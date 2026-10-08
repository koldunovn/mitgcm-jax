"""mitjax/ad/cg2d_rule.py with the real exchanges and Fortran-order global sums (plan Task 7c after 7b), tier 1x:
P=1 and P=4 under `shard_map(check_vma=True)` on fake CPU devices (forward and the rule's dot test only;
sharded driver gradients are tier 2 on GPUs, [L-PAR-11]).

TEST FIXTURE (not model code): a CG2D-like solve on the real tile layouts of two M1 experiments (probed maps):
tutorial_baroclinic_gyre (4 tiles of 31 x 31, P=4: one tile per device, every tile edge a partition edge) and
advect_xy (2 tiles of 20 x 10, P=4: two padding tiles, replicas of tile 1). Coefficients aW2d, aS2d > 0 (20 % of the
faces closed), aC2d = -(sum) - c0, exchanged with EXCH_XY_RL so the operator is symmetric across tile edges. The
fixture forward solve follows cg2d.F's structure: RHS normalised by its global max (`_GLOBAL_MAX_RL`), first guess
scaled on the interior and exchanged (cg2d.F:120-136), Jacobi-preconditioned CG whose search direction is filled by
EXCH_S3D_RL before the stencil, global sums `ex.global_sum_rl` (Fortran order), the answer un-normalised on the
interior only (cg2d.F:376-378), so its halo lanes hold the stale exchanged first guess, as CG2D's do. The caller then
applies EXCH_XY_RL (solve_for_pressure.F:315); the dot test is of that composite, with cotangents on the exchanged
halo lanes and on interior points next to partition edges.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax.ad import cg2d_rule as R
from mitjax.eesupp import exch_maps as EM
from mitjax.eesupp.exchange import Exchanger
from mitjax.eesupp.shard import TileSharding

EXPS = ("tutorial_baroclinic_gyre", "advect_xy")


def _interior(c):
    L = c["ex"].layout
    return jnp.asarray(L.interior())


def operator(A, c, v):
    """(aW x(i-1) + aW(i+1) x(i+1) + aS x(j-1) + aS(j+1) x(j+1) + aC x) / cg2dNorm on the interior (cg2d.F:164-169
    association), 0 on halo lanes; v's ring filled by EXCH_S3D_RL (as cg2d.F does for cg2d_s)."""
    ex = c["ex"]
    L = ex.layout
    INT = _interior(c)
    aW, aS, aC = A
    x = ex.EXCH_S3D_RL(jnp.where(INT, v, 0.0))
    j0, j1, i0, i1 = L.OLy, L.OLy + L.sNy, L.OLx, L.OLx + L.sNx
    out = (aW[:, j0:j1, i0:i1] * x[:, j0:j1, i0 - 1:i1 - 1]
           + aW[:, j0:j1, i0 + 1:i1 + 1] * x[:, j0:j1, i0 + 1:i1 + 1]
           + aS[:, j0:j1, i0:i1] * x[:, j0 - 1:j1 - 1, i0:i1]
           + aS[:, j0 + 1:j1 + 1, i0:i1] * x[:, j0 + 1:j1 + 1, i0:i1]
           + aC[:, j0:j1, i0:i1] * x[:, j0:j1, i0:i1]) / c["cg2dNorm"]
    return jnp.zeros_like(v).at[:, j0:j1, i0:i1].set(out)


def jacobi(A, c, r):
    INT = _interior(c)
    return jnp.where(INT, r * (c["cg2dNorm"] / jnp.where(INT, A[2], 1.0)), 0.0)


def forward_solve(A, b, x_first, c, s):
    """cg2d.F-structured fixture solve (module docstring); returns the array with CG2D's stale halos."""
    ex = c["ex"]
    INT = _interior(c)
    rhsMax = ex.global_max(jnp.where(INT, jnp.abs(b), 0.0))
    rhsNorm = 1.0 / rhsMax
    bn = jnp.where(INT, b * rhsNorm, 0.0)
    x0 = ex.EXCH_XY_RL(jnp.where(INT, x_first * rhsNorm, x_first))
    L = ex.layout
    gs = lambda p, q: ex.global_sum_rl((p * q)[:, L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx])   # noqa: E731
    r0 = bn - operator(A, c, x0)
    z0 = jacobi(A, c, r0)

    def cond(st):
        it, _, r, _, _ = st
        return (it < 400) & (gs(r, r) >= s["tol_sq"])

    def body(st):
        it, x, r, p, rz = st
        q = operator(A, c, p)
        alpha = rz / gs(p, q)
        x = jnp.where(INT, x + alpha * p, x)
        r = r - alpha * q
        z = jacobi(A, c, r)
        rz1 = gs(r, z)
        return it + 1, x, r, z + (rz1 / rz) * p, rz1

    it, x, r, _, _ = jax.lax.while_loop(cond, body, (jnp.zeros((), jnp.int32), x0, r0, z0, gs(r0, z0)))
    x = jnp.where(INT, x / rhsNorm, x)
    return x, {"numIters": it, "lastResidual": jnp.sqrt(gs(r, r))}


TIGHT = R.make_pcg_solve(operator, precond=jacobi, vdot=R.exchanger_vdot)
PER_DEVICE = R.make_pcg_solve(operator, precond=jacobi, vdot=lambda c, a, b: R.tile_vdot(a, b))


def make_case(maps, seed=0):
    L = maps.layout
    ex = Exchanger(maps)
    INT = L.interior()
    rng = np.random.default_rng(seed)
    shp = L.shape2d

    def interior_field(lo, hi, closed=0.0):
        a = rng.uniform(lo, hi, shp) * (rng.random(shp) >= closed)
        return np.asarray(ex.EXCH_XY_RL(np.where(INT, a, 0.0)))

    aW, aS = interior_field(0.5, 1.5, 0.2), interior_field(0.5, 1.5, 0.2)
    c0 = rng.uniform(0.1, 0.3, shp)
    aE = np.roll(aW, -1, axis=2)                          # aW(i+1) (halo filled by the exchange)
    aN = np.roll(aS, -1, axis=1)                          # aS(j+1)
    aC = np.asarray(ex.EXCH_XY_RL(np.where(INT, -(aW + aE + aS + aN) - c0, 0.0)))
    norm = 1.0 / 1.5
    A = tuple(norm * a for a in (aW, aS, aC))
    b = np.asarray(ex.EXCH_XY_RL(np.where(INT, rng.standard_normal(shp), 0.0)))
    x_first = np.asarray(ex.EXCH_XY_RL(np.where(INT, 0.1 * rng.standard_normal(shp), 0.0)))
    return A, b, x_first, norm


def edge_cotangent(L, seed):
    """Random on interior points next to tile edges (= partition edges at Tloc = 1) and on every halo lane."""
    rng = np.random.default_rng(seed)
    m = np.zeros((L.ny, L.nx), bool)
    m[:, :L.OLx + 1] = m[:, L.OLx + L.sNx - 1:] = m[:L.OLy + 1, :] = m[L.OLy + L.sNy - 1:, :] = True
    return rng.standard_normal(L.shape2d) * m


def solve_then_exchange(ex, A, b, x_first, norm, tol_sq, interior):
    """The rule, then SOLVE_FOR_PRESSURE's EXCH_XY_RL (the composite the model computes)."""
    c = {"ex": ex, "cg2dNorm": norm}
    x, aux = R.cg2d_implicit(forward_solve, operator, TIGHT, A, b, x_first, c, {"tol_sq": tol_sq}, interior)
    return ex.EXCH_XY_RL(x), x, aux["numIters"]


@pytest.mark.parametrize("exp", EXPS)
def test_rule_with_real_exchanger_p1(exp):
    """P=1, single-device Exchanger: the rule's forward value == the fixture solve on every lane (bitwise); the
    residual of the converged system is small; the dot test of exchange o rule with partition-edge and halo cotangents
    <= 1e-11 at tangent amplitudes 1 and 1e-8."""
    maps = EM.load_maps(exp)
    L = maps.layout
    ex = Exchanger(maps)
    A, b, x_first, norm = make_case(maps)
    interior = ex.tile_mask(L.interior())
    tol = jnp.asarray(1e-26)
    y, x, it = jax.jit(solve_then_exchange)(ex, A, b, x_first, norm, tol, interior)
    x_lit, _ = jax.jit(forward_solve)(A, b, x_first, {"ex": ex, "cg2dNorm": norm}, {"tol_sq": tol})
    assert np.array_equal(np.asarray(x).view(np.int64), np.asarray(x_lit).view(np.int64))
    res = operator(A, {"ex": ex, "cg2dNorm": jnp.asarray(norm)}, y) - jnp.where(L.interior(), b, 0.0)
    assert float(jnp.max(jnp.abs(res))) <= 1e-10 * float(jnp.max(jnp.abs(b))), float(jnp.max(jnp.abs(res)))
    rng = np.random.default_rng(3)
    v = (tuple(rng.standard_normal(L.shape2d) for _ in A), rng.standard_normal(L.shape2d))
    w = edge_cotangent(L, 4)
    f = lambda A, b: solve_then_exchange(ex, A, b, x_first, norm, tol, interior)[0]   # noqa: E731
    jvp = jax.jit(lambda a, t: jax.jvp(f, a, t)[1])
    rhs = jax.jit(lambda a, ct: jax.vjp(f, *a)[1](ct))((A, b), w)
    rhs = sum(float(np.vdot(p, q)) for p, q in zip(jax.tree.leaves(v), jax.tree.leaves(rhs)))
    rel = []
    for amp in (1.0, 1e-8):
        lhs = float(np.vdot(jvp((A, b), jax.tree.map(lambda z: amp * z, v)), w))
        rel.append(abs(lhs - amp * rhs) / np.max(np.abs([lhs, amp * rhs])))
    print(f"{exp} P=1 dot test (amp 1, 1e-8): {rel}, cg2d iterations {int(it)}")
    assert np.all(np.isfinite(rel)) and np.max(rel) <= 1e-11, rel


@pytest.mark.parametrize("exp", EXPS)
def test_rule_sharded_p4(exp):
    """P=4 under shard_map(check_vma=True): `interior` from ex.tile_mask (False on padding tiles) and cg2dNorm made
    varying with ex.vary before the call. Forward (rule output and the exchanged solution) == P=1 bitwise on every lane
    and the same iteration count; the dot test of exchange o rule with cotangents on partition-edge points and halo
    lanes (zero on padding) <= 1e-11; the tangent is 0 on the padding tiles' interior and the gradient is exactly 0 on
    every padding lane."""
    maps = EM.load_maps(exp)
    L = maps.layout
    sh = TileSharding(maps, 4)
    A, b, x_first, norm = make_case(maps)
    tol = jnp.asarray(1e-26)
    ex1 = Exchanger(maps)
    y1, x1, it1 = jax.jit(solve_then_exchange)(ex1, A, b, x_first, norm, tol, ex1.tile_mask(L.interior()))
    INT = L.interior()

    def body(ex, A, b, x_first, norm, tol):
        interior = ex.tile_mask(INT)
        norm, tol = ex.vary((norm, tol))
        y, x, it = solve_then_exchange(ex, A, b, x_first, norm, tol, interior)
        return y, x, ex.first_device(it)

    T, REP = sh.TILES, sh.REP
    f = sh.shard_map(body, in_specs=(T, T, T, T, REP, REP), out_specs=(T, T, REP))
    As = tuple(sh.put_tiles(a) for a in A)
    bs, xs = sh.put_tiles(b), sh.put_tiles(x_first)
    y4, x4, it4 = f(sh.ex, As, bs, xs, norm, tol)
    assert int(it4) == int(it1) > 0
    for one, four in ((y1, y4), (x1, x4)):
        assert np.array_equal(np.asarray(one).view(np.int64), sh.unpad(four).view(np.int64)), exp

    rng = np.random.default_rng(3)
    v = (tuple(sh.blocks.pad(rng.standard_normal(L.shape2d)) for _ in A), sh.blocks.pad(rng.standard_normal(L.shape2d)))
    w = sh.blocks.pad(edge_cotangent(L, 4))
    w[L.nTiles:] = 0.0

    def fy(ex, A, b, x_first, norm, tol):
        interior = ex.tile_mask(INT)
        norm, tol = ex.vary((norm, tol))
        return solve_then_exchange(ex, A, b, x_first, norm, tol, interior)[0]

    def tl(ex, A, b, x_first, norm, tol, dA, db):
        return jax.jvp(lambda A, b: fy(ex, A, b, x_first, norm, tol), (A, b), (dA, db))[1]

    def adj(ex, A, b, x_first, norm, tol, w):
        return jax.vjp(lambda A, b: fy(ex, A, b, x_first, norm, tol), A, b)[1](ex.zero_padding(w))

    ftl = sh.shard_map(tl, in_specs=(T, T, T, T, REP, REP, T, T), out_specs=T)
    fadj = sh.shard_map(adj, in_specs=(T, T, T, T, REP, REP, T), out_specs=T)
    gA, gb = fadj(sh.ex, As, bs, xs, norm, tol, sh.put_padded(w))
    rhs = sum(float(np.vdot(p, np.asarray(q))) for p, q in zip(jax.tree.leaves(v), jax.tree.leaves((gA, gb))))
    rel = []
    for amp in (1.0, 1e-8):
        dA = tuple(sh.put_padded(amp * a) for a in v[0])
        dy = np.asarray(ftl(sh.ex, As, bs, xs, norm, tol, dA, sh.put_padded(amp * v[1])))
        # padding tiles: solution tangent 0 on their interior (interior mask False); their halos receive tile 1's
        # halo sources through the exchange (replica padding) and are dropped with the padding
        assert np.all(dy[L.nTiles:][:, INT] == 0.0)
        lhs = float(np.vdot(dy, w))
        rel.append(abs(lhs - amp * rhs) / np.max(np.abs([lhs, amp * rhs])))
    print(f"{exp} P=4 (Tloc={sh.blocks.Tloc}, padding {sh.blocks.Tpad - L.nTiles}) dot test (amp 1, 1e-8): {rel}")
    assert np.all(np.isfinite(rel)) and np.max(rel) <= 1e-11, rel
    for g in jax.tree.leaves((gA, gb)):
        assert np.all(np.asarray(g)[L.nTiles:] == 0.0)


def test_negative_controls_sharded_traps():
    """The two sharded traps of [L-PAR-10] and [L-PAR-6], measured at P=4 on advect_xy (2 padding tiles):
    without `ex.vary` on cg2dNorm (and the forward tolerance) the adjoint fails to lower (JAX 0.10.1: "pvary is a
    invariant->variant collective" inside custom_linear_solve); with `interior` True on the padding tiles the
    gradient leaks onto padding lanes (measured max 8.9 over A and b), where the real rule gives exact zeros; a
    per-device inner product (`tile_vdot`) in the derivative solve is refused (it would sum one device's tiles)."""
    maps = EM.load_maps("advect_xy")
    L = maps.layout
    sh = TileSharding(maps, 4)
    A, b, x_first, norm = make_case(maps)
    tol = jnp.asarray(1e-26)
    INT = L.interior()
    As = tuple(sh.put_tiles(a) for a in A)
    bs, xs = sh.put_tiles(b), sh.put_tiles(x_first)
    w = sh.blocks.pad(edge_cotangent(L, 4))
    w[L.nTiles:] = 0.0

    def run(variant):
        def solve(ex, A, b, x_first, norm, tol, interior):
            if variant != "tile_vdot":
                return solve_then_exchange(ex, A, b, x_first, norm, tol, interior)[0]
            c = {"ex": ex, "cg2dNorm": norm}
            x, _ = R.cg2d_implicit(forward_solve, operator, PER_DEVICE, A, b, x_first, c, {"tol_sq": tol}, interior)
            return ex.EXCH_XY_RL(x)

        def fy(ex, A, b, x_first, norm, tol):
            if variant == "no_vary":
                interior = ex.tile_mask(INT)
            elif variant == "interior_on_padding":
                interior = jnp.broadcast_to(jnp.asarray(INT), (sh.blocks.Tloc,) + INT.shape)
                norm, tol = ex.vary((norm, tol))
            else:
                interior = ex.tile_mask(INT)
                norm, tol = ex.vary((norm, tol))
            return solve(ex, A, b, x_first, norm, tol, interior)

        def adj(ex, A, b, x_first, norm, tol, w):
            return jax.vjp(lambda A, b: fy(ex, A, b, x_first, norm, tol), A, b)[1](ex.zero_padding(w))

        T, REP = sh.TILES, sh.REP
        f = sh.shard_map(adj, in_specs=(T, T, T, T, REP, REP, T), out_specs=T)
        gA, gb = f(sh.ex, As, bs, xs, norm, tol, sh.put_padded(w))
        return max(float(np.max(np.abs(np.asarray(g)[L.nTiles:]))) for g in jax.tree.leaves((gA, gb)))

    assert run("rule") == 0.0
    leak = run("interior_on_padding")
    print(f"gradient on padding lanes with interior True on padding: {leak}")
    assert leak > 0.1
    with pytest.raises(ValueError, match="invariant->variant"):
        run("no_vary")
    # a per-device inner product in the derivative solve (the rule's old default) is refused under shard_map
    with pytest.raises(ValueError, match="one device's tiles"):
        run("tile_vdot")
