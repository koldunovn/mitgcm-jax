"""mitjax/ad/cg2d_rule.py (plan Task 7c): the implicit-derivative rule on a synthetic tiled 5-point operator.

TEST FIXTURE (not model code): a 10 x 8 channel (periodic in x, walls in y) on 2 x 2 tiles of sNx = 5, sNy = 4 with
OLx = OLy = 2, a self-contained halo exchange (gather from the global interior; a lane beyond a wall stays 0, as halo
points no exchange writes do in ini_cg2d.F), and a CG2D-like operator with MITgcm's signs: aW2d, aS2d > 0 on open
faces (0 on the walls), aC2d = -(aW + aW(i+1) + aS + aS(j+1)) - c0 with c0 > 0 (the free-surface term), all divided by
cg2dNorm (a traced setup constant). The rule with the real (probed) exchanges, the Fortran-order global sums and at
P=4 under shard_map: test_cg2d_rule_sharded.py (tier 1x).

Gates: the rule's forward value == the literal forward solve on every lane, halos included, bitwise (Nikolay
2026-10-01; planted halo-zeroed forward value fails); forward == dense solve; tangent and adjoint == the dense implicit derivative; FD sweep; dot test TL vs adjoint
<= 1e-11 with cotangents on tile-edge interior points and on halo lanes. Negative controls (measured to bite):
without the halo zeroing of the derivative solve's solution (the planted `_mask_solution` = identity, with a
derivative solve that returns exchanged halos as a literal CG2D does) the dot test and the dense comparison fail;
`custom_linear_solve`'s own JVP with the loose, warm-started forward solve fails the dot test ([L-AD-9]).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax.ad import cg2d_rule as R

# ---------------------------------------------------------------------------------------------------- test fixture
NTX, NTY, SNX, SNY, OL = 2, 2, 5, 4, 2
NX, NY = NTX * SNX, NTY * SNY
T, NYL, NXL = NTX * NTY, SNY + 2 * OL, SNX + 2 * OL
INTERIOR = R.interior_mask(SNX, SNY, OL, OL)


def _maps():
    """Gather map: lane (t, j, i) -> flat global interior index; valid = inside the channel (periodic in x)."""
    gidx = np.zeros((T, NYL, NXL), np.int64)
    valid = np.zeros((T, NYL, NXL), bool)
    for t in range(T):
        ty, tx = divmod(t, NTX)
        for j in range(NYL):
            for i in range(NXL):
                J, I = ty * SNY + j - OL, (tx * SNX + i - OL) % NX
                if 0 <= J < NY:
                    gidx[t, j, i], valid[t, j, i] = J * NX + I, True
    return gidx, valid


GIDX, VALID = _maps()


def to_global(v):
    """Interior lanes of tile arrays [T, NYL, NXL] -> global [NY, NX]."""
    a = v[:, OL:OL + SNY, OL:OL + SNX].reshape(NTY, NTX, SNY, SNX)
    return a.transpose(0, 2, 1, 3).reshape(NY, NX)


def exch(maps, v):
    """Fill every lane (halos included) from the global interior: the fixture's EXCH_XY_RL."""
    gidx, valid = maps
    return jnp.where(valid, to_global(v).reshape(-1)[gidx], 0.0)


def from_global(maps, G):
    return exch(maps, jnp.zeros((T, NYL, NXL)).at[:, OL:OL + SNY, OL:OL + SNX].set(
        jnp.asarray(G).reshape(NTY, SNY, NTX, SNX).transpose(0, 2, 1, 3).reshape(T, SNY, SNX)))


def operator(A, c, v):
    """(aW2d x(i-1) + aW2d(i+1) x(i+1) + aS2d x(j-1) + aS2d(j+1) x(j+1) + aC2d x) / cg2dNorm on the interior, 0 on
    halo lanes, in cg2d.F's association order (cg2d.F:164-169)."""
    aW, aS, aC = A
    x = exch(c["maps"], v)
    j0, j1, i0, i1 = OL, OL + SNY, OL, OL + SNX
    out = (aW[:, j0:j1, i0:i1] * x[:, j0:j1, i0 - 1:i1 - 1]
           + aW[:, j0:j1, i0 + 1:i1 + 1] * x[:, j0:j1, i0 + 1:i1 + 1]
           + aS[:, j0:j1, i0:i1] * x[:, j0 - 1:j1 - 1, i0:i1]
           + aS[:, j0 + 1:j1 + 1, i0:i1] * x[:, j0 + 1:j1 + 1, i0:i1]
           + aC[:, j0:j1, i0:i1] * x[:, j0:j1, i0:i1]) / c["cg2dNorm"]
    return jnp.zeros_like(v).at[:, j0:j1, i0:i1].set(out)


def jacobi(A, c, r):
    return jnp.where(INTERIOR, r * (c["cg2dNorm"] / jnp.where(INTERIOR, A[2], 1.0)), 0.0)


def forward_solve(A, b, x_first, c, s):
    """Fixture forward: Jacobi PCG from the first guess until err_sq < cg2dTolerance_sq (s["tol_sq"], a traced setup
    constant, relative to |b|^2 here) or 500 iterations, then the exchange SOLVE_FOR_PRESSURE applies (halos filled)."""
    x0 = jnp.where(INTERIOR, x_first, 0.0)
    r0 = jnp.where(INTERIOR, b, 0.0) - operator(A, c, x0)
    bb = R.tile_vdot(b * INTERIOR, b * INTERIOR)
    z0 = jacobi(A, c, r0)

    def cond(st):
        it, _, r, _, _ = st
        return (it < 500) & (R.tile_vdot(r, r) >= s["tol_sq"] * bb)

    def body(st):
        it, x, r, p, rz = st
        q = operator(A, c, p)
        alpha = rz / R.tile_vdot(p, q)
        x, r = x + alpha * p, r - alpha * q
        z = jacobi(A, c, r)
        rz1 = R.tile_vdot(r, z)
        return it + 1, x, r, z + (rz1 / rz) * p, rz1

    it, x, r, _, _ = jax.lax.while_loop(cond, body, (jnp.zeros((), jnp.int32), x0, r0, z0, R.tile_vdot(r0, z0)))
    return exch(c["maps"], x), {"numIters": it, "lastResidual": jnp.sqrt(R.tile_vdot(r, r))}


def literal_style_derivative_solve(A, c, s, r):
    """A derivative solve that returns exchanged halos, as a literal CG2D + EXCH_XY_RL would (tight tolerance)."""
    x, _ = forward_solve(A, r, jnp.zeros_like(r), c, {"tol_sq": jnp.asarray(1e-28)})
    return x


TIGHT = R.make_pcg_solve(operator, precond=jacobi)


def make_case(seed=0):
    """Random channel coefficients (global, consistent across tiles and halos), rhs, first guess, maps, consts."""
    rng = np.random.default_rng(seed)
    aWg = rng.uniform(0.5, 1.5, (NY, NX))                 # west faces; periodic in x: every face open
    aSg = rng.uniform(0.5, 1.5, (NY, NX))
    aSg[0, :] = 0.0                                        # south wall
    aSn = np.vstack([aSg[1:], np.zeros((1, NX))])          # aS(j+1); north wall
    aEg = np.roll(aWg, -1, axis=1)                         # aW(i+1)
    c0 = rng.uniform(0.1, 0.3, (NY, NX))
    aCg = -(aWg + aEg + aSg + aSn) - c0
    norm = 1.0 / 1.5                                       # cg2dNorm = 1/max|aW, aS| style constant
    maps = (jnp.asarray(GIDX), jnp.asarray(VALID))
    A = tuple(from_global(maps, norm * g) for g in (aWg, aSg, aCg))   # normalised coefficients, halos exchanged
    c = {"maps": maps, "cg2dNorm": jnp.asarray(norm)}
    s = {"tol_sq": jnp.asarray(1e-15 ** 2)}
    b = from_global(maps, rng.standard_normal((NY, NX)))
    x_first = from_global(maps, 0.1 * rng.standard_normal((NY, NX)))
    return A, b, x_first, c, s, (aWg, aSg, aCg)


def dense_matrix(glob):
    """The global matrix of operator(), physical units (normalised coefficients / cg2dNorm = global coefficients)."""
    aWg, aSg, aCg = glob
    n = NX * NY
    M = np.zeros((n, n))
    for J in range(NY):
        for I in range(NX):
            k = J * NX + I
            M[k, k] += aCg[J, I]
            M[k, J * NX + (I - 1) % NX] += aWg[J, I]
            M[k, J * NX + (I + 1) % NX] += aWg[J, (I + 1) % NX]
            if J > 0:
                M[k, (J - 1) * NX + I] += aSg[J, I]
            if J < NY - 1:
                M[k, (J + 1) * NX + I] += aSg[J + 1, I]
    return M


def solve_rule(A, b, x_first, c, s, derivative_solve=TIGHT):
    x, _ = R.cg2d_implicit(forward_solve, operator, derivative_solve, A, b, x_first, c, s, INTERIOR)
    return x


def edge_cotangent(seed):
    """Random cotangent on the interior points next to tile edges and on every halo lane (0 elsewhere)."""
    rng = np.random.default_rng(seed)
    m = np.zeros((NYL, NXL), bool)
    m[:, :OL + 1] = m[:, OL + SNX - 1:] = m[:OL + 1, :] = m[OL + SNY - 1:, :] = True
    return jnp.asarray(rng.standard_normal((T, NYL, NXL)) * m)


def fmax(xs):
    """max of a list of floats that fails on a NaN (Python's max() skips one at some positions)."""
    a = np.asarray(xs, np.float64)
    assert np.all(np.isfinite(a)), xs
    return float(np.max(a))


def fmin(xs):
    a = np.asarray(xs, np.float64)
    assert np.all(np.isfinite(a)), xs
    return float(np.min(a))


def dot_test(f, args, v, w):
    """|<J v, w> - <v, J^T w>| / max(|.|) for y = f(*args) (jit; same compiled TL for both amplitudes)."""
    jvp = jax.jit(lambda a, t: jax.jvp(f, a, t)[1])
    vjp = jax.jit(lambda a, ct: jax.vjp(f, *a)[1](ct))
    rhs = sum(float(jnp.vdot(x, y)) for x, y in zip(jax.tree.leaves(v), jax.tree.leaves(vjp(args, w))))
    out = []
    for amp in (1.0, 1e-8):
        lhs = float(jnp.vdot(jvp(args, jax.tree.map(lambda z: amp * z, v)), w))
        out.append(abs(lhs - amp * rhs) / np.max(np.abs([lhs, amp * rhs])))
    return out


@pytest.fixture(scope="module")
def case():
    return make_case(0)


# ------------------------------------------------------------------------------------------------------------ gates


def test_forward_equals_literal_and_dense(case, monkeypatch):
    """The rule's forward value is the literal forward solve's array on EVERY lane, halos included, bitwise (Nikolay
    2026-10-01: faithful to the model); the solution matches the dense solve to the forward tolerance; numIters is
    passed through. Negative control: a planted halo-zeroed forward value (the rule's earlier design) fails the gate."""
    A, b, x_first, c, s, glob = case

    def rule_forward(A, b, x0, c, s):
        return R.cg2d_implicit(forward_solve, operator, TIGHT, A, b, x0, c, s, INTERIOR)

    x, aux = jax.jit(rule_forward)(A, b, x_first, c, s)
    x_lit, _ = jax.jit(forward_solve)(A, b, x_first, c, s)

    def same_bits(p, q):
        return np.array_equal(np.asarray(p).view(np.int64), np.asarray(q).view(np.int64))

    assert np.any(np.asarray(x_lit)[:, ~INTERIOR] != 0.0)          # the halos the gate must keep are not zero
    assert same_bits(x, x_lit)
    xd = np.linalg.solve(dense_matrix(glob), np.asarray(to_global(b)).ravel())
    err = np.max(np.abs(np.asarray(to_global(x)).ravel() - xd)) / np.max(np.abs(xd))
    assert err < 1e-12, err
    assert 0 < int(aux["numIters"]) < 500
    # planted: the forward value with zeroed halo lanes
    monkeypatch.setattr(R, "_forward_value", lambda interior, x_full: R._mask_halo(interior, x_full))
    jax.clear_caches()
    x_bad, _ = jax.jit(rule_forward)(A, b, x_first, c, s)
    assert not same_bits(x_bad, x_lit)
    assert same_bits(np.asarray(x_bad)[:, INTERIOR], np.asarray(x_lit)[:, INTERIOR])


def test_tangent_and_adjoint_vs_dense(case):
    """jvp == M^-1 (db - dM x) and vjp(w) == M^-T w (interior; halo cotangents carry no derivative) of the dense
    system, to 1e-11 relative to the field max."""
    A, b, x_first, c, s, glob = case
    rng = np.random.default_rng(1)
    dglob = tuple(rng.standard_normal((NY, NX)) * (g != 0) for g in glob)
    maps = c["maps"]
    dA = tuple(from_global(maps, c["cg2dNorm"] * g) for g in dglob)
    db = from_global(maps, rng.standard_normal((NY, NX)))
    f = lambda A, b: solve_rule(A, b, x_first, c, s)   # noqa: E731
    x, dx = jax.jit(lambda A, b, dA, db: jax.jvp(f, (A, b), (dA, db)))(A, b, dA, db)
    M, dM = dense_matrix(glob), dense_matrix(dglob)
    xg = np.asarray(to_global(x)).ravel()
    dxd = np.linalg.solve(M, np.asarray(to_global(db)).ravel() - dM @ xg)
    dxg = np.asarray(to_global(dx)).ravel()
    assert np.max(np.abs(dxg - dxd)) / np.max(np.abs(dxd)) < 1e-11
    assert np.all(np.asarray(dx)[:, ~INTERIOR] == 0.0)

    w = edge_cotangent(2) + jnp.where(INTERIOR, jnp.asarray(rng.standard_normal((T, NYL, NXL))), 0.0)
    _, pull = jax.vjp(f, A, b)
    gA, gb = jax.jit(pull)(w)
    gbd = np.linalg.solve(M.T, np.asarray(to_global(w)).ravel())    # only the interior of w reaches x
    gbg = np.asarray(to_global(gb)).ravel()
    assert np.max(np.abs(gbg - gbd)) / np.max(np.abs(gbd)) < 1e-11
    assert np.all(np.asarray(gb)[:, ~INTERIOR] == 0.0)


def test_dot_test_tile_edges(case):
    """Dot test of (A, b) -> x with random tangents on every lane of aW2d, aS2d, aC2d, b (halos included) and a
    cotangent on tile-edge interior points and halo lanes: <= 1e-11 at tangent amplitudes 1 and 1e-8."""
    A, b, x_first, c, s, _ = case
    rng = np.random.default_rng(3)
    v = (tuple(jnp.asarray(rng.standard_normal(a.shape)) for a in A), jnp.asarray(rng.standard_normal(b.shape)))
    w = edge_cotangent(4)
    rel = dot_test(lambda A, b: solve_rule(A, b, x_first, c, s), (A, b), v, w)
    print(f"dot test (amp 1, 1e-8): {rel}")
    assert fmax(rel) <= 1e-11, rel


def test_fd_sweep(case):
    """Central FD of J = <w, x> along a direction of (A, b) against the AD directional derivative: a plateau with
    relative error <= 1e-8 (forward converged to 1e-15; FD error ~ h^2 at large h, noise / h at small h)."""
    A, b, x_first, c, s, glob = case
    rng = np.random.default_rng(5)
    w = jnp.where(INTERIOR, jnp.asarray(rng.standard_normal((T, NYL, NXL))), 0.0)
    dglob = tuple(rng.standard_normal((NY, NX)) * (g != 0) for g in glob)
    dA = tuple(from_global(c["maps"], c["cg2dNorm"] * g) for g in dglob)
    db = from_global(c["maps"], rng.standard_normal((NY, NX)))
    J = jax.jit(lambda A, b: jnp.vdot(w, solve_rule(A, b, x_first, c, s)))
    ad = float(jax.jit(lambda A, b: jax.jvp(J, (A, b), (dA, db))[1])(A, b))
    errs = []
    for h in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        p = (tuple(a + h * d for a, d in zip(A, dA)), b + h * db)
        m = (tuple(a - h * d for a, d in zip(A, dA)), b - h * db)
        fd = (float(J(*p)) - float(J(*m))) / (2 * h)
        errs.append(abs(fd - ad) / abs(ad))
    print(f"FD rel errors h=1e-2..1e-6: {errs}")
    assert fmin(errs) <= 1e-8, errs


def test_cg2dnorm_is_traced_and_differentiated(case):
    """cg2dNorm is a traced argument (one compilation for two values) and the rule gives the converged solution's
    derivative with respect to it: x scales as cg2dNorm when the normalised coefficients are held fixed."""
    A, b, x_first, c, s, _ = case
    w = edge_cotangent(6) * INTERIOR
    J = jax.jit(lambda n, maps: jnp.vdot(w, solve_rule(A, b, x_first, {"maps": maps, "cg2dNorm": n}, s)))
    n, maps = c["cg2dNorm"], c["maps"]
    g = jax.grad(J)(n, maps)
    # M = A_norm / n, so x = n A_norm^-1 b and dJ/dn = J / n
    assert abs(float(g) - float(J(n, maps)) / float(n)) <= 1e-11 * abs(float(g))
    J(n * 1.25, maps)
    assert J._cache_size() == 1


# ------------------------------------------------------------------------------------------------ negative controls


def test_negative_control_halo_zeroing(case, monkeypatch):
    """Planted error: the derivative solve returns exchanged halos (as a literal CG2D + EXCH_XY_RL does) and the
    rule's `_mask_solution` is the identity. custom_linear_solve(symmetric=True) then transposes E S^-1 as E S^-1
    instead of S^-1 E^T: the dot test with tile-edge (halo) cotangents fails, and the tangent is nonzero on halo lanes
    where the primal is a constant 0. With the zeroing (the real rule) the same derivative solve passes."""
    A, b, x_first, c, s, _ = case
    rng = np.random.default_rng(3)
    v = (tuple(jnp.asarray(rng.standard_normal(a.shape)) for a in A), jnp.asarray(rng.standard_normal(b.shape)))
    w = edge_cotangent(4)
    f = lambda A, b: solve_rule(A, b, x_first, c, s, literal_style_derivative_solve)   # noqa: E731
    good = dot_test(f, (A, b), v, w)
    assert fmax(good) <= 1e-11, good
    monkeypatch.setattr(R, "_mask_solution", lambda interior, y: y)
    jax.clear_caches()
    bad = dot_test(f, (A, b), v, w)
    print(f"halo zeroing: with {good}, without {bad}")
    assert fmin(bad) > 1e-3, bad
    dx = jax.jvp(f, (A, b), v)[1]
    assert np.max(np.abs(np.asarray(dx)[:, ~INTERIOR])) > 0.1
    # the planted error is invisible to an interior-only cotangent: why the gate needs halo-lane cotangents
    w_int = w * INTERIOR
    hidden = dot_test(f, (A, b), v, w_int)
    print(f"without zeroing, interior-only cotangent: {hidden}")
    assert fmax(hidden) <= 1e-11, hidden


def test_negative_control_linear_solve_jvp(case):
    """[L-AD-9]: lax.custom_linear_solve with the loose (1e-7), warm-started forward solve as `solve` differentiates
    by running that solve on the tangent; the tangent-linear model is then only as accurate as the loose solve (and not
    linear in the tangent: the first guess is the primal's), and the dot test misses the 1e-11 bar at both amplitudes,
    while the rule passes on the same input."""
    A, b, x_first, c, s, _ = case
    loose = {"tol_sq": jnp.asarray(1e-7 ** 2)}

    def f_cls(b):
        def matvec(v):
            return jnp.where(INTERIOR, operator(A, c, jnp.where(INTERIOR, v, 0.0)), 0.0)

        def solve(_, r):
            return jnp.where(INTERIOR, forward_solve(A, r, x_first, c, loose)[0], 0.0)
        return jax.lax.custom_linear_solve(matvec, b, solve, symmetric=True)

    rng = np.random.default_rng(7)
    v = jnp.where(INTERIOR, jnp.asarray(rng.standard_normal(b.shape)), 0.0)
    w = jnp.where(INTERIOR, jnp.asarray(rng.standard_normal(b.shape)), 0.0)
    bad = dot_test(f_cls, (b,), (v,), w)
    good = dot_test(lambda b: solve_rule(A, b, x_first, c, loose), (b,), (v,), w)
    print(f"custom_linear_solve JVP (loose solve): {bad}; rule: {good}")
    # measured 2026-10-01: 8.6e-9 (amp 1), 6.0e-10 (amp 1e-8); the ECCO port measured 1.9e-9. Both miss the bar 10x+
    assert fmin(bad) > 1e-10, bad
    assert fmax(good) <= 1e-11, good
