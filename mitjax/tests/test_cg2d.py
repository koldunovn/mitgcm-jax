"""Gates of the two-dimensional pressure solver (M1 sub-lane CG2D): INI_CG2D / UPDATE_CG2D operator, CG2D replay, the
rule's forward value, derivatives, STDOUT lines, unported options, and negative controls measured to bite.

Oracle and inputs: mitjax/tests/cg2d_gate.py (dumps of the registered dumps-on runs; geometry teacher-forced from the
same dumps). Every comparison is element equality on all points (halos included) with an isfinite check on both
sides, in numpy.
"""

import dataclasses
import inspect
import re
import sys
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax.config.params import fortran_e15
from mitjax.eesupp import global_sum as gs
from mitjax.eesupp.exchange import Exchanger
from mitjax.model.src import cg2d as cg2d_mod
from mitjax.model.src import ini_cg2d as ini_cg2d_mod
from mitjax.model.src.cg2d_h import ini_parms_cg2d
from mitjax.tests import cg2d_gate as G

ALL = G.SOLVER_VARIANTS
CONSTANT_OPERATOR = [v for v in ALL if v[0] != "global_ocean.90x40x15"]   # no UPDATE_CG2D: C01 = INI_CG2D's
DOT_FD = [("tutorial_barotropic_gyre", "input"), ("tutorial_baroclinic_gyre", "input")]
GO = ("global_ocean.90x40x15", "input")


def _ids(v):
    return f"{v[0]}/{v[1]}"


def _assert_all_equal(res, what):
    bad = {k: v for k, v in res.items() if any(v[1:])}
    assert not bad, f"{what}: (n, n differing, non-finite ours, non-finite oracle[, bit patterns]) {bad}"


# ---------------------------------------------------------------------------------------------------------------------
# parameters and unported options

_PRINT = {  # Cg2dParams field -> label of the oracle's parameter printout (config_summary.F)
    "debugLevel": "debugLevel =", "cg2dMaxIters": "cg2dMaxIters =", "cg2dUseMinResSol": "cg2dUseMinResSol=",
    "cg2dTargetResidual": "cg2dTargetResidual =", "cg2dPreCondFreq": "cg2dPreCondFreq =",
    "useSRCGSolver": "useSRCGSolver =", "useNSACGSolver": "useNSACGSolver =",
    "printResidualFreq": "printResidualFreq =", "deltaTMom": "deltaTMom =", "deltaTFreeSurf": "deltaTFreeSurf =",
    "implicSurfPress": "implicSurfPress =", "implicDiv2DFlow": "implicDiv2DFlow =", "freeSurfFac": "freeSurfFac =",
    "momStepping": "momStepping =", "nonlinFreeSurf": "nonlinFreeSurf =", "cg2dFullAdjoint": "cg2dFullAdjoint =",
}


def _printed(lines, label):
    pre = re.compile(r"^\(PID\.TID \d+\.\d+\) ")
    body = [pre.sub("", ln) for ln in lines]
    hits = [n for n, ln in enumerate(body) if ln.startswith(label)]
    if len(hits) != 1:
        return None
    return body[hits[0] + 1].strip()


@pytest.mark.parametrize("v", ALL, ids=_ids)
def test_params_match_stdout_printout(v):
    """ini_parms_cg2d's values printed in the parameter printout's own format equal the oracle's printout."""
    e = G.experiment(*v)
    p = ini_parms_cg2d(e)
    lines = G.stdout_lines(*v)
    compared = 0
    for field, label in _PRINT.items():
        got = _printed(lines, label)
        if got is None:
            assert field == "cg2dFullAdjoint" and not e.cfg.cpp.ALLOW_AUTODIFF, f"{label} not printed"
            continue
        val = getattr(p, field)
        ours = fortran_e15(float(val)) if isinstance(val, (float, np.floating)) else \
            ("T" if val else "F") if isinstance(val, bool) else str(val)
        assert ours == got, f"{field}: ours {ours} printed {got}"
        compared += 1
    assert compared >= 15
    assert p.cg2dUseMinResSol == 0 and p.printResidualFreq == -1 and p.cg2dTargetResWunit_le_0


def test_unported_options_raise():
    v = ("tutorial_baroclinic_gyre", "input")
    e = G.experiment(*v)
    cg2dh, params = G.cg2dh_from_dump(*v, 0, 1.0)
    sz = e.cfg.size
    b = G.xy(jnp.zeros((4, sz.sNy + 2 * sz.OLy, sz.sNx + 2 * sz.OLx)), "cg2d_b", sz)
    ex = G.exchanger(v[0])
    with pytest.raises(NotImplementedError, match="nIterMin >= 0"):
        cg2d_mod.cg2d(b, b, 10, 0, cfg=e.cfg, cg2dh=cg2dh, params=params, ex=ex)
    with pytest.raises(NotImplementedError, match="printResidualFreq"):
        cg2d_mod.cg2d(b, b, 10, -1, cfg=e.cfg, cg2dh=cg2dh, params=dataclasses.replace(params, printResidualFreq=1),
                      ex=ex)
    # cg2dNormaliseRHS = .FALSE. is ported (GO lane, global_ocean.cs32x15): its gate and control are
    # test_cs32_go.py::test_cg2d_without_rhs_normalisation (C01 -> C02 bitwise, literal and rule)
    with pytest.raises(NotImplementedError, match="useSRCGSolver"):
        cg2d_mod.cg2d_solve(b, b, 10, -1, cfg=e.cfg, cg2dh=cg2dh, params=dataclasses.replace(params, useSRCGSolver=True),
                            ex=ex)
    # cg2dFullAdjoint (CG2D_STORE, no forward value) is ported by the GOADK lane (input_ad.bottomdrag,
    # test_goadk_model.py): no longer an unported option
    grid, surface = G.teacher_geometry(*v)
    # PTRACERS lane: cg2dTargetResWunit > 0 (ini_cg2d.F:152-162) is ported (gated against the printed tolerance in
    # test_ptracers_adjsens.py): no normalisation of the rhs, a tolerance in W units
    h2 = ini_cg2d_mod.ini_cg2d(cfg=e.cfg, grid=grid, surface=surface,
                               params=dataclasses.replace(params, cg2dTargetResWunit_le_0=False), ex=ex)
    assert not h2.cg2dNormaliseRHS and float(h2.cg2dTolerance_sq) > 0.
    from mitjax.model.src.update_cg2d import update_cg2d
    with pytest.raises(NotImplementedError, match="deepAtmosphere"):
        update_cg2d(cg2dh, None, 1, cfg=e.cfg, grid=grid, surface=surface,
                    params=dataclasses.replace(params, deepAtmosphere=True), ex=ex)


# ---------------------------------------------------------------------------------------------------------------------
# operator: INI_CG2D (constant operator) and the INI_CG2D -> UPDATE_CG2D chain (global_ocean, r* / NLFS)

@pytest.mark.parametrize("v", CONSTANT_OPERATOR, ids=_ids)
def test_ini_cg2d_operator_bitwise(v):
    """INI_CG2D == the operator CG2D receives at every dumped step (C01), all points, equal bit patterns; cg2dNorm ==
    the double INI_CG2D prints (17 digits), and our printed line == the oracle's."""
    c, _ = G.run_ini_cg2d(*v)
    line, norm = G.stdout_cg2dnorm(*v)
    assert float(c.cg2dNorm) == norm
    assert ini_cg2d_mod.ini_cg2d_message(c.cg2dNorm) == line
    ds, _, _ = G.oracle(*v)
    for it in ds.iterations():
        _assert_all_equal(G.compare_operator(c, *v, it, "C01_cg2d_inputs"), f"{_ids(v)} C01 it={it}")


def _go_chain(update=None):
    """global_ocean: INI_CG2D, UPDATE_CG2D at nIter0 (INITIALISE_VARIA, hFac of S00_begin), then UPDATE_CG2D of every
    dumped step (hFac of S07_update_rstar_T, myIter = start + 1) chained on our own output."""
    c, params = G.run_ini_cg2d(*GO)
    ds, it0, _ = G.oracle(*GO)
    yield "ini", c
    c = G.run_update_cg2d(*GO, c, params.nIter0, hfac_stage="S00_begin", it=it0)
    for it in ds.iterations():
        c = G.run_update_cg2d(*GO, c, it + 1, hfac_stage="S07_update_rstar_T", it=it)
        yield it, c


def test_update_cg2d_chain_global_ocean_bitwise():
    ds, _, _ = G.oracle(*GO)
    n = 0
    for it, c in _go_chain():
        if it == "ini":
            assert float(c.cg2dNorm) == G.stdout_cg2dnorm(*GO)[1]
            continue
        _assert_all_equal(G.compare_operator(c, *GO, it, "S08_update_cg2d"), f"S08 it={it}")
        _assert_all_equal(G.compare_operator(c, *GO, it, "C01_cg2d_inputs"), f"C01 it={it}")
        n += 1
    assert n == len(ds.iterations()) == 3


# ---------------------------------------------------------------------------------------------------------------------
# solver replay: C01 -> C02, literal and through the rule

@pytest.mark.parametrize("v", ALL, ids=_ids)
def test_cg2d_replay_bitwise(v):
    """From the dumped C01 inputs, cg2d reproduces C02 (cg2d_x on all points, numIters, nIterMin, firstResidual,
    minResidualSq, lastResidual) bitwise at every dumped step; cg2d_solve (the rule) returns the same arrays and
    scalars bit for bit on every lane."""
    ds, _, _ = G.oracle(*v)
    for it in ds.iterations():
        lit = G.replay_cg2d(*v, it, solve="literal")
        _assert_all_equal(G.compare_solution(*v, it, lit), f"{_ids(v)} it={it}")
        rule = G.replay_cg2d(*v, it, solve="rule")
        for k, (a, b) in enumerate(zip(jax.tree.leaves(lit), jax.tree.leaves(rule))):
            assert G.same_bits(a, b), f"{_ids(v)} it={it}: rule output leaf {k} differs from the literal solve"


@pytest.mark.parametrize("v", ALL, ids=_ids)
def test_cg2d_stdout_lines(v):
    """The ' cg2d: Sum(rhs),rhsMax' line (every CG2D call) and SOLVE_FOR_PRESSURE's cg2d_init_res / cg2d_iters /
    cg2d_last_res lines formatted from our replay equal the oracle's STDOUT lines of the same steps (the n-th solve of
    the run is the n-th dumped iteration; monitorFreq <= deltaT in all four runs: 10 solves, 10 blocks)."""
    ds, _, _ = G.oracle(*v)
    sums, mon = G.stdout_solver_lines(*v)
    assert len(sums) == 10 and len(mon) == 30, (len(sums), len(mon))
    for n, it in enumerate(ds.iterations()):
        out = G.replay_cg2d(*v, it, solve="literal")
        _, _, first, minsq, last, nits, nmin, printed = out
        assert cg2d_mod.cg2d_sum_rhs_message(printed["sumRHS"], printed["rhsMax"]) == sums[n]
        assert cg2d_mod.solve_for_pressure_cg2d_messages(first, minsq, last, nits, nmin) == mon[3 * n:3 * n + 3]


@pytest.mark.parametrize("v", [x for x in ALL if x[0] != "tutorial_barotropic_gyre"], ids=_ids)
def test_cg2d_p4_bitwise(v):
    """P=4 (shard_map(check_vma=True) on 4 fake CPU devices, ShardedExchanger): cg2d_solve from the dumped C01 inputs
    == C02 bitwise on every point, numIters and lastResidual equal, at every dumped step (baroclinic and optim: one
    tile per device; global_ocean: 9 tiles per device)."""
    from mitjax.eesupp import exch_maps as EM
    from mitjax.eesupp.shard import TileSharding
    e = G.experiment(*v)
    sz = e.cfg.size
    ds, _, _ = G.oracle(*v)
    sh = TileSharding(EM.load_maps(v[0]), 4)
    T, REP = sh.TILES, sh.REP
    norm = G.stdout_cg2dnorm(*v)[1]
    for it in ds.iterations():
        cg2dh, params = G.cg2dh_from_dump(*v, it, norm)

        def body(ex, ops, b, x, nrm, tol):
            c = cg2dh.replace(**{k: G.xy(ops[k], k, sz) for k in G.OPERATOR}, cg2dNorm=nrm, cg2dTolerance_sq=tol)
            out = cg2d_mod.cg2d_solve(G.xy(b, "cg2d_b", sz), G.xy(x, "cg2d_x", sz), params.cg2dMaxIters, -1,
                                      cfg=e.cfg, cg2dh=c, params=params, ex=ex)
            return out[1].data, ex.first_device(out[5]), ex.first_device(out[4])
        f = sh.shard_map(body, in_specs=(T, {k: T for k in G.OPERATOR}, T, T, REP, REP), out_specs=(T, REP, REP))
        x4, n4, last4 = f(sh.ex, {k: sh.put_tiles(getattr(cg2dh, k).data) for k in G.OPERATOR},
                          sh.put_tiles(G.dump2d(ds, it, "C01_cg2d_inputs", "cg2d_b")),
                          sh.put_tiles(G.dump2d(ds, it, "C01_cg2d_inputs", "cg2d_x")), cg2dh.cg2dNorm,
                          cg2dh.cg2dTolerance_sq)
        ref = G.dump2d(ds, it, "C02_cg2d_solution", "cg2d_x")
        ours = sh.unpad(x4)
        assert np.all(np.isfinite(ours)) and int(np.sum(ours != ref)) == 0, (it, int(np.sum(ours != ref)))
        assert int(n4) == ds.scalar(it, "C02_cg2d_solution", "numIters")
        assert float(last4) == ds.scalar(it, "C02_cg2d_solution", "lastResidual")


# ---------------------------------------------------------------------------------------------------------------------
# derivative: dot test and FD

def _solve_fn(v, it, composite):
    e = G.experiment(*v)
    sz = e.cfg.size
    ds, _, _ = G.oracle(*v)
    cg2dh, params = G.cg2dh_from_dump(*v, it, G.stdout_cg2dnorm(*v)[1])
    x0 = jnp.asarray(G.dump2d(ds, it, "C01_cg2d_inputs", "cg2d_x"))

    def f(bd, c, ex):
        out = cg2d_mod.cg2d_solve(G.xy(bd, "cg2d_b", sz), G.xy(x0, "cg2d_x", sz), params.cg2dMaxIters, -1,
                                  cfg=e.cfg, cg2dh=c, params=params, ex=ex)
        x = out[1].data
        return ex.EXCH_XY_RL(x) if composite else x       # solve_for_pressure.F:315 _EXCH_XY_RL( cg2d_x )
    b0 = jnp.asarray(G.dump2d(ds, it, "C01_cg2d_inputs", "cg2d_b"))
    return f, b0, cg2dh, params, sz


def _edge_weights(shape, sz):
    """Cotangent weights: 10 on halo lanes and on the first/last interior row and column of every tile."""
    w = np.ones(shape)
    OLx, OLy = sz.OLx, sz.OLy
    w[:, :OLy + 1, :] = 10.
    w[:, -OLy - 1:, :] = 10.
    w[:, :, :OLx + 1] = 10.
    w[:, :, -OLx - 1:] = 10.
    return w


@pytest.mark.parametrize("composite", [False, True], ids=["cg2d_x", "exch_xy_of_cg2d_x"])
@pytest.mark.parametrize("v", DOT_FD, ids=_ids)
def test_cg2d_dot_test(v, composite):
    """<ct, TL(db)> == <AD(ct), db> to 1e-11 relative, with cotangents on halo lanes and tile edges (weighted 10x),
    for CG2D's output and for EXCH_XY_RL(CG2D) as SOLVE_FOR_PRESSURE uses it; the adjoint is finite on every lane and
    exactly 0 on cg2d_b's halo lanes (CG2D reads the interior of cg2d_b only)."""
    ds, _, _ = G.oracle(*v)
    it = ds.iterations()[0]
    f, b0, cg2dh, _, sz = _solve_fn(v, it, composite)
    ex = G.exchanger(v[0])
    rng = np.random.default_rng(20261001)
    db = jnp.asarray(rng.standard_normal(b0.shape) * float(jnp.max(jnp.abs(b0))))
    ct = jnp.asarray(rng.standard_normal(b0.shape) * _edge_weights(b0.shape, sz))
    _, dx = jax.jit(lambda b, d, c, ex: jax.jvp(lambda bb: f(bb, c, ex), (b,), (d,)))(b0, db, cg2dh, ex)
    bbar = jax.jit(lambda b, ct, c, ex: jax.vjp(lambda bb: f(bb, c, ex), b)[1](ct)[0])(b0, ct, cg2dh, ex)
    dx, bbar = np.asarray(dx), np.asarray(bbar)
    assert np.all(np.isfinite(dx)) and np.all(np.isfinite(bbar))
    lhs, rhs = float(np.sum(dx * np.asarray(ct))), float(np.sum(bbar * np.asarray(db)))
    assert abs(lhs - rhs) <= 1e-11 * max(abs(lhs), abs(rhs)), (lhs, rhs)
    halo = ~np.broadcast_to(G.interior_mask(sz), bbar.shape)
    assert np.all(bbar[halo] == 0.0)


@pytest.mark.parametrize("v", DOT_FD, ids=_ids)
def test_cg2d_fd(v):
    """FD of J(b) = sum(w * EXCH_XY_RL(CG2D(b))) with respect to single points of cg2d_b (a tile-edge point, an inner
    point, a halo point) against the adjoint. The forward of the FD is the literal solve at a tightened target
    residual (cg2dTolerance_sq = (1e-13)^2: the experiment, not the tolerance of the comparison): J is linear in b,
    so the FD error is the solver's residual over h; h sweep 1e-3, 1e-2 of max|b|; agreement 1e-7 relative at the
    best h. A halo point of b has FD and adjoint exactly 0."""
    ds, _, _ = G.oracle(*v)
    it = ds.iterations()[0]
    f, b0, cg2dh, _, sz = _solve_fn(v, it, composite=True)
    ex = G.exchanger(v[0])
    tight = cg2dh.replace(cg2dTolerance_sq=np.float64(1e-13) ** 2)
    w = jnp.asarray(np.random.default_rng(7).standard_normal(b0.shape))
    J = jax.jit(lambda b, c, ex: jnp.sum(w * f(b, c, ex)))
    g = np.asarray(jax.jit(jax.grad(lambda b, c, ex: jnp.sum(w * f(b, c, ex))))(b0, cg2dh, ex))
    OLx, OLy = sz.OLx, sz.OLy
    pts = [(0, OLy, OLx + 3), (0, OLy + sz.sNy // 2, OLx + sz.sNx // 2), (0, OLy - 1, OLx + 2)]
    bmax = float(jnp.max(jnp.abs(b0)))
    for p in pts:
        best = np.inf
        for hrel in (1e-3, 1e-2):
            h = hrel * bmax
            e_k = jnp.zeros_like(b0).at[p].set(h)
            fd = (float(J(b0 + e_k, tight, ex)) - float(J(b0 - e_k, tight, ex))) / (2 * h)
            if g[p] == 0.0:
                assert fd == 0.0, (p, fd)
                best = 0.0
            else:
                best = min(best, abs(fd - g[p]) / abs(g[p]))
        assert best <= 1e-7, (p, best, g[p])


# ---------------------------------------------------------------------------------------------------------------------
# negative controls: a planted error must make the gate fail (measured counts asserted)

def _mutant(module, edits):
    """A copy of `module` with textual edits [(old, new, count)] applied (each `old` must occur `count` times)."""
    src = inspect.getsource(module)
    for old, new, count in edits:
        assert src.count(old) == count, (old, src.count(old))
        src = src.replace(old, new)
    m = types.ModuleType(module.__name__ + "_mutant")
    m.__dict__["__file__"] = module.__file__
    exec(compile(src, module.__file__ + ":mutant", "exec"), m.__dict__)
    sys.modules[m.__name__] = m
    return m


def _replay_mismatches(v, cg2d_fn=None, ex=None):
    ds, _, _ = G.oracle(*v)
    n = 0
    for it in ds.iterations():
        out = G.replay_cg2d(*v, it, cg2d_fn=cg2d_fn) if ex is None else _replay_with_ex(v, it, ex, cg2d_fn)
        res = G.compare_solution(*v, it, out)
        n += sum(r[1] + r[2] for r in res.values())
    return n


def _replay_with_ex(v, it, ex, cg2d_fn=None):
    e = G.experiment(*v)
    sz = e.cfg.size
    ds, _, _ = G.oracle(*v)
    cg2dh, params = G.cg2dh_from_dump(*v, it, G.stdout_cg2dnorm(*v)[1])
    b = G.xy(G.dump2d(ds, it, "C01_cg2d_inputs", "cg2d_b"), "cg2d_b", sz)
    x = G.xy(G.dump2d(ds, it, "C01_cg2d_inputs", "cg2d_x"), "cg2d_x", sz)
    fn = cg2d_fn or cg2d_mod.cg2d
    f = jax.jit(lambda b, x, c, ex: fn(b, x, params.cg2dMaxIters, -1, cfg=e.cfg, cg2dh=c, params=params, ex=ex))
    return jax.tree.map(np.asarray, f(b, x, cg2dh, ex))


@jax.tree_util.register_pytree_node_class
class _ReversedTileSum(Exchanger):
    """Planted error: GLOBAL_SUM_TILE_RL adds the tiles in reverse order."""

    def global_sum_tile(self, phiTile):
        return gs.global_sum_tile(jnp.asarray(phiTile)[::-1])


def _reversed(ex):
    leaves, aux = ex.tree_flatten()
    return _ReversedTileSum.tree_unflatten(aux, leaves)


# measured counts of differing C02 values (cg2d_x points + scalars) over the three dumped steps
@pytest.mark.parametrize("v", ALL, ids=_ids)
def test_negative_control_residual_test_one_iteration_late(v):
    """Stopping test evaluated on the previous iteration's err_sq (one extra iteration): the replay gate fails."""
    m = _mutant(cg2d_mod, [
        ("        it2d, done, cg2d_x, cg2d_r, cg2d_s, cg2d_q, eta_qrNM1, err_sq, actualIts = c\n",
         "        it2d, done, cg2d_x, cg2d_r, cg2d_s, cg2d_q, eta_qrNM1, err_sq, actualIts = c\n"
         "        err_prev = err_sq\n", 1),
        ("        done = err_sq < cg2dTolerance_sq\n", "        done = err_prev < cg2dTolerance_sq\n", 1)])
    assert _replay_mismatches(v, cg2d_fn=m.cg2d) > 0


@pytest.mark.parametrize("v", [x for x in ALL if x[0] != "tutorial_barotropic_gyre"], ids=_ids)
def test_negative_control_global_sum_reversed_tile_order(v):
    """GLOBAL_SUM_TILE_RL in reverse tile order: the replay gate fails (multi-tile variants; the barotropic gyre has
    one tile, where the order cannot matter)."""
    assert _replay_mismatches(v, ex=_reversed(G.exchanger(v[0]))) > 0


def test_negative_control_global_sum_order_cannot_bite_on_one_tile():
    v = ("tutorial_barotropic_gyre", "input")
    assert _replay_mismatches(v, ex=_reversed(G.exchanger(v[0]))) == 0


@pytest.mark.parametrize("v", ALL, ids=_ids)
def test_negative_control_work_array_zeroing(v):
    """cg2d_r/cg2d_s zeroing (cg2d.F:142-147) removed entirely: cg2d_s(i,j) is read before it is written (:256-257
    at it2d=1), so the replay fails (NaN). With only the RING part of the zeroing removed (interior zeroed), the
    replay stays bitwise: EXCH_S3D_RL writes every ring point the stencils read (the four corners are never read),
    measured on all four layouts. Both measured here."""
    m_all = _mutant(cg2d_mod, [("    cg2d_r = cg2d_r.at[i1, j1].set(0.0)\n", "", 1),
                               ("    cg2d_s = cg2d_s.at[i1, j1].set(0.0)\n", "", 1)])
    assert _replay_mismatches(v, cg2d_fn=m_all.cg2d) > 0
    m_ring = _mutant(cg2d_mod, [("    cg2d_r = cg2d_r.at[i1, j1].set(0.0)\n", "    cg2d_r = cg2d_r.at[i, j].set(0.0)\n", 1),
                                ("    cg2d_s = cg2d_s.at[i1, j1].set(0.0)\n", "    cg2d_s = cg2d_s.at[i, j].set(0.0)\n", 1)])
    assert _replay_mismatches(v, cg2d_fn=m_ring.cg2d) == 0


@pytest.mark.parametrize("v", CONSTANT_OPERATOR, ids=_ids)
def test_negative_control_ini_cg2d_loop_bound(v):
    """INI_CG2D's main-diagonal loop from 1 instead of 0 (master's bounds `DO j=0,sNy; DO i=0,sNx`, ini_cg2d.F:199-200):
    the operator gate fails (aC2d(0,.) / aC2d(.,0) stay 0, so pW/pS of the first row and column change)."""
    m = _mutant(ini_cg2d_mod, [("    j0 = loop_j(0, sNy)", "    j0 = loop_j(1, sNy)", 1),
                               ("    i0 = loop_i(0, sNx)", "    i0 = loop_i(1, sNx)", 1)])
    e = G.experiment(*v)
    params = ini_parms_cg2d(e)
    grid, surface = G.teacher_geometry(*v)
    ex = G.exchanger(v[0])
    c = jax.jit(lambda g, s, p, ex: m.ini_cg2d(cfg=e.cfg, grid=g, surface=s, params=p, ex=ex))(grid, surface, params,
                                                                                                ex)
    res = G.compare_operator(c, *v, 0, "C01_cg2d_inputs")
    assert sum(r[1] for r in res.values()) > 0
