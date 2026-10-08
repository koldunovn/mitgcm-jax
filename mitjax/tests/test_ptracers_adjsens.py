"""tutorial_tracer_adjsens/input_ad, the forward Model (PTRACERS lane, plan Task 26; the adjoint is a later session).

The build (code_ad: ALLOW_AUTODIFF_TAMC, NONLIN_FRSURF r*, CG2D_NSA, SHORTWAVE_HEATING, pkg/kpp compiled with useKPP
off, GM/Redi with GM_AdvForm and GM_ExtraDiag, the genarr3d control xx_ptr1, COST_TRACER) runs through the driver
Model with these PTRACERS-lane arms: INI_CG2D's cg2dTargetResWunit tolerance (with INI_GLOBAL_DOMAIN's globalArea,
n2dWetPts), INI_NLFS_VARS' and CALC_R_STAR's ALLOW_AUTODIFF parts, CTRL_MAP_INI_GENARR of xx_ptr1, then
CONVECTIVE_ADJUSTMENT_INI after PTRACERS_INIT_VARIA, SWFrac3D (SWFRAC) in INI_FORCING and the short-wave terms,
KPP_CALC_DUMMY in DO_OCEANIC_PHYS, CG2D_NSA in SOLVE_FOR_PRESSURE, COST_TRACER in COST_TILE and COST_FINAL.

1. INI_CG2D's tolerance line and INI_GLOBAL_DOMAIN's globalArea, n2dWetPts against the oracle STDOUT.
2. The initial state (S00_begin of nIter0) bitwise.
3. FORWARD_STEP steps 1-3, every probe stage the jdon run dumps (P01-P06, S00-S18, T01-T23 incl. T04 and the cost
   scalars of S18) bitwise; negative control: a small change of the tracer's initial value shows at T04.
4. The whole run: every %MON record and banner, the cg2d_nsa / cg2d_init_res / cg2d_iters lines and the COST_FINAL
   lines (objf_tracer per tile, local and global fc) identical to the oracle STDOUT.

Exchanges from the run's own exchange probe (the M2 maps are not registered). Tier 1x (about 5 min).
"""

import functools

import numpy as np

from mitjax.tests.test_ptracers_latlon import _bad, model, planted_tracer, run_steps

EXP = ("tutorial_tracer_adjsens", "input_ad")


@functools.lru_cache(maxsize=None)
def oracle():
    from mitjax.tests import rstar_gate as rsg
    return rsg.oracle(*EXP, "jdon")


@functools.lru_cache(maxsize=None)
def shared_model():
    return model(EXP, "adjsens")


def _raw(name):
    from mitjax.tests import monitor_gate as mg
    return list(mg.oracle(*EXP).prm[name].values())[0].strip()


def test_ini_cg2d_tolerance_and_global_domain():
    from mitjax.model.src.ini_cg2d import ini_cg2d_tolerance, ini_cg2d_tolerance_message
    from mitjax.model.src.ini_global_domain import ini_global_domain_2d
    from mitjax.tests import monitor_gate as mg
    m = shared_model()
    assert not m.cg2dh.cg2dNormaliseRHS
    n2d, area = ini_global_domain_2d(cfg=m.cfg, grid=m.grid, ex=m.ex)
    # WRITE_0D_RL prints 16 significant digits ('(1PE24.15)'): our values printed alike
    assert [f"{float(n2d):.15E}", f"{float(area):.15E}"] == [_raw("n2dWetPts"), _raw("globalArea")]
    tol = ini_cg2d_tolerance(m.cg2dh.cg2dNorm, n2d, area, m.cg2d_params)
    assert float(tol*tol) == float(m.cg2dh.cg2dTolerance_sq)
    line = [r for r in mg.oracle(*EXP).raw if "INI_CG2D: cg2dTolerance" in r]
    assert line == ["(PID.TID 0000.0001) " + ini_cg2d_tolerance_message(tol, n2d, area)]


def test_initial_state():
    from mitjax.tests import r1_gate as rg
    m = shared_model()
    ds, its, _ = oracle()
    res = rg.compare_stage(ds, its[0], "S00_begin", m.state0)
    assert len(res) >= 30 and {"pTracer_01", "gpTrNm1_01", "theta", "salt"} <= set(res)
    assert not _bad(res), _bad(res)


def _cost_compare(ds, it, cost):
    from mitjax.tests import r1_gate as rg
    out = {}
    for (_, s_, n) in ds.keys(it):
        if s_ == "S18_cost_tile" and hasattr(cost, n):
            recs = ds.tiles(it, s_, n)
            ref = np.array([recs[t].data.ravel()[0] for t in sorted(recs)])
            ours = np.broadcast_to(np.asarray(getattr(cost, n), np.float64), ref.shape)
            out[n] = rg.compare_field(ours[:, None, None, None], ref[:, None, None, None])
    return out


def test_steps_1_3_every_stage():
    m = shared_model()
    ds, its, _ = oracle()
    carry0 = m.initial_carry()
    assert {"gm", "kpp", "genarr", "cost"} <= set(carry0[4])
    res = run_steps(m, carry0, ds, its, cost_compare=_cost_compare)
    bad = {k: _bad(v) for k, v in res.items() if _bad(v)}
    assert not bad, bad
    assert sum(1 for (k, st) in res if st == "T04_ptracers_integrate") == 3
    assert sum(1 for (k, st) in res if st == "S18_cost_tile" and len(res[(k, st)]) == 3) == 3
    assert len(res) >= 3 * 31                    # P01-P06, S00-S12, S15, S16, S18, T01-T23 per step
    ctrl = run_steps(m, (planted_tracer(m, carry0[0]),) + tuple(carry0[1:]), ds, its, n=1,
                     cost_compare=_cost_compare)
    assert ctrl[(0, "T04_ptracers_integrate")]["pTracer_01"][3] > 0


def test_whole_run_and_cost():
    from mitjax.drivers.run import forward
    from mitjax.tests import advect_gate as ag
    from mitjax.tests import monitor_gate as mg
    m = model(EXP, "adjsens")
    res = forward(m, write_pickups=False)
    o = mg.oracle(*EXP)
    diffs, rows, nblocks = ag.run_verdict(*EXP, res, o)
    assert diffs == {"mon": 0, "banner": 0} and nblocks == 5, diffs
    assert all(ours >= yard and vs >= 16 for _, ours, yard, vs in rows), rows
    for key in ("cg2d_nsa:", "cg2d_init_res", "cg2d_iters", "cg2d_last_res", "fc =", "objf_"):
        ours = [r for r in res.records if key in r]
        theirs = [r for r in o.raw if key in r]
        assert ours == theirs and ours, key


def test_som81_forward():
    """input_ad.som81 (staggerTimeStep, T/S SOM schemes 80/81: GAD_SOM_ADVECT's noFlowAcrossSurf and
    FORCING_SURF_RELAX's r* rescaling): initial state, steps 1-3 at every dumped stage (32 per step), the whole run's
    %MON / cg2d_nsa / COST_FINAL / ptracer records identical to the oracle."""
    from mitjax.drivers.run import forward
    from mitjax.tests import advect_gate as ag
    from mitjax.tests import monitor_gate as mg
    from mitjax.tests import r1_gate as rg
    from mitjax.tests import rstar_gate as rsg
    exp = ("tutorial_tracer_adjsens", "input_ad.som81")
    ds, its, _ = rsg.oracle(*exp, "jdon")
    m = model(exp, "som81")
    r0 = rg.compare_stage(ds, its[0], "S00_begin", m.state0)
    assert len(r0) >= 30 and not _bad(r0), _bad(r0)
    res = run_steps(m, m.initial_carry(), ds, its, cost_compare=_cost_compare)
    bad = {k: _bad(v) for k, v in res.items() if _bad(v)}
    assert not bad and len(res) >= 3 * 32, bad
    run = forward(m, write_pickups=False)
    o = mg.oracle(*exp)
    diffs, rows, nblocks = ag.run_verdict(*exp, run, o)
    assert diffs == {"mon": 0, "banner": 0}, diffs
    for key in ("cg2d_nsa:", "fc =", "objf_", "trcstat_ptracer01"):
        ours, theirs = [r for r in run.records if key in r], [r for r in o.raw if key in r]
        assert ours == theirs and ours, key
