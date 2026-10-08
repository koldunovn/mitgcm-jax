"""Lane M4ADCOL (M4 step 7, sessions 2-3): the forward of 1D_ocean_ice_column/input_ad (code_ad build) against lane A's
dumps-on oracle job27856057-jdon (helpers: mitjax/tests/adcol_gate.py), with useECCO as data.pkg sets it (session 3:
pkg/ecco ported; its own files are gated in test_m4adcol_ecco.py).

1. Steps 0-2 free from the Model's initial carry (INITIALISE_VARIA + CAL/EXF/SEAICE/KPP/CTRL with pkg/cal/COST set-up),
   every dumped stage (49 per iteration, 129 (iteration, stage) pairs, 2049 field records) bitwise on every point.
2. Negative controls of this session's arms, teacher-forced (adcol_gate.teacher_step), each measured: the 0.97
   longwave emissivity of SEAICE_SOLVE4TEMP planted to the namelist's SEAICE_ice_emiss (I04_growth from iteration 1,
   the first step with ice; EXF_LWDOWN_WITH_EMISSIVITY planted on bites first in EXF_RADIATION), IMPLDIFF of gU / gV skipped (D02_after_impl_visc), the Large & Yeager (2009) drag reached
   with umax = 1 (X04_exf_bulkformulae); blind spots asserted: ALLOW_DRAG_LARGEYEAGER09 off (with cdrag_8 = 0 and
   umax = 1000 the 2009 form is the old one bit for bit) and SEAICE_EXCLUDE_FOR_EXACT_AD_TESTING off (the excluded
   block only moves snow: HSNOW = 0 throughout this run, its MAX/MIN give +-0 that changes no stage).
3. CTRL_GET_GEN_REC's pkg/cal arm on the run's ten steps (period 864000 s, start 19790101): records 1 and 2, `first`
   at step 1 only, fac = 1 - t/864000; a planted period of 3600 s bites.
4. The ten-step run through the run driver: %MON records identical, the sea-ice cost line ` --> f_ice` and
   `local fc` / `global fc = 6.91935719996437E+05` identical to the oracle in every digit, costfunction_seaice.0000
   identical, pickups byte-identical; every COST_FINAL line of the oracle (f_gencost, f_gentim2d, f_genarr3d,
   f_ice, the Writing / Reading lines, early / local / global fc) identical and in order, costfunction.0000,
   costfunction_ctrl.0000 and costfunction_seaice.0000 byte-identical; negative control: lastinterval planted to
   18000 s changes f_ice and fc.
Costs: about 15 min on a CPU node.
"""

import jax
import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

from mitjax.tests import adcol_gate as A  # noqa: E402


@pytest.fixture(autouse=True)
def _release_executables():
    import gc
    yield
    jax.clear_caches()
    gc.collect()


def test_steps_0_2_free_every_dumped_stage_bitwise():
    r = A.adcol_run()
    bad, ncmp, miss = A.free_steps(r, 3)
    assert not bad, bad
    assert not any(miss.values()), miss
    for it in r.its[:3]:
        assert ncmp[(it, "P07_kpp")] == 6 and ncmp[(it, "P10_kpp_exch")] == 6, (it, ncmp)
        assert ncmp[(it, "I04_growth")] == 24 and ncmp[(it, "S03_ctrl_map_forcing")] == 12, (it, ncmp)
    print("free steps 0-2: (iteration, stage) pairs", len(ncmp), "field records", sum(ncmp.values()))
    assert len(ncmp) == 129 and sum(ncmp.values()) == 2049, (len(ncmp), sum(ncmp.values()))


def _impldiff_skip(monkeypatch):
    import mitjax.model.src.impldiff as I
    real = I.impldiff

    def planted(iMin, iMax, jMin, jMax, tracerId, KappaRX, recip_hFac, gTracer, **kw):
        if tracerId < 0:
            return gTracer                     # planted: no implicit vertical viscosity
        return real(iMin, iMax, jMin, jMax, tracerId, KappaRX, recip_hFac, gTracer, **kw)
    monkeypatch.setattr(I, "impldiff", planted)


@pytest.mark.parametrize("plant, expect", [
    ("lw_emissivity", [(0, "X02_exf_radiation"), (1, "X02_exf_radiation"), (2, "X02_exf_radiation")]),
    ("lw_hardwired", [(1, "I04_growth"), (2, "I04_growth")]),
    ("impldiff_skip", [(0, "D02_after_impl_visc"), (1, "D02_after_impl_visc"), (2, "D02_after_impl_visc")]),
    ("umax_1", [(0, "X04_exf_bulkformulae"), (1, "X04_exf_bulkformulae"), (2, "X04_exf_bulkformulae")]),
    ("ly09_off", []),
    ("exclude_off", []),
])
def test_negative_controls_new_arms(plant, expect, monkeypatch):
    from mitjax.tests import m4col_gate as C
    r = A.adcol_run()
    m = r.m
    kw = {}
    if plant == "lw_emissivity":                 # also flips EXF_RADIATION's arm (exf_radiation.F:71-79): first
        kw["cfg"] = C.cfg_with(m, on=("EXF_LWDOWN_WITH_EMISSIVITY",))     # bite there, at every step
    elif plant == "lw_hardwired":                # SEAICE_SOLVE4TEMP's 0.97 alone: the namelist's ice emissivity
        import mitjax.pkg.seaice.seaice_solve4temp as S4
        monkeypatch.setattr(S4, "LW_HARDWIRED", 0.97001763668430343479)  # data.seaice (input_ad) SEAICE_ice_emiss
    elif plant == "impldiff_skip":
        _impldiff_skip(monkeypatch)
    elif plant == "umax_1":
        kw["arrays"] = A.with_exfp(m, umax=1.0)
    elif plant == "ly09_off":
        kw["cfg"] = C.cfg_with(m, off=("ALLOW_DRAG_LARGEYEAGER09",))
    elif plant == "exclude_off":
        kw["cfg"] = C.cfg_with(m, off=("SEAICE_EXCLUDE_FOR_EXACT_AD_TESTING",))
    got = A.first_bad(r, **kw)
    print(plant, got)
    assert [(it, s) for it, s, _ in got] == expect, got


def test_ctrl_get_gen_rec_cal_arm():
    from mitjax.pkg.cal.cal_fulldate import cal_FullDate
    from mitjax.pkg.ctrl.ctrl_get_gen_rec import gen_rec_table
    m = A.adcol_run().m
    tp = m.prm.time
    sd = tuple(cal_FullDate(19790101, 0, cal=m.cal))           # data.ctrl xx_gentim2d_startdate1/2
    fac, first, changed, c0, c1 = gen_rec_table(sd, 864000.0, cal=m.cal, startTime=tp.startTime,
                                                deltaTClock=tp.deltaTClock, nIter0=tp.nIter0,
                                                nTimeSteps=tp.nTimeSteps)
    t = np.arange(10) * 3600.0
    assert list(c0) == [1]*10 and list(c1) == [2]*10, (c0, c1)
    assert list(first) == [True] + [False]*9 and not any(changed), (first, changed)
    assert np.array_equal(fac, 1.0 - t/864000.0), fac
    # the set-up's clock carries the same table
    clock = m.pks["clock"]
    assert clock["useCAL"] and all(np.array_equal(a, b) for a, b in zip(clock["table"][(sd, 864000.0)],
                                                                         (fac, first, changed, c0, c1)))
    # negative control: a period of one step moves the records every step
    _, _, ch2, c02, _ = gen_rec_table(sd, 3600.0, cal=m.cal, startTime=tp.startTime, deltaTClock=tp.deltaTClock,
                                     nIter0=tp.nIter0, nTimeSteps=tp.nTimeSteps)
    assert list(c02) == list(range(1, 11)) and list(ch2) == [False] + [True]*9, (c02, ch2)


@pytest.fixture(scope="module")
def whole():
    from mitjax import paths as P
    from mitjax.drivers.model import Model
    from mitjax.drivers.run import forward, load_experiment, make_rundir
    from mitjax.tests import advect_gate as ag
    from mitjax.tests import monitor_gate as mg
    exp_dir = P.UPSTREAM / "verification" / A.EXP[0]
    e = load_experiment(exp_dir, A.EXP[1])
    rundir = make_rundir(exp_dir, A.EXP[1], ag.out_dir("adcol-whole"))
    m = Model(e, rundir)
    res = forward(m)
    return m, res, mg.oracle(*A.EXP)


def _cost_lines(records):
    return [r for r in records if " fc = " in r or "--> f_" in r or "cost function info" in r]


def test_whole_run_monitor_cost_pickups(whole):
    from mitjax.tests import advect_gate as ag
    m, res, o = whole
    diffs, rows, nblocks = ag.run_verdict(*A.EXP, res, o)
    print(diffs, nblocks, rows)
    assert diffs == {"mon": 0, "banner": 0} and nblocks == 11, (diffs, nblocks)
    ours, theirs = _cost_lines(res.records), _cost_lines(o.raw)
    print("\n".join(ours))
    assert ours == theirs and len(ours) == 23, (ours, theirs)
    assert "(PID.TID 0000.0001)  global fc =   6.91935719996437E+05" in ours
    od = o.stdout_path.parent
    for f in ("costfunction_seaice.0000", "costfunction_ctrl.0000", "costfunction_ecco.0000"):
        assert (m.rundir / f).read_bytes() == (od / f).read_bytes(), f
    pk, extra = ag.pickup_diffs(m, o)
    assert pk and all(pk.values()) and not extra, (pk, extra)


def test_whole_run_cost_negative_control(whole):
    """lastinterval (cost.h, traced) planted to 18000 s: SEAICE_COST_TEST adds only the steps with myTime > 18000 s
    (seaice_cost_test.F:79) with tempVar = 1/((1+18000)/3600): f_ice and fc change."""
    import jax.numpy as jnp
    from mitjax.drivers.run import forward
    m, res, o = whole
    a = m.arrays
    m.arrays = a.replace(pkc=dict(a.pkc, lastinterval=jnp.float64(18000.0)))
    try:
        res2 = forward(m, write_pickups=False)
    finally:
        m.arrays = a
    l1, l2 = _cost_lines(res.records), _cost_lines(res2.records)
    f1 = [x for x in l1 if "--> f_ice" in x]
    f2 = [x for x in l2 if "--> f_ice" in x]
    print(f1, f2)
    assert f1 and f2 and f1 != f2
    assert [x for x in l1 if "global fc" in x] != [x for x in l2 if "global fc" in x]
