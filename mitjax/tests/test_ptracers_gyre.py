"""tutorial_advection_in_gyre/input end to end (PTRACERS lane, plan Task 26): the passive tracer wired into the model
(SOM advection with PTRACERS_ALLOW_DYN_STATE, pickup start at nIter0 = 259200 with the dye initialised from dye.bin).

1. The driver Model (mitjax/drivers/model.py: INI_PARMS, INITIALISE_FIXED, INITIALISE_VARIA with PTRACERS_INIT_VARIA)
   against S00_begin of nIter0: every dumped field bitwise (pTracer_01, gpTrNm1_01, surfaceForcingPTr_01 included).
2. FORWARD_STEP steps 1-3 (start iterations 259200-259202) from that state, every probe stage the jdon run dumps
   (S00..S16, T02/T03, T11-T13, T21-T23, T04_ptracers_integrate) bitwise on every point, halos included; negative
   control: a 1-ulp change of the dye's initial value makes T04 and the later stages differ.
3. The whole run through mitjax/drivers/run.forward: every %MON record (dynamic and ptracer blocks) and banner
   identical to the oracle STDOUT, testreport digits >= the yardstick for every check-list variable (pt1mn ...
   included) and >= 16 against the oracle; the driver's end-of-run pickups byte-identical.
4. The INI_PARMS values the PTRACERS lane ported (deltaTClock unset, viscAz, diffKzT, cAdjFreq < 0; CG2D_NSA's
   cg2dMinItersNSA and numItersMax) against the oracles' parameter printouts.

Exchanges: the exchanger is built in memory from the run's own exchange probe (X00_exch_probe), as the M2 maps are
not registered yet (eesupp/exch_maps.MAP_SHA256). useMNC: pkg/mnc is output only here (monitor_mnc, pickup_read_mnc
and pickup_write_mnc are .FALSE. in data.mnc; Model accepts it, GO lane). Tier 1x (about 1.5 min).
"""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

EXP = ("tutorial_advection_in_gyre", "input")


@functools.lru_cache(maxsize=None)
def oracle():
    from mitjax.tests import rstar_gate as rsg
    return rsg.oracle(*EXP, "jdon")


def model():
    """A fresh driver Model of the gyre (run directory linked under $MJX_RUNS/ptracers, never reused)."""
    import mitjax.drivers.model as dm
    from mitjax import paths
    from mitjax.drivers.run import load_experiment, make_rundir
    from mitjax.eesupp.exch_maps import build_maps
    from mitjax.eesupp.exchange import Exchanger
    exp_dir = paths.UPSTREAM / "verification" / EXP[0]
    e = load_experiment(exp_dir, EXP[1])
    rd = make_rundir(exp_dir, EXP[1], paths.RUNS / "ptracers" / f"gyre-{time.time_ns()}")
    ds, _, _ = oracle()
    return dm.Model(e, rd, ex=Exchanger(build_maps(ds)))


@functools.lru_cache(maxsize=None)
def shared_model():
    return model()


def _bad(res):
    return {k: v for k, v in res.items() if v[0] == "shape" or any(v[1:])}


def test_initial_state():
    from mitjax.tests import r1_gate as rg
    m = shared_model()
    ds, its, _ = oracle()
    res = rg.compare_stage(ds, its[0], "S00_begin", m.state0)
    assert len(res) >= 20 and {"pTracer_01", "gpTrNm1_01", "surfaceForcingPTr_01"} <= set(res)
    assert not _bad(res), _bad(res)


def run_steps(m, state0, n=3):
    """{(step, stage): compare_stage} for every probe stage the oracle dumps, n steps from state0."""
    from mitjax.model.src.forward_step import forward_step
    from mitjax.tests import r1_gate as rg
    from mitjax.tests import r2_gate as r2
    ds, its, _ = oracle()
    stages = sorted({s for (_, s, _) in ds.keys(its[0])})

    def step(a, state, ff, phi0, iloop, t, it):
        probes = {}

        def probe(stage, values):
            if stage in stages:
                probes[stage] = values
        out = forward_step(iloop, t, it, cfg=m.cfg, grid=a.grid, params=a.params, fp=m.fp, eos=a.eos,
                           cg2dh=a.cg2dh, cg2d_params=a.cg2d_params, state=state, ff=ff, phi0surf=phi0, ex=a.ex,
                           probe=probe)
        return out + (probes,)
    f = jax.jit(step)
    state, ff, phi0 = state0, m.ff0, m.phi0surf0
    t, it = m.start_counters()
    res = {}
    for k in range(n):
        state, ff, phi0, t, it, _, pr = f(m.arrays, state, ff, phi0, jnp.int32(k + 1), t, it)
        for st, v in pr.items():
            res[(k, st)] = rg.compare_stage(ds, its[0] + k, st, r2.stage_values(st, v))
    return res


def test_steps_1_3_every_stage():
    m = shared_model()
    res = run_steps(m, m.state0)
    bad = {k: _bad(v) for k, v in res.items() if _bad(v)}
    assert not bad, bad
    assert sum(1 for (k, st) in res if st == "T04_ptracers_integrate") == 3
    assert len(res) >= 3 * 19
    # negative control: the dye's initial value one ulp off
    from mitjax.farray import FArray
    p = np.array(m.state0.pTracer_01.data)
    q = np.unravel_index(np.argmax(p), p.shape)
    p[q] = np.nextafter(p[q], 0.)
    s0 = m.state0.replace(pTracer_01=FArray(jnp.asarray(p), m.state0.pTracer_01.name,
                                            tiled=m.state0.pTracer_01.tiled, _dims=m.state0.pTracer_01.dims))
    ctrl = run_steps(m, s0, n=1)
    assert ctrl[(0, "T04_ptracers_integrate")]["pTracer_01"][3] > 0


def test_whole_run():
    """The whole run (4 steps, 5 MONITOR + 5 ptracer blocks): records, digits, and the driver's end-of-run pickups
    (DO_WRITE_PICKUP with writePickupAtEnd: rolling 'ckptA'; PACKAGES_WRITE_PICKUP's PTRACERS_WRITE_PICKUP, then
    WRITE_PICKUP with useMNC and pickup_write_mnc = .FALSE., MDS_WRITE_FIELD with useSingleCpuIO): pickup,
    pickup_ptracers, pickup_somTRAC01 (.ckptA .data, .meta) byte-identical to the oracle's, with the PRINT_MESSAGE
    records of PTRACERS_WRITE_PICKUP and the %CHECKPOINT record."""
    import filecmp
    from pathlib import Path
    from mitjax.drivers.run import forward
    from mitjax.tests import advect_gate as ag
    from mitjax.tests import monitor_gate as mg
    m = model()
    res = forward(m)
    o = mg.oracle(*EXP)
    diffs, rows, nblocks = ag.run_verdict(*EXP, res, o)
    assert diffs == {"mon": 0, "banner": 0} and nblocks == 5, diffs
    ptr_ours = [r for r in res.records if "trcstat_ptracer01" in r]
    ptr_oracle = [r for r in o.raw if "trcstat_ptracer01" in r]
    assert ptr_ours == ptr_oracle and len(ptr_ours) == 25
    names = [r[0] for r in rows]
    assert {"pt1mn", "pt1mx", "pt1av", "pt1sd"} <= set(names)
    assert all(ours >= yard and vs >= 16 for _, ours, yard, vs in rows), rows
    _, _, ordir = oracle()
    names = sorted(p.name for p in Path(ordir).iterdir() if p.name.startswith("pickup") and ".ckptA." in p.name)
    assert len(names) == 6, names
    for fn in names:
        assert filecmp.cmp(Path(m.rundir) / fn, Path(ordir) / fn, shallow=False), fn
    keys = ("PTRACERS_WRITE_PICKUP", " to file: pickup_somTRAC", "%CHECKPOINT")
    assert [r for r in res.records if any(k in r for k in keys)] == [r for r in o.raw if any(k in r for k in keys)]


def _printed(exp, inp, name):
    from mitjax.io import stdout as so
    from mitjax.tests import monitor_gate as mg
    o = mg.oracle(exp, inp)
    return [so.fortran_number(v) if v not in ("T", "F") else v == "T" for v in o.prm[name].values()]


def test_ini_parms_ptracers_lane_values_vs_printout():
    """The INI_PARMS arms of this lane against the oracle printout: advection_in_gyre (deltaT and deltaTClock unset:
    deltaT = deltaTtracer, deltaTClock = deltaT; viscAz -> viscArNr; diffKzT -> diffKrNrT), tutorial_tracer_adjsens
    (cAdjFreq = -1 -> deltaTClock; cg2dMinItersNSA; numItersMax = 200 from its tamc.h, >= cg2dMaxIters)."""
    from mitjax.config.params import load
    from mitjax.model.src.cg2d_h import ini_parms_cg2d
    from mitjax.model.src.ini_parms import ini_parms_dyn
    from mitjax.model.src.ini_parms_tracer import ini_parms_tracer
    from mitjax.tests import init_gate as ig
    e = load(*EXP)
    prm = ig.params(*EXP)
    p = ini_parms_tracer(e, ini_parms_dyn(e, prm.grid, prm.time, prm.init), prm.time, prm.init)
    assert float(prm.time.deltaTClock) == _printed(*EXP, "deltaTClock")[0]
    assert list(np.asarray(p.viscArNr.data)) == _printed(*EXP, "viscArNr")
    assert list(np.asarray(p.diffKrNrT.data)) == _printed(*EXP, "diffKrNrT")
    assert list(np.asarray(p.dTtracerLev.data)) == _printed(*EXP, "dTtracerLev")
    assert ini_parms_cg2d(e).deltaTMom == _printed(*EXP, "deltaTMom")[0]
    ta = ("tutorial_tracer_adjsens", "input_ad")
    e2 = load(*ta)
    prm2 = ig.params(*ta)
    p2 = ini_parms_dyn(e2, prm2.grid, prm2.time, prm2.init)
    assert p2.cAdjFreq == _printed(*ta, "cAdjFreq")[0] == 86400.0
    c2 = ini_parms_cg2d(e2)
    assert c2.cg2dMinItersNSA == _printed(*ta, "cg2dMinItersNSA")[0]
    assert c2.numItersMax == 200 and c2.numItersMax >= c2.cg2dMaxIters
