"""tutorial_global_oce_latlon/input end to end (PTRACERS lane, plan Task 26): one passive tracer (the age-tracer
overrides of tutorial_global_oce_latlon/code: PTRACERS_FORCING_SURF, PTRACERS_APPLY_FORCING) with GM/Redi
(R5's `gm=` passed through PTRACERS_INTEGRATE's CALC_3D_DIFFUSIVITY and GAD_CALC_RHS) and DST3FL vertical advection.

1. The driver Model (mitjax/drivers/model.py) against S00_begin of nIter0: every dumped field bitwise.
2. FORWARD_STEP steps 1-3 from that state with the package state (pk: gm, genarr, cost) carried, every probe stage
   the jdon run dumps (P01-P06, S00-S16, T01-T04, T11-T23) bitwise on every point, halos included; negative control:
   a 1-ulp change of the tracer's initial value makes T04_ptracers_integrate differ.
3. The whole run through mitjax/drivers/run.forward: every %MON record (dynamic and ptracer blocks) and banner
   identical to the oracle STDOUT, testreport digits >= the yardstick and 16 against the oracle; the driver's
   end-of-run permanent pickups (pickup, pickup_cd, pickup_ptracers of iteration 20, both tiles, .data and .meta)
   byte-identical and the %CHECKPOINT record.

Exchanges: the exchanger is built in memory from the run's own exchange probe (X00_exch_probe), as the M2 maps are
not registered yet (eesupp/exch_maps.MAP_SHA256). Tier 1x (about 6 min).
"""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

EXP = ("tutorial_global_oce_latlon", "input")


@functools.lru_cache(maxsize=None)
def oracle():
    from mitjax.tests import rstar_gate as rsg
    return rsg.oracle(*EXP, "jdon")


def model(exp=EXP, tag="latlon"):
    """A fresh driver Model (run directory linked under $MJX_RUNS/ptracers, never reused)."""
    import mitjax.drivers.model as dm
    from mitjax import paths
    from mitjax.drivers.run import load_experiment, make_rundir
    from mitjax.eesupp.exch_maps import build_maps
    from mitjax.eesupp.exchange import Exchanger
    from mitjax.tests import rstar_gate as rsg
    exp_dir = paths.UPSTREAM / "verification" / exp[0]
    e = load_experiment(exp_dir, exp[1])
    rd = make_rundir(exp_dir, exp[1], paths.RUNS / "ptracers" / f"{tag}-{time.time_ns()}")
    ds, _, _ = rsg.oracle(*exp, "jdon")
    return dm.Model(e, rd, ex=Exchanger(build_maps(ds)))


@functools.lru_cache(maxsize=None)
def shared_model():
    return model()


def _bad(res):
    return {k: v for k, v in res.items() if v[0] == "shape" or any(v[1:])}


def run_steps(m, carry0, ds, its, n=3, cost_compare=None):
    """{(step, stage): compare_stage} for every probe stage the oracle `ds` dumps, n steps of FORWARD_STEP from the
    driver carry `carry0` (state, ff, phi0surf, flow[, pk]): the package state pk is carried when the Model has one.
    `cost_compare(ds, it, cost)`: the comparison of S18_cost_tile (the cost.h scalars, one record per tile)."""
    from mitjax.model.src.forward_step import forward_step
    from mitjax.tests import r1_gate as rg
    from mitjax.tests import r2_gate as r2
    stages = sorted({s for (_, s, _) in ds.keys(its[0])})

    def step(a, carry, iloop, t, it):
        probes = {}

        def probe(stage, values):
            if stage in stages:
                probes[stage] = values
        state, ff, phi0 = carry[:3]
        pk = carry[4] if len(carry) > 4 else None
        state, ff, phi0, t, it, o = forward_step(iloop, t, it, cfg=m.cfg, grid=a.grid, params=a.params, fp=m.fp,
                                                 eos=a.eos, cg2dh=a.cg2dh, cg2d_params=a.cg2d_params, state=state,
                                                 ff=ff, phi0surf=phi0, ex=a.ex, probe=probe, pk=pk, pkc=a.pkc,
                                                 pks=m.pks)
        return (state, ff, phi0, carry[3]) + ((o["pk"],) if pk is not None else ()), t, it, probes
    f = jax.jit(step)
    carry = carry0
    t, it = m.start_counters()
    res = {}
    for k in range(n):
        carry, t, it, pr = f(m.arrays, carry, jnp.int32(k + 1), t, it)
        for st, v in pr.items():
            if st == "S18_cost_tile" and cost_compare is not None:
                res[(k, st)] = cost_compare(ds, its[0] + k, v)
            else:
                res[(k, st)] = rg.compare_stage(ds, its[0] + k, st, r2.stage_values(st, v))
    return res


def test_initial_state():
    from mitjax.tests import r1_gate as rg
    m = shared_model()
    ds, its, _ = oracle()
    res = rg.compare_stage(ds, its[0], "S00_begin", m.state0)
    assert len(res) >= 20 and {"pTracer_01", "gpTrNm1_01"} <= set(res)
    assert not _bad(res), _bad(res)


def test_steps_1_3_every_stage():
    m = shared_model()
    ds, its, _ = oracle()
    carry0 = m.initial_carry()
    assert len(carry0) > 4 and "gm" in carry0[4]                                # GM/Redi package state carried
    res = run_steps(m, carry0, ds, its)
    bad = {k: _bad(v) for k, v in res.items() if _bad(v)}
    assert not bad, bad
    assert sum(1 for (k, st) in res if st == "T04_ptracers_integrate") == 3
    assert len(res) >= 3 * 25                    # 25 probe stages per step (P01-P06, S00-S16, T01-T23)
    # negative control: the tracer's initial value changed at one wet interior point of level 5
    ctrl = run_steps(m, (planted_tracer(m, carry0[0]),) + tuple(carry0[1:]), ds, its, n=1)
    assert ctrl[(0, "T04_ptracers_integrate")]["pTracer_01"][3] > 0


def planted_tracer(m, s0, k=5):
    """The State s0 with pTracer_01 changed at the first wet interior point of level k (storage order): by 2**-30
    of its value, or by 1e-6 where that is larger (latlon's age tracer starts from 0 and gains 86400 s in step 1:
    a change below half an ulp of 86400 would be rounded away)."""
    from mitjax.farray import FArray
    sz = m.cfg.size
    p = np.array(s0.pTracer_01.data)
    wet = np.zeros(p.shape, bool)
    wet[:, k-1, sz.OLy:sz.OLy+sz.sNy, sz.OLx:sz.OLx+sz.sNx] = \
        np.asarray(m.grid.maskC.data)[:, k-1, sz.OLy:sz.OLy+sz.sNy, sz.OLx:sz.OLx+sz.sNx] == 1.
    q = tuple(a[0] for a in np.nonzero(wet))
    p[q] = p[q] + max(abs(p[q])*2.**-30, 1e-6)
    return s0.replace(pTracer_01=FArray(jnp.asarray(p), s0.pTracer_01.name, tiled=s0.pTracer_01.tiled,
                                        _dims=s0.pTracer_01.dims))


def test_whole_run_and_pickups():
    """The whole run (20 steps, 21 MONITOR + 21 ptracer blocks): records, digits, and the driver's end-of-run
    permanent pickups (DO_WRITE_PICKUP at iteration 20: PACKAGES_WRITE_PICKUP's CD_CODE, PTRACERS and GMREDI (none)
    pickups, then WRITE_PICKUP) byte-identical to the oracle's."""
    import filecmp
    from pathlib import Path
    from mitjax.drivers.run import forward
    from mitjax.tests import advect_gate as ag
    from mitjax.tests import monitor_gate as mg
    m = model()
    res = forward(m)
    o = mg.oracle(*EXP)
    diffs, rows, nblocks = ag.run_verdict(*EXP, res, o)
    assert diffs == {"mon": 0, "banner": 0} and nblocks == 21, diffs
    ptr_ours = [r for r in res.records if "trcstat_ptracer01" in r]
    ptr_oracle = [r for r in o.raw if "trcstat_ptracer01" in r]
    assert ptr_ours == ptr_oracle and len(ptr_ours) == 21 * 5
    names = [r[0] for r in rows]
    assert {"pt1mn", "pt1mx", "pt1av", "pt1sd"} <= set(names)
    assert all(ours >= yard and vs >= 16 for _, ours, yard, vs in rows), rows
    assert int(res.myIter) == 20 and res.pickups
    _, _, ordir = oracle()
    names = sorted(p.name for p in Path(ordir).iterdir() if p.name.startswith("pickup"))
    assert len(names) == 12, names
    for fn in names:
        assert filecmp.cmp(Path(m.rundir) / fn, Path(ordir) / fn, shallow=False), fn
    ours = sorted(p.name for p in Path(m.rundir).iterdir() if p.name.startswith("pickup") and not p.is_symlink())
    assert ours == names, ours
    assert [r for r in res.records if "CHECKPOINT" in r] == [r for r in o.raw if "CHECKPOINT" in r]
