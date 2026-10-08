"""R4 global_ocean.90x40x15/input (plan Task 15c), M1 lane GO: the run driver's Model (INI_PARMS -> INITIALISE_FIXED ->
INITIALISE_VARIA incl. the r* sequence, CD_CODE_INI_VARS, GMREDI_INIT_VARIA, the periodic forcing preload) and its
FORWARD_STEP with no teacher forcing.

* every dumped stage of steps 36000-36002 (S00 ... S16 incl. S04 DO_OCEANIC_PHYS with GM/Redi and S05 THERMODYNAMICS)
  bitwise on every point incl. halos; negative control: rCD + 1 ulp changes S06;
* the whole run (python -m mitjax run's Model and run.forward, 10 steps): every %MON record (incl. advcfl) and every
  %SBO record identical to the oracle STDOUT, testreport digits >= the yardstick vs results/ and 16 vs the oracle on
  every check-list variable, the end-of-run pickups (pickup.ckptA, pickup_cd.ckptA; data and meta, 36 tiles) byte for
  byte, nothing on STDERR (no r* warning, no zero denominator);
* P=4 == P=1 (jit(shard_map(check_vma=True)) on 4 fake CPU devices, 36 tiles) for the whole run: every carry leaf and
  the per-step solver / CALC_R_STAR outputs bit for bit; the final carry's non-finite values exactly
  go_gate.NAN_STORAGE's.
Helpers: mitjax/tests/go_gate.py (driver_model, probed_steps, run_steps_p), advect_gate.py (whole_run,
run_verdict, pickup_diffs).
"""

import numpy as np

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

import jax  # noqa: E402

from mitjax.tests import advect_gate as ag  # noqa: E402
from mitjax.tests import go_gate as G  # noqa: E402
from mitjax.tests import r1_gate as rg  # noqa: E402
from mitjax.tests import r2_gate as r2  # noqa: E402

MUST = ("S00_begin", "S01_update_rstar_F", "S02_load_fields", "P01_external_forcing_surf", "P02_rho_sigma_ivdc",
        "P03_mxlayer", "P05_gmredi_tensor", "P06_gmredi_exch", "S04_oceanic_phys", "T01_residual_flow",
        "T11_temp_gT", "T13_temp_impl", "T02_temp_integrate", "T21_salt_gS", "T23_salt_impl", "T03_salt_integrate",
        "S05_thermodynamics_sync", "S06_dynamics", "S07_update_rstar_T", "S08_update_cg2d", "S09_solve_for_pressure",
        "S10_momentum_correction", "S11_integr_continuity", "S12_calc_rstar", "S15_tracers_correction",
        "S16_blocking_exchanges")


def test_steps_in_model_bitwise():
    m = G.driver_model("steps")
    ds, its, _ = G.rsg.oracle(*G.EXP)
    stages = tuple(ds.stages(its[0]))
    res, _ = G.probed_steps(m, 3, stages)
    for it, pr in res:
        assert set(MUST) <= set(pr), (it, sorted(set(MUST) - set(pr)))
        for st in stages:
            if st in pr:
                r = rg.compare_stage(ds, it, st, r2.stage_values(st, pr[st]))
                assert r and not r2.bad(r), (it, st, r2.bad(r))
    # negative control: the CD relaxation factor + 1 ulp
    p = m.arrays.params
    m.arrays = m.arrays.replace(params=p.replace(traced=dict(rCD=np.nextafter(np.float64(p.rCD), np.inf))))
    res, _ = G.probed_steps(m, 1, ("S06_dynamics",))
    r = rg.compare_stage(ds, its[0], "S06_dynamics", r2.stage_values("S06_dynamics", res[0][1]["S06_dynamics"]))
    assert r["gU"][3] > 0, r


def test_whole_run_monitor_sbo_digits_pickups():
    m, res, o = ag.whole_run(*G.EXP, "go")
    diffs, rows, nblocks = ag.run_verdict(*G.EXP, res, o)
    assert not any(diffs.values()), diffs
    assert nblocks == sum("%MON time_tsnumber" in r for r in o.raw) == 11 and len(rows) >= 21
    for name, ours, yard, vs_oracle in rows:
        assert ours >= yard, (name, ours, yard)
        assert vs_oracle >= 16, (name, vs_oracle)
    ours = [r for r in res.records if "%SBO" in r]
    theirs = [r.rstrip("\n") for r in o.raw if "%SBO" in r]
    assert len(ours) == len(theirs) == 44 and ours == theirs
    same, extra = ag.pickup_diffs(m, o)
    assert same and all(same.values()) and not extra, ([k for k, v in same.items() if not v], extra)
    assert any(k.startswith("pickup_cd.ckptA") for k in same) and any(k.startswith("pickup.ckptA") for k in same)
    assert res.stderr == []


def test_p4_equals_p1_whole_run():
    m = G.driver_model("p4")
    n = m.prm.time.nTimeSteps
    c1, o1 = G.run_steps_p(m, n)
    c4, o4 = G.run_steps_p(m, n, 4)
    la, lb = jax.tree.leaves(c1), jax.tree.leaves(c4)
    assert len(la) == len(lb) > 100
    nd = [int(np.count_nonzero(np.ascontiguousarray(a, np.float64).view(np.int64)
                                != np.ascontiguousarray(b, np.float64).view(np.int64))) for a, b in zip(la, lb)]
    assert not any(nd), nd
    for a, b in zip(o1, o4):
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            assert np.array_equal(np.asarray(x), np.asarray(y))
    # session 7: the final carry finite except the named NaN storage (go_gate.NAN_STORAGE: the dWtransU/V rim the
    # Fortran neither writes nor reads). The PTRACERS_FIELDS.h fields of the compiled but unused pkg/ptracers were NaN
    # there too until lane M4LAB session 2's audit (drivers/model.py): they now hold the never-written common's zero,
    # as the oracle dumps them, and are finite
    assert G.nonfinite_leaves(c1) == G.expected_nonfinite(G.EXP[0], m.cfg.size)
