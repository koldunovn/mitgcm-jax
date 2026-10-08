"""advect_xz/input.nlfs (plan Task 14's last variant; M1 lane GO, session 3): r* (select_rStar = 2, nonlinFreeSurf =
4) with staggerTimeStep = .TRUE. (DO_STAGGER_FIELDS_EXCHANGES and THERMODYNAMICS after the r* update,
forward_step.F:976-1009), exactConserv, the flux-limiter scheme 77 for theta and 3rd-order upwind for salt with
implicit vertical advection (GAD_IMPLICIT_R :162-259: GAD_FLUXLIMIT_IMPL_R + SOLVE_TRIDIAGONAL, GAD_U3C4_IMPL_R +
SOLVE_PENTADIAGONAL), momStepping = .FALSE.

* INITIALISE_VARIA (incl. its r* sequence) vs S00_begin bitwise; every dumped field of every stage of steps 1-3
  bitwise (element equality, bit patterns, finite; all points incl. the multi-wrap halo, sNy = 1 < OLy);
* the whole run through the run driver: every %MON record identical to the oracle STDOUT, testreport digits >= the
  yardstick vs results/ and >= 16 vs the oracle.
Helpers: mitjax/tests/advect_gate.py (ADVECT lane) with its Model (r2_gate.Model, which now runs INITIALISE_VARIA's
r* sequence)."""

import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

from mitjax.tests import advect_gate as ag  # noqa: E402
from mitjax.tests import r1_gate as rg  # noqa: E402

EXP = ("advect_xz", "input.nlfs")
# the probe stages FORWARD_STEP emits for this variant (r* and staggered stages included)
STAGES = ("S00_begin", "S01_update_rstar_F") + tuple(p for p in ag.PROBES if p != "S00_begin") + (
    "S07_update_rstar_T", "S12_calc_rstar", "S13_stagger_exchanges", "S14_thermodynamics_stagger")
MUST = ("S07_update_rstar_T", "S11_integr_continuity", "S12_calc_rstar",
        "S13_stagger_exchanges", "T11_temp_gT", "T13_temp_impl", "T23_salt_impl", "S14_thermodynamics_stagger",
        "S16_blocking_exchanges")


def _m():
    return ag.model(*EXP)


def test_initial_state_bitwise():
    m = _m()
    r = rg.compare_stage(m.ds, m.it0, "S00_begin", m.state0)
    assert len(r) >= 10 and not ag.bad(r), ag.bad(r)
    for n in ("hFacC", "rStarFacC", "wVel", "etaH"):
        assert n in r, n


def test_substeps_steps_1_to_3_bitwise():
    m = _m()
    dumped = m.ds.stages(m.it0)
    stages = tuple(s for s in STAGES if s in dumped)
    for st in MUST:
        assert st in stages, (st, stages)
    res, _ = ag.run_steps(m, m.params, 3, stages=stages)
    for k, per_stage in enumerate(res):
        assert set(MUST) <= set(per_stage), (k, sorted(per_stage))
        for st, r in per_stage.items():
            assert r, (k, st, "no fields compared")
            assert not ag.bad(r), (k, st, ag.bad(r))
            assert all(x[2] == 0 for x in r.values()), (k, st, "non-finite")


@pytest.mark.parametrize("planted", ["no_stagger_exchange"])
def test_negative_control_stagger_exchange(monkeypatch, planted):
    """DO_STAGGER_FIELDS_EXCHANGES skipped (the staggered THERMODYNAMICS reads unexchanged u, v, w) -> T11 differs
    by step 2 at the latest."""
    from mitjax.model.src import do_stagger_fields_exchanges as dsfe
    monkeypatch.setattr(dsfe, "do_stagger_fields_exchanges", lambda *a, state, **k: state)
    m = _m()
    res, _ = ag.run_steps(m, m.params, 2, stages=("T11_temp_gT", "T21_salt_gS"),
                          step_fn=m.step_fn(("T11_temp_gT", "T21_salt_gS")))
    nd = ag.ndiff(res)
    assert sum(sum(d.values()) for d in nd) > 0, nd


def test_whole_run_monitor_and_digits():
    m, res, o = ag.whole_run(*EXP, "go")
    diffs, rows, nblocks = ag.run_verdict(*EXP, res, o)
    assert not any(diffs.values()), diffs
    assert nblocks == sum("%MON time_tsnumber" in r for r in o.raw) > 1 and rows
    for name, ours, yard, vs_oracle in rows:
        assert ours >= yard, (name, ours, yard)
        assert vs_oracle >= 16, (name, vs_oracle)
