"""M3 Task 30: tutorial_reentrant_channel/input end to end (GM/Redi bolus form with the dm95 taper, pkg/rbcs
temperature relaxation, OS7MP advection in x, y and r, implicit vertical viscosity MOM_U/V_IMPLICIT_R, staggered
time step; pkg/layers output only).

1. Free-running steps (no teacher forcing) from the driver Model: every dumped stage of iterations 0-2 of lane A's
   dumps-on run (job27840386-jdon), every point of every tile (element equality, bit patterns, finite): the state,
   forcing, GM/Redi tensor and bolus flow, the tracer stages (T10 the OS7MP advective tendency, T11 with the RBCS
   relaxation term of APPLY_FORCING_T, T13 the implicit vertical diffusion), DYNAMICS' result with the implicit
   vertical viscosity (S06), the per-level DYNAMICS records (D00a CALC_PHI_HYD, D00b MOM_FLUXFORM: every level), the
   grid. Not compared (not probed): CG2D's internals (C01/C02: its inputs and solution are compared through S09), the
   exchange probe X00, INI_FIELDS' output I02, and the G00 / S00 / S04 fields our Grid / State does not carry.
2. Negative control: tauRelaxT x (1 + 1e-7) (RBCS_PARAMS.h, a traced Params value) changes T11 of iteration 1.
3. The whole run (10 steps through drivers/run.forward, as `python -m mitjax run`): every %MON record and MONITOR
   banner identical to the oracle STDOUT (9 blocks; the trAdv_CFL lines need the staggered-step CFL time of
   drivers/run.chunk_ends), testreport digits 16 vs results/ (= the yardstick) and 16 vs the oracle on all 17
   variables, the end-of-run pickup files byte-identical.
4. P=4 == P=1: the whole run inside jit(shard_map(check_vma=True)) on 4 fake CPU devices (one tile each), every
   carry leaf and per-step output bit for bit; the final carry finite.
Costs: about 15 min on a CPU compute node.
"""

import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

EXP = ("tutorial_reentrant_channel", "input")
JDON = "job27840386-jdon"
# I02_ini_fields: INI_FIELDS' output; INITIALISE_VARIA changes the State after it (the end of INITIALISE_VARIA is
# the S00_begin record of iteration 0, compared)
NOT_PROBED = ("C01_cg2d_inputs", "C02_cg2d_solution", "X00_exch_probe", "I02_ini_fields")
# G00 / S00 / S04 records of fields the Model keeps elsewhere: the reference profiles and unit factors (Params, not
# GRID.h), the cg2d operator (CG2D.h) and, without NONLIN_FRSURF, the State-independent hFac (GRID.h, compared in G00)
NOT_CARRIED = {("G00_geometry", n) for n in ("dBdrRef", "gravitySign", "phiRef", "rVel2wUnit", "rhoFacC", "rhoFacF",
                                             "rkSign", "sRef", "tRef", "wUnit2rVel")}
NOT_CARRIED |= {("S00_begin", n) for n in ("aC2d", "aS2d", "aW2d", "pC", "pS", "pW", "hFacC", "hFacS", "hFacW",
                                           "recip_hFacC")}
NOT_CARRIED |= {("S04_oceanic_phys", n) for n in ("hFacC", "hFacS", "hFacW", "recip_hFacC")}


@pytest.fixture(autouse=True)
def _release_executables():
    """Each test compiles whole-step programs: drop them afterwards (a process may hold only vm.max_map_count
    memory mappings; PORTING_LESSONS, 2026-10-02)."""
    import gc

    import jax
    yield
    jax.clear_caches()
    gc.collect()


def _model():
    from mitjax.config.params import load
    from mitjax import paths
    from mitjax.drivers.model import Model
    from mitjax.io.dump import DumpSet
    top = paths.REFERENCE_RUNS / EXP[0] / EXP[1] / JDON
    return Model(load(*EXP), top / "rundir"), DumpSet(top / "dumps")


def test_free_steps_bitwise_every_dumped_stage():
    from mitjax.tests import goadk_model_gate as M
    m, ds = _model()
    out, _ = M.run_free(m, ds, 3)
    assert [it for it, _, _ in out] == [0, 1, 2]
    for it, res, _ in out:
        assert not M.bad(res), (it, M.bad(res))
        left = [(s, n) for s, n in M.not_compared(ds, it, res) if s not in NOT_PROBED and (s, n) not in NOT_CARRIED]
        assert not left, (it, left)
        assert len(res["D00a_phi_hyd"]) == len(res["D00b_mom_fluxform"]) == 4 * 49, it   # every level's records
        assert sum(len(r) for r in res.values()) == 293 + 2 * 4 * 49, it
        for s in ("T10_temp_adv", "T11_temp_gT", "T13_temp_impl", "S06_dynamics", "P05_gmredi_tensor",
                  "T01_residual_flow"):
            assert s in res, (it, s)


def test_negative_control_rbcs_tau():
    """tauRelaxT x (1 + 1e-7): T11 of iteration 1 differs while T10 (before the forcing) of that iteration does
    not. Iteration 0 cannot show it: the initial theta is the relaxation record itself (hydrogThetaFile =
    relaxTFile = temperature.50km.bin), so the RBCS term is exactly zero there."""
    from mitjax.tests import goadk_model_gate as M
    m, ds = _model()
    m.arrays = m.arrays.replace(params=m.params.replace(traced={"tauRelaxT": m.params.tauRelaxT * (1 + 1e-7)}))
    out, _ = M.run_free(m, ds, 2)
    assert not M.bad(out[0][1]).get("T11_temp_gT")
    b = M.bad(out[1][1])
    assert "T11_temp_gT" in b and "T10_temp_adv" not in b, sorted(b)


def test_whole_run_monitor_digits_pickups():
    from mitjax.tests import advect_gate as ag
    from mitjax.tests import cube_run_gate as C
    m, res, o = C.whole_run(*EXP, tag="m3-channel-whole")
    diffs, rows, nblocks = ag.run_verdict(*EXP, res, o)
    print("tutorial_reentrant_channel digits (name, ours vs results/, yardstick, ours vs oracle):", rows)
    assert nblocks == 9 and diffs == {"mon": 0, "banner": 0}, (nblocks, diffs)
    assert len(rows) == 17 and all(a >= y and a == 16 and s == 16 for _, a, y, s in rows), rows
    assert sum("%MON trAdv_CFL_u_max" in r for r in res.records) > 0
    files, _ = C.output_file_diffs(m, o)
    pk = {f: ok for f, ok in files.items() if f.startswith("pickup")}
    assert len(pk) == 8 and all(pk.values()), pk
    assert res.stderr == [], res.stderr[:5]


def test_p4_equals_p1_whole_run_and_finite():
    from mitjax.tests import go_gate as G
    m, c1, o1, c4, o4, ndiff, same = G.whole_run_pn(*EXP, 4, tag="m3-channel-pn")
    assert len(o1) == m.prm.time.nTimeSteps == 10
    assert ndiff == 0 and same, (ndiff, same)
    assert G.nonfinite_leaves(c1) == {}
    assert np.all(np.isfinite(np.asarray(c1[0].theta.data)))
