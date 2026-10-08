"""MLAdjust end to end (M3 Task 30): verification/MLAdjust and its six variants through the driver Model.

The variants exercise the viscosity schemes of MOM_CALC_VISC with both momentum forms: `input` (biharmonic Leith +
LeithD + Smagorinsky, full Leith, vector invariant), AhVrDv / AhStTn (harmonic; AhStTn with the strain-tension form:
MOM_HDISSIP), QGLeith (QG Leith viscosity: MOM_VISC_QGL_STRETCH / _LIMIT, sigmaRfield), AhFlxF / A4FlxF (flux form
with useVariableVisc and ISOTROPIC_COS_SCALING; A4FlxF starts from pickup.0000000036), QGLthGM (QG Leith also in
GM/Redi: GMREDI_CALC_QGLEITH, 'linear' taper, GM_AdvForm). Every variant has implicitViscosity (MOM_U/V_IMPLICIT_R).

1. Free-running steps 1-3 (no teacher forcing): every dumped stage of lane A's dumps-on run (`jdon`), the per-level
   DYNAMICS stages (D00a, D00b / D00c) included, on every point of every tile, halos included (element equality, bit
   patterns, finite), plus the grid / CG2D operator / absent-GMREDI records (mladjust_gate.compare_extra).
   Negative control: QGLeith with the QG arm switched off (viscC2LeithQG_ne_0 = .FALSE.) differs at D00c (steps 2-3;
   step 1 starts from rest).
2. The whole run (12 steps) through drivers/run.forward: every %MON record and banner identical to the oracle STDOUT,
   testreport digits >= the yardstick and 16 against the oracle; the end-of-run pickups byte-identical; %CHECKPOINT.

Tier 1x (about 3 min per variant and test: one whole-step program per variant).
"""

import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

VARIANTS = ("input", "input.A4FlxF", "input.AhFlxF", "input.AhStTn", "input.AhVrDv", "input.QGLeith",
            "input.QGLthGM")
# dumped stages the step does not probe: CG2D's internals (C01/C02: compared through S09 and the cg2d lines of the
# whole run), the initialisation stages (I00-I02: the end of INITIALISE_VARIA is the S00_begin record of nIter0,
# compared), the exchange probe X00, and G00 (our Grid's geometry is gated by the grid lane)
NOT_PROBED = ("C01_cg2d_inputs", "C02_cg2d_solution", "I00_pickup_read", "I01_read_pickup", "I02_ini_fields",
              "X00_exch_probe", "G00_geometry")


@pytest.fixture(autouse=True)
def _release_executables():
    """Drop the compiled programs after each test (vm.max_map_count, PORTING_LESSONS 2026-10-02)."""
    import gc

    import jax
    yield
    jax.clear_caches()
    gc.collect()


@pytest.mark.parametrize("inp", VARIANTS)
def test_steps_1_3_every_stage(inp):
    from mitjax.tests import goadk_model_gate as gmg
    from mitjax.tests import mladjust_gate as G
    ds, its, _ = G.oracle(inp)
    m = G.model(inp)
    out, _ = G.run_steps(m, ds, its)
    assert [it for it, _ in out] == list(its[:3])
    dyn = "D00c_mom_vecinv" if m.params.vectorInvariantMomentum else "D00b_mom_fluxform"
    for it, res in out:
        assert not gmg.bad(res), (it, gmg.bad(res))
        left = [(s, n) for s, n in gmg.not_compared(ds, it, res) if s not in NOT_PROBED]
        assert not left, (it, left)
        assert len(res.get(dyn, {})) == 4 * m.cfg.size.Nr and len(res.get("D00a_phi_hyd", {})) == 4 * m.cfg.size.Nr
        assert "S06_dynamics" in res and "T23_salt_impl" in res, sorted(res)


def test_negative_control_qg_arm_off():
    """QGLeith with MOM_CALC_VISC's QG Leith arm switched off (the static flag of viscC2LeithQG .NE. 0): the per-level
    MOM_VECINV tendencies differ -- the QG code runs and the D00c comparison sees it. Not in step 1: the run starts
    from rest, so every viscosity term of step 1 multiplies a zero flow (job 27841246); steps 2-3 are compared."""
    from mitjax.tests import goadk_model_gate as gmg
    from mitjax.tests import mladjust_gate as G
    ds, its, _ = G.oracle("input.QGLeith")
    m = G.model("input.QGLeith")
    assert m.params.viscC2LeithQG_ne_0
    a = m.arrays.replace(params=m.arrays.params.replace(static={"viscC2LeithQG_ne_0": False}))
    out, _ = G.run_steps(m, ds, its, nsteps=3, arrays=a)
    bad = [gmg.bad(res) for _, res in out]
    assert not bad[0], bad[0]
    assert any("D00c_mom_vecinv" in b and sum(v[1] for v in b["D00c_mom_vecinv"].values()) > 0 for b in bad[1:]), \
        [sorted(b) for b in bad]


@pytest.mark.parametrize("inp", VARIANTS)
def test_whole_run_and_pickups(inp):
    from mitjax.drivers.run import forward
    from mitjax.tests import advect_gate as ag
    from mitjax.tests import mladjust_gate as G
    from mitjax.tests import monitor_gate as mg
    m = G.model(inp)
    res = forward(m)
    o = mg.oracle(G.EXP, inp)
    diffs, rows, nblocks = ag.run_verdict(G.EXP, inp, res, o)
    assert diffs == {"mon": 0, "banner": 0} and nblocks == 13, (diffs, nblocks)
    assert all(ours >= yard and vs >= 16 for _, ours, yard, vs in rows), rows
    pk, extra = ag.pickup_diffs(m, o)
    assert pk and all(pk.values()) and not extra, (pk, extra)
    assert [r for r in res.records if "CHECKPOINT" in r] == [r for r in o.raw if "CHECKPOINT" in r]
