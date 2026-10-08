"""global_ocean.cs32x15/input.seaice (M4 step 6, lane M4CS32ICE sessions 2-3): the sea ice on the cube, gated bitwise
(every point of every tile incl. halos and cube corners) against lane A's FTZ oracle, the dumps-on run job27878831-jdon
(plan decision 13: the standard build's objects linked with crtfastmath.o, FTZ/DAZ set at start, as XLA:CPU computes).

1. SEAICE_INIT_FIXED's curvilinear metric terms (seaice_init_fixed.F:295-356, gap S2) and the rest of SEAICE.h after
   the initialisation from pickup_seaice.0000036000 == I00_seaice_begin of 36000; control: SEAICEuseMetricTerms
   .FALSE. -> the eight k1/k2At* differ.
2. Each sea-ice routine teacher-forced from the oracle's fields before its stage, iterations 36000-36002
   (m4cs32ice_gate.kernel_diffs): Y01 SEAICE_GET_DYNFORCING from fu/fv (:207-237, gap S3), Y02/Y04 the forcing and
   ice strength (ATMOSPHERIC_LOADING phiSurf with useRealFreshWaterFlux), Y06 SEAICE_LSR with SEAICEuseStrImpCpl on
   the 12 tiles (S5), Y09 SEAICE_OCEAN_STRESS, I01 the clip (S4), I02 SEAICE_ADVECTION's three cube passes (S6), I03
   SEAICE_REG_RIDGE, I04 SEAICE_GROWTH with ALLOW_BALANCE_FLUXES compiled (S7). Each with a measured control.
3. The whole first step from the pickups (iteration 36000, the Model's own carry): every dumped stage S00 .. S16.

Subnormals (decision 11 inconsistency of session 2, resolved by decision 13): SEAICE_LSR's iterate decays through the
subnormal range at 36001 / 36002; the standard oracle (job27855988-jdon, gfortran keeps subnormals) differs from the
FTZ oracle there at 159 + 96 / 1 + 4 points of uIce / vIce, all below 2**-1021 (lane A's measurement,
reference/ftz_band.py). Against the FTZ oracle the port is bitwise everywhere; against the standard oracle it differs
at exactly the points where the FTZ oracle does (`test_standard_oracle_difference_is_the_ftz_difference`, a
measurement, not a band rule)."""

import dataclasses

import numpy as np
import pytest

from mitjax.tests import m4cs32ice_gate as G

KERNEL_STAGES = ("Y01_get_dynforcing", "Y02_ice_strength", "Y04_solver_inputs", "Y09_ocean_stress", "I01_dynsolver",
                 "I02_advdiff", "I03_reg_ridge", "I04_growth")
K_FIELDS = ("k1AtC", "k1AtU", "k1AtV", "k1AtZ", "k2AtC", "k2AtU", "k2AtV", "k2AtZ")


@pytest.fixture(scope="module")
def r():
    return G.run()


def _bad(d):
    return {it: {n: v for n, v in dd.items() if v} for it, dd in d.items()}


def test_seaice_init_metric_terms(r):
    d = G.seaice_init_diffs(r.m, r.ds, 36000)
    assert len(d) >= 19 and set(K_FIELDS) <= set(d) and {"AREA", "HEFF", "UICE", "TICES", "HEFFM"} <= set(d)
    assert {n: v for n, v in d.items() if v} == {}
    rr = G.planted_model(r, {(G.S, "SEAICE_PARM01", "SEAICEuseMetricTerms"): (False, "bool")})
    d2 = G.seaice_init_diffs(rr.m, r.ds, 36000)
    bad = {n: v for n, v in d2.items() if v}
    assert set(bad) == set(K_FIELDS) and min(bad.values()) > 10000, bad      # measured 10044 .. 11190


@pytest.mark.parametrize("stage", KERNEL_STAGES)
def test_kernel_teacher_forced(r, stage):
    d = G.kernel_diffs(r, stage)
    assert sorted(d) == r.its
    assert all(len(dd) >= {"Y01_get_dynforcing": 2, "Y09_ocean_stress": 10, "I02_advdiff": 6}.get(stage, 10)
               for dd in d.values()), d
    assert _bad(d) == {it: {} for it in r.its}


def test_lsr_bitwise_and_stdout(r):
    """SEAICE_LSR at 36000-36002 on every point (bitwise against the FTZ oracle); its STDOUT lines (residuals,
    iteration counts) equal the oracle's character for character."""
    from mitjax.pkg.seaice.seaice_lsr import lsr_stdout_lines
    import jax
    d = G.kernel_diffs(r, "Y06_lsr", values=True)
    ref = [ln.rstrip("\n") for ln in open(G.paths_rundir() / "output.txt") if ln.startswith(" SEAICE_LSR")]
    for k, it in enumerate(r.its):
        assert len(d[it][0]) >= 30 and not any(d[it][0].values()), (it, {n: v for n, v in d[it][0].items() if v})
        ours = lsr_stdout_lines(jax.tree_util.tree_map(jax.device_get, d[it][1]["_lsr_out"]))
        assert ours == ref[6*k:6*k + 6], it


def test_standard_oracle_difference_is_the_ftz_difference(r):
    """Measurement (decision 13): against the standard oracle (gfortran with subnormals) our SEAICE_LSR output differs
    at exactly as many points of each field as the FTZ oracle's own output does (both bit-counted at 36000-36002);
    the difference is not empty at 36001 (uIce: 159 points measured by lane A) and lies in the fields of lane A's band
    measurement (job 27878832): uIce / vIce, at 36002 also uIceNm1 / vIceNm1 and the signed zeros of e11 / e22 /
    FORCEX / FORCEY."""
    std = G.standard_dumps()
    d = G.kernel_diffs(r, "Y06_lsr", values=True)
    seen = {}
    for it, (dd, res) in d.items():
        for n in dd:
            ftz = G.bits_differ(r.ds.field(it, "Y06_lsr", n), std.field(it, "Y06_lsr", n))
            ours = G.bits_differ(res[n].data, std.field(it, "Y06_lsr", n))
            assert ours == ftz, (it, n, ours, ftz)
            if ftz:
                seen[(it, n)] = ftz
    assert {k for k in seen if k[0] == 36001} == {(36001, "UICE"), (36001, "VICE")}, seen
    assert {n for it, n in seen if it == 36002} == {"UICE", "VICE", "uIceNm1", "vIceNm1", "e11", "e22", "FORCEX",
                                                    "FORCEY"}, seen
    assert not any(it == 36000 for it, _ in seen) and seen[(36001, "UICE")] == 159, seen


def test_control_drag_south(r):
    """SEAICE_drag_south x (1 + 1e-12): Y01 differs (the yC < 0 arm, seaice_get_dynforcing.F:224-225)."""
    sp = r.m.arrays.pkc["sp"]
    sp = sp.replace(SEAICE_drag_south=sp.SEAICE_drag_south*(1.0 + 1e-12))
    d = G.kernel_diffs(r, "Y01_get_dynforcing", its=r.its[:1], sp=sp)
    assert d[36000]["TAUX"] > 0 and d[36000]["TAUY"] > 0, d


def test_control_rfw_phisurf(r):
    """useRealFreshWaterFlux planted .FALSE. in the sea ice's PARAMS.h view: phiSurf loses the sea-ice load
    (seaice_dynsolver.F:244-252 -> :253-259), FORCEX0 / FORCEY0 differ at Y02."""
    op = dataclasses.replace(r.m.arrays.pkc["op"], useRealFreshWaterFlux=False)
    d = G.kernel_diffs(r, "Y02_ice_strength", its=r.its[:1], op=op)
    assert d[36000]["FORCEX0"] > 0 and d[36000]["FORCEY0"] > 0, d


def test_control_strimpcpl_off(r):
    """SEAICEuseStrImpCpl planted .FALSE.: Y06 differs (seaice_lsr.F:1396-1397, :1704-1726, :1860-1882)."""
    sp = r.m.arrays.pkc["sp"].replace(SEAICEuseStrImpCpl=False)
    d = G.kernel_diffs(r, "Y06_lsr", its=r.its[:1], sp=sp)
    assert d[36000]["UICE"] > 1000 and d[36000]["VICE"] > 1000, d


def test_blind_clip_at_dumped_iterations(r):
    """Blind spot: at 36000-36002 no |uIce|, |vIce| of Y09 exceeds 0.40 (measured max 0.30), so the clip
    (seaice_dynsolver.F:388-410) changes nothing the dumps see; it binds at 36004 and 36006 (%MON seaice_uice_max =
    0.4), gated by the whole run and its clip-off control (test_m4cs32ice_run.py). The input of the clip is Y06's
    uIce / vIce (SEAICE_OCEAN_STRESS writes neither)."""
    sp = r.m.arrays.pkc["sp"].replace(SEAICE_clipVelocities=False)
    d = G.kernel_diffs(r, "I01_dynsolver", sp=sp)
    assert _bad(d) == {it: {} for it in r.its}
    for it in r.its:
        for n in ("UICE", "VICE"):
            assert float(np.abs(np.asarray(r.ds.field(it, "Y06_lsr", n))).max()) < 0.40


def test_control_advection_pass_order(r, monkeypatch):
    """Planted: SEAICE_ADVECTION's directional order swapped on every face (calc_fluxes_X <-> calc_fluxes_Y of
    seaice_advection.F:307-318): I02 differs."""
    import mitjax.pkg.seaice.seaice_advection as SA
    orig = SA._cs_pass_flags

    def swapped(nCFace, ipass):
        o, i, x, y = orig(nCFace, ipass)
        return o, i, y, x
    monkeypatch.setattr(SA, "_cs_pass_flags", swapped)
    d = G.kernel_diffs(r, "I02_advdiff", its=r.its[:1])
    assert d[36000]["HEFF"] > 0 and d[36000]["AREA"] > 0, d


def test_blind_advection_corner_fills(r, monkeypatch):
    """Blind spot (measured): FILL_CS_CORNER_TR_RL planted as a no-op in SEAICE_ADVECTION's cube passes (:356-359,
    :424-427, :570-573, :638-641) leaves I02 unchanged at 36000: the eight cube corners lie at low latitudes, where
    this run has no sea ice (the filled corner halos hold zeros either way)."""
    from mitjax.pkg.generic_advdiff import gad_advection as GA
    monkeypatch.setattr(GA, "_cs_fill", lambda fill4dir, A, flag, cs, kc: A)
    d = G.kernel_diffs(r, "I02_advdiff", its=r.its[:1])
    assert sum(d[36000].values()) == 0, d


def test_growth_balance_fluxes_switches(r):
    """ALLOW_BALANCE_FLUXES is compiled (CPP_OPTIONS.h) with selectBalanceEmPmR = 0 and balanceQnet = .FALSE.: the
    tile integrals and global means of seaice_growth.F:2440-2475, :2572-2664 write nothing a later statement reads
    (blind spot, asserted from the switches; I04 is gated bitwise above). With balanceQnet planted on the port
    refuses."""
    m = r.m
    op = m.arrays.pkc["op"]
    assert m.cfg.cpp.flag("ALLOW_BALANCE_FLUXES")
    assert op.selectBalanceEmPmR == 0 and op.balanceQnet is False
    with pytest.raises(NotImplementedError, match="ALLOW_BALANCE_FLUXES"):
        G.kernel_diffs(r, "I04_growth", its=r.its[:1], op=dataclasses.replace(op, balanceQnet=True))


def test_free_step_36000_every_stage(r):
    """FORWARD_STEP of iteration 36000 from the Model's own carry (the pickups), every dumped stage S00 .. S16
    (EXF, SEAICE_MODEL, the ocean) on every point."""
    out, ncmp = G.step_compare(r, 36000, carry=r.m.initial_carry())
    for s in ("I00_seaice_begin", "Y06_lsr", "I02_advdiff", "I04_growth", "P13_seaice_model", "S04_oceanic_phys",
              "S09_solve_for_pressure", "T03_salt_integrate", "S16_blocking_exchanges"):
        assert ncmp.get(s, 0) > 0, s
    assert len(ncmp) >= 45
    assert {s: v for s, v in out.items() if v} == {}


def test_seaice_model_teacher_forced(r):
    """SEAICE_MODEL from the oracle's fields before I00_seaice_begin, 36000-36002: every dumped stage (I00, Y01 ..
    Y09, I01 .. I04, P13) on every point, bitwise against the FTZ oracle (session 2 against the standard oracle: the
    LSR's subnormal uIce / vIce differed, carried unchanged to P13)."""
    res = G.seaice_model_diffs(r)
    for it, stages in res.items():
        assert {"I00_seaice_begin", "Y06_lsr", "I02_advdiff", "I04_growth", "P13_seaice_model"} <= set(stages), it
        for stage, d in stages.items():
            assert len(d) > 0 and {n: v for n, v in d.items() if v} == {}, (it, stage)


def test_blind_advection_interioronly(r, monkeypatch):
    """Blind spot (docs/ISSUES_UPSTREAM.md): SEAICE_ADVECTION sets interiorOnly in pass 1 only
    (seaice_advection.F:303-318), GAD_ADVECTION also in passes 2 and 3. With GAD's flags planted, I02 is unchanged
    at 36000-36002 (measured): the overlap points the seaice flags update in passes 2/3 are read by no later
    statement that reaches HEFF / AREA / HSNOW (the update :314-320 of SEAICE_ADVDIFF is interior-only)."""
    import mitjax.pkg.seaice.seaice_advection as SA
    from mitjax.pkg.generic_advdiff import gad_advection as GA
    monkeypatch.setattr(SA, "_cs_pass_flags", GA._cs_pass_flags)
    d = G.kernel_diffs(r, "I02_advdiff")
    assert _bad(d) == {it: {} for it in r.its}
