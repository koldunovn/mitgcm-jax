"""lab_sea/input (M4 step 5, lane M4LAB session 2): the ocean physics of lab_sea gated bitwise (every point incl.
halos) against lane A's dumps-on run job27855987-jdon, iterations 1-3, one step from the teacher carry (the oracle's
state at the start of the iteration; m4lab_gate.ocean_inputs / ocean_step_fn), each ported arm with a measured
negative control or an asserted blind spot. Session 4: the step runs the port's own SEAICE_MODEL (sessions 2-3 put
the oracle's P13_seaice_model values in its place while the sea-ice gaps L3-L12 were open); the sea-ice stages are
compared too.

Gaps (lane M4LAB's list): L14 KPP's SWFRAC arms (KPPuseSWfrac3D = .FALSE.: kpp_calc.F:652-663,
kpp_routines.F:504-511, :699-706, :835-842), L15 ALLOW_SALT_PLUME with useSALT_PLUME = .FALSE. (the argument arms of
KPP_CALC / KPP_FORCING_SURF / KPPMIX / BLDEPTH), L16 KPP_TRANSPORT_T/S with ALLOW_GMREDI (+ tmpFac*Kwz) and
ALLOW_SALT_PLUME (+ tmpFac1*saltPlumeFlux*...), L17 GMREDI_CALC_TENSOR's locMixLayer = KPPhbl, L18 'ldd97' (Lrho fields
and the taper), L19 the build without GM_NON_UNITY_DIAGONAL / GM_EXTRA_DIAGONAL (Kux = Kvy = GM_isopycK), L20 the
LONGSTEP checks of the GM transports (passive tracers only).
"""

import dataclasses

import numpy as np
import pytest

from mitjax.tests import m4lab_gate as L


@pytest.fixture(scope="module")
def md():
    return L.stub_model()


def test_ocean_physics_and_thermodynamics_teacher_forced(md):
    """The sea ice (I00 .. I04, Y01 .. Y09, P13: the port's own SEAICE_MODEL) and P01 .. S05 (EXTERNAL_FORCING_SURF,
    the k loop, the mixed layer, KPP_CALC, GMREDI_CALC_TENSOR, the exchanges, THERMODYNAMICS with KPP_TRANSPORT_T/S
    and the GM transports) bitwise on every point at iterations 1-3, one step from the teacher carry."""
    m, ds = md
    stages = L.DYN_STAGES + ("P13_seaice_model",) + L.OCEAN_STAGES
    res = L.ocean_diffs(m, ds, stages=stages)
    for it, r in res.items():
        assert set(stages) <= set(r), (it, sorted(set(stages) - set(r)))
        assert {"KPPhbl", "KPPfrac", "KPPghat", "KPPviscAz", "KPPdiffKzT", "KPPdiffKzS"} <= set(r["P07_kpp"])
        assert {"Kwx", "Kwy", "Kwz", "Kux", "Kvy"} <= set(r["P05_gmredi_tensor"])
        assert L.n_bad(r) == {}, (it, L.n_bad(r))


def test_kpp_swfrac_control(md):
    """Control of L14: the KPPuseSWfrac3D arm instead of SWFRAC changes KPPfrac (kpp_calc.F:644-651) and the
    boundary layer (BLDEPTH's bfsfc) at iteration 1."""
    m, ds = md
    kp = m.arrays.pkc["kpp_p"]
    a2 = m.arrays.replace(pkc=dict(m.arrays.pkc, kpp_p=kp.replace(KPPuseSWfrac3D=True)))
    bad = L.n_bad(L.ocean_diffs(m, ds, its=(1,), arrays=a2, stages=("P07_kpp",))[1])
    assert bad.get(("P07_kpp", "KPPfrac"), 0) > 50, bad


def test_kpp_transport_gmredi_control(md):
    """Control of L16: KPP_ghatUseTotalDiffus .FALSE. (tmpFac = 0: no Kwz in the non-local flux) changes gT and gS
    (T11, T21) at iteration 1."""
    m, ds = md
    kp = m.arrays.pkc["kpp_p"]
    a3 = m.arrays.replace(pkc=dict(m.arrays.pkc, kpp_p=kp.replace(KPP_ghatUseTotalDiffus=False)))
    bad = L.n_bad(L.ocean_diffs(m, ds, its=(1,), arrays=a3, stages=("P07_kpp", "T11_temp_gT", "T21_salt_gS"))[1])
    assert ("P07_kpp", "KPPfrac") not in bad
    assert bad.get(("T11_temp_gT", "gT_loc"), 0) > 0 and bad.get(("T21_salt_gS", "gS_loc"), 0) > 0, bad


def test_salt_plume_terms_blind(md):
    """L15/L16 salt plume: with useSALT_PLUME = .FALSE. the terms are tmpFac1 = 0 times saltPlumeFlux; they can change
    a bit only where surfaceForcingS is -0. (then -0. + +0. = +0.). Asserted blind here: the oracle's saltPlumeFlux
    is +0. on every point (SPsalFRAC is never set: SALT_PLUME_READPARMS returns before its defaults) and P01 holds no
    -0. in surfaceForcingS, iterations 1-3."""
    m, ds = md
    for it in (1, 2, 3):
        spf = np.asarray(ds.field(it, "I04_growth", "saltPlumeFlux"))
        assert np.all(spf == 0.) and not np.any(np.signbit(spf))
        sfs = np.asarray(ds.field(it, "P01_external_forcing_surf", "surfaceForcingS"))
        assert not np.any((sfs == 0.) & np.signbit(sfs))


def test_gmredi_tensor_ldd97_kpp_no_diagonal(md):
    """GMREDI_CALC_TENSOR teacher-forced from P02/P03/P07/S00 == P05 at iterations 1-3 (L17-L19). Controls at
    iteration 1: 'gkw91' instead of 'ldd97' changes Kwx/Kwy/Kwz; the build with both diagonal options changes
    Kux/Kvy. Blind: locMixLayer is read only by the 'fm07' taper (gmredi_slope_limit.F:229-391), so hMixLayer in
    place of KPPhbl changes nothing (measured)."""
    from mitjax.tests.m4col_gate import CppPlant
    m, ds = md
    for it, d in L.gm_tensor_diffs(m, ds).items():
        assert {"Kwx", "Kwy", "Kwz", "Kux", "Kvy"} <= set(d), it
        assert {n: v for n, v in d.items() if v} == {}, (it, d)
    d = L.gm_tensor_diffs(m, ds, its=(1,), gm_fix=lambda g: g.replace(GM_taper_scheme="gkw91"))[1]
    assert d["Kwx"] > 50 and d["Kwy"] > 50 and d["Kwz"] > 50, d
    cfg2 = dataclasses.replace(m.cfg, cpp=CppPlant(m.cfg.cpp, on=("GM_NON_UNITY_DIAGONAL", "GM_EXTRA_DIAGONAL")))
    d = L.gm_tensor_diffs(m, ds, its=(1,), cfg=cfg2)[1]
    assert d["Kux"] > 50 and d["Kvy"] > 50, d
    d = L.gm_tensor_diffs(m, ds, its=(1,), hbl_name="hMixLayer")[1]
    assert {n: v for n, v in d.items() if v} == {}, d


def test_longstep_and_salt_plume_checks(md):
    """L20: with ALLOW_LONGSTEP the GM transports and GMREDI_CALC_DIFF refuse a passive tracer (the LS_K* arms) and
    take the plain arm for T and S; L15: useSALT_PLUME raises in KPP."""
    from mitjax.pkg.gmredi.gmredi_calc_diff import GAD_TR1, gmredi_calc_diff
    from mitjax.pkg.gmredi.gmredi_xtransport import gmredi_xtransport
    m, _ = md
    assert m.cfg.cpp.ALLOW_LONGSTEP
    with pytest.raises(NotImplementedError, match="LS_Kux"):
        gmredi_xtransport(GAD_TR1, 1, 0, 1, 0, 1, None, None, None, None, cfg=m.cfg, grid=None, gm=None)
    with pytest.raises(NotImplementedError, match="LS_Kwz"):
        gmredi_calc_diff(0, 1, 0, 1, 0, 1, None, GAD_TR1, cfg=m.cfg, grid=None, gm=None)
    from mitjax.pkg.kpp.kpp_routines import _salt_plume_args
    cfg_sp = dataclasses.replace(m.cfg, use=tuple((k, True if k.lower() == "usesalt_plume" else v)
                                                  for k, v in m.cfg.use))
    with pytest.raises(NotImplementedError, match="useSALT_PLUME"):
        _salt_plume_args(cfg_sp, 0., 0., "BLDEPTH")
