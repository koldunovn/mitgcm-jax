"""global_ocean.cs32x15/input.seaice (M4 step 6, lane M4CS32ICE): the EXF front of the step (S00_begin ..
S02_load_fields), gated bitwise (every point of every tile incl. halos and cube corners) against lane A's FTZ oracle
(plan decision 13, session 3), the dumps-on run job27878831-jdon (the standard build's objects with crtfastmath.o;
session 2: job27855988-jdon, the same EXF values): iteration 36000 from the Model's own initial carry
(pickup.0000036000), 36001 and 36002 teacher-forced (m4cs32ice_gate.teacher_carry), with measured negative controls.

Gaps E1-E10 (lane M4CS32ICE's list): a build without ALLOW_ATM_WIND (useAtmWind .FALSE.), the C-grid stress
read by EXF_SET_UV and exchanged on the cube (exf_getforcing.F:200-204), EXF_WIND from the stress with a wspeed file,
the Large&Yeager04 bulk arms, monthly records without pkg/cal (EXF_GetFFieldRec / GET_PERIODIC_INTERVAL),
ALLOW_RUNOFTEMP; EXF_ALLOW_TIDES and ALLOW_ZENITHANGLE compiled but unused.
Blind spots: EXF_ALLOW_TIDES without a tide file (phiTide2d = +0. with or without the arm; asserted from the
parameters, phiTide2d is not dumped); the ALLOW_ZENITHANGLE fields zen_albedo / zen_fsol_* are dumped at X01 but not
carried (never written with useExfZenAlbedo / useExfZenIncoming .FALSE.).
Controls (measured in session 2, dev jobs 27877933, 27877954, 27877968): Large&Yeager04's huol clamp input planted
x(1+1e-12) (exf_bulkformulae.F:377, the LY04-only MIN site) -> X04 first differs; ALLOW_RUNOFTEMP planted off ->
X01-X05 differ only in `runoftemp` (dumped by every X stage, not loaded without the option), X06 in Qnet; the record
bookkeeping (atempStartTime shifted by one day) -> X01's atemp differs at every iteration. Planting
ALLOW_BULK_LARGEYEAGER04 off routes into solve4Stress = .FALSE. (wspeedfile without the LY04 arm, :230-236), which
the port refuses."""

import pytest

from mitjax.tests import m4cs32ice_gate as G

FRONT = ("S00_begin", "S01_update_rstar_F", "X01_exf_getffields", "X02_exf_radiation", "X03_exf_wind",
         "X04_exf_bulkformulae", "X05_exf_hflux_sflux", "X06_exf_mapfields", "S02_load_fields")
EARLY = ("S00_begin", "S01_update_rstar_F", "X01_exf_getffields", "X02_exf_radiation", "X03_exf_wind")


@pytest.fixture(scope="module")
def r():
    return G.run()


def _fields(out, it):
    return {s: set(v) for (i, s), v in out.items() if i == it and v}


def test_front_36000_free_36001_36002_teacher_forced(r):
    assert r.its == [36000, 36001, 36002]
    for it in r.its:
        out, ncmp = G.front(r, it=it)
        assert {s for (i, s) in ncmp if i == it} >= set(FRONT), it
        assert all(ncmp[(it, s)] > 0 for s in FRONT), it
        assert {k: v for k, v in out.items() if v} == {}, it


def test_control_largeyeager04_huol_clamp(r, monkeypatch):
    """Planted: the input of Large&Yeager04's clamp huol = SIGN(MIN(ABS(huol),10.),huol) (exf_bulkformulae.F:377,
    a MIN site only the LY04 arm has) x (1 + 1e-12): X04 is the first stage that differs (evap, hl, hs)."""
    import mitjax.pkg.exf.exf_bulkformulae as EB
    orig = EB.MIN

    def planted(*a, p=None):
        if len(a) == 2 and isinstance(a[1], float) and a[1] == 10.0:
            return orig(a[0]*(1.0 + 1e-12), a[1], p=p)
        return orig(*a, p=p)
    monkeypatch.setattr(EB, "MIN", planted)
    out, _ = G.front(r)
    bad = _fields(out, 36000)
    assert "X04_exf_bulkformulae" in bad and bad["X04_exf_bulkformulae"] >= {"hs", "hl", "evap"}
    assert not set(bad) & set(EARLY)


def test_control_largeyeager04_off(r):
    """Planted: ALLOW_BULK_LARGEYEAGER04 off. With useAtmWind .FALSE. solve4Stress is then .FALSE.
    (exf_bulkformulae.F:236), the arm global_ocean.cs32x15/code_ad runs (ported by lane M4ADCS32ICE, :324-333; this
    test refused it before): X04 is the first stage that differs."""
    out, _ = G.front(r, off=("ALLOW_BULK_LARGEYEAGER04",))
    bad = _fields(out, 36000)
    assert "X04_exf_bulkformulae" in bad and bad["X04_exf_bulkformulae"] >= {"hs", "hl", "evap"}
    assert not set(bad) & set(EARLY)


def test_control_runoftemp_off(r):
    """Planted: ALLOW_RUNOFTEMP undefined. The runoff heat content (exf_mapfields.F:199-209) leaves Qnet (X06, S02);
    X01-X05 differ only in `runoftemp`, which every X stage of the dump build writes (read from runoftempFile by the
    oracle, never loaded by the port without the option)."""
    out, _ = G.front(r, off=("ALLOW_RUNOFTEMP",))
    bad = _fields(out, 36000)
    for s in ("X01_exf_getffields", "X02_exf_radiation", "X03_exf_wind", "X04_exf_bulkformulae",
              "X05_exf_hflux_sflux"):
        assert bad.get(s) == {"runoftemp"}, (s, bad.get(s))
    assert bad["X06_exf_mapfields"] == {"Qnet", "runoftemp"}
    assert bad["S02_load_fields"] == {"Qnet"}
    assert not set(bad) & {"S00_begin", "S01_update_rstar_F"}


def test_control_record_bookkeeping(r):
    """Planted: atempStartTime = 1296000 + 86400 (data.exf EXF_NML_02 sets 1296000.): the record weights of
    EXF_GetFFieldRec without pkg/cal (exf_getffieldrec.F:202-263) change, so atemp differs from X01 on at every
    iteration (36000 sits on a record boundary: myTime - StartTime + period/2 = 100 repeat periods)."""
    rr = G.planted_model(r, {(G.X, "EXF_NML_02", "atempStartTime"): 1296000. + 86400.})
    for it in r.its:
        out, _ = G.front(rr, it=it)
        bad = _fields(out, it)
        assert "atemp" in bad.get("X01_exf_getffields", set()), it
        assert not set(bad) & {"S00_begin", "S01_update_rstar_F"}, it


def test_blind_tides(r):
    """EXF_ALLOW_TIDES blind spot (phiTide2d is not dumped): without a tide file tidePot = tidePotconst = 0.
    (exf_init_varia.F:331-342), so EXF_MAPFIELDS writes phiTide2d = exf_outscal_tidePot*0. = +0. (:327), the value
    INI_FFIELDS gave it (ini_ffields.F:53): the arm cannot change a bit in this run."""
    exf = r.m.arrays.pkc["exfp"]
    assert exf.tidePotfile.strip() == ""
    assert float(exf.tidePotconst) == 0.0 and float(exf.exf_outscal_tidePot) == 1.0
