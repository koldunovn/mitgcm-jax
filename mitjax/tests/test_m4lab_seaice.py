"""lab_sea/input (M4 step 5, lane M4LAB session 3): the sea-ice gaps L4-L12, gated bitwise (every point incl. halos)
against lane A's dumps-on run job27855987-jdon, iterations 1-3, teacher-forced from the oracle's fields before each
stage, each with a measured negative control or an asserted blind spot (lane M4LAB session 3).

The Model is m4lab_gate.stub_model(): lab_sea/input with only ALLOW_SITRACER planted off (gap L3, not ported; the
SItracer arms do not write a field these kernels read or return). Session 4: L3 is ported and the plant is gone
(stub_model is the plain Model); SItracer is gated end to end in test_m4lab_run.py.

1. SEAICE_GROWTH (I04_growth): the 7 non-ITD categories (L11) and saltPlumeFlux (L12) with SALT_PLUME_READPARMS's
   never-written commons (SPsalFRAC = 0., SaltPlumeSouthernOcean .FALSE.). Controls: SPsalFRAC = 1 (saltPlumeFlux
   differs: the oracle's field is +0. everywhere), SEAICE_multDim 1 (TICES and the fluxes differ).
2. SEAICE_ADVDIFF with SEAICEadvScheme 7, OS7MP (L10, I02_advdiff). Control: scheme 77.
3. SEAICE_FREEDRIFT (L9, Y03_freedrift). Control: SEAICE_waterDrag*(1+2**-40).
4. SEAICE_LSR without SEAICE_ALLOW_LSR_FLEX (the max-norm SOLV_NCHECK test), SEAICE_LSR_ZEBRA, bottom drag with
   SEAICEbasalDragK2 = 0, SEAICE_OLx = 0, no-slip, SEAICEetaZmethod 0, the spherical metric terms of RHSU/V
   (L4-L7, Y06_lsr from Y04_solver_inputs). Controls: ZEBRA planted off; SEAICEselectMetricTerms 0 in the LSR.
   Blind spot (asserted): CbotC = 0 adds +0. to dragSym (no -0. in DWATN*COSWAT), so bottom drag cannot change a bit.
6. The whole FORWARD_STEP free from the pickups, iterations 1-3, every dumped stage (session 4: with ALLOW_SITRACER).
5. The whole SEAICE_MODEL from the oracle's fields before I00_seaice_begin: every dumped stage Y01..I04 and P13
   (incl. ZETA without SEAICE_ZETA_SMOOTHREG, L8, computed in SEAICE_LSR). Control: SEAICE_ZETA_SMOOTHREG planted
   on (ZETA differs at Y06).
"""

import dataclasses

import numpy as np
import pytest

from mitjax.tests import m4lab_gate as L

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def md():
    return L.stub_model()


def _nz(d):
    return {k: v for k, v in d.items() if v}


def test_growth_categories_and_salt_plume(md):
    m, ds = md
    res = L.growth_diffs(m, ds)
    for it, d in res.items():
        assert {"TICES", "Qnet", "Qsw", "saltFlux", "saltPlumeFlux", "HEFF", "AREA", "EmPmR"} <= set(d), it
        assert _nz(d) == {}, it
    sp1 = dataclasses.replace(m.arrays.pkc["spp"], SPsalFRAC=np.float64(1.0))
    ctl = L.growth_diffs(m, ds, spp=sp1)
    for it, d in ctl.items():
        assert set(_nz(d)) == {"saltPlumeFlux"} and d["saltPlumeFlux"] > 0, (it, d)
    sp = m.arrays.pkc["sp"].replace(SEAICE_multDim=1, SEAICE_PDF=(1.0,) + (0.0,)*6)
    ctl = L.growth_diffs(m, ds, its=(1,), sp=sp)
    assert ctl[1]["TICES"] > 0 and ctl[1]["Qnet"] > 0, ctl


def test_advdiff_os7mp(md):
    m, ds = md
    res = L.advdiff_diffs(m, ds)
    for it, d in res.items():
        assert {"HEFF", "AREA", "HSNOW"} <= set(d), it
        assert _nz(d) == {}, it
    sp = m.arrays.pkc["sp"].replace(SEAICEadvScheme=77, SEAICEadvSchHeff=77, SEAICEadvSchArea=77,
                                    SEAICEadvSchSnow=77)
    ctl = L.advdiff_diffs(m, ds, its=(1,), sp=sp)
    assert all(ctl[1][n] > 0 for n in ("HEFF", "AREA", "HSNOW")), ctl


def test_freedrift(md):
    m, ds = md
    res = L.freedrift_diffs(m, ds)
    for it, d in res.items():
        assert set(d) == {"uice_fd", "vice_fd"}, it
        assert _nz(d) == {}, it
    sp = m.arrays.pkc["sp"]
    ctl = L.freedrift_diffs(m, ds, its=(1,), sp=sp.replace(SEAICE_waterDrag=np.float64(sp.SEAICE_waterDrag)
                                                             * (1 + 2.0**-40)))
    assert ctl[1]["uice_fd"] > 0 and ctl[1]["vice_fd"] > 0, ctl


def test_lsr_plain_zebra(md):
    m, ds = md
    res = L.lsr_diffs(m, ds)
    for it, (d, _) in res.items():
        assert {"UICE", "VICE", "ZETA", "ETA", "DWATN", "FORCEX", "FORCEY", "uice_fd", "CbotC"} <= set(d), it
        assert _nz(d) == {}, it
    from mitjax.tests.m4col_gate import cfg_with
    from mitjax.ad.seaice_lsr_rule import seaice_lsr_forward_only
    import jax
    cfg = cfg_with(m, off=("SEAICE_LSR_ZEBRA",))

    def f(sp, op, sf, st, myTime, myIter):
        return seaice_lsr_forward_only(myTime, myIter, sf, cfg=cfg, sp=sp, op=op, grid=m.arrays.grid, state=st,
                                       ex=m.ex)
    ctl = L.lsr_diffs(m, ds, its=(1,), fn=jax.jit(f))
    assert ctl[1][0]["UICE"] > 0 and ctl[1][0]["VICE"] > 0, ctl[1][0]
    ctl = L.lsr_diffs(m, ds, its=(1,), sp=m.arrays.pkc["sp"].replace(SEAICEselectMetricTerms=0))
    assert ctl[1][0]["UICE"] > 0 and ctl[1][0]["VICE"] > 0, ctl[1][0]
    # blind spot: CbotC (SEAICEbasalDragK2 = 0) is +0. on every point and DWATN*COSWAT holds no -0.
    from mitjax.tests import m4off_gate as G
    for it in (1, 2, 3):
        sf, _, _, _ = G.inputs_at(m, ds, it, "Y06_lsr")
        assert not np.any(np.asarray(sf["CbotC"].data).view(np.int64)), it
        assert not np.any(np.signbit(np.asarray(ds.field(it, "Y06_lsr", "DWATN")))), it


def test_seaice_model_teacher_forced_at_i00(md):
    m, ds = md
    res = L.seaice_model_diffs(m, ds)
    for it, d in res.items():
        for st in L.DYN_STAGES + ("P13_seaice_model",):
            if st not in d:
                continue
            assert _nz(d[st]) == {}, (it, st, _nz(d[st]))
        assert {"Y02_ice_strength", "Y03_freedrift", "Y06_lsr", "I04_growth", "P13_seaice_model"} <= set(d), it
    from mitjax.tests.m4col_gate import cfg_with
    fn = L.seaice_model_fn(m, cfg=cfg_with(m, on=("SEAICE_ZETA_SMOOTHREG",)))
    ctl = L.seaice_model_diffs(m, ds, its=(1,), fn=fn)
    assert ctl[1]["Y06_lsr"]["ZETA"] > 0, ctl[1]["Y06_lsr"]


def test_free_steps_1_3(md):
    """The whole FORWARD_STEP free from the pickups (the Model's own initial carry, chained), iterations 1-3: every
    field of every dumped stage bitwise (48 stages, 998 fields per iteration, measured in dev job 27873248), incl. the
    CD-scheme IMPLDIFF of uVelD/vVelD (L21, ported in session 1: read by the momentum of iterations 2, 3) and the sea
    ice end to end. Session 4: with ALLOW_SITRACER (ported; SItracer is not dumped and no dumped field reads it: its
    gate is test_m4lab_run). Not
    compared (not probed): X00_exch_probe, C01/C02 (cg2d), the I00-I02 pickup stages (compared by test_m4lab_init)."""
    import jax.numpy as jnp
    from mitjax.tests import goadk_model_gate as M
    m, ds = md
    f = M.step_fn(m)
    tp = m.prm.time
    carry = m.initial_carry()
    for it in (1, 2, 3):
        k = it - tp.nIter0
        carry, _, _, _, probes = f(m.arrays, carry, jnp.int32(k + 1), jnp.float64(tp.startTime + tp.deltaTClock*k),
                                   jnp.int32(it))
        res = M.compare_step(m, ds, it, probes)
        assert sum(len(d) for d in res.values()) >= 998, it
        assert {"Y03_freedrift", "Y06_lsr", "I04_growth", "P13_seaice_model", "P07_kpp", "P05_gmredi_tensor",
                "D00b_mom_fluxform", "S06_dynamics", "S17_monitor"} <= set(res), it
        bad = {(s, n): v for s, d in res.items() for n, v in d.items() if v[0] == "shape" or any(v[1:])}
        assert bad == {}, (it, sorted(bad)[:10])
