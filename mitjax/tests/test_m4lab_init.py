"""lab_sea/input (M4 step 5, lane M4LAB session 1): the start from the pickups, the front of FORWARD_STEP and the
first ported gaps, gated bitwise (every point incl. halos) against lane A's dumps-on run job27855987-jdon
(iterations 1-3; nIter0 = 1), each with a measured negative control.

The Model is built with the session-1 stubs of m4lab_gate (the sea-ice settings whose arms are not ported yet:
the lane's gap list L3-L11); no test here runs a kernel the stubs enter.

1. Exchange map: lab_sea-t4_10x8_ol4x4.npz (registered; scripts/make_exch_maps.py reproduced the X00 probe).
2. The front from the Model's own initial carry (READ_PICKUP of pickup.0000000001, pkg/cal, EXF with monthly records,
   climsss relaxation): G00_geometry and S00_begin .. S02_load_fields at iteration 1; teacher-forced at 2, 3.
3. SEAICE_INIT_FIXED / SEAICE_INIT_VARIA from pickup_seaice.0000000001 = I00_seaice_begin at iteration 1, incl. the
   spherical metric coefficients k2AtC/U/V/Z (seaice_init_fixed.F:279-294, gap L2).
4. EXTERNAL_FORCING_SURF -> FORCING_SURF_RELAX without restoring under sea ice (forcing_surf_relax.F:75-90, gap L13),
   teacher-forced from P13_seaice_model, iterations 1-3.
5. SEAICE_LSR refuses the build options it does not port (session 1: SEAICE_LSR_ZEBRA, then silent; session 3:
   ZEBRA ported, the check now plants SEAICE_VECTORIZE_LSR / SEAICE_ALLOW_SIDEDRAG on, SEAICE_ALLOW_FREEDRIFT off).
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from mitjax.tests import m4lab_gate as L

# PTRACERS compiled, usePTRACERS .FALSE. (data.pkg): the jaxdump writes the unused pTracer arrays (zeros of the
# never-written common); session 2 sets them to that zero (drivers/model.py; they held NaN before), so they are compared
DEAD = set()


@pytest.fixture(scope="module")
def md():
    return L.stub_model()


def _bad(res):
    return {(s, n): v for s, d in res.items() for n, v in d.items()
            if (s, n) not in DEAD and (v[0] == "shape" or any(v[1:]))}


def test_front_iteration1_free_and_2_3_teacher_forced(md):
    """S00_begin (after the pickups), G00, X01-X06, S02 at iteration 1 from the Model's own initial carry; X01..S02
    at iterations 2, 3 from the oracle's carry (EXF_GETFFIELDREC / the monthly record interpolation of pkg/cal)."""
    from mitjax.tests import goadk_model_gate as M
    from mitjax.tests import m4off_gate as G
    m, ds = md
    tp = m.prm.time
    f = G.front_fn(m)
    probes = f(m.arrays, m.initial_carry(), jnp.int32(1), jnp.float64(tp.startTime), jnp.int32(tp.nIter0))
    res = M.compare_step(m, ds, 1, probes)
    for st in ("S00_begin", "G00_geometry", "X01_exf_getffields", "X06_exf_mapfields", "S02_load_fields"):
        assert len(res.get(st, {})) > 10 or st == "S02_load_fields", (st, len(res.get(st, {})))
    assert _bad(res) == {}
    for it in (2, 3):                    # m4off_gate.run_front assumes nIter0 = 0: here k = it - nIter0
        k = it - tp.nIter0
        probes = f(m.arrays, G.teacher_carry(m, ds, it), jnp.int32(k + 1),
                   jnp.float64(tp.startTime + tp.deltaTClock*k), jnp.int32(it))
        r = M.compare_step(m, ds, it, probes)
        assert len(r.get("X06_exf_mapfields", {})) > 30
        assert _bad(r) == {}, it


def test_seaice_init_from_pickup_with_metric_terms(md):
    """SEAICE.h after SEAICE_INIT_FIXED / INIT_VARIA (pickup_seaice.0000000001) == I00_seaice_begin of iteration 1 on
    every point, incl. k2AtC/U/V/Z = -tanPhiAtU/V*recip_rSphere (SEAICEselectMetricTerms 2). Control: without the
    metric terms (SEAICEuseMetricTerms .FALSE. -> SEAICEselectMetricTerms 0) the four differ on every point (1152)."""
    m, ds = md
    d = L.seaice_init_diffs(m, ds)
    assert len(d) >= 19 and {"k2AtC", "k2AtU", "k2AtV", "k2AtZ", "AREA", "HEFF", "UICE", "TICES"} <= set(d)
    assert {n: v for n, v in d.items() if v} == {}
    from mitjax import paths
    from mitjax.drivers.model import Model, with_namelist
    e = with_namelist(L.stub_experiment(), {(L.S, "SEAICE_PARM01", "SEAICEuseMetricTerms"): (False, "bool")})
    m2 = Model(e, paths.REFERENCE_RUNS / L.EXP[0] / L.EXP[1] / L.JDON / "rundir")
    d2 = L.seaice_init_diffs(m2, ds)
    assert {n: v for n, v in d2.items() if v} == {n: 1152 for n in ("k2AtC", "k2AtU", "k2AtV", "k2AtZ")}


def test_forcing_surf_relax_no_restore_under_ice(md):
    """P01_external_forcing_surf teacher-forced from P13_seaice_model (FFIELDS.h, AREA), iterations 1-3: every field
    bitwise (surfaceForcingS = -lambdaSaltClimRelax*(1.-AREA)*(salt(1)-SSS)*drF(1)*hFacC). Control: the restoring
    arm (SEAICErestoreUnderIce .TRUE.) changes surfaceForcingS under the ice at every iteration."""
    m, ds = md
    out = L.external_forcing_surf_diffs(m, ds)
    for it, d in out.items():
        assert {"surfaceForcingS", "surfaceForcingT", "Qnet", "EmPmR"} <= set(d), it
        assert {n: v for n, v in d.items() if v} == {}, it
    sp = m.arrays.pkc["sp"].replace(SEAICErestoreUnderIce=True)
    ctl = L.external_forcing_surf_diffs(m, ds, sp=sp)
    for it, d in ctl.items():
        assert d["surfaceForcingS"] > 50, (it, d)


def test_lsr_refuses_unported_build_options(md):
    """SEAICE_LSR raises on the build options it does not port (session 1: SEAICE_LSR_ZEBRA ran the dyn_lsr arms
    silently; session 3 ported ZEBRA and SEAICE_ALLOW_BOTTOMDRAG, gated in test_m4lab_seaice): SEAICE_VECTORIZE_LSR
    and SEAICE_ALLOW_SIDEDRAG planted on (and SEAICE_ALLOW_CHECK_LSR_CONVERGENCE, SEAICE_ALLOW_MOM_ADVECTION). The
    build without SEAICE_ALLOW_FREEDRIFT is ported since lane M4CS32ICE (global_ocean.cs32x15, seaice_lsr.F:614-645
    not compiled), so planting it off no longer raises (regression job 27878079)."""
    from mitjax.config.params import load
    from mitjax.pkg.seaice.seaice_lsr import seaice_lsr
    from mitjax.tests.m4col_gate import CppPlant
    m, _ = md
    pkc = m.arrays.pkc
    cfg = load(*L.EXP).cfg
    for plant, msg in (({"on": ("SEAICE_VECTORIZE_LSR",)}, "SEAICE_VECTORIZE_LSR"),
                       ({"on": ("SEAICE_ALLOW_SIDEDRAG",)}, "SEAICE_ALLOW_SIDEDRAG"),
                       ({"on": ("SEAICE_ALLOW_CHECK_LSR_CONVERGENCE",)}, "SEAICE_ALLOW_CHECK_LSR_CONVERGENCE"),
                       ({"on": ("SEAICE_ALLOW_MOM_ADVECTION",)}, "SEAICE_ALLOW_MOM_ADVECTION")):
        cfg2 = dataclasses.replace(cfg, cpp=CppPlant(cfg.cpp, **plant))
        with pytest.raises(NotImplementedError, match=msg):
            seaice_lsr(0.0, 1, None, cfg=cfg2, sp=pkc["sp"], op=pkc["op"], grid=None, state=None, ex=None)
