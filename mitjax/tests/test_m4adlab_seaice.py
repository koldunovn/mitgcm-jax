"""lab_sea/input_ad (M4 step 7 part 2, lane M4ADLAB session 2): the sea-ice arms of the code_ad build, gated bitwise
(every point incl. halos) against lane A's dumps-on run job27855987-jdon (code_ad forward), iterations 0-2,
teacher-forced from the oracle's fields before each stage; each gate with a measured negative control
(lane M4ADLAB session 2). The Model is m4adlab_gate.stub_model() (since session 4 the real Model, no stub left);
the kernels run with the unplanted code_ad build and input_ad's own SEAICE_PARM01 values (m4adlab_gate.ad_sp).

1. SEAICE_LSR with ALLOW_AUTODIFF (G6: the resets seaice_lsr.F:201-220, no residual lines or WARNING :1003-1058,
   the deltaC guard seaice_calc_viscosities.F:350-353), Y06_lsr from Y04_solver_inputs, and the sweeps per pass
   ICOUNT1 / ICOUNT2 equal to the Fortran probe's (job 27882556). Control: ALLOW_AUTODIFF planted off (ETA, ZETA,
   etaZ, zetaZ differ at iteration 0: the resets matter where SEAICE_CALC_VISCOSITIES does not write).
2. SEAICE_MODEL's start through SEAICE_DYNSOLVER with ALLOW_AUTODIFF (seaice_model.F:137-153, seaice_dynsolver.F:96-112
   re-initialisations of PRESS0, SEAICE_zMax/zMin, uice_fd/vice_fd): Y01 .. Y09 and I01. Control: ALLOW_AUTODIFF
   planted off (uice_fd / vice_fd at Y02 and uIceNm1 / vIceNm1 differ at iterations 1-2).
3. SEAICE_ADVDIFF, the single-dimension arm (ADVECT) with SEAICE_DIFFUSION (G2, SEAICEdiffKhArea 200 and the
   derived SEAICEdiffKhHeff / Snow), I02_advdiff from I01_dynsolver. Control: SEAICEdiffKhArea 0 (AREA differs).
4. The reverse pass of one teacher-forced SEAICE_LSR call through the executed sweeps (plan decision 14, A1: since
   session 4 the kernel's own DO m loop for SEAICE_LSR_ADJOINT_ITER, mitjax/ad/lsr_sweeps.py; its forward bitwise the
   while_loop's of the same kernel with the option planted off; control: a 50-step tape, shorter than the 58 sweeps
   of iteration 0, is not bitwise): the gradient of a random projection of
   (UICE, VICE) with respect to every float64 SEAICE.h input is finite on every lane, the dot test <vjp, dx> = jvp(dx)
   holds to 1e-12 and central differences of the literal forward agree to 1e-8 (measured: dot 4e-16..2e-14, FD
   1e-10..1.3e-9; dev job 27893444). Controls: the former `where(cond, sqrt(tempVar), c)` of
   SEAICE_OCEANDRAG_COEFFS (the cause of session 1's NaN: 0*inf at tempVar = 0) and the unguarded SQRT of the
   forward build's SEAICE_CALC_VISCOSITIES each make the gradient non-finite (dev job 27893445).
   Also: the residual lines and the WARNING (seaice_lsr.F:1003-1058, #ifndef ALLOW_AUTODIFF_TAMC) are absent from the
   oracle's output.txt although debugLevel = 1 = debLevA and SEAICE_monFreq = 1 (printResidual .TRUE.); the port
   marks them unprinted. Control: ALLOW_AUTODIFF_TAMC planted off marks them printed.
5. SEAICE_GROWTH (I04_growth from I03_reg_ridge) with useMaykutSatVapPoly and postSolvTempIter 0 (G4,
   seaice_solve4temp.F:385-388, :447-458, :563-564), the saltPlumeFlux of SEAICE_VARIABLE_SALINITY (G5,
   seaice_growth.F:2070-2104, :2133-2138) and areaGainFormula 2 / areaLossFormula 3 (G3, session 3: :1825,
   :1840-1848 with plan decision 15's build-dependent winners at :1847-1848): every dumped field bitwise incl. AREA.
   Controls: each formula set back to 1 (AREA differs); MINMAX_BUILD planted with the wrong winner, or another build
   (lab_sea/code), is refused; the table equals the oracle's site table. Blind spot (asserted, dev job 27894227): the
   wrong winner planted in the call itself changes no dumped field (+0/-0 ties absorbed by AREA's sum); the audit
   catches it. Controls: useMaykutSatVapPoly .FALSE. (TICES differs), postSolvTempIter 2 (HEFF, HSNOW, saltFlux
   differ).
6. G7 EXF_SEAICE_FRACTION (session 3): the EXF front (areamask, exf_iceFraction at X01-X06) and the whole SEAICE_MODEL
   with d_AREAbyRLX / d_HEFFbyRLX (test_exf_seaice_fraction_front, test_seaice_model_whole_and_rlx; their docstrings
   give the controls and blind spots).
Blind spot (asserted): iteration 0 starts from zero ice velocities, so the diffusion control changes nothing there.
"""

import dataclasses
import re

import numpy as np
import pytest

from mitjax.tests import m4adlab_gate as A

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def md():
    return A.stub_model()


def _nz(d):
    return {k: v for k, v in d.items() if v}


def _planted(off):
    from mitjax.tests.m4col_gate import CppPlant
    return dataclasses.replace(A.ad_cfg(), cpp=CppPlant(A.ad_cfg().cpp, off=off))


def test_lsr_autodiff_arms_and_sweep_counts(md):
    m, ds = md
    res = A.lsr_diffs(m, ds)
    for it, (d, out) in res.items():
        assert {"UICE", "VICE", "ETA", "ZETA", "deltaC", "PRESS", "DWATN", "e11", "uIceNm1"} <= set(d), it
        assert _nz(d) == {}, it
        got = tuple(zip(np.asarray(out["ICOUNT1"]).tolist(), np.asarray(out["ICOUNT2"]).tolist()))
        assert got == A.ICOUNT[it], (it, got)
        assert not np.any(out["printResid"]) and not np.any(out["printWarn"]), it
        assert not np.any(out["notConverged"]), it
    ctl = A.lsr_diffs(m, ds, its=(0,), fn=A.lsr_fn(m, _planted(("ALLOW_AUTODIFF",))))
    assert set(_nz(ctl[0][0])) == {"ETA", "ZETA", "etaZ", "zetaZ"}, ctl[0][0]
    from mitjax import paths
    text = (paths.REFERENCE_RUNS / A.EXP[0] / A.EXP[1] / A.JDON / "rundir" / "output.txt").read_text()
    assert "SEAICE_LSR (ipass=" not in text and "SEAICE_LSR: Residual" not in text
    assert re.search(r"debugLevel =\s+/\* select debug printing level \*/\n\(PID\.TID 0000\.0001\)\s+1\n", text)
    ctl = A.lsr_diffs(m, ds, its=(1,), fn=A.lsr_fn(m, _planted(("ALLOW_AUTODIFF_TAMC",))))
    assert _nz(ctl[1][0]) == {} and np.all(ctl[1][1]["printResid"]) and np.all(ctl[1][1]["printWarn"]), ctl[1][1]


def test_dynsolver_autodiff_arms(md):
    m, ds = md
    res = A.dynsolver_diffs(m, ds)
    for it, (d, lo) in res.items():
        assert {"Y01_get_dynforcing", "Y02_ice_strength", "Y03_freedrift", "Y04_solver_inputs", "Y06_lsr",
                "Y09_ocean_stress", "I01_dynsolver"} <= set(d), it
        assert {s: _nz(v) for s, v in d.items() if _nz(v)} == {}, it
        got = tuple(zip(np.asarray(lo["ICOUNT1"]).tolist(), np.asarray(lo["ICOUNT2"]).tolist()))
        assert got == A.ICOUNT[it], (it, got)
    ctl = A.dynsolver_diffs(m, ds, its=(1, 2), fn=A.dynsolver_fn(m, _planted(("ALLOW_AUTODIFF",))))
    for it in (1, 2):
        assert {"uice_fd", "vice_fd"} <= set(_nz(ctl[it][0]["Y02_ice_strength"])), (it, ctl[it][0])


def test_advdiff_diffusion(md):
    m, ds = md
    res = A.advdiff_diffs(m, ds)
    for it, d in res.items():
        assert {"AREA", "HEFF", "HSNOW", "UICE", "VICE"} <= set(d), it
        assert _nz(d) == {}, it
    ctl = A.advdiff_diffs(m, ds, its=(0, 1), sp=A.ad_sp(m).replace(SEAICEdiffKhArea=np.float64(0.0)))
    assert _nz(ctl[0]) == {}, ctl[0]                     # blind spot: iteration 0 (zero ice velocities)
    assert ctl[1]["AREA"] > 0, ctl[1]


def test_lsr_reverse_pass_finite_dot_fd(md):
    m, ds = md
    for it in (0, 1, 2):
        r = A.lsr_reverse(m, ds, it)
        assert r["fwd_bitwise"], it
        assert _nz(r["nonfinite"]) == {}, (it, _nz(r["nonfinite"]))
        assert r["dot"][2] < 1e-12, (it, r["dot"])
        assert r["fd_rel"] < 1e-8, (it, r["fd"], r["dot"])


def test_lsr_a1_taped_sweeps_control(md):
    """Plan decision 14 (A1) in the kernel (mitjax/ad/lsr_sweeps.py): the bitwise check of the taped forward bites --
    a scan of 50 steps (fewer than the 58 v-sweeps of pass 1 at iteration 0) differs from the while_loop."""
    m, ds = md
    assert not A.lsr_reverse(m, ds, 0, n_sweeps=50)["fwd_bitwise"]
    assert A.lsr_reverse(m, ds, 0, n_sweeps=58)["fwd_bitwise"]


def test_lsr_reverse_pass_controls(md, monkeypatch):
    import jax.numpy as jnp
    import mitjax.pkg.seaice.seaice_oceandrag_coeffs as D
    m, ds = md
    monkeypatch.setattr(D, "safe_sqrt", lambda x, mask, fill=0.0: jnp.where(mask, jnp.sqrt(x), fill))
    r = A.lsr_reverse(m, ds, 1)
    assert r["fwd_bitwise"] and sum(r["nonfinite"].values()) > 0, _nz(r["nonfinite"])
    monkeypatch.undo()
    r = A.lsr_reverse(m, ds, 1, cfg=_planted(("ALLOW_AUTODIFF",)))
    assert sum(r["nonfinite"].values()) > 0, _nz(r["nonfinite"])


def test_growth_maykut_postsolv0_variable_salinity_plume_area_formulas(md):
    from mitjax.tests import m4lab_gate as L
    m, ds = md
    fn = L.growth_fn(m, cfg=A.ad_cfg())
    sp = A.ad_sp(m)
    res = L.growth_diffs(m, ds, its=(0, 1, 2), sp=sp, fn=fn)
    for it, d in res.items():
        assert {"TICES", "HEFF", "HSNOW", "Qnet", "Qsw", "saltFlux", "saltPlumeFlux", "EmPmR", "AREA"} <= set(d), it
        assert _nz(d) == {}, (it, _nz(d))
    ctl = L.growth_diffs(m, ds, its=(1,), sp=sp.replace(useMaykutSatVapPoly=False), fn=fn)
    assert ctl[1]["TICES"] > 0, ctl
    ctl = L.growth_diffs(m, ds, its=(1,), sp=sp.replace(postSolvTempIter=2), fn=fn)
    assert ctl[1]["HEFF"] > 0 and ctl[1]["saltFlux"] > 0 and ctl[1]["TICES"] == 0, ctl
    # G3 controls: each formula alone set back to 1 changes AREA
    for kw in (dict(SEAICE_areaGainFormula=1), dict(SEAICE_areaLossFormula=1)):
        ctl = L.growth_diffs(m, ds, its=(1,), sp=sp.replace(**kw), fn=fn)
        assert set(_nz(ctl[1])) == {"AREA"}, (kw, ctl)


def test_growth_area_loss_formula3_build_winners(md, monkeypatch):
    """Plan decision 15 at seaice_growth.F:1847-1848: the kernel's MINMAX_BUILD names the executing build's winners
    (lab_sea-code_ad: MAX "a", MIN "b"); a planted wrong winner in the table, or another measured build, is refused; the table
    equals the oracle's site table $MJX_REFERENCE/minmax_sites/lab_sea-code_ad-63cdc0b-704fd6b.json."""
    import json
    from mitjax import paths
    from mitjax.tests import m4lab_gate as L
    import mitjax.pkg.seaice.seaice_growth as G
    m, ds = md
    t = json.loads((paths.REFERENCE / "minmax_sites" / "lab_sea-code_ad-63cdc0b-704fd6b.json").read_text())
    got = {f"{s['F_file']}:{s['F_line']}": {"lab_sea-code_ad-63cdc0b-704fd6b": s["p"]} for s in t["sites"]
           if s["F_file"] == "pkg/seaice/seaice_growth.F" and s["F_line"] in (1847, 1848)}
    assert got == G.MINMAX_BUILD, got
    sp = A.ad_sp(m)
    cfg = A.ad_cfg()
    planted = {**G.MINMAX_BUILD, "pkg/seaice/seaice_growth.F:1847": {"lab_sea-code_ad-63cdc0b-704fd6b": "b"}}
    monkeypatch.setattr(G, "MINMAX_BUILD", planted)
    with pytest.raises(NotImplementedError, match="1847-1848 winners"):
        L.growth_diffs(m, ds, its=(1,), sp=sp, fn=L.growth_fn(m, cfg=cfg))
    monkeypatch.undo()
    # another measured build without a table entry (lab_sea/code: formula 3 never measured there) is refused; the
    # build is found by its identity (mitjax/config/build_identity.py, docs plan 20261006 Task 4), so the stand-in
    # names lab_sea/code's identity label for this configuration
    from mitjax.config import build_identity as bi
    assert bi.label(cfg) == "lab_sea-code_ad-63cdc0b"
    with monkeypatch.context() as mp:
        mp.setattr(bi, "label", lambda c: "lab_sea-code-63cdc0b")
        with pytest.raises(NotImplementedError, match="decision 15"):
            L.growth_diffs(m, ds, its=(1,), sp=sp, fn=L.growth_fn(m, cfg=cfg))
    # blind spot (asserted; dev job 27894227): the wrong winner planted in the call itself changes no dumped field at
    # iterations 0-2 although the sites see +0/-0 ties (MAX :1847 8/4/2 points, MIN :1848 178/186/189): the sign of a
    # zero tmpscal3 is absorbed by AREA's sum (:1854-1857); the audit (test_minmax_sites.py) catches that plant
    for which, line in (("MAX", ":1847;"), ("MIN", ":1848;")):
        flipped = _flip_at(getattr(G, which), G, line)
        with monkeypatch.context() as mp:
            mp.setattr(G, which, flipped)
            ctl = L.growth_diffs(m, ds, its=(0, 1, 2), sp=sp, fn=L.growth_fn(m, cfg=cfg))
        assert flipped.hits == 1, (which, flipped.hits)                 # the plant reached the site (one trace)
        assert not any(_nz(d) for d in ctl.values()), (which, ctl)


def _flip_at(f, module, marker):
    """`f` (MAX or MIN) with the other winner at the call on the line of `module` holding `marker` (a test plant)."""
    import sys
    src = open(module.__file__).read().splitlines()
    (line,) = [n + 1 for n, s in enumerate(src) if marker in s]

    def g(*a, p):
        if sys._getframe(1).f_lineno == line:
            g.hits += 1
            return f(*a, p={"a": "b", "b": "a"}[p])
        return f(*a, p=p)
    g.hits = 0
    return g


def test_exf_seaice_fraction_front(md):
    """G7 (EXF_SEAICE_FRACTION), EXF side: FORWARD_STEP's front (EXF_GETFORCING X01-X06, S02) teacher-forced at
    iterations 0-2, every point incl. halos, with areamask (EXF_INIT_VARIA :344-355, EXF_GETFFIELDS :406-419) and
    exf_iceFraction (EXF_MAPFIELDS :340-348, exchange :375-377) compared at X01 and X06. Control: areamask planted 2.
    in the carry at iteration 1 (areamask differs at X01-X06, exf_iceFraction = MIN(MAX(2,0),1) = 1 at X06).
    Blind spot (asserted): input_ad sets no areamask variable, so areamask = areamaskconst = 0 on every point of every
    dump and the clamp and the exchange act on zeros."""
    from mitjax.tests import m4off_gate as G
    m, ds = md
    res = G.run_front(m, ds, its=(0, 1, 2))
    for it, r in res.items():
        for st in ("X01_exf_getffields", "X06_exf_mapfields"):
            assert {"areamask", "exf_iceFraction"} <= set(r[st]), (it, st)
            for n in ("areamask", "exf_iceFraction"):
                assert not np.any(np.asarray(ds.field(it, st, n))), (it, st, n)
        assert G.bad(r) == {}, (it, G.bad(r))
    import jax.numpy as jnp
    from mitjax.tests import goadk_model_gate as M
    carry = G.teacher_carry(m, ds, 1)
    pk = dict(carry[4])
    pk["exf"] = {**pk["exf"], "areamask": G.like(pk["exf"]["areamask"], np.full(pk["exf"]["areamask"].data.shape,
                                                                               2.0))}
    tp = m.prm.time
    probes = G.front_fn(m)(m.arrays, carry[:4] + (pk,), jnp.int32(2), jnp.float64(tp.startTime + tp.deltaTClock),
                           jnp.int32(tp.nIter0 + 1))
    bad = G.bad(M.compare_step(m, ds, 1, probes))
    assert ("X01_exf_getffields", "areamask") in bad and ("X06_exf_mapfields", "exf_iceFraction") in bad, bad
    assert {n for _, n in bad} == {"areamask", "exf_iceFraction"}, bad


def test_seaice_model_whole_and_rlx(md):
    """SEAICE_MODEL of input_ad teacher-forced from I00_seaice_begin at iterations 0-2, every dumped sea-ice stage and
    P13 bitwise (G2-G7 together), incl. G7's d_AREAbyRLX / d_HEFFbyRLX (reset on all points, seaice_reg_ridge.F:
    101-104) at I03 and I04. Controls: d_HEFFbyRLX planted 1e-6 at the growth's input (Qnet and EmPmR differ: the sums
    seaice_growth.F:2214-2216, :2375-2377 read it); d_AREAbyRLX / d_HEFFbyRLX planted 1. at SEAICE_MODEL's input
    leave every stage bitwise (the reset overwrites them). Blind spot (asserted): the sums add +0 and change no bit at
    iterations 0-2 (EXF_SEAICE_FRACTION planted off in SEAICE_GROWTH alone)."""
    from mitjax.tests import m4lab_gate as L
    m, ds = md
    sp = A.ad_sp(m)
    fn = L.seaice_model_fn(m, cfg=A.ad_cfg())
    res = L.seaice_model_diffs(m, ds, its=(0, 1, 2), sp=sp, fn=fn)
    for it, r in res.items():
        assert {"I01_dynsolver", "I02_advdiff", "I03_reg_ridge", "I04_growth", "P13_seaice_model"} <= set(r), it
        assert {"d_AREAbyRLX", "d_HEFFbyRLX", "HSALT", "saltFluxAdjust"} <= set(r["I03_reg_ridge"]), r["I03_reg_ridge"]
        assert {s: _nz(d) for s, d in r.items() if _nz(d)} == {}, (it, {s: _nz(d) for s, d in r.items() if _nz(d)})

    from mitjax.tests import m4off_gate as G

    def plant(f, n, v):
        def g(sp_, op_, spp_, sf, *rest):
            return f(sp_, op_, spp_, {**sf, **{k: G.like(sf[k], np.full(sf[k].data.shape, v)) for k in n}}, *rest)
        return g
    ctl = L.seaice_model_diffs(m, ds, its=(1,), sp=sp, fn=plant(fn, ("d_AREAbyRLX", "d_HEFFbyRLX"), 1.0))
    assert {s: _nz(d) for s, d in ctl[1].items() if _nz(d)} == {}, ctl
    g = L.growth_fn(m, cfg=A.ad_cfg())
    ctl = L.growth_diffs(m, ds, its=(1,), sp=sp, fn=plant(g, ("d_HEFFbyRLX",), 1e-6))
    assert ctl[1]["Qnet"] > 0 and ctl[1]["EmPmR"] > 0, ctl
    ctl = L.growth_diffs(m, ds, its=(0, 1, 2), sp=sp, fn=L.growth_fn(m, cfg=_planted(("EXF_SEAICE_FRACTION",))))
    assert not any(_nz(d) for d in ctl.values()), ctl
