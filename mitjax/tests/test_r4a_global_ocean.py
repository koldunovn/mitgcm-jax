"""R4 global_ocean.90x40x15 (plan Task 15a), M1 lane GO: the NONLIN_FRSURF blocks of FORWARD_STEP (forward_step.py
nlfs_reset, nlfs_update_hfac, nlfs_update_cg2d, nlfs_calc_r_star), DYNAMICS with the C-D scheme, quasi-hydrostatic and
NH metric terms, 3-D Coriolis, viscA4, the r* terms of CALC_GRAD_PHI_HYD / DIAGS_PHI_* / MOM_FLUXFORM /
MOM_CALC_RTRANS / TIMESTEP, and CALC_R_STAR's zero-denominator count.

Teacher forcing from the oracle's substep dumps of steps 36000-36002 (pickup start, mitjax/tests/go_gate.py says
which stage feeds which input); bitwise on every point of every tile incl. halos (element equality, bit patterns,
finite). The stages that need GM/Redi, the periodic forcing preload (lane R5) or SALT_INTEGRATE (lane ADVECT) are
not run here: THERMODYNAMICS (S05) and DO_OCEANIC_PHYS (S04) are inputs, not gated.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax.farray import FArray
from mitjax.model.state import NLFS_GRID
from mitjax.tests import go_gate as G

R_STATE = G.rsg.R_STATE
S06_FIELDS = ("gU", "gV", "guNm1", "gvNm1", "totPhiHyd", "phiHydLow", "uVel", "vVel", "wVel", "etaN", "etaH")


def _s():
    return G.setup()


def _start_state(s, it):
    st = G.teacher_state(s, it, [("S00_begin", None), ("G00_geometry", R_STATE)])
    g = G.teacher_grid(s, it, "S00_begin")
    return st.replace(**{n: getattr(g, n) for n in NLFS_GRID}), g


def _s01(s, it):
    st0, g0 = _start_state(s, it)
    return G.run_nlfs_reset(s, st0, g0, it)


def _dyn_inputs(s, it, grid):
    st = G.teacher_state(s, it, [("S00_begin", None), ("G00_geometry", R_STATE), ("S01_update_rstar_F", None),
                                 ("S04_oceanic_phys", None), ("S05_thermodynamics_sync", None)])
    st = st.replace(**{n: getattr(grid, n) for n in NLFS_GRID})
    ff, phi0 = G.teacher_ff(s, it)
    return st, ff, phi0


def _cd_carry(s, st_prev_out, st_next, it_prev):
    """CD_CODE_VARS.h of the next step: uVelD, vVelD, uNM1, vNM1 from our DYNAMICS of the previous step, after the
    end-of-step EXCH_UV_DGRID_3D_RL (do_fields_blocking_exchanges.F:83); etaNm1 = etaN as SOLVE_FOR_PRESSURE sets it
    (solve_for_pressure.F:127) = etaN at S06 of the previous step (the value the solver starts from)."""
    ex = s.ex
    u, v = ex.EXCH_UV_DGRID_3D_RL(st_prev_out.uVelD.data, st_prev_out.vVelD.data, True)
    uD = FArray(u, "uVelD", tiled=True, _dims=st_prev_out.uVelD.dims)
    vD = FArray(v, "vVelD", tiled=True, _dims=st_prev_out.vVelD.dims)
    eta = G._farr(G.dumped(s, it_prev, "S06_dynamics")["etaN"], st_next.etaNm1)
    return st_next.replace(uVelD=uD, vVelD=vD, uNM1=st_prev_out.uNM1, vNM1=st_prev_out.vNM1, etaNm1=eta)


def _dynamics_chain(s, params=None, steps=None):
    """[(it, State after DYNAMICS)] for the dumped steps, CD state carried from our own previous step."""
    p = s.params if params is None else params
    out, prev = [], None
    for it in (s.its if steps is None else steps):
        _, g1 = _s01(s, it)
        st, ff, phi0 = _dyn_inputs(s, it, g1)
        st = G.cd_initial(s, st) if prev is None else _cd_carry(s, prev[1], st, prev[0])
        sp = s.__class__(**{**vars(s), "params": p})
        res = G.run_dynamics(sp, st, g1, ff, phi0, it)
        out.append((it, res))
        prev = (it, res)
    return out


def _vals(st):
    return {n: getattr(st, n) for n in st.names()}


# ---------------------------------------------------------------------------------------------------------------

def test_s01_nlfs_reset_bitwise():
    """forward_step.F:461-494 (RESET_NLFS_VARS, UPDATE_R_STAR(.FALSE.)) from S00/G00 == S01 at every dumped step."""
    s = _s()
    for it in s.its:
        st1, _ = _s01(s, it)
        r = G.compare_stage(s, it, "S01_update_rstar_F", _vals(st1))
        assert len(r) >= 10 and not G.bad(r), (it, G.bad(r))


def test_s06_dynamics_bitwise_chain():
    """DYNAMICS (C-D scheme, quasi-hydrostatic, NH metric, 3-D Coriolis, viscA4, r*) teacher-forced at S05 == S06
    at steps 36000-36002 on every point; the CD state from pickup_cd (36000) and from our own previous step."""
    s = _s()
    for it, res in _dynamics_chain(s):
        r = G.compare_stage(s, it, "S06_dynamics", _vals(res), names=S06_FIELDS)
        assert set(S06_FIELDS) <= set(r), sorted(r)
        assert not G.bad(r), (it, G.bad(r))


def test_s06_negative_controls_bite():
    """rCD + 1 ulp (CD relaxation) and the r* divergence term dropped (rStarExpW/S := 1) each change S06 gU at 36000."""
    s = _s()
    it = s.its[0]
    p = s.params.replace(traced=dict(rCD=np.nextafter(np.float64(s.params.rCD), np.inf)))
    res = _dynamics_chain(s, params=p, steps=(it,))[0][1]
    r = G.compare_stage(s, it, "S06_dynamics", _vals(res), names=("gU", "gV"))
    assert r["gU"][3] > 0 and r["gV"][3] > 0, r
    _, g1 = _s01(s, it)
    st, ff, phi0 = _dyn_inputs(s, it, g1)
    st = G.cd_initial(s, st)
    one = {n: FArray(jnp.ones_like(getattr(st, n).data), n, tiled=True, _dims=getattr(st, n).dims)
           for n in ("rStarExpW", "rStarExpS")}
    res2 = G.run_dynamics(s, st.replace(**one), g1, ff, phi0, it)
    r2 = G.compare_stage(s, it, "S06_dynamics", _vals(res2), names=("gU",))
    assert r2["gU"][3] > 1000, r2


def test_s07_s08_update_hfac_cg2d_bitwise():
    """forward_step.F:832-875: UPDATE_R_STAR(.TRUE.) == S07 (hFacC/W/S, recip_hFacC), UPDATE_CG2D == S08 (aW2d, aS2d,
    aC2d) at every dumped step; inputs: our S01 grid, rStarFacC/W/S of S00 (CALC_R_STAR runs only at :949)."""
    from mitjax.model.src.forward_step import nlfs_update_cg2d, nlfs_update_hfac
    s = _s()
    cfg = s.cfg
    for it in s.its:
        st1, g1 = _s01(s, it)
        t, i = G.counters(s, it + 1)                   # forward_step.F:807-808: the counters are updated before
        st1 = st1.replace(**{n: getattr(g1, n) for n in NLFS_GRID})
        from mitjax.model.state import NLFS_CG2D
        st1 = st1.replace(**{n: getattr(s.cg2dh, n) for n in NLFS_CG2D})
        f = jax.jit(lambda st, g, p, cp, c, t, i: nlfs_update_cg2d(
            t, i, cfg=cfg, grid=nlfs_update_hfac(t, i, cfg=cfg, grid=g, params=p, state=st)[1], params=p,
            cg2d_params=cp, cg2dh=c, state=nlfs_update_hfac(t, i, cfg=cfg, grid=g, params=p, state=st)[0], ex=s.ex))
        st8, _ = f(st1, g1, s.params, s.cg2d_params, s.cg2dh, t, i)
        r7 = G.compare_stage(s, it, "S07_update_rstar_T", _vals(st8))
        assert len(r7) >= 4 and not G.bad(r7), (it, G.bad(r7))
        r8 = G.compare_stage(s, it, "S08_update_cg2d", _vals(st8))
        assert set(r8) >= {"aW2d", "aS2d", "aC2d"} and not G.bad(r8), (it, G.bad(r8))


def _s12(s, it, plant=None):
    from mitjax.model.src.forward_step import nlfs_calc_r_star
    cfg = s.cfg
    st = G.teacher_state(s, it, [("S00_begin", None), ("G00_geometry", R_STATE), ("S11_integr_continuity", None)])
    g = G.teacher_grid(s, it, "S00_begin")
    if plant is not None:
        st = st.replace(rStarFacC=plant(st.rStarFacC))
    t, i = G.counters(s, it + 1)
    f = jax.jit(lambda st, g, p, t, i: nlfs_calc_r_star(t, i, cfg=cfg, grid=g, params=p, state=st, ex=s.ex))
    return f(st, g, s.params, t, i)


def test_s12_calc_r_star_bitwise_and_zero_count():
    """forward_step.F:939-961 CALC_R_STAR(etaH of S11) == S12 (r) and G00 R of the next step; the accepted guard
    at calc_r_star.F:306-311 counts 0 zero denominators at every dumped step (host report empty)."""
    from mitjax.model.src.calc_r_star import calc_r_star_host, calc_r_star_zero_report
    s = _s()
    for n, it in enumerate(s.its):
        st, cnt = _s12(s, it)
        r = G.compare_stage(s, it, "S12_calc_rstar", _vals(st), names=("rStarFacC", "rStarFacW", "rStarFacS",
                                                                            "rStarDhCDt"))
        assert len(r) == 4 and not G.bad(r), (it, G.bad(r))
        if n + 1 < len(s.its):
            r2 = G.compare_stage(s, s.its[n + 1], "G00_geometry", _vals(st), names=G.rsg.CALC_OUT_G00)
            assert len(r2) == len(G.rsg.CALC_OUT_G00) and not G.bad(r2), (it, G.bad(r2))
        nz, lines = calc_r_star_zero_report(cnt, it + 1)
        assert nz == (0, 0, 0) and lines == []
        calc_r_star_host(cnt, it + 1, 0, cfg=s.cfg)        # no STOP


def test_zero_denominator_planted_is_counted():
    """Negative control of the count: rStarFacC := 0 at one interior wet point of tile 5 before CALC_R_STAR (it
    becomes rStarExpC, the denominator of :306-307) -> the host report counts exactly (1, 0, 0) and prints one line;
    the forward value there is rStarFacC/1 (the guard), finite."""
    from mitjax.model.src.calc_r_star import calc_r_star_zero_report
    s = _s()
    it = s.its[0]
    sz = s.cfg.size

    def plant(a):
        jj, ii = 5 + sz.OLy - 1, 5 + sz.OLx - 1             # Fortran (5, 5) of tile 5 (storage index 4)
        return FArray(a.data.at[4, jj, ii].set(0.), a.name, tiled=True, _dims=a.dims)
    st, cnt = _s12(s, it, plant)
    nz, lines = calc_r_star_zero_report(cnt, it + 1)
    assert nz == (1, 0, 0), nz
    assert len(lines) == 1 and "calc_r_star.F:306-311" in lines[0]
    assert np.all(np.isfinite(np.asarray(st.rStarExpC.data)))


def test_dynamics_gradient_finite_and_fd():
    """d J / d uVel with J = sum(gU*w) (w a fixed random field) through DYNAMICS at 36000: finite on every lane
    (halos, land); FD at a smooth interior wet point, central differences, h-sweep 1e-2..1e-5 of |u|; threshold
    stated beforehand: best relative error < 1e-6."""
    s = _s()
    it = s.its[0]
    _, g1 = _s01(s, it)
    st, ff, phi0 = _dyn_inputs(s, it, g1)
    st = G.cd_initial(s, st)
    w = jnp.asarray(np.random.default_rng(0).standard_normal(st.gU.data.shape))
    mask = jnp.asarray(np.asarray(g1.maskW.data))
    from mitjax.model.src.dynamics import dynamics
    t, i = G.counters(s, it)

    def J(u):
        st2 = st.replace(uVel=FArray(u, "uVel", tiled=True, _dims=st.uVel.dims))
        out = dynamics(t, i, cfg=s.cfg, grid=g1, params=s.params, state=st2, ff=ff, fp=s.fp, phi0surf=phi0)
        return jnp.sum(out.gU.data*w*mask)
    u0 = st.uVel.data
    Jj = jax.jit(J)
    gJ = jax.jit(jax.grad(J))(u0)
    assert np.all(np.isfinite(np.asarray(gJ)))
    idx = (14, 0, 3 + 5, 3 + 5)
    assert float(mask[idx]) == 1.
    x = float(u0[idx])
    best = np.inf
    for h in (1e-2, 1e-3, 1e-4, 1e-5):
        dh = h*max(abs(x), 1e-3)
        fd = (float(Jj(u0.at[idx].add(dh))) - float(Jj(u0.at[idx].add(-dh))))/(2*dh)
        best = min(best, abs(fd - float(gJ[idx]))/max(abs(float(gJ[idx])), 1e-30))
    assert best < 1e-6, best


def test_package_switches_live_and_refused_at_setup():
    """The `IF (useX)` guards of DO_OCEANIC_PHYS / TEMP_INTEGRATE read live switches (ini_parms_dyn from cfg.use; they
    were dead `static_items().get` reads before), and the integrated Model refuses a switched-on package it does not
    run at set-up. Negative controls: useKL10 = .TRUE. overlaid on global_ocean's cfg -> params.useKL10 is True and
    check_packages raises (pkg/kl10 is not ported; the planted switch was useGGL90 until the vermix lane integrated
    GGL90); global_ocean itself passes (gmredi and sbo are integrated)."""
    import dataclasses
    from mitjax.config.params import UnportedPackage
    from mitjax.drivers.model import INTEGRATED_PACKAGES, check_packages
    from mitjax.model.src.ini_parms import ini_parms_dyn
    s = _s()
    assert "kl10" not in INTEGRATED_PACKAGES
    assert s.params.useKL10 is False and s.params.usePP81 is False
    cfg2 = dataclasses.replace(s.cfg, use=tuple((k, True if k == "useKL10" else v) for k, v in s.cfg.use))
    e2 = dataclasses.replace(s.e, cfg=cfg2)
    p2 = ini_parms_dyn(e2, s.prm.grid, s.prm.time, s.prm.init)
    assert p2.useKL10 is True
    with pytest.raises(UnportedPackage, match="useKL10"):
        check_packages(cfg2)
    check_packages(s.cfg)            # global_ocean's useGMRedi (R5 merge) and useSBO (GO lane) are integrated
    check_packages(G.gg.experiment("tutorial_barotropic_gyre", "input").cfg)


def test_s09_s11_solve_correct_continuity_bitwise():
    """forward_step.F:897-930: SOLVE_FOR_PRESSURE (CD etaNm1, real fresh-water source, exactConserv) ->
    MOMENTUM_CORRECTION_STEP -> INTEGR_CONTINUITY (r* rStarDhDt, INTEGRATE_FOR_W r* arm, exactConserv with real fresh
    water, UPDATE_ETAH) from the S08 inputs == S09, S10, S11 (group d) at every dumped step, solver iterations ==
    C02. Negative controls: useRealFreshWaterFlux := .FALSE. and select_rStar := 0 (INTEGR_CONTINUITY / INTEGRATE_FOR_W
    only) each change the outputs at 36000."""
    s = _s()
    for it in s.its:
        st, g, c, ff = G.after_s08(s, it)
        out, diag = G.run_s09_s11(s, it, st, g, c, ff)
        for stage, v in out.items():
            r = G.compare_stage(s, it, stage, _vals(v))
            assert set(r) >= {"uVel", "vVel", "wVel", "etaN", "etaH", "dEtaHdt"} and not G.bad(r), (it, stage,
                                                                                                    G.bad(r))
        assert int(diag["numIters"]) == int(s.ds.scalar(it, "C02_cg2d_solution", "numIters"))
    it = s.its[0]
    st, g, c, ff = G.after_s08(s, it)
    out, _ = G.run_s09_s11(s, it, st, g, c, ff, params=s.params.replace(static=dict(useRealFreshWaterFlux=False)))
    r = G.compare_stage(s, it, "S09_solve_for_pressure", _vals(out["S09_solve_for_pressure"]), names=("etaN",))
    assert r["etaN"][3] > 100, r
    out, _ = G.run_s09_s11(s, it, st, g, c, ff, params=s.params.replace(static=dict(select_rStar=0)))
    r = G.compare_stage(s, it, "S11_integr_continuity", _vals(out["S11_integr_continuity"]), names=("wVel",))
    assert r["wVel"][3] > 100, r


def test_t13_t23_recip_hFacNew_bitwise():
    """THERMODYNAMICS' r* recip_hFacNew = recip_hFacC/rStarExpC (thermodynamics.F:207-218) through the implicit
    vertical step (GAD_IMPLICIT_R, temp_integrate.F:480-504 / salt_integrate.F:478-502): the dumped T12/T22 gT_loc,
    gS_loc and T13/T23 kappaRk, wVel of S01 (no GM bolus velocity enters GAD_IMPLICIT_R without implicit vertical
    advection) -> T13/T23 bitwise at every dumped step. Negative control: recip_hFacC without the r* factor bites."""
    from mitjax.pkg.generic_advdiff.gad_h import GAD_SALINITY, GAD_TEMPERATURE
    from mitjax.pkg.generic_advdiff.gad_implicit_r import gad_implicit_r
    s = _s()
    cfg = s.cfg
    sz = cfg.size
    from mitjax.farray import loops_kji

    def run(it, rstar=True):
        _, g1 = _s01(s, it)
        # the State THERMODYNAMICS reads: S00 + G00 R + S01 + S04 (theta, salt before the tracer step)
        st = G.teacher_state(s, it, [("S00_begin", None), ("G00_geometry", R_STATE), ("S01_update_rstar_F", None),
                                     ("S04_oceanic_phys", None)])
        k, j, i = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
        rh = st.theta.local("recip_hFacNew").at[i, j, k].set(0.)
        if rstar:
            rh = rh.at[i, j, k].set(g1.recip_hFacC[i, j, k]/st.rStarExpC[i, j])      # thermodynamics.F:212-213
        else:
            rh = rh.at[i, j, k].set(g1.recip_hFacC[i, j, k])
        out = {}
        t, i_ = G.counters(s, it)
        for tid, fld, pre, post in ((GAD_TEMPERATURE, "theta", "T12_temp_step", "T13_temp_impl"),
                                    (GAD_SALINITY, "salt", "T22_salt_step", "T23_salt_impl")):
            d12, d13 = G.dumped(s, it, pre), G.dumped(s, it, post)
            name = "gT_loc" if tid == GAD_TEMPERATURE else "gS_loc"
            scheme = s.params.tempVertAdvScheme if tid == GAD_TEMPERATURE else s.params.saltVertAdvScheme
            gloc = G._farr(d12[name], st.theta.local("g_loc"))
            kap = G._farr(d13["kappaRk"], st.theta.local("kappaRk"))
            impl = s.params.tempImplVertAdv if tid == GAD_TEMPERATURE else s.params.saltImplVertAdv
            f = jax.jit(lambda kap, rh, w, tr, gl, p, t, i_: gad_implicit_r(
                impl, scheme, tid, p.dTtracerLev, kap, rh, w, tr, gl, t, i_, cfg=cfg, grid=g1, params=p))
            res = f(kap, rh, st.wVel, getattr(st, fld), gloc, s.params, t, i_)
            out[post] = G.compare(res.data, d13[name])
        return out
    for it in s.its:
        r = run(it)
        assert all(not any(v[1:]) for v in r.values()), (it, r)
    r = run(s.its[0], rstar=False)
    assert all(v[3] > 100 for v in r.values()), r


def test_initialise_varia_rstar_sequence_bitwise():
    """INITIALISE_VARIA's NONLIN_FRSURF sequence (initialise_varia.F:297-347: CALC_R_STAR(etaH, startTime, -1),
    UPDATE_R_STAR(.TRUE.), UPDATE_CG2D, INTEGR_CONTINUITY with exactConserv and real fresh water, CALC_R_STAR(etaH,
    startTime, nIter0)) and CD_CODE_INI_VARS from the pickups == S00_begin (+ G00 R) of 36000 on every State field
    the oracle dumps there, except the fields of the routines still pending (PACKAGES_INIT_VARIABLES' other
    packages, INI_FFIELDS / INI_FORCING: FFIELDS.h). Host: both CALC_R_STAR calls report nothing (no warning, no
    zero denominator)."""
    from mitjax.drivers.run import _rstar_host
    from mitjax.model.src.initialise_varia import executed_pending, initialise_varia
    from mitjax.pkg.rw.read_rec import RW
    from mitjax.tests import init_gate as ig
    s = _s()
    cfg = s.cfg
    pend = tuple(r for r in executed_pending(cfg, s.prm)
                 if r not in ("INTEGR_CONTINUITY", "CALC_R_STAR", "UPDATE_R_STAR", "UPDATE_CG2D"))
    rw = RW(s.rundir, s.prm.init.readBinaryPrec, cfg.size)
    host = []
    st = initialise_varia(s.grid, cfg=cfg, params=s.prm, ex=s.ex, rw=rw, pending=pend, dyn=s.params, cg2dh=s.cg2dh,
                          cg2d_params=s.cg2d_params, host=host)
    res = ig.compare(st, *G.EXP)
    fails = ig.failures(res, exempt=ig.exempt_fields(pend))
    assert not fails, fails
    for n in ("hFacC", "hFacW", "hFacS", "recip_hFacC", "aW2d", "aS2d", "aC2d", "rStarFacC", "rStarExpC",
              "rStarDhCDt", "wVel", "etaH", "dEtaHdt"):
        assert n in res, n                                                     # compared, not skipped
    stderr = []
    nw = 0
    # READ_PICKUP's CHECK_PICKUP start levels (GO lane, cs32x15 input_ad) share the host list: keep CALC_R_STAR's
    assert [h[0] for h in host] == ["CHECK_PICKUP", "CALC_R_STAR", "CALC_R_STAR"], [h[:2] for h in host]
    rstar = [h for h in host if h[0] == "CALC_R_STAR"]
    assert [h[1] for h in rstar] == [-1, s.prm.time.nIter0]
    for _, it_, cnt in rstar:
        nw = _rstar_host(cnt, it_, nw, cfg, stderr)
    assert stderr == []


def test_chain_p4_equals_p1():
    """The GO stages of step 36000 in one program (nlfs_reset, DYNAMICS, nlfs_update_hfac, nlfs_update_cg2d,
    SOLVE_FOR_PRESSURE, MOMENTUM_CORRECTION_STEP, INTEGR_CONTINUITY, nlfs_calc_r_star; go_gate.chain_body) at P = 1
    == S11, S12 dumps, and inside jit(shard_map(check_vma=True)) on 4 fake CPU devices (36 tiles, 9 per device) == P = 1
    on every leaf (both States, the CALC_R_STAR counters, the solver scalars), every point finite. Padding lanes:
    36 tiles divide evenly for every P <= 4 (the gate flags' device count), so no padding tile exists here; padding
    tiles are replicas of tile 1 (eesupp/shard.py), whose r* denominators are the real tile's."""
    import numpy as np
    from mitjax.tests.r2_gate import tree_bits_differ
    s = _s()
    it = s.its[0]
    one = G.run_chain(s, it)
    for stage in ("S11_integr_continuity", "S12_calc_rstar"):
        r = G.compare_stage(s, it, stage, _vals(one[1]))
        assert len(r) >= 6 and not G.bad(r), (stage, G.bad(r))
    four = G.run_chain(s, it, 4)
    la, lb = jax.tree.leaves(one), jax.tree.leaves(four)
    assert len(la) == len(lb) > 50
    nd = [int(np.count_nonzero(np.ascontiguousarray(a, np.float64).view(np.int64)
                                != np.ascontiguousarray(b, np.float64).view(np.int64))) for a, b in zip(la, lb)]
    assert not any(nd), nd
    assert not tree_bits_differ(one, four)
    written = ("uVel", "vVel", "wVel", "etaN", "etaH", "dEtaHdt", "PmEpR", "gU", "gV", "guNm1", "gvNm1",
               "totPhiHyd", "phiHydLow", "uVelD", "vVelD", "uNM1", "vNM1", "etaNm1", "hFacC", "hFacW", "hFacS",
               "recip_hFacC", "recip_hFacW", "recip_hFacS", "aW2d", "aS2d", "aC2d", "rStarFacC", "rStarFacW",
               "rStarFacS", "rStarExpC", "rStarExpW", "rStarExpS", "rStarDhCDt", "rStarDhWDt", "rStarDhSDt")
    for n in written:                         # every field the chain writes (teacher-forced inputs it never
        a = np.asarray(getattr(four[1], n).data)    # writes, e.g. hFac_surf*, stay NaN by construction)
        assert np.all(np.isfinite(a)), n


def test_cd_code_pickup_round_trip(tmp_path):
    """CD_CODE_READ_PICKUP of pickup_cd.0000036000 then PACKAGES_WRITE_PICKUP -> CD_CODE_WRITE_PICKUP (as a global
    file) writes the input file's bytes back (records uVelD, vVelD, uNM1, vNM1 of Nr levels, etaNm1 at 4*Nr+1);
    control: uVelD + 1 ulp at one point changes the bytes. MDS_WRITE_FIELD's exch2 I/O layout (W2_useE2ioLayOut) is
    not ported (pkg/mdsio); on global_ocean's single 90x40 facet it coincides with the plain global layout, which
    this test uses (MdsContext exch2=False)."""
    import dataclasses
    import numpy as np
    from mitjax.model.src.ini_parms import ini_parms_io
    from mitjax.pkg.cd_code.cd_code_write_pickup import cd_code_write_pickup
    from mitjax.pkg.mdsio.mdsio_write_field import MdsContext
    s = _s()
    st = G.cd_initial(s, G.teacher_state(s, s.its[0], [("S00_begin", None)]))
    io = dataclasses.replace(ini_parms_io(s.e), globalFiles=True, pickup_write_mdsio=True)
    want = (s.rundir / "pickup_cd.0000036000").read_bytes()
    for tag, state, same in (("same", st, True),
                             ("planted", st.replace(uVelD=st.uVelD.__class__(
                                 st.uVelD.data.at[4, 0, 5, 5].set(np.nextafter(float(st.uVelD.data[4, 0, 5, 5]),
                                                                              np.inf)),
                                 "uVelD", tiled=True, _dims=st.uVelD.dims)), False)):
        d = tmp_path / tag
        d.mkdir()
        mds = MdsContext(d, s.cfg.size, exch2=False, useSingleCpuIO=io.useSingleCpuIO, mdsioLocalDir=io.mdsioLocalDir,
                         the_run_name=io.the_run_name)
        fn = cd_code_write_pickup(True, "0000036000", 0., 36000, cfg=s.cfg, io=io, state=state, mds=mds)
        got = (d / (fn + ".data")).read_bytes()
        assert (got == want) == same, (tag, len(got), len(want))


def _read_tiled(rundir, fn, size, nrec):
    """The oracle's tiled MDS file `fn`.<iG>.<jG>.data of one process (iG = bi, jG = bj): [tile, rec, j, i] interior
    values, tile = (bj-1)*nSx + (bi-1)."""
    import numpy as np
    out = np.empty((size.nSx*size.nSy, nrec, size.sNy, size.sNx))
    for bj in range(1, size.nSy+1):
        for bi in range(1, size.nSx+1):
            a = np.fromfile(rundir / f"{fn}.{bi:03d}.{bj:03d}.data", ">f8")
            out[(bj-1)*size.nSx + bi-1] = a.reshape(nrec, size.sNy, size.sNx)
    return out


def _into(st, name, vals, OLx, OLy):
    """State field `name` with the interior replaced by vals [tile, (k,) j, i]."""
    f = getattr(st, name)
    d = np.asarray(f.data).copy()
    if d.ndim == 4:
        d[:, :, OLy:OLy+vals.shape[-2], OLx:OLx+vals.shape[-1]] = vals
    else:
        d[:, OLy:OLy+vals.shape[-2], OLx:OLx+vals.shape[-1]] = vals
    return st.replace(**{name: FArray(jnp.asarray(d), f.name, tiled=True, _dims=f.dims)})


def test_exch2_io_layout_pickups_byte_identical(tmp_path):
    """MDS_WRITE_FIELD / MDS_WR_METAFILES with the exch2 I/O layout (W2_useE2ioLayOut, GO lane): the oracle's own
    end-of-run pickups of the jdon run (pickup.ckptA and pickup_cd.ckptA, 36 tiles, iteration 36010) read back into a
    State and written again by WRITE_PICKUP and CD_CODE_WRITE_PICKUP give every .data and .meta file byte for byte
    (72 + 72 files: record placement, tile names, meta dimList with exch2_txGlobalo/tyGlobalo, map2gl, fldList,
    timeInterval). Control: the plain (non-exch2) layout path is refused for this build (raises), and a field changed
    by 1 ulp at one point changes its tile's data file."""
    import filecmp
    from mitjax.model.src.ini_parms import ini_parms_io
    from mitjax.model.src.write_pickup import write_pickup
    from mitjax.pkg.cd_code.cd_code_write_pickup import cd_code_write_pickup
    from mitjax.pkg.exch2.w2_eeboot import w2_eeboot
    from mitjax.pkg.mdsio.mdsio_write_field import MdsContext
    s = _s()
    sz = s.cfg.size
    Nr, OLx, OLy = sz.Nr, sz.OLx, sz.OLy
    rd = s.rundir
    io = ini_parms_io(s.e)
    assert not io.globalFiles and io.pickup_write_mdsio
    w2 = w2_eeboot(s.e)[0]
    assert w2.W2_useE2ioLayOut
    st = G.teacher_state(s, s.its[-1], [("S00_begin", None)])
    # pickup_cd.ckptA: uVelD, vVelD, uNM1, vNM1 (records 1-4, Nr levels each), etaNm1 (record 4*Nr+1)
    cd = _read_tiled(rd, "pickup_cd.ckptA", sz, 4*Nr+1)
    for n, r in (("uVelD", 0), ("vVelD", 1), ("uNM1", 2), ("vNM1", 3)):
        st = _into(st, n, cd[:, r*Nr:(r+1)*Nr], OLx, OLy)
    st = _into(st, "etaNm1", cd[:, 4*Nr], OLx, OLy)
    # pickup.ckptA: the fldList of its meta file, 3-D fields of Nr records, 2-D of one
    meta = (rd / "pickup.ckptA.001.001.meta").read_text()
    flds = [x.strip() for x in meta.split("fldList = {")[1].split("}")[0].replace("'", " ' ").split("'") if x.strip()]
    names = {"Uvel": "uVel", "Vvel": "vVel", "Theta": "theta", "Salt": "salt", "GuNm1": "guNm1", "GvNm1": "gvNm1",
             "GtNm1": "gtNm1", "GsNm1": "gsNm1", "PhiHyd": "totPhiHyd", "EtaN": "etaN", "dEtaHdt": "dEtaHdt",
             "EtaH": "etaHnm1"}                                    # write_pickup.F:359: EtaH is etaHnm1
    nlev = [1 if n in ("EtaN", "dEtaHdt", "EtaH") else Nr for n in flds]
    pk = _read_tiled(rd, "pickup.ckptA", sz, sum(nlev))
    r0 = 0
    for n, nl in zip(flds, nlev):
        st = _into(st, names[n], pk[:, r0] if nl == 1 else pk[:, r0:r0+nl], OLx, OLy)
        r0 += nl
    myIter = s.its[0] + 10                                                     # the run's last iteration (36010)
    myTime = float(s.prm.time.startTime + s.prm.time.deltaTClock*10)
    d = tmp_path / "w"
    d.mkdir()
    mds = MdsContext(d, sz, exch2=True, w2=w2, useSingleCpuIO=io.useSingleCpuIO, mdsioLocalDir=io.mdsioLocalDir,
                     the_run_name=io.the_run_name)
    write_pickup(False, "ckptA", myTime, myIter, cfg=s.cfg, params=s.params, ip=s.prm.init, io=io, state=st,
                 mds=mds)
    cd_code_write_pickup(False, "ckptA", myTime, myIter, cfg=s.cfg, io=io, state=st, mds=mds)
    files = sorted(p.name for p in rd.glob("pickup*.ckptA.*"))
    assert len(files) == 144
    bad = [f for f in files if not filecmp.cmp(d / f, rd / f, shallow=False)]
    assert not bad, bad[:6]
    # controls
    with pytest.raises(NotImplementedError):
        mds0 = MdsContext(tmp_path / "x", sz, exch2=True)                     # an exch2 build without the W2 blocks
        cd_code_write_pickup(False, "ckptA", myTime, myIter, cfg=s.cfg, io=io, state=st, mds=mds0)
    st2 = _into(st, "uVelD", np.nextafter(cd[:, 0:Nr], np.inf), OLx, OLy)
    d2 = tmp_path / "c"
    d2.mkdir()
    mds2 = MdsContext(d2, sz, exch2=True, w2=w2, useSingleCpuIO=False, mdsioLocalDir=io.mdsioLocalDir,
                      the_run_name=io.the_run_name)
    cd_code_write_pickup(False, "ckptA", myTime, myIter, cfg=s.cfg, io=io, state=st2, mds=mds2)
    assert not filecmp.cmp(d2 / "pickup_cd.ckptA.001.001.data", rd / "pickup_cd.ckptA.001.001.data", shallow=False)


def test_sbo_host_records_equal_oracle():
    """run.SboHost (DO_THE_MODEL_IO's SBO_CALC + SBO_OUTPUT in the driver, GO lane) on teacher-forced carries: the
    initial call (INITIALISE_VARIA's DO_THE_MODEL_IO: S00 State, hFac of S00) and the end of every dumped step (S16
    State, rhoInSitu of S04, hFac of S12 as the State carries them under NONLIN_FRSURF, sIceLoad of P01) print the
    oracle STDOUT's %SBO records character for character (16 records). Control: hFacC of S00 instead of S12 at the
    first step changes a record."""
    from types import SimpleNamespace
    from mitjax.drivers.run import SboHost
    s = _s()
    m = SimpleNamespace(cfg=s.cfg, exp=s.e, fp=s.fp, grid=s.grid, ex=s.ex, prm=s.prm)
    sh = SboHost(m)
    want = [ln.rstrip("\n") for ln in open(s.rundir / "output.txt") if "%SBO" in ln]

    def carry(it, stage, rho_stage, hfac_stage, ice_stage):
        st = G.teacher_state(s, it, [("S00_begin", None), (stage, None), (rho_stage, ("rhoInSitu",)),
                                     (hfac_stage, ("hFacC", "hFacW", "hFacS", "recip_hFacC"))])
        g00 = G.dumped(s, it, "G00_geometry")
        st = st.replace(recip_hFacW=G._farr(g00["recip_hFacW"], st.recip_hFacW),
                        recip_hFacS=G._farr(g00["recip_hFacS"], st.recip_hFacS))
        ff, phi0 = G.teacher_ff(s, it, ice_stage)
        return (st, ff, phi0, None)
    tp = s.prm.time
    got = sh.records(float(tp.startTime), int(tp.nIter0),
                     carry(s.its[0], "S00_begin", "S00_begin", "S00_begin", "S00_begin"))
    for it in s.its:
        c = carry(it, "S16_blocking_exchanges", "S04_oceanic_phys", "S12_calc_rstar", "P01_external_forcing_surf")
        k = it + 1 - tp.nIter0
        assert sh.prints(tp.startTime + tp.deltaTClock*k, it + 1)
        got += sh.records(float(tp.startTime + tp.deltaTClock*k), it + 1, c)
    assert len(got) == 16 and got == want[:16], (got, want[:16])
    c = carry(s.its[0], "S16_blocking_exchanges", "S04_oceanic_phys", "S00_begin", "P01_external_forcing_surf")
    bad = sh.records(float(tp.startTime + tp.deltaTClock), s.its[0] + 1, c)
    assert bad != want[4:8]


def test_mnc_output_only_read_switches_refused():
    """mnc is integrated as output only (decided 2026-10-01 by Nikolay): tutorial_baroclinic_gyre (useMNC=.TRUE.)
    passes the set-up check; negative controls: each MNC read switch (pickup_read_mnc, mnc_read_bathy, ...) set
    .TRUE. by a namelist overlay raises UnsupportedOption. The output-only gate: lane A's mncoff / diagoff oracle
    runs, scripts/tests/test_m2_oracle.py::test_output_request_runs_change_no_model_number."""
    from mitjax.config.params import UnsupportedOption
    from mitjax.drivers.model import MNC_READ_SWITCHES, check_packages, with_namelist
    e = G.gg.experiment("tutorial_baroclinic_gyre", "input")
    assert dict(e.cfg.use)["useMNC"]
    check_packages(e.cfg, exp=e)
    for f, g, k, _ in MNC_READ_SWITCHES:
        e2 = with_namelist(e, {(f, g, k): (True, "bool")})
        with pytest.raises(UnsupportedOption, match=k):
            check_packages(e2.cfg, exp=e2)


def test_write_pickup_mnc_branch_guard():
    """WRITE_PICKUP's MNC branch runs only for `useMNC .AND. pickup_write_mnc` (write_pickup.F:389): the Model reads
    pickup_write_mnc from data.mnc (mnc_readparms.F:97 default .FALSE.; tutorial_baroclinic_gyre leaves it unset), and
    write_pickup goes on to the MDSIO part; negative control: pickup_write_mnc=.TRUE. raises (the MNC pickup is not
    ported). The whole baroclinic-gyre run with its pickup: docs/M1_ACCEPTANCE.md."""
    import dataclasses
    from types import SimpleNamespace
    from mitjax.model.src.ini_parms import IOParams
    from mitjax.model.src.write_pickup import write_pickup
    from mitjax.params_io import RunParams, fortran_default
    e = G.gg.experiment("tutorial_baroclinic_gyre", "input")
    assert dict(e.cfg.use)["useMNC"] and e.cfg.cpp.ALLOW_MNC
    assert RunParams(e.run).get("data.mnc", "MNC_01", "pickup_write_mnc", default=fortran_default(
        "pkg/mnc/mnc_readparms.F:97", "pickup_write_mnc", e)) is False
    ip = SimpleNamespace(useMNC=True)
    io = IOParams(pickup_write_mdsio=False)                 # :98 returns before any state is touched
    assert write_pickup(False, "ckptA", 0.0, 0, cfg=e.cfg, params=None, ip=ip, io=io, state=None, mds=None) is None
    with pytest.raises(NotImplementedError, match="MNC pickups"):
        write_pickup(False, "ckptA", 0.0, 0, cfg=e.cfg, params=None, ip=ip,
                     io=dataclasses.replace(io, pickup_write_mnc=True), state=None, mds=None)
