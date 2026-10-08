"""TEMP_INTEGRATE: model/src/temp_integrate.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.ad.approx_advection import gad_advection_call
from mitjax.farray import FArray, loop_i, loop_j, loops_kji
from mitjax.model.src.adams_bashforth2 import adams_bashforth2
from mitjax.model.src.adams_bashforth3 import adams_bashforth3
from mitjax.model.src.apply_forcing import apply_forcing_t
from mitjax.model.src.calc_3d_diffusivity import calc_3d_diffusivity
from mitjax.model.src.calc_adv_flow import calc_adv_flow
from mitjax.model.src.cycle_tracer import cycle_tracer
from mitjax.model.src.impldiff import impldiff
from mitjax.model.src.timestep_tracer import timestep_tracer
from mitjax.pkg.generic_advdiff.gad_calc_rhs import gad_calc_rhs, gad_som_cfg
from mitjax.pkg.generic_advdiff.gad_h import GAD_TEMPERATURE
from mitjax.pkg.generic_advdiff.gad_implicit_r import gad_implicit_r
from mitjax.pkg.generic_advdiff.gad_som_advect import gad_som_advect

_OPT = "GAD_OPTIONS.h"      # temp_integrate.F:6-8  #include "GAD_OPTIONS.h" (ALLOW_GENERIC_ADVDIFF)


def _level(A, k):
    """A(1-OLx,1-OLy,k) passed to a 2-D dummy argument (sequence association, KERNEL_GUIDE §4): level k."""
    (_, ilo, ihi), (_, jlo, jhi), (_, klo, _) = A.dims
    kk = k - klo
    return FArray(A.data[:, getattr(kk, "value", kk)], A.name, i=(ilo, ihi), j=(jlo, jhi), tiled=A.tiled)


def _local2(like, name):
    """A local `_RL name(1-OLx:sNx+OLx,1-OLy:sNy+OLy)` of the tiles of `like` (3-D), every point NaN."""
    return _level(like, like.dims[2][1]).local(name)


def _local_kupdw(like, name):
    """A local `_RL name(1-OLx:sNx+OLx,1-OLy:sNy+OLy,2)` (the kUp/kDown slots), every point NaN."""
    (_, ilo, ihi), (_, jlo, jhi), _ = like.dims
    d = jnp.full((like.data.shape[0], 2) + tuple(like.data.shape[2:]), jnp.nan, like.data.dtype)
    return FArray(d, name, i=(ilo, ihi), j=(jlo, jhi), k=(1, 2), tiled=like.tiled)


def temp_integrate(recip_hFac, uFld, vFld, wFld, KappaRk, myTime, myIter, *, cfg, grid, params, fp, ff, state,
                   probe=None, gm=None, mix=None, rbcs=None):
    """TEMP_INTEGRATE( bi, bj, recip_hFac, uFld, vFld, wFld, KappaRk, myTime, myIter, myThid )
    @63cdc0b model/src/temp_integrate.F:13-532

    C     | SUBROUTINE TEMP_INTEGRATE
    C     | o Calculate tendency for temperature and integrates
    C     |   forward in time. The temperature array is updated here
    C     |   while adjustments (filters, conv.adjustment) are applied
    C     |   later, in S/R TRACERS_CORRECTION_STEP.
    C recip_hFac :: reciprocal of cell open-depth factor (@ next iter)
    C uFld,vFld  :: Local copy of horizontal velocity field
    C wFld       :: Local copy of vertical velocity field
    C KappaRk    :: Vertical diffusion for Tempertature

    Returns (state, KappaRk): State with theta, gtNm1 (or the gtNm pair under ALLOW_ADAMSBASHFORTH_3) and som_T
    (under GAD_ALLOW_TS_SOM_ADV) updated. recip_hFac, uFld, vFld, wFld, KappaRk:
    (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr). `fp`, `ff`: the FORCING lane's ForcingParams and FFIELDS.h (APPLY_FORCING_T).
    `probe(stage, values)`: optional, called with the values of the dump stages T11_temp_gT, T12_temp_step,
    T13_temp_impl (reference/jaxdump/SUBSTEPS.md).

    The level loop DO k=Nr,1,-1 (:290-443) calls per-level routines and carries rTrans and fVer from level to level:
    an unrolled Python loop in the Fortran order (KERNEL_GUIDE §4). Ported for the M1 variants: AB on the tendency
    (AdamsBashforthGt: ADAMS_BASHFORTH2 per level), the forcing inside or outside AB (tracForcingOutAB), the
    explicit tendency of GAD_CALC_RHS (its ported schemes), TIMESTEP_TRACER, the implicit vertical step
    (GAD_IMPLICIT_R when INCLUDE_IMPLVERTADV_CODE, else IMPLDIFF), CYCLE_TRACER; under ALLOW_AUTODIFF the kappaRk
    reset (:212-220). Raise: AB on the tracer (AdamsBashforth_T: ADAMS_BASHFORTH2/3 with kArg = 0, CYCLE_AB_TRACER),
    diagnostics (GO lane: FREESURF_RESCALE_G of gT/gS and the AB history under nonlinFreeSurf > 0 is ported; lane
    M4ADLAB: DWNSLP_APPLY with useDOWN_SLOPE, :445-466, the state in `mix["dwnslp"]`)
    (useDiagnostics is .FALSE. for the kernels: ini_parms_tracer). `gm`: GMREDI.h (useGMRedi: GMREDI_CALC_DIFF in
    CALC_3D_DIFFUSIVITY and the GM/Redi fluxes of GAD_CALC_RHS; R5 arm).

    ADVECT lane (plan Task 14), marked arms: GAD_SOM_ADVECT with tempSOM_Advection (GAD_ALLOW_TS_SOM_ADV; the
    moments som_T are State fields, GAD_SOM_VARS.h) and GAD_ADVECTION with tempMultiDimAdvec (:243-284; probe stage
    T10_temp_adv after GAD_ADVECTION, where the oracle dumps it: inside the ELSEIF arm, :277); ALLOW_ADAMSBASHFORTH_3
    (gtNm pair of DYNVARS.h): ADAMS_BASHFORTH3 per level with AdamsBashforthGt (:376-383) and the TracAB argument
    gtNm(...,m2), m2 = 1+MOD(iterNb,2) (:335), of GAD_CALC_RHS (read only with AdamsBashforth_T, which raises)."""
    pr = probe if probe is not None else (lambda stage, values: None)
    ab3 = bool(cfg.cpp.ALLOW_ADAMSBASHFORTH_3)                                 # ADVECT lane
    if params.useDiagnostics:
        raise NotImplementedError("TEMP_INTEGRATE: DIAGNOSTICS_FILL (useDiagnostics) is not ported")
    sz = cfg.size
    Nr = sz.Nr
    g = grid
    theta = state.theta
    gtNm1 = None if ab3 else state.gtNm1                                        # DYNVARS.h:64 (AB2)
    gtNm = state.gtNm if ab3 else None                                          # DYNVARS.h:59 (AB3), ADVECT lane
    som_T = state.som_T if "som_T" in state else None                           # GAD_SOM_VARS.h:27, ADVECT lane
    iterNb = myIter                                                             # :148
    if params.staggerTimeStep:                                                  # :149
        iterNb = myIter - 1
    iMin = 0                                                                    # :160-163
    iMax = sz.sNx+1
    jMin = 0
    jMax = sz.sNy+1

    if params.AdamsBashforth_T:                                                 # :179-196
        raise NotImplementedError("TEMP_INTEGRATE: AB on the tracer (AdamsBashforth_T) is not ported")

    k3, j3, i3 = loops_kji((1, Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
    gT_loc = theta.local("gT_loc").at[i3, j3, k3].set(0.)                       # :199-205
    fVer = _local_kupdw(theta, "fVer")
    jA = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    iA = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    fVer = fVer.at[iA, jA, 1].set(0.)                                           # :206-211
    fVer = fVer.at[iA, jA, 2].set(0.)
    if cfg.cpp.ALLOW_AUTODIFF:                                                  # :212-220
        KappaRk = KappaRk.at[i3, j3, k3].set(0.)

    if cfg.cpp.INCLUDE_CALC_DIFFUSIVITY_CALL:                                   # :232-241
        KappaRk = calc_3d_diffusivity(iMin, iMax, jMin, jMax, GAD_TEMPERATURE, params.useGMRedi, params.useKPP,
                                      KappaRk, cfg=cfg, grid=grid, params=params, state=state, gm=gm,
                                      mix=mix)                          # vermix lane: KPP.h

    if not cfg.cpp.flag("DISABLE_MULTIDIM_ADVECTION", _OPT):                    # :243-284 (ADVECT lane arm)
        if cfg.cpp.flag("GAD_ALLOW_TS_SOM_ADV", _OPT) and params.tempSOM_Advection:   # :259-269
            som_T, gT_loc = gad_som_advect(
                params.tempImplVertAdv, params.tempAdvScheme, params.tempVertAdvScheme, GAD_TEMPERATURE,
                params.dTtracerLev, uFld, vFld, wFld, theta, som_T, gT_loc, myTime, myIter,
                cfg=gad_som_cfg(cfg, params), grid=grid, params=params)
        elif params.tempMultiDimAdvec:                                          # :270-283
            gT_loc = gad_advection_call(           # lane M4ADCS32ICE: mitjax/ad/approx_advection.py
                params.tempImplVertAdv, params.tempAdvScheme, params.tempVertAdvScheme, GAD_TEMPERATURE,
                params.dTtracerLev, uFld, vFld, wFld, theta, gT_loc, myTime, myIter,
                cfg=cfg, grid=grid, params=params)
            pr("T10_temp_adv", {"gT_loc": gT_loc})

    calcAdvection = params.tempAdvection and not params.tempMultiDimAdvec      # :289
    rTrans = _local2(theta, "rTrans")
    uTrans = _local2(theta, "uTrans")
    vTrans = _local2(theta, "vTrans")
    rTransKp = _local2(theta, "rTransKp")
    maskUp = _local2(theta, "maskUp")
    xA = _local2(theta, "xA")
    yA = _local2(theta, "yA")
    fZon = _local2(theta, "fZon")
    fMer = _local2(theta, "fMer")
    gtForc = _local2(theta, "gtForc")
    gt_AB = _local2(theta, "gt_AB")
    # DO k=Nr,1,-1 (KERNEL_GUIDE §4): the level body below; k = Nr, 2, 1 static, k = Nr-1 .. 3 in a level scan
    # (mitjax/ops/scan_k.scan_levels: k is then a traced level index); `c` carries what the Fortran carries
    def _level_body(k, c):
        rTrans, uTrans, vTrans, rTransKp, maskUp, xA, yA, gtForc, fZon, fMer, fVer, gT_loc, gtNm1, gtNm, gt_AB = c
        kM1 = max(1, k-1)                                                       # :294; MINMAX-INT: integer (no tie or NaN case)
        kUp = 1+(k+1) % 2                                                       # :295
        kDown = 1+k % 2                                                         # :296
        rTrans, uTrans, vTrans, rTransKp, maskUp, xA, yA = calc_adv_flow(       # :308-313
            uFld, vFld, wFld, rTrans, uTrans, vTrans, rTransKp, maskUp, xA, yA, k,
            cfg=cfg, grid=grid, params=params)
        gtForc = gtForc.at[iA, jA].set(0.)                                      # :316-320
        if params.tempForcing:                                                  # :321-331
            gtForc = apply_forcing_t(gtForc, iMin, iMax, jMin, jMax, k, myTime, myIter,
                                     cfg=cfg, grid=grid, fp=fp, ff=ff, params=params,
                                     rbcs=None if rbcs is None else dict(rbcs, theta=state.theta))   # M3: RBCS
        if ab3:                                                                 # :333-348 (ADVECT lane)
            m2_even = (iterNb % 2) == 0                                         # :335  m2 = 1+MOD(iterNb,2)
            TracAB = FArray(jnp.where(m2_even, gtNm[0].data, gtNm[1].data), "gtNm", tiled=gtNm[0].tiled,
                            _dims=gtNm[0].dims)                                 # gtNm(1-OLx,1-OLy,1,bi,bj,m2)
        else:                                                                   # :349-364
            TracAB = gtNm1
        fZon, fMer, fVer, gT_loc = gad_calc_rhs(                                # :336-348 / :351-364
            iMin, iMax, jMin, jMax, k, kM1, kUp, kDown, xA, yA, maskUp, _level(uFld, k),
            _level(vFld, k), _level(wFld, k), uTrans, vTrans, rTrans, rTransKp,
            params.diffKhT, params.diffK4T, _level(KappaRk, k), params.diffKr4T,
            theta, TracAB, params.dTtracerLev,
            GAD_TEMPERATURE, params.tempAdvScheme, params.tempVertAdvScheme,
            calcAdvection, params.tempImplVertAdv, params.AdamsBashforth_T,
            params.tempVertDiff4, params.useGMRedi, params.useKPP, params.temp_stayPositive,
            fZon, fMer, fVer, gT_loc, myTime, myIter,
            cfg=cfg, grid=grid, params=params,
            diffKh_ne_0=params.diffKhT_ne_0, diffK4_ne_0=params.diffK4T_ne_0, gm=gm,
            mix=None if mix is None else dict(mix, ff=ff, fp=fp))      # vermix lane: KPP_TRANSPORT_T
        if params.tempForcing and params.tracForcingOutAB != 1:                 # :368-374
            gT_loc = gT_loc.at[iA, jA, k].set(gT_loc[iA, jA, k] + gtForc[iA, jA])
        if params.AdamsBashforthGt:                                             # :376-395
            if ab3:                                                             # :378-382 (ADVECT lane)
                gT_loc, gtNm, gt_AB = adams_bashforth3(k, Nr, gT_loc, gtNm, gt_AB,
                                                       params.tempStartAB, iterNb, cfg=cfg, params=params)
            else:
                gT_loc, gtNm1, gt_AB = adams_bashforth2(k, Nr, gT_loc, gtNm1, gt_AB,  # :384-388
                                                        params.tempStartAB, iterNb, cfg=cfg, params=params)
        if params.tempForcing and params.tracForcingOutAB == 1:                 # :398-404
            gT_loc = gT_loc.at[iA, jA, k].set(gT_loc[iA, jA, k] + gtForc[iA, jA])
        if cfg.cpp.NONLIN_FRSURF and params.nonlinFreeSurf > 0:                 # :406-440 (GO lane)
            from mitjax.model.src.freesurf_rescale_g import freesurf_rescale_g
            gT_loc = freesurf_rescale_g(k, gT_loc, cfg=cfg, params=params, state=state, grid=grid)
            if params.AdamsBashforthGt:
                if ab3:                                                         # gtNm(..,1), gtNm(..,2)
                    gtNm = tuple(freesurf_rescale_g(k, a, cfg=cfg, params=params, state=state, grid=grid) for a in gtNm)
                else:
                    gtNm1 = freesurf_rescale_g(k, gtNm1, cfg=cfg, params=params, state=state, grid=grid)
        return (rTrans, uTrans, vTrans, rTransKp, maskUp, xA, yA, gtForc, fZon, fMer, fVer, gT_loc, gtNm1, gtNm, gt_AB)

    from mitjax.ops.scan_k import scan_levels
    c = (rTrans, uTrans, vTrans, rTransKp, maskUp, xA, yA, gtForc, fZon, fMer, fVer, gT_loc, gtNm1, gtNm, gt_AB)
    c = _level_body(Nr, c)                     # CALC_ADV_FLOW and GAD_CALC_RHS branch on k = Nr, k = 1 and
    if Nr >= 4:                                # (kM1 = MAX(1,k-1)) k = 2: those levels run as static calls
        c = scan_levels(_level_body, c, 3, Nr-1, down=True)
    if Nr >= 3:
        c = _level_body(2, c)
    if Nr >= 2:
        c = _level_body(1, c)
    rTrans, uTrans, vTrans, rTransKp, maskUp, xA, yA, gtForc, fZon, fMer, fVer, gT_loc, gtNm1, gtNm, gt_AB = c

    if cfg.cpp.ALLOW_DOWN_SLOPE and params.useDOWN_SLOPE:                       # :445-466   (lane M4ADLAB session 3)
        if mix is None or "dwnslp" not in mix:
            raise ValueError("TEMP_INTEGRATE: useDOWN_SLOPE needs the down-slope state `mix['dwnslp']`")
        from mitjax.pkg.down_slope.dwnslp_apply import dwnslp_apply
        gT_loc = dwnslp_apply(theta, gT_loc, recip_hFac, grid.recip_rA, grid.recip_drF, mix["dwnslp"], cfg=cfg,
                              usingZCoords=params.usingZCoords)        # kBottom = kLowC

    pr("T11_temp_gT", {"gT_loc": gT_loc})
    gT_loc = timestep_tracer(params.dTtracerLev, theta, gT_loc, myTime, myIter, cfg=cfg)  # :469-473
    pr("T12_temp_step", {"gT_loc": gT_loc})

    if cfg.cpp.INCLUDE_IMPLVERTADV_CODE:                                        # :480-504
        if params.tempImplVertAdv or params.implicitDiffusion:
            gT_loc = gad_implicit_r(params.tempImplVertAdv, params.tempVertAdvScheme, GAD_TEMPERATURE,
                                    params.dTtracerLev, KappaRk, recip_hFac, wFld, theta, gT_loc,
                                    myTime, myIter, cfg=cfg, grid=grid, params=params)
    elif params.implicitDiffusion:                                              # :497-503
        gT_loc = impldiff(iMin, iMax, jMin, jMax, GAD_TEMPERATURE, KappaRk, recip_hFac, gT_loc,
                          cfg=cfg, grid=grid, params=params)
    pr("T13_temp_impl", {"gT_loc": gT_loc, "kappaRk": KappaRk})

    # :506-527 (AdamsBashforth_T raised above): T(n) = T**
    theta = cycle_tracer(theta, gT_loc, myTime, myIter, cfg=cfg)                # :523-526
    if ab3:                                                                     # ADVECT lane
        state = state.replace(gtNm=gtNm)
    else:
        state = state.replace(gtNm1=gtNm1)
    if som_T is not None:                                                       # ADVECT lane
        state = state.replace(som_T=som_T)
    return state.replace(theta=theta), KappaRk
