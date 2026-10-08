"""SALT_INTEGRATE: model/src/salt_integrate.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.ad.approx_advection import gad_advection_call
from mitjax.farray import FArray, loop_i, loop_j, loops_kji
from mitjax.model.src.adams_bashforth2 import adams_bashforth2
from mitjax.model.src.adams_bashforth3 import adams_bashforth3
from mitjax.model.src.apply_forcing import apply_forcing_s
from mitjax.model.src.calc_3d_diffusivity import calc_3d_diffusivity
from mitjax.model.src.calc_adv_flow import calc_adv_flow
from mitjax.model.src.cycle_tracer import cycle_tracer
from mitjax.model.src.impldiff import impldiff
from mitjax.model.src.timestep_tracer import timestep_tracer
from mitjax.pkg.generic_advdiff.gad_calc_rhs import gad_calc_rhs, gad_som_cfg
from mitjax.pkg.generic_advdiff.gad_h import GAD_SALINITY
from mitjax.pkg.generic_advdiff.gad_implicit_r import gad_implicit_r
from mitjax.pkg.generic_advdiff.gad_som_advect import gad_som_advect

_OPT = "GAD_OPTIONS.h"      # salt_integrate.F:6-8  #include "GAD_OPTIONS.h" (ALLOW_GENERIC_ADVDIFF)


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


def salt_integrate(recip_hFac, uFld, vFld, wFld, KappaRk, myTime, myIter, *, cfg, grid, params, fp, ff, state,
                   probe=None, gm=None, mix=None):
    """SALT_INTEGRATE( bi, bj, recip_hFac, uFld, vFld, wFld, KappaRk, myTime, myIter, myThid )
    @63cdc0b model/src/salt_integrate.F:13-530

    C     | SUBROUTINE SALT_INTEGRATE
    C     | o Calculate tendency for salinity and integrates
    C     |   forward in time. The salinity array is updated here
    C     |   while adjustments (filters, conv.adjustment) are applied
    C     |   later, in S/R TRACERS_CORRECTION_STEP.
    C recip_hFac :: reciprocal of cell open-depth factor (@ next iter)
    C uFld,vFld  :: Local copy of horizontal velocity field
    C wFld       :: Local copy of vertical velocity field
    C KappaRk    :: Vertical diffusion for Salinity

    Returns (state, KappaRk): State with salt, the AB history gsNm1 (or the gsNm pair under ALLOW_ADAMSBASHFORTH_3)
    and the SOM moments som_S (GAD_SOM_VARS.h, under GAD_ALLOW_TS_SOM_ADV) updated. recip_hFac, uFld, vFld, wFld,
    KappaRk: (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr). `fp`, `ff`: the FORCING lane's ForcingParams and FFIELDS.h
    (APPLY_FORCING_S). `probe(stage, values)`: optional, called with the values of the dump stages T20_salt_adv
    (after GAD_ADVECTION, inside its ELSEIF arm as the oracle dumps it, :275), T21_salt_gS, T22_salt_step,
    T23_salt_impl (reference/jaxdump/SUBSTEPS.md).

    The literal mirror of TEMP_INTEGRATE (mitjax/model/src/temp_integrate.py; the Fortran files differ only in the
    tracer's names, comments and line numbers): created by the ADVECT lane (plan Task 14); the R5 lane adds its GM
    arms here. The level loop DO k=Nr,1,-1 (:288-441) calls per-level routines and carries rTrans and fVer from
    level to level: an unrolled Python loop in the Fortran order (KERNEL_GUIDE §4). Ported: GAD_SOM_ADVECT
    (saltSOM_Advection, :257-267) or GAD_ADVECTION (saltMultiDimAdvec, :268-281); AB on the tendency
    (AdamsBashforthGs: ADAMS_BASHFORTH2, or ADAMS_BASHFORTH3 under ALLOW_ADAMSBASHFORTH_3 with the TracAB argument
    gsNm(...,m2), m2 = 1+MOD(iterNb,2), :333); the forcing inside or outside AB (tracForcingOutAB); the explicit
    tendency of GAD_CALC_RHS (its ported schemes); TIMESTEP_TRACER; the implicit vertical step (GAD_IMPLICIT_R when
    INCLUDE_IMPLVERTADV_CODE, else IMPLDIFF); CYCLE_TRACER; under ALLOW_AUTODIFF the kappaRk reset (:210-218).
    Raise: AB on the tracer (AdamsBashforth_S: ADAMS_BASHFORTH2/3 with kArg = 0, CYCLE_AB_TRACER), NONLIN_FRSURF
    rescaling: ported (GO lane, FREESURF_RESCALE_G :404-438), DOWN_SLOPE: ported (lane M4ADLAB, DWNSLP_APPLY
    :443-464, `mix["dwnslp"]`), diagnostics (useDiagnostics is
    .FALSE. for the kernels: ini_parms_tracer). `gm`: GMREDI.h (useGMRedi: GMREDI_CALC_DIFF in CALC_3D_DIFFUSIVITY
    and the GM/Redi fluxes of GAD_CALC_RHS; R5 arm)."""
    pr = probe if probe is not None else (lambda stage, values: None)
    ab3 = bool(cfg.cpp.ALLOW_ADAMSBASHFORTH_3)                                 # ADVECT lane
    if params.useDiagnostics:
        raise NotImplementedError("SALT_INTEGRATE: DIAGNOSTICS_FILL (useDiagnostics) is not ported")
    sz = cfg.size
    Nr = sz.Nr
    g = grid
    salt = state.salt
    gsNm1 = None if ab3 else state.gsNm1                                        # DYNVARS.h:64 (AB2)
    gsNm = state.gsNm if ab3 else None                                          # DYNVARS.h:59 (AB3), ADVECT lane
    som_S = state.som_S if "som_S" in state else None                           # GAD_SOM_VARS.h:27, ADVECT lane
    iterNb = myIter                                                             # :146
    if params.staggerTimeStep:                                                  # :147
        iterNb = myIter - 1
    iMin = 0                                                                    # :158-161
    iMax = sz.sNx+1
    jMin = 0
    jMax = sz.sNy+1

    if params.AdamsBashforth_S:                                                 # :177-194
        raise NotImplementedError("SALT_INTEGRATE: AB on the tracer (AdamsBashforth_S) is not ported")

    k3, j3, i3 = loops_kji((1, Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
    gS_loc = salt.local("gS_loc").at[i3, j3, k3].set(0.)                       # :197-203
    fVer = _local_kupdw(salt, "fVer")
    jA = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    iA = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    fVer = fVer.at[iA, jA, 1].set(0.)                                           # :204-209
    fVer = fVer.at[iA, jA, 2].set(0.)
    if cfg.cpp.ALLOW_AUTODIFF:                                                  # :210-218
        KappaRk = KappaRk.at[i3, j3, k3].set(0.)

    if cfg.cpp.INCLUDE_CALC_DIFFUSIVITY_CALL:                                   # :230-239
        KappaRk = calc_3d_diffusivity(iMin, iMax, jMin, jMax, GAD_SALINITY, params.useGMRedi, params.useKPP,
                                      KappaRk, cfg=cfg, grid=grid, params=params, state=state, gm=gm,
                                      mix=mix)                          # vermix lane: KPP.h

    if not cfg.cpp.flag("DISABLE_MULTIDIM_ADVECTION", _OPT):                    # :241-282 (ADVECT lane arm)
        if cfg.cpp.flag("GAD_ALLOW_TS_SOM_ADV", _OPT) and params.saltSOM_Advection:   # :257-267
            som_S, gS_loc = gad_som_advect(
                params.saltImplVertAdv, params.saltAdvScheme, params.saltVertAdvScheme, GAD_SALINITY,
                params.dTtracerLev, uFld, vFld, wFld, salt, som_S, gS_loc, myTime, myIter,
                cfg=gad_som_cfg(cfg, params), grid=grid, params=params)
        elif params.saltMultiDimAdvec:                                          # :268-281
            gS_loc = gad_advection_call(           # lane M4ADCS32ICE: mitjax/ad/approx_advection.py
                params.saltImplVertAdv, params.saltAdvScheme, params.saltVertAdvScheme, GAD_SALINITY,
                params.dTtracerLev, uFld, vFld, wFld, salt, gS_loc, myTime, myIter,
                cfg=cfg, grid=grid, params=params)
            pr("T20_salt_adv", {"gS_loc": gS_loc})

    calcAdvection = params.saltAdvection and not params.saltMultiDimAdvec      # :287
    rTrans = _local2(salt, "rTrans")
    uTrans = _local2(salt, "uTrans")
    vTrans = _local2(salt, "vTrans")
    rTransKp = _local2(salt, "rTransKp")
    maskUp = _local2(salt, "maskUp")
    xA = _local2(salt, "xA")
    yA = _local2(salt, "yA")
    fZon = _local2(salt, "fZon")
    fMer = _local2(salt, "fMer")
    gsForc = _local2(salt, "gsForc")
    gs_AB = _local2(salt, "gs_AB")
    # DO k=Nr,1,-1 (KERNEL_GUIDE §4): the level body below; k = Nr, 2, 1 static, k = Nr-1 .. 3 in a level scan
    # (mitjax/ops/scan_k.scan_levels: k is then a traced level index); `c` carries what the Fortran carries
    def _level_body(k, c):
        rTrans, uTrans, vTrans, rTransKp, maskUp, xA, yA, gsForc, fZon, fMer, fVer, gS_loc, gsNm1, gsNm, gs_AB = c
        kM1 = max(1, k-1)                                                       # :292; MINMAX-INT: integer (no tie or NaN case)
        kUp = 1+(k+1) % 2                                                       # :293
        kDown = 1+k % 2                                                         # :294
        rTrans, uTrans, vTrans, rTransKp, maskUp, xA, yA = calc_adv_flow(       # :306-311
            uFld, vFld, wFld, rTrans, uTrans, vTrans, rTransKp, maskUp, xA, yA, k,
            cfg=cfg, grid=grid, params=params)
        gsForc = gsForc.at[iA, jA].set(0.)                                      # :314-318
        if params.saltForcing:                                                  # :319-329
            gsForc = apply_forcing_s(gsForc, iMin, iMax, jMin, jMax, k, myTime, myIter,
                                     cfg=cfg, grid=grid, fp=fp, ff=ff, params=params)
        if ab3:                                                                 # :331-346 (ADVECT lane)
            m2_even = (iterNb % 2) == 0                                         # :333  m2 = 1+MOD(iterNb,2)
            TracAB = FArray(jnp.where(m2_even, gsNm[0].data, gsNm[1].data), "gsNm", tiled=gsNm[0].tiled,
                            _dims=gsNm[0].dims)                                 # gsNm(1-OLx,1-OLy,1,bi,bj,m2)
        else:                                                                   # :347-362
            TracAB = gsNm1
        fZon, fMer, fVer, gS_loc = gad_calc_rhs(                                # :334-346 / :349-362
            iMin, iMax, jMin, jMax, k, kM1, kUp, kDown, xA, yA, maskUp, _level(uFld, k),
            _level(vFld, k), _level(wFld, k), uTrans, vTrans, rTrans, rTransKp,
            params.diffKhS, params.diffK4S, _level(KappaRk, k), params.diffKr4S,
            salt, TracAB, params.dTtracerLev,
            GAD_SALINITY, params.saltAdvScheme, params.saltVertAdvScheme,
            calcAdvection, params.saltImplVertAdv, params.AdamsBashforth_S,
            params.saltVertDiff4, params.useGMRedi, params.useKPP, params.salt_stayPositive,
            fZon, fMer, fVer, gS_loc, myTime, myIter,
            cfg=cfg, grid=grid, params=params,
            diffKh_ne_0=params.diffKhS_ne_0, diffK4_ne_0=params.diffK4S_ne_0, gm=gm,
            mix=None if mix is None else dict(mix, ff=ff, fp=fp))      # vermix lane: KPP_TRANSPORT_S
        if params.saltForcing and params.tracForcingOutAB != 1:                 # :366-372
            gS_loc = gS_loc.at[iA, jA, k].set(gS_loc[iA, jA, k] + gsForc[iA, jA])
        if params.AdamsBashforthGs:                                             # :374-393
            if ab3:                                                             # :376-380 (ADVECT lane)
                gS_loc, gsNm, gs_AB = adams_bashforth3(k, Nr, gS_loc, gsNm, gs_AB,
                                                       params.saltStartAB, iterNb, cfg=cfg, params=params)
            else:
                gS_loc, gsNm1, gs_AB = adams_bashforth2(k, Nr, gS_loc, gsNm1, gs_AB,  # :382-386
                                                        params.saltStartAB, iterNb, cfg=cfg, params=params)
        if params.saltForcing and params.tracForcingOutAB == 1:                 # :396-402
            gS_loc = gS_loc.at[iA, jA, k].set(gS_loc[iA, jA, k] + gsForc[iA, jA])
        if cfg.cpp.NONLIN_FRSURF and params.nonlinFreeSurf > 0:                 # :404-438 (GO lane)
            from mitjax.model.src.freesurf_rescale_g import freesurf_rescale_g
            gS_loc = freesurf_rescale_g(k, gS_loc, cfg=cfg, params=params, state=state, grid=grid)
            if params.AdamsBashforthGs:
                if ab3:                                                         # gtNm(..,1), gtNm(..,2)
                    gsNm = tuple(freesurf_rescale_g(k, a, cfg=cfg, params=params, state=state, grid=grid) for a in gsNm)
                else:
                    gsNm1 = freesurf_rescale_g(k, gsNm1, cfg=cfg, params=params, state=state, grid=grid)
        return (rTrans, uTrans, vTrans, rTransKp, maskUp, xA, yA, gsForc, fZon, fMer, fVer, gS_loc, gsNm1, gsNm, gs_AB)

    from mitjax.ops.scan_k import scan_levels
    c = (rTrans, uTrans, vTrans, rTransKp, maskUp, xA, yA, gsForc, fZon, fMer, fVer, gS_loc, gsNm1, gsNm, gs_AB)
    c = _level_body(Nr, c)                     # CALC_ADV_FLOW and GAD_CALC_RHS branch on k = Nr, k = 1 and
    if Nr >= 4:                                # (kM1 = MAX(1,k-1)) k = 2: those levels run as static calls
        c = scan_levels(_level_body, c, 3, Nr-1, down=True)
    if Nr >= 3:
        c = _level_body(2, c)
    if Nr >= 2:
        c = _level_body(1, c)
    rTrans, uTrans, vTrans, rTransKp, maskUp, xA, yA, gsForc, fZon, fMer, fVer, gS_loc, gsNm1, gsNm, gs_AB = c

    if cfg.cpp.ALLOW_DOWN_SLOPE and params.useDOWN_SLOPE:   # :443-464   (lane M4ADLAB session 3)
        if mix is None or "dwnslp" not in mix:
            raise ValueError("SALT_INTEGRATE: useDOWN_SLOPE needs the down-slope state `mix['dwnslp']`")
        from mitjax.pkg.down_slope.dwnslp_apply import dwnslp_apply
        gS_loc = dwnslp_apply(salt, gS_loc, recip_hFac, grid.recip_rA, grid.recip_drF, mix["dwnslp"], cfg=cfg,
                              usingZCoords=params.usingZCoords)        # kBottom = kLowC

    pr("T21_salt_gS", {"gS_loc": gS_loc})
    gS_loc = timestep_tracer(params.dTtracerLev, salt, gS_loc, myTime, myIter, cfg=cfg)  # :467-471
    pr("T22_salt_step", {"gS_loc": gS_loc})

    if cfg.cpp.INCLUDE_IMPLVERTADV_CODE:                                        # :478-502
        if params.saltImplVertAdv or params.implicitDiffusion:
            gS_loc = gad_implicit_r(params.saltImplVertAdv, params.saltVertAdvScheme, GAD_SALINITY,
                                    params.dTtracerLev, KappaRk, recip_hFac, wFld, salt, gS_loc,
                                    myTime, myIter, cfg=cfg, grid=grid, params=params)
    elif params.implicitDiffusion:                                              # :495-501
        gS_loc = impldiff(iMin, iMax, jMin, jMax, GAD_SALINITY, KappaRk, recip_hFac, gS_loc,
                          cfg=cfg, grid=grid, params=params)
    pr("T23_salt_impl", {"gS_loc": gS_loc, "kappaRk": KappaRk})

    # :502-523 (AdamsBashforth_S raised above): S(n) = S**
    salt = cycle_tracer(salt, gS_loc, myTime, myIter, cfg=cfg)                # :521-524
    if ab3:                                                                     # ADVECT lane
        state = state.replace(gsNm=gsNm)
    else:
        state = state.replace(gsNm1=gsNm1)
    if som_S is not None:                                                       # ADVECT lane
        state = state.replace(som_S=som_S)
    return state.replace(salt=salt), KappaRk
