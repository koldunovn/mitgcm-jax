"""PTRACERS_INTEGRATE: pkg/ptracers/ptracers_integrate.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import FArray, loop_i, loop_j, loops_kji
from mitjax.model.src.adams_bashforth2 import adams_bashforth2
from mitjax.model.src.calc_3d_diffusivity import calc_3d_diffusivity
from mitjax.model.src.calc_adv_flow import calc_adv_flow
from mitjax.model.src.cycle_tracer import cycle_tracer
from mitjax.model.src.freesurf_rescale_g import freesurf_rescale_g
from mitjax.model.src.impldiff import impldiff
from mitjax.model.src.timestep_tracer import timestep_tracer
from mitjax.pkg.generic_advdiff.gad_advection import gad_advection
from mitjax.pkg.generic_advdiff.gad_calc_rhs import gad_calc_rhs, gad_som_cfg
from mitjax.pkg.generic_advdiff.gad_h import GAD_TR1
from mitjax.pkg.generic_advdiff.gad_implicit_r import gad_implicit_r
from mitjax.pkg.generic_advdiff.gad_som_advect import gad_som_advect
from mitjax.pkg.ptracers.ptracers_forcing_surf import routine_of_build

_OPT = "PTRACERS_OPTIONS.h"     # ptracers_integrate.F:1  #include "PTRACERS_OPTIONS.h"


def _level(A, k):
    """A(1-OLx,1-OLy,k) passed to a 2-D dummy argument (sequence association, KERNEL_GUIDE §4): level k."""
    (_, ilo, ihi), (_, jlo, jhi), (_, klo, _) = A.dims
    kk = k - klo                                                    # k: a Python int or a traced level (KIdx)
    return FArray(A.data[:, getattr(kk, "value", kk)], A.name, i=(ilo, ihi), j=(jlo, jhi), tiled=A.tiled)


def _local2(like, name):
    """A local `_RL name(1-OLx:sNx+OLx,1-OLy:sNy+OLy)` of the tiles of `like` (3-D), every point NaN."""
    return _level(like, like.dims[2][1]).local(name)


def _local_kupdw(like, name):
    """A local `_RL name(1-OLx:sNx+OLx,1-OLy:sNy+OLy,2)` (the kUp/kDown slots), every point NaN."""
    (_, ilo, ihi), (_, jlo, jhi), _ = like.dims
    d = jnp.full((like.data.shape[0], 2) + tuple(like.data.shape[2:]), jnp.nan, like.data.dtype)
    return FArray(d, name, i=(ilo, ihi), j=(jlo, jhi), k=(1, 2), tiled=like.tiled)


def ptracers_integrate(recip_hFac, uFld, vFld, wFld, KappaRk, myTime, myIter, *, cfg, grid, params, ptr, ptf,
                       state, gm=None):
    """PTRACERS_INTEGRATE( bi, bj, recip_hFac, uFld, vFld, wFld, KappaRk, myTime, myIter, myThid )
    @63cdc0b pkg/ptracers/ptracers_integrate.F:10-528

    C     Calculates tendency for passive tracers and integrates forward in
    C     time. The tracer array is updated here while adjustments (filters,
    C     conv.adjustment) are applied later, in S/R TRACERS_CORRECTION_STEP
    C  recip_hFac       :: reciprocal of cell open-depth factor (@ next iter)
    C  uFld, vFld, wFld :: Local copy of velocity field (3 components)
    C  KappaRk          :: vertical diffusion used for one passive tracer

    Returns (ptf, KappaRk): PTRACERS_FIELDS.h `ptf` with pTracer, gpTrNm1 (and the SOM moments `ptf.som`) of every
    tracer 1..PTRACERS_numInUse with PTRACERS_StepFwd updated. `ptr`: PTRACERS_PARAMS.h/PTRACERS_START.h; `params`:
    PARAMS.h (the kernels see useDiagnostics = .FALSE.; DIAGNOSTICS_FILL calls are output only); `state`: DYNVARS.h
    (IVDConvCount, diffKr for CALC_3D_DIFFUSIVITY).

    The same structure as TEMP_INTEGRATE (mitjax/model/src/temp_integrate.py), per tracer: tendency reset (:159-165),
    fVer reset (:199-204), the ALLOW_AUTODIFF kappaRk reset (:205-213), CALC_3D_DIFFUSIVITY (:217-222), the
    multi-dimensional advection (GAD_SOM_ADVECT with PTRACERS_ALLOW_DYN_STATE and SOM advection, :234-249, moments in
    `ptf.som`; GAD_ADVECTION with PTRACERS_MultiDimAdv, :251-265), then the level loop DO k=Nr,1,-1 (:271-400, in
    the Fortran order, k = Nr-1 .. 3 as a level scan, KERNEL_GUIDE §4: CALC_ADV_FLOW, PTRACERS_APPLY_FORCING (the
    build's own version: tutorial_global_oce_latlon has one), GAD_CALC_RHS, the forcing inside or outside AB
    (tracForcingOutAB),
    ADAMS_BASHFORTH2 on the tendency (PTRACERS_AdamsBashGtr)), TIMESTEP_TRACER (:428-432), the implicit vertical step
    (GAD_IMPLICIT_R under INCLUDE_IMPLVERTADV_CODE, else IMPLDIFF with implicitDiffusion, :453-474), CYCLE_TRACER
    (:503-509). iMin..jMax = 0..sNx+1, 0..sNy+1 (:148-151); `dummy(Nr)` (:112, never read: trUseDiffKr4 is .FALSE.)
    is a NaN array.

    Raise: ALLOW_LONGSTEP (:131-133), useDiagnostics (:167-181, :306-311, :359-364, :476-494), AB on the tracer
    (PTRACERS_AdamsBash_Tr: ADAMS_BASHFORTH2 with kArg = 0 and CYCLE_AB_TRACER, :189-197, :496-502), DOWN_SLOPE
    (:402-425), useMATRIX (:351, :436-445), OBCS (:511-519). Under NONLIN_FRSURF with nonlinFreeSurf > 0 the tendency
    and, with AB on it, gpTrNm1 are rescaled per level by FREESURF_RESCALE_G (:376-397; lane GO's port, `state`
    holds rStarExpC). GM/Redi for a passive tracer (PTRACERS_useGMRedi): GMREDI_CALC_DIFF in CALC_3D_DIFFUSIVITY
    and the GMREDI_X/Y/RTRANSPORT fluxes of GAD_CALC_RHS with the GMREDI.h state `gm` (R5 lane's arms)."""
    if cfg.cpp.flag("ALLOW_LONGSTEP", _OPT):                                    # :131-133
        raise NotImplementedError("PTRACERS_INTEGRATE: ALLOW_LONGSTEP is not ported")
    if params.useDiagnostics:
        raise NotImplementedError("PTRACERS_INTEGRATE: DIAGNOSTICS_FILL (useDiagnostics) is not ported")
    if cfg.cpp.flag("ALLOW_MATRIX", _OPT):                                      # :351, :436-445
        raise NotImplementedError("PTRACERS_INTEGRATE: useMATRIX is not ported")
    sz = cfg.size
    Nr = sz.Nr
    iterNb = myIter                                                             # :135
    if params.staggerTimeStep:                                                  # :136
        iterNb = myIter - 1
    iMin = 0                                                                    # :148-151
    iMax = sz.sNx+1
    jMin = 0
    jMax = sz.sNy+1
    apply_forcing = routine_of_build(cfg, "ptracers_apply_forcing")
    dummy = FArray(jnp.full((Nr,), jnp.nan), "dummy", k=(1, Nr), tiled=False)  # :112  _RL dummy(Nr)

    for iTracer in range(1, ptr.PTRACERS_numInUse+1):                           # :154
        if not ptr.PTRACERS_StepFwd[iTracer-1]:                                 # :155
            continue
        n = iTracer-1
        GAD_TR = GAD_TR1 + iTracer - 1                                          # :156
        pTracer = ptf.pTracer[n]
        gpTrNm1 = ptf.gpTrNm1[n]
        k3, j3, i3 = loops_kji((1, Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
        gTracer = pTracer.local("gTracer").at[i3, j3, k3].set(0.)               # :159-165
        if ptr.PTRACERS_AdamsBash_Tr[n]:                                        # :189-197
            raise NotImplementedError("PTRACERS_INTEGRATE: AB on the tracer (PTRACERS_AdamsBash_Tr) is not ported")
        fVer = _local_kupdw(pTracer, "fVer")
        jA = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
        iA = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
        fVer = fVer.at[iA, jA, 1].set(0.)                                       # :199-204
        fVer = fVer.at[iA, jA, 2].set(0.)
        if cfg.cpp.flag("ALLOW_AUTODIFF", _OPT):                                # :205-213
            KappaRk = KappaRk.at[i3, j3, k3].set(0.)
        KappaRk = calc_3d_diffusivity(iMin, iMax, jMin, jMax, GAD_TR,           # :217-222
                                      ptr.PTRACERS_useGMRedi[n], ptr.PTRACERS_useKPP[n], KappaRk,
                                      cfg=cfg, grid=grid, params=params, state=state, ptr=ptr, gm=gm)

        if not cfg.cpp.flag("DISABLE_MULTIDIM_ADVECTION", _OPT):                # :228-266
            if cfg.cpp.flag("PTRACERS_ALLOW_DYN_STATE", _OPT) and ptr.PTRACERS_SOM_Advection[n]:   # :234-248
                som, gTracer = gad_som_advect(
                    ptr.PTRACERS_ImplVertAdv[n], ptr.PTRACERS_advScheme[n], ptr.PTRACERS_advScheme[n], GAD_TR,
                    ptr.PTRACERS_dTLev, uFld, vFld, wFld, pTracer, ptf.som[n], gTracer, myTime, myIter,
                    cfg=gad_som_cfg(cfg, params), grid=grid, params=params)
                ptf = ptf.set("som", iTracer, som)
            elif ptr.PTRACERS_MultiDimAdv[n]:                                   # :249-265
                gTracer = gad_advection(
                    ptr.PTRACERS_ImplVertAdv[n], ptr.PTRACERS_advScheme[n], ptr.PTRACERS_advScheme[n], GAD_TR,
                    ptr.PTRACERS_dTLev, uFld, vFld, wFld, pTracer, gTracer, myTime, myIter,
                    cfg=cfg, grid=grid, params=params)

        calcAdvection = (not ptr.PTRACERS_MultiDimAdv[n]                        # :269-270
                         and ptr.PTRACERS_advScheme[n] != 0)
        rTrans = _local2(pTracer, "rTrans")
        uTrans = _local2(pTracer, "uTrans")
        vTrans = _local2(pTracer, "vTrans")
        rTransKp = _local2(pTracer, "rTransKp")
        maskUp = _local2(pTracer, "maskUp")
        xA = _local2(pTracer, "xA")
        yA = _local2(pTracer, "yA")
        fZon = _local2(pTracer, "fZon")
        fMer = _local2(pTracer, "fMer")
        gTrForc = _local2(pTracer, "gTrForc")
        gTr_AB = _local2(pTracer, "gTr_AB")
        # DO k=Nr,1,-1 (:271, KERNEL_GUIDE §4): the level body below; k = Nr, 2, 1 static, k = Nr-1 .. 3 in a level
        # scan (mitjax/ops/scan_k.scan_levels: k is then a traced level index); `c` carries what the Fortran carries
        def level_k(k, c):
            rTrans, uTrans, vTrans, rTransKp, maskUp, xA, yA, gTrForc, fZon, fMer, fVer, gTracer, gpTrNm1, gTr_AB = c
            kM1 = max(1, k-1)                                       # :276; MINMAX-INT: integer (no tie or NaN case)
            kUp = 1+(k+1) % 2                                                   # :277
            kDown = 1+k % 2                                                     # :278
            rTrans, uTrans, vTrans, rTransKp, maskUp, xA, yA = calc_adv_flow(   # :288-293
                uFld, vFld, wFld, rTrans, uTrans, vTrans, rTransKp, maskUp, xA, yA, k,
                cfg=cfg, grid=grid, params=params)
            gTrForc = gTrForc.at[iA, jA].set(0.)                                # :296-300
            gTrForc = apply_forcing(gTrForc, ptf.surfaceForcingPTr[n],          # :301-305
                                    iMin, iMax, jMin, jMax, k, iTracer, myTime, myIter,
                                    cfg=cfg, grid=grid, params=params, ptr=ptr)
            fZon, fMer, fVer, gTracer = gad_calc_rhs(                           # :315-335
                iMin, iMax, jMin, jMax, k, kM1, kUp, kDown, xA, yA, maskUp, _level(uFld, k),
                _level(vFld, k), _level(wFld, k), uTrans, vTrans, rTrans, rTransKp,
                ptr.PTRACERS_diffKh[n], ptr.PTRACERS_diffK4[n], _level(KappaRk, k), dummy,
                pTracer, gpTrNm1, ptr.PTRACERS_dTLev, GAD_TR,
                ptr.PTRACERS_advScheme[n], ptr.PTRACERS_advScheme[n],
                calcAdvection, ptr.PTRACERS_ImplVertAdv[n],
                ptr.PTRACERS_AdamsBash_Tr[n], False,
                ptr.PTRACERS_useGMRedi[n], ptr.PTRACERS_useKPP[n], ptr.PTRACERS_stayPositive[n],
                fZon, fMer, fVer, gTracer, myTime, myIter,
                cfg=cfg, grid=grid, params=params,
                diffKh_ne_0=ptr.PTRACERS_diffKh_ne_0[n], diffK4_ne_0=ptr.PTRACERS_diffK4_ne_0[n], gm=gm)
            if params.tracForcingOutAB != 1:                                    # :338-344
                gTracer = gTracer.at[iA, jA, k].set(gTracer[iA, jA, k] + gTrForc[iA, jA])
            if ptr.PTRACERS_AdamsBashGtr[n]:                                    # :351-365 (.NOT.useMATRIX above)
                gTracer, gpTrNm1, gTr_AB = adams_bashforth2(k, Nr, gTracer, gpTrNm1, gTr_AB,
                                                            ptr.PTRACERS_startAB[n], iterNb,
                                                            cfg=cfg, params=params)
            if params.tracForcingOutAB == 1:                                    # :368-374
                gTracer = gTracer.at[iA, jA, k].set(gTracer[iA, jA, k] + gTrForc[iA, jA])
            if cfg.cpp.flag("NONLIN_FRSURF", _OPT) and params.nonlinFreeSurf > 0:   # :376-397
                gTracer = freesurf_rescale_g(k, gTracer, cfg=cfg, params=params, state=state)    # :382-385
                if ptr.PTRACERS_AdamsBashGtr[n]:                                # :386-395
                    gpTrNm1 = freesurf_rescale_g(k, gpTrNm1, cfg=cfg, params=params, state=state)
            return rTrans, uTrans, vTrans, rTransKp, maskUp, xA, yA, gTrForc, fZon, fMer, fVer, gTracer, gpTrNm1, gTr_AB

        from mitjax.ops.scan_k import scan_levels
        c = (rTrans, uTrans, vTrans, rTransKp, maskUp, xA, yA, gTrForc, fZon, fMer, fVer, gTracer, gpTrNm1, gTr_AB)
        c = level_k(Nr, c)                     # CALC_ADV_FLOW and GAD_CALC_RHS branch on k = Nr, k = 1 and
        if Nr >= 4:                            # (kM1 = MAX(1,k-1)) k = 2: those levels run as static calls
            c = scan_levels(level_k, c, 3, Nr-1, down=True)
        if Nr >= 3:
            c = level_k(2, c)
        if Nr >= 2:
            c = level_k(1, c)
        rTrans, uTrans, vTrans, rTransKp, maskUp, xA, yA, gTrForc, fZon, fMer, fVer, gTracer, gpTrNm1, gTr_AB = c

        if cfg.cpp.flag("ALLOW_DOWN_SLOPE", _OPT) and ptr.PTRACERS_useDWNSLP[n]:   # :402-425
            raise NotImplementedError("PTRACERS_INTEGRATE: DWNSLP_APPLY is not ported")

        gTracer = timestep_tracer(ptr.PTRACERS_dTLev, pTracer, gTracer,        # :428-432
                                  myTime, myIter, cfg=cfg)

        if cfg.cpp.flag("INCLUDE_IMPLVERTADV_CODE", _OPT):                      # :453-474
            if ptr.PTRACERS_ImplVertAdv[n] or params.implicitDiffusion:
                gTracer = gad_implicit_r(ptr.PTRACERS_ImplVertAdv[n], ptr.PTRACERS_advScheme[n], GAD_TR,
                                         ptr.PTRACERS_dTLev, KappaRk, recip_hFac, wFld, pTracer, gTracer,
                                         myTime, myIter, cfg=cfg, grid=grid, params=params)
        elif params.implicitDiffusion:                                          # :467-473
            gTracer = impldiff(iMin, iMax, jMin, jMax, GAD_TR, KappaRk, recip_hFac, gTracer,
                               cfg=cfg, grid=grid, params=params, ptr=ptr)

        # :496-509  (PTRACERS_AdamsBash_Tr raised above): pTr(n) = pTr**
        pTracer = cycle_tracer(pTracer, gTracer, myTime, myIter, cfg=cfg)      # :505-508
        if cfg.cpp.flag("ALLOW_OBCS", _OPT) and cfg.use_flag("useOBCS"):        # :511-519
            raise NotImplementedError("PTRACERS_INTEGRATE: OBCS_APPLY_PTRACER is not ported")
        ptf = ptf.set("pTracer", iTracer, pTracer).set("gpTrNm1", iTracer, gpTrNm1)
    return ptf, KappaRk
