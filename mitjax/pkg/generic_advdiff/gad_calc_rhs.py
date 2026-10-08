"""GAD_CALC_RHS: pkg/generic_advdiff/gad_calc_rhs.F @63cdc0b (the driver of the explicit advective and diffusive
fluxes of one tracer at one level, and their divergence)."""

from typing import NamedTuple

from mitjax.farray import loop_i, loop_j
from mitjax.pkg.generic_advdiff.gad_c2_adv_r import gad_c2_adv_r
from mitjax.pkg.generic_advdiff.gad_c2_adv_x import gad_c2_adv_x
from mitjax.pkg.generic_advdiff.gad_c2_adv_y import gad_c2_adv_y
from mitjax.pkg.generic_advdiff.gad_diff_r import gad_diff_r
from mitjax.pkg.generic_advdiff.gad_diff_x import gad_diff_x
from mitjax.pkg.generic_advdiff.gad_diff_y import gad_diff_y
from mitjax.pkg.generic_advdiff.gad_c4_adv_x import gad_c4_adv_x
from mitjax.pkg.generic_advdiff.gad_c4_adv_y import gad_c4_adv_y
from mitjax.pkg.generic_advdiff.gad_dst3_adv_r import gad_dst3_adv_r
from mitjax.pkg.generic_advdiff.gad_dst3_adv_x import gad_dst3_adv_x
from mitjax.pkg.generic_advdiff.gad_dst3_adv_y import gad_dst3_adv_y
from mitjax.pkg.generic_advdiff.gad_h import ENUM_CENTERED_2ND, ENUM_CENTERED_4TH, ENUM_DST3, ENUM_UPWIND_3RD
from mitjax.pkg.generic_advdiff.gad_h import GAD_SALINITY, GAD_TEMPERATURE, GAD_TR1
from mitjax.pkg.generic_advdiff.gad_u3_adv_x import gad_u3_adv_x
from mitjax.pkg.generic_advdiff.gad_u3_adv_y import gad_u3_adv_y

_OPT = "GAD_OPTIONS.h"      # gad_calc_rhs.F:1  #include "GAD_OPTIONS.h"


class GadKernelCfg(NamedTuple):
    """The static configuration the GAD-A kernels read (`cfg.sNx`, `cfg.ALLOW_SMAG_3D_DIFFUSIVITY`, ...): SIZE.h and
    the CPP options as seen after GAD_OPTIONS.h (the kernels' own include). Same fields as the GAD-A replay
    harness's KernelCfg (mitjax/tests/gad_a_replay.py), built here from the model's config."""
    sNx: int
    sNy: int
    OLx: int
    OLy: int
    Nr: int
    ALLOW_AUTODIFF: bool
    TARGET_NEC_SX: bool
    OLD_DST3_FORMULATION: bool
    ALLOW_SMAG_3D_DIFFUSIVITY: bool
    ISOTROPIC_COS_SCALING: bool


def gad_kernel_cfg(cfg):
    s = cfg.size
    return GadKernelCfg(sNx=s.sNx, sNy=s.sNy, OLx=s.OLx, OLy=s.OLy, Nr=s.Nr,
                        **{n: cfg.cpp.flag(n, _OPT) for n in ("ALLOW_AUTODIFF", "TARGET_NEC_SX",
                                                               "OLD_DST3_FORMULATION", "ALLOW_SMAG_3D_DIFFUSIVITY",
                                                               "ISOTROPIC_COS_SCALING")})


class GadSomCfg(NamedTuple):
    """ADVECT lane (plan Task 14): the static configuration the SOM routines read (gad_som_advect.py and callees,
    gad_som_exchanges.py), built from the model's config and Params. Same fields as the GAD-B replay harness's
    SomCfg (mitjax/tests/gad_som_replay.py; a test asserts the equality)."""
    sNx: int
    sNy: int
    OLx: int
    OLy: int
    Nr: int
    GAD_ALLOW_TS_SOM_ADV: bool
    PTRACERS_ALLOW_DYN_STATE: bool
    ALLOW_AUTODIFF: bool
    ALLOW_DIAGNOSTICS: bool
    ALLOW_OBCS: bool
    useDiagnostics: bool
    useCubedSphereExchange: bool
    rigidLid: bool
    nonlinFreeSurf: int
    select_rStar: int
    uniformFreeSurfLev: bool
    tempSOM_Advection: bool
    saltSOM_Advection: bool
    ALLOW_EXCH2: bool = False           # ADVECT lane (M2): the cube passes read the W2 tile view


def gad_som_cfg(cfg, params):
    """GadSomCfg of the model: SIZE.h, the CPP options as GAD_OPTIONS.h sees them, the PARAMS.h / GAD.h switches
    from `params` (useDiagnostics is .FALSE. for every kernel: ini_parms_tracer)."""
    s = cfg.size
    flags = {n: cfg.cpp.flag(n, _OPT) for n in ("GAD_ALLOW_TS_SOM_ADV", "PTRACERS_ALLOW_DYN_STATE",
                                                "ALLOW_AUTODIFF", "ALLOW_DIAGNOSTICS", "ALLOW_OBCS")}
    if cfg.cpp.flag("ALLOW_PTRACERS", _OPT):
        # PTRACERS lane: the SOM routines include PTRACERS_OPTIONS.h after GAD_OPTIONS.h under ALLOW_PTRACERS
        # (gad_som_advect.F:1-4), which defines PTRACERS_ALLOW_DYN_STATE (tutorial_advection_in_gyre/code)
        flags["PTRACERS_ALLOW_DYN_STATE"] = cfg.cpp.flag("PTRACERS_ALLOW_DYN_STATE", "PTRACERS_OPTIONS.h")
    return GadSomCfg(sNx=s.sNx, sNy=s.sNy, OLx=s.OLx, OLy=s.OLy, Nr=s.Nr, **flags,
                     useDiagnostics=bool(params.useDiagnostics),
                     useCubedSphereExchange=bool(params.useCubedSphereExchange), rigidLid=bool(params.rigidLid),
                     nonlinFreeSurf=int(params.nonlinFreeSurf), select_rStar=int(params.select_rStar),
                     uniformFreeSurfLev=bool(params.uniformFreeSurfLev),
                     tempSOM_Advection=bool(params.tempSOM_Advection), saltSOM_Advection=bool(params.saltSOM_Advection),
                     ALLOW_EXCH2=bool(cfg.cpp.ALLOW_EXCH2))


def _level2(A, name):
    """A local 2-D array declared like the 2-D argument A, every point NaN (KERNEL_GUIDE §2)."""
    return A.local(name)


def gad_calc_rhs(iMin, iMax, jMin, jMax, k, kM1, kUp, kDown, xA, yA, maskUp, uFld, vFld, wFld,
                 uTrans, vTrans, rTrans, rTransKp1, diffKh, diffK4, KappaR, diffKr4, TracerN, TracAB,
                 deltaTLev, trIdentity, advectionSchArg, vertAdvecSchArg, calcAdvection, implicitAdvection,
                 applyAB_onTracer, trUseDiffKr4, trUseGMRedi, trUseKPP, trUseSmolHack,
                 fZon, fMer, fVerT, gTracer, myTime, myIter, *, cfg, grid, params, diffKh_ne_0, diffK4_ne_0,
                 gm=None, mix=None):
    """GAD_CALC_RHS( bi,bj,iMin,iMax,jMin,jMax,k,kM1,kUp,kDown, xA, yA, maskUp, uFld, vFld, wFld,
    uTrans, vTrans, rTrans, rTransKp1, diffKh, diffK4, KappaR, diffKr4, TracerN, TracAB, deltaTLev, trIdentity,
    advectionSchArg, vertAdvecSchArg, calcAdvection, implicitAdvection, applyAB_onTracer,
    trUseDiffKr4, trUseGMRedi, trUseKPP, trUseSmolHack, fZon, fMer, fVerT, gTracer, myTime, myIter, myThid )
    @63cdc0b pkg/generic_advdiff/gad_calc_rhs.F:10-798

    C Calculates the tendency of a tracer due to advection and diffusion.
    C It calculates the fluxes in each direction indepentently and then
    C sets the tendency to the divergence of these fluxes. The advective
    C fluxes are only calculated here when using the linear advection schemes
    C otherwise only the diffusive and parameterized fluxes are calculated.
    C The tendency is assumed to contain data on entry.
    C xA, yA           :: areas of X and Y face of tracer cells
    C maskUp           :: 2-D array for mask at W points
    C uFld, vFld, wFld :: Local copy of velocity field (3 components)
    C uTrans, vTrans   :: 2-D arrays of volume transports at U,V points
    C rTrans           :: 2-D arrays of volume transports at W points
    C rTransKp1        :: 2-D array of volume trans at W pts, interf k+1
    C diffKh           :: horizontal diffusion coefficient
    C diffK4           :: horizontal bi-harmonic diffusion coefficient
    C KappaR           :: 2-D array for vertical diffusion coefficient, interf k
    C diffKr4          :: 1-D array for vertical bi-harmonic diffusion coefficient
    C TracerN          :: tracer field @ time-step n
    C TracAB           :: current tracer field (or extrapolated fwd in time to n+1/2 if applying AB on Tr)
    C gTracer          :: tendency array;  fZon, fMer :: zonal, meridional flux
    C fVerT            :: 2 1/2D arrays for vertical advective flux

    Returns (fZon, fMer, fVerT, gTracer) (the O and U arguments in the Fortran order). `k`, `kM1`, `kUp`, `kDown`,
    `trIdentity`, the schemes and the LOGICAL arguments are static Python values; xA ... rTransKp1, KappaR, fZon,
    fMer are 2-D (1-OLx:sNx+OLx,1-OLy:sNy+OLy) (the caller's level sections), TracerN, TracAB, gTracer
    (...,Nr), fVerT (...,2), deltaTLev(Nr) and diffKr4(Nr) not tiled. The REAL tests `diffKh.NE.0.` (:328, :457) and
    `diffK4 .NE. 0.` (:229, :339, :468) select code: decided on the host by the caller (`diffKh_ne_0`,
    `diffK4_ne_0`, keyword-only: the one addition to the Fortran signature, KERNEL_GUIDE §4); a gradient with respect
    to diffKh must not cross zero.

    Ported: the centred 2nd-order advective fluxes (ENUM_CENTERED_2ND, :256-257, :385-386, :517-518 / :549-550),
    DST-3 (ENUM_DST3, :283-286, :412-415, :532-535 / :564-567; lane M4COL, 1D_ocean_ice_column),
    the Laplacian diffusive fluxes (GAD_DIFF_X/Y, :328-336, :457-465), the explicit vertical diffusion (GAD_DIFF_R,
    :611-615) or its omission under implicitDiffusion (:605-610), and the divergence (:770-784). Raise: the other
    advection schemes (plan Task 14), the bi-harmonic terms (diffK4, diffKr4), GM/Redi, KPP, OBCS, AIM, the
    Smolarkiewicz hack, useDiagnostics (DIAGNOSTICS_FILL / LAYERS_FILL, output only: pkg/diagnostics is not ported).
    ALLOW_AUTODIFF: :174 (`fVerT(1,1,kDown) = fVerT(1,1,kDown)`) changes no value; :176-186 run only in adjoint mode
    (inAdMode is .FALSE. in every forward step). The point loops are independent (each point reads only inputs or
    values this routine wrote in an earlier statement); the macros `_recip_hFacC`, `_maskW`, `_maskS` are the plain
    GRID.h fields (ALLOW_DEPTH_CONTROL raises).
    """
    if cfg.cpp.ALLOW_DEPTH_CONTROL:
        raise NotImplementedError("GAD_CALC_RHS: ALLOW_DEPTH_CONTROL (_recip_hFacC macro) is not ported")
    if cfg.cpp.flag("ALLOW_OBCS", _OPT) and params.useOBCS:
        raise NotImplementedError("GAD_CALC_RHS: OBCS_U1_ADV_TRACER (useOBCS) is not ported")
    if cfg.cpp.flag("ALLOW_AIM", _OPT):
        raise NotImplementedError("GAD_CALC_RHS: the ALLOW_AIM vertical-advection condition is not ported")
    if cfg.cpp.flag("ALLOW_DIAGNOSTICS", _OPT) and params.useDiagnostics:
        raise NotImplementedError("GAD_CALC_RHS: DIAGNOSTICS_FILL (useDiagnostics) is not ported")
    sz = cfg.size
    OLx, OLy, sNx, sNy = sz.OLx, sz.OLy, sz.sNx, sz.sNy
    kc = gad_kernel_cfg(cfg)
    g = grid
    advectionScheme = advectionSchArg                                           # :163
    vertAdvecScheme = vertAdvecSchArg                                           # :164

    advFac = 0.                                                                 # :196  0. _d 0
    if calcAdvection:                                                           # :197
        advFac = 1.                                                             #       1. _d 0
    rAdvFac = g.rkSign*advFac                                                   # :198
    if implicitAdvection:                                                       # :199
        rAdvFac = g.rkSign

    jA = loop_j(1-OLy, sNy+OLy)
    iA = loop_i(1-OLx, sNx+OLx)
    df4 = _level2(fZon, "df4")
    af = _level2(fZon, "af")
    df = _level2(fZon, "df")
    localT = _level2(fZon, "localT")
    locABT = _level2(fZon, "locABT")
    fZon = fZon.at[iA, jA].set(0.)                                              # :201-209
    fMer = fMer.at[iA, jA].set(0.)
    fVerT = fVerT.at[iA, jA, kUp].set(0.)
    df = df.at[iA, jA].set(0.)
    df4 = df4.at[iA, jA].set(0.)

    if applyAB_onTracer:                                                        # :212-218
        localT = localT.at[iA, jA].set(TracerN[iA, jA, k])
        locABT = locABT.at[iA, jA].set(TracAB[iA, jA, k])
    else:                                                                       # :219-226
        localT = localT.at[iA, jA].set(TracerN[iA, jA, k])
        locABT = locABT.at[iA, jA].set(TracerN[iA, jA, k])

    if diffK4_ne_0:                                                             # :229-245
        raise NotImplementedError("GAD_CALC_RHS: bi-harmonic diffusion (GAD_GRAD_X/Y, GAD_DEL2) is not ported")

    fZon = fZon.at[iA, jA].set(0.)                                              # :248-252

    if calcAdvection:                                                           # :255-325
        if advectionScheme == ENUM_CENTERED_2ND:                                # :256-257
            af = gad_c2_adv_x(k, uTrans, locABT, af, cfg=kc)
        elif advectionScheme in (ENUM_UPWIND_3RD, ENUM_CENTERED_4TH, ENUM_DST3):   # ADVECT lane, :263-284
            maskLocW = _level2(fZon, "maskLocW").at[iA, jA].set(g.maskW[iA, jA, k])   # :264-271
            if advectionScheme == ENUM_UPWIND_3RD:                              # :276-278
                af = gad_u3_adv_x(k, uTrans, maskLocW, locABT, af, cfg=kc)
            elif advectionScheme == ENUM_CENTERED_4TH:                          # :279-281
                af = gad_c4_adv_x(k, uTrans, maskLocW, locABT, af, cfg=kc, grid=g)
            else:                                                               # :283-286 (lane M4COL)
                af = gad_dst3_adv_x(k, True, deltaTLev[k], uTrans, uFld, maskLocW, locABT, af, cfg=kc, grid=g)
        else:
            raise NotImplementedError(f"GAD_CALC_RHS: advectionScheme {advectionScheme} (X) is not ported here "
                                      "(plan Task 14)")
        fZon = fZon.at[iA, jA].set(fZon[iA, jA] + af[iA, jA])                    # :308-312

    if diffKh_ne_0:                                                             # :328-336
        df = gad_diff_x(k, xA, diffKh, localT, df, cfg=kc, grid=g)
    else:
        df = df.at[iA, jA].set(0.)
    if cfg.cpp.flag("ALLOW_GMREDI", _OPT) and trUseGMRedi:                      # :343-352 (R5 arm)
        from mitjax.pkg.gmredi.gmredi_xtransport import gmredi_xtransport
        df = gmredi_xtransport(trIdentity, k, iMin, iMax+1, jMin, jMax, xA, maskUp, TracerN, df,
                               cfg=cfg, grid=grid, gm=gm)
    fZon = fZon.at[iA, jA].set(fZon[iA, jA] + df[iA, jA]*params.rhoFacC[k])     # :354-358

    fMer = fMer.at[iA, jA].set(0.)                                              # :377-381

    if calcAdvection:                                                           # :384-454
        if advectionScheme == ENUM_CENTERED_2ND:                                # :385-386
            af = gad_c2_adv_y(k, vTrans, locABT, af, cfg=kc)
        elif advectionScheme in (ENUM_UPWIND_3RD, ENUM_CENTERED_4TH, ENUM_DST3):   # ADVECT lane, :392-413
            maskLocS = _level2(fMer, "maskLocS").at[iA, jA].set(g.maskS[iA, jA, k])   # :393-400
            if advectionScheme == ENUM_UPWIND_3RD:                              # :406-408
                af = gad_u3_adv_y(k, vTrans, maskLocS, locABT, af, cfg=kc)
            elif advectionScheme == ENUM_CENTERED_4TH:                          # :409-411
                af = gad_c4_adv_y(k, vTrans, maskLocS, locABT, af, cfg=kc, grid=g)
            else:                                                               # :412-415 (lane M4COL)
                af = gad_dst3_adv_y(k, True, deltaTLev[k], vTrans, vFld, maskLocS, locABT, af, cfg=kc, grid=g)
        else:
            raise NotImplementedError(f"GAD_CALC_RHS: advectionScheme {advectionScheme} (Y) is not ported here "
                                      "(plan Task 14)")
        fMer = fMer.at[iA, jA].set(fMer[iA, jA] + af[iA, jA])                    # :437-441

    if diffKh_ne_0:                                                             # :457-465
        df = gad_diff_y(k, yA, diffKh, localT, df, cfg=kc, grid=g)
    else:
        df = df.at[iA, jA].set(0.)
    if cfg.cpp.flag("ALLOW_GMREDI", _OPT) and trUseGMRedi:                      # :472-481 (R5 arm)
        from mitjax.pkg.gmredi.gmredi_ytransport import gmredi_ytransport
        df = gmredi_ytransport(trIdentity, k, iMin, iMax, jMin, jMax+1, yA, maskUp, TracerN, df,
                               cfg=cfg, grid=grid, gm=gm)
    fMer = fMer.at[iA, jA].set(fMer[iA, jA] + df[iA, jA]*params.rhoFacC[k])     # :483-487

    if calcAdvection and not implicitAdvection and k >= 2:                      # :513-600
        tr = TracAB if applyAB_onTracer else TracerN                            # :515-516 / :547-548
        if vertAdvecScheme == ENUM_CENTERED_2ND:                                # :517-518 / :549-550
            af = gad_c2_adv_r(k, rTrans, tr, af, cfg=kc, grid=g)
        elif vertAdvecScheme == ENUM_DST3:                                      # :532-535 / :564-567 (lane M4COL)
            af = gad_dst3_adv_r(k, deltaTLev[k], rTrans, wFld, tr, af, cfg=kc, grid=g)
        else:
            raise NotImplementedError(f"GAD_CALC_RHS: vertAdvecScheme {vertAdvecScheme} (R) is not ported here "
                                      "(plan Task 14)")
        fVerT = fVerT.at[iA, jA, kUp].set(fVerT[iA, jA, kUp] + af[iA, jA]*g.maskInC[iA, jA])   # :581-585

    if params.implicitDiffusion:                                                # :605-610
        df = df.at[iA, jA].set(0.)
    else:                                                                       # :611-615
        df = gad_diff_r(k, maskUp, KappaR, TracerN, df, cfg=kc, grid=g, params=params)
    if trUseDiffKr4:                                                            # :617-621
        raise NotImplementedError("GAD_CALC_RHS: GAD_BIHARM_R (vertical bi-harmonic diffusion) is not ported")
    if cfg.cpp.flag("ALLOW_GMREDI", _OPT) and trUseGMRedi:                      # :623-632 (R5 arm)
        from mitjax.pkg.gmredi.gmredi_rtransport import gmredi_rtransport
        df = gmredi_rtransport(trIdentity, k, iMin, iMax, jMin, jMax, maskUp, TracerN, df,
                               cfg=cfg, grid=grid, gm=gm)
    fVerT = fVerT.at[iA, jA, kUp].set(fVerT[iA, jA, kUp] + df[iA, jA])          # :634-638

    if cfg.cpp.flag("ALLOW_KPP", _OPT) and trUseKPP and k >= 2:                 # :656-709 (vermix lane, M3)
        # `mix`: {"kppf": KPP.h fields, "ff": FFIELDS.h, "fp": forcing parameters; lane M4LAB: "kpp_p" KPP_PARAMS.h,
        # "salt_plume" SALT_PLUME.h} (KPP_TRANSPORT_T/S read them)
        df = df.at[iA, jA].set(0.)                                              # :658-662  0. _d 0
        if trIdentity == GAD_TEMPERATURE:                                       # :663-667
            from mitjax.pkg.kpp.kpp_transport_t import kpp_transport_t
            df = kpp_transport_t(iMin, iMax, jMin, jMax, k, kM1, df, myTime, myIter, cfg=cfg, grid=grid,
                                 params=params, fp=mix["fp"], ff=mix["ff"], kppf=mix["kppf"],
                                 kpp=mix.get("kpp_p"), gm=gm)                   # ALLOW_GMREDI arm (lane M4LAB)
        elif trIdentity == GAD_SALINITY:                                        # :668-672
            from mitjax.pkg.kpp.kpp_transport_s import kpp_transport_s
            df = kpp_transport_s(iMin, iMax, jMin, jMax, k, kM1, df, myTime, myIter, cfg=cfg, grid=grid,
                                 ff=mix["ff"], kppf=mix["kppf"], params=params, kpp=mix.get("kpp_p"), gm=gm,
                                 salt_plume=mix.get("salt_plume"))      # GMREDI / SALT_PLUME arms (lane M4LAB)
        elif cfg.cpp.flag("ALLOW_PTRACERS", _OPT) and trIdentity >= GAD_TR1:     # :673-679
            raise NotImplementedError("GAD_CALC_RHS: KPP_TRANSPORT_PTR is not ported")
        else:                                                                   # :680-684
            raise ValueError(f" tracer identity = {trIdentity} is not valid => STOP\n"
                             "ABNORMAL END: S/R GAD_CALC_RHS: invalid tracer identity")
        fVerT = fVerT.at[iA, jA, kUp].set(fVerT[iA, jA, kUp]                     # :685-689
                                          + df[iA, jA]*maskUp[iA, jA]*params.rhoFacF[k])
    if cfg.cpp.flag("GAD_SMOLARKIEWICZ_HACK", _OPT) and trUseSmolHack:          # :711-765
        raise NotImplementedError("GAD_CALC_RHS: the Smolarkiewicz hack is not ported")

    j = loop_j(1-OLy, sNy+OLy-1)                                                # :770-784
    i = loop_i(1-OLx, sNx+OLx-1)
    gTracer = gTracer.at[i, j, k].set(
        gTracer[i, j, k]
        - g.recip_hFacC[i, j, k]*g.recip_drF[k]
        * g.recip_rA[i, j]*g.recip_deepFac2C[k]*params.recip_rhoFacC[k]
        * ((fZon[i+1, j]-fZon[i, j])*g.maskInC[i, j]
           + (fMer[i, j+1]-fMer[i, j])*g.maskInC[i, j]
           + (fVerT[i, j, kDown]-fVerT[i, j, kUp])*g.rkSign
           - localT[i, j]*((uTrans[i+1, j]-uTrans[i, j])*advFac
                           + (vTrans[i, j+1]-vTrans[i, j])*advFac
                           + (rTransKp1[i, j]-rTrans[i, j])*rAdvFac
                           )*g.maskInC[i, j]
           ))
    return fZon, fMer, fVerT, gTracer
