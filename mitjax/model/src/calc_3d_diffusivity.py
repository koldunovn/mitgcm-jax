"""CALC_3D_DIFFUSIVITY: model/src/calc_3d_diffusivity.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loops_kji
from mitjax.model.grid import PI
from mitjax.model.src.ini_parms import _use
from mitjax.pkg.generic_advdiff.gad_h import GAD_SALINITY, GAD_TEMPERATURE, GAD_TR1


def calc_3d_diffusivity(iMin, iMax, jMin, jMax, trIdentity, trUseGMRedi, trUseKPP, KappaRTr, *, cfg, grid, params,
                        state, gm=None, ptr=None, mix=None):
    """CALC_3D_DIFFUSIVITY( bi, bj, iMin,iMax,jMin,jMax, trIdentity, trUseGMRedi, trUseKPP, KappaRTr, myThid )
    @63cdc0b model/src/calc_3d_diffusivity.F:10-428

    C     | SUBROUTINE CALC_3D_DIFFUSIVITY
    C     | o Calculate net (3D) vertical diffusivity for 1 tracer
    C     | Combines spatially varying diffusion coefficients from
    C     | KPP and/or GM and/or convective stability test.
    C     trIdentity :: tracer identifier
    C     trUseGMRedi:: this tracer use GM-Redi
    C     trUseKPP   :: this tracer use KPP
    C     KappaRTr   :: Net diffusivity for this tracer (trIdentity)

    Returns KappaRTr (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr). trIdentity, trUseGMRedi, trUseKPP static. Ported: the
    background + convective diffusivity without KPP (:83-118: IVDConvCount*ivdc_kappa + KbryanLewis79, DYNVARS.h
    IVDConvCount from `state`), the tracer's diffKrNr (:119-144, temperature and salinity). The k loops are
    independent across k (each level reads only inputs): k-vectorised nests; KbryanLewis79 is computed per level as
    the Fortran does (:85-86, `atan` is XLA's arctangent: bitwise agreement with glibc's atan is not measured; in the
    M1 variants diffKrBL79surf = diffKrBL79deep = 0, so the product with (deep-surf) = 0 gives +0 whatever the
    arctangent's last bit). `0.5 _d 0` exact; PI from PARAMS.h:16. Raise: ALLOW_LONGSTEP with a passive tracer,
    ALLOW_BL79_LAT_VARY, the package contributions (KPP, GM/Redi, PP81, KL10, MY82, GGL90, Smag-3D: plan Tasks 15b,
    M3), the partial-cell hacks interDiffKr_pCell and pCellMix_select (:261-414).

    PTRACERS lane (M2, marked arms): the passive-tracer branch trIdentity >= GAD_TR1 (:145-161, `ptr`:
    PTRACERS_PARAMS.h PTRACERS_diffKrNr(k,iTr)) and the ALLOW_3D_DIFFKR arms (:124-125, :137-138, :153-154: DYNVARS.h
    diffKr from `state`, tutorial_tracer_adjsens/code_ad); an invalid tracer Id STOPs (:162-167)."""
    for opt in ("ALLOW_BL79_LAT_VARY",):
        if getattr(cfg.cpp, opt):
            raise NotImplementedError(f"CALC_3D_DIFFUSIVITY: {opt} is not ported")
    diffKr3d = bool(cfg.cpp.ALLOW_3D_DIFFKR)                                    # PTRACERS lane: ALLOW_3D_DIFFKR arms
    sz = cfg.size
    g = grid
    if not trUseKPP:                                                            # :83-168
        k, j, i = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))   # :84, :107-108
        KbryanLewis79 = (params.diffKrBL79surf + (params.diffKrBL79deep-params.diffKrBL79surf)  # :85-86
                         * (jnp.arctan(-(g.rF[k]-params.diffKrBL79Ho)/params.diffKrBL79scl)/PI+0.5))
        if cfg.cpp.ALLOW_LONGSTEP and trIdentity >= 3:                          # :91-103
            raise NotImplementedError("CALC_3D_DIFFUSIVITY: ALLOW_LONGSTEP (passive tracers) is not ported")
        KappaRTr = KappaRTr.at[i, j, k].set(                                    # :109-111
            state.IVDConvCount[i, j, k]*params.ivdc_kappa
            + KbryanLewis79)
        if diffKr3d and trIdentity in (GAD_TEMPERATURE, GAD_SALINITY):         # PTRACERS lane (tracer_adjsens):
            KappaRTr = KappaRTr.at[i, j, k].set(KappaRTr[i, j, k]               # :124-125 / :137-138
                                                + state.diffKr[i, j, k])        # #ifdef ALLOW_3D_DIFFKR: diffKr
        elif trIdentity == GAD_TEMPERATURE:                                     # :119-131
            KappaRTr = KappaRTr.at[i, j, k].set(KappaRTr[i, j, k]
                                                + params.diffKrNrT[k])
        elif trIdentity == GAD_SALINITY:                                        # :132-144
            KappaRTr = KappaRTr.at[i, j, k].set(KappaRTr[i, j, k]
                                                + params.diffKrNrS[k])
        elif cfg.cpp.ALLOW_PTRACERS and trIdentity >= GAD_TR1:                  # :145-161 (PTRACERS lane)
            iTr = trIdentity - GAD_TR1 + 1                                      # :148
            KappaRTr = KappaRTr.at[i, j, k].set(KappaRTr[i, j, k]               # :150-158
                                                + (state.diffKr[i, j, k] if diffKr3d
                                                   else ptr.PTRACERS_diffKrNr[iTr-1][k]))
        else:                                                                   # :162-167
            raise ValueError(f" CALC_3D_DIFFUSIVITY: Invalid tracer Id: {trIdentity:4d}\n"
                             "ABNORMAL END: S/R CALC_3D_DIFFUSIVITY")

    if cfg.cpp.ALLOW_KPP and trUseKPP:                                          # :172-200 (vermix lane, M3)
        # `mix`: the mixing packages' state {"kppf": KPP.h fields, "ggl": GGL90.h, "pp81", "my82"}; KPP_CALC_DIFF_T/S with kArg = 0, kSize = Nr (:175-184)
        if mix is None:
            raise ValueError("CALC_3D_DIFFUSIVITY: trUseKPP needs the KPP.h fields (`mix`)")
        if trIdentity == GAD_TEMPERATURE:                                       # :174-178
            from mitjax.pkg.kpp.kpp_calc_diff_t import kpp_calc_diff_t
            KappaRTr = kpp_calc_diff_t(iMin, iMax, jMin, jMax, 0, sz.Nr, KappaRTr, cfg=cfg, kppf=mix["kppf"])
        elif trIdentity == GAD_SALINITY:                                        # :179-183
            from mitjax.pkg.kpp.kpp_calc_diff_s import kpp_calc_diff_s
            KappaRTr = kpp_calc_diff_s(iMin, iMax, jMin, jMax, 0, sz.Nr, KappaRTr, cfg=cfg, kppf=mix["kppf"])
        elif cfg.cpp.ALLOW_PTRACERS and trIdentity >= GAD_TR1:                  # :185-190
            raise NotImplementedError("CALC_3D_DIFFUSIVITY: KPP_CALC_DIFF_PTR is not ported")
        else:                                                                   # :192-197
            raise ValueError(f" CALC_3D_DIFFUSIVITY: Invalid tracer Id: {trIdentity:4d}\n"
                             "ABNORMAL END: S/R CALC_3D_DIFFUSIVITY")
    if cfg.cpp.ALLOW_GMREDI and trUseGMRedi:                                    # :202-209 (R5 arm)
        from mitjax.pkg.gmredi.gmredi_calc_diff import gmredi_calc_diff
        KappaRTr = gmredi_calc_diff(iMin, iMax, jMin, jMax, 0, cfg.size.Nr, KappaRTr, trIdentity,
                                    cfg=cfg, grid=grid, gm=gm)
    # :211-245 (vermix lane): PP81_CALC_DIFF, MY82_CALC_DIFF, GGL90_CALC_DIFF with kArg = 0, kSize = Nr (`mix`)
    if cfg.cpp.ALLOW_PP81 and _use(cfg, "usePP81"):                             # :211-218
        from mitjax.pkg.pp81.pp81_calc_diff import pp81_calc_diff
        KappaRTr = pp81_calc_diff(iMin, iMax, jMin, jMax, 0, sz.Nr, KappaRTr, cfg=cfg, params=params, pp=mix["pp81"])
    if cfg.cpp.ALLOW_KL10 and _use(cfg, "useKL10"):                             # :220-227
        raise NotImplementedError("CALC_3D_DIFFUSIVITY: useKL10 is not ported")
    if cfg.cpp.ALLOW_MY82 and _use(cfg, "useMY82"):                             # :229-236
        from mitjax.pkg.my82.my82_calc_diff import my82_calc_diff
        KappaRTr = my82_calc_diff(iMin, iMax, jMin, jMax, 0, sz.Nr, KappaRTr, cfg=cfg, params=params, my=mix["my82"])
    if cfg.cpp.ALLOW_GGL90 and _use(cfg, "useGGL90"):                           # :238-245
        from mitjax.pkg.ggl90.ggl90_calc_diff import ggl90_calc_diff
        KappaRTr = ggl90_calc_diff(iMin, iMax, jMin, jMax, 0, sz.Nr, KappaRTr, cfg=cfg, params=params,
                                   ggl=mix["ggl"])
    if cfg.cpp.ALLOW_SMAG_3D_DIFFUSIVITY:                                       # :247-259
        raise NotImplementedError("CALC_3D_DIFFUSIVITY: ALLOW_SMAG_3D_DIFFUSIVITY is not ported")
    if not cfg.cpp.EXCLUDE_PCELL_MIX_CODE:                                      # :261-414
        if params.interDiffKr_pCell or params.pCellMix_select > 0:
            raise NotImplementedError("CALC_3D_DIFFUSIVITY: interDiffKr_pCell / pCellMix_select is not ported")
    return KappaRTr
