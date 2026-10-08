"""GMREDI_YTRANSPORT: pkg/gmredi/gmredi_ytransport.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j
from mitjax.pkg.gmredi.gmredi_calc_diff import GAD_TR1
from mitjax.pkg.gmredi.gmredi_h import op5

# tamc.h PARAMETER( maxpass = 2 ): pkg/autodiff/tamc.h:88 and its override
# verification/tutorial_global_oce_optim/code_ad/tamc.h:78 (the only M1 build with ALLOW_AUTODIFF_TAMC)
maxpass = 2
from mitjax.pkg.gmredi.gmredi_xtransport import maxpass_of  # noqa: E402  (PTRACERS lane)


def gmredi_ytransport(trIdentity, k, iMin, iMax, jMin, jMax, yA, maskFk, Tracer, df, *, cfg, grid, gm):
    """GMREDI_YTRANSPORT( trIdentity, bi, bj, k, iMin, iMax, jMin, jMax, yA, maskFk, Tracer, df, myThid )
    @63cdc0b pkg/gmredi/gmredi_ytransport.F:9-265

    C     | o SUBROUTINE GMREDI_YTRANSPORT
    C     |   Add horizontal y transport terms from GM/Redi
    C     |   parameterization.
    C     trIdentity :: tracer Id number
    C     k          :: current level index
    C     iMin,iMax  :: Range of 1rst index where results will be set
    C     jMin,jMax  :: Range of 2nd  index where results will be set
    C     yA         :: Area of Y face
    C     maskFk     :: 2-D mask for vertical interface k (between level k-1 & k)
    C     Tracer     :: 3D Tracer field
    C     df         :: Diffusive flux component work array.

    Returns df. trIdentity, k, iMin..jMax: Python ints. Ported: the diagonal flux :129-137 (IF ( useGMRedi ), :104).
    GOADK lane (M2, global_ocean.90x40x15/code_ad): GM_ExtraDiag = .TRUE. (the Kvz term with the vertical
    gradient dTdz, :142-191; reads GRID.h maskC, recip_drC; the unary + of :153 is no operation). Not ported
    (raise): the bolus advective flux (GM_AdvForm with
    GM_AdvSeparate, :195-236), ALLOW_LONGSTEP for a passive tracer
    (:116-140). The AD-only key computation of ALLOW_AUTODIFF_TAMC (:92-94)
    has no forward effect; its trIdentity check (:95-101) is the Fortran STOP. The GM_vbT diagnostics block
    (:241-256) is output only (not ported). The point loop runs on the whole (i,j) range at once.
    """
    if cfg.cpp.ALLOW_LONGSTEP and trIdentity >= GAD_TR1:                  # IF ( trIdentity .GE. GAD_TR1 ): LS_Kvy
        raise NotImplementedError("GMREDI_YTRANSPORT: ALLOW_LONGSTEP with a passive tracer (LS_Kvy) is not ported")
    if cfg.cpp.ALLOW_AUTODIFF_TAMC and trIdentity > maxpass_of(cfg):    # :95-101 (maxpass_of: PTRACERS lane)
        raise RuntimeError(f"GMREDI_YTRANSPORT: trIdentity > maxpass {trIdentity} {maxpass_of(cfg)}: "
                           "ABNORMAL END: S/R GMREDI_YTRANSPORT")
    recip_dyC, recip_deepFacC = grid.recip_dyC, grid.recip_deepFacC
    Kvy = gm.Kvy

    if cfg.use_flag("useGMRedi"):                                   # :104
        j = loop_j(jMin, jMax)                                      # :115-137  Area integrated meridional flux
        i = loop_i(iMin, iMax)
        df = df.at[i, j].set(df[i, j]
                             - yA[i, j]
                             * Kvy[i, j, k]
                             * recip_dyC[i, j]*recip_deepFacC[k]
                             * (Tracer[i, j, k] - Tracer[i, j-1, k]))

        if cfg.cpp.GM_EXTRA_DIAGONAL and gm.GM_ExtraDiag:           # :142-191 (GOADK lane)
            Nr = cfg.size.Nr
            km1 = max(k-1, 1)  # MINMAX-INT: integer level index (no tie or NaN case)
            kp1 = min(k+1, Nr)  # MINMAX-INT: integer level index (no tie or NaN case)
            maskp1 = 1.                                             # :150  1. _d 0
            if k >= Nr:
                maskp1 = 0.                                         # :151  0. _d 0
            maskC, recip_drC = grid.maskC, grid.recip_drC
            dTdz = op5*(                                            # :152-166  vertical gradients at V points
                + op5*recip_drC[k]
                * (maskFk[i, j-1]
                   * (Tracer[i, j-1, km1]-Tracer[i, j-1, k])
                   + maskFk[i, j]
                   * (Tracer[i, j, km1]-Tracer[i, j, k])
                   )
                + op5*recip_drC[kp1]
                * (maskC[i, j-1, k]*maskC[i, j-1, kp1]*maskp1
                   * (Tracer[i, j-1, k]-Tracer[i, j-1, kp1])
                   + maskC[i, j, k]*maskC[i, j, kp1]*maskp1
                   * (Tracer[i, j, k]-Tracer[i, j, kp1])
                   ))
            df = df.at[i, j].set(df[i, j] - yA[i, j]*gm.Kvz[i, j, k]*dTdz)   # :182-186
        if cfg.cpp.GM_BOLUS_ADVEC and (gm.GM_AdvForm and gm.GM_AdvSeparate
                                       and not gm.GM_InMomAsStress):  # :194-236
            raise NotImplementedError("GMREDI_YTRANSPORT: the bolus advective flux (GM_AdvForm) is not ported")
    return df
