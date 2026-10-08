"""GMREDI_XTRANSPORT: pkg/gmredi/gmredi_xtransport.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j
from mitjax.pkg.gmredi.gmredi_calc_diff import GAD_TR1
from mitjax.pkg.gmredi.gmredi_h import op5

# tamc.h PARAMETER( maxpass = 2 ): pkg/autodiff/tamc.h:88 and its override
# verification/tutorial_global_oce_optim/code_ad/tamc.h:78 (the only M1 build with ALLOW_AUTODIFF_TAMC)
maxpass = 2


def maxpass_of(cfg):
    """PTRACERS lane: maxpass of the build. tamc.h declares it only #ifndef ALLOW_PTRACERS (tutorial_tracer_adjsens/
    code_ad/tamc.h:80-84; pkg/autodiff/tamc.h:85-89): with ALLOW_PTRACERS it is PTRACERS_SIZE.h:19-20
    PARAMETER( maxpass = PTRACERS_num + 2 ), else 2."""
    if cfg.cpp.ALLOW_PTRACERS:
        from mitjax.pkg.ptracers.ptracers_readparms import _num
        return _num(cfg) + 2
    return maxpass


def gmredi_xtransport(trIdentity, k, iMin, iMax, jMin, jMax, xA, maskFk, Tracer, df, *, cfg, grid, gm):
    """GMREDI_XTRANSPORT( trIdentity, bi, bj, k, iMin, iMax, jMin, jMax, xA, maskFk, Tracer, df, myThid )
    @63cdc0b pkg/gmredi/gmredi_xtransport.F:9-265

    C     | o SUBROUTINE GMREDI_XTRANSPORT
    C     |   Add horizontal x transport terms from GM/Redi
    C     |   parameterization.
    C     trIdentity :: tracer Id number
    C     k          :: current level index
    C     iMin,iMax  :: Range of 1rst index where results will be set
    C     jMin,jMax  :: Range of 2nd  index where results will be set
    C     xA         :: Area of X face
    C     maskFk     :: 2-D mask for vertical interface k (between level k-1 & k)
    C     Tracer     :: 3D Tracer field
    C     df         :: Diffusive flux component work array.

    Returns df. trIdentity, k, iMin..jMax: Python ints. Ported: the diagonal flux :129-137 (IF ( useGMRedi ), :104).
    GOADK lane (M2, global_ocean.90x40x15/code_ad): GM_ExtraDiag = .TRUE. (the Kuz term with the vertical
    gradient dTdz, :142-191; reads GRID.h maskC, recip_drC; the unary + of :153 is no operation). Not ported
    (raise): the bolus advective flux (GM_AdvForm with
    GM_AdvSeparate, :195-236), ALLOW_LONGSTEP for a passive tracer
    (:116-140). The AD-only key computation of ALLOW_AUTODIFF_TAMC (:92-94)
    has no forward effect; its trIdentity check (:95-101) is the Fortran STOP. The GM_ubT diagnostics block
    (:241-256) is output only (not ported). The point loop runs on the whole (i,j) range at once.
    """
    if cfg.cpp.ALLOW_LONGSTEP and trIdentity >= GAD_TR1:                  # IF ( trIdentity .GE. GAD_TR1 ): LS_Kux
        raise NotImplementedError("GMREDI_XTRANSPORT: ALLOW_LONGSTEP with a passive tracer (LS_Kux) is not ported")
    if cfg.cpp.ALLOW_AUTODIFF_TAMC and trIdentity > maxpass_of(cfg):    # :95-101
        raise RuntimeError(f"GMREDI_XTRANSPORT: trIdentity > maxpass {trIdentity} {maxpass_of(cfg)}: "
                           "ABNORMAL END: S/R GMREDI_XTRANSPORT")
    recip_dxC, recip_deepFacC = grid.recip_dxC, grid.recip_deepFacC
    Kux = gm.Kux

    if cfg.use_flag("useGMRedi"):                                   # :104
        j = loop_j(jMin, jMax)                                      # :115-137  Area integrated zonal flux
        i = loop_i(iMin, iMax)
        df = df.at[i, j].set(df[i, j]
                             - xA[i, j]
                             * Kux[i, j, k]
                             * recip_dxC[i, j]*recip_deepFacC[k]
                             * (Tracer[i, j, k] - Tracer[i-1, j, k]))

        if cfg.cpp.GM_EXTRA_DIAGONAL and gm.GM_ExtraDiag:           # :142-191 (GOADK lane)
            Nr = cfg.size.Nr
            km1 = max(k-1, 1)  # MINMAX-INT: integer level index (no tie or NaN case)
            kp1 = min(k+1, Nr)  # MINMAX-INT: integer level index (no tie or NaN case)
            maskp1 = 1.                                             # :150  1. _d 0
            if k >= Nr:
                maskp1 = 0.                                         # :151  0. _d 0
            maskC, recip_drC = grid.maskC, grid.recip_drC
            dTdz = op5*(                                            # :152-166  vertical gradients at U points
                + op5*recip_drC[k]
                * (maskFk[i-1, j]
                   * (Tracer[i-1, j, km1]-Tracer[i-1, j, k])
                   + maskFk[i, j]
                   * (Tracer[i, j, km1]-Tracer[i, j, k])
                   )
                + op5*recip_drC[kp1]
                * (maskC[i-1, j, k]*maskC[i-1, j, kp1]*maskp1
                   * (Tracer[i-1, j, k]-Tracer[i-1, j, kp1])
                   + maskC[i, j, k]*maskC[i, j, kp1]*maskp1
                   * (Tracer[i, j, k]-Tracer[i, j, kp1])
                   ))
            df = df.at[i, j].set(df[i, j] - xA[i, j]*gm.Kuz[i, j, k]*dTdz)   # :182-186
        if cfg.cpp.GM_BOLUS_ADVEC and (gm.GM_AdvForm and gm.GM_AdvSeparate
                                       and not gm.GM_InMomAsStress):  # :194-236
            raise NotImplementedError("GMREDI_XTRANSPORT: the bolus advective flux (GM_AdvForm) is not ported")
    return df
