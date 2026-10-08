"""GMREDI_RTRANSPORT: pkg/gmredi/gmredi_rtransport.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j
from mitjax.pkg.gmredi.gmredi_calc_diff import GAD_TR1
from mitjax.pkg.gmredi.gmredi_h import op5
from mitjax.pkg.gmredi.gmredi_xtransport import maxpass_of


def gmredi_rtransport(trIdentity, k, iMin, iMax, jMin, jMax, maskUp, Tracer, df, *, cfg, grid, gm):
    """GMREDI_RTRANSPORT( trIdentity, bi, bj, k, iMin, iMax, jMin, jMax, maskUp, Tracer, df, myThid )
    @63cdc0b pkg/gmredi/gmredi_rtransport.F:8-216

    C     | o SUBROUTINE GMREDI_RTRANSPORT                           |
    C     |   Add vertical transport terms from GM/Redi              |
    C     |   parameterization.                                      |
    C     trIdentity   :: tracer Id number
    C     k            :: current level index
    C     iMin,iMax    :: Range of 1rst index where results will be set
    C     jMin,jMax    :: Range of 2nd  index where results will be set
    C     maskUp       :: 2-D array for mask at W points
    C     Tracer       :: 3D Tracer field
    C     df           :: Diffusive flux component work array.

    Returns df. trIdentity, k, iMin..jMax: Python ints. Ported: :91-153 (surface flux is zero: nothing for k = 1;
    IF ( useGMRedi .AND. k.GT.1 )). Not ported (raise): the bolus advective flux (GM_AdvForm with GM_AdvSeparate,
    :158-203), ALLOW_LONGSTEP for a passive tracer
    (:134-154). The ALLOW_AUTODIFF_TAMC key (:78-80) has no forward effect; its trIdentity
    check (:81-87) is the Fortran STOP. The vertical diffusion term (:206-210) is added through Kwz in
    GMREDI_CALC_DIFF. The point loops run on the whole (i,j) range at once.
    """
    if cfg.cpp.ALLOW_LONGSTEP and trIdentity >= GAD_TR1:                  # :134-135 (LS_Kwx, LS_Kwy)
        raise NotImplementedError("GMREDI_RTRANSPORT: ALLOW_LONGSTEP with a passive tracer (LS_Kwx, LS_Kwy) is "
                                  "not ported")
    if cfg.cpp.ALLOW_AUTODIFF_TAMC and trIdentity > maxpass_of(cfg):    # :81-87 (maxpass_of: PTRACERS lane)
        raise RuntimeError(f"GMREDI_RTRANSPORT: trIdentity > maxpass {trIdentity} {maxpass_of(cfg)}: "
                           "ABNORMAL END: S/R GMREDI_RTRANSPORT")
    maskW, maskS, recip_dxC, recip_dyC = grid.maskW, grid.maskS, grid.recip_dxC, grid.recip_dyC
    recip_deepFacC, deepFac2F, rA, maskInC = grid.recip_deepFacC, grid.deepFac2F, grid.rA, grid.maskInC
    Kwx, Kwy = gm.Kwx, gm.Kwy

    if cfg.use_flag("useGMRedi") and k > 1:                         # :90-91  Surface flux is zero
        j = loop_j(jMin, jMax)                                      # :93-124  Horizontal gradients interpolated
        i = loop_i(iMin, iMax)                                      #   to W points
        dTdx = op5*(
            op5*recip_deepFacC[k]
            * (maskW[i+1, j, k]*recip_dxC[i+1, j]
               * (Tracer[i+1, j, k] - Tracer[i, j, k])
               + maskW[i, j, k]*recip_dxC[i, j]
               * (Tracer[i, j, k] - Tracer[i-1, j, k])
               )
            + op5*recip_deepFacC[k-1]
            * (maskW[i+1, j, k-1]*recip_dxC[i+1, j]
               * (Tracer[i+1, j, k-1] - Tracer[i, j, k-1])
               + maskW[i, j, k-1]*recip_dxC[i, j]
               * (Tracer[i, j, k-1] - Tracer[i-1, j, k-1])
               ))

        dTdy = op5*(
            op5*recip_deepFacC[k]
            * (maskS[i, j+1, k]*recip_dyC[i, j+1]
               * (Tracer[i, j+1, k] - Tracer[i, j, k])
               + maskS[i, j, k]*recip_dyC[i, j]
               * (Tracer[i, j, k] - Tracer[i, j-1, k])
               )
            + op5*recip_deepFacC[k-1]
            * (maskS[i, j+1, k-1]*recip_dyC[i, j+1]
               * (Tracer[i, j+1, k-1] - Tracer[i, j, k-1])
               + maskS[i, j, k-1]*recip_dyC[i, j]
               * (Tracer[i, j, k-1] - Tracer[i, j-1, k-1])
               ))

        df = df.at[i, j].set(df[i, j]                               # :133-153  Off-diagonal components of
                             - rA[i, j]*deepFac2F[k]*maskInC[i, j]   #   vertical flux
                             * (Kwx[i, j, k]*dTdx
                                + Kwy[i, j, k]*dTdy)*maskUp[i, j])

        if cfg.cpp.GM_BOLUS_ADVEC and (gm.GM_AdvForm and gm.GM_AdvSeparate
                                       and not gm.GM_InMomAsStress):  # :158-203
            raise NotImplementedError("GMREDI_RTRANSPORT: the bolus advective flux (GM_AdvForm) is not ported")
    return df
