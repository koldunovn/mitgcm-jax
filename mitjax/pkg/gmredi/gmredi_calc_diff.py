"""GMREDI_CALC_DIFF: pkg/gmredi/gmredi_calc_diff.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j

# pkg/generic_advdiff/GAD.h:123-124  INTEGER GAD_TR1; PARAMETER(GAD_TR1=3)
GAD_TR1 = 3


def gmredi_calc_diff(iMin, iMax, jMin, jMax, kArg, kSize, KappaRx, tracerIdentity, *, cfg, grid, gm):
    """GMREDI_CALC_DIFF( bi, bj, iMin, iMax, jMin, jMax, kArg, kSize, KappaRx, tracerIdentity, myThid )
    @63cdc0b pkg/gmredi/gmredi_calc_diff.F:3-91

    C     | SUBROUTINE GMREDI_CALC_DIFF                              |
    C     | o Add contribution to net diffusivity from GM/Redi       |
    C     iMin,iMax :: Range of points for which calculation is done
    C     jMin,jMax :: Range of points for which calculation is done
    C     kArg      :: = 0 -> do the k-loop here and treat all levels
    C                  > 0 -> k-loop is done outside and treat only level k=kArg
    C     kSize     :: 3rd Dimension of the vertical diffusivity array KappaRx
    C     KappaRx   :: vertical diffusivity array

    Returns KappaRx (an FArray declared (1-OLx:sNx+OLx, 1-OLy:sNy+OLy, kSize)). iMin..jMax, kArg, kSize,
    tracerIdentity: Python ints. Both IF arms of :54-64 / :73-83 are the same statement without ALLOW_LONGSTEP; with
    ALLOW_LONGSTEP (lane M4LAB: lab_sea compiles pkg/longstep, useLONGSTEP off) the passive-tracer arm
    (tracerIdentity >= GAD_TR1) reads LS_Kwz and raises; T and S (tracerIdentity < GAD_TR1) take the Kwz arm.
    kArg = 0 (all levels, the call from CALC_3D_DIFFUSIVITY) runs as a Python loop over k (each level reads and
    writes only level k); kArg > 0 writes level MIN(kArg,kSize) from Kwz(kArg).
    """
    sNx, Nr = cfg.size.sNx, cfg.size.Nr
    if cfg.cpp.ALLOW_LONGSTEP and tracerIdentity >= GAD_TR1:                 # :58-59, :77-78 (LS_Kwz)
        raise NotImplementedError("GMREDI_CALC_DIFF: ALLOW_LONGSTEP with a passive tracer (LS_Kwz) is not ported")
    maskInC, Kwz = grid.maskInC, gm.Kwz
    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    if kArg == 0:                                                   # :49-67  do all levels
        for k in range(1, min(Nr, kSize) + 1):  # MINMAX-INT: integer (no tie or NaN case)
            if tracerIdentity < GAD_TR1:                            # :54-56
                KappaRx = KappaRx.at[i, j, k].set(KappaRx[i, j, k]+Kwz[i, j, k]
                                                  * maskInC[i, j])
            else:                                                   # :57-63 (#else of ALLOW_LONGSTEP)
                KappaRx = KappaRx.at[i, j, k].set(KappaRx[i, j, k]+Kwz[i, j, k]
                                                  * maskInC[i, j])
    else:                                                           # :68-85  do level k=kArg only
        k = min(kArg, kSize)  # MINMAX-INT: integer (no tie or NaN case)
        if tracerIdentity < GAD_TR1:                                # :73-75
            KappaRx = KappaRx.at[i, j, k].set(KappaRx[i, j, k]+Kwz[i, j, kArg]
                                              * maskInC[i, j])
        else:                                                       # :76-82 (#else of ALLOW_LONGSTEP)
            KappaRx = KappaRx.at[i, j, k].set(KappaRx[i, j, k]+Kwz[i, j, kArg]
                                              * maskInC[i, j])
    return KappaRx
