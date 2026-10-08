"""MY82_CALC_DIFF: pkg/my82/my82_calc_diff.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j


def my82_calc_diff(iMin, iMax, jMin, jMax, kArg, kSize, KappaRx, *, cfg, params, my):
    """MY82_CALC_DIFF( bi, bj, iMin, iMax, jMin, jMax, kArg, kSize, KappaRx, myThid )
    @63cdc0b pkg/my82/my82_calc_diff.F:3-79

    C     | SUBROUTINE MY82_CALC_DIFF                                |
    C     | o Add contrubution to net diffusivity from MY82 mixing    |
    C     iMin,iMax :: Range of points for which calculation is done
    C     jMin,jMax :: Range of points for which calculation is done
    C     kArg      :: = 0 -> do the k-loop here and treat all levels
    C                  > 0 -> k-loop is done outside and treat only level k=kArg
    C     kSize     :: 3rd Dimension of the vertical diffusivity array KappaRx
    C     KappaRx   :: vertical diffusivity array

    KappaRx: FArray (1-OLx:sNx+OLx,1-OLy:sNy+OLy,kSize) (kSize, kArg static ints); returned with the MY82 increment
    added on iMin:iMax, jMin:jMax. kArg = 0: every level 1..MIN(Nr,kSize) (:45-59; the levels are independent, one
    statement per level in the Fortran order); kArg > 0: level MIN(kArg,kSize) (:60-74). #else of ALLOW_3D_DIFFKR
    (diffKrNrS); ALLOW_3D_DIFFKR raises.
    """
    sz = cfg.size
    if cfg.cpp.flag("ALLOW_3D_DIFFKR", "MY82_OPTIONS.h"):
        raise NotImplementedError("MY82_CALC_DIFF: ALLOW_3D_DIFFKR is not ported")
    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    if kArg == 0:                                                       # :45
        for k in range(1, min(sz.Nr, kSize)+1):                         # :47  MINMAX-INT: integer loop bound
            KappaRx = KappaRx.at[i, j, k].set(KappaRx[i, j, k]          # :50-56
                                              + (my.MYdiffKr[i, j, k]
                                                 - params.diffKrNrS[k]))
    else:
        k = min(kArg, kSize)                                            # :62  MINMAX-INT: integer level index
        KappaRx = KappaRx.at[i, j, k].set(KappaRx[i, j, k]              # :65-71
                                          + (my.MYdiffKr[i, j, kArg]
                                             - params.diffKrNrS[kArg]))
    return KappaRx
