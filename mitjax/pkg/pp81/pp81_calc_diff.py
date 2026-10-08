"""PP81_CALC_DIFF: pkg/pp81/pp81_calc_diff.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j


def pp81_calc_diff(iMin, iMax, jMin, jMax, kArg, kSize, KappaRx, *, cfg, params, pp):
    """PP81_CALC_DIFF( bi, bj, iMin, iMax, jMin, jMax, kArg, kSize, KappaRx, myThid )
    @63cdc0b pkg/pp81/pp81_calc_diff.F:3-81

    C     | SUBROUTINE PP81_CALC_DIFF                                |
    C     | o Add contrubution to net diffusivity from PP81 mixing   |
    C     iMin,iMax :: Range of points for which calculation is done
    C     jMin,jMax :: Range of points for which calculation is done
    C     kArg      :: = 0 -> do the k-loop here and treat all levels
    C                  > 0 -> k-loop is done outside and treat only level k=kArg
    C     kSize     :: 3rd Dimension of the vertical diffusivity array KappaRx
    C     KappaRx   :: vertical diffusivity array

    KappaRx: FArray (1-OLx:sNx+OLx,1-OLy:sNy+OLy,kSize) (kSize, kArg static ints); returned with the PP81 increment
    added on iMin:iMax, jMin:jMax. kArg = 0: every level 1..MIN(Nr,kSize) (:47-61; the levels are independent, one
    statement per level in the Fortran order); kArg > 0: level MIN(kArg,kSize) (:62-76). #else of ALLOW_3D_DIFFKR
    (diffKrNrS); ALLOW_3D_DIFFKR raises.
    """
    sz = cfg.size
    if cfg.cpp.flag("ALLOW_3D_DIFFKR", "PP81_OPTIONS.h"):
        raise NotImplementedError("PP81_CALC_DIFF: ALLOW_3D_DIFFKR is not ported")
    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    if kArg == 0:                                                       # :47
        for k in range(1, min(sz.Nr, kSize)+1):                         # :49  MINMAX-INT: integer loop bound
            KappaRx = KappaRx.at[i, j, k].set(KappaRx[i, j, k]          # :52-58
                                              + (pp.PPdiffKr[i, j, k]
                                                 - params.diffKrNrS[k]))
    else:
        k = min(kArg, kSize)                                            # :64  MINMAX-INT: integer level index
        KappaRx = KappaRx.at[i, j, k].set(KappaRx[i, j, k]              # :67-73
                                          + (pp.PPdiffKr[i, j, kArg]
                                             - params.diffKrNrS[kArg]))
    return KappaRx
