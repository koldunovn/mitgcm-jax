"""GGL90_CALC_DIFF: pkg/ggl90/ggl90_calc_diff.F @63cdc0b."""

from mitjax.farray import loops_kji


def ggl90_calc_diff(iMin, iMax, jMin, jMax, kArg, kSize, KappaRx, *, cfg, params, ggl):
    """GGL90_CALC_DIFF( bi, bj, iMin, iMax, jMin, jMax, kArg, kSize, KappaRx, myThid )
    @63cdc0b pkg/ggl90/ggl90_calc_diff.F:7-74

    C     | SUBROUTINE GGL90_CALC_DIFF                               |
    C     | o Add contribution to net diffusivity from GGL90 mixing  |
    C     iMin,iMax :: Range of points for which calculation is done
    C     jMin,jMax :: Range of points for which calculation is done
    C     kArg      :: = 0 -> do the k-loop here and treat all levels
    C                  > 0 -> k-loop is done outside and treat only level k=kArg
    C     kSize     :: 3rd Dimension of the vertical diffusivity array KappaRx
    C     KappaRx   :: vertical diffusivity array

    KappaRx: FArray (1-OLx:sNx+OLx,1-OLy:sNy+OLy,kSize); kArg, kSize Python ints. Returns KappaRx. The levels of
    the kArg = 0 loop (:50-58) are independent (vectorised)."""
    Nr = cfg.size.Nr
    if kArg == 0:                                                       # :48-58
        k, j, i = loops_kji((1, min(Nr, kSize)), (jMin, jMax), (iMin, iMax))   # MINMAX-INT: MIN(Nr,kSize)
        return KappaRx.at[i, j, k].set(KappaRx[i, j, k]                 # :53-55
                                       + (ggl.GGL90diffKr[i, j, k]
                                          - params.diffKrNrS[k]))
    k = min(kArg, kSize)                                                # :61  MINMAX-INT: MIN(kArg,kSize)
    kk, j, i = loops_kji((k, k), (jMin, jMax), (iMin, iMax))
    return KappaRx.at[i, j, kk].set(KappaRx[i, j, kk]                   # :64-66
                                    + (ggl.GGL90diffKr[i, j, kArg]
                                       - params.diffKrNrS[kArg]))
