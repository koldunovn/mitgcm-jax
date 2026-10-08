"""MY82_CALC_VISC: pkg/my82/my82_calc_visc.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX


def my82_calc_visc(iMin, iMax, jMin, jMax, k, KappaRU, KappaRV, *, grid, params, my):
    """MY82_CALC_VISC( bi, bj, iMin, iMax, jMin, jMax, k, KappaRU, KappaRV, myThid )
    @63cdc0b pkg/my82/my82_calc_visc.F:3-56

    C     | SUBROUTINE MY82_CALC_VISC
    C     | o Add contrubution to net viscosity from MY82 mixing
    C     iMin, iMax, jMin, jMax :: Range of points for which calculation

    k a Python int; KappaRU, KappaRV: FArrays (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr) (the caller's (Nr+1) arrays by
    sequence association: level k of either is the same storage); returned with level k updated on iMin:iMax,
    jMin:jMax (:37-43, :45-51). `_maskW`/`_maskS` are maskW/maskS (no NONLIN_FRSURF macro change in these builds).
    The literal `0.5 _d 0` is the double 0.5. The (i, j) loops read only inputs: vectorised.
    """
    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    KappaRU = KappaRU.at[i, j, k].set(MAX(KappaRU[i, j, k],                                   # :39-41
                                          KappaRU[i, j, k] - params.viscArNr[k] + grid.maskW[i, j, k] *
                                          0.5*(my.MYviscAr[i, j, k]+my.MYviscAr[i-1, j, k]), p="a"))
    KappaRV = KappaRV.at[i, j, k].set(MAX(KappaRV[i, j, k],                                   # :47-49
                                          KappaRV[i, j, k] - params.viscArNr[k] + grid.maskS[i, j, k] *
                                          0.5*(my.MYviscAr[i, j, k]+my.MYviscAr[i, j-1, k]), p="a"))
    return KappaRU, KappaRV
