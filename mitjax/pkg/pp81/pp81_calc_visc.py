"""PP81_CALC_VISC: pkg/pp81/pp81_calc_visc.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX


def pp81_calc_visc(iMin, iMax, jMin, jMax, k, KappaRU, KappaRV, *, grid, params, pp):
    """PP81_CALC_VISC( bi, bj, iMin, iMax, jMin, jMax, k, KappaRU, KappaRV, myThid )
    @63cdc0b pkg/pp81/pp81_calc_visc.F:3-59

    C     | SUBROUTINE PP81_CALC_VISC                                |
    C     | o Add contrubution to net viscosity from PP81 mixing     |
    C     iMin, iMax, jMin, jMax :: Range of points for which calculation

    k a Python int; KappaRU, KappaRV: FArrays (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr) (the caller's (Nr+1) arrays by
    sequence association: level k of either is the same storage); returned with level k updated on iMin:iMax,
    jMin:jMax (:40-46, :48-54). `_maskW`/`_maskS` are maskW/maskS (no NONLIN_FRSURF macro change in these builds).
    The literal `0.5` is REAL*4 (exact). The (i, j) loops read only inputs: vectorised.
    """
    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    KappaRU = KappaRU.at[i, j, k].set(MAX(KappaRU[i, j, k],                                   # :42-44
                                          KappaRU[i, j, k] - params.viscArNr[k] + grid.maskW[i, j, k] *
                                          0.5*(pp.PPviscAr[i, j, k]+pp.PPviscAr[i-1, j, k]), p="a"))
    KappaRV = KappaRV.at[i, j, k].set(MAX(KappaRV[i, j, k],                                   # :50-52
                                          KappaRV[i, j, k] - params.viscArNr[k] + grid.maskS[i, j, k] *
                                          0.5*(pp.PPviscAr[i, j, k]+pp.PPviscAr[i, j-1, k]), p="a"))
    return KappaRU, KappaRV
