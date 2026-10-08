"""GGL90_CALC_VISC: pkg/ggl90/ggl90_calc_visc.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j


def ggl90_calc_visc(iMin, iMax, jMin, jMax, k, KappaRU, KappaRV, *, params, ggl):
    """GGL90_CALC_VISC( bi, bj, iMin, iMax, jMin, jMax, k, KappaRU, KappaRV, myThid )
    @63cdc0b pkg/ggl90/ggl90_calc_visc.F:7-64

    C     | SUBROUTINE GGL90_CALC_VISC                               |
    C     | o Add contribution to net viscosity from GGL90 mixing    |
    C     iMin,iMax :: Range of points for which calculation is done
    C     jMin,jMax :: Range of points for which calculation is done
    C     k         :: current level index
    C     KappaRU   :: vertical viscosity array for U-component
    C     KappaRV   :: vertical viscosity array for V-component

    KappaRU, KappaRV: FArrays (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr) as declared here (the caller's arrays have Nr+1
    levels; the routine touches level k <= Nr only: pass the caller's array, its declared k range covers k).
    k: a Python int. Returns (KappaRU, KappaRV)."""
    j, i = loop_j(jMin, jMax), loop_i(iMin, iMax)
    KappaRU = KappaRU.at[i, j, k].set(KappaRU[i, j, k]                  # :45-46
                                      + (ggl.GGL90viscArU[i, j, k] - params.viscArNr[k]))
    KappaRV = KappaRV.at[i, j, k].set(KappaRV[i, j, k]                  # :52-53
                                      + (ggl.GGL90viscArV[i, j, k] - params.viscArNr[k]))
    return KappaRU, KappaRV
