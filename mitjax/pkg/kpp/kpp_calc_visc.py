"""KPP_CALC_VISC: pkg/kpp/kpp_calc_visc.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX


def kpp_calc_visc(iMin, iMax, jMin, jMax, K, KappaRU, KappaRV, *, cfg, grid, params, kppf):
    """KPP_CALC_VISC( bi, bj, iMin, iMax, jMin, jMax, K, KappaRU, KappaRV, myThid )
    @63cdc0b pkg/kpp/kpp_calc_visc.F:3-52

    C     | SUBROUTINE KPP_CALC_VISC
    C     | o Add contrubution to net viscosity from KPP mixing

    KappaRU, KappaRV: (…,Nr) FArrays; returns (KappaRU, KappaRV), level K updated over iMin..iMax, jMin..jMax
    (:35-49; `_maskW`/`_maskS` are GRID.h maskW/maskS; `0.5` REAL*4, exact). K a static int."""
    k = K
    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    vz = kppf["KPPviscAz"]
    KappaRU = KappaRU.at[i, j, k].set(MAX(KappaRU[i, j, k],                    # :37-39
                                          KappaRU[i, j, k] - params.viscArNr[1] + grid.maskW[i, j, k] *
                                          0.5*(vz[i, j, k]+vz[i-1, j, k]), p="a"))
    KappaRV = KappaRV.at[i, j, k].set(MAX(KappaRV[i, j, k],                    # :45-47
                                          KappaRV[i, j, k] - params.viscArNr[1] + grid.maskS[i, j, k] *
                                          0.5*(vz[i, j, k]+vz[i, j-1, k]), p="a"))
    return KappaRU, KappaRV
