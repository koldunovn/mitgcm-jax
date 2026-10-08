"""CALC_GRAD_PHI_SURF: model/src/calc_grad_phi_surf.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j


def calc_grad_phi_surf(iMin, iMax, jMin, jMax, etaFld, phiSurfX, phiSurfY, *, cfg, grid):
    """CALC_GRAD_PHI_SURF( bi, bj, iMin, iMax, jMin, jMax, etaFld, phiSurfX, phiSurfY, myThid )
    @63cdc0b model/src/calc_grad_phi_surf.F:6-64

    C     | SUBROUTINE CALC_GRAD_PHI_SURF
    C     | o Calculate the gradient of the surface Potential anomaly
    C     etaFld   :: free-surface r-anomaly (r unit)
    C     phiSurfX :: gradient of Surface potential - X component
    C     phiSurfY :: gradient of Surface potential - Y component

    Returns (phiSurfX, phiSurfY); points outside iMin..iMax, jMin..jMax keep their values. Reads Bo_surf
    (SURFACE.h) and recip_dxC/recip_dyC (GRID.h). Points are independent."""
    Bo_surf, recip_dxC, recip_dyC = grid.Bo_surf, grid.recip_dxC, grid.recip_dyC
    j = loop_j(jMin, jMax)                                                      # :46
    i = loop_i(iMin, iMax)                                                      # :47
    phiSurfX = phiSurfX.at[i, j].set(recip_dxC[i, j]*(                         # :48-50
        Bo_surf[i, j]*etaFld[i, j]
        - Bo_surf[i-1, j]*etaFld[i-1, j]))
    phiSurfY = phiSurfY.at[i, j].set(recip_dyC[i, j]*(                         # :55-59
        Bo_surf[i, j]*etaFld[i, j]
        - Bo_surf[i, j-1]*etaFld[i, j-1]))
    return phiSurfX, phiSurfY
