"""pkg/mom_common/mom_calc_strain.F: strain of the horizontal flow at vorticity points (MOM_CALC_STRAIN)."""

from mitjax.farray import loop_i, loop_j


def mom_calc_strain(k, uFld, vFld, hFacZ, strain, *, cfg, grid):
    """MOM_CALC_STRAIN(bi,bj,k, uFld, vFld, hFacZ, strain, myThid)   @63cdc0b pkg/mom_common/mom_calc_strain.F:7-83

    C Calculates the strain of the horizontal flow field (at vorticity points):
    C D_S = \\frac{\\Delta y_u}{\\Delta x_v} \\delta_i \\frac{v}{\\Delta y_c}
    C     + \\frac{\\Delta x_v}{\\Delta y_u} \\delta_j \\frac{u}{\\Delta x_c}
    C assuming free-slip boundaries.
    C  k                    :: vertical level
    C  uFld                 :: zonal flow
    C  vFld                 :: meridional flow
    C  hFacZ                :: open-water thickness at vorticity points
    C  strain               :: strain of horizontal flow

    Returns strain (points outside DO j=2-OLy,sNy+OLy / DO i=2-OLx,sNx+OLx keep their prior values). hFacZ is not
    read (the masking :70-72 is commented out). The `IF (useCubedSphereExchange)` block (:78-80) holds only a
    commented-out STOP: nothing to port. The commented-out alternative (:60-66) is not ported.
    """
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    dxC, dyC, recip_rAz, recip_deepFacC = grid.dxC, grid.dyC, grid.recip_rAz, grid.recip_deepFacC

    j = loop_j(2-OLy, sNy+OLy)                                      # :50-75
    i = loop_i(2-OLx, sNx+OLx)
    strain = strain.at[i, j].set(
        (dyC[i, j]*vFld[i, j]
         -dyC[i-1, j]*vFld[i-1, j]
         +dxC[i, j]*uFld[i, j]
         -dxC[i, j-1]*uFld[i, j-1]
         )*recip_rAz[i, j]*recip_deepFacC[k])
    return strain
