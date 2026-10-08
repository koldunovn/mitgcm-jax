"""pkg/mom_common/mom_hdissip.F: horizontal dissipation in terms of tension and strain (MOM_HDISSIP)."""

from mitjax.farray import loop_i, loop_j


def mom_hdissip(k, tension, strain, hFacZ, viscAh_s, viscAh_t, viscA4_s, viscA4_t,
                harmonic, biharmonic, useVariableViscosity, uDissip, vDissip, *, cfg, grid):
    """MOM_HDISSIP( bi, bj, k, tension, strain, hFacZ, viscAh_s, viscAh_t, viscA4_s, viscA4_t,
                    harmonic, biharmonic, useVariableViscosity, uDissip, vDissip, myThid )
    @63cdc0b pkg/mom_common/mom_hdissip.F:3-89

    C     Calculate horizontal dissipation terms in terms of tension and strain
    C       Du = d/dx At Tension + d/dy As Strain
    C       Dv = d/dx As Strain  - d/dy At Tension

    Returns (uDissip, vDissip). M3 lane MLAdjust (input.AhStTn: useStrainTensionVisc, harmonic). `harmonic`,
    `biharmonic` are static (they select code); with `harmonic` the point loop :42-82 runs on the whole (i,j) range
    at once (each point reads only inputs), the products and differences left to right as written; without it the
    arrays keep the caller's values. `biharmonic` is the Fortran STOP (:84-86). hFacZ, viscA4_s, viscA4_t and
    useVariableViscosity are not read (as in the Fortran)."""
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    g = grid
    if harmonic:                                                    # :41
        j = loop_j(2-OLy, sNy+OLy-1)                                # :42
        i = loop_i(2-OLx, sNx+OLx-1)                                # :43
        uDissip = uDissip.at[i, j].set(                             # :45-61
            g.recip_dyG[i, j]*g.recip_dyG[i, j]
            * g.recip_dxC[i, j]
            * (
                g.dyF[i, j]*g.dyF[i, j]
                * viscAh_t[i, j]*tension[i, j]
                - g.dyF[i-1, j]*g.dyF[i-1, j]
                * viscAh_t[i-1, j]*tension[i-1, j]
            )
            + g.recip_dxC[i, j]*g.recip_dxC[i, j]
            * g.recip_dyG[i, j]
            * (
                g.dxV[i, j+1]*g.dxV[i, j+1]
                * viscAh_s[i, j+1]*strain[i, j+1]
                - g.dxV[i, j]*g.dxV[i, j]
                * viscAh_s[i, j]*strain[i, j]
            ))
        vDissip = vDissip.at[i, j].set(                             # :63-79
            g.recip_dyC[i, j]*g.recip_dyC[i, j]
            * g.recip_dxG[i, j]
            * (
                g.dyU[i+1, j]*g.dyU[i+1, j]
                * viscAh_s[i+1, j]*strain[i+1, j]
                - g.dyU[i, j]*g.dyU[i, j]
                * viscAh_s[i, j]*strain[i, j]
            )
            - g.recip_dxG[i, j]*g.recip_dxG[i, j]
            * g.recip_dyC[i, j]
            * (
                g.dxF[i, j]*g.dxF[i, j]
                * viscAh_t[i, j]*tension[i, j]
                - g.dxF[i, j-1]*g.dxF[i, j-1]
                * viscAh_t[i, j-1]*tension[i, j-1]
            ))
    if biharmonic:                                                  # :84-86
        raise RuntimeError("STOP MOM_HDISSIP: BIHARMONIC NOT ALLOWED WITH STRAIN-TENSION")
    return uDissip, vDissip
