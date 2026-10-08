"""DIAGS_PHI_HYD: model/src/diags_phi_hyd.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX


def diags_phi_hyd(k, iMin, iMax, jMin, jMax, phiHydC, myTime, myIter, *, cfg, grid, params, state, phi0surf):
    """DIAGS_PHI_HYD( k, bi, bj, iMin,iMax, jMin,jMax, phiHydC, myTime, myIter, myThid )
    @63cdc0b model/src/diags_phi_hyd.F:7-133

    C     | S/R DIAGS_PHI_HYD
    C     | o Diagnose full hydrostatic Potential at cell center ; used for output & with EOS funct. of P

    Returns the State with totPhiHyd (DYNVARS.h) of level k: :60-69 phiHydC + Bo_surf*etaN + phi0surf; under
    NONLIN_FRSURF with select_rStar >= 1 and nonlinFreeSurf >= 4 the r* form (:73-118, z coordinates; GO lane).
    The diagnostics fills (useDiagnostics), fluidIsAir and the p-coordinate arms raise."""
    nlfs_rstar = cfg.cpp.NONLIN_FRSURF and params.select_rStar >= 1 and params.nonlinFreeSurf >= 4   # :73 (GO)
    if nlfs_rstar and (params.fluidIsAir or params.usingPCoords):
        raise NotImplementedError("DIAGS_PHI_HYD: the NONLIN_FRSURF fluidIsAir / p-coordinate arms are not ported")
    if params.useDiagnostics:
        raise NotImplementedError("DIAGS_PHI_HYD: the diagnostics block is not ported")
    j = loop_j(jMin, jMax)                                                      # :60
    i = loop_i(iMin, iMax)                                                      # :61
    totPhiHyd = state.totPhiHyd.at[i, j, k].set(phiHydC[i, j]                   # :62-64
                                                + grid.Bo_surf[i, j]*state.etaN[i, j]
                                                + phi0surf[i, j])
    # :65-67 phiHydCstR (NONLIN_FRSURF): read only by the diagnostics fill (:120-124, useDiagnostics raises above)
    if nlfs_rstar:                                                              # :73-118 (GO lane; z coordinates)
        dPhiRef = (grid.Ro_surf[i, j]-grid.rC[k])*params.gravity                # :108
        totPhiHyd = totPhiHyd.at[i, j, k].set(phiHydC[i, j]*state.rStarFacC[i, j]   # :109-113
                                              + MAX(dPhiRef, 0., p="b")         # :112
                                              * (state.rStarFacC[i, j] - 1.)    # 1. _d 0
                                              + phi0surf[i, j])
    return state.replace(totPhiHyd=totPhiHyd)
