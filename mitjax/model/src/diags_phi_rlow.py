"""DIAGS_PHI_RLOW: model/src/diags_phi_rlow.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX, MIN


def diags_phi_rlow(k, iMin, iMax, jMin, jMax, phiHydF, phiHydC, alphRho, myTime, myIter, *, cfg, grid, params,
                   state, phi0surf):
    """DIAGS_PHI_RLOW( k, bi, bj, iMin,iMax, jMin,jMax, phiHydF, phiHydC, alphRho, myTime, myIter, myThid )
    @63cdc0b model/src/diags_phi_rlow.F:6-195

    C     | S/R DIAGS_PHI_RLOW
    C     | o Diagnose Phi-Hydrostatic at r-lower boundary (ocean bottom in z-coord.)

    Returns the State with phiHydLow (DYNVARS.h). Ported: usingZCoords (:61-115) with both integr_GeoPot arms, then
    at k = Nr (:117-190) the `IF (.TRUE.)` surface term (:176-186), or under NONLIN_FRSURF with select_rStar >= 1
    and nonlinFreeSurf >= 4 the r* rescaling (:130-174, z coordinates; GO lane). Lane B (Task 25): usingPCoords
    (:121-128: phiHydLow = phiHydF at k = Nr; the usingZCoords block is skipped). The NONLIN_FRSURF r* rescalings of
    fluidIsAir and usingPCoords raise.
    The IF on kLowC (an integer field) is a `where` on the point. `MIN(zeroRL, ...)`, `MAX(zeroRL, ...)` with
    mitjax.ops.fortran_minmax (gfortran's values)."""
    nlfs_rstar = cfg.cpp.NONLIN_FRSURF and params.select_rStar >= 1 and params.nonlinFreeSurf >= 4   # :132 (GO)
    if nlfs_rstar and params.fluidIsAir:
        raise NotImplementedError("DIAGS_PHI_RLOW: the NONLIN_FRSURF fluidIsAir rescaling (:134-146) is not ported")
    if nlfs_rstar and params.usingPCoords:
        raise NotImplementedError("DIAGS_PHI_RLOW: the NONLIN_FRSURF p-coordinate rescaling (:147-160) is not ported")
    if not (params.usingZCoords or params.usingPCoords):
        raise NotImplementedError("DIAGS_PHI_RLOW: neither usingZCoords nor usingPCoords")
    import jax.numpy as jnp
    sz = cfg.size
    g = grid
    phiHydLow = state.phiHydLow
    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    if params.usingZCoords:                                                     # :61-115
        if k == 1:                                                              # :68-74
            jA = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
            iA = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
            phiHydLow = phiHydLow.at[iA, jA].set(0.)                            # :71  0. _d 0
        atLow = g.kLowC[i, j] == k
        ddRloc = g.rC[k]-g.R_low[i, j]                                          # :82 / :102
        if params.integr_GeoPot == 1:                                           # :76-87
            phiHydLow = phiHydLow.at[i, j].set(jnp.where(
                atLow,
                phiHydC[i, j] + ddRloc*params.gravFacC[k]*params.gravity*alphRho[i, j]*params.recip_rhoConst,  # :83-84
                phiHydLow[i, j]))
        else:                                                                   # :89-112
            ratioRm = 1.                                                        # :92  oneRL
            ratioRp = 1.                                                        # :93
            if k > 1:
                ratioRm = 0.5*g.drC[k]/(g.rF[k]-g.rC[k])                        # :94  halfRL
            if k < sz.Nr:
                ratioRp = 0.5*g.drC[k+1]/(g.rC[k]-g.rF[k+1])                    # :95
            ratioRm = ratioRm*params.gravFacF[k]                                # :96
            ratioRp = ratioRp*params.gravFacF[k+1]                              # :97
            phiHydLow = phiHydLow.at[i, j].set(jnp.where(
                atLow,
                phiHydC[i, j]                                                   # :103-106
                + (MIN(0., ddRloc, p="b")*ratioRm                            # :103-106
                   + MAX(0., ddRloc, p="b")*ratioRp
                   )*params.gravity*alphRho[i, j]*params.recip_rhoConst,
                phiHydLow[i, j]))
    if k == sz.Nr and params.usingPCoords:                                      # :121-128 (lane B, Task 25)
        phiHydLow = phiHydLow.at[i, j].set(phiHydF[i, j])                       # :125
    if k == sz.Nr and nlfs_rstar:                                               # :130-174 (GO lane; usingZCoords)
        dPhiRef = (g.Ro_surf[i, j]-g.R_low[i, j])*params.gravity                # :163-164
        phiHydLow = phiHydLow.at[i, j].set(phiHydLow[i, j]*state.rStarFacC[i, j]   # :165-168
                                           + dPhiRef*(state.rStarFacC[i, j] - 1.)  # 1. _d 0
                                           + phi0surf[i, j])
    elif k == sz.Nr:                                                            # :117-190
        phiHydLow = phiHydLow.at[i, j].set(phiHydLow[i, j]                      # :181-183
                                           + g.Bo_surf[i, j]*state.etaN[i, j]
                                           + phi0surf[i, j])
    return state.replace(phiHydLow=phiHydLow)
