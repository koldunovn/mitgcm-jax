"""CALC_GRAD_PHI_HYD: model/src/calc_grad_phi_hyd.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j


def calc_grad_phi_hyd(k, iMin, iMax, jMin, jMax, phiHydC, alphRho, dPhiHydX, dPhiHydY, myTime, myIter, *, cfg,
                      grid, params, phi0surf, state=None):
    """CALC_GRAD_PHI_HYD( k, bi, bj, iMin,iMax, jMin,jMax, phiHydC, alphRho, dPhiHydX, dPhiHydY, myTime, myIter,
    myThid )   @63cdc0b model/src/calc_grad_phi_hyd.F:7-281

    C     | SUBROUTINE CALC_GRAD_PHI_HYD
    C     | o Calculate the gradient of Hydrostatic potential anom.
    C     k         :: level index
    C     phiHydC   :: hydrostatic potential anomaly at cell center
    C     alphRho   :: Density (z-coord) or specific volume (p-coord)
    C     dPhiHydX,Y :: Gradient (X & Y directions) of hyd. potential

    Returns (dPhiHydX, dPhiHydY). `phi0surf`: SURFACE.h /SURF_FIXED/ (written by EXTERNAL_FORCING_SURF; passed explicitly, FORCING lane). Ported: :127-134 varLoc = phiHydC +
    phi0surf (`IF (.TRUE.)`), :137-154 the gradients, :271-276 the masks. GO lane (plan Task 15a), NONLIN_FRSURF
    with select_rStar = 2 (`state`: rStarFacC, etaH): varLoc = phiHydC*rStarFacC + phi0surf (:62-82, fluidIsWater)
    and the z* slope term (:156-203, flat-top or general form); raise: fluidIsAir, select_rStar = 1 with
    nonlinFreeSurf >= 4 (:84-122), the p* terms (:205-264). varLoc is a
    local not initialised outside jMin..jMax, iMin..iMax (read only inside). alphRho is read by the r* slope term only."""
    rstar = cfg.cpp.NONLIN_FRSURF and not cfg.cpp.DISABLE_RSTAR_CODE
    if rstar and params.select_rStar >= 1 and params.nonlinFreeSurf >= 1:
        if state is None:
            raise ValueError("CALC_GRAD_PHI_HYD: the r* terms read SURFACE.h / DYNVARS.h: pass state=")
        if params.fluidIsAir:
            raise NotImplementedError("CALC_GRAD_PHI_HYD: the fluidIsAir r* terms (pStarFacK, atm_*) are not ported")
        if params.select_rStar < 2 and params.nonlinFreeSurf >= 4:
            raise NotImplementedError("CALC_GRAD_PHI_HYD: select_rStar = 1 with nonlinFreeSurf >= 4 (:84-122) is "
                                      "not ported")
    if not cfg.cpp.INCLUDE_PHIHYD_CALCULATION_CODE:                             # :48, :278
        raise NotImplementedError("CALC_GRAD_PHI_HYD: a build without INCLUDE_PHIHYD_CALCULATION_CODE")
    recip_dxC, recip_dyC, maskW, maskS = grid.recip_dxC, grid.recip_dyC, grid.maskW, grid.maskS
    recip_deepFacC, recip_rhoFacC = grid.recip_deepFacC, params.recip_rhoFacC
    sz = cfg.size
    varLoc = phiHydC.local("varLoc")
    j = loop_j(jMin, jMax)                                                      # :129
    i = loop_i(iMin, iMax)                                                      # :130
    if rstar and params.select_rStar >= 2 and params.nonlinFreeSurf >= 4:      # :62-82 (GO lane; fluidIsWater)
        varLoc = varLoc.at[i, j].set(phiHydC[i, j]*state.rStarFacC[i, j]       # :77-78
                                     + phi0surf[i, j])
    else:                                                                       # :123-134
        varLoc = varLoc.at[i, j].set(phiHydC[i, j]+phi0surf[i, j])           # :131
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)                                         # :137
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)                                         # :138
    dPhiHydX = dPhiHydX.at[i, j].set(0.)                                        # :139  0. _d 0
    dPhiHydY = dPhiHydY.at[i, j].set(0.)                                        # :140
    j = loop_j(jMin, jMax)                                                      # :143
    i = loop_i(iMin+1, iMax)                                                    # :144
    dPhiHydX = dPhiHydX.at[i, j].set(recip_dxC[i, j]*recip_deepFacC[k]          # :145-146
                                     * (varLoc[i, j]-varLoc[i-1, j])*recip_rhoFacC[k])
    j = loop_j(jMin+1, jMax)                                                    # :149
    i = loop_i(iMin, iMax)                                                      # :150
    dPhiHydY = dPhiHydY.at[i, j].set(recip_dyC[i, j]*recip_deepFacC[k]          # :151-152
                                     * (varLoc[i, j]-varLoc[i, j-1])*recip_rhoFacC[k])
    if rstar and params.select_rStar >= 2 and params.nonlinFreeSurf >= 1:      # :156-268 (GO lane)
        # :160-162 generalForm: static, decided in ini_parms_dyn (useShelfIce, rF(1), topoFile)
        generalForm = params.gradPhiHyd_generalForm
        if params.fluidIsWater and (params.usingZCoords or generalForm):        # :164
            if params.usingZCoords:
                factorP = params.gravity*params.recip_rhoConst*recip_rhoFacC[k]*0.5   # :167  0.5 _d 0
            else:
                factorP = 0.5                                                   # :170
            j = loop_j(jMin, jMax)
            i = loop_i(iMin, iMax)
            if generalForm:                                                     # :172-179
                varLoc = varLoc.at[i, j].set(state.etaH[i, j]*grid.recip_Rcol[i, j]
                                             * (grid.rC[k] - grid.R_low[i, j]))  # :176-177
            else:                                                               # :180-188
                varLoc = varLoc.at[i, j].set(state.etaH[i, j]
                                             * (1. + grid.rC[k]*grid.recip_Rcol[i, j]))   # :184-185  1. _d 0
            j = loop_j(jMin, jMax)
            i = loop_i(iMin+1, iMax)
            dPhiHydX = dPhiHydX.at[i, j].set(dPhiHydX[i, j]                     # :191-194
                                             + factorP*(alphRho[i-1, j]+alphRho[i, j])
                                             * (varLoc[i, j]-varLoc[i-1, j])
                                             * recip_dxC[i, j]*recip_deepFacC[k])
            j = loop_j(jMin+1, jMax)
            i = loop_i(iMin, iMax)
            dPhiHydY = dPhiHydY.at[i, j].set(dPhiHydY[i, j]                     # :199-202
                                             + factorP*(alphRho[i, j-1]+alphRho[i, j])
                                             * (varLoc[i, j]-varLoc[i, j-1])
                                             * recip_dyC[i, j]*recip_deepFacC[k])
        elif params.fluidIsWater:                                               # :205-224
            raise NotImplementedError("CALC_GRAD_PHI_HYD: the p* slope term (:205-224) is not ported")
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)                                         # :271
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)                                         # :272
    dPhiHydX = dPhiHydX.at[i, j].set(dPhiHydX[i, j]*maskW[i, j, k])             # :273
    dPhiHydY = dPhiHydY.at[i, j].set(dPhiHydY[i, j]*maskS[i, j, k])             # :274
    return dPhiHydX, dPhiHydY
