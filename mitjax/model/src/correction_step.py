"""CORRECTION_STEP: model/src/correction_step.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j


def correction_step(iMin, iMax, jMin, jMax, phiSurfX, phiSurfY, myTime, myIter, *, cfg, grid, params, state):
    """CORRECTION_STEP( bi, bj, iMin, iMax, jMin, jMax, phiSurfX, phiSurfY, myTime, myIter, myThid )
    @63cdc0b model/src/correction_step.F:7-300

    C     | S/R CORRECTION_STEP
    C     | o Corrects the horizontal flow fields with the surface pressure (and Non-Hydrostatic pressure).
    C     phiSurfX :: gradient of Surface potential - X component
    C     phiSurfY :: gradient of Surface potential - Y component

    Returns the State with uVel, vVel. The level loop DO k=1,Nr (:87-237) has independent iterations (each level
    reads gU, gV and writes uVel, vVel of its own level): a level scan (KERNEL_GUIDE §4). use3Dsolver
    (:156-169, :179-190) raises (non-hydrostatic pressure, not in M1). gU_dpx, gV_dpy: locals, written on
    iMin..iMax, jMin..jMax and read there only."""
    if params.use3Dsolver:
        raise NotImplementedError("CORRECTION_STEP: use3Dsolver (non-hydrostatic pressure) is not ported")
    for opt in ("ALLOW_OBCS",):    # GO lane: no ALLOW_CD_CODE / NONLIN_FRSURF lines at 63cdc0b
        if getattr(cfg.cpp, opt):
            raise NotImplementedError(f"CORRECTION_STEP: the {opt} lines are not ported")
    g = grid
    uVel, vVel, gU, gV = state.uVel, state.vVel, state.gU, state.gV
    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    # :87  DO k=1,Nr as a level scan (KERNEL_GUIDE §4; no level branches)
    def level_k(k, c):
        uVel, vVel = c
        psFac = (params.pfFacMom*params.implicSurfPress                         # :152-153
                 * g.recip_deepFacC[k]*params.recip_rhoFacC[k])
        gU_dpx = phiSurfX.local("gU_dpx").at[i, j].set(-psFac*phiSurfX[i, j]*g.maskW[i, j, k])   # :173
        gV_dpy = phiSurfY.local("gV_dpy").at[i, j].set(-psFac*phiSurfY[i, j]*g.maskS[i, j, k])   # :194
        uVel = uVel.at[i, j, k].set((gU[i, j, k]                                 # :215-217
                                     + params.deltaTMom*gU_dpx[i, j]
                                     )*g.maskW[i, j, k])
        vVel = vVel.at[i, j, k].set((gV[i, j, k]                                 # :227-229
                                     + params.deltaTMom*gV_dpy[i, j]
                                     )*g.maskS[i, j, k])
        return uVel, vVel
    from mitjax.ops.scan_k import scan_levels
    uVel, vVel = scan_levels(level_k, (uVel, vVel), 1, cfg.size.Nr)
    return state.replace(uVel=uVel, vVel=vVel)
