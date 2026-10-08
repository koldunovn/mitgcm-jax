"""RESET_NLFS_VARS: model/src/reset_nlfs_vars.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j


def reset_nlfs_vars(myTime, myIter, *, cfg, params, state):
    """RESET_NLFS_VARS( myTime, myIter, myThid )   @63cdc0b model/src/reset_nlfs_vars.F:6-69

    C     | SUBROUTINE RESET_NLFS_VARS
    C     | o Re-set some Non-Linear Free-Surface variables
    C     |   in order to facilitate the AD tool task of solving
    C     |   dependency rules.
    C     myTime    :: Current time in simulation
    C     myIter    :: Current iteration number in simulation

    Called by FORWARD_STEP when doResetHFactors (forward_step.F:465-467; the caller decides). Returns the State with
    pStarFacK (SURFACE.h): `1. _d 0` on every point (:53-58). Raise: fluidIsAir with select_rStar >= 1 (:47-52,
    pStarFacK = rStarFacC**atm_kappa). With DISABLE_RSTAR_CODE (global_ocean.90x40x15/code_ad) the body :36-65 is
    not compiled: the routine writes nothing."""
    if cfg.cpp.DISABLE_RSTAR_CODE:                                  # :36-65 not compiled (GOADK lane: code_ad)
        return state
    if params.fluidIsAir and params.select_rStar >= 1:
        raise NotImplementedError("RESET_NLFS_VARS: fluidIsAir (reset_nlfs_vars.F:47-52) is not ported")
    sz = cfg.size
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    pStarFacK = state.pStarFacK.at[i, j].set(1.)                                # :56  1. _d 0
    return state.replace(pStarFacK=pStarFacK)
