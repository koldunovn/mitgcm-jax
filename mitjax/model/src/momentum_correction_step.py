"""MOMENTUM_CORRECTION_STEP: model/src/momentum_correction_step.F @63cdc0b."""

from mitjax.eesupp.exch_rs import EXCH_UV_3D_RL
from mitjax.farray import loop_i, loop_j
from mitjax.model.src.calc_grad_phi_surf import calc_grad_phi_surf
from mitjax.model.src.correction_step import correction_step


def momentum_correction_step(myTime, myIter, *, cfg, grid, params, state, ex):
    """MOMENTUM_CORRECTION_STEP( myTime, myIter, myThid )   @63cdc0b model/src/momentum_correction_step.F:7-131

    C     | SUBROUTINE MOMENTUM_CORRECTION_STEP
    C     | o Update the momentum after the surface pressure (and NH pressure) solves.

    Returns the State. phiSurfX/Y: locals zeroed on every point (:65-70); iMin = 1-OLx+1, iMax = sNx+OLx, jMin =
    1-OLy+1, jMax = sNy+OLy (:74-77). The NONLIN_FRSURF / OBCS / diagnostics lines are not compiled in the builds
    this is gated on and raise. :127-128 EXCH_UV_3D_RL when applyExchUV_early."""
    for opt in ("ALLOW_OBCS",):                     # GO lane: no NONLIN_FRSURF lines at 63cdc0b
        if getattr(cfg.cpp, opt):
            raise NotImplementedError(f"MOMENTUM_CORRECTION_STEP: the {opt} lines are not ported")
    sz = cfg.size
    if params.momStepping:                                                      # :63
        jA = loop_j(1-sz.OLy, sz.sNy+sz.OLy)                                    # :65
        iA = loop_i(1-sz.OLx, sz.sNx+sz.OLx)                                    # :66
        phiSurfX = state.etaN.local("phiSurfX").at[iA, jA].set(0.)              # :67
        phiSurfY = state.etaN.local("phiSurfY").at[iA, jA].set(0.)              # :68
        iMin = 1-sz.OLx+1                                                       # :74
        iMax = sz.sNx+sz.OLx                                                    # :75
        jMin = 1-sz.OLy+1                                                       # :76
        jMax = sz.sNy+sz.OLy                                                    # :77
        phiSurfX, phiSurfY = calc_grad_phi_surf(iMin, iMax, jMin, jMax, state.etaN, phiSurfX, phiSurfY,   # :80-84
                                                cfg=cfg, grid=grid)
        state = correction_step(iMin, iMax, jMin, jMax, phiSurfX, phiSurfY, myTime, myIter,             # :87-90
                                cfg=cfg, grid=grid, params=params, state=state)
    if params.applyExchUV_early:                                                # :127-128
        uVel, vVel = EXCH_UV_3D_RL(state.uVel, state.vVel, True, sz.Nr, ex=ex)
        state = state.replace(uVel=uVel, vVel=vVel)
    return state
