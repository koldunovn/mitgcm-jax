"""DO_FIELDS_BLOCKING_EXCHANGES: model/src/do_fields_blocking_exchanges.F @63cdc0b."""

from mitjax.eesupp.exch_rs import EXCH_UV_XYZ_RL, EXCH_XYZ_RL


def do_fields_blocking_exchanges(*, cfg, params, state, ex):
    """DO_FIELDS_BLOCKING_EXCHANGES( myThid )   @63cdc0b model/src/do_fields_blocking_exchanges.F:7-103

    C     | SUBROUTINE DO_FIELDS_BLOCKING_EXCHANGES
    C     | o Controlling routine for exchanging edge info.

    Returns the State. GAD_SOM_EXCHANGES (:77-80, tempSOM_Advection / saltSOM_Advection): ADVECT lane arm.
    PTRACERS_FIELDS_BLOCKING_EXCH (:88-100, usePTRACERS): PTRACERS lane arm. Raise: the package exchanges compiled
    in other builds (:64-77: GGL90, ...)."""
    if params.useOffLine:                                                       # :49
        return state
    uVel, vVel, wVel = state.uVel, state.vVel, state.wVel
    theta, salt, totPhiHyd = state.theta, state.salt, state.totPhiHyd
    if not params.staggerTimeStep:                                              # :52-57
        if not params.applyExchUV_early:
            uVel, vVel = EXCH_UV_XYZ_RL(uVel, vVel, True, ex=ex)               # :54
        if not params.implicitIntGravWave:
            wVel = EXCH_XYZ_RL(wVel, ex=ex)                                     # :56
    if not params.implicitIntGravWave:                                          # :60-63
        theta = EXCH_XYZ_RL(theta, ex=ex)                                       # :61
        salt = EXCH_XYZ_RL(salt, ex=ex)                                         # :62
    som = {}
    if cfg.cpp.ALLOW_GENERIC_ADVDIFF and (params.tempSOM_Advection or params.saltSOM_Advection):   # :77-80
        # ADVECT lane (plan Task 14): GAD_SOM_EXCHANGES on the SOM moments (GAD_SOM_VARS.h, State som_T, som_S)
        from mitjax.pkg.generic_advdiff.gad_calc_rhs import gad_som_cfg
        from mitjax.pkg.generic_advdiff.gad_som_exchanges import gad_som_exchanges
        som["som_T"], som["som_S"] = gad_som_exchanges(cfg=gad_som_cfg(cfg, params), ex=ex,     # :79
                                                       som_T=state.som_T, som_S=state.som_S)
    if cfg.cpp.ALLOW_CD_CODE:                                                   # :82-84 (GO lane)
        from mitjax.farray import FArray
        u, v = ex.EXCH_UV_DGRID_3D_RL(state.uVelD.data, state.vVelD.data, True)   # :83  .TRUE., Nr
        state = state.replace(uVelD=FArray(u, state.uVelD.name, tiled=state.uVelD.tiled, _dims=state.uVelD.dims),
                              vVelD=FArray(v, state.vVelD.name, tiled=state.vVelD.tiled, _dims=state.vVelD.dims))
    ptf = None
    if cfg.cpp.ALLOW_PTRACERS and params.usePTRACERS:                           # :88-100 (PTRACERS lane)
        from mitjax.pkg.ptracers.ptracers_fields_blocking_exch import ptracers_fields_blocking_exch
        from mitjax.pkg.ptracers.ptracers_fields_h import ptf_of_state
        ptf = ptracers_fields_blocking_exch(cfg=cfg, ptr=params, ptf=ptf_of_state(state, params), ex=ex)   # :95-98
    if params.storePhiHyd4Phys:                                                 # :85-86
        totPhiHyd = EXCH_XYZ_RL(totPhiHyd, ex=ex)
    state = state.replace(uVel=uVel, vVel=vVel, wVel=wVel, theta=theta, salt=salt, totPhiHyd=totPhiHyd, **som)
    if ptf is not None:                                                         # PTRACERS lane
        from mitjax.pkg.ptracers.ptracers_fields_h import state_with_ptf
        state = state_with_ptf(state, ptf)
    return state
