"""DO_STAGGER_FIELDS_EXCHANGES: model/src/do_stagger_fields_exchanges.F @63cdc0b."""

from mitjax.eesupp.exch_rs import EXCH_3D_RL, EXCH_UV_3D_RL


def do_stagger_fields_exchanges(myTime, myIter, *, cfg, params, state, ex):
    """DO_STAGGER_FIELDS_EXCHANGES( myTime, myIter, myThid )   @63cdc0b model/src/do_stagger_fields_exchanges.F:6-58

    C     | SUBROUTINE DO_STAGGER_FIELDS_EXCHANGES
    C     | o Exchange edge info of Active tracers fields (U,V) (and W in NH case)
    C     |   which are needed to compute the staggered Thermodynamics

    Returns the State. Under .NOT.useOffLine (:32): with staggerTimeStep, EXCH_UV_3D_RL(uVel, vVel, .TRUE., Nr)
    unless applyExchUV_early (:38-39) and EXCH_3D_RL(wVel, Nr) unless implicitIntGravWave (:41-42). Raise: the
    implicitIntGravWave exchanges of theta, salt (:46-51, EXCH_SM_3D_RL; implicitIntGravWave is refused by
    FORWARD_STEP), useOffLine."""
    if params.useOffLine:
        raise NotImplementedError("DO_STAGGER_FIELDS_EXCHANGES: useOffLine is not ported")
    Nr = cfg.size.Nr
    if params.staggerTimeStep:                                                  # :37
        if not params.applyExchUV_early:
            uVel, vVel = EXCH_UV_3D_RL(state.uVel, state.vVel, True, Nr, ex=ex)    # :38-39
            state = state.replace(uVel=uVel, vVel=vVel)
        if not params.implicitIntGravWave:
            state = state.replace(wVel=EXCH_3D_RL(state.wVel, Nr, ex=ex))      # :41-42
    if params.implicitIntGravWave:                                              # :46-51
        raise NotImplementedError("DO_STAGGER_FIELDS_EXCHANGES: the implicitIntGravWave arm (EXCH_SM_3D_RL) is "
                                  "not ported")
    return state
