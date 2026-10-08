"""PTRACERS_SWITCH_ONOFF: pkg/ptracers/ptracers_switch_onoff.F @63cdc0b."""


def ptracers_switch_onoff(myTime, myIter, *, cfg, ptr):
    """PTRACERS_SWITCH_ONOFF( myTime, myIter, myThid )   @63cdc0b pkg/ptracers/ptracers_switch_onoff.F:7-64

    C     Decide to switch on/off individual tracer time-stepping

    Returns `ptr`. Acts only when .NOT.PTRACERS_startAllTrc (:37-60: under ALLOW_AUTODIFF a STOP, otherwise
    PTRACERS_StepFwd from myTime .GE. PTRACERS_startStepFwd); PTRACERS_READPARMS raises for PTRACERS_startStepFwd,
    so PTRACERS_startAllTrc is .TRUE. in every ported run and the routine changes nothing."""
    if not ptr.PTRACERS_startAllTrc:                                            # :37
        raise NotImplementedError("PTRACERS_SWITCH_ONOFF: PTRACERS_startAllTrc = .FALSE. is not ported")
    return ptr
