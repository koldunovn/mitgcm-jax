"""PTRACERS_RESET: pkg/ptracers/ptracers_reset.F @63cdc0b."""


def ptracers_reset(myTime, myIter, *, cfg, ptr, ptf):
    """PTRACERS_RESET( myTime, myIter, myThid )   @63cdc0b pkg/ptracers/ptracers_reset.F:9-144

    C     Re-initialize PTRACERS if it is the correct time to do so

    Returns `ptf`. The reset (:55-133) runs only for a tracer with PTRACERS_resetFreq > 0 (:53-55); no ported run
    sets PTRACERS_resetFreq (PTRACERS_READPARMS raises when data.ptracers does; its default is 0., ptracers_readparms.F:114), so the
    routine leaves every field unchanged."""
    if any(getattr(ptr, "PTRACERS_resetFreq_gt_0", ())):
        raise NotImplementedError("PTRACERS_RESET: PTRACERS_resetFreq > 0 is not ported")
    return ptf
