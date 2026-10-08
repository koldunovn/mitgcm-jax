"""CAL_INIT_FIXED: pkg/cal/cal_init_fixed.F @63cdc0b."""

from mitjax.pkg.cal.cal_set import cal_Set


def cal_init_fixed(cal, *, startTime, endTime, deltaTClock, nIter0, nEndIter, nTimeSteps, stderr=None):
    """CAL_INIT_FIXED( myThid )   @63cdc0b pkg/cal/cal_init_fixed.F:8-46

    C     Initialise the calendar: CAL_SET with the time-stepping parameters of PARAMS.h.

    The PARAMS.h values are passed as keywords (host values of ini_parms' TimeParams). CAL_SUMMARY (:39) is
    print-out only (not ported)."""
    return cal_Set(startTime, endTime, deltaTClock, nIter0, nEndIter, nTimeSteps, cal=cal, stderr=stderr)   # :34-37
