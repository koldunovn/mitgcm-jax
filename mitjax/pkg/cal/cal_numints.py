"""CAL_NUMINTS: pkg/cal/cal_numints.F @63cdc0b (lane M4ADCOL, pkg/ecco's COST_AVERAGESFLAGS)."""

import numpy as np

from mitjax.pkg.cal.cal_timepassed import cal_TimePassed
from mitjax.pkg.cal.cal_toseconds import cal_ToSeconds


def cal_NumInts(date_a, date_b, timeint, *, cal):
    """INTEGER FUNCTION cal_NumInts( date_a, date_b, timeint, mythid )   @63cdc0b pkg/cal/cal_numints.F:3-67

    c     Return the number of time intervals between two dates.

    :53-59: with timeint(4) = -1 (a time interval): passed = cal_TimePassed(date_a, date_b), cal_NumInts =
    abs(passedsecs/timeintsecs), the REAL*8 quotient assigned to the INTEGER function value (truncation toward
    zero); otherwise error 2501 and STOP (:60-64)."""
    if timeint[3] == -1:
        passed = cal_TimePassed(date_a, date_b, cal=cal)
        passedsecs = np.float64(cal_ToSeconds(passed, cal=cal))
        timeintsecs = np.float64(cal_ToSeconds(timeint, cal=cal))
        return int(np.abs(passedsecs/timeintsecs))
    raise RuntimeError("cal_NumInts: error 2501\n stopped in cal_NumInts.")
