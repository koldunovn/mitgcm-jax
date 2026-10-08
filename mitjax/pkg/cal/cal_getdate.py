"""CAL_GETDATE: pkg/cal/cal_getdate.F @63cdc0b."""

import numpy as np

from mitjax.pkg.cal.cal_addtime import cal_AddTime
from mitjax.pkg.cal.cal_timeinterval import cal_TimeInterval


def cal_GetDate(myIter, myTime, *, cal):
    """CAL_GETDATE( myIter, myTime, mydate, myThid )   @63cdc0b pkg/cal/cal_getdate.F:3-93

    C     Determine the current date given the iteration number and/or the
    C     current time of integration.

    Concrete myIter (int) and myTime (float64); returns `mydate` (4-list)."""
    if myIter == -1:                                                           # :47-53
        return [cal.startdate_1, cal.startdate_2, 1, 1]
    if cal.cal_setStatus < 3:                                                  # :55-63
        raise RuntimeError(f"CAL_GETDATE: myIter={myIter:10d} , myTime={float(myTime):19.2f}\nCAL_GETDATE: called "
                           f"too early (cal_setStatus={cal.cal_setStatus:2d} )\nABNORMAL END: S/R CAL_GETDATE")
    if np.float64(myTime) == cal.modelStart:                                   # :66-72
        return list(cal.modelStartDate)
    secs = np.float64(myTime) - cal.modelStart                                 # :84
    workdate = cal_TimeInterval(secs, "secs", cal=cal)                         # :87
    return cal_AddTime(cal.modelStartDate, workdate, cal=cal)                  # :88
