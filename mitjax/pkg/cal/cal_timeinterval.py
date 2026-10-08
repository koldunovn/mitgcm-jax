"""CAL_TIMEINTERVAL: pkg/cal/cal_timeinterval.F @63cdc0b."""

import math

import numpy as np

from mitjax.pkg.cal.cal_h import idiv, imod


def cal_TimeInterval(timeint, timeunit, *, cal):
    """CAL_TIMEINTERVAL( timeint, timeunit, date, myThid )   @63cdc0b pkg/cal/cal_timeinterval.F:3-102

    C     Create an array in date format given a time interval measured in
    C     units of timeunit.

    `timeint` REAL*8; returns `date` (4-list: days, hhmmss, 0, -1). `float(secondsperday)` is a REAL*4 conversion
    (86400. is exact in binary32); INT truncates toward zero."""
    timeint = np.float64(timeint)
    fac = 1                                                                    # :52
    if timeint < 0:                                                            # :53
        fac = -1
    date = [0, 0, 0, -1]                                                       # :55-56 date(4) = -1, date(3) = 0
    if timeunit == "secs":                                                     # :57
        if cal.cal_setStatus < 1:                                              # :59-67
            raise RuntimeError("CAL_TIMEINTERVAL: called too early\nABNORMAL END: S/R CAL_TIMEINTERVAL")
        date[0] = int(math.trunc(timeint / np.float64(np.float32(cal.secondsPerDay))))   # :68
        tmp1 = np.float64(date[0])                                             # :69
        tmp2 = np.float64(cal.secondsPerDay)                                   # :70
        nsecs = int(math.trunc(timeint - tmp1 * tmp2))                         # :71
    elif timeunit == "model":                                                  # :73-86
        raise NotImplementedError("CAL_TIMEINTERVAL: timeunit 'model' is not ported (no caller in the M4 runs)")
    else:                                                                      # :88-93  ierr = 701
        raise RuntimeError("cal_TimeInterval: error 701 (invalid time unit)\n stopped in cal_TimeInterval.")
    hhmmss = idiv(nsecs, cal.secondsPerMinute)                                 # :96
    date[1] = (idiv(hhmmss, cal.minutesPerHour)*10000                          # :97-99
               + (imod(fac*hhmmss, cal.minutesPerHour)*100 + imod(fac*nsecs, cal.secondsPerMinute))*fac)
    return date
