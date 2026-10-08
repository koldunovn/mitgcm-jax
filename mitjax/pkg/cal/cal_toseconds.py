"""CAL_TOSECONDS: pkg/cal/cal_toseconds.F @63cdc0b."""

import numpy as np

from mitjax.pkg.cal.cal_h import idiv, imod


def cal_ToSeconds(date, *, cal):
    """CAL_TOSECONDS( date, timeint, myThid )   @63cdc0b pkg/cal/cal_toseconds.F:3-99

    C     Transform the calendar time of a time interval given in date format
    C     into seconds.

    Returns `timeint` (REAL*8). fac, nsecs, ndays are _RL locals (:48): `ndays*secondsperday` is a REAL*8 product
    of the converted INTEGER; the hhmmss terms are INTEGER expressions converted on the sum."""
    if cal.cal_setStatus < 1:                                                  # :56-64
        raise RuntimeError("CAL_TOSECONDS: called too early\nABNORMAL END: S/R CAL_CONVDATE")
    check_sign = 1                                                             # :66-69
    if (date[0] < 0 and date[1] > 0) or (date[0] > 0 and date[1] < 0):
        check_sign = -1
    if (date[3] == -1 and date[2] == 0 and check_sign >= 0) or cal.usingModelCalendar:   # :71-74
        if date[0] < 0 or date[1] < 0:                                         # :75-84
            ndays = np.float64(-date[0])
            hhmmss = -date[1]
            fac = np.float64(-1)
        else:
            ndays = np.float64(date[0])
            hhmmss = date[1]
            fac = np.float64(1)
        # :85-88  nsecs = ndays*secondsperday + (hhmmss/10000)*secondsperhour + mod(hhmmss/100,100)*secondsperminute
        #                 + mod(hhmmss,100)   (left to right: REAL*8 + INTEGER terms, each converted on its addition)
        nsecs = ndays*np.float64(cal.secondsPerDay)
        nsecs = nsecs + np.float64(idiv(hhmmss, 10000)*cal.secondsPerHour)
        nsecs = nsecs + np.float64(imod(idiv(hhmmss, 100), 100)*cal.secondsPerMinute)
        nsecs = nsecs + np.float64(imod(hhmmss, 100))
        timeint = fac*nsecs                                                    # :89
    else:                                                                      # :90-96  ierr = 1001
        raise RuntimeError("cal_ToSeconds: error 1001 (not a time interval)\n stopped in cal_ToSeconds.")
    return timeint
