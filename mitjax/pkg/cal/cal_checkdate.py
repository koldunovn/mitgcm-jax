"""CAL_CHECKDATE: pkg/cal/cal_checkdate.F @63cdc0b."""

from mitjax.pkg.cal.cal_convdate import cal_ConvDate
from mitjax.pkg.cal.cal_h import idiv, imod, nMonthYear


def cal_CheckDate(date, *, cal, stderr=None):
    """CAL_CHECKDATE( date, valid, calerr, myThid )   @63cdc0b pkg/cal/cal_checkdate.F:3-143

    C     Check whether the date is valid.

    Returns (valid, calerr). PRINT_ERROR / warning messages go to the list `stderr` (None: discarded)."""
    def err(msg):
        if stderr is not None:
            stderr.append(msg)
    valid = True                                                               # :52
    calerr = 0                                                                 # :53
    wrong_sign = (date[0] < 0 and date[1] > 0) or (date[0] > 0 and date[1] < 0)   # :56-57
    if wrong_sign:                                                             # :59-63
        calerr = 1803
        valid = False
    elif cal.cal_setStatus < 1:                                                # :64-71
        err(f"CAL_CHECKDATE: date={date[0]:9d}{date[1]:9d}{date[2]:9d}{date[3]:9d}")
        err(f"CAL_CHECKDATE: called too early (cal_setStatus={cal.cal_setStatus:2d} )")
    elif date[3] <= 0:                                                         # :73-82
        if date[3] != -1:
            calerr = 1801
        elif date[2] != 0:
            calerr = 1802
    else:                                                                      # :84-112
        yy, mm, dd, nsecs, lp, wd = cal_ConvDate(date, cal=cal)                # :87
        if mm == 0 or abs(mm) > nMonthYear:                                    # :88-93
            err(f"CAL_CHECKDATE: Invalid month in date(1)={date[0]:10d}")
            valid = False
        elif wd < 1 or wd > 7:                                                 # :94-97
            calerr = 1805
        elif lp != 1 and lp != 2:                                              # :98-102
            calerr = 1806
            valid = False
        elif dd == 0 or abs(dd) > cal.nMaxDayMonth:                            # :103-108
            err(f"CAL_CHECKDATE: Invalid day in date(1)={date[0]:10d}")
        elif date[0] < cal.refDate[0]:                                         # :109-112
            calerr = 1807
    if valid and cal.cal_setStatus >= 1:                                       # :116-140 (warnings only)
        hhmmss = abs(date[1])
        hh = idiv(hhmmss, 10000)
        mn = imod(idiv(hhmmss, 100), 100)
        ss = imod(hhmmss, 100)
        if ss >= cal.secondsPerMinute:
            err(f"** WARNING ** CAL_CHECKDATE: Invalid Seconds in date(2)={date[1]:10d}")
        if mn >= cal.minutesPerHour:
            err(f"** WARNING ** CAL_CHECKDATE: Invalid Minutes in date(2)={date[1]:10d}")
        if hh >= cal.hoursPerDay:
            err(f"** WARNING ** CAL_CHECKDATE: Invalid  Hours  in date(2)={date[1]:10d}")
    return valid, calerr
