"""CAL_CONVDATE: pkg/cal/cal_convdate.F @63cdc0b."""

from mitjax.pkg.cal.cal_h import idiv, imod


def cal_ConvDate(date, *, cal):
    """CAL_CONVDATE( date, yy, mm, dd, ss, lp, wd, myThid )   @63cdc0b pkg/cal/cal_convdate.F:3-105

    C     Decompose the calendar date into year, month, day, second, leap-year
    C     index and weekday.

    `date` a 4-list (date(1..4)); returns (yy, mm, dd, ss, lp, wd)."""
    if cal.cal_setStatus < 1:                                                  # :50-58
        raise RuntimeError("CAL_CONVDATE: called too early\nABNORMAL END: S/R CAL_CONVDATE")
    fac = 1                                                                    # :61
    wrong_sign = (date[0] < 0 and date[1] > 0) or (date[0] > 0 and date[1] < 0)   # :62-63
    if wrong_sign:                                                             # :65-68  ierr = 901
        raise RuntimeError("cal_ConvDate: error 901 (signs of the date components differ)\n"
                           " stopped in cal_ConvDate.")
    if date[0] < 0 or date[1] < 0:                                             # :70-78
        date_1 = -date[0]
        date_2 = -date[1]
        fac = -1
    else:
        date_1 = date[0]
        date_2 = date[1]
        fac = 1
    if date[3] != -1:                                                          # :82-90
        yy = idiv(date_1, 10000)
        mm = imod(idiv(date_1, 100), 100)
        dd = imod(date_1, 100)
    else:
        yy = 0
        mm = 0
        dd = date_1
    ss = (imod(date_2, 100) + imod(idiv(date_2, 100), 100)*cal.secondsPerMinute   # :91-93
          + idiv(date_2, 10000)*cal.secondsPerHour)
    yy = fac*yy                                                                # :96-99
    mm = fac*mm
    dd = fac*dd
    ss = fac*ss
    lp = date[2]                                                               # :101
    wd = date[3]                                                               # :102
    return yy, mm, dd, ss, lp, wd
