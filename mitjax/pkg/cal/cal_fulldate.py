"""CAL_FULLDATE: pkg/cal/cal_fulldate.F @63cdc0b."""

from mitjax.pkg.cal.cal_checkdate import cal_CheckDate
from mitjax.pkg.cal.cal_h import idiv, imod
from mitjax.pkg.cal.cal_isleap import cal_IsLeap
from mitjax.pkg.cal.cal_timepassed import cal_TimePassed


def cal_FullDate(yymmdd, hhmmss, *, cal, stderr=None):
    """CAL_FULLDATE( yymmdd, hhmmss, date, myThid )   @63cdc0b pkg/cal/cal_fulldate.F:3-105

    C     Complete a given date to a full calendar date (leap-year index and weekday).

    Returns `date` (4-list)."""
    date = [yymmdd, hhmmss, 1, 1]                                              # :53-56
    if cal.cal_setStatus < 1:                                                  # :58-66
        raise RuntimeError(f"CAL_FULLDATE: yymmdd={yymmdd:9d} , hhmmss={hhmmss:9d}\nCAL_FULLDATE: called too early "
                           f"(cal_setStatus={cal.cal_setStatus:2d} )\nABNORMAL END: S/R CAL_FULLDATE")
    valid, calerr = cal_CheckDate(date, cal=cal, stderr=stderr)                # :69
    if calerr != 0 and stderr is not None:                                     # :70-75 (print only)
        stderr.append(f"CAL_FULLDATE: yymmdd={yymmdd:9d} , hhmmss={hhmmss:9d} (calerr {calerr})")
    if valid:                                                                  # :77
        theyear = idiv(yymmdd, 10000)                                          # :79
        date[2] = cal_IsLeap(theyear, cal=cal)                                 # :80
        numberOfDays = cal_TimePassed(cal.refDate, date, cal=cal)              # :83
        if numberOfDays[0] < 0:                                                # :84-93 (print only)
            if stderr is not None:
                stderr.append(" in CAL_FULLDATE: numberOfDays < 0")
        else:
            date[3] = imod(numberOfDays[0], 7) + 1                             # :95
    else:                                                                      # :97-102
        raise RuntimeError("CAL_FULLDATE: fatal error from cal_CheckDate\nABNORMAL END: S/R CAL_FULLDATE")
    return date
