"""CAL_ISLEAP: pkg/cal/cal_isleap.F @63cdc0b."""

from mitjax.pkg.cal.cal_h import imod


def cal_IsLeap(year, *, cal):
    """INTEGER FUNCTION CAL_ISLEAP( year, myThid )   @63cdc0b pkg/cal/cal_isleap.F:3-71

    C     In case the Gregorian calendar is used determine whether the
    C     given year is a leap year or not.

    Returns 1 (no leap year) or 2 (leap year)."""
    if cal.cal_setStatus < 1:                                                  # :40-48
        raise RuntimeError(f"CAL_ISLEAP: year={year:9d}\nCAL_ISLEAP: called too early (cal_setStatus="
                           f"{cal.cal_setStatus:2d} )\nABNORMAL END: FUNCTION CAL_ISLEAP")
    if cal.usingGregorianCalendar:                                             # :50
        if imod(year, 4) != 0:                                                 # :51
            r = 1
        else:
            r = 2                                                              # :54
            if imod(year, 100) == 0 and imod(year, 400) != 0:                  # :55-56
                r = 1
    elif cal.usingJulianCalendar:                                              # :60
        if imod(year, 4) != 0:
            r = 1
        else:
            r = 2
    else:
        r = 1                                                                  # :67
    return r
