"""EXF_GETFFIELD_START: pkg/exf/exf_getffield_start.F @63cdc0b."""

import numpy as np

from mitjax.model.grid import UNSET_RL
from mitjax.pkg.cal.cal_fulldate import cal_FullDate
from mitjax.pkg.cal.cal_getdate import cal_GetDate
from mitjax.pkg.cal.cal_timepassed import cal_TimePassed
from mitjax.pkg.cal.cal_toseconds import cal_ToSeconds


def exf_getffield_start(useYearlyFields, pkg_name, fld_name, fld_period, fld_startdate1, fld_startdate2,
                        fld_start_time, errCount, *, useCAL, cal, nIter0, startTime, stderr=None):
    """EXF_GETFFIELD_START( useYearlyFields, pkg_name, fld_name, fld_period, fld_startdate1, fld_startdate2,
    fld_start_time, errCount, myThid )   @63cdc0b pkg/exf/exf_getffield_start.F:3-148

    C     | o Get the start time (in seconds) of a forcing field from its start date

    Host routine (calendar integers). Returns (fld_start_time, errCount). `float(...)` REAL*4 literals: `0.` exact."""
    fld_period = np.float64(fld_period)
    msgs = [] if stderr is None else stderr
    if fld_start_time == UNSET_RL:                                             # :63-64
        fld_start_time = np.float64(0.)
    elif useCAL:                                                               # :65-77
        msgs.append(f"S/R EXF_GETFFIELD_START: start-time for {pkg_name}-field \"{fld_name}\" = "
                    f"{fld_name}StartTime is computed (useCAL) from startdate1 & date2 and cannot be set "
                    f"(in data.{pkg_name})")
        errCount = errCount + 1
    if useCAL and (fld_period > 0. or (fld_period == -1. and not useYearlyFields)):   # :80-81
        date_array = cal_FullDate(fld_startdate1, fld_startdate2, cal=cal, stderr=stderr)   # :83-84
        if useYearlyFields:                                                    # :85-92
            yearStartDate = [int(np.trunc(np.float64(date_array[0]) / np.float64(np.float32(10000.)))) * 10000 + 101,
                             0, date_array[2], date_array[3]]
            difftime = cal_TimePassed(yearStartDate, date_array, cal=cal)
            fld_start_time = cal_ToSeconds(difftime, cal=cal)
        else:
            gcm_startdate = cal_GetDate(nIter0, startTime, cal=cal)            # :99
            difftime = cal_TimePassed(gcm_startdate, date_array, cal=cal)      # :100-101
            fld_start_time = cal_ToSeconds(difftime, cal=cal)                  # :102
            fld_start_time = np.float64(startTime) + fld_start_time            # :103
    elif not useCAL:                                                           # :106-145
        if (fld_startdate1 != 0 or fld_startdate2 != 0) and fld_period > 0.:
            msgs.append(f"S/R EXF_GETFFIELD_START: start-date for {pkg_name}-field \"{fld_name}\" is not allowed "
                        "when pkg/cal is not used (useCAL=F)")
            errCount = errCount + 1
        if fld_period < 0.:
            msgs.append(f"S/R EXF_GETFFIELD_START: Invalid record period for {pkg_name}-field \"{fld_name}\"")
            errCount = errCount + 1
    return fld_start_time, errCount
