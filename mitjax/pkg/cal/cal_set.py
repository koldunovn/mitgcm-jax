"""CAL_SET: pkg/cal/cal_set.F @63cdc0b."""

import math

import numpy as np

from mitjax.config.fortran import real4

from mitjax.pkg.cal.cal_addtime import cal_AddTime
from mitjax.pkg.cal.cal_fulldate import cal_FullDate
from mitjax.pkg.cal.cal_h import idiv, imod, nMonthYear
from mitjax.pkg.cal.cal_timeinterval import cal_TimeInterval


def _nint(x):
    """Fortran NINT of a REAL*8 (round half away from zero)."""
    x = float(x)
    n = math.trunc(x)
    if x - n >= 0.5:
        n += 1
    elif x - n <= -0.5:
        n -= 1
    return int(n)


def cal_Set(modstart, modend, modstep, moditerini, moditerend, modintsteps, *, cal, stderr=None):
    """CAL_SET( modstart, modend, modstep, moditerini, moditerend, modintsteps, myThid )
    @63cdc0b pkg/cal/cal_set.F:3-253

    C     Given the parameters specified by the user in data.cal determine
    C     the calendar settings and the start/end dates of the integration.

    Writes the common block `cal` (as the Fortran) and returns it."""
    cal.usingNoLeapYearCal = False                                             # :85-88
    cal.usingGregorianCalendar = False
    cal.usingModelCalendar = False
    cal.usingJulianCalendar = False
    cal.hoursPerDay = 24                                                       # :91-96
    cal.minutesPerHour = 60
    cal.minutesPerDay = cal.minutesPerHour*cal.hoursPerDay
    cal.secondsPerMinute = 60
    cal.secondsPerHour = cal.secondsPerMinute*cal.minutesPerHour
    cal.secondsPerDay = cal.secondsPerMinute*cal.minutesPerDay
    theCalendar = cal.theCalendar.rstrip()
    if theCalendar == "gregorian":                                             # :99-115
        cal.usingGregorianCalendar = True
    elif theCalendar == "noLeapYear":
        cal.usingNoLeapYearCal = True
    elif theCalendar == "model":
        cal.usingModelCalendar = True
    else:                                                                      # ierr = 101
        raise RuntimeError(f"cal_Set: error 101 (calendar type {cal.theCalendar!r} not implemented)")
    if cal.usingGregorianCalendar or cal.usingNoLeapYearCal:                   # :119-158
        cal.refDate = [15821015, 0, 1, 1]
        cal.nDaysNoLeap = 365
        cal.nDaysLeap = 366
        cal.nMaxDayMonth = 31
        k = 2773                                                               # :140-148
        for i in range(1, nMonthYear + 1):
            j = imod(k, 2)
            k = idiv(k-j, 2)
            cal.nDayMonth[(i, 1)] = 30+j
            cal.nDayMonth[(i, 2)] = 30+j
        cal.nDayMonth[(2, 1)] = 28
        cal.nDayMonth[(2, 2)] = 29
        cal.dayOfWeek = ["FRI", "SAT", "SUN", "MON", "TUE", "WED", "THU"]       # :151-157
    if cal.usingModelCalendar:                                                 # :160-186
        cal.refDate = [101, 0, 1, 1]
        cal.nDaysNoLeap = 360
        cal.nDaysLeap = 360
        cal.nMaxDayMonth = 30
        for i in range(1, nMonthYear + 1):
            cal.nDayMonth[(i, 1)] = 30
            cal.nDayMonth[(i, 2)] = 30
        cal.dayOfWeek = ["MD1", "MD2", "MD3", "MD4", "MD5", "MD6", "MD7"]
    cal.cal_setStatus = 1                                                      # :189
    cal.modelStart = np.float64(modstart)                                      # :192-197
    cal.modelEnd = np.float64(modend)
    cal.modelStep = np.float64(modstep)
    cal.modelIter0 = int(moditerini)
    cal.modelIterEnd = int(moditerend)
    cal.modelIntSteps = int(modintsteps)
    if cal.modelStep <= 0.:                                                    # :201-205  ierr = 102
        raise RuntimeError("cal_Set: error 102\n stopped in cal_Set.")
    if cal.modelStep < 1.:                                                     # :206-210  ierr = 103
        raise RuntimeError("cal_Set: error 103\n stopped in cal_Set.")
    if abs(cal.modelStep - np.float64(_nint(cal.modelStep))) > real4("0.000001"):     # :211-217 (REAL*4 literal)
        raise RuntimeError("cal_Set: error 104\n stopped in cal_Set.")
    else:
        cal.modelStep = np.float64(np.float32(_nint(cal.modelStep)))           # :216 FLOAT(NINT(modelStep)), REAL*4
    cal.cal_setStatus = 2                                                      # :220
    modelBaseDate = cal_FullDate(cal.startdate_1, cal.startdate_2, cal=cal, stderr=stderr)   # :223-224
    runtimesecs = np.float64(cal.modelIntSteps)*cal.modelStep                  # :230
    iterinisecs = cal.modelStart                                               # :235
    iterinitime = cal_TimeInterval(iterinisecs, "secs", cal=cal)               # :236
    cal.modelStartDate = cal_AddTime(modelBaseDate, iterinitime, cal=cal)      # :237-238
    timediff = cal_TimeInterval(runtimesecs, "secs", cal=cal)                  # :240
    cal.modelEndDate = cal_AddTime(cal.modelStartDate, timediff, cal=cal)      # :241-242
    cal.cal_setStatus = 3                                                      # :245
    return cal
