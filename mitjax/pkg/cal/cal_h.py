"""cal.h of pkg/cal @63cdc0b (pkg/cal/cal.h): the calendar common blocks, and the Fortran INTEGER operations the
calendar routines use.

The calendar is host-side integer bookkeeping (dates as YYYYMMDD / HHMMSS integer pairs, set once at initialisation
and queried by EXF with concrete model times), so `Cal` is a plain mutable Python object, as the Fortran common
blocks: CAL_READPARMS and CAL_SET write it, every other routine reads it. Its REAL members (modelStart, modelEnd,
modelStep, /CALENDAR_RL/) are numpy float64.

Fortran INTEGER division truncates toward zero and MOD takes the sign of the dividend; Python's `//` and `%` floor.
`idiv` and `imod` are the Fortran operations on Python ints (every cal routine uses them, never `//` or `%`).
"""

import numpy as np

nMonthYear = 12            # cal.h:48-49  PARAMETER ( nMonthYear = 12 )


def idiv(a, b):
    """Fortran INTEGER a/b: the quotient truncated toward zero."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def imod(a, b):
    """Fortran MOD(a, b) of INTEGERs: a - (a/b)*b, the sign of a."""
    return a - idiv(a, b) * b


class Cal:
    """The calendar common blocks of cal.h:51-129 (/CALENDAR_RL/, /CALENDAR_I/, /CALENDAR_L/, /CALENDAR_C/).
    Arrays keep the Fortran index: `nDayMonth[(m, l)]` for nDayMonth(m,l), `refDate[1..4]` as a 5-list with an
    unused element 0 (dates are 4-lists `d` with d[0] = date(1), ... in the routines' arguments)."""

    def __init__(self):
        # /CALENDAR_RL/
        self.modelStart = np.float64(0.)
        self.modelEnd = np.float64(0.)
        self.modelStep = np.float64(0.)
        # /CALENDAR_I/
        self.refDate = [0, 0, 0, 0]
        self.nDayMonth = {}
        self.nDaysNoLeap = 0
        self.nDaysLeap = 0
        self.nMaxDayMonth = 0
        self.hoursPerDay = 0
        self.minutesPerDay = 0
        self.minutesPerHour = 0
        self.secondsPerDay = 0
        self.secondsPerHour = 0
        self.secondsPerMinute = 0
        self.modelStartDate = [0, 0, 0, 0]
        self.modelEndDate = [0, 0, 0, 0]
        self.modelIter0 = 0
        self.modelIterEnd = 0
        self.modelIntSteps = 0
        self.cal_setStatus = 0
        self.startdate_1 = 0
        self.startdate_2 = 0
        # /CALENDAR_L/
        self.calendarDumps = False
        self.usingModelCalendar = False
        self.usingNoLeapYearCal = False
        self.usingJulianCalendar = False
        self.usingGregorianCalendar = False
        # /CALENDAR_C/
        self.theCalendar = " "
        self.dayOfWeek = [" "] * 7
