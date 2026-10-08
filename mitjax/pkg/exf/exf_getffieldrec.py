"""EXF_GETFFIELDREC: pkg/exf/exf_getffieldrec.F @63cdc0b (host: record numbers and the interpolation weight)."""

import math

import numpy as np

from mitjax.model.src.external_fields_load import get_periodic_interval

halfRL = np.float64(0.5)                    # eesupp/inc/EEPARAMS.h:73  halfRL = 0.5 _d 0


def _mod_r8(a, p):
    """Fortran MOD of REAL*8 arguments (fmod: exact remainder, the sign of a)."""
    return np.float64(np.fmod(np.float64(a), np.float64(p)))


def exf_GetFFieldRec(fldStartTime, fldPeriod, fldRepeatCycle, fldName, usefldyearlyfields, myTime, myIter, *,
                     useCAL, cal, nIter0, deltaTClock):
    """EXF_GETFFIELDREC( fldStartTime, fldPeriod, fldRepeatCycle, fldName, usefldyearlyfields, fac, first,
    changed, count0, count1, year0, year1, myTime, myIter, myThid )   @63cdc0b pkg/exf/exf_getffieldrec.F:3-269

    C     | o Get flags, counters, and the linear interpolation factor for a given field.

    Host routine: concrete myTime (float64) and myIter. Returns (fac, first, changed, count0, count1, year0, year1)
    with fac a float64. REAL*4 literals: `0.5`, `1.` exact. Ported: the useCAL branch with fldPeriod /= 0 and
    usefldyearlyfields = .FALSE. (:90-150, :195-198) and fldPeriod = 0 (:97-109); without useCAL (or without
    ALLOW_CAL: `IF ( .TRUE. )`, :202-205) the GET_PERIODIC_INTERVAL branch :207-263 (lane M4CS32ICE: the monthly
    records of global_ocean.cs32x15/input.seaice, a build without pkg/cal). Raises: useExfYearlyFields (:151-193)."""
    if not useCAL:                                                             # :200-263
        return _no_cal(fldStartTime, fldPeriod, fldRepeatCycle, fldName, myTime, myIter, nIter0=nIter0,
                       deltaTClock=deltaTClock)
    myTime = np.float64(myTime)
    fldPeriod = np.float64(fldPeriod)
    fldRepeatCycle = np.float64(fldRepeatCycle)
    fldStartTime = np.float64(fldStartTime)
    first = bool((myTime - cal.modelStart) < np.float64(0.5)*cal.modelStep)   # :94
    changed = False                                                            # :95
    year0 = year1 = 0
    if fldPeriod == 0.:                                                        # :97-109
        first = bool((myTime - cal.modelStart) < np.float64(0.5)*cal.modelStep)
        changed = False
        fac = np.float64(1.)
        count0 = 1
        count1 = count0
        year0 = 0
        year1 = year0
        return fac, first, changed, count0, count1, year0, year1
    if usefldyearlyfields:                                                     # :151-193
        raise NotImplementedError("EXF_GetFFieldRec: useExfYearlyFields is not ported")
    fldsectot = myTime - fldStartTime                                          # :116
    if fldRepeatCycle == 0.:                                                   # :119-133
        if fldsectot < 0.:
            raise RuntimeError(f"EXF_GetFFieldRec for field \"{fldName}\": myTime={float(myTime):17.10E} earlier "
                               "than 1rst reccord\nABNORMAL END: S/R EXF_GetFFieldRec")
        count0 = int(math.trunc((fldsectot + np.float64(0.5))/fldPeriod)) + 1  # :130
        count1 = count0 + 1                                                    # :131
        fldsecs = _mod_r8(fldsectot, fldPeriod)                                # :132
    else:                                                                      # :134-146
        if fldsectot < 0.:                                                     # :138-139
            fldsectot = fldsectot + fldRepeatCycle
        fldsecs0 = _mod_r8(fldsectot, fldRepeatCycle)                          # :140
        count0 = int(math.trunc((fldsecs0 + np.float64(0.5))/fldPeriod)) + 1   # :141
        fldsecs1 = _mod_r8(fldsectot + fldPeriod, fldRepeatCycle)              # :142
        count1 = int(math.trunc((fldsecs1 + np.float64(0.5))/fldPeriod)) + 1   # :143
        fldsecs = _mod_r8(fldsecs0, fldPeriod)                                 # :144
    fac = np.float64(1.) - fldsecs/fldPeriod                                   # :149
    if fldsecs - cal.modelStep < 0.:                                           # :196
        changed = True
    return fac, first, changed, count0, count1, year0, year1


def _no_cal(fldStartTime, fldPeriod, fldRepeatCycle, fldName, myTime, myIter, *, nIter0, deltaTClock):
    """exf_getffieldrec.F:202-263, the branch without pkg/cal: (fac, first, changed, count0, count1, year0, year1)."""
    year0 = 0                                                                  # :207
    year1 = 0                                                                  # :208
    fldPeriod = np.float64(fldPeriod)
    if fldPeriod == np.float64(0.):                                            # :210-218
        fac = np.float64(1.)
        first = int(myIter) == int(nIter0)
        changed = False
        count0 = 1
        count1 = 1
        return fac, first, changed, count0, count1, year0, year1
    locTime = np.float64(myTime) - np.float64(fldStartTime) + fldPeriod*halfRL   # :221
    intimeP, intime0, intime1, bWght, aWght = get_periodic_interval(             # :222-225
        fldRepeatCycle, fldPeriod, deltaTClock, locTime)
    del aWght
    fac = bWght                                                                # :228
    first = int(myIter) == int(nIter0)                                         # :229
    changed = intime0 != intimeP                                               # :230
    count0 = intime0                                                           # :231
    count1 = intime1                                                           # :232
    if intime0 <= 0:                                                           # :234-259
        raise RuntimeError(f"EXF_GetFFieldRec: for field \"{fldName}\" @ Iter={int(myIter)} , "
                           f"myTime={float(myTime):17.10E}\nEXF_GetFFieldRec: Reccord number \"intime0\" not "
                           "valid ; possible cause:\nEXF_GetFFieldRec:  myTime earlier than field-StartTime="
                           f"{float(fldStartTime):18.10E}\nABNORMAL END: S/R EXF_GetFFieldRec")
    return fac, first, changed, count0, count1, year0, year1
