"""CTRL_GET_GEN_REC   @63cdc0b pkg/ctrl/ctrl_get_gen_rec.F:3-228 (host-side record bookkeeping)"""

from mitjax.config.fortran import real4

_HALF = real4("0.5")      # ctrl_get_gen_rec.F:175 `0.5*deltaTClock`: REAL*4 0.5 (exact)


def ctrl_get_gen_rec(xx_genstartdate, xx_genperiod, myTime, myIter, *, useCAL, startTime, deltaTClock,
                     externForcingCycle, cal=None, **_):
    """CTRL_GET_GEN_REC( xx_genstartdate, xx_genperiod, fac, first, changed, count0, count1, myTime, myIter,
                         myThid )

    C     o Get flags, counters, and the linear interpolation factor for a
    C       given control vector contribution.

    Returns (fac, first, changed, count0, count1). Host floats: which records to read is set-up/driver logic on the
    model clock, not a differentiated value. Lane M4ADCOL (1D_ocean_ice_column/code_ad, ALLOW_CAL): the useCAL arm
    (:79-166) with `cal` (cal.h) on host values of myTime / myIter (`ctrl_get_gen_rec_cal`; a traced clock reads
    the per-step table `gen_rec_table` builds from it). The GET_PERIODIC_INTERVAL branch (:186-197, a periodic control
    without pkg/cal) is not ported (period 0 in R5). The debugLevel >= debLevC printout (:204-225) is not reached
    (debugLevel = debLevA, set_defaults.F:245)."""
    shiftRec = 0                                                       # :78
    if useCAL:                                                         # :79
        return ctrl_get_gen_rec_cal(xx_genstartdate, xx_genperiod, myTime, myIter, cal=cal,
                                    startTime=startTime, deltaTClock=deltaTClock)
    myRelTime = myTime - startTime                                     # :174
    first = myRelTime < _HALF * deltaTClock                            # :175 (a traced bool in a scan step)
    if xx_genperiod == 0.0 or externForcingCycle == 0.0:               # :176-177
        changed = False                                                # :180
        count0 = 1                                                     # :181
        count1 = 1                                                     # :182
        fac = 1.0                                                      # :183  1. _d 0
    else:
        raise NotImplementedError("CTRL_GET_GEN_REC: periodic control (GET_PERIODIC_INTERVAL, :186-197) "
                                  "not ported")
    del shiftRec
    return fac, first, changed, count0, count1


def ctrl_get_gen_rec_cal(xx_genstartdate, xx_genperiod, myTime, myIter, *, cal, startTime, deltaTClock):
    """The useCAL arm of CTRL_GET_GEN_REC (ctrl_get_gen_rec.F:80-166, #ifdef ALLOW_CAL) on host values (myTime a
    float, myIter an int). Returns (fac, first, changed, count0, count1)."""
    import numpy as np
    from mitjax.pkg.cal.cal_addtime import cal_AddTime
    from mitjax.pkg.cal.cal_getdate import cal_GetDate
    from mitjax.pkg.cal.cal_timeinterval import cal_TimeInterval
    from mitjax.pkg.cal.cal_timepassed import cal_TimePassed
    from mitjax.pkg.cal.cal_toseconds import cal_ToSeconds
    myTime = np.float64(myTime)
    fldstartdate = list(xx_genstartdate)                               # :85-89 cal_CopyDate
    fldperiod = np.float64(xx_genperiod)                               # :81, :90
    if xx_genperiod == -12.0:                                          # :92 (-12. _d 0)
        raise NotImplementedError("CTRL_GET_GEN_REC: xx_genperiod = -12 (cal_GetMonthsRec, :95-99) not ported")
    if fldperiod == 0.0:                                               # :100
        first = bool((myTime - cal.modelStart) < _HALF * cal.modelStep)    # :104
        changed = False                                                # :105
        fac = 1.0                                                      # :106  1. _d 0
        count0 = 1                                                     # :107
        count1 = count0                                                # :108
        return fac, first, changed, count0, count1
    mydate = cal_GetDate(myIter, myTime, cal=cal)                      # :112
    difftime = cal_TimePassed(fldstartdate, cal.modelStartDate, cal=cal)   # :115-116
    fldsecs = np.float64(cal_ToSeconds(difftime, cal=cal))             # :117
    shiftRec = int(fldsecs / fldperiod)                                # :120
    difftime = cal_TimePassed(fldstartdate, mydate, cal=cal)           # :123-124
    fldsecs = np.float64(cal_ToSeconds(difftime, cal=cal))             # :125
    fldsecs = int((fldsecs + _HALF) / fldperiod) * fldperiod           # :126
    fldcount = int((fldsecs + _HALF) / fldperiod) + 1                  # :127
    first = bool((myTime - np.float64(startTime)) < _HALF * np.float64(deltaTClock))   # :130
    if first:                                                          # :132-133
        changed = False
    else:
        previousdate = cal_GetDate(myIter - 1, myTime - cal.modelStep, cal=cal)   # :135-136
        difftime = cal_TimePassed(fldstartdate, previousdate, cal=cal)            # :138-139
        prevfldsecs = np.float64(cal_ToSeconds(difftime, cal=cal))                # :140
        prevfldsecs = int((prevfldsecs + _HALF) / fldperiod) * fldperiod          # :141
        prevfldcount = int((prevfldsecs + _HALF) / fldperiod) + 1                 # :142
        changed = fldcount != prevfldcount                                        # :144-148
    count0 = fldcount - shiftRec                                       # :151
    count1 = count0 + 1                                                # :152
    difftime = cal_TimeInterval(fldsecs, "secs", cal=cal)              # :154
    flddate = cal_AddTime(fldstartdate, difftime, cal=cal)             # :155
    difftime = cal_TimePassed(flddate, mydate, cal=cal)                # :156
    fldsecs = np.float64(cal_ToSeconds(difftime, cal=cal))             # :157
    fac = float(np.float64(1.0) - fldsecs / fldperiod)                 # :162  1. _d 0 - fldsecs/fldperiod
    return fac, first, changed, count0, count1


def gen_rec_table(xx_genstartdate, xx_genperiod, *, cal, startTime, deltaTClock, nIter0, nTimeSteps):
    """The host values of the useCAL arm for the steps iloop = 1..nTimeSteps (myIter = nIter0 + iloop - 1, myTime =
    startTime + deltaTClock*(iloop - 1), forward_step.F:429-430): numpy arrays (fac, first, changed, count0,
    count1) indexed by iloop - 1, read by a traced clock (a scan step) in CTRL_GET_GEN."""
    import numpy as np
    rows = [ctrl_get_gen_rec_cal(xx_genstartdate, xx_genperiod,
                                 np.float64(startTime) + np.float64(deltaTClock) * np.float64(it - 1),
                                 nIter0 + (it - 1), cal=cal, startTime=startTime, deltaTClock=deltaTClock)
            for it in range(1, nTimeSteps + 1)]
    return tuple(np.asarray([r[c] for r in rows]) for c in range(5))
