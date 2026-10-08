"""CTRL_INIT_REC   @63cdc0b pkg/ctrl/ctrl_init_rec.F:5-128 (host-side set-up)"""


def ctrl_init_rec(fldname, fldstartdate1, fldstartdate2, fldperiod, nfac, *, useCAL, startTime, endTime, cal=None):
    """CTRL_INIT_REC( fldname, fldstartdate1, fldstartdate2, fldperiod, nfac,
                      fldstartdate, diffrec, startrec, endrec, myThid )

    Returns (fldstartdate[4], diffrec, startrec, endrec). The debugLevel >= debLevB messages (:71-77, :116-126)
    are not printed (debugLevel = debLevA in this AD build, set_defaults.F:245). Lane M4ADCOL
    (1D_ocean_ice_column/code_ad, ALLOW_CAL): the useCAL arm (:92-110) with `cal` (cal.h after CAL_INIT_FIXED;
    host integer date arithmetic of the ported pkg/cal routines). Its `fldstartdate` is CAL_FULLDATE's (:96-97)."""
    fldstartdate = [0, 0, 0, 0]                                        # :80-82
    startrec = 0                                                       # :83-85
    endrec = 0
    diffrec = 0
    if fldperiod == -12.0:                                             # :86 (REAL*4 -12. is exact)
        startrec = 1
        endrec = 12 * nfac
    elif fldperiod == 0.0:                                             # :89
        startrec = 1
        endrec = 1 * nfac
    elif useCAL:                                                       # :92-93 (ALLOW_CAL)
        import numpy as np
        from mitjax.pkg.cal.cal_fulldate import cal_FullDate
        from mitjax.pkg.cal.cal_timepassed import cal_TimePassed
        from mitjax.pkg.cal.cal_toseconds import cal_ToSeconds
        modelBaseDate = cal_FullDate(cal.startdate_1, cal.startdate_2, cal=cal)              # :94-95
        fldstartdate = cal_FullDate(fldstartdate1, fldstartdate2, cal=cal)                    # :96-97
        difftime = cal_TimePassed(modelBaseDate, fldstartdate, cal=cal)                       # :98-99
        diffsecs = np.float64(cal_ToSeconds(difftime, cal=cal))                               # :100
        startrec = int((cal.modelStart - diffsecs) / np.float64(fldperiod)) + 1             # :101
        endrec = int((cal.modelEnd - diffsecs + cal.modelStep / 2)                           # :102-103
                     / np.float64(fldperiod)) + 2
        if nfac != 1:                                                  # :104-108
            startrec = (startrec - 1) * nfac + 1
            endrec = endrec * nfac
    else:                                                              # :111-112
        startrec = 1
        endrec = (int((endTime - startTime) / fldperiod) + 1) * nfac
    diffrec = endrec - startrec + 1                                    # :114
    return fldstartdate, diffrec, startrec, endrec
