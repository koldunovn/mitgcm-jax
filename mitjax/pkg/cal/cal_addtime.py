"""CAL_ADDTIME: pkg/cal/cal_addtime.F @63cdc0b."""

from mitjax.pkg.cal.cal_convdate import cal_ConvDate
from mitjax.pkg.cal.cal_h import idiv, imod, nMonthYear
from mitjax.pkg.cal.cal_isleap import cal_IsLeap


def cal_AddTime(date, interval, *, cal):
    """CAL_ADDTIME( date, interval, added, myThid )   @63cdc0b pkg/cal/cal_addtime.F:3-236

    C     Add a time interval either to a time or a date.

    Returns `added` (4-list)."""
    if cal.cal_setStatus < 1:                                                  # :65-76
        raise RuntimeError("CAL_ADDTIME: called too early\nABNORMAL END: S/R CAL_ADDTIME")
    if interval[3] != -1:                                                      # :78-82  ierr = 601
        raise RuntimeError("cal_AddTime: error 601 (interval is not a time interval)\n stopped in cal_AddTime.")
    spd, sph, spm, mph = cal.secondsPerDay, cal.secondsPerHour, cal.secondsPerMinute, cal.minutesPerHour
    date_1 = 0                                                                 # :84-86
    date_2 = 0
    fac = 1
    if date[3] == -1:                                                          # :88-108
        if date[0] >= 0:
            date_1 = date[0]
            date_2 = date[1]
            intv_1 = interval[0]
            intv_2 = interval[1]
        else:
            if interval[0] < 0:
                date_1 = -date[0]
                date_2 = -date[1]
                intv_1 = -interval[0]
                intv_2 = -interval[1]
                fac = -1
            else:
                date_1 = interval[0]
                date_2 = interval[1]
                intv_1 = date[0]
                intv_2 = date[1]
                fac = 1
    else:                                                                      # :109-118
        if interval[0] >= 0:
            intv_1 = interval[0]
            intv_2 = interval[1]
        else:
            intv_1 = -interval[0]
            intv_2 = -interval[1]
            fac = -1
    intsecs = fac*(idiv(intv_2, 10000)*sph + (imod(idiv(intv_2, 100), 100)*spm + imod(intv_2, 100)))   # :120-122
    if date[3] == -1:                                                          # :124-140
        datesecs = idiv(date_2, 10000)*sph + imod(idiv(date_2, 100), 100)*spm + imod(date_2, 100)
        date_1 = date_1 + intv_1
        nsecs = datesecs + intsecs
        if date_1 > 0 and nsecs < 0:
            date_1 = date_1 - 1
            nsecs = nsecs + spd
        nsecs = fac*nsecs
        yi = 0
        mi = 0
        di = fac*date_1
        li = 0
        wi = -1
    else:
        yi, mi, di, si, li, wi = cal_ConvDate(date, cal=cal)                   # :142
        if interval[0] >= 0 and interval[1] >= 0:                              # :143-144
            nsecs = si + intsecs                                               # :145
            ndays = interval[0] + idiv(nsecs, spd)                             # :146
            nsecs = imod(nsecs, spd)                                           # :147
            ndays_left = ndays                                                 # :164
            if cal.usingGregorianCalendar:                                     # :167-173
                if mi == 2 and di == 29 and ndays_left > 1:
                    mi = 3
                    di = 1
                    ndays_left = ndays_left - 1
            days_in_year = cal.nDaysNoLeap                                     # :176-179
            if (mi > 2 and cal_IsLeap(yi+1, cal=cal) == 2) or (mi <= 2 and cal_IsLeap(yi, cal=cal) == 2):
                days_in_year = cal.nDaysLeap
            while ndays_left >= days_in_year:                                  # :180-187
                ndays_left = ndays_left - days_in_year
                yi = yi + 1
                days_in_year = cal.nDaysNoLeap
                if (mi > 2 and cal_IsLeap(yi+1, cal=cal) == 2) or (mi <= 2 and cal_IsLeap(yi, cal=cal) == 2):
                    days_in_year = cal.nDaysLeap
            li = cal_IsLeap(yi, cal=cal)                                       # :188
            for iday in range(1, ndays_left + 1):                              # :191-201
                di = di + 1
                if di > cal.nDayMonth[(mi, li)]:
                    di = 1
                    mi = mi + 1
                switch = idiv(mi-1, nMonthYear)
                yi = yi + switch
                mi = imod(mi-1, nMonthYear) + 1
                if switch == 1:
                    li = cal_IsLeap(yi, cal=cal)
            wi = imod(wi+ndays-1, 7) + 1                                       # :202
        else:                                                                  # :204-223
            nsecs = si + intsecs
            if nsecs >= 0:
                ndayssub = intv_1
            else:
                nsecs = nsecs + spd
                ndayssub = intv_1 + 1
            for iday in range(1, ndayssub + 1):
                di = di - 1
                if di == 0:
                    mi = imod(mi+10, nMonthYear) + 1
                    switch = idiv(mi, nMonthYear)
                    yi = yi - switch
                    if switch == 1:
                        li = cal_IsLeap(yi, cal=cal)
                    di = cal.nDayMonth[(mi, li)]
            wi = imod(wi+6-imod(ndayssub, 7), 7) + 1
    added = [0, 0, 0, 0]
    added[0] = yi*10000 + mi*100 + di                                          # :227
    hhmmss = idiv(nsecs, spm)                                                  # :228
    added[1] = idiv(hhmmss, mph)*10000 + (imod(fac*hhmmss, mph)*100 + imod(fac*nsecs, spm))*fac   # :229-231
    added[2] = li                                                              # :232
    added[3] = wi                                                              # :233
    return added
