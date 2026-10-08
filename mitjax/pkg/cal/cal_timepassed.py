"""CAL_TIMEPASSED: pkg/cal/cal_timepassed.F @63cdc0b."""

from mitjax.pkg.cal.cal_convdate import cal_ConvDate
from mitjax.pkg.cal.cal_h import idiv, imod
from mitjax.pkg.cal.cal_isleap import cal_IsLeap


def cal_TimePassed(initialdate, finaldate, *, cal):
    """CAL_TIMEPASSED( initialdate, finaldate, numdays, myThid )   @63cdc0b pkg/cal/cal_timepassed.F:3-189

    C     Calculate the time that passes between initialdate and
    C     finaldate.

    Returns `numdays` (4-list: days, hhmmss, 0, -1)."""
    if cal.cal_setStatus < 1:                                                  # :66-77
        raise RuntimeError("CAL_TIMEPASSED: called too early\nABNORMAL END: S/R CAL_TIMEPASSED")
    nothingtodo = False                                                        # :79
    numdays = [0, 0, 0, -1]                                                    # :82-85
    spd, spm, mph = cal.secondsPerDay, cal.secondsPerMinute, cal.minutesPerHour
    if (initialdate[3] > 0) == (finaldate[3] > 0):                             # :87-88
        caldates = (initialdate[3] > 0) and (finaldate[3] > 0)                 # :90-91
        if initialdate[0] == finaldate[0]:                                     # :94-106
            if initialdate[1] == finaldate[1]:
                nothingtodo = True
            elif initialdate[1] > finaldate[1]:
                swap = True
            else:
                swap = False
        elif initialdate[0] > finaldate[0]:
            swap = True
        else:
            swap = False
        if not nothingtodo:                                                    # :108
            if swap:                                                           # :110-116
                yi, mi, di, si, li, wi = cal_ConvDate(finaldate, cal=cal)
                yf, mf, df, sf, lf, wf = cal_ConvDate(initialdate, cal=cal)
            else:
                yi, mi, di, si, li, wi = cal_ConvDate(initialdate, cal=cal)
                yf, mf, df, sf, lf, wf = cal_ConvDate(finaldate, cal=cal)
            if not caldates:                                                   # :119-127
                ndays = df - di
                nsecs = sf - si
                if nsecs < 0:
                    nsecs = nsecs + spd
                    ndays = ndays - 1
                ndays = ndays + idiv(nsecs, spd)
                nsecs = imod(nsecs, spd)
            else:                                                              # :128-162
                si = si + (di-1)*spd
                sf = sf + (df-1)*spd
                cdi = 0
                for imon in range(1, imod(mi-1, 12) + 1):
                    cdi = cdi + cal.nDayMonth[(imon, li)]
                csi = si
                cdf = 0
                for imon in range(1, imod(mf-1, 12) + 1):
                    cdf = cdf + cal.nDayMonth[(imon, lf)]
                csf = sf
                if yi == yf:                                                   # :142-150
                    ndays = (cdf + idiv(csf, spd)) - (cdi + idiv(csi, spd))
                    nsecs = (csf - idiv(csf, spd)*spd) - (csi - idiv(csi, spd)*spd)
                    if nsecs < 0:
                        nsecs = nsecs + spd
                        ndays = ndays - 1
                else:                                                          # :151-161
                    ndays = (cal.nDaysNoLeap - 1) + cal_IsLeap(yi, cal=cal) - cdi - cal.nDayMonth[(mi, li)]
                    for iyr in range(yi+1, yf):
                        ndays = ndays + (cal.nDaysNoLeap - 1) + cal_IsLeap(iyr, cal=cal)
                    ndays = ndays + cdf
                    csi = cal.nDayMonth[(mi, li)]*spd - csi
                    nsecs = csi + csf
            numdays[0] = ndays + idiv(nsecs, spd)                              # :165-170
            nsecs = imod(nsecs, spd)
            hhmmss = idiv(nsecs, spm)
            numdays[1] = idiv(hhmmss, mph)*10000 + imod(hhmmss, mph)*100 + imod(nsecs, spm)
            if swap:                                                           # :171-174
                numdays[0] = -numdays[0]
                numdays[1] = -numdays[1]
    else:                                                                      # :180-185  ierr = 501
        raise RuntimeError("cal_TimePassed: error 501 (one date is a calendar date, the other a time interval)\n"
                           " stopped in cal_TimePassed")
    return numdays
