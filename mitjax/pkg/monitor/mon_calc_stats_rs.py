"""MON_CALC_STATS_RS   @63cdc0b pkg/monitor/mon_calc_stats_rs.F:8-158"""

from mitjax.pkg.monitor.mon_calc_stats_rl import mon_calc_stats_rl


def mon_calc_stats_rs(myNr, arr, arrhFac, arrMask, arrArea, arrDr, *, cfg, ex=None):
    """MON_CALC_STATS_RS( myNr, arr, arrhFac, arrMask, arrArea, arrDr,
                          theMin, theMax, theMean, theSD, theDel2, theVol, myThid )

    C     Calculate statistics of global array ``\\_RS arr''.
    C     account for volume and mask

    The file differs from mon_calc_stats_rl.F only in the declared type of arr (`_RS`, :23; `diff` of the two files
    @63cdc0b: the routine name, comment and that declaration). `_RS` is REAL*8 in every build of this project (no
    -use_real4), so the statements are those of MON_CALC_STATS_RL, executed by that function."""
    return mon_calc_stats_rl(myNr, arr, arrhFac, arrMask, arrArea, arrDr, cfg=cfg, ex=ex)
