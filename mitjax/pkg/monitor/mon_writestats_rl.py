"""MON_WRITESTATS_RL   @63cdc0b pkg/monitor/mon_writestats_rl.F:8-62"""

from mitjax.pkg.monitor.mon_calc_stats_rl import mon_calc_stats_rl
from mitjax.pkg.monitor.mon_out import mon_out_rl
from mitjax.pkg.monitor.monitor_h import mon_foot_del2, mon_foot_max, mon_foot_mean, mon_foot_min, mon_foot_sd


def mon_writestats_rl(myNr, arr, arrName, arrhFac, arrMask, arrArea, arrDr, arrStats, *, cfg, mon, ex=None):
    """MON_WRITESTATS_RL( myNr, arr, arrName, arrhFac, arrMask, arrArea, arrDr, arrStats, myThid )

    C     Compute the statistics of global array "\\_RL arr" (account for
    C     volume and mask) and write them to STDOUT with label "arrName".
    C     arrStats :: statistics of the global array (min, max ...)

    arrStats (O, `_RL arrStats(*)`) is a Python list of at least 6 entries; the updated list is returned."""
    theMin, theMax, theMean, theSD, theDel2, theVol = mon_calc_stats_rl(      # :42-45
        myNr, arr, arrhFac, arrMask, arrArea, arrDr, cfg=cfg, ex=ex)

    arrStats = list(arrStats)                                                 # :47-52
    arrStats[0] = theMin
    arrStats[1] = theMax
    arrStats[2] = theMean
    arrStats[3] = theSD
    arrStats[4] = theDel2
    arrStats[5] = theVol

    mon_out_rl(arrName, theMax, mon_foot_max, cfg=cfg, mon=mon)              # :54-58
    mon_out_rl(arrName, theMin, mon_foot_min, cfg=cfg, mon=mon)
    mon_out_rl(arrName, theMean, mon_foot_mean, cfg=cfg, mon=mon)
    mon_out_rl(arrName, theSD, mon_foot_sd, cfg=cfg, mon=mon)
    mon_out_rl(arrName, theDel2, mon_foot_del2, cfg=cfg, mon=mon)
    return arrStats
