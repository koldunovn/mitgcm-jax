"""MON_PRINTSTATS_RS   @63cdc0b pkg/monitor/mon_printstats_rs.F:8-46"""

from mitjax.pkg.monitor.mon_out import mon_out_rl
from mitjax.pkg.monitor.mon_stats_rs import mon_stats_rs
from mitjax.pkg.monitor.monitor_h import mon_foot_max, mon_foot_mean, mon_foot_min, mon_foot_sd


def mon_printstats_rs(myNr, arr, arrName, *, cfg, mon, ex=None):
    """MON_PRINTSTATS_RS( myNr, arr, arrName, myThid )

    C     Prints to STDOUT the bare statistics of global array "\\_RS arr"
    C     with label "arrName".

    Called by INI_GRID (model/src/ini_grid.F:209-226) and INI_CORI (ini_cori.F:208-210) for the grid fields."""
    theMin, theMax, theMean, theSD = mon_stats_rs(myNr, arr, cfg=cfg, ex=ex)    # :35-38

    mon_out_rl(arrName, theMax, mon_foot_max, cfg=cfg, mon=mon)                # :40-43
    mon_out_rl(arrName, theMin, mon_foot_min, cfg=cfg, mon=mon)
    mon_out_rl(arrName, theMean, mon_foot_mean, cfg=cfg, mon=mon)
    mon_out_rl(arrName, theSD, mon_foot_sd, cfg=cfg, mon=mon)
