"""MON_SOLUTION   @63cdc0b pkg/monitor/mon_solution.F:8-66"""

from mitjax.eesupp.print import ERROR_MESSAGE_UNIT as errorMessageUnit
from mitjax.eesupp.print import SQUEEZE_RIGHT, print_message
from mitjax.io.fortran_format import fortran_write
from mitjax.pkg.monitor.mon_calc_stats_rl import mon_calc_stats_rl


class MonSolutionStop(RuntimeError):
    """STOP 'ABNORMAL END: S/R MON_SOLUTION, stops due to EXTREME Pot.Temp' (mon_solution.F:61-62)."""


def mon_solution(statsTemp, myTime, myIter, *, cfg, grid, state, mon, ex=None):
    """MON_SOLUTION( statsTemp, myTime, myIter, myThid )

    C     Checks that the solutions is within bounds

    statsTemp: the 6 statistics of theta from MONITOR (min, max, ...). When statsTemp(1) > statsTemp(2) the theta
    statistics are recomputed (:41-44). Out of range: the three messages go to errorMessageUnit and the run stops
    (ALL_PROC_DIE + STOP), here a MonSolutionStop exception carrying the Fortran message. PRINT_MESSAGE's copy of an
    error-unit record to unit 0 (print.F:148-157) is not ported in mitjax/eesupp/print.py and raises there first."""
    if statsTemp[0] <= statsTemp[1]:                                # :35-45
        tMin = statsTemp[0]
        tMax = statsTemp[1]
    else:
        tMin, tMax, tMean, tSD, tDel2, tVol = mon_calc_stats_rl(   # :41-44
            cfg.Nr, state.theta, grid.hFacC, grid.maskInC, grid.rA, grid.drF, cfg=cfg, ex=ex)

    if (tMax - tMin) > mon.monSolutionMaxRange:                     # :47-63
        msgBuf = fortran_write("(A,1P2E11.3)", "SOLUTION IS HEADING OUT OF BOUNDS: tMin,tMax=", tMin, tMax)
        print_message(msgBuf, errorMessageUnit, SQUEEZE_RIGHT, mon.myThid, io=mon.io)
        msgBuf = fortran_write("(2A,1PE11.3,A)", "  exceeds allowed range ", "(monSolutionMaxRange=",
                               mon.monSolutionMaxRange, ")")
        print_message(msgBuf, errorMessageUnit, SQUEEZE_RIGHT, mon.myThid, io=mon.io)
        msgBuf = fortran_write("(A,I10)", "MON_SOLUTION: STOPPING CALCULATION at Iter=", int(myIter))
        print_message(msgBuf, errorMessageUnit, SQUEEZE_RIGHT, mon.myThid, io=mon.io)
        raise MonSolutionStop("ABNORMAL END: S/R MON_SOLUTION, stops due to EXTREME Pot.Temp")
