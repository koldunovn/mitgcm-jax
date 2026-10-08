"""MON_STATS_RS   @63cdc0b pkg/monitor/mon_stats_rs.F:8-120"""

import jax.numpy as jnp
import numpy as np

from mitjax.farray import loops_kji
from mitjax.ops.fortran_minmax_host import first_masked, max_chain, min_chain
from mitjax.pkg.monitor.monitor_h import global_max_rl, global_sum_rl_scalar, global_sum_tile_rl, tile_sums


def mon_stats_rs(myNr, arr, *, cfg, ex=None):
    """MON_STATS_RS( myNr, arr, theMin, theMax, theMean, theSD, myThid )

    C     Calculate bare statistics of global array ``\\_RS arr''.

    arr: FArray (1-OLx:sNx+OLx,1-OLy:sNy+OLy,myNr,nSx,nSy). Returns (theMin, theMax, theMean, theSD), host float64.
    Every interior point counts (the `tmpVal.NE.0.` tests are commented out, :54, :60, :101). Sums: per-tile chains
    in k, j, i order then GLOBAL_SUM_TILE_RL (mitjax/eesupp). `tmpVal=FLOAT(numPnts)` (:79) is a default-REAL
    (REAL*4) conversion assigned to an _RL: exact for the point counts of the M1 grids (< 2**24, checked)."""
    sNx, sNy = cfg.sNx, cfg.sNy
    theMin = np.float64(0.0)                                        # :38-44
    theMax = np.float64(0.0)
    theMean = np.float64(0.0)
    theSD = np.float64(0.0)
    theVar = np.float64(0.0)
    numPnts = 0
    noPnts = True

    k, j, i = loops_kji((1, myNr), (1, sNy), (1, sNx))              # :46-73
    tmpVal = arr[i, j, k]                                           # :53
    every = jnp.ones(tmpVal.shape, bool)
    first, found = first_masked(tmpVal, every)                      # :55-59
    if found:
        theMin = first
        theMax = first
        noPnts = False
    theMin = min_chain(theMin, tmpVal, p="a")                      # :61
    theMax = max_chain(theMax, tmpVal, p="a")                      # :62
    tileMean = tile_sums(tmpVal)                                    # :63
    tileVar = tile_sums(tmpVal*tmpVal)                              # :64
    numPnts = numPnts + int(np.prod(tmpVal.shape))                  # :65

    theMean = global_sum_tile_rl(tileMean, ex)                      # :77
    theVar = global_sum_tile_rl(tileVar, ex)                        # :78
    if numPnts >= 2 ** 24:
        raise ValueError("MON_STATS_RS: FLOAT(numPnts) (REAL*4) is not exact above 2**24 points")
    tmpVal_s = np.float64(np.float32(numPnts))                      # :79
    tmpVal_s = global_sum_rl_scalar(tmpVal_s, ex)                   # :80
    numPnts = int(np.rint(tmpVal_s))                                # :81 (an integer value: NINT is exact)

    if tmpVal_s > 0.0:                                              # :83-117
        rNumPnts = np.float64(1.0)/tmpVal_s
        theMean = theMean*rNumPnts
        theVar = theVar*rNumPnts
        if noPnts:
            theMin = theMean
        theMin = -theMin
        theMin = global_max_rl(theMin, ex)
        theMin = -theMin
        if noPnts:
            theMax = theMean
        theMax = global_max_rl(theMax, ex)

        tileSD = tile_sums((tmpVal-theMean)*(tmpVal-theMean))       # :94-110
        theSD = global_sum_tile_rl(tileSD, ex)                      # :113
        theSD = np.sqrt(theSD*rNumPnts)                             # :115

    return theMin, theMax, theMean, theSD
