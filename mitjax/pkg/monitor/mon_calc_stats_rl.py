"""MON_CALC_STATS_RL   @63cdc0b pkg/monitor/mon_calc_stats_rl.F:8-158"""

import jax.numpy as jnp
import numpy as np

from mitjax.farray import loops_kji
from mitjax.ops.fortran_minmax_host import first_masked, max_chain, min_chain
from mitjax.pkg.monitor.monitor_h import global_max_rl, global_sum_tile_rl, tile_sums

# eesupp/inc/EEPARAMS.h:71-72: zeroRL = 0.0 _d 0, oneRL = 1.0 _d 0
oneRL = 1.0
zeroRL = 0.0


def mon_calc_stats_rl(myNr, arr, arrhFac, arrMask, arrArea, arrDr, *, cfg, ex=None):
    """MON_CALC_STATS_RL( myNr, arr, arrhFac, arrMask, arrArea, arrDr,
                          theMin, theMax, theMean, theSD, theDel2, theVol, myThid )

    C     Calculate statistics of global array ``\\_RL arr''.
    C     account for volume and mask

    Arguments (Fortran declarations): arr, arrhFac (1-OLx:sNx+OLx,1-OLy:sNy+OLy,myNr,nSx,nSy); arrMask, arrArea
    (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy); arrDr(myNr), all FArrays. Returns (theMin, theMax, theMean, theSD, theDel2,
    theVol) as host float64.

    Order: every sum of the Fortran is a per-tile chain over k, j, i (`tileX(bi,bj) = tileX(bi,bj) + ...`, :60-109,
    :132-149), then GLOBAL_SUM_TILE_RL; both through mitjax/eesupp (`tile_sums`, `global_sum_tile_rl`). The k loop is
    vectorised for the per-point expressions only (they are independent); the sums keep the k, j, i order. A point
    the Fortran skips (`IF ( tmpMask.GT.0. _d 0 )`) adds +0, which is exact: a chain starting at `0.` is never -0.
    tileDel2 adds two terms per point, `tileDel2 + ddx*ddx + ddy*ddy` (:97), so its chain runs over (ddx*ddx, ddy*ddy)
    pairs in point order. theMin/theMax are one chain over all tiles with the measured operand order of these two
    statements (the new value wins ties and NaN: $MJX_REFERENCE/minmax_sites); the running min starts from the first masked point
    (:70-74). tileVar is formed (:103) but its global sum is
    commented out in the Fortran (:115), so theVar stays 0 (:52, :124); theVar is not an output.
    Host-side finish: the divisions and SQRT of the global values are IEEE float64 on the host (numpy)."""
    sNx, sNy = cfg.sNx, cfg.sNy
    theMin = np.float64(0.0)                                        # :48-56
    theMax = np.float64(0.0)
    theMean = np.float64(0.0)
    theSD = np.float64(0.0)
    theVar = np.float64(0.0)
    theDel2 = np.float64(0.0)
    theVol = np.float64(0.0)
    theNbPt = np.float64(0.0)
    noPnts = True

    k, j, i = loops_kji((1, myNr), (1, sNy), (1, sNx))              # :58-67
    tmpVal = arr[i, j, k]                                           # :68
    tmpMask = arrMask[i, j]*arrhFac[i, j, k]                        # :69
    wet = tmpMask > 0.0                                             # :70, :75
    first, found = first_masked(tmpVal, wet)                        # :70-74
    if found:
        theMin = first
        theMax = first
        noPnts = False
    theMin = min_chain(theMin, tmpVal, wet, p="b")                 # :76
    theMax = max_chain(theMax, tmpVal, wet, p="b")                 # :77
    ddx = arrhFac[i+1, j, k]*arrhFac[i-1, j, k]                     # :87-91
    ddx = jnp.where(ddx > 0.0, (arr[i+1, j, k]-tmpVal)
                    + (arr[i-1, j, k]-tmpVal), ddx)
    ddy = arrhFac[i, j+1, k]*arrhFac[i, j-1, k]                     # :92-96
    ddy = jnp.where(ddy > 0.0, (arr[i, j+1, k]-tmpVal)
                    + (arr[i, j-1, k]-tmpVal), ddy)
    zero = jnp.zeros((), tmpVal.dtype)
    del2 = jnp.stack([ddx*ddx, ddy*ddy], axis=-1)                   # :97 (two adds per point)
    tileDel2 = tile_sums(jnp.where(wet[..., None], del2, zero).reshape(
        del2.shape[0], -1, 2*sNx))
    tileNbPt = tile_sums(jnp.where(wet, oneRL, zero))               # :99
    tmpVol = arrArea[i, j]*arrDr[k]*tmpMask                         # :100
    tileVol = tile_sums(jnp.where(wet, tmpVol, zero))               # :101
    tileMean = tile_sums(jnp.where(wet, tmpVol*tmpVal, zero))       # :102

    theNbPt = global_sum_tile_rl(tileNbPt, ex)                      # :111
    theDel2 = global_sum_tile_rl(tileDel2, ex)                      # :112
    theVol = global_sum_tile_rl(tileVol, ex)                        # :113
    theMean = global_sum_tile_rl(tileMean, ex)                      # :114

    if theNbPt > zeroRL:                                            # :117-120
        theDel2 = np.sqrt(theDel2)/theNbPt

    if theVol > 0.0:                                                # :122-155
        theMean = theMean/theVol
        theVar = theVar/theVol
        if noPnts:
            theMin = theMean
        theMin = -theMin
        theMin = global_max_rl(theMin, ex)
        theMin = -theMin
        if noPnts:
            theMax = theMean
        theMax = global_max_rl(theMax, ex)

        tileSD = tile_sums(jnp.where(                               # :132-149
            wet, tmpVol*(tmpVal-theMean)*(tmpVal-theMean), zero))
        theSD = global_sum_tile_rl(tileSD, ex)                      # :151
        theSD = np.sqrt(theSD/theVol)                               # :153

    return theMin, theMax, theMean, theSD, theDel2, theVol
