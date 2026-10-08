"""MON_VORT3   @63cdc0b pkg/monitor/mon_vort3.F:8-358"""

import jax.numpy as jnp
import numpy as np

from mitjax.farray import loops_kji
from mitjax.ops.safe import safe_div
from mitjax.ops.fortran_minmax_host import max_chain, min_chain
from mitjax.pkg.monitor.mon_out import mon_out_rl
from mitjax.pkg.monitor.mon_set_pref import mon_set_pref
from mitjax.pkg.monitor.monitor_h import global_max_rl, global_sum_tile_rl, mon_string_none, tile_sums


def mon_vort3(myIter, *, cfg, grid, state, mon, ex=None):
    """MON_VORT3( myIter, myThid )

    C     Calculates stats for Vorticity (z-component).

    Inputs: state.uVel, state.vVel; grid.hFacW, hFacS, dxC, dyC, rAz, recip_rAz, drF, fCoriG, yG.

    Per level k (:87-307): hFacZ and vort3 at i=1..sNx, j=1..sNy (:91-139, the standard hFacZ: MONITOR_TEST_HFACZ is
    #undef, MONITOR_OPTIONS.h:13), then the sums over points with hFacZ > 0 in j, i order; with the tile and k loops
    around them each tile's chain runs over k, j, i (sums through mitjax/eesupp). theMin/theMax are one chain over
    all tiles from 1.D20 / -1.D20 (the new value wins ties and NaN: $MJX_REFERENCE/minmax_sites). Lane B (Task 25):
    the cubed-sphere arm (useCubedSphereExchange, :143-239): i, j run to sNx+1, sNy+1 with hFacZ = vort3 = 0. on
    the added row and column, then the three-point corners (AZcorner = 1. _d 0, :66) of the tiles that
    `cfg.cs_vort3_corners` ([tile] x (SW, SE, NW): the W2 edge flags with :174-177's face tests, NE = .FALSE.)
    marks. Not ported (raise): the polar rows of a lat-lon grid that reaches +-90 (:242-280, a tile-dependent jMax);
    no M1 grid has them (coverage: :244-260, :263-278 never run)."""
    sNx, sNy, Nr = cfg.sNx, cfg.sNy, cfg.Nr
    g, s = grid, state
    if cfg.MONITOR_TEST_HFACZ:
        raise NotImplementedError("MON_VORT3: MONITOR_TEST_HFACZ (mon_vort3.F:76-121) is not ported")
    cube = bool(cfg.useCubedSphereExchange)
    if cube and getattr(cfg, "cs_vort3_corners", None) is None:
        raise NotImplementedError("MON_VORT3: the cubed-sphere corners (:143-239) need cfg.cs_vort3_corners")
    if cfg.usingSphericalPolarGrid:                                 # :242-280
        yG = np.asarray(g.yG.data)
        OLx, OLy = cfg.OLx, cfg.OLy
        north = yG[:, sNy + 1 - 1 + OLy, 1 - 1 + OLx] == 90.0       # yG(1,sNy+1,bi,bj).EQ.90.
        south = yG[:, 1 - 1 + OLy, 1 - 1 + OLx] == -90.0            # yG(1,1,bi,bj).EQ.-90.
        if north.any() or south.any():
            raise NotImplementedError("MON_VORT3: polar rows (yG = +-90, mon_vort3.F:243-279) are not ported")

    theMin = np.float64(1.0e20)                                     # :57-66 (1. _d 20)
    theMax = np.float64(-1.0e20)
    theArea = np.float64(0.0)
    theMean = np.float64(0.0)
    theVar = np.float64(0.0)
    theVol = np.float64(0.0)
    volMean = np.float64(0.0)
    volVar = np.float64(0.0)
    theSD = np.float64(0.0)

    iMax, jMax = (sNx+1, sNy+1) if cube else (sNx, sNy)             # :89-90; cube: :145-146
    k, j, i = loops_kji((1, Nr), (1, jMax), (1, iMax))              # :87-92 (cube: the added row / column too)
    hFacZ = 0.25*(g.hFacW[i, j-1, k]                                # :125-130 (0.25 _d 0)
                  + g.hFacW[i, j, k]
                  + g.hFacS[i-1, j, k]
                  + g.hFacS[i, j, k])
    vort3 = g.recip_rAz[i, j]*(                                     # :132-137
        s.vVel[i, j, k]*g.dyC[i, j]
        - s.vVel[i-1, j, k]*g.dyC[i-1, j]
        - s.uVel[i, j, k]*g.dxC[i, j]
        + s.uVel[i, j-1, k]*g.dxC[i, j-1])
    if cube:                                                        # :143-239 (lane B)
        hFacZ, vort3 = _cs_corners(hFacZ, vort3, cfg=cfg, grid=g, state=s)

    wet = hFacZ > 0.0                                               # :284-305
    zero = jnp.zeros((), vort3.dtype)
    tmpVal = vort3
    tmpAre = g.rAz[i, j]*g.drF[k]
    tmpVol = g.rAz[i, j]*g.drF[k]*hFacZ
    tileArea = tile_sums(jnp.where(wet, tmpAre, zero))
    theMin = min_chain(theMin, tmpVal, wet, p="b")                 # :292
    theMax = max_chain(theMax, tmpVal, wet, p="b")                 # :293
    tmpVal = tmpVal + g.fCoriG[i, j]
    tileSum = tile_sums(jnp.where(wet, tmpAre*tmpVal, zero))
    tileVar = tile_sums(jnp.where(wet, tmpAre*tmpVal*tmpVal, zero))
    tmpVal = safe_div(tmpVal, hFacZ, wet)                           # tmpVal / hFacZ(i,j) where hFacZ > 0
    tileVol = tile_sums(jnp.where(wet, tmpVol, zero))
    tileVSum = tile_sums(jnp.where(wet, tmpVol*tmpVal, zero))
    tileVSq = tile_sums(jnp.where(wet, tmpVol*tmpVal*tmpVal, zero))

    theMin = -theMin                                                # :317-320
    theMin = global_max_rl(theMin, ex)
    theMax = global_max_rl(theMax, ex)
    theMin = -theMin
    theArea = global_sum_tile_rl(tileArea, ex)                      # :327-332
    theVol = global_sum_tile_rl(tileVol, ex)
    theMean = global_sum_tile_rl(tileSum, ex)
    theVar = global_sum_tile_rl(tileVar, ex)
    volMean = global_sum_tile_rl(tileVSum, ex)
    volVar = global_sum_tile_rl(tileVSq, ex)
    if theArea > 0.0:                                               # :333-339
        theMean = theMean/theArea
        theVar = theVar/theArea
        theVar = theVar - theMean*theMean
        if theVar > 0.0:
            theVar = np.sqrt(theVar)
    if theVol > 0.0:                                                # :340-345
        volMean = volMean/theVol
        volVar = volVar/theVol
        volVar = volVar - volMean*volMean
        if volVar > 0.0:
            theSD = np.sqrt(volVar)

    mon_set_pref("vort", mon=mon)                                   # :348-354
    mon_out_rl(mon_string_none, theMin, "_r_min", cfg=cfg, mon=mon)
    mon_out_rl(mon_string_none, theMax, "_r_max", cfg=cfg, mon=mon)
    mon_out_rl(mon_string_none, theMean, "_a_mean", cfg=cfg, mon=mon)
    mon_out_rl(mon_string_none, theVar, "_a_sd", cfg=cfg, mon=mon)
    mon_out_rl(mon_string_none, volMean, "_p_mean", cfg=cfg, mon=mon)
    mon_out_rl(mon_string_none, theSD, "_p_sd", cfg=cfg, mon=mon)


def _cs_corners(hFacZ, vort3, *, cfg, grid, state):
    """mon_vort3.F:143-239 (useCubedSphereExchange; lane B): hFacZ, vort3 [tile, k, j = 1..sNy+1, i = 1..sNx+1] with
    the row j = sNy+1 and the column i = sNx+1 set to 0. (:147-155, REAL*4 zero), then per tile the S.W. (1,1),
    S.E. (sNx+1,1) and N.W. (1,sNy+1) corners (:180-224) where cfg.cs_vort3_corners says so:
    vort3 = recip_rAz/AZcorner*( +-vVel*dyC - uVel(i,j)*dxC(i,j) + uVel(i,j-1)*dxC(i,j-1) ),
    hFacZ = ( hFacW(i,j-1) + hFacW(i,j) + hFacS(i or i-1,j) )/3. _d 0, AZcorner = 1. _d 0 (:66)."""
    sNx, sNy, OLx, OLy = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy
    AZcorner = 1.0                                                  # :66  1. _d 0
    hz = jnp.asarray(hFacZ)
    v3 = jnp.asarray(vort3)
    hz = hz.at[:, :, sNy, :].set(0.).at[:, :, :, sNx].set(0.)      # :147-155 (j = jMax row, i = iMax column)
    v3 = v3.at[:, :, sNy, :].set(0.).at[:, :, :, sNx].set(0.)
    P = lambda i, j: (j - 1 + OLy, i - 1 + OLx)                     # noqa: E731  Fortran (i, j) -> padded (j0, i0)
    rAz, dxC, dyC = (jnp.asarray(getattr(grid, n).data) for n in ("recip_rAz", "dxC", "dyC"))
    hW, hS = jnp.asarray(grid.hFacW.data), jnp.asarray(grid.hFacS.data)
    u, v = jnp.asarray(state.uVel.data), jnp.asarray(state.vVel.data)
    for t, (sw, se, nw) in enumerate(np.asarray(cfg.cs_vort3_corners, bool)):
        for on, i, j, west in ((sw, 1, 1, True), (se, sNx+1, 1, False), (nw, 1, sNy+1, True)):
            if not on:
                continue
            c, cm = P(i, j), P(i, j-1)
            if west:                                                # :186-190 (S.W.), :214-218 (N.W.)
                vv = v[t, :, c[0], c[1]]*dyC[t][c]
                hs = hS[t, :, c[0], c[1]]
            else:                                                   # :200-204 (S.E.): -vVel(i-1,j)*dyC(i-1,j)
                ci = P(i-1, j)
                vv = -v[t, :, ci[0], ci[1]]*dyC[t][ci]
                hs = hS[t, :, ci[0], ci[1]]
            val = rAz[t][c]/AZcorner*(vv
                                      - u[t, :, c[0], c[1]]*dxC[t][c]
                                      + u[t, :, cm[0], cm[1]]*dxC[t][cm])
            hz3 = (hW[t, :, cm[0], cm[1]] + hW[t, :, c[0], c[1]] + hs)/3.0   # 3. _d 0
            v3 = v3.at[t, :, j-1, i-1].set(val)
            hz = hz.at[t, :, j-1, i-1].set(hz3)
    return hz, v3
