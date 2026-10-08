"""MON_SURFCOR   @63cdc0b pkg/monitor/mon_surfcor.F:8-204"""

import jax.numpy as jnp
import numpy as np

from mitjax.farray import loop_i, loop_j, loops_kji
from mitjax.pkg.monitor.mon_out import mon_out_rl
from mitjax.pkg.monitor.mon_set_pref import mon_set_pref
from mitjax.pkg.monitor.monitor_h import global_sum_tile_rl, mon_foot_mean, tile_sums


def _at_level(fld, ks, Nr, cfg):
    """fld(i,j,ks(i,j),bi,bj) for i=1..sNx, j=1..sNy where ks <= Nr (the level index is clipped elsewhere; those
    points are masked by the caller): [tile, sNy, sNx]."""
    sNx, sNy, OLx, OLy = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy
    d = fld.data[:, :, OLy:OLy+sNy, OLx:OLx+sNx]                    # [tile, k, j, i] interior
    kk = jnp.clip(ks, 1, Nr) - 1      # MINMAX-RAW: index clamp of a gather, not a Fortran MAX/MIN
    return jnp.take_along_axis(d, kk[:, None], axis=1)[:, 0]


def mon_surfcor(*, cfg, params, grid, state, mon, ex=None):
    """MON_SURFCOR( myThid )

    C     Compute and write area-mean surface expansion term (also called
    C     ``surface correction'' with Linear FS).

    Inputs: state.wVel, theta, salt; grid.kSurfC (integer, GRID.h), rA, maskInC, drF, and with NONLIN_FRSURF and
    select_rStar != 0: grid.h0FacC and state.rStarDhCDt (SURFACE.h); params.rUnit2mass.

    Per tile: tileArea, tile_wT, tile_wS are chains over j, i of the points with ks <= Nr (:80-99; through
    mitjax/eesupp). With NONLIN_FRSURF and select_rStar != 0 (:138-168), vT_Mean and vS_Mean are per-tile chains over
    k, j, i started at 0. (:142-145) and added once to tile_wT / tile_wS (:163-164). Lane B (Task 25): fluidIsAir
    (:93-96 tileWHeat, :122-136 tileTh2pe, :177-180, :195-199) with the Exner factors (rC(k)/atm_po)**atm_kappa of
    glibc's pow (params.rC_Po_kappa, from Params: ini_parms._atm_traced) and params.atm_Cp. Not ported (raise): the
    r* air term vT_Heat (:147-148, :159-160) and pkg/aim (:100-117). The divisions by theArea and the products with
    rUnit2mass of wT_Heat and theta2PE (:184-187) are kept (zero values for an ocean)."""
    sNx, sNy, Nr = cfg.sNx, cfg.sNy, cfg.Nr
    g, s = grid, state
    if cfg.fluidIsAir and cfg.NONLIN_FRSURF and cfg.select_rStar != 0:
        raise NotImplementedError("MON_SURFCOR: the r* air term vT_Heat (mon_surfcor.F:147-160) is not ported")
    if cfg.ALLOW_AIM and cfg.useAIM:
        raise NotImplementedError("MON_SURFCOR: pkg/aim branch (mon_surfcor.F:100-117) is not ported")
    theArea = np.float64(0.0)                                       # :67-71
    wT_Mean = np.float64(0.0)
    wS_Mean = np.float64(0.0)
    wT_Heat = np.float64(0.0)
    theta2PE = np.float64(0.0)

    j, i = loop_j(1, sNy), loop_i(1, sNx)                           # :80-99
    ks = g.kSurfC[i, j]
    wet = ks <= Nr                                                  # :83
    zero = jnp.zeros((), jnp.float64)
    w_ks = _at_level(s.wVel, ks, Nr, cfg)
    tileArea = tile_sums(jnp.where(wet, g.rA[i, j]*g.maskInC[i, j], zero))           # :84-85
    tmpVal = g.rA[i, j]*g.maskInC[i, j]*w_ks*_at_level(s.theta, ks, Nr, cfg)       # :86-87
    tile_wT = tile_sums(jnp.where(wet, tmpVal, zero))                               # :88
    tile_wS = tile_sums(jnp.where(wet, g.rA[i, j]*g.maskInC[i, j]                   # :89-91
                                  * w_ks*_at_level(s.salt, ks, Nr, cfg), zero))

    if cfg.fluidIsAir:                                              # :93-96, :122-136 (lane B)
        pk = jnp.asarray(params.rC_Po_kappa)                        # (rC(k)/atm_po)**atm_kappa, k = 1..Nr
        tileWHeat = tile_sums(jnp.where(wet, tmpVal*params.atm_Cp*pk[jnp.clip(ks, 1, Nr) - 1],   # MINMAX-RAW: gather index clamp
                                        zero))                                                  # :94-95
        tileTh2pe = jnp.zeros_like(tileArea)
        if Nr >= 2:
            k3, j3, i3 = loops_kji((2, Nr), (1, sNy), (1, sNx))     # :124-135, k = 2..Nr
            pkk = pk[1:][None, :, None, None]
            pkm = pk[:-1][None, :, None, None]
            ddPI = params.atm_Cp*(pkm - pkk)                         # :125-126
            tileTh2pe = tile_sums(-(ddPI*g.rA[i3, j3]*s.wVel[i3, j3, k3]          # :129-133 (acc - term)
                                    * (s.theta[i3, j3, k3]+s.theta[i3, j3, k3-1])*0.5
                                    * g.maskC[i3, j3, k3-1]*g.maskC[i3, j3, k3]
                                    * g.maskInC[i3, j3]))

    if cfg.NONLIN_FRSURF and cfg.select_rStar != 0:                 # :138-168
        k3, j3, i3 = loops_kji((1, Nr), (1, sNy), (1, sNx))
        tmpVol = g.rA[i3, j3]*g.h0FacC[i3, j3, k3]*g.drF[k3]*g.maskInC[i3, j3]   # :151-152
        tmpVal3 = s.rStarDhCDt[i3, j3]*s.theta[i3, j3, k3]                       # :153
        vT_Mean = tile_sums(tmpVol*tmpVal3)                                      # :154
        vS_Mean = tile_sums(tmpVol*s.rStarDhCDt[i3, j3]*s.salt[i3, j3, k3])     # :155-156
        tile_wT = tile_wT + vT_Mean                                              # :163
        tile_wS = tile_wS + vS_Mean                                              # :164

    theArea = global_sum_tile_rl(tileArea, ex)                      # :174-176
    wT_Mean = global_sum_tile_rl(tile_wT, ex)
    wS_Mean = global_sum_tile_rl(tile_wS, ex)
    if cfg.fluidIsAir:                                              # :177-180 (lane B)
        wT_Heat = global_sum_tile_rl(tileWHeat, ex)
        theta2PE = global_sum_tile_rl(tileTh2pe, ex)
    if theArea > 0.0:                                               # :181-188
        wT_Mean = wT_Mean/theArea
        wS_Mean = wS_Mean/theArea
        wT_Heat = wT_Heat/theArea
        theta2PE = theta2PE/theArea
        wT_Heat = wT_Heat*params.rUnit2mass
        theta2PE = theta2PE*params.rUnit2mass

    mon_set_pref("surfExpan", mon=mon)                              # :192-194
    mon_out_rl("_theta", wT_Mean, mon_foot_mean, cfg=cfg, mon=mon)
    mon_out_rl("_salt", wS_Mean, mon_foot_mean, cfg=cfg, mon=mon)
    if cfg.fluidIsAir:                                              # :195-199 (lane B)
        mon_out_rl("_Heat", wT_Heat, mon_foot_mean, cfg=cfg, mon=mon)
        mon_set_pref("En_Budget", mon=mon)
        mon_out_rl("_T2PE", theta2PE, mon_foot_mean, cfg=cfg, mon=mon)
