"""CTRL_MAP_GENTIM2D   @63cdc0b pkg/ctrl/ctrl_map_gentim2d.F:6-150"""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.pkg.ctrl.ctrl_get_gen import ctrl_get_gen
from mitjax.pkg.ctrl.ctrl_get_mask import ctrl_get_mask2d
from mitjax.pkg.ctrl.ctrl_readparms import fstr_blank
from mitjax.pkg.ctrl.ctrl_toolbox import xy

# eesupp/inc/EEPARAMS.h:71: zeroRL = 0.0 _d 0
zeroRL = 0.0


def ctrl_map_gentim2d(myTime, myIter, *, cfg, gentim2d, maskC, genarr, effective, clock):
    """CTRL_MAP_GENTIM2D( myTime, myIter, myThid )

    `genarr`: the CTRL_GENARR.h state {"xx_gentim2d", "xx_gentim2d0", "xx_gentim2d1", "wgentim2d"}, each
    {iarr: FArray}; returns the updated dict (the Fortran updates the common block). `effective`: {iarr: records}
    from CTRL_MAP_INI_GENTIM2D. The `xx_gentim2d_glosum` branch (:102-141, global mean with GLOBAL_SUM_TILE_RL) is not
    ported (F in R5)."""
    sz = cfg.size
    genarr = {k: dict(v) for k, v in genarr.items()}
    for g in gentim2d:                                                 # :51 DO iarr = 1, maxCtrlTim2D
        iarr = g.iarr
        if fstr_blank(g.xx_gentim2d_weight):                           # :53
            continue
        some = genarr["xx_gentim2d"][iarr]
        xx_gentim2d_loc = xy(jnp.zeros_like(some.data), "xx_gentim2d_loc", sz)   # :55-63 0. _d 0 (all points)
        mask2D = xy(jnp.zeros_like(some.data), "mask2D", sz)           # local _RS (every point written below)
        mask2D = ctrl_get_mask2d(g.xx_gentim2d_file, mask2D, cfg=cfg, maskC=maskC)    # :65
        # xx_gentim2d_startdate(1,iarr): CTRL_INIT_REC sets 0 without useCAL; with useCAL (lane M4ADCOL) CAL_FULLDATE
        # of startdate1/2 (ctrl_init_rec.F:96-97), kept in the clock by the set-up
        startdate = tuple(clock.get("startdate", {}).get(iarr, (0, 0, 0, 0)))
        xx_gentim2d_loc, x0, x1 = ctrl_get_gen(                        # :66-77
            g.xx_gentim2d_file, startdate, g.xx_gentim2d_period, mask2D, xx_gentim2d_loc,
            genarr["xx_gentim2d0"][iarr], genarr["xx_gentim2d1"][iarr], zeroRL, zeroRL,
            genarr["wgentim2d"][iarr], myTime, myIter, cfg=cfg, effective=effective[iarr], clock=clock)
        genarr["xx_gentim2d0"][iarr] = x0
        genarr["xx_gentim2d1"][iarr] = x1
        j = loop_j(1, sz.sNy)
        i = loop_i(1, sz.sNx)
        if g.xx_gentim2d_cumsum:                                       # :79-89
            genarr["xx_gentim2d"][iarr] = genarr["xx_gentim2d"][iarr].at[i, j].set(
                genarr["xx_gentim2d"][iarr][i, j] + xx_gentim2d_loc[i, j])
        else:                                                          # :90-99
            genarr["xx_gentim2d"][iarr] = genarr["xx_gentim2d"][iarr].at[i, j].set(xx_gentim2d_loc[i, j])
        if g.xx_gentim2d_glosum:                                       # :102
            raise NotImplementedError("CTRL_MAP_GENTIM2D: xx_gentim2d_glosum (:102-141) not ported")
    return genarr
