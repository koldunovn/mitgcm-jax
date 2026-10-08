"""CTRL_MAP_FORCING   @63cdc0b pkg/ctrl/ctrl_map_forcing.F:9-181"""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.pkg.ctrl.ctrl_readparms import fstr_prefix
from mitjax.pkg.ctrl.ctrl_toolbox import xy
from mitjax.pkg.ctrl.rotate_uv2en_standin import rotate_uv2en_rl

FIELDS = ("fu", "fv", "Qnet", "EmPmR", "Qsw", "SST", "SSS", "pLoad", "saltFlux")


def ctrl_map_forcing(myTime, myIter, *, cfg, gentim2d, xx_gentim2d, ffields, grid, ex, usingPCoords):
    """CTRL_MAP_FORCING( myTime, myIter, myThid )

    c     | Add the surface flux anomalies of the control vector
    c     | to the model flux fields and update the tile halos.

    `ffields`: FFIELDS.h fields {fu, fv, Qnet, EmPmR, Qsw, SST, SSS, pLoad, saltFlux} as 2-D FArrays; returns the
    updated dict. `xx_gentim2d`: {iarr: FArray} (CTRL_GENARR.h, after CTRL_MAP_GENTIM2D). `grid`: angleCosC,
    angleSinC (2-D), maskW, maskS (3-D) FArrays. `ex`: the experiment's exchanger (EXCH_XY_RS and EXCH_UV_XY_RS
    are the RL exchange code with _RS = REAL*8 in every M1 build: the probe maps XY and UVs).
    ALLOW_STREAMICE (:129-157, :170-177) is not compiled in this build. The file-name tests (:79-123) are
    static (namelist strings): they select at trace time which fields get the control."""
    sz = cfg.size
    if cfg.cpp.ALLOW_STREAMICE:
        raise NotImplementedError("CTRL_MAP_FORCING: ALLOW_STREAMICE not ported")
    ff = dict(ffields)
    zero = jnp.zeros_like(ff["Qnet"].data)
    tmpUE = xy(zero, "tmpUE", sz)                                       # :62-73 0. _d 0 (all points)
    tmpVN = xy(zero, "tmpVN", sz)
    tmpUX = xy(zero, "tmpUX", sz)
    tmpVY = xy(zero, "tmpVY", sz)
    names = {g.iarr: g.xx_gentim2d_file for g in gentim2d}
    j = loop_j(1, sz.sNy)                                               # :77
    i = loop_i(1, sz.sNx)                                               # :78
    for iarr, fnam in names.items():                                    # :79-85 (k-independent: per point, the
        if fstr_prefix(fnam, 5, "xx_fe"):                               #  iarr loop order is kept per field)
            tmpUE = tmpUE.at[i, j].set(tmpUE[i, j] + xx_gentim2d[iarr][i, j])
        if fstr_prefix(fnam, 5, "xx_fn"):
            tmpVN = tmpVN.at[i, j].set(tmpVN[i, j] + xx_gentim2d[iarr][i, j])
    tmpUE = xy(ex.EXCH_XY_RL(tmpUE.data), "tmpUE", sz)                  # :91
    tmpVN = xy(ex.EXCH_XY_RL(tmpVN.data), "tmpVN", sz)                  # :92
    tmpUX, tmpVY, tmpUE, tmpVN = rotate_uv2en_rl(                       # :93-94
        tmpUX, tmpVY, tmpUE, tmpVN, False, True, True, 1, sz=sz, usingPCoords=usingPCoords,
        angleCosC=grid["angleCosC"], angleSinC=grid["angleSinC"], maskW=grid["maskW"], maskS=grid["maskS"])
    ff["fu"] = ff["fu"].at[i, j].set(ff["fu"][i, j] + tmpUX[i, j])      # :100
    ff["fv"] = ff["fv"].at[i, j].set(ff["fv"][i, j] + tmpVY[i, j])      # :101
    targets = (("xx_qnet", 7, "Qnet"), ("xx_empmr", 8, "EmPmR"), ("xx_qsw", 6, "Qsw"), ("xx_sst", 6, "SST"),
               ("xx_sss", 6, "SSS"), ("xx_pload", 8, "pLoad"), ("xx_saltflux", 11, "saltFlux"),
               ("xx_fu", 5, "fu"), ("xx_fv", 5, "fv"))                  # :104-122, in the Fortran's order
    for iarr, fnam in names.items():                                    # :102
        for lit, n, fld in targets:
            if fstr_prefix(fnam, n, lit):
                ff[fld] = ff[fld].at[i, j].set(ff[fld][i, j] + xx_gentim2d[iarr][i, j])
    for fld in ("Qnet", "EmPmR", "Qsw", "SST", "SSS", "pLoad", "saltFlux"):   # :159-165 EXCH_XY_RS
        ff[fld] = xy(ex.EXCH_XY_RL(ff[fld].data), fld, sz)
    fu, fv = ex.EXCH_UV_XY_RL(ff["fu"].data, ff["fv"].data, True)       # :166 EXCH_UV_XY_RS( fu, fv, .TRUE. )
    ff["fu"], ff["fv"] = xy(fu, "fu", sz), xy(fv, "fv", sz)
    return ff
