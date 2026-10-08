"""EXF_WIND: pkg/exf/exf_wind.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import FArray, loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX
from mitjax.ops.safe import div, safe_sqrt


def exf_wind(myTime, myIter, f, *, cfg, exf, params, state, ctrl=None):
    """EXF_WIND( myTime, myIter, myThid )   @63cdc0b pkg/exf/exf_wind.F:7-310

    C     | o Prepare surface wind speed and direction (and stress) from the atmospheric state.

    `f` EXF_FIELDS.h (dict), `state` (uVel, vVel: useRelativeWind). Returns the new dict. Ported: useAtmWind
    (:126-151) with useRelativeWind or not (:102-118), the wind-speed lower limit (:265-269). With
    ALLOW_BULKFORMULAE and ALLOW_ATM_TEMP the stress is computed in EXF_BULKFORMULAE (:271-303 not compiled).
    Lane M4CS32ICE: useAtmWind = .FALSE. with ALLOW_BULKFORMULAE (:154-243) and a wspeed file: wStress, cw, sw
    from the stress (stressIsOnCgrid: the mean of the squares of the four face values, cw / sw from the west /
    south face value alone, :158-180), uwind / vwind = wspeed*cw / wspeed*sw (:237-242). Raises: the wind-speed
    inversion without a wspeed file (:183-225). Lane M4ADCOL: ALLOW_GENTIM2D_CONTROL's xx_wspeed (:248-258).
    `IF ( wsSq .NE. 0. )` (:133-141) and `IF ( usSq .NE. 0. )` (:170-178): both branches computed on every lane,
    the root and the divisions guarded by the same test (finite derivatives where the square is 0)."""
    if not exf.useAtmWind:
        if not cfg.cpp.flag("ALLOW_BULKFORMULAE", "EXF_OPTIONS.h"):           # :154 #ifdef ALLOW_BULKFORMULAE
            raise NotImplementedError("EXF_WIND: useAtmWind = .FALSE. without ALLOW_BULKFORMULAE is not ported")
        if exf.wspeedfile.strip() == "":                                       # :183-225
            raise NotImplementedError("EXF_WIND: the wind speed from the stress (no wspeedfile, :183-225) is not "
                                      "ported")
    # lane M4ADCOL: ALLOW_GENTIM2D_CONTROL (exf_wind.F includes CTRL_OPTIONS.h, :37-41) adds xx_gentim2d to
    # wspeed where the file name starts with xx_wspeed (:248-258, every step, the loop is outside IF ( useCTRL ):
    # `ctrl` None means no gentim2d control state, which with ALLOW_GENTIM2D_CONTROL raises)
    gentim = cfg.cpp.flag("ALLOW_CTRL") and cfg.cpp.flag("ALLOW_GENTIM2D_CONTROL", "CTRL_OPTIONS.h")   # CTRL_OPTIONS.h
    # exists only in a build with pkg/ctrl (exf_wind.F:3-5 includes it #ifdef ALLOW_CTRL): ALLOW_CTRL first
    if gentim and ctrl is None:
        raise NotImplementedError("EXF_WIND: ALLOW_GENTIM2D_CONTROL without the gentim2d state is not ported")
    if not (cfg.cpp.flag("ALLOW_BULKFORMULAE", "EXF_OPTIONS.h") and cfg.cpp.flag("ALLOW_ATM_TEMP", "EXF_OPTIONS.h")):
        raise NotImplementedError("EXF_WIND: the stress without ALLOW_BULKFORMULAE/ALLOW_ATM_TEMP (:271-303) is "
                                  "not ported")
    sz = cfg.size
    f = dict(f)
    ks = sz.Nr if params.usingPCoords else 1                                   # :78-79
    j = loop_j(1, sz.sNy)
    i = loop_i(1, sz.sNx)
    T = f["uwind"].data.shape[0]
    loc = dict(i=(1, sz.sNx), j=(1, sz.sNy))
    wsLoc = FArray(jnp.zeros((T, sz.sNy, sz.sNx)), "wsLoc", **loc)            # :92 wsLoc = 0.
    urelw = FArray(jnp.zeros((T, sz.sNy, sz.sNx)), "urelw", **loc)
    vrelw = FArray(jnp.zeros((T, sz.sNy, sz.sNx)), "vrelw", **loc)
    urelw = urelw.at[i, j].set(f["uwind"][i, j])                               # :93
    vrelw = vrelw.at[i, j].set(f["vwind"][i, j])                               # :94
    for n in ("cw", "sw", "sh", "wStress"):                                    # :95-98
        f[n] = f[n].at[i, j].set(0.)
    if exf.useRelativeWind:                                                    # :102-118
        urelw = urelw.at[i, j].set(f["uwind"][i, j] - 0.5
                                   * (state.uVel[i, j, ks]+state.uVel[i+1, j, ks]))   # :106-107
        vrelw = vrelw.at[i, j].set(f["vwind"][i, j] - 0.5
                                   * (state.vVel[i, j, ks]+state.vVel[i, j+1, ks]))   # :108-109
    if not exf.useAtmWind:                                                     # :154-243
        us, vs = f["ustress"], f["vstress"]
        if exf.stressIsOnCgrid:                                                # :160-165
            usSq = (us[i, j]*us[i, j]
                    + us[i+1, j]*us[i+1, j]
                    + vs[i, j]*vs[i, j]
                    + vs[i, j+1]*vs[i, j+1]
                    )*0.5
        else:                                                                  # :166-169
            usSq = us[i, j]*us[i, j] + vs[i, j]*vs[i, j]
        nz = usSq != 0.                                                        # :170
        wSt = safe_sqrt(usSq, nz)                                              # :171 (0. where usSq = 0, :176)
        wSd = jnp.where(nz, wSt, 1.0)
        f["wStress"] = f["wStress"].at[i, j].set(wSt)
        f["cw"] = f["cw"].at[i, j].set(jnp.where(nz, div(us[i, j], wSd), 0.))  # :173, :177
        f["sw"] = f["sw"].at[i, j].set(jnp.where(nz, div(vs[i, j], wSd), 0.))  # :174, :178
        f["uwind"] = f["uwind"].at[i, j].set(f["wspeed"][i, j]*f["cw"][i, j])  # :239
        f["vwind"] = f["vwind"].at[i, j].set(f["wspeed"][i, j]*f["sw"][i, j])  # :240
        if gentim:
            f = _gentim2d_wspeed(f, ctrl, i, j)                                # :248-258
        f["sh"] = f["sh"].at[i, j].set(MAX(f["wspeed"][i, j], exf.umin, p="a"))   # :267
        return f
    # :129-143 wind speed and direction
    wsSq = urelw[i, j]*urelw[i, j] + vrelw[i, j]*vrelw[i, j]                   # :131-132
    nz = wsSq != 0.                                                            # :133
    ws = safe_sqrt(wsSq, nz)                                                   # :134 (0. where wsSq = 0, :138)
    wsd = jnp.where(nz, ws, 1.0)
    wsLoc = wsLoc.at[i, j].set(ws)
    f["cw"] = f["cw"].at[i, j].set(jnp.where(nz, div(urelw[i, j], wsd), 0.))   # :135, :139
    f["sw"] = f["sw"].at[i, j].set(jnp.where(nz, div(vrelw[i, j], wsd), 0.))   # :136, :140
    if exf.wspeedfile.strip() == "":                                           # :144-151
        f["wspeed"] = f["wspeed"].at[i, j].set(wsLoc[i, j])                    # :148
    if gentim:
        f = _gentim2d_wspeed(f, ctrl, i, j)                                    # :248-258
    f["sh"] = f["sh"].at[i, j].set(MAX(f["wspeed"][i, j], exf.umin, p="a"))    # :267
    return f


def _gentim2d_wspeed(f, ctrl, i, j):
    """exf_wind.F:248-258: DO iarr = 1, maxCtrlTim2D: IF (xx_gentim2d_file(iarr)(1:9).EQ.'xx_wspeed') wspeed =
    wspeed + xx_gentim2d(iarr) (interior)."""
    for iarr in sorted(ctrl["xx_gentim2d"]):
        if ctrl["files"][iarr][:9] == "xx_wspeed":
            f["wspeed"] = f["wspeed"].at[i, j].set(f["wspeed"][i, j] + ctrl["xx_gentim2d"][iarr][i, j])
    return f
