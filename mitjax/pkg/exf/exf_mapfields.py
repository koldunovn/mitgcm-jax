"""EXF_MAPFIELDS: pkg/exf/exf_mapfields.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.eesupp.exch_rs import EXCH_UV_XY_RS, EXCH_XY_RS
from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX, MIN
from mitjax.pkg.exf.exf_param_h import exf_half


def exf_mapfields(myTime, myIter, f, ff, *, cfg, exf, grid, params, fp, state, ex, tp):
    """EXF_MAPFIELDS( myTime, myIter, myThid )   @63cdc0b pkg/exf/exf_mapfields.F:3-380

    C     | o Map the EXF fields onto the model's forcing arrays (FFIELDS.h): Qnet, EmPmR, fu, fv, Qsw, SST, SSS,
    C     |   pLoad, saltFlux; the wind stress limited to +-windstressmax

    `f` EXF_FIELDS.h (dict; ustress/vstress are clipped in place, :216-233, :254-271), `ff` FFIELDS.h, `fp` the
    forcing parameters (rhoConstFresh, HeatCapacity_Cp; `fp.temp_EvPrRn_set` the static host flag of
    `temp_EvPrRn .NE. UNSET_RL`, :134), `state` (theta: the temp_EvPrRn block). Returns (f, ff).
    The `IF ( ustress .GT. windstressmax )` clips are REAL field tests: pointwise `where`. Lane M4CS32ICE:
    ALLOW_RUNOFTEMP (the runoff heat content with a runoftempfile, :199-209) and EXF_ALLOW_TIDES (phiTide2d of
    FFIELDS.h, :324-330, exchanged :372-374). Lane M4ADLAB session 3: EXF_SEAICE_FRACTION (exf_iceFraction of
    EXF_FIELDS.h from areamask, clipped to [0, 1], :340-348, exchanged :375-377; zeroRS / oneRS = 0, 1 (EEPARAMS.h),
    _RS = REAL*8 with REAL4_IS_SLOW)."""
    sz = cfg.size
    f = dict(f)
    ks = sz.Nr if params.usingPCoords else 1                                   # :89-90
    imin, imax = 1-sz.OLx, sz.sNx+sz.OLx                                       # :81-82
    jmin, jmax = 1-sz.OLy, sz.sNy+sz.OLy
    j = loop_j(jmin, jmax)
    i = loop_i(imin, imax)
    Qnet = ff.Qnet.at[i, j].set(exf.exf_outscal_hflux*f["hflux"][i, j])       # :102
    if exf.hfluxfile.strip() == "":                                            # :105-113
        Qnet = Qnet.at[i, j].set(Qnet[i, j] - exf.exf_outscal_hflux*(exf.hflux_exfremo_intercept
                                                                     + exf.hflux_exfremo_slope*(myTime-tp.startTime)))
    EmPmR = ff.EmPmR.at[i, j].set(exf.exf_outscal_sflux*f["sflux"][i, j]      # :118-119
                                  * fp.rhoConstFresh)
    if exf.sfluxfile.strip() == "":                                            # :122-130
        EmPmR = EmPmR.at[i, j].set(EmPmR[i, j] - fp.rhoConstFresh*exf.exf_outscal_sflux
                                   * (exf.sflux_exfremo_intercept + exf.sflux_exfremo_slope*(myTime-tp.startTime)))
    if cfg.cpp.flag("ALLOW_ATM_TEMP", "EXF_OPTIONS.h"):                        # :132-198
        if fp.temp_EvPrRn_set:                                                 # :134
            raise NotImplementedError("EXF_MAPFIELDS: temp_EvPrRn /= UNSET_RL (:134-196) is not ported")
    if (cfg.cpp.flag("ALLOW_RUNOFF", "EXF_OPTIONS.h") and cfg.cpp.flag("ALLOW_RUNOFTEMP", "EXF_OPTIONS.h")
            and exf.runoftempfile.strip()):                                    # :199-209
        js = loop_j(1, sz.sNy)
        is_ = loop_i(1, sz.sNx)
        Qnet = Qnet.at[is_, js].set(Qnet[is_, js]                              # :203-206
                                    + fp.HeatCapacity_Cp
                                    * (state.theta[is_, js, ks] - f["runoftemp"][is_, js])
                                    * f["runoff"][is_, js]*fp.rhoConstFresh)
    jj = loop_j(jmin, jmax)
    ii = loop_i(imin, imax)
    us = f["ustress"][ii, jj]
    us = jnp.where(us > exf.windstressmax, exf.windstressmax, us)              # :216-223
    us = jnp.where(us < -exf.windstressmax, -exf.windstressmax, us)            # :227-233
    f["ustress"] = f["ustress"].at[ii, jj].set(us)
    i1 = loop_i(imin+1, imax)
    if exf.stressIsOnCgrid:                                                    # :234-249
        fu = ff.fu.at[i1, jj].set(exf.exf_outscal_ustress*f["ustress"][i1, jj])
    else:
        fu = ff.fu.at[i1, jj].set(exf.exf_outscal_ustress                      # :244-246
                                  * (f["ustress"][i1, jj]+f["ustress"][i1-1, jj])
                                  * exf_half*grid.maskW[i1, jj, ks])
    vs = f["vstress"][ii, jj]
    vs = jnp.where(vs > exf.windstressmax, exf.windstressmax, vs)              # :254-261
    vs = jnp.where(vs < -exf.windstressmax, -exf.windstressmax, vs)            # :265-271
    f["vstress"] = f["vstress"].at[ii, jj].set(vs)
    j1 = loop_j(jmin+1, jmax)
    if exf.stressIsOnCgrid:                                                    # :272-287
        fv = ff.fv.at[ii, j1].set(exf.exf_outscal_vstress*f["vstress"][ii, j1])
    else:
        fv = ff.fv.at[ii, j1].set(exf.exf_outscal_vstress                      # :282-284
                                  * (f["vstress"][ii, j1]+f["vstress"][ii, j1-1])
                                  * exf_half*grid.maskS[ii, j1, ks])
    upd = dict(Qnet=Qnet, EmPmR=EmPmR, fu=fu, fv=fv)
    if cfg.cpp.flag("ALLOW_ATM_TEMP", "EXF_OPTIONS.h") or cfg.cpp.flag("SHORTWAVE_HEATING"):   # :289-296
        upd["Qsw"] = ff.Qsw.at[i, j].set(exf.exf_outscal_swflux*f["swflux"][i, j])
    if cfg.cpp.flag("ALLOW_CLIMSST_RELAXATION", "EXF_OPTIONS.h"):             # :298-304
        upd["SST"] = ff.SST.at[i, j].set(exf.exf_outscal_sst*f["climsst"][i, j])
    if cfg.cpp.flag("ALLOW_CLIMSSS_RELAXATION", "EXF_OPTIONS.h"):             # :306-312
        upd["SSS"] = ff.SSS.at[i, j].set(exf.exf_outscal_sss*f["climsss"][i, j])
    if cfg.cpp.flag("ATMOSPHERIC_LOADING"):                                    # :314-322
        upd["pLoad"] = ff.pLoad.at[i, j].set(exf.exf_outscal_apressure*f["apressure"][i, j] - params.surf_pRef)
    if cfg.cpp.flag("EXF_ALLOW_TIDES", "EXF_OPTIONS.h"):                       # :324-330
        upd["phiTide2d"] = ff.phiTide2d.at[i, j].set(exf.exf_outscal_tidePot*f["tidePot"][i, j])   # :327
    if cfg.cpp.flag("ALLOW_SALTFLX", "EXF_OPTIONS.h"):                         # :332-338
        upd["saltFlux"] = ff.saltFlux.at[i, j].set(f["saltflx"][i, j])
    if cfg.cpp.flag("EXF_SEAICE_FRACTION", "EXF_OPTIONS.h"):                   # :340-348
        iceFr = exf.exf_outscal_areamask*f["areamask"][i, j]                   # :343-344
        iceFr = MIN(MAX(iceFr, 0.0, p="b"), 1.0, p="a")                        # :345-346 zeroRS, oneRS
        f["exf_iceFraction"] = f["exf_iceFraction"].at[i, j].set(iceFr)
    # :354-377 exchanges
    upd["Qnet"] = EXCH_XY_RS(upd["Qnet"], ex=ex)                              # :355
    upd["EmPmR"] = EXCH_XY_RS(upd["EmPmR"], ex=ex)                            # :356
    upd["fu"], upd["fv"] = EXCH_UV_XY_RS(upd["fu"], upd["fv"], True, ex=ex)   # :357
    if cfg.cpp.flag("SHORTWAVE_HEATING"):                                      # :359-362
        upd["Qsw"] = EXCH_XY_RS(upd["Qsw"], ex=ex)
    if cfg.cpp.flag("ALLOW_CLIMSST_RELAXATION", "EXF_OPTIONS.h"):             # :363-365
        upd["SST"] = EXCH_XY_RS(upd["SST"], ex=ex)
    if cfg.cpp.flag("ALLOW_CLIMSSS_RELAXATION", "EXF_OPTIONS.h"):             # :366-368
        upd["SSS"] = EXCH_XY_RS(upd["SSS"], ex=ex)
    if cfg.cpp.flag("ATMOSPHERIC_LOADING"):                                    # :369-371
        upd["pLoad"] = EXCH_XY_RS(upd["pLoad"], ex=ex)
    if cfg.cpp.flag("EXF_ALLOW_TIDES", "EXF_OPTIONS.h"):                       # :372-374
        upd["phiTide2d"] = EXCH_XY_RS(upd["phiTide2d"], ex=ex)
    if cfg.cpp.flag("EXF_SEAICE_FRACTION", "EXF_OPTIONS.h"):                   # :375-377
        f["exf_iceFraction"] = EXCH_XY_RS(f["exf_iceFraction"], ex=ex)
    return f, ff.replace(**upd)

