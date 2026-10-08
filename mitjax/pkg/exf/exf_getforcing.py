"""EXF_GETFORCING: pkg/exf/exf_getforcing.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.eesupp.exch_rs import _rewrap
from mitjax.farray import FArray, loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX
from mitjax.pkg.exf.exf_bulkformulae import exf_bulkformulae
from mitjax.pkg.exf.exf_getclim import exf_getclim
from mitjax.pkg.exf.exf_getffields import exf_getffields
from mitjax.pkg.exf.exf_getsurfacefluxes import exf_getsurfacefluxes
from mitjax.pkg.exf.exf_mapfields import exf_mapfields
from mitjax.pkg.exf.exf_radiation import exf_radiation
from mitjax.pkg.exf.exf_wind import exf_wind


def exf_tsf(f, *, cfg, exf, grid, params, state, ff):
    """exf_getforcing.F:163-190: the local exf_Tsf (surface temperature in K) from gcmSST, or extrapolated to the
    surface with sstExtrapol > 0 (an IF on a REAL namelist value that selects code: host decision)."""
    sz = cfg.size
    ks = sz.Nr if params.usingPCoords else 1                                   # :163-164
    kl = sz.Nr-1 if params.usingPCoords else 2                                 # :170-171
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    T = ff.gcmSST.data.shape[0]
    exf_Tsf = FArray(jnp.zeros((T, sz.sNy+2*sz.OLy, sz.sNx+2*sz.OLx)), "exf_Tsf", i=(1-sz.OLx, sz.sNx+sz.OLx),
                     j=(1-sz.OLy, sz.sNy+sz.OLy))
    if sz.Nr >= 2 and exf.sstExtrapol_gt_0:                                    # :174 (host flag, EXF_READPARMS)
        deltaSST = (exf.sstExtrapol*(state.theta[i, j, ks]-state.theta[i, j, kl])   # :177-179
                    * grid.maskC[i, j, kl])
        exf_Tsf = exf_Tsf.at[i, j].set(ff.gcmSST[i, j] + exf.cen2kel           # :180-181
                                       + MAX(deltaSST, 0., p="b"))           # :181
    else:
        exf_Tsf = exf_Tsf.at[i, j].set(ff.gcmSST[i, j] + exf.cen2kel)         # :187
    return exf_Tsf


def exf_getforcing(myTime, myIter, f, ff, *, cfg, exf, cal, grid, params, fp, state, rw, tp, ex, probe=None):
    """EXF_GETFORCING( myTime, myIter, myThid )   @63cdc0b pkg/exf/exf_getforcing.F:94-393

    C     | o Get the forcing fields for the current time step.

    `f` EXF_FIELDS.h (dict), `ff` FFIELDS.h (gcmSST read; Qnet, EmPmR, fu, fv, Qsw, SST, SSS, pLoad, saltFlux
    written by EXF_MAPFIELDS), `state` (theta, uVel, vVel), `rw` the run directory reader, `tp` the time
    parameters. Returns (f, ff). `probe(stage, f, ff)` (optional, host) is called at the jaxdump stages X01-X06.
    Concrete myTime/myIter: EXF_GETFFIELDS / EXF_GETCLIM read records on the host (as EXTERNAL_FIELDS_LOAD).
    Not ported (raise): useExfCheckRange checks with exf_debugLev >= debLevC (EXF_CHECK_RANGE :337-340 is a check
    with print-out and STOP on out-of-range values; called at myIter = nIter0 here: see EXF_CHECK_RANGE below),
    EXF_DIAGNOSTICS_FILL and EXF_MONITOR are output only."""
    sz = cfg.size
    ks = sz.Nr if params.usingPCoords else 1                                   # :163-164
    exf_Tsf = exf_tsf(f, cfg=cfg, exf=exf, grid=grid, params=params, state=state, ff=ff)   # :163-190
    f = exf_getclim(myTime, myIter, f, cfg=cfg, exf=exf, cal=cal, grid=grid, params=params, rw=rw, tp=tp, ex=ex)
    f = exf_getffields(myTime, myIter, f, cfg=cfg, exf=exf, cal=cal, grid=grid, params=params, rw=rw, tp=tp)
    if probe is not None:
        probe("X01_exf_getffields", f, ff)
    f = exf_stress_exch(f, exf=exf, ex=ex)                                     # :200-204
    f, ff = exf_getforcing_fluxes(exf_Tsf, myTime, myIter, f, ff, cfg=cfg, exf=exf, grid=grid, params=params,
                                  fp=fp, state=state, tp=tp, ex=ex, probe=probe)
    return f, ff


def exf_stress_exch(f, *, exf, ex):
    """exf_getforcing.F:200-204: with useAtmWind = .FALSE. and the C-grid stress read from both files, the stress
    is exchanged right after EXF_GETFFIELDS (EXCH_UV_XY_RL, withSigns: on the cube the vector exchange with the
    component swaps and signs; lane M4CS32ICE). Returns the new dict."""
    if not exf.useAtmWind:                                                     # :200
        if exf.stressIsOnCgrid and exf.ustressfile.strip() and exf.vstressfile.strip():   # :201-202
            f = dict(f)
            u, v = ex.EXCH_UV_XY_RL(f["ustress"].data, f["vstress"].data, True)   # :203
            f["ustress"], f["vstress"] = _rewrap(u, f["ustress"]), _rewrap(v, f["vstress"])
    return f


def exf_getforcing_fluxes(exf_Tsf, myTime, myIter, f, ff, *, cfg, exf, grid, params, fp, state, tp, ex,
                          probe=None, mon=None, ctrl=None):
    """exf_getforcing.F:259-388: everything after the reads (EXF_RADIATION, EXF_WIND, EXF_BULKFORMULAE, the hflux /
    sflux sums, the stress exchange, EXF_GETSURFACEFLUXES, the SHORTWAVE_HEATING hflux, EXF_MAPFIELDS): no host
    work, traceable (myTime may be traced). Split from EXF_GETFORCING only at the boundary between its host part
    (the record reads) and the rest, so that a traced time loop can call this part (the reads preloaded).
    `mon` (a dict, optional): receives EXF_FIELDS.h as EXF_MONITOR (:380, after EXF_DIAGNOSTICS_FILL :377, output
    only) sees it; the driver prints the monitor block on the host (exf_monitor.exf_monitor)."""
    sz = cfg.size
    ks = sz.Nr if params.usingPCoords else 1
    if cfg.cpp.flag("ALLOW_DOWNWARD_RADIATION", "EXF_OPTIONS.h"):              # :259-262
        f = exf_radiation(exf_Tsf, myTime, myIter, f, cfg=cfg, exf=exf)
    if probe is not None:
        probe("X02_exf_radiation", f, ff)
    f = exf_wind(myTime, myIter, f, cfg=cfg, exf=exf, params=params, state=state,   # :266
                 ctrl=ctrl)                                                    # ctrl: lane M4ADCOL (gentim2d)
    if probe is not None:
        probe("X03_exf_wind", f, ff)
    if cfg.cpp.flag("ALLOW_ATM_TEMP", "EXF_OPTIONS.h") and cfg.cpp.flag("ALLOW_BULKFORMULAE", "EXF_OPTIONS.h"):
        f = exf_bulkformulae(exf_Tsf, myTime, myIter, f, cfg=cfg, exf=exf, params=params, state=state)   # :281
    if probe is not None:
        probe("X04_exf_bulkformulae", f, ff)
    f = dict(f)
    j = loop_j(1, sz.sNy)
    i = loop_i(1, sz.sNx)
    if cfg.cpp.flag("ALLOW_ATM_TEMP", "EXF_OPTIONS.h"):                        # :291-308
        if cfg.cpp.flag("SHORTWAVE_HEATING"):
            f["hflux"] = f["hflux"].at[i, j].set(- f["hs"][i, j]               # :296-302
                                                 - f["hl"][i, j]
                                                 + f["lwflux"][i, j])
        else:
            f["hflux"] = f["hflux"].at[i, j].set(- f["hs"][i, j]
                                                 - f["hl"][i, j]
                                                 + f["lwflux"][i, j]
                                                 + f["swflux"][i, j])
        f["sflux"] = f["sflux"].at[i, j].set(f["evap"][i, j] - f["precip"][i, j])   # :304
    if cfg.cpp.flag("ALLOW_RUNOFF", "EXF_OPTIONS.h"):                          # :312-314
        f["sflux"] = f["sflux"].at[i, j].set(f["sflux"][i, j] - f["runoff"][i, j])
    f["hflux"] = f["hflux"].at[i, j].set(f["hflux"][i, j]*grid.maskC[i, j, ks])   # :315
    f["sflux"] = f["sflux"].at[i, j].set(f["sflux"][i, j]*grid.maskC[i, j, ks])   # :316
    if exf.stressIsOnCgrid:                                                    # :328-332
        u, v = ex.EXCH_UV_XY_RL(f["ustress"].data, f["vstress"].data, True)
    else:
        u, v = ex.EXCH_UV_AGRID_3D_RL(f["ustress"].data, f["vstress"].data, True)
    f["ustress"], f["vstress"] = _rewrap(u, f["ustress"]), _rewrap(v, f["vstress"])
    if probe is not None:
        probe("X05_exf_hflux_sflux", f, ff)
    f = exf_getsurfacefluxes(myTime, myIter, f, cfg=cfg, ctrl=ctrl)            # :344 (ctrl: lane M4ADCOL)
    # :346-349 EXF_CHECK_RANGE (useExfCheckRange at nIter0): a range check that prints and STOPs; see the gate
    if cfg.cpp.flag("ALLOW_ATM_TEMP", "EXF_OPTIONS.h") and cfg.cpp.flag("SHORTWAVE_HEATING"):   # :357-375
        jh = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
        ih = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
        f["hflux"] = f["hflux"].at[ih, jh].set(f["hflux"][ih, jh] + f["swflux"][ih, jh])   # :368
    if mon is not None:                                                        # :380 EXF_MONITOR (host)
        mon.update(f)
    f, ff = exf_mapfields(myTime, myIter, f, ff, cfg=cfg, exf=exf, grid=grid, params=params, fp=fp, state=state,
                          ex=ex, tp=tp)                                        # :383
    if probe is not None:
        probe("X06_exf_mapfields", f, ff)
    return f, ff
