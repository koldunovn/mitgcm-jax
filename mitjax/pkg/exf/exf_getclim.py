"""EXF_GETCLIM: pkg/exf/exf_getclim.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.eesupp.exch_rs import _rewrap
from mitjax.farray import loop_i, loop_j
from mitjax.pkg.exf.exf_set_fld import exf_set_fld


def exf_getclim(myTime, myIter, f, *, cfg, exf, cal, grid, params, rw, tp, ex, rec=None, pre=None):
    """EXF_GETCLIM( myTime, myIter, myThid )   @63cdc0b pkg/exf/exf_getclim.F:3-180

    C     | o Get the climatological fields for the current time step.

    `f` EXF_FIELDS.h (dict). Returns the new dict. Ported: ALLOW_CLIMSST_RELAXATION (:70-97: EXF_SET_FLD, the
    climtempfreeze floor, the exchange) and ALLOW_CLIMSSS_RELAXATION (:99-114). Raises: ALLOW_CLIMSTRESS_RELAXATION,
    ALLOW_BULK_OFFLINE. `IF (climsst .LT. climtempfreeze)` is a REAL field test: a pointwise `where`.
    `rec` / `pre`: as in exf_set_fld (the traced time loop)."""
    for o in ("ALLOW_CLIMSTRESS_RELAXATION", "ALLOW_BULK_OFFLINE"):
        if cfg.cpp.flag(o, "EXF_OPTIONS.h"):
            raise NotImplementedError(f"EXF_GETCLIM: {o} is not ported")
    sz = cfg.size
    f = dict(f)
    kw = dict(cfg=cfg, exf=exf, cal=cal, grid=grid, params=params, rw=rw, tp=tp, rec=rec, pre=pre)
    if cfg.cpp.flag("ALLOW_CLIMSST_RELAXATION", "EXF_OPTIONS.h"):              # :70-97
        f["climsst"], f["climsst0"], f["climsst1"] = exf_set_fld(
            "climsst", exf.climsstfile, exf.climsstmask, exf.climsstStartTime, exf.climsstperiod,
            exf.climsstRepCycle, exf.exf_inscal_climsst, exf.climsst_exfremo_intercept, exf.climsst_exfremo_slope,
            f["climsst"], f["climsst0"], f["climsst1"], myTime, myIter, **kw)  # :72-82
        j = loop_j(1, sz.sNy)
        i = loop_i(1, sz.sNx)
        f["climsst"] = f["climsst"].at[i, j].set(                              # :88-90
            jnp.where(f["climsst"][i, j] < exf.climtempfreeze, exf.climtempfreeze, f["climsst"][i, j]))
        f["climsst"] = _rewrap(ex.EXCH_XY_RL(f["climsst"].data), f["climsst"])  # :96
    if cfg.cpp.flag("ALLOW_CLIMSSS_RELAXATION", "EXF_OPTIONS.h"):              # :99-114
        f["climsss"], f["climsss0"], f["climsss1"] = exf_set_fld(
            "climsss", exf.climsssfile, exf.climsssmask, exf.climsssStartTime, exf.climsssperiod,
            exf.climsssRepCycle, exf.exf_inscal_climsss, exf.climsss_exfremo_intercept, exf.climsss_exfremo_slope,
            f["climsss"], f["climsss0"], f["climsss1"], myTime, myIter, **kw)  # :101-111
        f["climsss"] = _rewrap(ex.EXCH_XY_RL(f["climsss"].data), f["climsss"])  # :113
    return f
