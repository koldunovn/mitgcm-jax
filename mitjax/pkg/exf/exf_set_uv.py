"""EXF_SET_UV: pkg/exf/exf_set_uv.F @63cdc0b (the no-interpolation branch)."""

from mitjax.pkg.exf.exf_set_fld import exf_set_fld


def exf_set_uv(uVecName, uVecFile, uVecMask, uVecStartTime, uVecPeriod, uVecRepeatCycle, uVec_inScale,
               uVec_remove_intercept, uVec_remove_slope, uVec, uVec0, uVec1,
               vVecName, vVecFile, vVecMask, vVecStartTime, vVecPeriod, vVecRepeatCycle, vVec_inScale,
               vVec_remove_intercept, vVec_remove_slope, vVec, vVec0, vVec1, myTime, myIter, *, cfg, exf, cal, grid,
               params, rw, tp, rec=None, pre=None):
    """EXF_SET_UV( uVecName, uVecFile, ..., uVec, uVec0, uVec1, vVecName, ..., vVec, vVec0, vVec1, myTime, myIter,
    myThid )   @63cdc0b pkg/exf/exf_set_uv.F:3-595

    C  | o Read-in, interpolate, and rotate wind or wind stress vectors from a spherical-polar input grid to an
    C  |   arbitrary output grid.

    Without USE_EXF_INTERPOLATION the routine is the `IF ( .TRUE. )` branch (:535-593): EXF_SET_FLD for each
    component (:538-557), then the A-grid rotation when rotateStressOnAgrid (:559-591; raises in EXF_READPARMS).
    Returns (uVec, uVec0, uVec1, vVec, vVec0, vVec1)."""
    if cfg.cpp.flag("USE_EXF_INTERPOLATION", "EXF_OPTIONS.h"):
        raise NotImplementedError("EXF_SET_UV: USE_EXF_INTERPOLATION is not ported")
    kw = dict(cfg=cfg, exf=exf, cal=cal, grid=grid, params=params, rw=rw, tp=tp, rec=rec, pre=pre)
    uVec, uVec0, uVec1 = exf_set_fld(uVecName, uVecFile, uVecMask, uVecStartTime, uVecPeriod, uVecRepeatCycle,
                                     uVec_inScale, uVec_remove_intercept, uVec_remove_slope, uVec, uVec0, uVec1,
                                     myTime, myIter, **kw)                     # :538-547
    vVec, vVec0, vVec1 = exf_set_fld(vVecName, vVecFile, vVecMask, vVecStartTime, vVecPeriod, vVecRepeatCycle,
                                     vVec_inScale, vVec_remove_intercept, vVec_remove_slope, vVec, vVec0, vVec1,
                                     myTime, myIter, **kw)                     # :548-557
    if exf.rotateStressOnAgrid:                                                # :559-591
        raise NotImplementedError("EXF_SET_UV: rotateStressOnAgrid is not ported")
    return uVec, uVec0, uVec1, vVec, vVec0, vVec1
