"""EXF_SET_FLD: pkg/exf/exf_set_fld.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j
from mitjax.pkg.exf.exf_filter_rl import exf_filter_rl
from mitjax.pkg.exf.exf_getffieldrec import exf_GetFFieldRec
from mitjax.pkg.exf.exf_param_h import exf_one
from mitjax.pkg.exf.exf_swapffields import exf_SwapFFields


def exf_set_fld(fldName, fldFile, fldMask, fldStartTime, fldPeriod, fldRepeatCycle, fld_inScale,
                fldRemove_intercept, fldRemove_slope, fldArr, fld0, fld1, myTime, myIter, *, cfg, exf, cal, grid,
                params, rw, tp, rec=None, pre=None):
    """EXF_SET_FLD( fldName, fldFile, fldMask, fldStartTime, fldPeriod, fldRepeatCycle, fld_inScale,
    fldRemove_intercept, fldRemove_slope, fldArr, fld0, fld1, myTime, myIter, myThid )
    @63cdc0b pkg/exf/exf_set_fld.F:3-319

    C  | o Set value of one generic external forcing field

    Returns (fldArr, fld0, fld1). The record bookkeeping (EXF_GETFFIELDREC, the reads READ_REC_3D_RL, EXF_FILTER_RL,
    EXF_SWAPFFIELDS) runs on the host with the concrete myTime/myIter, as the Fortran's I/O; the time interpolation
    (:300-314) is the jnp statement on the record arrays with the host weight `fac`. `fldStartTime`, `fldPeriod`,
    `fldRepeatCycle` are host values (the setup's ExfParams). ALLOW_GENTIM2D_CONTROL (:117-119), fldPeriod -12/-1
    (monthly records, :133-153) and USE_EXF_INTERPOLATION are not ported (raise).

    Traced time loop (session 3 wiring): `rec` (host, a dict) receives {fldName: (fac, fld0, fld1)} after the record
    bookkeeping of this call (exf_preload.exf_preload records them per step); `pre` (the step's preloaded
    {fldName: (fac, fld0, fld1)}, traced) replaces the bookkeeping :120-297: fld0 / fld1 / fac are this step's, then
    the interpolation :300-314 runs on them as below. A field that :120 skips is in neither."""
    # :117-121 (exf_set_fld.F includes CTRL_OPTIONS.h after EXF_OPTIONS.h under ALLOW_CTRL): with
    # ALLOW_GENTIM2D_CONTROL the test is `fldFile .NE. ' '` (:118) instead of :120. Lane M4ADCOL: the two agree
    # unless a field has a file and fldPeriod = 0 (read every step), which is not ported (raises)
    # (checked on the host bookkeeping pass, pre is None: the preload runs it for every step)
    if (pre is None and cfg.cpp.flag("ALLOW_CTRL") and cfg.cpp.flag("ALLOW_GENTIM2D_CONTROL", "CTRL_OPTIONS.h")
            and fldFile.strip() and float(fldPeriod) == 0.):
        raise NotImplementedError("EXF_SET_FLD: ALLOW_GENTIM2D_CONTROL with a file of period 0 (:118) is not ported")
    if pre is None and not (fldFile.strip() and float(fldPeriod) != 0.):       # :120
        return fldArr, fld0, fld1
    if pre is not None:                                                        # preloaded (docstring)
        if fldName not in pre:                # :120 decided on the host by the preload (fldPeriod is traced here)
            return fldArr, fld0, fld1
        fac, fld0, fld1 = pre[fldName]
        return _interpolate(fldArr, fld0, fld1, fac, fld_inScale, fldRemove_intercept, fldRemove_slope, myTime,
                            cfg=cfg, tp=tp)
    useCAL = cfg.use_flag("useCAL")
    if useCAL and float(fldPeriod) == -12.:                                    # :133-141
        raise NotImplementedError("EXF_SET_FLD: fldPeriod = -12 (cal_GetMonthsRec) is not ported")
    elif useCAL and float(fldPeriod) == -1.:                                   # :142-153
        raise NotImplementedError("EXF_SET_FLD: fldPeriod = -1 (EXF_GetMonthsRec) is not ported")
    elif float(fldPeriod) < 0.:                                                # :154-160
        raise RuntimeError(f"EXF_SET_FLD: \"{fldName}\", Invalid fldPeriod={float(fldPeriod):16.8E} for file: "
                           f"{fldFile.strip()}\nABNORMAL END: S/R EXF_SET_FLD")
    else:                                                                      # :161-169
        fac, first, changed, count0, count1, year0, year1 = exf_GetFFieldRec(
            fldStartTime, fldPeriod, fldRepeatCycle, fldName, exf.useExfYearlyFields, myTime, myIter,
            useCAL=useCAL, cal=cal, nIter0=tp.nIter0, deltaTClock=tp.deltaTClock)
    if first:                                                                  # :184-240
        locFile0 = fldFile                                                     # exf_GetYearlyFieldName (no yearly)
        fld1 = rw.READ_REC_3D_RL(locFile0, exf.exf_iprec, 1, fld1, count0, myIter)   # :230-231
        fld1 = exf_filter_rl(fld1, fldMask, cfg=cfg, grid=grid, params=params)     # :237
    if first or changed:                                                       # :242-297
        fld0, fld1 = exf_SwapFFields(fld0, fld1, cfg=cfg)                      # :243
        locFile1 = fldFile
        fld1 = rw.READ_REC_3D_RL(locFile1, exf.exf_iprec, 1, fld1, count1, myIter)   # :287-288
        fld1 = exf_filter_rl(fld1, fldMask, cfg=cfg, grid=grid, params=params)     # :294
    if rec is not None:
        rec[fldName] = (fac, fld0, fld1)
    return _interpolate(fldArr, fld0, fld1, fac, fld_inScale, fldRemove_intercept, fldRemove_slope, myTime,
                        cfg=cfg, tp=tp)


def _interpolate(fldArr, fld0, fld1, fac, fld_inScale, fldRemove_intercept, fldRemove_slope, myTime, *, cfg, tp):
    """exf_set_fld.F:300-314: the time interpolation between the two records and the trend removal."""
    sz = cfg.size
    j = loop_j(1, sz.sNy)
    i = loop_i(1, sz.sNx)
    fldArr = fldArr.at[i, j].set(fld_inScale * (fac * fld0[i, j]                # :305-307
                                                + (exf_one - fac)*fld1[i, j]))
    fldArr = fldArr.at[i, j].set(fldArr[i, j]                                  # :308-310
                                 - fld_inScale*(fldRemove_intercept
                                                + fldRemove_slope*(myTime-tp.startTime)))
    return fldArr, fld0, fld1
