"""EXF_GETFFIELDS: pkg/exf/exf_getffields.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX
from mitjax.pkg.exf.exf_set_fld import exf_set_fld
from mitjax.pkg.exf.exf_set_uv import exf_set_uv


def _set(f, n, exf, myTime, myIter, kw):
    f[n], f[n + "0"], f[n + "1"] = exf_set_fld(
        n, getattr(exf, n + "file"), getattr(exf, n + "mask"), getattr(exf, n + "StartTime"),
        getattr(exf, n + "period"), getattr(exf, n + "RepCycle"), getattr(exf, "exf_inscal_" + n),
        getattr(exf, n + "_exfremo_intercept"), getattr(exf, n + "_exfremo_slope"), f[n], f[n + "0"], f[n + "1"],
        myTime, myIter, **kw)


def exf_getffields(myTime, myIter, f, *, cfg, exf, cal, grid, params, rw, tp, rec=None, pre=None, ctrl=None):
    """EXF_GETFFIELDS( myTime, myIter, myThid )   @63cdc0b pkg/exf/exf_getffields.F:3-582

    C     | o Read-in atmospheric state and/or surface fluxes from files.

    `f` EXF_FIELDS.h (dict). Returns the new dict. Host routine for the record bookkeeping (EXF_SET_FLD).
    Ported: useAtmWind (:126-145; :78-109 stress = 0), wspeed, hflux, sflux, ALLOW_ATM_TEMP (atemp + exf_offset_atemp
    :188-207, aqh, lwflux, precip, snowprecip, the precip/snowprecip MAX :314-325), swflux, ALLOW_DOWNWARD_RADIATION
    (swdown, lwdown), ATMOSPHERIC_LOADING (apressure), ALLOW_RUNOFF (runoff), ALLOW_SALTFLX (saltflx).
    Lane M4CS32ICE: useAtmWind = .FALSE. (EXF_SET_UV of ustress/vstress :78-97, uwind = vwind = 0 :146-157),
    EXF_ALLOW_TIDES (tidePot, :390-404; no tide file: EXF_SET_FLD returns at its :120) and ALLOW_RUNOFTEMP
    (runoftemp with runoff's mask / start / period / cycle, :437-452).
    Raises: ALLOW_READ_TURBFLUXES, EXF_READ_EVAP,
    ALLOW_ROTATE_UV_CONTROLS with useCTRL (:468-483, :549-577). Lane M4ADLAB session 3: EXF_SEAICE_FRACTION
    (areamask, :406-419; no areamask file: EXF_SET_FLD returns at its :120) and its gentim2d control xx_areamask
    (:536-540).
    Lane M4ADCOL (1D_ocean_ice_column/input_ad): ALLOW_GENTIM2D_CONTROL with useCTRL (:484-561): `ctrl` = dict(
    xx_gentim2d={iarr: FArray} (CTRL_GENARR.h after this step's CTRL_MAP_GENTIM2D), files={iarr: xx_gentim2d_file})
    adds the control of every iarr whose file name starts with the field's control name to the field on the
    interior points (i, j = 1..sNx, sNy), iarr = 1..maxCtrlTim2D in order. The host record capture (`rec`, the EXF
    preload) runs without it: the controls enter no EXF_SET_FLD record (every field is re-interpolated from its
    records each step, exf_set_fld.F), so the traced step adds them (exf_preload.exf_getforcing_preloaded).
    `rec` / `pre`: the traced time loop's record capture / preloaded records of every EXF_SET_FLD call (exf_set_fld)."""
    def opt(n):
        return cfg.cpp.flag(n, "EXF_OPTIONS.h")
    for o in ("ALLOW_READ_TURBFLUXES", "EXF_READ_EVAP"):
        if opt(o):
            raise NotImplementedError(f"EXF_GETFFIELDS: {o} is not ported")
    use_ctrl = cfg.cpp.flag("ALLOW_CTRL") and cfg.use_flag("useCTRL")          # :468-577
    if use_ctrl and cfg.cpp.flag("ALLOW_ROTATE_UV_CONTROLS", "CTRL_OPTIONS.h"):
        raise NotImplementedError("EXF_GETFFIELDS: ALLOW_ROTATE_UV_CONTROLS (:468-483, :549-577) is not ported")
    gentim = use_ctrl and cfg.cpp.flag("ALLOW_GENTIM2D_CONTROL", "CTRL_OPTIONS.h")
    if gentim and ctrl is None and rec is None:
        raise ValueError("EXF_GETFFIELDS: useCTRL with ALLOW_GENTIM2D_CONTROL needs `ctrl` (xx_gentim2d)")
    sz = cfg.size
    f = dict(f)
    kw = dict(cfg=cfg, exf=exf, cal=cal, grid=grid, params=params, rw=rw, tp=tp, rec=rec, pre=pre)
    jh = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    ih = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    if not exf.useAtmWind:                                                     # :78-97
        (f["ustress"], f["ustress0"], f["ustress1"], f["vstress"], f["vstress0"], f["vstress1"]) = exf_set_uv(
            "ustress", exf.ustressfile, exf.ustressmask, exf.ustressStartTime, exf.ustressperiod,
            exf.ustressRepCycle, exf.exf_inscal_ustress, exf.ustress_exfremo_intercept, exf.ustress_exfremo_slope,
            f["ustress"], f["ustress0"], f["ustress1"],
            "vstress", exf.vstressfile, exf.vstressmask, exf.vstressStartTime, exf.vstressperiod,
            exf.vstressRepCycle, exf.exf_inscal_vstress, exf.vstress_exfremo_intercept, exf.vstress_exfremo_slope,
            f["vstress"], f["vstress0"], f["vstress1"], myTime, myIter, **kw)
    else:                                                                      # :98-109
        f["ustress"] = f["ustress"].at[ih, jh].set(0.)
        f["vstress"] = f["vstress"].at[ih, jh].set(0.)
    _set(f, "wspeed", exf, myTime, myIter, kw)                                 # :112-123
    if exf.useAtmWind:                                                         # :126-145
        (f["uwind"], f["uwind0"], f["uwind1"], f["vwind"], f["vwind0"], f["vwind1"]) = exf_set_uv(
            "uwind", exf.uwindfile, exf.uwindmask, exf.uwindStartTime, exf.uwindperiod, exf.uwindRepCycle,
            exf.exf_inscal_uwind, exf.uwind_exfremo_intercept, exf.uwind_exfremo_slope,
            f["uwind"], f["uwind0"], f["uwind1"],
            "vwind", exf.vwindfile, exf.vwindmask, exf.vwindStartTime, exf.vwindperiod, exf.vwindRepCycle,
            exf.exf_inscal_vwind, exf.vwind_exfremo_intercept, exf.vwind_exfremo_slope,
            f["vwind"], f["vwind0"], f["vwind1"], myTime, myIter, **kw)
    else:                                                                      # :146-157
        f["uwind"] = f["uwind"].at[ih, jh].set(0.)
        f["vwind"] = f["vwind"].at[ih, jh].set(0.)
    _set(f, "hflux", exf, myTime, myIter, kw)                                  # :160-170
    _set(f, "sflux", exf, myTime, myIter, kw)                                  # :173-183
    j = loop_j(1, sz.sNy)
    i = loop_i(1, sz.sNx)
    if opt("ALLOW_ATM_TEMP"):                                                  # :185-327
        _set(f, "atemp", exf, myTime, myIter, kw)                              # :188-198
        f["atemp"] = f["atemp"].at[i, j].set(f["atemp"][i, j] + exf.exf_offset_atemp)   # :203
        _set(f, "aqh", exf, myTime, myIter, kw)                                # :210-220
        _set(f, "lwflux", exf, myTime, myIter, kw)                             # :253-263
        _set(f, "precip", exf, myTime, myIter, kw)                             # :281-291
        _set(f, "snowprecip", exf, myTime, myIter, kw)                         # :294-306
        if exf.snowprecipfile.strip():                                         # :314-325
            f["precip"] = f["precip"].at[i, j].set(MAX(f["precip"][i, j], f["snowprecip"][i, j], p="b"))   # :320
    if opt("ALLOW_ATM_TEMP") or cfg.cpp.flag("SHORTWAVE_HEATING"):             # :329-342
        _set(f, "swflux", exf, myTime, myIter, kw)
    if opt("ALLOW_DOWNWARD_RADIATION"):                                        # :344-372
        _set(f, "swdown", exf, myTime, myIter, kw)                             # :347-357
        _set(f, "lwdown", exf, myTime, myIter, kw)                             # :360-370
    if cfg.cpp.flag("ATMOSPHERIC_LOADING"):                                    # :374-388
        _set(f, "apressure", exf, myTime, myIter, kw)
    if opt("EXF_ALLOW_TIDES"):                                                 # :390-404
        f["tidePot"], f["tidePot0"], f["tidePot1"] = exf_set_fld(
            "tidePot", exf.tidePotfile, exf.tidePotmask, exf.tidePotStartTime, exf.tidePotperiod,
            exf.tidePotRepCycle, exf.exf_inscal_tidePot, exf.tidePot_exfremo_intercept, exf.tidePot_exfremo_slope,
            f["tidePot"], f["tidePot0"], f["tidePot1"], myTime, myIter, **kw)
    if opt("EXF_SEAICE_FRACTION"):                                             # :406-419 (lane M4ADLAB)
        _set(f, "areamask", exf, myTime, myIter, kw)
    if opt("ALLOW_RUNOFF"):                                                    # :422-435
        _set(f, "runoff", exf, myTime, myIter, kw)
    if opt("ALLOW_RUNOFTEMP"):                                                 # :437-450
        f["runoftemp"], f["runoftemp0"], f["runoftemp1"] = exf_set_fld(
            "runoftemp", exf.runoftempfile, exf.runoffmask, exf.runoffStartTime, exf.runoffperiod,
            exf.runoffRepCycle, exf.exf_inscal_runoftemp, exf.runoftemp_exfremo_intercept,
            exf.runoftemp_exfremo_slope, f["runoftemp"], f["runoftemp0"], f["runoftemp1"], myTime, myIter, **kw)
    if opt("ALLOW_SALTFLX"):                                                   # :452-466
        _set(f, "saltflx", exf, myTime, myIter, kw)
    if gentim and ctrl is not None:                                            # :484-561 (lane M4ADCOL)
        f = _gentim2d_controls(f, ctrl, cfg=cfg, opt=opt)
    return f


def _gentim2d_controls(f, ctrl, *, cfg, opt):
    """exf_getffields.F:484-561 (#if ALLOW_CTRL && ALLOW_GENTIM2D_CONTROL, IF ( useCTRL ), no
    ALLOW_ROTATE_UV_CONTROLS): `fld(i,j) = fld(i,j) + xx_gentim2d(i,j,iarr)` for i, j = 1..sNx, sNy where
    `xx_gentim2d_file(iarr)(1:n) .EQ. 'xx_<fld>'` (a test of the first n characters of the file name)."""
    sz = cfg.size
    j = loop_j(1, sz.sNy)
    i = loop_i(1, sz.sNx)
    tests = []
    if opt("ALLOW_ATM_TEMP"):                                                  # :491-507
        tests += [("xx_atemp", "atemp"), ("xx_aqh", "aqh"), ("xx_precip", "precip"),
                  ("xx_snowprecip", "snowprecip"), ("xx_lwflux", "lwflux")]
    if opt("ALLOW_ATM_TEMP") or cfg.cpp.flag("SHORTWAVE_HEATING"):            # :508-512
        tests += [("xx_swflux", "swflux")]
    if opt("ALLOW_DOWNWARD_RADIATION"):                                        # :513-520
        tests += [("xx_swdown", "swdown"), ("xx_lwdown", "lwdown")]
    if opt("ALLOW_RUNOFF"):                                                    # :521-525
        tests += [("xx_runoff", "runoff")]
    if opt("EXF_READ_EVAP"):                                                   # :526-530 (raises above)
        raise NotImplementedError("EXF_GETFFIELDS: the xx_evap control is not ported")
    if cfg.cpp.flag("ATMOSPHERIC_LOADING"):                                    # :531-535
        tests += [("xx_apressure", "apressure")]
    if opt("EXF_SEAICE_FRACTION"):                                             # :536-540 (lane M4ADLAB)
        tests += [("xx_areamask", "areamask")]
    tests += [("xx_uwind", "uwind"), ("xx_vwind", "vwind")]                    # :541-548
    f = dict(f)
    for iarr in sorted(ctrl["xx_gentim2d"]):                                   # DO iarr = 1, maxCtrlTim2D
        fname = ctrl["files"][iarr]
        for cname, fld in tests:
            if fname[:len(cname)] == cname:                                    # (1:n) .EQ. 'xx_...'
                f[fld] = f[fld].at[i, j].set(f[fld][i, j] + ctrl["xx_gentim2d"][iarr][i, j])
    return f
