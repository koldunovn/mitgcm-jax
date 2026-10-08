"""EXF_INIT_FIXED: pkg/exf/exf_init_fixed.F @63cdc0b (the masks and the field start times)."""

from mitjax.pkg.exf.exf_getffield_start import exf_getffield_start

# EXF_INIT_FIXED's EXF_GETFFIELD_START calls in their order (:124-445), with the CPP option that compiles each and
# whether it needs useAtmWind (True), .NOT.useAtmWind (False) or neither (None)
_START_CALLS = (("uwind", None, True), ("vwind", None, True), ("wspeed", None, None),
                ("ustress", None, False), ("vstress", None, False), ("hflux", None, None), ("sflux", None, None),
                ("swflux", "ALLOW_ATM_TEMP|SHORTWAVE_HEATING", None), ("atemp", "ALLOW_ATM_TEMP", None),
                ("aqh", "ALLOW_ATM_TEMP", None), ("lwflux", "ALLOW_ATM_TEMP", None), ("precip", "ALLOW_ATM_TEMP", None),
                ("snowprecip", "ALLOW_ATM_TEMP", None), ("evap", "EXF_READ_EVAP", None), ("runoff", "ALLOW_RUNOFF", None),
                ("saltflx", "ALLOW_SALTFLX", None), ("swdown", "ALLOW_DOWNWARD_RADIATION", None),
                ("lwdown", "ALLOW_DOWNWARD_RADIATION", None), ("apressure", "ATMOSPHERIC_LOADING", None),
                ("areamask", "EXF_SEAICE_FRACTION", None),                    # :384-396 (lane M4ADLAB)
                ("climsst", "ALLOW_CLIMSST_RELAXATION", None), ("climsss", "ALLOW_CLIMSSS_RELAXATION", None))


def _compiled(cfg, opt):
    if opt is None:
        return True
    return any(cfg.cpp.flag(o, "EXF_OPTIONS.h") for o in opt.split("|"))


def exf_init_fixed(exf, *, cfg, cal, nIter0, startTime, stderr=None):
    """EXF_INIT_FIXED( myThid )   @63cdc0b pkg/exf/exf_init_fixed.F:3-628

    C     | o Routine to initialize EXF variables that are kept fixed during the run.

    Returns ExfParams with the field masks (:52-115) and every `<fld>StartTime` of a field read from a file
    (EXF_GETFFIELD_START, :124-445); errCount > 0 STOPs (:587-594). The ALLOW_ZENITHANGLE / USE_EXF_INTERPOLATION /
    ALLOW_OBCS / EXF_ALLOW_TIDES parts raise (not ported) when compiled and used. EXF_SEAICE_FRACTION (lane M4ADLAB
    session 3): areamask's start time (:384-396) when areamaskfile is set (EXF_READPARMS refuses a file)."""
    if cfg.cpp.flag("USE_EXF_INTERPOLATION", "EXF_OPTIONS.h"):
        raise NotImplementedError("EXF_INIT_FIXED: USE_EXF_INTERPOLATION is not ported")
    if cfg.cpp.flag("ALLOW_OBCS") and cfg.use_flag("useOBCS"):
        raise NotImplementedError("EXF_INIT_FIXED: useOBCS (obcs start times) is not ported")
    m = {}
    errCount = 0                                                               # :50
    for n in ("hflux", "sflux", "atemp", "aqh", "hs_", "hl_", "evap", "precip", "snowprecip", "runoff",   # hs_mask
              "saltflx"):                                                      # :52-62
        m[n + "mask"] = "c"
    if exf.stressIsOnCgrid:                                                    # :63-69
        m["ustressmask"], m["vstressmask"] = "w", "s"
    else:
        m["ustressmask"], m["vstressmask"] = "c", "c"
    for n in ("uwind", "vwind", "wspeed", "swflux", "lwflux", "swdown", "lwdown", "apressure", "tidePot",
              "areamask", "climsst", "climsss"):                               # :70-81
        m[n + "mask"] = "c"
    m["climustrmask"], m["climvstrmask"] = "w", "s"                            # :82-83
    if cfg.use_flag("useSEAICE"):                                              # :86-115
        for n in ("hflux", "sflux", "ustress", "vstress", "uwind", "vwind", "wspeed", "swflux", "swdown",
                  "apressure", "climustr", "climvstr"):
            m[n + "mask"] = " "
    st = {}
    for fld, opt, wind in _START_CALLS:
        if not _compiled(cfg, opt):
            continue
        if wind is not None and bool(exf.useAtmWind) != wind:
            continue
        if not getattr(exf, fld + "file").strip():
            continue
        t, errCount = exf_getffield_start(exf.useExfYearlyFields, "exf", fld, getattr(exf, fld + "period"),
                                          getattr(exf, fld + "startdate1"), getattr(exf, fld + "startdate2"),
                                          getattr(exf, fld + "StartTime"), errCount, useCAL=cfg.use_flag("useCAL"),
                                          cal=cal, nIter0=nIter0, startTime=startTime, stderr=stderr)
        st[fld + "StartTime"] = float(t)
    if errCount > 0:                                                           # :587-594
        raise RuntimeError(f"EXF_INIT_FIXED: found {errCount} errors in EXF_GETFFIELD_START\n"
                           "ABNORMAL END: S/R EXF_INIT_FIXED")
    return exf.replace(**m, **st)
