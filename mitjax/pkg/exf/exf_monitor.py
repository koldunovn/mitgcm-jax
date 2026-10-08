"""EXF_MONITOR: pkg/exf/exf_monitor.F @63cdc0b (host side, through pkg/monitor: an output routine; its records go to
`mon.units[mon.mon_ioUnit]` as MONITOR's)."""

from mitjax.eesupp.different_multiple import different_multiple
from mitjax.pkg.monitor.mon_out import mon_out_i, mon_out_rl
from mitjax.pkg.monitor.mon_set_pref import mon_set_pref
from mitjax.pkg.monitor.mon_writestats_rl import mon_writestats_rl
from mitjax.pkg.monitor.monitor import level_view
from mitjax.pkg.monitor.monitor_h import mon_string_none
from mitjax.pkg.ptracers.ptracers_monitor import _banner


def _blank(s):
    return str(s).strip() == ""


def exf_monitor(myTime, myIter, f, *, cfg, exf, exf_monFreq, deltaTClock, grid, mon, ex=None):
    """EXF_MONITOR( myTime, myIter, myThid )   @63cdc0b pkg/exf/exf_monitor.F:9-269

    C     | o Do EXF monitor output

    `f` EXF_FIELDS.h as EXF_GETFORCING holds it at the call (exf_getforcing.F:380; concrete arrays), `cfg` the monitor
    configuration (Nr, monitor_stdio, ALLOW_MNC, useMNC, monitor_mnc), `exf` ExfParams, `exf_monFreq`
    (EXF_PARAM.h: data.exf's or monitorFreq, exf_readparms.F:304), `grid` (maskInC, maskInW, maskInS, rA, rAw,
    rAs, drF), `mon` MONITOR.h. MASTER_CPU_IO is true (one process). computed = ALLOW_BULKFORMULAE (:58-62).
    Raises: output to MNC (useMNC with monitor_mnc, :77-91), the BLING / ALLOW_RUNOFTEMP /
    ALLOW_CLIMSTRESS_RELAXATION fields when compiled. EXF_SEAICE_FRACTION (lane M4ADLAB session 3): its statistics
    (:207-212) only with an areamaskfile, which EXF_READPARMS refuses (raises here too)."""
    if not different_multiple(exf_monFreq, myTime, deltaTClock):              # :64-65
        return
    for o in ("ALLOW_RUNOFTEMP", "ALLOW_CLIMSTRESS_RELAXATION"):
        if exf_cfg_flag(cfg, o):
            raise NotImplementedError(f"EXF_MONITOR: the {o} fields are not ported")
    if exf_cfg_flag(cfg, "EXF_SEAICE_FRACTION") and exf.areamaskfile.strip():   # :207-212
        raise NotImplementedError("EXF_MONITOR: the areamask statistics (areamaskfile) are not ported")
    computed = exf_cfg_flag(cfg, "ALLOW_BULKFORMULAE")                         # :58-62
    mon.mon_write_stdout = bool(cfg.monitor_stdio)                             # :71-75
    mon.mon_write_mnc = False                                                  # :76
    if cfg.ALLOW_MNC and cfg.useMNC and cfg.monitor_mnc:                       # :77-91
        raise NotImplementedError("EXF_MONITOR: monitor output to MNC is not ported")
    if mon.mon_write_stdout:                                                   # :93-102
        _banner("// Begin MONITOR EXF statistics", mon)
    mon_set_pref("exf", mon=mon)                                               # :107
    mon_out_i("_tsnumber", myIter, mon_string_none, cfg=cfg, mon=mon)          # :108
    mon_out_rl("_time_sec", myTime, mon_string_none, cfg=cfg, mon=mon)         # :109
    d = [0.0] * 6
    g = grid

    def stats(name, arr, mask, area):
        # the 2-D fields go to dummies declared (...,myNr,nSx,nSy) with myNr = 1 (monitor.level_view)
        mon_writestats_rl(1, level_view(arr), name, level_view(mask), mask, area, g.drF, d, cfg=cfg, mon=mon,
                          ex=ex)

    if exf.stressIsOnCgrid:                                                    # :112-122
        stats("_ustress", f["ustress"], g.maskInW, g.rAw)
        stats("_vstress", f["vstress"], g.maskInS, g.rAs)
    else:
        stats("_ustress", f["ustress"], g.maskInC, g.rA)
        stats("_vstress", f["vstress"], g.maskInC, g.rA)
    if computed or not _blank(exf.hfluxfile):                                  # :123-126
        stats("_hflux", f["hflux"], g.maskInC, g.rA)
    if computed or not _blank(exf.sfluxfile):                                  # :127-130
        stats("_sflux", f["sflux"], g.maskInC, g.rA)
    if exf.useAtmWind:                                                         # :131-136
        stats("_uwind", f["uwind"], g.maskInC, g.rA)
        stats("_vwind", f["vwind"], g.maskInC, g.rA)
    if computed or not _blank(exf.wspeedfile):                                 # :137-140
        stats("_wspeed", f["wspeed"], g.maskInC, g.rA)
    if exf_cfg_flag(cfg, "ALLOW_ATM_TEMP"):                                    # :141-166
        if not _blank(exf.atempfile):
            stats("_atemp", f["atemp"], g.maskInC, g.rA)
        if not _blank(exf.aqhfile):
            stats("_aqh", f["aqh"], g.maskInC, g.rA)
        if not _blank(exf.lwdownfile) or not _blank(exf.lwfluxfile):
            stats("_lwflux", f["lwflux"], g.maskInC, g.rA)
        if computed or not _blank(exf.evapfile):
            stats("_evap", f["evap"], g.maskInC, g.rA)
        if not _blank(exf.precipfile):
            stats("_precip", f["precip"], g.maskInC, g.rA)
        if not _blank(exf.snowprecipfile):
            stats("_snowprecip", f["snowprecip"], g.maskInC, g.rA)
    if exf_cfg_flag(cfg, "ALLOW_ATM_TEMP") or cfg.SHORTWAVE_HEATING:          # :167-172
        if not _blank(exf.swdownfile) or not _blank(exf.swfluxfile):
            stats("_swflux", f["swflux"], g.maskInC, g.rA)
    if exf_cfg_flag(cfg, "ALLOW_DOWNWARD_RADIATION"):                          # :173-182
        if not _blank(exf.swdownfile):
            stats("_swdown", f["swdown"], g.maskInC, g.rA)
        if not _blank(exf.lwdownfile):
            stats("_lwdown", f["lwdown"], g.maskInC, g.rA)
    if cfg.ATMOSPHERIC_LOADING:                                                # :183-188
        if not _blank(exf.apressurefile):
            stats("_apressure", f["apressure"], g.maskInC, g.rA)
    if exf_cfg_flag(cfg, "ALLOW_RUNOFF"):                                      # :189-194
        if not _blank(exf.runofffile):
            stats("_runoff", f["runoff"], g.maskInC, g.rA)
    if exf_cfg_flag(cfg, "ALLOW_SALTFLX"):                                     # :201-206
        if not _blank(exf.saltflxfile):
            stats("_saltflx", f["saltflx"], g.maskInC, g.rA)
    if exf_cfg_flag(cfg, "ALLOW_BLING"):                                       # :213-218
        raise NotImplementedError("EXF_MONITOR: ALLOW_BLING (apco2) is not ported")
    if exf_cfg_flag(cfg, "ALLOW_CLIMSST_RELAXATION"):                          # :219-224
        if not _blank(exf.climsstfile):
            stats("_climsst", f["climsst"], g.maskInC, g.rA)
    if exf_cfg_flag(cfg, "ALLOW_CLIMSSS_RELAXATION"):                          # :225-230
        if not _blank(exf.climsssfile):
            stats("_climsss", f["climsss"], g.maskInC, g.rA)
    if mon.mon_write_stdout:                                                   # :246-255
        _banner("// End MONITOR EXF statistics", mon)
    mon.mon_write_stdout = False                                               # :257-258
    mon.mon_write_mnc = False


def exf_cfg_flag(cfg, name):
    """A CPP flag of the build as EXF_MONITOR's compilation sees it (`cfg.exf_flags`: the EXF_OPTIONS.h / CPP_OPTIONS.h
    flags the caller resolved)."""
    return bool(cfg.exf_flags.get(name, False))
