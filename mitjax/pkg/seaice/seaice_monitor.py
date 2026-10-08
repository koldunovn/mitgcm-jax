"""SEAICE_MONITOR: pkg/seaice/seaice_monitor.F @63cdc0b (host side, through pkg/monitor: an output routine; its
records go to `mon.units[mon.mon_ioUnit]` as MONITOR's). Called by SEAICE_OUTPUT (seaice_output.F:171), which
DO_THE_MODEL_IO calls with useSEAICE (do_the_model_io.F:192-196); its other work (SEAICE_dumpFreq snapshots,
seaice_output.F:59-169) writes files only (SEAICE_dumpFreq = dumpFreq = 0 in the M4 column: nothing)."""

from mitjax.eesupp.different_multiple import different_multiple
from mitjax.pkg.monitor.mon_out import mon_out_i, mon_out_rl
from mitjax.pkg.monitor.mon_set_pref import mon_set_pref
from mitjax.pkg.monitor.mon_writestats_rl import mon_writestats_rl
from mitjax.pkg.monitor.monitor import level_view
from mitjax.pkg.monitor.monitor_h import mon_string_none
from mitjax.pkg.ptracers.ptracers_monitor import _banner


def seaice_monitor(myTime, myIter, sf, *, cfg, sp, SEAICE_monFreq, deltaTClock, grid, mon, useThSIce=False, ex=None):
    """SEAICE_MONITOR( myTime, myIter, myThid )   @63cdc0b pkg/seaice/seaice_monitor.F:8-163

    C     | o Do SEAICE monitor output

    `sf` SEAICE.h (concrete arrays), `cfg` the monitor configuration (Nr, ALLOW_MNC, useMNC, seaice_flags: the
    SEAICE_OPTIONS.h flags), `SEAICE_monFreq` (SEAICE_PARAMS.h: data.seaice's or monitorFreq,
    seaice_readparms.F:559), SEAICE_mon_stdio = .TRUE. (:1438; :1443-1444 only with useMNC), `grid` (maskInC,
    rA, rAz, drF; lane M4OFF: maskInW, maskInS, rAw, rAs; UVM is SEAICE_GRID.h, in `sf`), `mon` MONITOR.h.
    MASTER_CPU_IO is true (one process). SEAICE_CGRID (lane M4OFF): UICE/VICE on maskInW/maskInS (:102-108).
    ALLOW_SITRACER (lane M4LAB session 4): the SItracer statistics of the tracers in use (:127-134). Raises: output
    to MNC (useMNC with SEAICE_mon_mnc = monitor_mnc, :69-82)."""
    if not different_multiple(SEAICE_monFreq, myTime, deltaTClock):             # :55-56
        return
    flags = cfg.seaice_flags
    # SEAICE_mon_mnc = monitor_mnc (seaice_readparms.F:563, ALLOW_MNC; :566 .FALSE. otherwise) unless data.seaice
    # sets it (lane M4ADCOL: 1D_ocean_ice_column/input_ad sets .TRUE.; sp.SEAICE_mon_mnc, None = the default).
    # Lane M4LAB session 4: useMNC with monitor_mnc .FALSE. (lab_sea's data.mnc) keeps SEAICE_mon_stdio .TRUE.
    # (:1440-1445) and writes no MNC monitor (:69-82); without useMNC the value is never read
    mon_mnc = getattr(sp, "SEAICE_mon_mnc", None)
    mon_mnc = cfg.monitor_mnc if mon_mnc is None else mon_mnc
    if cfg.ALLOW_MNC and cfg.useMNC and mon_mnc:                                # :69-82 / readparms :1440-1445
        raise NotImplementedError("SEAICE_MONITOR: useMNC with SEAICE_mon_mnc (MNC monitor output) is not ported")
    SEAICE_mon_stdio = True                                                     # seaice_readparms.F:1438
    mon.mon_write_stdout = SEAICE_mon_stdio                                     # :62-66
    mon.mon_write_mnc = False                                                   # :67
    if mon.mon_write_stdout:                                                    # :84-93
        _banner("// Begin MONITOR SEAICE statistics", mon)
    mon_set_pref("seaice", mon=mon)                                             # :98
    mon_out_i("_tsnumber", myIter, mon_string_none, cfg=cfg, mon=mon)           # :99
    mon_out_rl("_time_sec", myTime, mon_string_none, cfg=cfg, mon=mon)          # :100
    d = [0.0] * 6
    g = grid

    def stats(arr, suff, mask, area):
        # the 2-D fields go to dummies declared (...,myNr,nSx,nSy) with myNr = 1 (monitor.level_view)
        mon_writestats_rl(1, level_view(arr), suff, level_view(mask), mask, area, g.drF, d, cfg=cfg, mon=mon, ex=ex)
    if flags.get("SEAICE_CGRID"):                                               # :102-108 (lane M4OFF)
        stats(sf["UICE"], "_uice", g.maskInW, g.rAw)
        stats(sf["VICE"], "_vice", g.maskInS, g.rAs)
    if flags.get("SEAICE_BGRID_DYNAMICS"):                                      # :109-114
        stats(sf["UICE"], "_uice", sf["UVM"], g.rAz)
        stats(sf["VICE"], "_vice", sf["UVM"], g.rAz)
    if not useThSIce:                                                           # :115-126
        for name, suff in (("AREA", "_area"), ("HEFF", "_heff"), ("HSNOW", "_hsnow")):
            stats(sf[name], suff, g.maskInC, g.rA)
        if flags.get("SEAICE_VARIABLE_SALINITY"):                               # :122-125
            stats(sf["HSALT"], "_hsalt", g.maskInC, g.rA)
    if flags.get("ALLOW_SITRACER"):                                             # :127-134 (lane M4LAB session 4)
        from mitjax.pkg.seaice.seaice_h import level
        for iTracer in range(1, sp.SItrNumInUse + 1):                           # :128
            suff = f"_sitracer{iTracer:02d}"                                    # :129 '(A9,I2.2)'
            stats(level(sf["SItracer"], iTracer), suff, g.maskInC, g.rA)        # :130-132
    if mon.mon_write_stdout:                                                    # :140-149
        _banner("// End MONITOR SEAICE statistics", mon)
    mon.mon_write_stdout = False                                                # :151-152
    mon.mon_write_mnc = False
