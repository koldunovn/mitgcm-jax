"""ADPTRACERS_MONITOR: pkg/ptracers/ptracers_monitor_ad.F @63cdc0b (the adjoint build's ptracer monitor, called at
the end of ADMONITOR; host side, through pkg/monitor: its records go to `mon.units[mon.mon_ioUnit]`)."""

from mitjax.eesupp.different_multiple import different_multiple
from mitjax.io.fortran_format import fortran_write
from mitjax.pkg.monitor.mon_set_pref import mon_set_pref
from mitjax.pkg.monitor.mon_writestats_rl import mon_writestats_rl
from mitjax.pkg.ptracers.ptracers_monitor import _banner


def adptracers_monitor(myTime, myIter, *, cfg, params, grid, ptr, adptracer, mon, adjMonitorFreq, ex=None):
    """ADPTRACERS_MONITOR( myTime, myIter, myThid )   @63cdc0b pkg/ptracers/ptracers_monitor_ad.F:7-148

    C writes out ptracer statistics

    `adptracer`: [FArray per tracer] the adjoint ptracer fields at this point of the reverse sweep (ptracers_adcommon.h
    adptracer; printed as they are: no COPY_ADVAR_OUTP / ADEXCH here). `adjMonitorFreq` (PARAMS.h); the monitor
    namespaces as PTRACERS_MONITOR's (`cfg`: Nr, monitor_stdio, ALLOW_MNC, useMNC; `grid`: hFacC, maskInC, rA, drF);
    `ptr`: PTRACERS_PARAMS.h (PTRACERS_numInUse, PTRACERS_ioLabel, PTRACERS_monitor_mnc). MASTER_CPU_IO is true (one
    process). Raise: monitor output to MNC (useMNC with PTRACERS_monitor_mnc, :76-86)."""
    if not different_multiple(adjMonitorFreq, myTime, params.deltaTClock):     # :60-61
        return
    mon.mon_write_stdout = bool(cfg.monitor_stdio)                              # :69-74
    mon.mon_write_mnc = False                                                   # :75
    if cfg.ALLOW_MNC and cfg.useMNC and ptr.PTRACERS_monitor_mnc:               # :76-86
        raise NotImplementedError("ADPTRACERS_MONITOR: monitor output to MNC (PTRACERS_monitor_mnc) is not ported")
    if mon.mon_write_stdout:                                                    # :88-98
        _banner("// Begin AD_MONITOR ptracer field statistics", mon)
    mon_set_pref("ad_trcstat_", mon=mon)                                        # :105
    dummyRL = [0.0] * 6
    for ip in range(1, ptr.PTRACERS_numInUse + 1):                              # :106
        suff = fortran_write("(A9,A2)", "adptracer", ptr.PTRACERS_ioLabel[ip-1])  # :107
        dummyRL = mon_writestats_rl(cfg.Nr, adptracer[ip-1], suff,              # :108-110
                                    grid.hFacC, grid.maskInC, grid.rA, grid.drF, dummyRL, cfg=cfg, mon=mon, ex=ex)
    if mon.mon_write_stdout:                                                    # :117-127
        _banner("// End AD_MONITOR ptracer field statistics", mon)
    mon.mon_write_stdout = False                                                # :130-131
    mon.mon_write_mnc = False
