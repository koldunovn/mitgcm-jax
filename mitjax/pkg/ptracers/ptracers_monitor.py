"""PTRACERS_MONITOR: pkg/ptracers/ptracers_monitor.F @63cdc0b (host side, through pkg/monitor: an output routine;
its records go to `mon.units[mon.mon_ioUnit]` as MONITOR's)."""

from mitjax.eesupp.different_multiple import different_multiple
from mitjax.eesupp.print import SQUEEZE_RIGHT, print_message
from mitjax.io.fortran_format import fortran_write
from mitjax.pkg.monitor.mon_out import mon_out_i, mon_out_rl
from mitjax.pkg.monitor.mon_set_pref import mon_set_pref
from mitjax.pkg.monitor.mon_writestats_rl import mon_writestats_rl
from mitjax.pkg.monitor.monitor_h import mon_string_none


def _banner(text, mon):
    """The three PRINT_MESSAGE records around the block (ptracers_monitor.F:78-86, 118-126)."""
    line = fortran_write("(2A)", "// ==========================", "=============================")
    print_message(line, mon.mon_ioUnit, SQUEEZE_RIGHT, mon.myThid, io=mon.io)
    print_message(fortran_write("(A)", text), mon.mon_ioUnit, SQUEEZE_RIGHT, mon.myThid, io=mon.io)
    print_message(line, mon.mon_ioUnit, SQUEEZE_RIGHT, mon.myThid, io=mon.io)


def ptracers_monitor(myTime, myIter, *, cfg, params, grid, ptr, ptf, mon, ex=None):
    """PTRACERS_MONITOR( myTime, myIter, myThid )   @63cdc0b pkg/ptracers/ptracers_monitor.F:7-148

    C     writes out ptracer statistics

    The monitor namespaces of pkg/monitor (mitjax/pkg/monitor/monitor.py): `cfg` (Nr, monitor_stdio, ALLOW_MNC,
    useMNC), `params` (monitorFreq, deltaTClock), `grid` (hFacC, maskInC, rA, drF), `mon` (MONITOR.h); `ptr`:
    PTRACERS_PARAMS.h (PTRACERS_monitorFreq, PTRACERS_monitor_mnc, PTRACERS_ioLabel, PTRACERS_numInUse); `ptf`:
    PTRACERS_FIELDS.h (concrete arrays). MASTER_CPU_IO is true (one process, one thread). Raise: monitor output to
    MNC (useMNC with PTRACERS_monitor_mnc, :65-75)."""
    if not different_multiple(ptr.PTRACERS_monitorFreq, myTime, params.deltaTClock):   # :52-53
        return
    mon.mon_write_stdout = bool(cfg.monitor_stdio)                              # :59-63
    mon.mon_write_mnc = False                                                   # :64
    if cfg.ALLOW_MNC and cfg.useMNC and ptr.PTRACERS_monitor_mnc:               # :65-75
        raise NotImplementedError("PTRACERS_MONITOR: monitor output to MNC (PTRACERS_monitor_mnc) is not ported")
    if mon.mon_write_stdout:                                                    # :77-87
        _banner("// Begin MONITOR ptracer field statistics", mon)
    if (ptr.PTRACERS_monitorFreq != params.monitorFreq                          # :93-101
            or (cfg.useMNC and ptr.PTRACERS_monitor_mnc)):
        mon_set_pref("trctime", mon=mon)
        mon_out_i("_tsnumber", myIter, mon_string_none, cfg=cfg, mon=mon)
        mon_out_rl("_secondsf", myTime, mon_string_none, cfg=cfg, mon=mon)
    mon_set_pref("trcstat_", mon=mon)                                           # :103
    dummyRL = [0.0] * 6
    for ip in range(1, ptr.PTRACERS_numInUse+1):                                # :104
        suff = fortran_write("(A7,A2)", "ptracer", ptr.PTRACERS_ioLabel[ip-1])   # :105
        dummyRL = mon_writestats_rl(cfg.Nr, ptf.pTracer[ip-1], suff,           # :107-109
                                    grid.hFacC, grid.maskInC, grid.rA, grid.drF, dummyRL, cfg=cfg, mon=mon, ex=ex)
    if mon.mon_write_stdout:                                                    # :117-127
        _banner("// End MONITOR ptracer field statistics", mon)
    mon.mon_write_stdout = False                                                # :129-130
    mon.mon_write_mnc = False
