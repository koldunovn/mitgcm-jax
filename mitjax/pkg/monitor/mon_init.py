"""MON_INIT   @63cdc0b pkg/monitor/mon_init.F:8-55 (host side, on the MONITOR.h common block)."""

from mitjax.eesupp.print import STANDARD_MESSAGE_UNIT as standardMessageUnit   # eesupp/src/eeboot.F:91
from mitjax.pkg.monitor.mon_set_iounit import mon_set_iounit
from mitjax.pkg.monitor.mon_set_pref import mon_set_pref
from mitjax.pkg.monitor.monitor_h import mon_string_none


def mon_init(*, cfg, params, mon):
    """MON_INIT( myThid )

    C     !DESCRIPTION:
    C     Set default monitor internal setup.

    cfg: fluidIsWater, fluidIsAir, useCoriolis (logical PARAMS), selectCoriMap (integer PARAMS).
    params: unused (kept for the uniform signature).
    `1. _d +4` and `1. _d +3` are the doubles 1.0D4 and 1.0D3; `0.` (REAL*4 zero) is exact."""
    mon_set_iounit(standardMessageUnit, mon=mon)                    # :33
    mon_set_pref(mon_string_none, mon=mon)                          # :34
    mon.monSolutionMaxRange = 1.0e4                                 # :41
    if cfg.fluidIsWater:                                            # :42
        mon.monSolutionMaxRange = 1.0e3
    mon.mon_output_AM = (cfg.fluidIsAir and cfg.useCoriolis         # :44-45
                         and cfg.selectCoriMap >= 2)
    mon.mon_trAdvCFL = [0.0, 0.0, 0.0]                              # :46-48
