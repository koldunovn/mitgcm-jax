"""MONITOR.h: parameters and common blocks of pkg/monitor, plus the host-side services the monitor routines call.

    @63cdc0b pkg/monitor/MONITOR.h:1-83

The common blocks /MON_I/, /MON_C/, /MON_R/, /MON_L/, /MON_F/ are the fields of `MonitorCommon` with their Fortran
names. The monitor runs on concrete values outside the differentiated program: `monitor.monitor` is called by the
driver between steps (and once after initialisation, initialise_varia.F:383), reads the state, and appends the text
records PRINT_MESSAGE writes to `MonitorCommon.units[unit]`.

Services from eesupp: PRINT_MESSAGE, IFNBLNK/ILNBLNK (mitjax/eesupp/print.py), DIFFERENT_MULTIPLE
(mitjax/eesupp/different_multiple.py); GLOBAL_SUM_TILE_RL / _GLOBAL_SUM_RL / _GLOBAL_MAX_RL through mitjax/eesupp
(global_sum.py; Exchanger.global_max) in `global_sum_tile_rl`, `global_sum_rl_scalar`, `global_max_rl` below. The
records PRINT_MESSAGE writes go to `MonitorCommon.io` (an eesupp MessageUnits); `MonitorCommon.units` is its unit
dictionary. `print_message(..., mon=)` is kept as the adapter pkg/sbo (sbo_output.py) calls.
"""

from dataclasses import dataclass, field

import numpy as np

from mitjax.eesupp import global_sum as gs
from mitjax.eesupp import print as eeprint
from mitjax.eesupp.different_multiple import different_multiple  # noqa: F401 (re-exported for pkg/sbo)
from mitjax.eesupp.print import MAX_LEN_FNAM, MAX_LEN_MBUF, SQUEEZE_RIGHT, ifnblnk, ilnblnk  # noqa: F401

# --- MONITOR.h:19-38  monitor head and tail strings (PARAMETERs)
mon_head = "%MON"
mon_foot_min = "_min"
mon_foot_max = "_max"
mon_foot_sd = "_sd"
mon_foot_mean = "_mean"
mon_foot_volint = "_volint"
mon_foot_volmean = "_volmean"
mon_foot_del2 = "_del2"
mon_foot_vol = "_vol"
mon_string_none = "NONE"


@dataclass
class MonitorCommon:
    """/MON_I/ /MON_C/ /MON_R/ /MON_L/ /MON_F/ (MONITOR.h:42-77) and the output units."""
    mon_ioUnit: int = None                  # /MON_I/
    mon_prefL: int = None
    mon_pref: str = " " * MAX_LEN_MBUF      # /MON_C/ CHARACTER*(MAX_LEN_MBUF)
    monSolutionMaxRange: float = None       # /MON_R/
    mon_trAdvCFL: list = None               # _RL mon_trAdvCFL(3)
    mon_output_AM: bool = None              # /MON_L/
    mon_write_stdout: bool = None
    mon_write_mnc: bool = None
    mon_fname: str = " " * MAX_LEN_FNAM     # /MON_F/
    io: eeprint.MessageUnits = field(default_factory=eeprint.MessageUnits)   # the Fortran units (eesupp)
    myThid: int = 1

    @property
    def units(self):
        """unit number -> list of records written by PRINT_MESSAGE."""
        return self.io.units


# ---------------------------------------------------------------------------------------------------------------
# helpers

def fstr_eq(a, b):
    """Fortran character comparison `a .EQ. b`: the shorter operand is padded with blanks."""
    n = max(len(a), len(b))  # MINMAX-RAW: string length, not a Fortran MAX
    return a.ljust(n) == b.ljust(n)


def print_message(message, unit, sq, *, mon):
    """PRINT_MESSAGE(message, unit, sq, myThid) through mitjax/eesupp/print.py, into mon.io."""
    eeprint.print_message(message, unit, sq, mon.myThid, io=mon.io)


def tile_sums(contrib):
    """Per-tile partial sums of a [tile, rows..., i] array in Fortran loop order (rows = the outer loops in order,
    i innermost), starting from `tile = 0.`: mitjax/eesupp/global_sum.py `tile_sum_fortran`."""
    import jax.numpy as jnp

    a = jnp.asarray(contrib)
    return gs.tile_sum_fortran(a.reshape(a.shape[0], -1, a.shape[-1]))


def global_sum_tile_rl(phiTile, ex=None):
    """CALL GLOBAL_SUM_TILE_RL(phiTile, sum, myThid) (eesupp/src/global_sum_tile.F, path without MPI :198-204):
    0. + tile 1 + tile 2 + ... through mitjax/eesupp; returns a host float64."""
    s = gs.global_sum_tile(phiTile) if ex is None else ex.global_sum_tile(phiTile)
    return np.float64(np.asarray(s))


def global_sum_rl_scalar(x, ex=None):
    """_GLOBAL_SUM_RL(x) = GLOBAL_SUM_R8 (eesupp/src/global_sum.F, one thread: `tmp = 0. _d 0; tmp = tmp + x`)."""
    import jax.numpy as jnp

    return global_sum_tile_rl(jnp.asarray([np.float64(x)]), ex)


def global_max_rl(x, ex=None):
    """_GLOBAL_MAX_RL(x) = GLOBAL_MAX_R8 (eesupp/src/global_max.F: one process, one thread -> x unchanged); through
    Exchanger.global_max when an exchanger is given."""
    if ex is None:
        return np.float64(x)
    import jax.numpy as jnp

    return np.float64(np.asarray(ex.global_max(jnp.asarray([np.float64(x)]))))
