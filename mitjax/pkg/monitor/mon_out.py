"""MON_OUT_I, MON_OUT_RS, MON_OUT_RL, MON_OUT_ALL   @63cdc0b pkg/monitor/mon_out.F:8-227

Formatted monitor records: `%MON <prefix><name><suffix>` with '=' in column 35 and the value in columns 36-57, written
with the Fortran edit descriptors of mon_out.F:195,197 by mitjax/io/fortran_format.py. Host side: values are concrete
Python/numpy numbers.
"""

import numpy as np

from mitjax.eesupp.print import MAX_LEN_MBUF, SQUEEZE_RIGHT, ifnblnk, ilnblnk, print_message
from mitjax.io.fortran_format import fortran_write
from mitjax.pkg.monitor.monitor_h import fstr_eq, mon_head, mon_string_none


def mon_out_i(pref, value, foot, *, cfg, mon):
    """MON_OUT_I( pref, value, foot, myThid )   mon_out.F:8-25

    C     Formatted integer I/O for monitor print out.
    C     pref   - Field prefix ( ignored if == mon_string_none )
    C     value  - Value to print
    C     foot   - Field suffix ( ignored if == mon_string_none )"""
    mon_out_all(pref, foot, 1, int(value), 0.0, cfg=cfg, mon=mon)          # :23 (0.0d0)


def mon_out_rs(pref, value, foot, *, cfg, mon):
    """MON_OUT_RS( pref, value, foot, myThid )   mon_out.F:32-51 (`dtmp = value`: _RS is REAL*8 here)."""
    dtmp = np.float64(value)                                               # :47
    mon_out_all(pref, foot, 2, 0, dtmp, cfg=cfg, mon=mon)                  # :49


def mon_out_rl(pref, value, foot, *, cfg, mon):
    """MON_OUT_RL( pref, value, foot, myThid )   mon_out.F:58-77

    C     Formatted RL I/O for monitor print out."""
    dtmp = np.float64(value)                                               # :73
    mon_out_all(pref, foot, 2, 0, dtmp, cfg=cfg, mon=mon)                  # :75


def mon_out_all(pref, foot, itype, ival, dval, *, cfg, mon):
    """MON_OUT_ALL( pref, foot, itype, ival, dval, myThid )   mon_out.F:84-227

    C     Formatted I/O for monitor output.
    C     pref   - Field prefix ( ignored if == mon_string_none )
    C     foot   - Field suffix ( ignored if == mon_string_none )

    msgBuf is a CHARACTER*(MAX_LEN_MBUF) buffer, built character by character as the Fortran does (lBuf logic,
    :139-185), '=' in column 35 (:191), the value written into msgBuf(36:57) (:194-197). MASTER_CPU_IO is true for
    the one thread of the oracle. #ifdef ALLOW_MNC (:210-221): writing the monitor to MNC (useMNC with
    monitor_mnc) is not ported and raises in monitor.monitor; here mon_write_mnc must be false."""
    buf = [" "] * MAX_LEN_MBUF

    def put(lo, hi, text):                     # msgBuf(lo:hi) = text (1-based, inclusive; blank-padded)
        n = hi - lo + 1
        t = text[:n].ljust(n)
        buf[lo - 1:hi] = list(t)

    lBuf = 0                                                               # :133
    I0 = ifnblnk(mon_head)                                                 # :139-141
    I1 = ilnblnk(mon_head)
    IL = I1 - I0 + 1
    if IL > 0 and lBuf + IL + 1 <= MAX_LEN_MBUF:                           # :142-146
        put(1, IL, mon_head)
        lBuf = IL + 1
        put(lBuf, lBuf, " ")

    prefL = mon.mon_prefL
    if (not fstr_eq(mon.mon_pref[:prefL], mon_string_none)                 # :148-157
            and lBuf + prefL + 1 <= MAX_LEN_MBUF):
        lBuf = lBuf + 1
        put(lBuf, lBuf + prefL - 1, mon.mon_pref[:prefL])
        lBuf = lBuf + prefL - 1

    for part in (pref, foot):                                              # :159-171 (pref), :173-185 (foot)
        I0 = ifnblnk(part)
        I1 = ilnblnk(part)
        IL = I1 - I0 + 1
        if IL > 0:
            if (not fstr_eq(part[I0 - 1:I1], mon_string_none)
                    and lBuf + IL + 1 <= MAX_LEN_MBUF):
                lBuf = lBuf + 1
                put(lBuf, lBuf + IL - 1, part[I0 - 1:I1])
                lBuf = lBuf + IL - 1

    put(35, 35, "=")                                                       # :191

    if mon.mon_write_stdout:                                               # :193-208
        if itype == 1:
            put(36, 57, _internal_write(fortran_write("(1X,I21)", int(ival)), 22))
        if itype == 2:
            put(36, 57, _internal_write(fortran_write("(1X,1P1E21.13)", float(dval)), 22))
        print_message("".join(buf), mon.mon_ioUnit, SQUEEZE_RIGHT, mon.myThid, io=mon.io)

    if cfg.ALLOW_MNC and cfg.useMNC and mon.mon_write_mnc:                 # :210-221
        raise NotImplementedError("MON_OUT_ALL: monitor output to MNC (useMNC with monitor_mnc) is not ported")


def _internal_write(record, n):
    """An internal WRITE into a CHARACTER*(n) variable: the record blank-padded to n; a longer record is the Fortran
    run-time error 'End of record'."""
    if len(record) > n:
        raise ValueError(f"internal WRITE: record {record!r} longer than the {n}-character variable")
    return record.ljust(n)
