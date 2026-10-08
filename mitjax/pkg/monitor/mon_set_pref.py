"""MON_SET_PREF   @63cdc0b pkg/monitor/mon_set_pref.F:8-53 (host side, on the MONITOR.h common block)."""

from mitjax.eesupp.print import MAX_LEN_MBUF, ifnblnk, ilnblnk


def mon_set_pref(prefString, *, mon):
    """MON_SET_PREF( prefString, myThid )

    C     !DESCRIPTION:
    C     Set default monitor prefix string.
    C     !INPUT PARAMETERS:
    C     prefString - String to use for prefixing monitor output

    The BAR2 barriers (:35, :50) are no-ops with one thread."""
    I0 = ifnblnk(prefString)                                        # :39-41
    I1 = ilnblnk(prefString)
    IL = I1 - I0 + 1
    if IL <= MAX_LEN_MBUF:                                          # :42-46
        mon.mon_pref = " " * MAX_LEN_MBUF
        mon.mon_prefL = IL
        mon.mon_pref = prefString[I0 - 1:I1] + mon.mon_pref[IL:]
