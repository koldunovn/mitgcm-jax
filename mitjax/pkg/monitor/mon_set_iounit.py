"""MON_SET_IOUNIT   @63cdc0b pkg/monitor/mon_set_iounit.F:8-38 (host side, on the MONITOR.h common block)."""


def mon_set_iounit(monUnit, *, mon):
    """MON_SET_IOUNIT( monUnit, myThid )

    C     !DESCRIPTION:
    C     Set default monitor unit for I/O.
    C     !INPUT PARAMETERS:
    C     monUnit :: Unit number to use for monitor output

    The BAR2 barriers around the master-thread update (:27, :35) are no-ops with one thread."""
    mon.mon_ioUnit = monUnit                                        # :31
