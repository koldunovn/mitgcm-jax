"""DO_WRITE_PICKUP: model/src/do_write_pickup.F @63cdc0b (host side, called by the driver at the end of a step on
the concrete State), with RESTART.h's alternating suffixes (INI_MODEL_IO sets them, ini_model_io.F:138-140)."""

from mitjax.pkg.monitor.monitor_h import different_multiple
from mitjax.model.src.write_pickup import write_pickup

maxNoChkptLev = 2                           # RESTART.h:24  PARAMETER ( maxNoChkptLev = 2 )


class Restart:
    """RESTART.h /RESTART_I/ nCheckLev and /RESTART_C/ checkPtSuff as INI_MODEL_IO sets them (ini_model_io.F:138-140:
    nCheckLev = 1, checkPtSuff(1) = 'ckptA', checkPtSuff(2) = 'ckptB')."""

    def __init__(self):
        self.nCheckLev = 1
        self.checkPtSuff = ["ckptA", "ckptB"]


def pickup_due(modelEnd, myTime, *, io, deltaTClock):
    """(write?, permPickup) of do_write_pickup.F:58-77 (without ALLOW_CAL)."""
    permPickup = different_multiple(io.pChkPtFreq, myTime, deltaTClock)         # :60-61
    tempPickup = different_multiple(io.chkPtFreq, myTime, deltaTClock)          # :62-63
    return (modelEnd and io.writePickupAtEnd) or permPickup or tempPickup, permPickup


def do_write_pickup(modelEnd, myTime, myIter, *, cfg, params, ip, io, state, mds, restart, deltaTClock, exp=None,
                    pk=None, cal=None, sp=None):
    """DO_WRITE_PICKUP( modelEnd, myTime, myIter, myThid )   @63cdc0b model/src/do_write_pickup.F:8-123

    C     | o Control writing of checkpoint/pickup files

    Returns the STDOUT records it prints (PRINT_MESSAGE without the '(PID.TID ...)' prefix) and the pickup file name
    or None. ALLOW_CAL (useCAL: CAL_TIME2DUMP) and rwSuffixType != 0 (RW_GET_SUFFIX) raise; PACKAGES_WRITE_PICKUP
    (:90-91, ADVECT lane: mitjax/model/src/packages_write_pickup.py, needs `exp`) runs when a package switch is on."""
    if dict(cfg.use).get("useCAL", False):                                      # :65-74
        # lane M4COL: CAL_TIME2DUMP (pkg/cal/cal_time2dump.F:31-55) changes time2write only under
        # `calendarDumps .AND. freq .NE. 0.` (:31); otherwise it returns it unchanged. That arm is not ported.
        if cal is None:
            raise ValueError("DO_WRITE_PICKUP: useCAL needs pkg/cal's common block (cal=)")
        if cal.calendarDumps and (io.pChkPtFreq != 0. or io.chkPtFreq != 0.):
            raise NotImplementedError("DO_WRITE_PICKUP: CAL_TIME2DUMP with calendarDumps is not ported")
    write, permPickup = pickup_due(modelEnd, myTime, io=io, deltaTClock=deltaTClock)
    lines, fn = [], None
    if write:                                                                   # :76-77
        if permPickup and ip.rwSuffixType == 0:                                 # :81-82  '(I10.10)'
            suffix = f"{int(myIter):010d}"
        elif permPickup:                                                        # :83-84
            raise NotImplementedError("DO_WRITE_PICKUP: RW_GET_SUFFIX (rwSuffixType != 0) is not ported")
        else:                                                                   # :85-86  '(A)'
            suffix = f"{restart.checkPtSuff[restart.nCheckLev - 1]:<10.10s}".rstrip()
        if any(v for _, v in cfg.use):                                          # :90-91 (ADVECT lane arm)
            from mitjax.model.src.packages_write_pickup import packages_write_pickup
            if exp is None:
                raise NotImplementedError("DO_WRITE_PICKUP: PACKAGES_WRITE_PICKUP needs the experiment (exp=)")
            packages_write_pickup(permPickup, suffix, myTime, myIter, exp=exp, cfg=cfg, params=params, io=io,
                                  state=state, mds=mds, printed=lines,          # printed: PTRACERS lane
                                  pk=pk, sp=sp)                                 # pk: vermix lane (GGL90.h)
        if not ip.useOffLine or ip.nonlinFreeSurf > 0:                          # :94-97
            fn = write_pickup(permPickup, suffix, myTime, myIter, cfg=cfg, params=params, ip=ip, io=io, state=state,
                              mds=mds)
        lines.append(("%CHECKPOINT " [:11] + f"{int(myIter):10d}" + " " + f"{suffix:<10.10s}").rstrip())   # :102-105
        if not permPickup:                                                      # :108-110
            restart.nCheckLev = restart.nCheckLev % maxNoChkptLev + 1
    elif modelEnd:                                                              # :113-117
        lines.append("Did not write pickup because writePickupAtEnd = FALSE")
    return lines, fn
