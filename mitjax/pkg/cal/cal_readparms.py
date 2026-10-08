"""CAL_READPARMS: pkg/cal/cal_readparms.F @63cdc0b."""

from mitjax.params_io import RunParams
from mitjax.pkg.cal.cal_h import Cal


def cal_readparms(exp):
    """CAL_READPARMS( myThid )   @63cdc0b pkg/cal/cal_readparms.F:3-118

    C     o This routine initialises the calendar according to the user
    C       specifications in "data.cal".

    Returns a new `Cal` (cal.h) after the namelist read of CAL_NML (TheCalendar, startDate_1, startDate_2,
    calendarDumps) with the initialisations of :82-86, or a `Cal` with cal_setStatus = -1 when useCAL is .FALSE.
    (:68-77)."""
    cal = Cal()
    if not exp.cfg.use_flag("useCAL"):                                         # :68-77
        cal.cal_setStatus = -1
        return cal
    cal.cal_setStatus = 0                                                      # :82
    rp = RunParams(exp.run)
    f, g = "data.cal", "CAL_NML"
    # :83-86 initial values, then READ(unit = iUnit, nml = cal_nml) (:97): a variable the file sets wins
    cal.theCalendar = rp.get(f, g, "TheCalendar") if rp.has(f, g, "TheCalendar") else " "
    cal.startdate_1 = int(rp.get(f, g, "startDate_1")) if rp.has(f, g, "startDate_1") else 0
    cal.startdate_2 = int(rp.get(f, g, "startDate_2")) if rp.has(f, g, "startDate_2") else 0
    cal.calendarDumps = bool(rp.get(f, g, "calendarDumps")) if rp.has(f, g, "calendarDumps") else False
    return cal
