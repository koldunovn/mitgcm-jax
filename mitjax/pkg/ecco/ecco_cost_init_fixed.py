"""ECCO_INIT_FIXED, ECCO_COST_INIT_FIXED   @63cdc0b pkg/ecco/ecco_init_fixed.F:8-52, ecco_cost_init_fixed.F:7-250
(lane M4ADCOL session 3). Host-side.

ECCO_INIT_FIXED (packages_init_fixed.F:381) calls ECCO_COST_INIT_FIXED (ecco_init_fixed.F:31); ECCO_DIAGNOSTICS_INIT
needs useDiagnostics (off). ECCO_COST_INIT_FIXED: eccoiter = optimcycle (:85, ALLOW_CTRL; 0 without a data.optim
override), eccoVol_0 (:89-103; read only by ECCO_PHYS' areavolTile, see ecco_phys.py), the record counts (:108-112,
ALLOW_CAL), and per gencost the barskip, time-averaging and start/end dates (:114-202). ALLOW_GENCOST_1D is undefined.
ECCO_COST_INIT_BARFILES (:237-240, when no file `costfinal` exists) writes the adjoint bar files only under
ALLOW_ADJOINT_RUN (ecco_cost_init_barfiles.F:117-129), which AD_CONFIG.h of the forward build leaves undefined: its
forward effect is the zeroing of two locals.
"""

import numpy as np

from mitjax.pkg.cal.cal_copydate import cal_CopyDate
from mitjax.pkg.cal.cal_fulldate import cal_FullDate
from mitjax.pkg.cal.cal_intmonths import cal_IntMonths, cal_IntYears
from mitjax.pkg.ecco.ecco_readparms import _eq


def ecco_cost_init_fixed(ep, *, cal, optimcycle, nTimeSteps, dTtracerLev1):
    """ECCO_COST_INIT_FIXED on EccoParams `ep` (updated in place: the gencost fields of ECCO.h) -> dict(eccoiter,
    nyearsrec, nmonsrec). `optimcycle`: OPTIMCYCLE.h (data.optim)."""
    eccoiter = optimcycle                                                    # :85
    nyearsrec = cal_IntYears(cal=cal)                                        # :110
    nmonsrec = cal_IntMonths(cal=cal)                                        # :111
    # :112 ndaysrec = cal_IntDays( myThid ): only a 'day' gencost reads it (:121-123), which raises below
    for k, g in enumerate(ep.gencost, start=1):                              # :114-206
        g.gencost_barskip = False                                            # :119
        if _eq(g.gencost_barfile, " "):                                      # :120-121
            g.gencost_barskip = True
        for g2 in ep.gencost[:k - 1]:                                        # :122-125
            if _eq(g2.gencost_barfile, g.gencost_barfile):
                g.gencost_barskip = True
        if g.using_gencost and (g.gencost_flag >= 1 or not _eq(g.gencost_avgperiod, "     ")):   # :128-154
            ap = g.gencost_avgperiod.rstrip(" ")
            if ap in ("day", "DAY"):
                raise NotImplementedError("ECCO_COST_INIT_FIXED: a 'day' gencost (cal_IntDays) is not ported")
            elif ap in ("month", "MONTH"):                                   # :134-137
                g.gencost_nrec = nmonsrec
                g.gencost_period = np.float64(0.)
            elif ap in ("step", "STEP"):                                     # :138-141
                g.gencost_nrec = nTimeSteps + 1
                g.gencost_period = np.float64(dTtracerLev1)
            elif ap in ("const", "CONST"):                                   # :142-145
                g.gencost_nrec = 1
                g.gencost_period = np.float64(dTtracerLev1)
            elif ap in ("year", "YEAR"):                                     # :146-149
                raise RuntimeError("ecco_cost_init_fixed: yearly data not yet implemented")
            else:                                                            # :150-152
                raise RuntimeError("ecco_cost_init_fixed: gencost_avgperiod wrongly specified")
        if g.gencost_startdate1 > 0:                                         # :157-166
            g.gencost_startdate = list(cal_FullDate(g.gencost_startdate1, g.gencost_startdate2, cal=cal))
        else:
            g.gencost_startdate = cal_CopyDate(cal.modelStartDate)
            g.gencost_startdate1 = cal.startdate_1
            g.gencost_startdate2 = cal.startdate_2
        if g.gencost_enddate1 > 0:                                           # :168-175
            g.gencost_enddate = list(cal_FullDate(g.gencost_enddate1, g.gencost_enddate2, cal=cal))
        else:
            g.gencost_enddate = cal_CopyDate(cal.modelEndDate)
    return dict(eccoiter=eccoiter, nyearsrec=nyearsrec, nmonsrec=nmonsrec)
