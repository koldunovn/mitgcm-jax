"""SBO_OUTPUT: pkg/sbo/sbo_output.F @63cdc0b."""

import numpy as np

from mitjax.io.fortran_format import fortran_write
from mitjax.pkg.monitor.monitor_h import MonitorCommon, SQUEEZE_RIGHT, different_multiple, print_message

narr = 24                   # sbo_output.F:44  PARAMETER( narr = 24 )
standardMessageUnit = 6     # eesupp/inc/EEPARAMS.h:115  PARAMETER ( standardMessageUnit = 6 )


def sbo_output(myTime, myIter, sbo, *, sp, nIter0, deltaTClock, stdout):
    """SBO_OUTPUT( myTime, myIter, myThid )   @63cdc0b pkg/sbo/sbo_output.F:7-130

    C     | SUBROUTINE SBO_OUTPUT
    C     | o Do SBO diagnostic output.

    `sbo` the SBO.h scalars (sbo_calc.SboCommon, concrete), `sp` SboParams (sbo_monFreq), `stdout` a
    MonitorCommon-like holder whose `units[6]` receives the PRINT_MESSAGE records (mitjax/pkg/monitor/monitor_h.py
    stand-in of eesupp PRINT_MESSAGE). Host-side, on concrete values. Returns sbo_diag(narr) (:56-79), the record
    MDS_WRITEVEC_LOC writes to SBO_global.<nIter0> at record myIter - nIter0 + 1 (:84-100); that file write itself
    is output only and not ported (no reader in the model or in testreport).

    :103-123: at myIter = nIter0 or DIFFERENT_MULTIPLE(sbo_monFreq, myTime, deltaTClock), four '(A,1PE21.13)'
    records %SBO sbo_mass, sbo_mass_fw, sbo_zoamc, sbo_zoamp through PRINT_MESSAGE(.., SQUEEZE_RIGHT, ..)."""
    v = {n: float(np.asarray(getattr(sbo, n))) for n in sbo.names()}
    sbo_diag = np.array([float(myTime), v["xoamc"], v["yoamc"], v["zoamc"], v["xoamp"], v["yoamp"], v["zoamp"],
                         v["mass"], v["xcom"], v["ycom"], v["zcom"], v["sboarea"], v["xoamc_si"], v["yoamc_si"],
                         v["zoamc_si"], v["mass_si"], v["xoamp_fw"], v["yoamp_fw"], v["zoamp_fw"], v["mass_fw"],
                         v["xcom_fw"], v["ycom_fw"], v["zcom_fw"], v["mass_gc"]], dtype=np.float64)   # :56-79
    assert sbo_diag.size == narr
    if myIter == nIter0 or different_multiple(sp.sbo_monFreq, myTime, deltaTClock):   # :103-105
        ioUnit = standardMessageUnit                                            # :107
        for label, name in (("%SBO sbo_mass                     = ", "mass"),     # :108-110
                            ("%SBO sbo_mass_fw                  = ", "mass_fw"),  # :111-113
                            ("%SBO sbo_zoamc                    = ", "zoamc"),    # :114-116
                            ("%SBO sbo_zoamp                    = ", "zoamp")):   # :117-119
            msgBuf = fortran_write("(A,1PE21.13)", label, v[name])
            print_message(msgBuf, ioUnit, SQUEEZE_RIGHT, mon=stdout)
    return sbo_diag


def new_stdout():
    """A holder of the PRINT_MESSAGE records (unit -> list of lines), as pkg/monitor uses it."""
    return MonitorCommon()
