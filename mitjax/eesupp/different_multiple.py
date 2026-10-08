"""DIFFERENT_MULTIPLE   @63cdc0b eesupp/src/different_multiple.F:7-65 (host side, on Python floats).

Moved here from the monitor lane's stand-in (mitjax/pkg/monitor/monitor_h.py, Task 11 follow-up). Used by MONITOR
(pkg/monitor/monitor.F:48), SBO_OUTPUT (pkg/sbo/sbo_output.F:103-105) and, when ported, THERMODYNAMICS's monOutputCFL
(model/src/thermodynamics.F:131-137) and the other output schedulers. The arguments are _RL (REAL*8): the arithmetic
below is IEEE float64, the same operations in the same order.
"""

import math


def different_multiple(freq, val1, step):
    """LOGICAL FUNCTION DIFFERENT_MULTIPLE( freq, val1, step )

    C     | o Checks if a multiple of freq exist
    C     |   around val1 +/- step/2
    C     freq       :: Frequency by which time is divided.
    C     val1       :: time that is checked
    C     step       :: length of time interval (around val1) that is checked

    NINT (different_multiple.F:54) rounds half away from zero to a default INTEGER (INTEGER*4); a quotient outside
    its range is undefined behaviour in gfortran, so it raises here instead of guessing."""
    freq, val1, step = float(freq), float(val1), float(step)
    out = False                                                     # :41
    if freq != 0.0:                                                 # :43
        if abs(step) > freq:                                        # :44-45
            out = True
        else:
            v1 = val1                                               # :49-51
            v2 = val1 - step
            v3 = val1 + step
            q = v1 / freq                                           # :54 NINT(v1/freq)*freq
            a = abs(q)
            n = math.floor(a)
            if a - n >= 0.5:
                n += 1
            n = n if q >= 0 else -n
            if not (-2 ** 31 <= n < 2 ** 31):
                raise OverflowError(f"DIFFERENT_MULTIPLE: NINT({q}) outside INTEGER*4")
            v4 = n * freq
            d1 = v1 - v4                                            # :55-57
            d2 = v2 - v4
            d3 = v3 - v4
            if abs(d1) < abs(d2) and abs(d1) <= abs(d3):            # :58-59
                out = True
    return out
