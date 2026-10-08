"""pkg/generic_advdiff/gad_plm_fun.F: routines for the monotone piecewise linear method.

    o GAD_PLM_FUN_U   ported (called by GAD_PPM_HAT_* and GAD_PQM_HAT_* for the MONO and WENO limiters)
    o GAD_PLM_FUN_V   not ported: no routine of MITgcm @63cdc0b calls it (raises)

Point routines: the Fortran works on scalars, one grid cell per call; here every argument is an array of all the
cells of the caller's loop at once (each cell reads only its own arguments).
"""

import jax.numpy as jnp

from mitjax.ops.fortran_minmax import MAX, MIN

# gad_plm_fun.F:26-27   _RL epsil; PARAMETER( epsil = 1. _d -16 )
epsil = 1.e-16


def gad_plm_fun_u(ffll, ff00, ffrr):
    """GAD_PLM_FUN_U(ffll, ff00, ffrr, dfds)   @63cdc0b pkg/generic_advdiff/gad_plm_fun.F:10-64

    C     | PLM_FUN_U: monotone piecewise linear method.                   |
    C     |     - uniform grid-spacing variant.                            |
    I  ffll,ff00,ffrr
    O  dfds(-1:+1)

    Returns dfds as {-1: dfds(-1), 0: dfds(+0), +1: dfds(+1)} (every element is written). The IF of :32 becomes a
    `where`: the THEN statements (:35-49) run on every lane (finite there: the divisor is at least epsil), the ELSE
    value (:54) is selected where the product is not positive. `min`/`max` are mitjax.ops.fortran_minmax.MIN/MAX
    (gfortran's value on ties and NaN at each site; at an exact tie JAX's own derivative, PORTING_RULES §3).
    """
    dfds = {}
    dfds[-1] = ff00 - ffll                                              # :29-30
    dfds[+1] = ffrr - ff00

    then = dfds[-1] * dfds[+1] > 0.0                                    # :32

#     ======================================= calc. ll//rr edge values
    fell = 0.5 * (ffll + ff00)                                          # :35-36
    ferr = 0.5 * (ff00 + ffrr)

#     ======================================= calc. centred derivative
    dfds0 = 0.5 * (ferr - fell)                                         # :39-40

#     ======================================= monotonic slope-limiting
    scal = (MIN(jnp.abs(dfds[-1]),                                     # :43-45
                jnp.abs(dfds[+1]), p="a")
            / MAX(jnp.abs(dfds0), epsil, p="a"))
    scal = MIN(scal, 1.0, p="a")                                        # :47

    dfds0 = scal * dfds0                                                # :49

#     ======================================= flatten if local extrema
    dfds[+0] = jnp.where(then, dfds0, 0.0)                              # :54

    dfds[-1] = 0.5 * dfds[-1]                                           # :58-59
    dfds[+1] = 0.5 * dfds[+1]
    return dfds


def gad_plm_fun_v(ffll, ff00, ffrr, ddll, dd00, ddrr):
    """GAD_PLM_FUN_V   @63cdc0b pkg/generic_advdiff/gad_plm_fun.F:68-127 -- not ported (no caller in MITgcm)."""
    raise NotImplementedError("GAD_PLM_FUN_V is not ported (no routine of MITgcm @63cdc0b calls it)")
