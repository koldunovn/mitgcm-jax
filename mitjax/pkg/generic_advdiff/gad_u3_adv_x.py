"""GAD_U3_ADV_X: zonal advective flux, upwind-biased third order (@63cdc0b pkg/generic_advdiff/gad_u3_adv_x.F)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.pkg.generic_advdiff.gad_h import oneSixth


def gad_u3_adv_x(k, uTrans, maskLocW, tracer, uT, *, cfg):
    """GAD_U3_ADV_X( bi,bj,k, uTrans, maskLocW, tracer, uT, myThid )
    @63cdc0b pkg/generic_advdiff/gad_u3_adv_x.F:7-75

    C !DESCRIPTION:
    C Calculates the area integrated zonal flux due to advection of a tracer
    C using upwind biased third-order interpolation (or the $\\kappa=1/3$ scheme):
    C F^x_{adv} = U \\overline{ \\theta  - \\frac{1}{6} \\delta_{ii} \\theta }^i
    C                 + \\frac{1}{12} |U| \\delta_{iii} \\theta
    C Near boundaries, mask all the gradients ==> still 3rd O.
    C !INPUT PARAMETERS:
    C  k            :: vertical level
    C  uTrans       :: zonal volume transport
    C  maskLocW     :: mask (either 0 or 1) at grid-cell western edge
    C  tracer       :: tracer field
    C !OUTPUT PARAMETERS:
    C  uT           :: zonal advective flux

    No CPP branches; GRID.h is not included (:26 is commented out). `k` is not used by the Fortran. The point loop
    body (:61-70) runs on the whole (i,j) range at once: each point reads only inputs. `0.` is a REAL*4 literal;
    `0.5 _d 0` is a double; oneSixth is GAD.h:97.
    """
    sNx, sNy, OLx, OLy = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy

    j = loop_j(1-OLy, sNy+OLy)                                      # :54-58
    uT = uT.at[1-OLx, j].set(0.)
    uT = uT.at[2-OLx, j].set(0.)
    uT = uT.at[sNx+OLx, j].set(0.)

    i = loop_i(1-OLx+2, sNx+OLx-1)                                  # :59-72
    Rjp = (tracer[i+1, j]-tracer[i, j])*maskLocW[i+1, j]
    Rj = (tracer[i, j]-tracer[i-1, j])*maskLocW[i, j]
    Rjm = (tracer[i-1, j]-tracer[i-2, j])*maskLocW[i-1, j]
    Rjjp = Rjp-Rj
    Rjjm = Rj-Rjm
    uT = uT.at[i, j].set(
          uTrans[i, j]*(
            tracer[i, j]+tracer[i-1, j]-oneSixth*(Rjjp+Rjjm)
                      )*0.5
         +jnp.abs(uTrans[i, j])*0.5*oneSixth*(Rjjp-Rjjm))
    return uT
