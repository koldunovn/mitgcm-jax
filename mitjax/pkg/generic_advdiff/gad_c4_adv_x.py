"""GAD_C4_ADV_X: zonal advective flux, centered fourth-order (@63cdc0b pkg/generic_advdiff/gad_c4_adv_x.F)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.pkg.generic_advdiff.gad_h import oneSixth


def gad_c4_adv_x(k, uTrans, maskLocW, tracer, uT, *, cfg, grid):
    """GAD_C4_ADV_X( bi,bj,k, uTrans, maskLocW, tracer, uT, myThid )
    @63cdc0b pkg/generic_advdiff/gad_c4_adv_x.F:7-78

    C !DESCRIPTION:
    C Calculates the area integrated zonal flux due to advection of a tracer
    C using centered fourth-order interpolation:
    C F^x_{adv} = U \\overline{ \\theta - \\frac{1}{6} \\delta_{ii} \\theta }^i
    C Near boundaries, the scheme reduces to a second if the flow is away
    C from the boundary and to third order if the flow is towards
    C the boundary.
    C !INPUT PARAMETERS:
    C  k            :: vertical level
    C  uTrans       :: zonal volume transport
    C  maskLocW     :: mask (either 0 or 1) at grid-cell western edge
    C  tracer       :: tracer field
    C !OUTPUT PARAMETERS:
    C  uT           :: zonal advective flux

    No CPP branches. GRID.h: maskW (the live line :72; the maskLocW alternative :73 is commented out). The point loop
    body (:62-72) works on scalars per (i,j); here every statement runs on the whole (i,j) range at once, which is
    the same computation because each point reads only inputs. `0.` is a REAL*4 literal; `0.5 _d 0`, `1. _d 0` are
    doubles; oneSixth is GAD.h:97.
    """
    sNx, sNy, OLx, OLy = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy
    maskW = grid.maskW

    j = loop_j(1-OLy, sNy+OLy)                                      # :55-59
    uT = uT.at[1-OLx, j].set(0.)
    uT = uT.at[2-OLx, j].set(0.)
    uT = uT.at[sNx+OLx, j].set(0.)

    i = loop_i(1-OLx+2, sNx+OLx-1)                                  # :60-75
    Rjp = (tracer[i+1, j]-tracer[i, j])*maskLocW[i+1, j]
    Rj = (tracer[i, j]-tracer[i-1, j])*maskLocW[i, j]
    Rjm = (tracer[i-1, j]-tracer[i-2, j])*maskLocW[i-1, j]
    Rjjp = (Rjp-Rj)
    Rjjm = (Rj-Rjm)
    uT = uT.at[i, j].set(
          uTrans[i, j]*(
            tracer[i, j]+tracer[i-1, j]-oneSixth*(Rjjp+Rjjm)
                      )*0.5
         +jnp.abs(uTrans[i, j])*0.5*oneSixth*(Rjjp-Rjjm)
           *(1. - maskW[i-1, j, k]*maskW[i+1, j, k]))
    return uT
