"""GAD_C4_ADV_Y: meridional advective flux, centered fourth-order (@63cdc0b pkg/generic_advdiff/gad_c4_adv_y.F)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.pkg.generic_advdiff.gad_h import oneSixth


def gad_c4_adv_y(k, vTrans, maskLocS, tracer, vT, *, cfg, grid):
    """GAD_C4_ADV_Y( bi,bj,k, vTrans, maskLocS, tracer, vT, myThid )
    @63cdc0b pkg/generic_advdiff/gad_c4_adv_y.F:7-78

    C !DESCRIPTION:
    C Calculates the area integrated meridional flux due to advection of a tracer
    C using centered fourth-order interpolation:
    C F^y_{adv} = V \\overline{ \\theta - \\frac{1}{6} \\delta_{jj} \\theta }^j
    C Near boundaries, the scheme reduces to a second if the flow is away
    C from the boundary and to third order if the flow is towards
    C the boundary.
    C !INPUT PARAMETERS:
    C  k            :: vertical level
    C  vTrans       :: meridional volume transport
    C  maskLocS     :: mask (either 0 or 1) at grid-cell southern edge
    C  tracer       :: tracer field
    C !OUTPUT PARAMETERS:
    C  vT           :: meridional advective flux

    No CPP branches. GRID.h: maskS (the live line :72; the maskLocS alternative :73 is commented out). The point loop
    body (:62-72) runs on the whole (i,j) range at once: each point reads only inputs. `0.` is a REAL*4 literal;
    `0.5 _d 0`, `1. _d 0` are doubles; oneSixth is GAD.h:97.
    """
    sNx, sNy, OLx, OLy = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy
    maskS = grid.maskS

    i = loop_i(1-OLx, sNx+OLx)                                      # :55-59
    vT = vT.at[i, 1-OLy].set(0.)
    vT = vT.at[i, 2-OLy].set(0.)
    vT = vT.at[i, sNy+OLy].set(0.)

    j = loop_j(1-OLy+2, sNy+OLy-1)                                  # :60-75
    Rjp = (tracer[i, j+1]-tracer[i, j])*maskLocS[i, j+1]
    Rj = (tracer[i, j]-tracer[i, j-1])*maskLocS[i, j]
    Rjm = (tracer[i, j-1]-tracer[i, j-2])*maskLocS[i, j-1]
    Rjjp = (Rjp-Rj)
    Rjjm = (Rj-Rjm)
    vT = vT.at[i, j].set(
          vTrans[i, j]*(
            tracer[i, j]+tracer[i, j-1]-oneSixth*(Rjjp+Rjjm)
                      )*0.5
         +jnp.abs(vTrans[i, j])*0.5*oneSixth*(Rjjp-Rjjm)
           *(1. - maskS[i, j-1, k]*maskS[i, j+1, k]))
    return vT
