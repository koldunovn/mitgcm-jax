"""GAD_DST3FL_ADV_Y: meridional advective flux, DST-3 with flux limiting
(@63cdc0b pkg/generic_advdiff/gad_dst3fl_adv_y.F)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX, MIN
from mitjax.ops.safe import safe_div
from mitjax.pkg.generic_advdiff.gad_h import oneSixth

# gad_dst3fl_adv_y.F:35-36   _RL thetaMax; PARAMETER( thetaMax = 1.D+20 )
thetaMax = 1.0e+20


def gad_dst3fl_adv_y(k, calcCFL, deltaTloc, vTrans, vFld, maskLocS, tracer, vT, *, cfg, grid):
    """GAD_DST3FL_ADV_Y( bi,bj,k, calcCFL, deltaTloc, vTrans, vFld, maskLocS, tracer, vT, myThid )
    @63cdc0b pkg/generic_advdiff/gad_dst3fl_adv_y.F:3-98

    C     | SUBROUTINE GAD_DST3FL_ADV_Y                              |
    C     | o Compute Meridional advective Flux of Tracer using      |
    C     |   3rd Order DST Sceheme with flux limiting               |

    As GAD_DST3FL_ADV_X (see there: static calcCFL, `where` with the division guarded before the operation, SIGN as
    copysign, MAX/MIN with gfortran's tie values (mitjax.ops.fortran_minmax), the TARGET_NEC_SX initialisation :45-49 ported). GRID.h:
    recip_dyC, recip_deepFacC. The point loop body (:50-92) runs on the whole (i,j) range at once (each point reads
    only inputs). `0.5*` (:89, :91) is a REAL*4 literal, exact; the other literals are doubles.
    """
    sNx, sNy, OLx, OLy = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy
    recip_dyC, recip_deepFacC = grid.recip_dyC, grid.recip_deepFacC

    i = loop_i(1-OLx, sNx+OLx)                                      # :38-42
    vT = vT.at[i, 1-OLy].set(0.)
    vT = vT.at[i, 2-OLy].set(0.)
    vT = vT.at[i, sNy+OLy].set(0.)

    j = loop_j(1-OLy+2, sNy+OLy-1)                                  # :43-95
    if cfg.ALLOW_AUTODIFF and cfg.TARGET_NEC_SX:                    # :45-49
        thetaP = 0.
        thetaM = 0.
    Rjp = (tracer[i, j+1]-tracer[i, j])*maskLocS[i, j+1]
    Rj = (tracer[i, j]-tracer[i, j-1])*maskLocS[i, j]
    Rjm = (tracer[i, j-1]-tracer[i, j-2])*maskLocS[i, j-1]

    vCFL = vFld[i, j]                                               # :54-58
    if calcCFL:
        vCFL = jnp.abs(vFld[i, j]*deltaTloc
                       *recip_dyC[i, j]*recip_deepFacC[k])
    d0 = (2.-vCFL)*(1.-vCFL)*oneSixth
    d1 = (1.-vCFL*vCFL)*oneSixth

    bigP = jnp.abs(Rj)*thetaMax <= jnp.abs(Rjm)                     # :70-74
    thetaP = jnp.where(bigP, jnp.copysign(thetaMax, Rjm*Rj),
                       safe_div(Rjm, Rj, ~bigP))
    bigM = jnp.abs(Rj)*thetaMax <= jnp.abs(Rjp)                     # :75-79
    thetaM = jnp.where(bigM, jnp.copysign(thetaMax, Rjp*Rj),
                       safe_div(Rjp, Rj, ~bigM))

    psiP = d0+d1*thetaP                                             # :81-86
    psiP = MAX(0., MIN(MIN(1., psiP, p="b"),                       # :82-83
                       thetaP*(1.-vCFL)/(vCFL+1.e-20), p="b"), p="b")
    psiM = d0+d1*thetaM
    psiM = MAX(0., MIN(MIN(1., psiM, p="b"),                       # :85-86
                       thetaM*(1.-vCFL)/(vCFL+1.e-20), p="b"), p="b")

    vT = vT.at[i, j].set(                                           # :88-92
          0.5*(vTrans[i, j]+jnp.abs(vTrans[i, j]))
             *(tracer[i, j-1] + psiP*Rj)
         +0.5*(vTrans[i, j]-jnp.abs(vTrans[i, j]))
             *(tracer[i, j] - psiM*Rj))
    return vT
