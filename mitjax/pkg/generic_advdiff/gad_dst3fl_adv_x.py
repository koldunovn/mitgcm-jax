"""GAD_DST3FL_ADV_X: zonal advective flux, DST-3 with flux limiting (@63cdc0b pkg/generic_advdiff/gad_dst3fl_adv_x.F)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX, MIN
from mitjax.ops.safe import safe_div
from mitjax.pkg.generic_advdiff.gad_h import oneSixth

# gad_dst3fl_adv_x.F:35-36   _RL thetaMax; PARAMETER( thetaMax = 1.D+20 )
thetaMax = 1.0e+20


def gad_dst3fl_adv_x(k, calcCFL, deltaTloc, uTrans, uFld, maskLocW, tracer, uT, *, cfg, grid):
    """GAD_DST3FL_ADV_X( bi,bj,k, calcCFL, deltaTloc, uTrans, uFld, maskLocW, tracer, uT, myThid )
    @63cdc0b pkg/generic_advdiff/gad_dst3fl_adv_x.F:3-102

    C     | SUBROUTINE GAD_DST3FL_ADV_X                              |
    C     | o Compute Zonal advective Flux of Tracer using           |
    C     |   3rd Order DST Sceheme with flux limiting               |

    (The routine has no argument comments; GAD_DST3_ADV_X's: calcCFL =T: calculate CFL number, =F: take uFld as
    CFL; deltaTloc local time-step (s); uTrans zonal volume transport; uFld zonal flow / CFL number; tracer tracer
    field; uT zonal advective flux.)

    `calcCFL` is a static Python bool (GAD_ADVECTION / GAD_CALC_RHS pass .TRUE.). GRID.h: recip_dxC,
    recip_deepFacC. The point loop body (:54-96) works on scalars per (i,j); here every statement runs on the whole
    (i,j) range at once (each point reads only inputs). The two-way IFs (:74-83) are `where`s; the division of the
    ELSE branch is guarded before the operation (`safe_div`, value bitwise Rjm/Rj where the Fortran computes it), so
    the branch not taken has a finite derivative [L-AD-1]. SIGN(a,b) is `copysign` (gfortran's default -fsign-zero:
    a negative zero b gives -|a|). MAX/MIN are `mitjax.ops.fortran_minmax.MAX/MIN`: gfortran's value on a tie
    (MAX(0.D0, -0.D0) = -0 reaches uT here; jnp.maximum gives +0) and JAX's own derivative (subgradient at a switch
    [L-AD-7], [L-CONF-12]). The `#if (ALLOW_AUTODIFF && TARGET_NEC_SX)`
    initialisation of thetaP, thetaM (:49-53) is ported; both are overwritten at :74-83. `0.5*` (:93, :95) is a
    REAL*4 literal, exact; `0. _d 0`, `1. _d 0`, `2. _d 0`, `1. _d -20` are doubles; oneSixth is GAD.h:97.
    The commented-out "old version" (:64-72) is not ported.
    """
    sNx, sNy, OLx, OLy = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy
    recip_dxC, recip_deepFacC = grid.recip_dxC, grid.recip_deepFacC

    j = loop_j(1-OLy, sNy+OLy)                                      # :42-46
    uT = uT.at[1-OLx, j].set(0.)
    uT = uT.at[2-OLx, j].set(0.)
    uT = uT.at[sNx+OLx, j].set(0.)

    i = loop_i(1-OLx+2, sNx+OLx-1)                                  # :47-99
    if cfg.ALLOW_AUTODIFF and cfg.TARGET_NEC_SX:                    # :49-53
        thetaP = 0.
        thetaM = 0.
    Rjp = (tracer[i+1, j]-tracer[i, j])*maskLocW[i+1, j]
    Rj = (tracer[i, j]-tracer[i-1, j])*maskLocW[i, j]
    Rjm = (tracer[i-1, j]-tracer[i-2, j])*maskLocW[i-1, j]

    uCFL = uFld[i, j]                                               # :58-62
    if calcCFL:
        uCFL = jnp.abs(uFld[i, j]*deltaTloc
                       *recip_dxC[i, j]*recip_deepFacC[k])
    d0 = (2.-uCFL)*(1.-uCFL)*oneSixth
    d1 = (1.-uCFL*uCFL)*oneSixth

    bigP = jnp.abs(Rj)*thetaMax <= jnp.abs(Rjm)                     # :74-78
    thetaP = jnp.where(bigP, jnp.copysign(thetaMax, Rjm*Rj),
                       safe_div(Rjm, Rj, ~bigP))
    bigM = jnp.abs(Rj)*thetaMax <= jnp.abs(Rjp)                     # :79-83
    thetaM = jnp.where(bigM, jnp.copysign(thetaMax, Rjp*Rj),
                       safe_div(Rjp, Rj, ~bigM))

    psiP = d0+d1*thetaP                                             # :85-90
    psiP = MAX(0., MIN(MIN(1., psiP, p="b"),                       # :86-87
                       thetaP*(1.-uCFL)/(uCFL+1.e-20), p="b"), p="b")
    psiM = d0+d1*thetaM
    psiM = MAX(0., MIN(MIN(1., psiM, p="b"),                       # :89-90
                       thetaM*(1.-uCFL)/(uCFL+1.e-20), p="b"), p="b")

    uT = uT.at[i, j].set(                                           # :92-96
          0.5*(uTrans[i, j]+jnp.abs(uTrans[i, j]))
             *(tracer[i-1, j] + psiP*Rj)
         +0.5*(uTrans[i, j]-jnp.abs(uTrans[i, j]))
             *(tracer[i, j] - psiM*Rj))
    return uT
