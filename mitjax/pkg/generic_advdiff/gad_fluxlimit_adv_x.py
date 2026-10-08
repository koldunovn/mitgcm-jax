"""GAD_FLUXLIMIT_ADV_X: zonal advective flux, second order with a flux limiter
(@63cdc0b pkg/generic_advdiff/gad_fluxlimit_adv_x.F)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.ops.safe import safe_div
from mitjax.pkg.generic_advdiff.gad_flux_limiter_h import Limiter

# eesupp/inc/EEPARAMS.h:72   PARAMETER ( zeroRL = 0.0 _d 0 , oneRL  = 1.0 _d 0 )
zeroRL = 0.0
oneRL = 1.0
# gad_fluxlimit_adv_x.F:62-63   _RL CrMax; PARAMETER( CrMax = 1.D+6 )
CrMax = 1.0e+6


def gad_fluxlimit_adv_x(k, calcCFL, deltaTloc, uTrans, uFld, maskLocW, tracer, uT, *, cfg, grid):
    """GAD_FLUXLIMIT_ADV_X( bi, bj, k, calcCFL, deltaTloc, uTrans, uFld, maskLocW, tracer, uT, myThid )
    @63cdc0b pkg/generic_advdiff/gad_fluxlimit_adv_x.F:7-106

    C !DESCRIPTION:
    C Calculates the area integrated zonal flux due to advection of a tracer
    C using second-order interpolation with a flux limiter:
    C F^x_{adv} = U \\overline{ \\theta }^i
    C - \\frac{1}{2} \\left(
    C     [ 1 - \\psi(C_r) ] |U|
    C    + U \\frac{u \\Delta t}{\\Delta x_c} \\psi(C_r)
    C              \\right) \\delta_i \\theta
    C where the $\\psi(C_r)$ is the limiter function and $C_r$ is
    C the slope ratio.
    C !INPUT PARAMETERS:
    C  k                 :: vertical level
    C  calcCFL           :: =T: calculate CFL number ; =F: take uFld as CFL
    C  deltaTloc         :: local time-step (s)
    C  uTrans            :: zonal volume transport
    C  uFld              :: zonal flow / CFL number
    C  tracer            :: tracer field
    C !OUTPUT PARAMETERS:
    C  uT                :: zonal advective flux

    `calcCFL` is a static Python bool. GRID.h: recip_dxC, recip_deepFacC. Limiter(Cr) is the statement function of
    GAD_FLUX_LIMITER.h (gad_flux_limiter_h.py). The point loop body (:77-101) runs on the whole (i,j) range at once
    (each point reads only inputs). The IFs (:84-88, :89-93) are `where`s; the division of the ELSE branch is guarded
    before the operation (`safe_div`: value bitwise Cr/Rj where the Fortran computes it, finite derivative on the
    branch not taken [L-AD-1]); SIGN(a,b) is `copysign` (gfortran's default -fsign-zero). Limiters are
    differentiated as written (subgradient, JAX's own derivative at a switch [L-AD-7], [L-CONF-12]). Literals
    `0. _d 0`, `0.5 _d 0` are doubles.
    """
    sNx, sNy, OLx, OLy = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy
    recip_dxC, recip_deepFacC = grid.recip_dxC, grid.recip_deepFacC

    j = loop_j(1-OLy, sNy+OLy)                                      # :69-73
    uT = uT.at[1-OLx, j].set(0.)
    uT = uT.at[2-OLx, j].set(0.)
    uT = uT.at[sNx+OLx, j].set(0.)

    i = loop_i(1-OLx+2, sNx+OLx-1)                                  # :74-103
    uCFL = uFld[i, j]                                               # :77-79
    if calcCFL:
        uCFL = jnp.abs(uFld[i, j]*deltaTloc
                       *recip_dxC[i, j]*recip_deepFacC[k])
    Rjp = (tracer[i+1, j]-tracer[i, j])*maskLocW[i+1, j]            # :80-82
    Rj = (tracer[i, j]-tracer[i-1, j])*maskLocW[i, j]
    Rjm = (tracer[i-1, j]-tracer[i-2, j])*maskLocW[i-1, j]

    Cr = jnp.where(uTrans[i, j] > zeroRL, Rjm, Rjp)                 # :84-88
    big = jnp.abs(Rj)*CrMax <= jnp.abs(Cr)                          # :89-93
    Cr = jnp.where(big, jnp.copysign(CrMax, Cr)*jnp.copysign(oneRL, Rj),
                   safe_div(Cr, Rj, ~big))

#       calculate Limiter Function:
    Cr = Limiter(Cr, p=("b", "a", "b", "b"))                        # :96

    uT = uT.at[i, j].set(                                           # :98-101
            uTrans[i, j]*(tracer[i, j]+tracer[i-1, j])*0.5
          - jnp.abs(uTrans[i, j])*((oneRL-Cr) + uCFL*Cr)
                            *Rj*0.5)
    return uT
