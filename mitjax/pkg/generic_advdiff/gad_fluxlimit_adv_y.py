"""GAD_FLUXLIMIT_ADV_Y: meridional advective flux, second order with a flux limiter
(@63cdc0b pkg/generic_advdiff/gad_fluxlimit_adv_y.F)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.ops.safe import safe_div
from mitjax.pkg.generic_advdiff.gad_flux_limiter_h import Limiter

# eesupp/inc/EEPARAMS.h:72   PARAMETER ( zeroRL = 0.0 _d 0 , oneRL  = 1.0 _d 0 )
zeroRL = 0.0
oneRL = 1.0
# gad_fluxlimit_adv_y.F:62-63   _RL CrMax; PARAMETER( CrMax = 1.D+6 )
CrMax = 1.0e+6


def gad_fluxlimit_adv_y(k, calcCFL, deltaTloc, vTrans, vFld, maskLocS, tracer, vT, *, cfg, grid):
    """GAD_FLUXLIMIT_ADV_Y( bi, bj, k, calcCFL, deltaTloc, vTrans, vFld, maskLocS, tracer, vT, myThid )
    @63cdc0b pkg/generic_advdiff/gad_fluxlimit_adv_y.F:7-106

    C !DESCRIPTION:
    C Calculates the area integrated meridional flux due to advection of a tracer
    C using second-order interpolation with a flux limiter:
    C F^y_{adv} = V \\overline{ \\theta }^j
    C - \\frac{1}{2} \\left(
    C     [ 1 - \\psi(C_r) ] |V|
    C    + V \\frac{v \\Delta t}{\\Delta y_c} \\psi(C_r)
    C              \\right) \\delta_j \\theta
    C where the $\\psi(C_r)$ is the limiter function and $C_r$ is
    C the slope ratio.
    C !INPUT PARAMETERS:
    C  k                 :: vertical level
    C  calcCFL           :: =T: calculate CFL number ; =F: take vFld as CFL
    C  deltaTloc         :: local time-step (s)
    C  vTrans            :: meridional volume transport
    C  vFld              :: meridional flow / CFL number
    C  tracer            :: tracer field
    C !OUTPUT PARAMETERS:
    C  vT                :: meridional advective flux

    As GAD_FLUXLIMIT_ADV_X (see there). GRID.h: recip_dyC, recip_deepFacC. The point loop body (:77-101) runs on
    the whole (i,j) range at once (each point reads only inputs).
    """
    sNx, sNy, OLx, OLy = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy
    recip_dyC, recip_deepFacC = grid.recip_dyC, grid.recip_deepFacC

    i = loop_i(1-OLx, sNx+OLx)                                      # :69-73
    vT = vT.at[i, 1-OLy].set(0.)
    vT = vT.at[i, 2-OLy].set(0.)
    vT = vT.at[i, sNy+OLy].set(0.)

    j = loop_j(1-OLy+2, sNy+OLy-1)                                  # :74-103
    vCFL = vFld[i, j]                                               # :77-79
    if calcCFL:
        vCFL = jnp.abs(vFld[i, j]*deltaTloc
                       *recip_dyC[i, j]*recip_deepFacC[k])
    Rjp = (tracer[i, j+1]-tracer[i, j])*maskLocS[i, j+1]            # :80-82
    Rj = (tracer[i, j]-tracer[i, j-1])*maskLocS[i, j]
    Rjm = (tracer[i, j-1]-tracer[i, j-2])*maskLocS[i, j-1]

    Cr = jnp.where(vTrans[i, j] > zeroRL, Rjm, Rjp)                 # :84-88
    big = jnp.abs(Rj)*CrMax <= jnp.abs(Cr)                          # :89-93
    Cr = jnp.where(big, jnp.copysign(CrMax, Cr)*jnp.copysign(oneRL, Rj),
                   safe_div(Cr, Rj, ~big))

#       calculate Limiter Function:
    Cr = Limiter(Cr, p=("b", "a", "b", "b"))                        # :96

    vT = vT.at[i, j].set(                                           # :98-101
            vTrans[i, j]*(tracer[i, j]+tracer[i, j-1])*0.5
          - jnp.abs(vTrans[i, j])*((oneRL-Cr) + vCFL*Cr)
                            *Rj*0.5)
    return vT
