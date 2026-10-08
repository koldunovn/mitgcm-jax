"""GAD_DST3_ADV_X: zonal advective flux, DST-3 (@63cdc0b pkg/generic_advdiff/gad_dst3_adv_x.F); from the Task 8
prototype `dev/prototype/style_farray.py` (gated there, job 27826890), unchanged apart from the imports."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.pkg.generic_advdiff.gad_h import oneSixth


def gad_dst3_adv_x(k, calcCFL, deltaTloc, uTrans, uFld, maskLocW, tracer, uT, *, cfg, grid):
    """GAD_DST3_ADV_X(bi,bj,k, calcCFL, deltaTloc, uTrans, uFld, maskLocW, tracer, uT, myThid)
    @63cdc0b pkg/generic_advdiff/gad_dst3_adv_x.F:7-121

    C !DESCRIPTION:
    C  Calculates the area integrated zonal flux due to advection of a
    C  tracer using 3rd-order Direct Space and Time (DST-3) Advection Scheme
    C !INPUT PARAMETERS:
    C  k                 :: vertical level
    C  calcCFL           :: =T: calculate CFL number ; =F: take uFld as CFL.
    C  deltaTloc         :: local time-step (s)
    C  uTrans            :: zonal volume transport
    C  uFld              :: zonal flow / CFL number
    C  tracer            :: tracer field
    C !OUTPUT PARAMETERS:
    C  uT                :: zonal advective flux

    The Fortran spells the argument `tracer` also as `Tracer` (:112, :114; Fortran is case-insensitive). The point
    loop body (:78-114) works on scalars per (i,j); here every statement runs on the whole (i,j) range at once, which
    is the same computation because each point reads only inputs. `2.`, `1.`, `0.5`, `0.` are REAL*4 literals,
    exact in binary. #ifdef OLD_DST3_FORMULATION (:87-108) is not ported: not defined in any M1 build. No M1
    experiment executes this routine (docs/coverage); it is the Task 8 canonical example.
    """
    if cfg.OLD_DST3_FORMULATION:
        raise NotImplementedError("GAD_DST3_ADV_X: OLD_DST3_FORMULATION is not ported")
    sNx, sNy, OLx, OLy = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy
    recip_dxC, recip_deepFacC = grid.recip_dxC, grid.recip_deepFacC

    j = loop_j(1-OLy, sNy+OLy)                                      # :71-75
    uT = uT.at[1-OLx, j].set(0.)
    uT = uT.at[2-OLx, j].set(0.)
    uT = uT.at[sNx+OLx, j].set(0.)

    j = loop_j(1-OLy, sNy+OLy)                                      # :76-118
    i = loop_i(1-OLx+2, sNx+OLx-1)
    Rjp = (tracer[i+1, j]-tracer[i, j])*maskLocW[i+1, j]
    Rj = (tracer[i, j]-tracer[i-1, j])*maskLocW[i, j]
    Rjm = (tracer[i-1, j]-tracer[i-2, j])*maskLocW[i-1, j]

    uCFL = uFld[i, j]
    if calcCFL:
        uCFL = jnp.abs(uFld[i, j]*deltaTloc
                       *recip_dxC[i, j]*recip_deepFacC[k])
    d0 = (2.-uCFL)*(1.-uCFL)*oneSixth
    d1 = (1.-uCFL*uCFL)*oneSixth
    uT = uT.at[i, j].set(
          0.5*(uTrans[i, j]+jnp.abs(uTrans[i, j]))
             *(tracer[i-1, j] + (d0*Rj+d1*Rjm))
         +0.5*(uTrans[i, j]-jnp.abs(uTrans[i, j]))
             *(tracer[i, j] - (d0*Rj+d1*Rjp)))
    return uT
