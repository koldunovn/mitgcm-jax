"""GAD_DST3_ADV_Y: meridional advective flux, DST-3 (@63cdc0b pkg/generic_advdiff/gad_dst3_adv_y.F); GOADK lane (M2,
global_ocean.90x40x15/code_ad, tempAdvScheme = saltAdvScheme = 30); the meridional twin of gad_dst3_adv_x.py."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.pkg.generic_advdiff.gad_h import oneSixth


def gad_dst3_adv_y(k, calcCFL, deltaTloc, vTrans, vFld, maskLocS, tracer, vT, *, cfg, grid):
    """GAD_DST3_ADV_Y(bi,bj,k, calcCFL, deltaTloc, vTrans, vFld, maskLocS, tracer, vT, myThid)
    @63cdc0b pkg/generic_advdiff/gad_dst3_adv_y.F:7-120

    C !DESCRIPTION:
    C  Calculates the area integrated Meridional flux due to advection of a
    C  tracer using 3rd-order Direct Space and Time (DST-3) Advection Scheme
    C !INPUT PARAMETERS:
    C  k                 :: vertical level
    C  calcCFL           :: =T: calculate CFL number ; =F: take vFld as CFL
    C  deltaTloc         :: local time-step (s)
    C  vTrans            :: meridional volume transport
    C  vFld              :: meridional flow / CFL number
    C  tracer            :: tracer field
    C !OUTPUT PARAMETERS:
    C  vT                :: meridional advective flux

    `cfg`: the GAD kernel configuration (gad_calc_rhs.gad_kernel_cfg: sNx, sNy, OLx, OLy, OLD_DST3_FORMULATION);
    `grid`: GRID.h (recip_dyC, recip_deepFacC). The Fortran spells the argument `tracer` also as `Tracer` (:111, :113;
    Fortran is case-insensitive). The point loop body (:77-113) works on scalars per (i,j); here every statement runs
    on the whole (i,j) range at once, which is the same computation because each point reads only inputs. `2.`, `1.`,
    `0.5`, `0.` are REAL*4 literals, exact in binary. #ifdef OLD_DST3_FORMULATION (:86-107) is not ported (raise): not
    defined in the global_ocean.90x40x15/code_ad build (GAD_OPTIONS.h of pkg/generic_advdiff).
    """
    if cfg.OLD_DST3_FORMULATION:
        raise NotImplementedError("GAD_DST3_ADV_Y: OLD_DST3_FORMULATION is not ported")
    sNx, sNy, OLx, OLy = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy
    recip_dyC, recip_deepFacC = grid.recip_dyC, grid.recip_deepFacC

    i = loop_i(1-OLx, sNx+OLx)                                      # :70-74
    vT = vT.at[i, 1-OLy].set(0.)
    vT = vT.at[i, 2-OLy].set(0.)
    vT = vT.at[i, sNy+OLy].set(0.)

    j = loop_j(1-OLy+2, sNy+OLy-1)                                  # :75-117
    i = loop_i(1-OLx, sNx+OLx)
    Rjp = (tracer[i, j+1]-tracer[i, j])*maskLocS[i, j+1]
    Rj = (tracer[i, j]-tracer[i, j-1])*maskLocS[i, j]
    Rjm = (tracer[i, j-1]-tracer[i, j-2])*maskLocS[i, j-1]

    vCFL = vFld[i, j]
    if calcCFL:
        vCFL = jnp.abs(vFld[i, j]*deltaTloc
                       *recip_dyC[i, j]*recip_deepFacC[k])
    d0 = (2.-vCFL)*(1.-vCFL)*oneSixth
    d1 = (1.-vCFL*vCFL)*oneSixth
    vT = vT.at[i, j].set(
          0.5*(vTrans[i, j]+jnp.abs(vTrans[i, j]))
             *(tracer[i, j-1] + (d0*Rj+d1*Rjm))
         +0.5*(vTrans[i, j]-jnp.abs(vTrans[i, j]))
             *(tracer[i, j] - (d0*Rj+d1*Rjp)))
    return vT
