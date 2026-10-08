"""GAD_DST3_ADV_R: vertical advective flux, DST-3 (@63cdc0b pkg/generic_advdiff/gad_dst3_adv_r.F). Lane M4COL
(the vertical scheme of 1D_ocean_ice_column/input: tempVertAdvScheme = saltVertAdvScheme = 30 through GAD_CALC_RHS,
multiDimAdvection = .FALSE.)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.pkg.generic_advdiff.gad_h import oneSixth


def gad_dst3_adv_r(k, dTarg, rTrans, wFld, tracer, wT, *, cfg, grid):
    """GAD_DST3_ADV_R( bi,bj,k, dTarg, rTrans, wFld, tracer, wT, myThid )
    @63cdc0b pkg/generic_advdiff/gad_dst3_adv_r.F:7-122

    C  Calculates the area integrated vertical flux due to advection of a tracer
    C  using 3rd-order Direct Space and Time (DST-3) Advection Scheme
    C  k                 :: vertical level
    C  deltaTloc         :: local time-step (s)
    C  rTrans            :: vertical volume transport
    C  wFld              :: vertical flow
    C  tracer            :: tracer field
    C  wT                :: vertical advective flux

    k: a Python int or a traced level (KIdx, the caller's level scan); dTarg: REAL (traced); rTrans, wFld, wT:
    (1-OLx:sNx+OLx,1-OLy:sNy+OLy); tracer: the same with k=(1,Nr). Returns wT, written on every point (:74-119).
    km2, km1, kp1 (:70-72) are level indices (MAX/MIN of INTEGERs). The point loop body works on scalars per (i,j);
    here every statement runs on the whole range at once (each point reads only inputs). `2.`, `1.`, `0.5` are REAL*4
    literals, exact in binary. #ifdef OLD_DST3_FORMULATION (:59-68, :88-109) is not ported (raise): not defined in
    1D_ocean_ice_column/code (GAD_OPTIONS.h). GRID.h: maskC, recip_drC. The commented-out `wLoc = rTrans*recip_rA`
    (:84) is not code."""
    if cfg.OLD_DST3_FORMULATION:
        raise NotImplementedError("GAD_DST3_ADV_R: OLD_DST3_FORMULATION is not ported")
    sNx, sNy, OLx, OLy, Nr = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy, cfg.Nr
    maskC, recip_drC = grid.maskC, grid.recip_drC

    # :70-72  km2=MAX(1,k-2), km1=MAX(1,k-1), kp1=MIN(Nr,k+1) as `x if x >= 1 else 1` (the same INTEGER; written so
    # that a traced k (KIdx of a level scan) decides each test statically from its range, e.g. k-2 >= 1 on 3..Nr-1)
    km2 = k-2 if k-2 >= 1 else 1                                    # :70; MINMAX-INT: integer (no tie or NaN case)
    km1 = k-1 if k-1 >= 1 else 1                                    # :71; MINMAX-INT: integer (no tie or NaN case)
    kp1 = k+1 if k+1 <= Nr else Nr                                  # :72; MINMAX-INT: integer (no tie or NaN case)

    j = loop_j(1-OLy, sNy+OLy)                                      # :74
    i = loop_i(1-OLx, sNx+OLx)                                      # :75
    Rjp = ((tracer[i, j, k]-tracer[i, j, kp1])                      # :76-77
           * maskC[i, j, kp1])
    Rj = ((tracer[i, j, km1]-tracer[i, j, k])                       # :78-79
          * maskC[i, j, k]*maskC[i, j, km1])
    Rjm = ((tracer[i, j, km2]-tracer[i, j, km1])                    # :80-81
           * maskC[i, j, km1])

    wLoc = wFld[i, j]                                               # :83
    cfl = jnp.abs(wLoc*dTarg*recip_drC[k])                          # :85
    d0 = (2.-cfl)*(1.-cfl)*oneSixth                                 # :86
    d1 = (1.-cfl*cfl)*oneSixth                                      # :87

    wT = wT.at[i, j].set(                                           # :111-115
          0.5*(rTrans[i, j]+jnp.abs(rTrans[i, j]))
             *(tracer[i, j, k] + (d0*Rj+d1*Rjp))
         +0.5*(rTrans[i, j]-jnp.abs(rTrans[i, j]))
             *(tracer[i, j, km1] - (d0*Rj+d1*Rjm)))
    return wT
