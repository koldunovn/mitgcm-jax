"""GAD_DST3FL_ADV_R: vertical advective flux, DST-3 with flux limiting (@63cdc0b pkg/generic_advdiff/gad_dst3fl_adv_r.F).
PTRACERS lane (the vertical scheme of tutorial_global_oce_latlon's passive tracer, PTRACERS_advScheme = 33)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX, MIN
from mitjax.ops.safe import safe_div
from mitjax.pkg.generic_advdiff.gad_h import oneSixth

# gad_dst3fl_adv_r.F:56-57   _RL thetaMax; PARAMETER( thetaMax = 1.D+20 )
thetaMax = 1.0e+20


def gad_dst3fl_adv_r(k, dTarg, rTrans, wFld, tracer, wT, *, cfg, grid):
    """GAD_DST3FL_ADV_R( bi,bj,k, dTarg, rTrans, wFld, tracer, wT, myThid )
    @63cdc0b pkg/generic_advdiff/gad_dst3fl_adv_r.F:7-120

    C  Calculates the area integrated vertical flux due to advection of a tracer
    C  using 3rd Order DST Scheme with flux limiting
    C  k                 :: vertical level
    C  deltaTloc         :: local time-step (s)
    C  rTrans            :: vertical volume transport
    C  wFld              :: vertical flow
    C  tracer            :: tracer field
    C  wT                :: vertical advective flux

    k: Python int; dTarg: REAL (traced); rTrans, wFld, wT: (1-OLx:sNx+OLx,1-OLy:sNy+OLy); tracer: the same with
    k=(1,Nr). Returns wT, written on every point (:63-117). km2, km1, kp1 (:59-61) are host integers. The point
    loop body works on scalars per (i,j); here every statement runs on the whole range at once (each point reads
    only inputs). As GAD_DST3FL_ADV_X (mitjax/pkg/generic_advdiff/gad_dst3fl_adv_x.py): the two-way IFs (:92-101)
    are `where`s with the division guarded (`safe_div`, bitwise Rjm/Rj where the Fortran divides), SIGN is
    `copysign`, MAX/MIN carry gfortran's per-site winners (tables of the tutorial_global_oce_latlon and
    tutorial_tracer_adjsens builds), `0.5*` REAL*4 exact; the `#if (ALLOW_AUTODIFF && TARGET_NEC_SX)` initialisation
    (:65-69) is ported. GRID.h: maskC, recip_drC. The commented-out old versions (:84-90) are not ported."""
    sNx, sNy, OLx, OLy, Nr = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy, cfg.Nr
    maskC, recip_drC = grid.maskC, grid.recip_drC

    km2 = max(1, k-2)                                               # :59; MINMAX-INT: integer (no tie or NaN case)
    km1 = max(1, k-1)                                               # :60; MINMAX-INT: integer (no tie or NaN case)
    kp1 = min(Nr, k+1)                                              # :61; MINMAX-INT: integer (no tie or NaN case)

    j = loop_j(1-OLy, sNy+OLy)                                      # :63
    i = loop_i(1-OLx, sNx+OLx)                                      # :64
    if cfg.ALLOW_AUTODIFF and cfg.TARGET_NEC_SX:                    # :65-69
        thetaP = 0.
        thetaM = 0.
    Rjp = ((tracer[i, j, k]-tracer[i, j, kp1])                      # :70-71
           * maskC[i, j, kp1])
    Rj = ((tracer[i, j, km1]-tracer[i, j, k])                       # :72-73
          * maskC[i, j, k]*maskC[i, j, km1])
    Rjm = ((tracer[i, j, km2]-tracer[i, j, km1])                    # :74-75
           * maskC[i, j, km1])

    wLoc = wFld[i, j]                                               # :77
    wCFL = jnp.abs(wLoc*dTarg*recip_drC[k])                         # :78
    d0 = (2.-wCFL)*(1.-wCFL)*oneSixth                               # :79
    d1 = (1.-wCFL*wCFL)*oneSixth                                    # :80

    bigP = jnp.abs(Rj)*thetaMax <= jnp.abs(Rjm)                     # :92-96
    thetaP = jnp.where(bigP, jnp.copysign(thetaMax, Rjm*Rj),
                       safe_div(Rjm, Rj, ~bigP))
    bigM = jnp.abs(Rj)*thetaMax <= jnp.abs(Rjp)                     # :97-101
    thetaM = jnp.where(bigM, jnp.copysign(thetaMax, Rjp*Rj),
                       safe_div(Rjp, Rj, ~bigM))

    psiP = d0+d1*thetaP                                             # :103
    psiP = MAX(0., MIN(MIN(1., psiP, p="b"),                       # :104-105
                       thetaP*(1.-wCFL)/(wCFL+1.e-20), p="b"), p="b")
    psiM = d0+d1*thetaM                                             # :106
    psiM = MAX(0., MIN(MIN(1., psiM, p="b"),                       # :107-108
                       thetaM*(1.-wCFL)/(wCFL+1.e-20), p="b"), p="b")

    wT = wT.at[i, j].set(                                           # :110-114
          0.5*(rTrans[i, j]+jnp.abs(rTrans[i, j]))
             *(tracer[i, j, k] + psiM*Rj)
         +0.5*(rTrans[i, j]-jnp.abs(rTrans[i, j]))
             *(tracer[i, j, km1] - psiP*Rj))
    return wT
