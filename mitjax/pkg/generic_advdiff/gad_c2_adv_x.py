"""GAD_C2_ADV_X: zonal advective flux, centered second-order (@63cdc0b pkg/generic_advdiff/gad_c2_adv_x.F)."""

from mitjax.farray import loop_i, loop_j


def gad_c2_adv_x(k, uTrans, tracer, uT, *, cfg):
    """GAD_C2_ADV_X( bi,bj,k, uTrans, tracer, uT, myThid )   @63cdc0b pkg/generic_advdiff/gad_c2_adv_x.F:7-57

    C !DESCRIPTION:
    C Calculates the area integrated zonal flux due to advection of a tracer using
    C centered second-order interpolation:
    C F^x_{adv} = U \\overline{\\theta}^i
    C !INPUT PARAMETERS:
    C  k                    :: vertical level
    C  uTrans               :: zonal volume transport
    C  tracer               :: tracer field
    C !OUTPUT PARAMETERS:
    C  uT                   :: zonal advective flux

    No CPP branches. `k` is not used by the Fortran (kept: same arguments). The Fortran spells `tracer` also as
    `Tracer` (:52; case-insensitive). The (i,j) nest runs on the whole range at once: each point reads only inputs.
    `0.` is a REAL*4 literal, exact; `0.5 _d 0` is the double 0.5.
    """
    sNx, sNy, OLx, OLy = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy

    j = loop_j(1-OLy, sNy+OLy)                                      # :46-48
    uT = uT.at[1-OLx, j].set(0.)
    i = loop_i(1-OLx+1, sNx+OLx)                                    # :49-54
    uT = uT.at[i, j].set(
          uTrans[i, j]*(tracer[i, j]+tracer[i-1, j])*0.5)
    return uT
