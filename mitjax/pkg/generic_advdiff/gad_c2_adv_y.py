"""GAD_C2_ADV_Y: meridional advective flux, centered second-order (@63cdc0b pkg/generic_advdiff/gad_c2_adv_y.F)."""

from mitjax.farray import loop_i, loop_j


def gad_c2_adv_y(k, vTrans, tracer, vT, *, cfg):
    """GAD_C2_ADV_Y( bi,bj,k, vTrans, tracer, vT, myThid )   @63cdc0b pkg/generic_advdiff/gad_c2_adv_y.F:7-57

    C !DESCRIPTION:
    C Calculates the area integrated meridional flux due to advection of a tracer
    C using centered second-order interpolation:
    C F^y_{adv} = V \\overline{\\theta}^j
    C !INPUT PARAMETERS:
    C  k                    :: vertical level
    C  vTrans               :: meridional volume transport
    C  tracer               :: tracer field
    C !OUTPUT PARAMETERS:
    C  vT                   :: meridional advective flux

    No CPP branches. `k` is not used by the Fortran (kept: same arguments). The (i,j) nest runs on the whole range
    at once: each point reads only inputs. `0.` is a REAL*4 literal, exact; `0.5 _d 0` is the double 0.5.
    """
    sNx, sNy, OLx, OLy = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy

    i = loop_i(1-OLx, sNx+OLx)                                      # :46-48
    vT = vT.at[i, 1-OLy].set(0.)
    j = loop_j(1-OLy+1, sNy+OLy)                                    # :49-54
    vT = vT.at[i, j].set(
          vTrans[i, j]*(tracer[i, j]+tracer[i, j-1])*0.5)
    return vT
