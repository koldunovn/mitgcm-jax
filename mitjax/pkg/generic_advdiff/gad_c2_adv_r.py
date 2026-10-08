"""GAD_C2_ADV_R: vertical advective flux, centered second-order (@63cdc0b pkg/generic_advdiff/gad_c2_adv_r.F)."""

from mitjax.farray import loop_i, loop_j


def gad_c2_adv_r(k, rTrans, tracer, wT, *, cfg, grid):
    """GAD_C2_ADV_R( bi, bj, k, rTrans, tracer, wT, myThid )   @63cdc0b pkg/generic_advdiff/gad_c2_adv_r.F:7-68

    C !DESCRIPTION:
    C Calculates the area integrated vertical flux due to advection of a tracer
    C using centered second-order interpolation:
    C F^r_{adv} = W \\overline{\\theta}^k
    C !INPUT PARAMETERS:
    C  k                    :: vertical level
    C  rTrans               :: vertical volume transport
    C  tracer               :: tracer field
    C !OUTPUT PARAMETERS:
    C  wT                   :: vertical advective flux

    `tracer` is declared (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr). GRID.h: maskC. The branch on k (:51) is static (`k` is a
    Python int); GAD_CALC_RHS calls this routine for k >= 2 only, the k = 1 branch is ported as written. The (i,j)
    nest runs on the whole range at once: each point reads only inputs. `0.` is a REAL*4 literal, exact.
    """
    sNx, sNy, OLx, OLy, Nr = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy, cfg.Nr
    maskC = grid.maskC

    km1 = max(1, k-1)                                               # :49; MINMAX-INT: integer (no tie or NaN case)

    j = loop_j(1-OLy, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx)
    if k == 1 or k > Nr:                                            # :51-56
        wT = wT.at[i, j].set(0.)
    else:                                                           # :57-65
        wT = wT.at[i, j].set(maskC[i, j, km1]*
            rTrans[i, j]*
               (tracer[i, j, k]+tracer[i, j, km1])*0.5)
    return wT
