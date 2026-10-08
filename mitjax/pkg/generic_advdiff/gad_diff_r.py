"""GAD_DIFF_R: vertical diffusive flux (@63cdc0b pkg/generic_advdiff/gad_diff_r.F)."""

from mitjax.farray import loop_i, loop_j


def gad_diff_r(k, maskUp, KappaR, tracer, dfr, *, cfg, grid, params):
    """GAD_DIFF_R( bi, bj, k, maskUp, KappaR, tracer, dfr, myThid )   @63cdc0b pkg/generic_advdiff/gad_diff_r.F:7-70

    C !DESCRIPTION:
    C Calculates the area integrated vertical flux due to down-gradient
    C diffusion of a tracer:
    C F^r_{diff} = - A^r \\kappa_r \\frac{1}{\\Delta r_c} \\delta_k \\theta
    C !INPUT PARAMETERS:
    C  k                    :: vertical level
    C  maskUp               :: 2-D array for mask at W points
    C  KappaR               :: vertical diffusivity
    C  tracer               :: tracer field
    C !OUTPUT PARAMETERS:
    C  dfr                  :: vertical diffusive flux

    `tracer` is declared (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr). GRID.h: rA (`_rA` expands to rA(i,j,bi,bj) in every M1
    build), deepFac2F, recip_drC, rkSign; PARAMS.h: rhoFacF (`params`). The branch on k (:52) is static (`k` is a
    Python int); GAD_CALC_RHS calls this routine for k = 1..Nr. Unary minus as in GAD_DIFF_X. The (i,j) nest runs on
    the whole range at once: each point reads only inputs. `0.` is a REAL*4 literal, exact. No CPP branches.
    """
    sNx, sNy, OLx, OLy, Nr = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy, cfg.Nr
    rA, deepFac2F, recip_drC, rkSign = grid.rA, grid.deepFac2F, grid.recip_drC, grid.rkSign
    rhoFacF = params.rhoFacF

    km1 = max(1, k-1)                                               # :50; MINMAX-INT: integer (no tie or NaN case)

    j = loop_j(1-OLy, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx)
    if k == 1 or k > Nr:                                            # :52-57
        dfr = dfr.at[i, j].set(0.)
    else:                                                           # :58-67
        dfr = dfr.at[i, j].set(-(KappaR[i, j]*maskUp[i, j]
                   *rA[i, j]*deepFac2F[k]*rhoFacF[k]
                   *recip_drC[k]
                   *(tracer[i, j, k]-tracer[i, j, km1])*rkSign))
    return dfr
