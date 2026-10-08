"""pkg/mom_common/mom_u_metric_nh.F: non-hydrostatic zonal metric term (MOM_U_METRIC_NH)."""

from mitjax.farray import loop_i, loop_j


def mom_u_metric_nh(k, uFld, wFld, uMetricTerms, *, cfg, grid, params):
    """MOM_U_METRIC_NH(bi,bj,k, uFld, wFld, uMetricTerms, myThid)   @63cdc0b pkg/mom_common/mom_u_metric_nh.F:3-63

    C Calculates the zonal metric term due to non-hydrostaticity on the sphere:
    C -\\frac{u}{a} \\overline{w}^{ik}
    C  k                    :: vertical level
    C  uFld                 :: zonal flow
    C  wFld                 :: vertical flow
    C  uMetricTerms         :: metric term

    Returns uMetricTerms. `k` is static; kp1 and wVelBottomOverride (:47-49) are trace-time values. wFld is the 3-D
    field. `0.25`, `1.`, `0.` are REAL*4 literals (exact). The point loop runs on the whole (i,j) range.
    """
    sNx, sNy, OLx, OLy, Nr = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy, cfg.size.Nr
    recip_deepFacC, gravitySign = grid.recip_deepFacC, grid.gravitySign
    recip_rSphere, rVel2wUnit = params.recip_rSphere, params.rVel2wUnit

    kp1 = min(k+1, Nr)                                              # :47-49; MINMAX-INT: integer (no tie or NaN case)
    wVelBottomOverride = 1.
    if k == Nr:
        wVelBottomOverride = 0.

    j = loop_j(1-OLy, sNy+OLy)                                      # :51-60
    i = loop_i(1-OLx+1, sNx+OLx)
    uMetricTerms = uMetricTerms.at[i, j].set(
        uFld[i, j]*recip_rSphere*recip_deepFacC[k]
        * 0.25*((wFld[i-1, j, kp1]+wFld[i, j, kp1])
                * rVel2wUnit[kp1]*wVelBottomOverride
                + (wFld[i-1, j, k]+wFld[i, j, k])
                * rVel2wUnit[k]
                )*gravitySign)
    return uMetricTerms
