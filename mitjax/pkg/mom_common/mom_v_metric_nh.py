"""pkg/mom_common/mom_v_metric_nh.F: non-hydrostatic meridional metric term (MOM_V_METRIC_NH)."""

from mitjax.farray import loop_i, loop_j


def mom_v_metric_nh(k, vFld, wFld, vMetricTerms, *, cfg, grid, params):
    """MOM_V_METRIC_NH(bi,bj,k, vFld, wFld, vMetricTerms, myThid)   @63cdc0b pkg/mom_common/mom_v_metric_nh.F:3-63

    C Calculates the zonal metric term due to non-hydrostaticity on the sphere:
    C -\\frac{v}{a} \\overline{w}^{jk}
    C  k                    :: vertical level
    C  vFld                 :: merdional flow
    C  wFld                 :: vertical flow
    C  vMetricTerms         :: metric term

    Returns vMetricTerms. As MOM_U_METRIC_NH with the V-point stencil.
    """
    sNx, sNy, OLx, OLy, Nr = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy, cfg.size.Nr
    recip_deepFacC, gravitySign = grid.recip_deepFacC, grid.gravitySign
    recip_rSphere, rVel2wUnit = params.recip_rSphere, params.rVel2wUnit

    kp1 = min(k+1, Nr)                                              # :47-49; MINMAX-INT: integer (no tie or NaN case)
    wVelBottomOverride = 1.
    if k == Nr:
        wVelBottomOverride = 0.

    j = loop_j(1-OLy+1, sNy+OLy)                                    # :51-60
    i = loop_i(1-OLx, sNx+OLx)
    vMetricTerms = vMetricTerms.at[i, j].set(
        vFld[i, j]*recip_rSphere*recip_deepFacC[k]
        * 0.25*((wFld[i, j-1, kp1]+wFld[i, j, kp1])
                * rVel2wUnit[kp1]*wVelBottomOverride
                + (wFld[i, j-1, k]+wFld[i, j, k])
                * rVel2wUnit[k]
                )*gravitySign)
    return vMetricTerms
