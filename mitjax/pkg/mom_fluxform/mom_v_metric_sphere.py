"""pkg/mom_fluxform/mom_v_metric_sphere.F: meridional metric term of the spherical grid (MOM_V_METRIC_SPHERE)."""

from mitjax.farray import loop_i, loop_j


def mom_v_metric_sphere(k, uFld, vMetricTerms, *, cfg, grid, params):
    """MOM_V_METRIC_SPHERE(bi, bj, k, uFld, vMetricTerms, myThid)
    @63cdc0b pkg/mom_fluxform/mom_v_metric_sphere.F:3-82

    C Calculates the meridional metric term due to sphericity:
    C \\frac{1}{a} \\overline{u}^{ij} \\overline{u}^{ij} \\tan{\\phi}
    C  k                    :: vertical level
    C  uFld                 :: zonal velocity
    C  vMetricTerms         :: metric term

    Returns vMetricTerms. selectMetricTerms (PARAMS.h) is static; both arms are ported. `_tanPhiAtV` is tanPhiAtV.
    `-recip_rSphere*...` is (-recip_rSphere)*..., the same value as -(recip_rSphere*...); `+halfRL*` is halfRL*.
    The point loops run on the whole (i,j) range.
    """
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    recip_deepFacC, tanPhiAtV, dxC, recip_rAz = grid.recip_deepFacC, grid.tanPhiAtV, grid.dxC, grid.recip_rAz
    dyG, hFacW, recip_hFacS, recip_dyC = grid.dyG, grid.hFacW, grid.recip_hFacS, grid.recip_dyC
    recip_rSphere = params.recip_rSphere
    halfRL = 0.5                                                    # EEPARAMS.h:73

    if params.selectMetricTerms == 1:                               # :44-57
#-    Use analytical expression for tan(Phi) (stored in: tanPhiAtV)
        j = loop_j(1-OLy+1, sNy+OLy)
        i = loop_i(1-OLx, sNx+OLx-1)
        vMetricTerms = vMetricTerms.at[i, j].set(
            -recip_rSphere*recip_deepFacC[k]
            * 0.25*(uFld[i, j] + uFld[i+1, j]
                    + uFld[i, j-1] + uFld[i+1, j-1]
                    )
            * 0.25*(uFld[i, j] + uFld[i+1, j]
                    + uFld[i, j-1] + uFld[i+1, j-1]
                    )
            * tanPhiAtV[i, j])
    else:                                                           # :58-79
#-    Using grid-spacing gradient: estimates tan(Phi)/rSphere as -del^j(dxC)/rAz
        j = loop_j(2-OLy, sNy+OLy)
        i = loop_i(1-OLx, sNx+OLx-1)
        vMetricTerms = vMetricTerms.at[i, j].set(
            +halfRL
            * ((uFld[i, j-1] + uFld[i, j])*halfRL
               * (dxC[i, j]
                  - dxC[i, j-1])*recip_rAz[i, j]
               * (uFld[i, j-1]*dyG[i, j-1]*hFacW[i, j-1, k]
                  + uFld[i, j]*dyG[i, j]*hFacW[i, j, k]
                  )*halfRL
               + (uFld[i+1, j-1] + uFld[i+1, j])*halfRL
               * (dxC[i+1, j]
                  - dxC[i+1, j-1])*recip_rAz[i+1, j]
               * (uFld[i+1, j-1]*dyG[i+1, j-1]*hFacW[i+1, j-1, k]
                  + uFld[i+1, j]*dyG[i+1, j]*hFacW[i+1, j, k]
                  )*halfRL
               )*recip_hFacS[i, j, k]
            * recip_dyC[i, j]*recip_deepFacC[k])
    return vMetricTerms
