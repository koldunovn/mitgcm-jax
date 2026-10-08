"""pkg/mom_fluxform/mom_u_metric_sphere.F: zonal metric term of the spherical grid (MOM_U_METRIC_SPHERE)."""

from mitjax.farray import loop_i, loop_j


def mom_u_metric_sphere(k, uFld, vFld, uMetricTerms, *, cfg, grid, params):
    """MOM_U_METRIC_SPHERE(bi, bj, k, uFld, vFld, uMetricTerms, myThid)
    @63cdc0b pkg/mom_fluxform/mom_u_metric_sphere.F:3-79

    C Calculates the zonal metric term due to sphericity:
    C \\frac{u}{a} \\overline{v}^{ij} \\tan{\\phi}
    C  k                    :: vertical level
    C  uFld                 :: zonal velocity
    C  vFld                 :: meridional velocity
    C  uMetricTerms         :: metric term

    Returns uMetricTerms. selectMetricTerms (PARAMS.h) is static; both arms are ported. `_tanPhiAtU` is tanPhiAtU.
    `-halfRL*(...)` is (-0.5)*(...), the same value as -(0.5*(...)). The point loops run on the whole (i,j) range.
    """
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    recip_deepFacC, tanPhiAtU, dxC, recip_rAz = grid.recip_deepFacC, grid.tanPhiAtU, grid.dxC, grid.recip_rAz
    dxG, hFacS, recip_hFacW, recip_dxC = grid.dxG, grid.hFacS, grid.recip_hFacW, grid.recip_dxC
    recip_rSphere = params.recip_rSphere
    halfRL = 0.5                                                    # EEPARAMS.h:73

    if params.selectMetricTerms == 1:                               # :46-56
#-    Using analytical expression for tan(Phi) (stored in: tanPhiAtU)
        j = loop_j(1-OLy, sNy+OLy-1)
        i = loop_i(1-OLx+1, sNx+OLx)
        uMetricTerms = uMetricTerms.at[i, j].set(
            uFld[i, j]*recip_rSphere*recip_deepFacC[k]*0.25
            * (vFld[i, j] + vFld[i-1, j]
               + vFld[i, j+1] + vFld[i-1, j+1]
               )*tanPhiAtU[i, j])
    else:                                                           # :57-76
#-    Using grid-spacing gradient: estimates tan(Phi)/rSphere as -del^j(dxC)/rAz
        j = loop_j(2-OLy, sNy+OLy-1)
        i = loop_i(2-OLx, sNx+OLx)
        uMetricTerms = uMetricTerms.at[i, j].set(
            -halfRL
            * ((uFld[i, j-1] + uFld[i, j])*halfRL
               * (dxC[i, j]-dxC[i, j-1])*recip_rAz[i, j]
               * (vFld[i-1, j]*dxG[i-1, j]*hFacS[i-1, j, k]
                  + vFld[i, j]*dxG[i, j]*hFacS[i, j, k]
                  )*halfRL
               + (uFld[i, j] + uFld[i, j+1])*halfRL
               * (dxC[i, j+1]-dxC[i, j])*recip_rAz[i, j+1]
               * (vFld[i-1, j+1]*dxG[i-1, j+1]*hFacS[i-1, j+1, k]
                  + vFld[i, j+1]*dxG[i, j+1]*hFacS[i, j+1, k]
                  )*halfRL
               )*recip_hFacW[i, j, k]
            * recip_dxC[i, j]*recip_deepFacC[k])
    return uMetricTerms
