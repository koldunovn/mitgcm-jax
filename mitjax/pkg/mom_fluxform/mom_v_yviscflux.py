"""pkg/mom_fluxform/mom_v_yviscflux.F: meridional viscous flux of V (MOM_V_YVISCFLUX)."""

from mitjax.farray import loop_i, loop_j


def _opt(cfg):
    """mom_v_yviscflux.F:1-4: MOM_FLUXFORM_OPTIONS.h, then MOM_COMMON_OPTIONS.h under #ifdef ALLOW_MOM_COMMON."""
    if cfg.cpp.flag("ALLOW_MOM_COMMON", "MOM_FLUXFORM_OPTIONS.h"):
        return "MOM_COMMON_OPTIONS.h"
    return "MOM_FLUXFORM_OPTIONS.h"


def mom_v_yviscflux(k, vFld, del2v, yViscFluxV, viscAh_D, viscA4_D, *, cfg, grid):
    """MOM_V_YVISCFLUX(bi,bj,k, vFld, del2v, yViscFluxV, viscAh_D, viscA4_D, myThid)
    @63cdc0b pkg/mom_fluxform/mom_v_yviscflux.F:6-79

    C Calculates the area integrated meridional viscous fluxes of V:
    C F^y = - \\frac{ \\Delta x_f \\Delta r_f h_c }{\\Delta y_f}
    C  ( A_h \\delta_j v - A_4 \\delta_j \\nabla^2 v )
    C  k                    :: vertical level
    C  vFld                 :: meridional flow
    C  del2v                :: Laplacian of meridional flow
    C  yViscFluxV           :: viscous fluxes

    Returns yViscFluxV. #ifdef ISOTROPIC_COS_SCALING (:60-70) follows the build (defined by
    MLAdjust/code/MOM_COMMON_OPTIONS.h:22): cosFacU on the harmonic term, sqCosFacU (COSINEMETH_III) else cosFacU on
    the biharmonic one. `_dxF`, `_hFacC`,
    `_recip_dyF` are dxF, hFacC, recip_dyF. The point loop runs on the whole (i,j) range.
    """
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    dxF, drF, hFacC, recip_dyF = grid.dxF, grid.drF, grid.hFacC, grid.recip_dyF

#     - Laplacian  and bi-harmonic terms
    j = loop_j(1-OLy, sNy+OLy-1)                                    # :54-76
    i = loop_i(1-OLx, sNx+OLx-1)
    if cfg.cpp.flag("ISOTROPIC_COS_SCALING", _opt(cfg)):            # :60-70
        cosFacU = grid.cosFacU
        cF4 = grid.sqCosFacU if cfg.cpp.flag("COSINEMETH_III", _opt(cfg)) else grid.cosFacU   # :65-69
        yViscFluxV = yViscFluxV.at[i, j].set(
            dxF[i, j]*drF[k]*hFacC[i, j, k]
            * (
               -viscAh_D[i, j]*(vFld[i, j+1]-vFld[i, j])
               * cosFacU[j]
               + viscA4_D[i, j]*(del2v[i, j+1]-del2v[i, j])
               * cF4[j]
              )*recip_dyF[i, j])
        return yViscFluxV
    yViscFluxV = yViscFluxV.at[i, j].set(
        dxF[i, j]*drF[k]*hFacC[i, j, k]
        * (
           -viscAh_D[i, j]*(vFld[i, j+1]-vFld[i, j])
           + viscA4_D[i, j]*(del2v[i, j+1]-del2v[i, j])
          )*recip_dyF[i, j])
    return yViscFluxV
