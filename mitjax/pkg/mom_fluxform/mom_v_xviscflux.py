"""pkg/mom_fluxform/mom_v_xviscflux.F: zonal viscous flux of V (MOM_V_XVISCFLUX)."""

from mitjax.farray import loop_i, loop_j


def _opt(cfg):
    """mom_v_xviscflux.F:1-4: MOM_FLUXFORM_OPTIONS.h, then MOM_COMMON_OPTIONS.h under #ifdef ALLOW_MOM_COMMON."""
    if cfg.cpp.flag("ALLOW_MOM_COMMON", "MOM_FLUXFORM_OPTIONS.h"):
        return "MOM_COMMON_OPTIONS.h"
    return "MOM_FLUXFORM_OPTIONS.h"


def mom_v_xviscflux(k, vFld, del2v, hFacZ, xViscFluxV, viscAh_Z, viscA4_Z, *, cfg, grid):
    """MOM_V_XVISCFLUX(bi,bj,k, vFld, del2v, hFacZ, xViscFluxV, viscAh_Z, viscA4_Z, myThid)
    @63cdc0b pkg/mom_fluxform/mom_v_xviscflux.F:6-75

    C Calculates the area integrated zonal viscous fluxes of V:
    C F^x = - \\frac{ \\Delta x_u \\Delta r_f h_z }{\\Delta x_v}
    C  ( A_h \\delta_i v - A_4 \\delta_i \\nabla^2 v )
    C  k                    :: vertical level
    C  vFld                 :: meridional flow
    C  del2v                :: Laplacian of meridional flow
    C  xViscFluxU           :: viscous fluxes

    Returns xViscFluxV. #ifdef COSINEMETH_III (:63-67) follows the build. `_dyU`, `_recip_dxV` are dyU, recip_dxV.
    The point loop runs on the whole (i,j) range.
    """
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    dyU, drF, recip_dxV = grid.dyU, grid.drF, grid.recip_dxV
    cosFacV = grid.cosFacV
    cF4 = grid.sqCosFacV if cfg.cpp.flag("COSINEMETH_III", _opt(cfg)) else grid.cosFacV   # :63-67

#     - Laplacian  and bi-harmonic terms
    j = loop_j(1-OLy, sNy+OLy)                                      # :55-72
    i = loop_i(1-OLx+1, sNx+OLx)
    xViscFluxV = xViscFluxV.at[i, j].set(
        dyU[i, j]*drF[k]*hFacZ[i, j]
        * (
           -viscAh_Z[i, j]*(vFld[i, j]-vFld[i-1, j])
           * cosFacV[j]
           + viscA4_Z[i, j]*(del2v[i, j]-del2v[i-1, j])
           * cF4[j]
          )*recip_dxV[i, j])
    return xViscFluxV
