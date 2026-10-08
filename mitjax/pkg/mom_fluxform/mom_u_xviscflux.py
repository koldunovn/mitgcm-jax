"""pkg/mom_fluxform/mom_u_xviscflux.F: zonal viscous flux of U (MOM_U_XVISCFLUX)."""

from mitjax.farray import loop_i, loop_j


def _opt(cfg):
    """mom_u_xviscflux.F:1-4: MOM_FLUXFORM_OPTIONS.h, then MOM_COMMON_OPTIONS.h under #ifdef ALLOW_MOM_COMMON."""
    if cfg.cpp.flag("ALLOW_MOM_COMMON", "MOM_FLUXFORM_OPTIONS.h"):
        return "MOM_COMMON_OPTIONS.h"
    return "MOM_FLUXFORM_OPTIONS.h"


def mom_u_xviscflux(k, uFld, del2u, xViscFluxU, viscAh_D, viscA4_D, *, cfg, grid):
    """MOM_U_XVISCFLUX(bi,bj,k, uFld, del2u, xViscFluxU, viscAh_D, viscA4_D, myThid)
    @63cdc0b pkg/mom_fluxform/mom_u_xviscflux.F:6-74

    C Calculates the area integrated zonal viscous fluxes of U:
    C F^x = - \\frac{ \\Delta y_f \\Delta r_f h_c }{\\Delta x_f}
    C  ( A_h \\delta_i u - A_4 \\delta_i \\nabla^2 u )
    C  k                    :: vertical level
    C  uFld                 :: zonal flow
    C  del2u                :: Laplacian of zonal flow
    C  xViscFluxU           :: viscous fluxes

    Returns xViscFluxU. #ifdef COSINEMETH_III (:62-66) follows the build (defined in every M1 build). `_dyF`,
    `_hFacC`, `_recip_dxF` are dyF, hFacC, recip_dxF. The point loop runs on the whole (i,j) range.
    """
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    dyF, drF, hFacC, recip_dxF = grid.dyF, grid.drF, grid.hFacC, grid.recip_dxF
    cosFacU = grid.cosFacU
    cF4 = grid.sqCosFacU if cfg.cpp.flag("COSINEMETH_III", _opt(cfg)) else grid.cosFacU   # :62-66

#     - Laplacian  and bi-harmonic terms
    j = loop_j(1-OLy, sNy+OLy-1)                                    # :54-71
    i = loop_i(1-OLx, sNx+OLx-1)
    xViscFluxU = xViscFluxU.at[i, j].set(
        dyF[i, j]*drF[k]*hFacC[i, j, k]
        * (
           -viscAh_D[i, j]*(uFld[i+1, j]-uFld[i, j])
           * cosFacU[j]
           + viscA4_D[i, j]*(del2u[i+1, j]-del2u[i, j])
           * cF4[j]
          )*recip_dxF[i, j])
    return xViscFluxU
