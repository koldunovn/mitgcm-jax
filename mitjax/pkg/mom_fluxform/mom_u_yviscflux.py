"""pkg/mom_fluxform/mom_u_yviscflux.F: meridional viscous flux of U (MOM_U_YVISCFLUX)."""

from mitjax.farray import loop_i, loop_j


def _opt(cfg):
    """mom_u_yviscflux.F:1-4: MOM_FLUXFORM_OPTIONS.h, then MOM_COMMON_OPTIONS.h under #ifdef ALLOW_MOM_COMMON."""
    if cfg.cpp.flag("ALLOW_MOM_COMMON", "MOM_FLUXFORM_OPTIONS.h"):
        return "MOM_COMMON_OPTIONS.h"
    return "MOM_FLUXFORM_OPTIONS.h"


def mom_u_yviscflux(k, uFld, del2u, hFacZ, yViscFluxU, viscAh_Z, viscA4_Z, *, cfg, grid):
    """MOM_U_YVISCFLUX(bi,bj,k, uFld, del2u, hFacZ, yViscFluxU, viscAh_Z, viscA4_Z, myThid)
    @63cdc0b pkg/mom_fluxform/mom_u_yviscflux.F:6-79

    C Calculates the area integrated meridional viscous fluxes of U:
    C F^y = - \\frac{ \\Delta y_v \\Delta r_f h_z }{\\Delta y_u}
    C  ( A_h \\delta_j u - A_4 \\delta_j \\nabla^2 u )
    C  k                    :: vertical level
    C  uFld                 :: zonal flow
    C  del2u                :: Laplacian of zonal flow
    C  yViscFluxU           :: viscous fluxes

    Returns yViscFluxU. #ifdef ISOTROPIC_COS_SCALING (:61-71) follows the build (#undef in pkg/mom_common's
    MOM_COMMON_OPTIONS.h, defined by MLAdjust/code/MOM_COMMON_OPTIONS.h:22): cosFacV on the harmonic term, sqCosFacV
    (COSINEMETH_III) else cosFacV on the biharmonic one. `_dxV`, `_recip_dyU` are dxV, recip_dyU. The point loop runs on the whole (i,j) range.
    """
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    dxV, drF, recip_dyU = grid.dxV, grid.drF, grid.recip_dyU

#     - Laplacian  and bi-harmonic terms
    j = loop_j(1-OLy+1, sNy+OLy)                                    # :55-76
    i = loop_i(1-OLx, sNx+OLx)
    if cfg.cpp.flag("ISOTROPIC_COS_SCALING", _opt(cfg)):            # :61-71
        cosFacV = grid.cosFacV
        cF4 = grid.sqCosFacV if cfg.cpp.flag("COSINEMETH_III", _opt(cfg)) else grid.cosFacV   # :66-70
        yViscFluxU = yViscFluxU.at[i, j].set(
            dxV[i, j]*drF[k]*hFacZ[i, j]
            * (
               -viscAh_Z[i, j]*(uFld[i, j]-uFld[i, j-1])
               * cosFacV[j]
               + viscA4_Z[i, j]*(del2u[i, j]-del2u[i, j-1])
               * cF4[j]
              )*recip_dyU[i, j])
        return yViscFluxU
    yViscFluxU = yViscFluxU.at[i, j].set(
        dxV[i, j]*drF[k]*hFacZ[i, j]
        * (
           -viscAh_Z[i, j]*(uFld[i, j]-uFld[i, j-1])
           + viscA4_Z[i, j]*(del2u[i, j]-del2u[i, j-1])
          )*recip_dyU[i, j])
    return yViscFluxU
