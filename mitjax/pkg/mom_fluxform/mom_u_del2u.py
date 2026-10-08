"""pkg/mom_fluxform/mom_u_del2u.F: Laplacian of the zonal flow (MOM_U_DEL2U)."""

from mitjax.farray import loop_i, loop_j


def _opt(cfg):
    """mom_u_del2u.F:1-4: MOM_FLUXFORM_OPTIONS.h, then MOM_COMMON_OPTIONS.h under #ifdef ALLOW_MOM_COMMON."""
    if cfg.cpp.flag("ALLOW_MOM_COMMON", "MOM_FLUXFORM_OPTIONS.h"):
        return "MOM_COMMON_OPTIONS.h"
    return "MOM_FLUXFORM_OPTIONS.h"


def mom_u_del2u(k, uFld, hFacZ, h0FacZ, del2u, *, cfg, grid, params):
    """MOM_U_DEL2U(bi, bj, k, uFld, hFacZ, h0FacZ, del2u, myThid)   @63cdc0b pkg/mom_fluxform/mom_u_del2u.F:6-125

    C Calculates the Laplacian of zonal flow
    C  k                    :: vertical level
    C  uFld                 :: zonal flow
    C  hFacZ                :: fractional thickness at vorticity points
    C  h0FacZ               :: fixed fractional thickness at vorticity points
    C  del2u                :: Laplacian

    Returns del2u. no_slip_sides (PARAMS.h) is static, sideDragFactor a float parameter. Locals fZon, fMer (:47-48)
    start as NaN (every point read is written first). #ifdef COSINEMETH_III (:59-61) follows the build;
    #ifdef NONLIN_FRSURF (:104-110) is ported (global_ocean.90x40x15). Not ported (raise): ALLOW_OBCS (:62-64,
    :94-96) and ISOTROPIC_COS_SCALING with COSINEMETH_III (:77-79). `_hFacC`, `_dyF`, `_recip_dxF`, `_dxV`,
    `_recip_dyU`, `_recip_hFacW`, `_maskW`, `_hFacW` are the plain GRID.h fields. The point loops run on the whole
    (i,j) range: the flux loops read inputs only, the del2u loops read the fluxes and their own earlier value.
    """
    opt = _opt(cfg)
    if cfg.cpp.flag("ALLOW_OBCS", opt):
        raise NotImplementedError("MOM_U_DEL2U: ALLOW_OBCS is not ported")
    if cfg.cpp.flag("ISOTROPIC_COS_SCALING", opt) and cfg.cpp.flag("COSINEMETH_III", opt):
        raise NotImplementedError("MOM_U_DEL2U: ISOTROPIC_COS_SCALING with COSINEMETH_III is not ported")
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    drF, hFacC, dyF, recip_dxF, dxV, recip_dyU = (grid.drF, grid.hFacC, grid.dyF, grid.recip_dxF, grid.dxV,
                                                  grid.recip_dyU)
    recip_drF, recip_hFacW, recip_rAw, recip_deepFac2C, maskW = (grid.recip_drF, grid.recip_hFacW, grid.recip_rAw,
                                                                 grid.recip_deepFac2C, grid.maskW)
    h0W = grid.h0FacW if cfg.cpp.flag("NONLIN_FRSURF", opt) else grid.hFacW   # :104-110

    fZon = del2u.local("fZon")                                      # :47  _RL fZon(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
    fMer = del2u.local("fMer")                                      # :48

#     Zonal flux d/dx U
    j = loop_j(1-OLy+1, sNy+OLy-1)                                  # :53-68
    i = loop_i(1-OLx, sNx+OLx-1)
    f = (drF[k]*hFacC[i, j, k]
         * dyF[i, j]
         * recip_dxF[i, j]
         * (uFld[i+1, j]-uFld[i, j]))
    if cfg.cpp.flag("COSINEMETH_III", opt):                         # :59-61
        f = f * grid.sqCosFacU[j]
    fZon = fZon.at[i, j].set(f)

#     Meridional flux d/dy U
    j = loop_j(1-OLy+1, sNy+OLy)                                    # :71-83
    i = loop_i(1-OLx+1, sNx+OLx-1)
    fMer = fMer.at[i, j].set(drF[k]*hFacZ[i, j]
                             * dxV[i, j]
                             * recip_dyU[i, j]
                             * (uFld[i, j]-uFld[i, j-1]))

#     del^2 U
    j = loop_j(1-OLy+1, sNy+OLy-1)                                  # :86-98
    i = loop_i(1-OLx+1, sNx+OLx-1)
    del2u = del2u.at[i, j].set(
        recip_drF[k]*recip_hFacW[i, j, k]
        * recip_rAw[i, j]*recip_deepFac2C[k]
        * (fZon[i, j] - fZon[i-1, j]
           + fMer[i, j+1] - fMer[i, j]
           )*maskW[i, j, k])

    if params.no_slip_sides:                                        # :100-122
#-- No-slip BCs impose a drag at walls...
        hFacZClosedS = h0W[i, j, k] - h0FacZ[i, j]
        hFacZClosedN = h0W[i, j, k] - h0FacZ[i, j+1]
        del2u = del2u.at[i, j].set(
            del2u[i, j]
            - recip_hFacW[i, j, k]
            * recip_rAw[i, j]*recip_deepFac2C[k]
            * (hFacZClosedS*dxV[i, j]
               * recip_dyU[i, j]
               + hFacZClosedN*dxV[i, j+1]
               * recip_dyU[i, j+1]
               )*uFld[i, j]*params.sideDragFactor
            * maskW[i, j, k])
    return del2u
