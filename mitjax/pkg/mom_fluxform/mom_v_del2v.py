"""pkg/mom_fluxform/mom_v_del2v.F: Laplacian of the meridional flow (MOM_V_DEL2V)."""

from mitjax.farray import loop_i, loop_j


def _opt(cfg):
    """mom_v_del2v.F:1-4: MOM_FLUXFORM_OPTIONS.h, then MOM_COMMON_OPTIONS.h under #ifdef ALLOW_MOM_COMMON."""
    if cfg.cpp.flag("ALLOW_MOM_COMMON", "MOM_FLUXFORM_OPTIONS.h"):
        return "MOM_COMMON_OPTIONS.h"
    return "MOM_FLUXFORM_OPTIONS.h"


def mom_v_del2v(k, vFld, hFacZ, h0FacZ, del2v, *, cfg, grid, params):
    """MOM_V_DEL2V(bi, bj, k, vFld, hFacZ, h0FacZ, del2v, myThid)   @63cdc0b pkg/mom_fluxform/mom_v_del2v.F:6-124

    C Calculates the Laplacian of meridional flow
    C  k                    :: vertical level
    C  vFld                 :: meridional flow
    C  hFacZ                :: fractional thickness at vorticity points
    C  h0FacZ               :: fixed fractional thickness at vorticity points
    C  del2v                :: Laplacian

    Returns del2v. As MOM_U_DEL2U with the V-point stencils: #ifdef COSINEMETH_III (:58-60) follows the build,
    NONLIN_FRSURF (:103-109) is ported, ALLOW_OBCS (:76-78, :93-95) and ISOTROPIC_COS_SCALING with COSINEMETH_III
    (:73-75) raise.
    """
    opt = _opt(cfg)
    if cfg.cpp.flag("ALLOW_OBCS", opt):
        raise NotImplementedError("MOM_V_DEL2V: ALLOW_OBCS is not ported")
    if cfg.cpp.flag("ISOTROPIC_COS_SCALING", opt) and cfg.cpp.flag("COSINEMETH_III", opt):
        raise NotImplementedError("MOM_V_DEL2V: ISOTROPIC_COS_SCALING with COSINEMETH_III is not ported")
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    drF, hFacC, dxF, recip_dyF, dyU, recip_dxV = (grid.drF, grid.hFacC, grid.dxF, grid.recip_dyF, grid.dyU,
                                                  grid.recip_dxV)
    recip_drF, recip_hFacS, recip_rAs, recip_deepFac2C, maskS = (grid.recip_drF, grid.recip_hFacS, grid.recip_rAs,
                                                                 grid.recip_deepFac2C, grid.maskS)
    h0S = grid.h0FacS if cfg.cpp.flag("NONLIN_FRSURF", opt) else grid.hFacS   # :103-109

    fZon = del2v.local("fZon")                                      # :46  _RL fZon(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
    fMer = del2v.local("fMer")                                      # :47

#     Zonal flux d/dx V
    j = loop_j(1-OLy+1, sNy+OLy-1)                                  # :52-64
    i = loop_i(1-OLx+1, sNx+OLx)
    f = (drF[k]*hFacZ[i, j]
         * dyU[i, j]
         * recip_dxV[i, j]
         * (vFld[i, j]-vFld[i-1, j]))
    if cfg.cpp.flag("COSINEMETH_III", opt):                         # :58-60
        f = f * grid.sqCosFacV[j]
    fZon = fZon.at[i, j].set(f)

#     Meridional flux d/dy V
    j = loop_j(1-OLy, sNy+OLy-1)                                    # :67-82
    i = loop_i(1-OLx+1, sNx+OLx-1)
    fMer = fMer.at[i, j].set(drF[k]*hFacC[i, j, k]
                             * dxF[i, j]
                             * recip_dyF[i, j]
                             * (vFld[i, j+1]-vFld[i, j]))

#     del^2 V
    j = loop_j(1-OLy+1, sNy+OLy-1)                                  # :85-97
    i = loop_i(1-OLx+1, sNx+OLx-1)
    del2v = del2v.at[i, j].set(
        recip_drF[k]*recip_hFacS[i, j, k]
        * recip_rAs[i, j]*recip_deepFac2C[k]
        * (fZon[i+1, j] - fZon[i, j]
           + fMer[i, j] - fMer[i, j-1]
           )*maskS[i, j, k])

    if params.no_slip_sides:                                        # :99-121
#-- No-slip BCs impose a drag at walls...
        hFacZClosedW = h0S[i, j, k] - h0FacZ[i, j]
        hFacZClosedE = h0S[i, j, k] - h0FacZ[i+1, j]
        del2v = del2v.at[i, j].set(
            del2v[i, j]
            - recip_hFacS[i, j, k]
            * recip_rAs[i, j]*recip_deepFac2C[k]
            * (hFacZClosedW*dyU[i, j]
               * recip_dxV[i, j]
               + hFacZClosedE*dyU[i+1, j]
               * recip_dxV[i+1, j]
               )*vFld[i, j]*params.sideDragFactor
            * maskS[i, j, k])
    return del2v
