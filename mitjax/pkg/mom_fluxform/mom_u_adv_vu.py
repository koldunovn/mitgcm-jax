"""pkg/mom_fluxform/mom_u_adv_vu.F: meridional advective flux of zonal momentum (MOM_U_ADV_VU)."""

from mitjax.farray import loop_i, loop_j

_OPT = "MOM_FLUXFORM_OPTIONS.h"     # mom_u_adv_vu.F:1


def mom_u_adv_vu(k, vTrans, uFld, AdvectFluxVU, *, cfg, grid, params):
    """MOM_U_ADV_VU(bi, bj, k, vTrans, uFld, AdvectFluxVU, myThid)   @63cdc0b pkg/mom_fluxform/mom_u_adv_vu.F:3-81

    C Calculates the meridional advective flux of zonal momentum:
    C F^y = \\overline{V}^i \\overline{u}^{j}
    C  k                    :: vertical level
    C  vTrans               :: meridional transport
    C  uFld                 :: zonal velocity
    C  AdvectFluxVU         :: advective flux

    Returns AdvectFluxVU. selectMetricTerms (PARAMS.h) is static; both arms (:46-62 and :63-78, u*dxC advected) are
    ported. `0.25 _d 0` is the double 0.25 (exact). #ifdef MOM_BOUNDARY_CONSERVE (:51-53, :69-71) and OLD_ADV_BCS
    (:57-60) are not ported (raise). The point loops run on the whole (i,j) range.
    """
    if cfg.cpp.flag("MOM_BOUNDARY_CONSERVE", _OPT):
        raise NotImplementedError("MOM_U_ADV_VU: MOM_BOUNDARY_CONSERVE is not ported")
    if cfg.cpp.flag("OLD_ADV_BCS", _OPT):
        raise NotImplementedError("MOM_U_ADV_VU: OLD_ADV_BCS is not ported")
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    dxC = grid.dxC

    j = loop_j(1-OLy+1, sNy+OLy)
    i = loop_i(1-OLx+1, sNx+OLx)
    if params.selectMetricTerms != 3:                               # :46-62
        AdvectFluxVU = AdvectFluxVU.at[i, j].set(
            0.25
            * (vTrans[i, j] + vTrans[i-1, j])
            * (uFld[i, j] + uFld[i, j-1]))
    else:                                                           # :63-78
#-    Advect u*dxC (--> account for metric term u*v*tanPhi/R)
        AdvectFluxVU = AdvectFluxVU.at[i, j].set(
            0.25
            * (vTrans[i, j] + vTrans[i-1, j])
            * (uFld[i, j]*dxC[i, j]
               + uFld[i, j-1]*dxC[i, j-1]))
    return AdvectFluxVU
