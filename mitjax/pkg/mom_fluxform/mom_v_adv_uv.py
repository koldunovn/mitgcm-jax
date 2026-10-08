"""pkg/mom_fluxform/mom_v_adv_uv.F: zonal advective flux of meridional momentum (MOM_V_ADV_UV)."""

from mitjax.farray import loop_i, loop_j

_OPT = "MOM_FLUXFORM_OPTIONS.h"     # mom_v_adv_uv.F:1


def mom_v_adv_uv(k, uTrans, vFld, AdvectFluxUV, *, cfg, grid):
    """MOM_V_ADV_UV(bi,bj,k, uTrans, vFld, AdvectFluxUV, myThid)   @63cdc0b pkg/mom_fluxform/mom_v_adv_uv.F:3-63

    C Calculates the zonal advective flux of meridional momentum:
    C F^x = \\overline{U}^j \\overline{v}^{i}
    C  k                    :: vertical level
    C  uTrans               :: zonal transport
    C  vFld                 :: meridional flow
    C  AdvectFluxUV         :: advective flux

    Returns AdvectFluxUV. `0.25` is a REAL*4 literal (exact). #ifdef MOM_BOUNDARY_CONSERVE (:49-51) and
    OLD_ADV_BCS (:55-58) are not ported (raise). The point loop runs on the whole (i,j) range.
    """
    if cfg.cpp.flag("MOM_BOUNDARY_CONSERVE", _OPT):
        raise NotImplementedError("MOM_V_ADV_UV: MOM_BOUNDARY_CONSERVE is not ported")
    if cfg.cpp.flag("OLD_ADV_BCS", _OPT):
        raise NotImplementedError("MOM_V_ADV_UV: OLD_ADV_BCS is not ported")
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy

    j = loop_j(1-OLy+1, sNy+OLy)                                    # :45-60
    i = loop_i(1-OLx+1, sNx+OLx)
    AdvectFluxUV = AdvectFluxUV.at[i, j].set(
        0.25*(uTrans[i, j] + uTrans[i, j-1])
        * (vFld[i, j] + vFld[i-1, j]))
    return AdvectFluxUV
