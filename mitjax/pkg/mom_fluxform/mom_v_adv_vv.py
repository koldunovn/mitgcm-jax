"""pkg/mom_fluxform/mom_v_adv_vv.F: meridional advective flux of meridional momentum (MOM_V_ADV_VV)."""

from mitjax.farray import loop_i, loop_j

_OPT = "MOM_FLUXFORM_OPTIONS.h"     # mom_v_adv_vv.F:1


def mom_v_adv_vv(k, vTrans, vFld, AdvectFluxVV, *, cfg, grid):
    """MOM_V_ADV_VV(bi,bj,k, vTrans, vFld, AdvectFluxVV, myThid)   @63cdc0b pkg/mom_fluxform/mom_v_adv_vv.F:3-60

    C Calculates the meridional advective flux of meridional momentum:
    C F^y = \\overline{V}^j \\overline{v}^{j}
    C  k                    :: vertical level
    C  vTrans               :: meridional transport
    C  vFld                 :: meridional flow
    C  AdvectFluxVV         :: advective flux

    Returns AdvectFluxVV. `0.25` is a REAL*4 literal (exact). #ifdef MOM_BOUNDARY_CONSERVE (:50-52) is not ported
    (raises). The point loop runs on the whole (i,j) range.
    """
    if cfg.cpp.flag("MOM_BOUNDARY_CONSERVE", _OPT):
        raise NotImplementedError("MOM_V_ADV_VV: MOM_BOUNDARY_CONSERVE is not ported")
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy

    j = loop_j(1-OLy, sNy+OLy-1)                                    # :46-57
    i = loop_i(1-OLx, sNx+OLx-1)
    AdvectFluxVV = AdvectFluxVV.at[i, j].set(
        0.25*(vTrans[i, j] + vTrans[i, j+1])
        * (vFld[i, j] + vFld[i, j+1]))
    return AdvectFluxVV
