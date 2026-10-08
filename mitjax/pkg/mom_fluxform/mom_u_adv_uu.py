"""pkg/mom_fluxform/mom_u_adv_uu.F: zonal advective flux of zonal momentum (MOM_U_ADV_UU)."""

from mitjax.farray import loop_i, loop_j

_OPT = "MOM_FLUXFORM_OPTIONS.h"     # mom_u_adv_uu.F:1


def mom_u_adv_uu(k, uTrans, uFld, AdvectFluxUU, *, cfg, grid):
    """MOM_U_ADV_UU(bi,bj,k, uTrans, uFld, AdvectFluxUU, myThid)   @63cdc0b pkg/mom_fluxform/mom_u_adv_uu.F:3-60

    C Calculates the zonal advective flux of zonal momentum:
    C F^x = \\overline{U}^i \\overline{u}^{i}
    C  k                    :: vertical level
    C  uTrans               :: zonal transport
    C  uFld                 :: zonal flow
    C  AdvectFluxUU         :: advective flux

    Returns AdvectFluxUU. `0.25` is a REAL*4 literal (exact). #ifdef MOM_BOUNDARY_CONSERVE (:50-52; #undef in
    MOM_FLUXFORM_OPTIONS.h) is not ported (raises). The point loop runs on the whole (i,j) range.
    """
    if cfg.cpp.flag("MOM_BOUNDARY_CONSERVE", _OPT):
        raise NotImplementedError("MOM_U_ADV_UU: MOM_BOUNDARY_CONSERVE is not ported")
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy

    j = loop_j(1-OLy, sNy+OLy-1)                                    # :46-57
    i = loop_i(1-OLx, sNx+OLx-1)
    AdvectFluxUU = AdvectFluxUU.at[i, j].set(
        0.25*(uTrans[i, j] + uTrans[i+1, j])
        * (uFld[i, j] + uFld[i+1, j]))
    return AdvectFluxUU
