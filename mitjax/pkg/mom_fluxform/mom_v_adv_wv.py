"""pkg/mom_fluxform/mom_v_adv_wv.F: vertical advective flux of meridional momentum (MOM_V_ADV_WV)."""

from mitjax.farray import loop_i, loop_j

_OPT = "MOM_FLUXFORM_OPTIONS.h"     # mom_v_adv_wv.F:1


def mom_v_adv_wv(k, deepFacA, vFld, wFld, rTrans, advectiveFluxWV, *, cfg, grid, params):
    """MOM_V_ADV_WV(bi, bj, k, deepFacA, vFld, wFld, rTrans, advectiveFluxWV, myThid)
    @63cdc0b pkg/mom_fluxform/mom_v_adv_wv.F:3-116

    C Calculates the vertical advective flux of meridional momentum:
    C F^r = \\overline{W}^j \\overline{v}^{k}
    C  k                    :: vertical level
    C  deepFacA             :: deep-model grid factor at level center
    C  vFld                 :: meridional velocity
    C  wFld                 :: vertical velocity
    C  rTrans               :: vertical transport (above V point)
    C  advectiveFluxWV      :: advective flux

    Returns advectiveFluxWV. `k` (1..Nr+1; the caller passes k and k+1) and the PARAMS.h selectors
    (useRealFreshWaterFlux, usingPCoords, rigidLid, select_rStar) are static. deepFacA is `deepFacA(Nr)` (not tiled),
    vFld and wFld are the 3-D fields. All four arms are ported. `halfRL` = 0.5 _d 0 (EEPARAMS.h:73), `0.25 _d 0`
    and `0.` are exact. #ifdef MOM_BOUNDARY_CONSERVE (:84-86) is not ported (raises). The point loops run on the
    whole (i,j) range (each point reads inputs and its own earlier value).
    """
    if cfg.cpp.flag("MOM_BOUNDARY_CONSERVE", _OPT):
        raise NotImplementedError("MOM_V_ADV_WV: MOM_BOUNDARY_CONSERVE is not ported")
    sNx, sNy, OLx, OLy, Nr = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy, cfg.size.Nr
    rA, maskC, deepFac2F = grid.rA, grid.maskC, grid.deepFac2F
    rhoFacF = params.rhoFacF
    halfRL = 0.5                                                    # EEPARAMS.h:73

    if (k == Nr+1 and
            params.useRealFreshWaterFlux and params.usingPCoords):  # :50-57
        j = loop_j(1-OLy+1, sNy+OLy)
        i = loop_i(1-OLx+1, sNx+OLx)
        advectiveFluxWV = advectiveFluxWV.at[i, j].set(rTrans[i, j]*vFld[i, j, k-1]
                                                       * deepFacA[k-1])

    elif k > Nr or (k == 1 and params.rigidLid):                    # :59-66
#     Advective flux = 0  at k=Nr+1 ; = 0 at k=1 if rigid-lid
        j = loop_j(1-OLy, sNy+OLy)
        i = loop_i(1-OLx, sNx+OLx)
        advectiveFluxWV = advectiveFluxWV.at[i, j].set(0.)

    elif k == 1:                                                    # :68-76
#     (linear) Free-surface correction at k=1
        j = loop_j(1-OLy+1, sNy+OLy)
        i = loop_i(1-OLx+1, sNx+OLx)
        advectiveFluxWV = advectiveFluxWV.at[i, j].set(rTrans[i, j]*vFld[i, j, k]
                                                       * deepFacA[k])

    else:                                                           # :78-113
#     Vertical advection - interior ; assume vFld & wFld are masked
        j = loop_j(1-OLy+1, sNy+OLy)
        i = loop_i(1-OLx+1, sNx+OLx)
        advectiveFluxWV = advectiveFluxWV.at[i, j].set(
            rTrans[i, j]*halfRL
            * (vFld[i, j, k]*deepFacA[k]
               + vFld[i, j, k-1]*deepFacA[k-1]))

        if params.select_rStar == 0 and not params.rigidLid:        # :94-110
#     (linear) Free-surface correction at k>1
            advectiveFluxWV = advectiveFluxWV.at[i, j].set(
                advectiveFluxWV[i, j]
                + 0.25*(
                    wFld[i, j, k]*rA[i, j]
                    * (maskC[i, j, k]-maskC[i, j, k-1])
                    + wFld[i, j-1, k]*rA[i, j-1]
                    * (maskC[i, j-1, k]-maskC[i, j-1, k-1])
                    )*deepFac2F[k]*rhoFacF[k]
                * vFld[i, j, k]*deepFacA[k])
    return advectiveFluxWV
