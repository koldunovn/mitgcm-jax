"""pkg/mom_common/mom_calc_hdiv.F: horizontal divergence of the flow (MOM_CALC_HDIV)."""

from mitjax.farray import loop_i, loop_j

_OPT = "MOM_COMMON_OPTIONS.h"       # mom_calc_hdiv.F:1


def mom_calc_hdiv(k, hDivScheme, uFld, vFld, hDiv, *, cfg, grid):
    """MOM_CALC_HDIV(bi,bj,k,hDivScheme, uFld, vFld, hDiv, myThid)   @63cdc0b pkg/mom_common/mom_calc_hdiv.F:3-78

    C     | S/R MOM_CALC_HDIV
    C     hDivScheme = 1: the straight forward horizontal divergence that only considers the horizontal grid
    C                     variations
    C     hDivScheme = 2: takes into account the fractional areas due to the lopping

    Returns hDiv. `hDivScheme` is static (an argument that selects the branch; MOM_VECINV passes 2). #ifdef
    ALLOW_AUTODIFF (:30-37) is ported (global_ocean.90x40x15/code_ad); #ifdef ALLOW_OBCS (:49-51, :67-69) is not
    (undefined in every M2 build; raises). The point loops run on the whole (i,j) range.
    """
    if cfg.cpp.flag("ALLOW_OBCS", _OPT):
        raise NotImplementedError("MOM_CALC_HDIV: ALLOW_OBCS (maskInC) is not ported")
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    dxG, dyG, recip_rA, recip_deepFacC = grid.dxG, grid.dyG, grid.recip_rA, grid.recip_deepFacC
    hFacW, hFacS, recip_hFacC = grid.hFacW, grid.hFacS, grid.recip_hFacC

    if cfg.cpp.flag("ALLOW_AUTODIFF", _OPT):                       # :30-37
        jA = loop_j(1-OLy, sNy+OLy)
        iA = loop_i(1-OLx, sNx+OLx)
        hDiv = hDiv.at[iA, jA].set(0.)

    j = loop_j(1-OLy, sNy+OLy-1)
    i = loop_i(1-OLx, sNx+OLx-1)
    if hDivScheme == 1:                                             # :38-53
        hDiv = hDiv.at[i, j].set((
              uFld[i+1, j]*dyG[i+1, j]
             -uFld[i, j]*dyG[i, j]
             +vFld[i, j+1]*dxG[i, j+1]
             -vFld[i, j]*dxG[i, j]
                    )*recip_rA[i, j]*recip_deepFacC[k])

    elif hDivScheme == 2:                                           # :55-71
        hDiv = hDiv.at[i, j].set(
           ((uFld[i+1, j]*dyG[i+1, j]*hFacW[i+1, j, k]
             -uFld[i, j]*dyG[i, j]*hFacW[i, j, k])
            +(vFld[i, j+1]*dxG[i, j+1]*hFacS[i, j+1, k]
              -vFld[i, j]*dxG[i, j]*hFacS[i, j, k])
           )*recip_rA[i, j]*recip_deepFacC[k]
            *recip_hFacC[i, j, k])

    else:                                                           # :73-75
        raise ValueError("S/R MOM_CALC_HDIV: We should never reach this point!")

    return hDiv
