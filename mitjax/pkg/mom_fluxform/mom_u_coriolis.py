"""pkg/mom_fluxform/mom_u_coriolis.F: horizontal Coriolis term of the zonal momentum equation (MOM_U_CORIOLIS)."""

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX


def mom_u_coriolis(k, vFld, uCoriolisTerm, *, cfg, grid, params):
    """MOM_U_CORIOLIS(bi, bj, k, vFld, uCoriolisTerm, myThid)   @63cdc0b pkg/mom_fluxform/mom_u_coriolis.F:3-96

    C Calculates the horizontal Coriolis term in the zonal equation:
    C \\overline{f}^i \\overline{v}^{ij}
    C  k                    :: vertical level
    C  vFld                 :: meridional flow
    C  uCoriolisTerm        :: Coriolis term

    Returns uCoriolisTerm. selectCoriScheme (PARAMS.h) is static; all arms are ported. `_fCori` is fCori
    (FCORI_MACROS.h). `halfRL` = 0.5 _d 0, `0.25 _d 0`, `4. _d 0`, `oneRS` = 1 are exact. The point loops run on the
    whole (i,j) range (each point reads inputs and its own earlier value).
    """
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    fCori, dxG, hFacS, recip_dxC, recip_hFacW, maskS = (grid.fCori, grid.dxG, grid.hFacS, grid.recip_dxC,
                                                        grid.recip_hFacW, grid.maskS)
    halfRL, oneRS = 0.5, 1.0                                        # EEPARAMS.h:73, :69
    sel = params.selectCoriScheme

    j = loop_j(1-OLy, sNy+OLy-1)
    i = loop_i(1-OLx+1, sNx+OLx)
    if sel <= 1:                                                    # :44-54
#-    Original discretization
        uCoriolisTerm = uCoriolisTerm.at[i, j].set(
            halfRL
            * (fCori[i, j] + fCori[i-1, j])
            * 0.25*(vFld[i, j] + vFld[i, j+1]
                    + vFld[i-1, j] + vFld[i-1, j+1]
                    ))
    elif sel <= 3:                                                  # :55-65
#-    Energy conserving discretization
        uCoriolisTerm = uCoriolisTerm.at[i, j].set(
            halfRL
            * (fCori[i, j]
               * halfRL*(vFld[i, j] + vFld[i, j+1])
               + fCori[i-1, j]
               * halfRL*(vFld[i-1, j] + vFld[i-1, j+1])))
    else:                                                           # :66-79
#-    Using averaged transport:
        uCoriolisTerm = uCoriolisTerm.at[i, j].set(
            halfRL
            * (fCori[i, j] + fCori[i-1, j])
            * (vFld[i, j]*dxG[i, j]*hFacS[i, j, k]
               + vFld[i, j+1]*dxG[i, j+1]*hFacS[i, j+1, k]
               + vFld[i-1, j]*dxG[i-1, j]*hFacS[i-1, j, k]
               + vFld[i-1, j+1]*dxG[i-1, j+1]*hFacS[i-1, j+1, k]
               )*0.25*recip_dxC[i, j]*recip_hFacW[i, j, k])

    if sel == 1 or sel == 3:                                        # :81-93
#-    Scale term so that only "wet" points are used
        uCoriolisTerm = uCoriolisTerm.at[i, j].set(
            uCoriolisTerm[i, j]
            * 4./MAX(oneRS,                                         # :87-90
                     maskS[i, j, k]+maskS[i, j+1, k]
                     + maskS[i-1, j, k]+maskS[i-1, j+1, k], p="a"))
    return uCoriolisTerm
