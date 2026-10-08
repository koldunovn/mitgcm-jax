"""pkg/mom_fluxform/mom_v_coriolis.F: horizontal Coriolis term of the meridional momentum equation (MOM_V_CORIOLIS)."""

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX


def mom_v_coriolis(k, uFld, vCoriolisTerm, *, cfg, grid, params):
    """MOM_V_CORIOLIS(bi, bj, k, uFld, vCoriolisTerm, myThid)   @63cdc0b pkg/mom_fluxform/mom_v_coriolis.F:3-96

    C Calculates the horizontal Coriolis term in the meridional equation:
    C -\\overline{f}^j \\overline{u}^{ij}
    C  k                    :: vertical level
    C  uFld                 :: zonal flow
    C  vCoriolisTerm        :: Coriolis term

    Returns vCoriolisTerm. As MOM_U_CORIOLIS with the V-point stencil; `-halfRL*(...)` is (-0.5)*(...), the same
    value as -(0.5*(...)) (negation is exact).
    """
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    fCori, dyG, hFacW, recip_dyC, recip_hFacS, maskW = (grid.fCori, grid.dyG, grid.hFacW, grid.recip_dyC,
                                                        grid.recip_hFacS, grid.maskW)
    halfRL, oneRS = 0.5, 1.0                                        # EEPARAMS.h:73, :69
    sel = params.selectCoriScheme

    j = loop_j(1-OLy+1, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx-1)
    if sel <= 1:                                                    # :44-54
#-    Original discretization
        vCoriolisTerm = vCoriolisTerm.at[i, j].set(
            -halfRL
            * (fCori[i, j] + fCori[i, j-1])
            * 0.25*(uFld[i, j] + uFld[i+1, j]
                    + uFld[i, j-1] + uFld[i+1, j-1]
                    ))
    elif sel <= 3:                                                  # :55-65
#-    Energy conserving discretization
        vCoriolisTerm = vCoriolisTerm.at[i, j].set(
            -halfRL
            * (fCori[i, j]
               * halfRL*(uFld[i, j] + uFld[i+1, j])
               + fCori[i, j-1]
               * halfRL*(uFld[i, j-1] + uFld[i+1, j-1])))
    else:                                                           # :66-79
#-    Using averaged transport:
        vCoriolisTerm = vCoriolisTerm.at[i, j].set(
            -halfRL
            * (fCori[i, j] + fCori[i, j-1])
            * (uFld[i, j]*dyG[i, j]*hFacW[i, j, k]
               + uFld[i+1, j]*dyG[i+1, j]*hFacW[i+1, j, k]
               + uFld[i, j-1]*dyG[i, j-1]*hFacW[i, j-1, k]
               + uFld[i+1, j-1]*dyG[i+1, j-1]*hFacW[i+1, j-1, k]
               )*0.25*recip_dyC[i, j]*recip_hFacS[i, j, k])

    if sel == 1 or sel == 3:                                        # :81-93
#-    Scale term so that only "wet" points are used
        vCoriolisTerm = vCoriolisTerm.at[i, j].set(
            vCoriolisTerm[i, j]
            * 4./MAX(oneRS,                                         # :87-90
                     maskW[i, j, k]+maskW[i+1, j, k]
                     + maskW[i, j-1, k]+maskW[i+1, j-1, k], p="a"))
    return vCoriolisTerm
