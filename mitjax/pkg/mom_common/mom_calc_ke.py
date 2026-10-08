"""pkg/mom_common/mom_calc_ke.F: kinetic energy of the horizontal flow (MOM_CALC_KE)."""

from mitjax.farray import loop_i, loop_j

_OPT = "MOM_COMMON_OPTIONS.h"       # mom_calc_ke.F:1


def mom_calc_ke(k, KEscheme, uFld, vFld, KE, *, cfg, grid):
    """MOM_CALC_KE(bi,bj,k,KEscheme, uFld, vFld, KE, myThid)   @63cdc0b pkg/mom_common/mom_calc_ke.F:7-141

    C !DESCRIPTION:
    C Calculates the Kinetic energy of horizontal flow
    C KE = \\frac{1}{2} \\left( h_w \\overline{u^2}^i + h_s \\overline{v^2}^j \\right)
    C !INPUT PARAMETERS:
    C  k                    :: vertical level
    C  KEscheme             :: spacial discretisation scheme for KE
    C  uFld                 :: zonal flow
    C  vFld                 :: meridional flow
    C !OUTPUT PARAMETERS:
    C  KE                   :: Kinetic energy

    Moved from the Task 8 prototype (dev/prototype/style_farray.py) with the real configuration (`cfg.size`,
    `cfg.cpp`). `0.125`, `0.25` and `0.` are Fortran REAL*4 literals, exact in binary, so their float64 values are the
    same. `KEscheme` is static (an argument that selects the branch). #ifdef ALLOW_AUTODIFF (:53-59) is ported
    (tutorial_global_oce_optim/code_ad).
    """
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    rAw, rAs, recip_rA = grid.rAw, grid.rAs, grid.recip_rA
    hFacW, hFacS, recip_hFacC = grid.hFacW, grid.hFacS, grid.recip_hFacC

    if cfg.cpp.flag("ALLOW_AUTODIFF", _OPT):                       # :53-59
        j = loop_j(1-OLy, sNy+OLy)
        i = loop_i(1-OLx, sNx+OLx)
        KE = KE.at[i, j].set(0.)

    j = loop_j(1-OLy, sNy+OLy-1)                                    # DO j=1-OLy,sNy+OLy-1 (:62, 77, ...)
    i = loop_i(1-OLx, sNx+OLx-1)
    if KEscheme == -1:                                              # :61-68
        KE = KE.at[i, j].set(0.125*(
                   (uFld[i, j]+uFld[i+1, j])**2
                  +(vFld[i, j]+vFld[i, j+1])**2))

    elif KEscheme == 0:                                             # :70-86
        KE = KE.at[i, j].set(0.25*(
                 (uFld[i, j]*uFld[i, j]
                  +uFld[i+1, j]*uFld[i+1, j])
               + (vFld[i, j]*vFld[i, j]
                  +vFld[i, j+1]*vFld[i, j+1])
                        ))

    elif KEscheme == 1:                                             # :88-99
        KE = KE.at[i, j].set(0.25*(
                 (uFld[i, j]*uFld[i, j]*rAw[i, j]
                  +uFld[i+1, j]*uFld[i+1, j]*rAw[i+1, j])
               + (vFld[i, j]*vFld[i, j]*rAs[i, j]
                  +vFld[i, j+1]*vFld[i, j+1]*rAs[i, j+1])
                        )*recip_rA[i, j])

    elif KEscheme == 2:                                             # :101-113
        KE = KE.at[i, j].set(0.25*(
                 (uFld[i, j]*uFld[i, j]*hFacW[i, j, k]
                  +uFld[i+1, j]*uFld[i+1, j]*hFacW[i+1, j, k])
               + (vFld[i, j]*vFld[i, j]*hFacS[i, j, k]
                  +vFld[i, j+1]*vFld[i, j+1]*hFacS[i, j+1, k])
                  )*recip_hFacC[i, j, k])

    elif KEscheme == 3:                                             # :115-134
        KE = KE.at[i, j].set(0.25*(
                 (
          uFld[i, j]*uFld[i, j]
              *hFacW[i, j, k]*rAw[i, j]
         +uFld[i+1, j]*uFld[i+1, j]
              *hFacW[i+1, j, k]*rAw[i+1, j]
                 )
               + (
          vFld[i, j]*vFld[i, j]
              *hFacS[i, j, k]*rAs[i, j]
         +vFld[i, j+1]*vFld[i, j+1]
              *hFacS[i, j+1, k]*rAs[i, j+1]
                 ))*recip_hFacC[i, j, k]
                         *recip_rA[i, j])

    else:                                                           # :136-137
        raise ValueError("S/R MOM_CALC_KE: We should never reach this point!")

    return KE
