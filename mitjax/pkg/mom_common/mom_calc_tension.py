"""pkg/mom_common/mom_calc_tension.F: tension of the horizontal flow at tracer points (MOM_CALC_TENSION)."""

from mitjax.farray import loop_i, loop_j

_OPT = "MOM_COMMON_OPTIONS.h"       # mom_calc_tension.F:1


def mom_calc_tension(k, uFld, vFld, tension, *, cfg, grid):
    """MOM_CALC_TENSION(bi,bj,k, uFld, vFld, tension, myThid)   @63cdc0b pkg/mom_common/mom_calc_tension.F:7-72

    C Calculates the tension of the horizontal flow field (at tracer points):
    C D_T = \\frac{\\Delta y_f}{\\Delta x_f} \\delta_i \\frac{u}{\\Delta y_g}
    C     - \\frac{\\Delta x_f}{\\Delta y_f} \\delta_j \\frac{v}{\\Delta x_g}
    C  k                    :: vertical level
    C  uFld                 :: zonal flow
    C  vFld                 :: meridional flow
    C  tension              :: tension of horizontal flow

    Returns tension (points outside DO j=1-OLy,sNy+OLy-1 / DO i=1-OLx,sNx+OLx-1 keep their prior values). #ifdef
    ALLOW_OBCS (:57-59) is not ported (raises); the commented-out alternative (:60-66) is not ported.
    """
    if cfg.cpp.flag("ALLOW_OBCS", _OPT):
        raise NotImplementedError("MOM_CALC_TENSION: ALLOW_OBCS (maskInC) is not ported")
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    dxG, dyG, recip_rA, recip_deepFacC = grid.dxG, grid.dyG, grid.recip_rA, grid.recip_deepFacC

    j = loop_j(1-OLy, sNy+OLy-1)                                    # :46-69
    i = loop_i(1-OLx, sNx+OLx-1)
    tension = tension.at[i, j].set(
        (dyG[i+1, j]*uFld[i+1, j]
         -dyG[i, j]*uFld[i, j]
         -dxG[i, j+1]*vFld[i, j+1]
         +dxG[i, j]*vFld[i, j]
         )*recip_rA[i, j]*recip_deepFacC[k])
    return tension
