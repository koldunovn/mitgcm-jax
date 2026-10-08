"""pkg/mom_common/mom_calc_absvort3.F: absolute vorticity (MOM_CALC_ABSVORT3)."""

from mitjax.farray import loop_i, loop_j


def mom_calc_absvort3(k, vort3, omega3, *, cfg, grid, params):
    """MOM_CALC_ABSVORT3(bi,bj,k, vort3, omega3, myThid)   @63cdc0b pkg/mom_common/mom_calc_absvort3.F:3-50

    C     | S/R MOM_CALC_ABSVORT3

    Returns omega3 = fCoriG*useCoriolisFac + vort3*nonLinFac. momAdvection and useCoriolis (PARAMS.h LOGICAL) are
    static; `1.` and `0.` are REAL*4 literals (exact).
    """
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    fCoriG = grid.fCoriG

    if params.momAdvection:                                         # :30-34
        nonLinFac = 1.
    else:
        nonLinFac = 0.
    if params.useCoriolis:                                          # :35-39
        useCoriolisFac = 1.
    else:
        useCoriolisFac = 0.

    j = loop_j(1-OLy, sNy+OLy)                                      # :41-47
    i = loop_i(1-OLx, sNx+OLx)
    omega3 = omega3.at[i, j].set(
        fCoriG[i, j]*useCoriolisFac
        + vort3[i, j]*nonLinFac)
    return omega3
