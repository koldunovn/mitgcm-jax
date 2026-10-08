"""INI_LINEAR_PHISURF: model/src/ini_linear_phisurf.F @63cdc0b (the Bo_surf / recip_Bo part)."""

from mitjax.farray import loop_i, loop_j
from mitjax.model.grid import declare


def ini_linear_phisurf(grid, *, cfg, params):
    """INI_LINEAR_PHISURF( myThid )   @63cdc0b model/src/ini_linear_phisurf.F:7-243

    C     | SUBROUTINE INI_LINEAR_PHISURF
    C     | o Initialise the Linear Relation Phi_surf(eta)
    C     | Bo_surf = Buoyancy|1/rho [ocean|atmos] at surface level [=g|alpha(p_o)]

    Returns the grid with Bo_surf, recip_Bo (SURFACE.h:24-25) set. Ported: the usingZCoords branch (:78-89:
    Bo_surf = gBaro, recip_Bo = 1. _d 0 / gBaro on every point) and uniformLin_PhiSurf in p coordinates (:90-102:
    recip_rhoConst, rhoConst; lane B). The other branches (fluidIsWater in p coordinates, fluidIsAir, :103-183)
    raise; the WRITE_FLD_XY_RL of Bo_surf without uniformLin_PhiSurf (:191-193) is output only. phi0surf (:200-213: zero, geoPotAnomFile, EXCH_XY_RS) is a forcing
    field that EXTERNAL_FORCING_SURF rewrites every step: it lives with FFIELDS (forcing lane); geoPotAnomFile raises.
    Eager, host side (INITIALISE_FIXED, initialise_fixed.F:226)."""
    sz = cfg.size
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)                                         # :82
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)                                         # :83
    if params.usingZCoords:                                                     # :78-89
        Bo_surf = declare("Bo_surf", sz).at[i, j].set(params.gBaro)            # :84
        recip_Bo = declare("recip_Bo", sz).at[i, j].set(1.0 / params.gBaro)    # :85  1. _d 0 / gBaro
    elif params.uniformLin_PhiSurf:                                             # :90-102 (lane B, Task 25)
        Bo_surf = declare("Bo_surf", sz).at[i, j].set(params.recip_rhoConst)   # :98
        recip_Bo = declare("recip_Bo", sz).at[i, j].set(params.rhoConst)       # :99
    else:                                                                       # :103-183
        raise NotImplementedError("INI_LINEAR_PHISURF: the fluidIsWater (p coordinates) and fluidIsAir branches "
                                  "(:103-183) are not ported")
    return grid.replace(Bo_surf=Bo_surf, recip_Bo=recip_Bo)
