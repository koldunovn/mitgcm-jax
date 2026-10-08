"""INI_SPHERICAL_POLAR_GRID: model/src/ini_spherical_polar_grid.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.model.grid import deg2rad
from mitjax.model.src.ini_local_grid import ini_local_grid


def ini_spherical_polar_grid(grid, delX, delY, *, cfg, params):
    """INI_SPHERICAL_POLAR_GRID( myThid )   @63cdc0b model/src/ini_spherical_polar_grid.F:8-284

    C     | SUBROUTINE INI_SPHERICAL_POLAR_GRID
    C     | o Initialise model coordinate system arrays
    C     | These arrays are used throughout the code in evaluating
    C     | gradients, integrals and spatial avarages. This routine
    C     | is called separately by each thread and initialise only
    C     | the region of the domain it is "responsible" for.
    C     | Under the spherical polar grid mode primitive distances
    C     | in X and Y are in degrees. Distance in Z are in m or Pa
    C     | depending on the vertical gridding mode.

    `delX`, `delY`: SET_GRID.h arrays (read by INI_LOCAL_GRID). Returns `grid` with xG, yG, xC, yC, dxF, dyF, dxG,
    dyG, dxC, dyC, dxV, dyU, rA, rAw, rAs, rAz, tanPhiAtU, tanPhiAtV, cosFacU, cosFacV, sqCosFacU, sqCosFacV.

    The scalar temporaries `lat`, `dlat`, `dlon` of each point loop become arrays over the loop's (i,j) range; every
    loop is pointwise, so it runs as one vectorised statement (points outside a `NOTE range` loop keep the zero
    INI_GRID wrote). Statement order and association as in the Fortran, e.g. dxF = ((rSphere*COS(lat*deg2rad))*dlon)
    *deg2rad. COS, SIN, TAN: XLA:CPU's are bitwise glibc 2.28 (mitjax/ops/libm.py, Task 7c). `#undef
    USE_BACKWARD_COMPATIBLE_GRID` (:3), so :189-194 and :219-221 are not compiled. `IF (cosPower.NE.0.)` (:257-263)
    is not executed by any M1 variant (cosPower = 0.): raises; `rotateGrid` (:276-281, ROTATE_SPHERICAL_POLAR_GRID,
    CALC_GRID_ANGLES) raises. Literals: `0.25 _d 0`, `0.5 _d 0` double; `1.`, `0.`, `90.` REAL*4, exact in binary.
    """
    sz = cfg.size
    sNx, sNy, OLx, OLy = sz.sNx, sz.sNy, sz.OLx, sz.OLy
    rSphere = params.rSphere

    xGloc, yGloc, delXloc, delYloc, gridNx, gridNy = ini_local_grid(delX, delY, cfg=cfg, params=params)  # :68-72

    j = loop_j(1-OLy, sNy+OLy)                                      # :75-80
    i = loop_i(1-OLx, sNx+OLx)
    xG = grid.xG.at[i, j].set(xGloc[i, j])
    yG = grid.yG.at[i, j].set(yGloc[i, j])

    xC = grid.xC.at[i, j].set(0.25*(                                # :83-91
        xGloc[i, j]+xGloc[i+1, j]+xGloc[i, j+1]+xGloc[i+1, j+1]))
    yC = grid.yC.at[i, j].set(0.25*(
        yGloc[i, j]+yGloc[i+1, j]+yGloc[i, j+1]+yGloc[i+1, j+1]))

    lat = yC[i, j]                                                  # :94-106  by formula
    dlon = delXloc[i]
    dlat = delYloc[j]
    dxF = grid.dxF.at[i, j].set(rSphere*jnp.cos(lat*deg2rad)*dlon*deg2rad)
    dyF = grid.dyF.at[i, j].set(rSphere*dlat*deg2rad)

    lat = 0.5*(yGloc[i, j]+yGloc[i+1, j])                           # :109-122  by formula
    dlon = delXloc[i]
    dlat = delYloc[j]
    dxG = grid.dxG.at[i, j].set(rSphere*jnp.cos(deg2rad*lat)*dlon*deg2rad)
    dxG = dxG.at[i, j].set(jnp.where(dxG[i, j] < 1., 0., dxG[i, j]))   # IF (dxG(i,j,bi,bj).LT.1.) dxG(i,j,bi,bj)=0.
    dyG = grid.dyG.at[i, j].set(rSphere*dlat*deg2rad)

    j = loop_j(1-OLy, sNy+OLy)                                      # :140-153
    i = loop_i(1-OLx+1, sNx+OLx)                                    # NOTE range
    dxC = grid.dxC.at[i, j].set(0.5*(dxF[i, j]+dxF[i-1, j]))       # by averaging

    j = loop_j(1-OLy+1, sNy+OLy)                                    # :156-167  NOTE range
    i = loop_i(1-OLx, sNx+OLx)
    dyC = grid.dyC.at[i, j].set(0.5*(dyF[i, j]+dyF[i, j-1]))       # by averaging

    j = loop_j(1-OLy+1, sNy+OLy)                                    # :170-179  NOTE range
    i = loop_i(1-OLx+1, sNx+OLx)                                    # NOTE range
    dxV = grid.dxV.at[i, j].set(0.5*(dxG[i, j]+dxG[i-1, j]))       # by averaging (method I)
    dyU = grid.dyU.at[i, j].set(0.5*(dyG[i, j]+dyG[i, j-1]))

    j = loop_j(1-OLy, sNy+OLy)                                      # :182-196  tracer cells
    i = loop_i(1-OLx, sNx+OLx)
    lat = 0.5*(yGloc[i, j]+yGloc[i+1, j])
    dlon = delXloc[i]
    dlat = delYloc[j]
    rA = grid.rA.at[i, j].set(rSphere*rSphere*dlon*deg2rad
                              * jnp.abs(jnp.sin((lat+dlat)*deg2rad)-jnp.sin(lat*deg2rad)))

    j = loop_j(1-OLy, sNy+OLy)                                      # :199-210  u cells
    i = loop_i(1-OLx+1, sNx+OLx)                                    # NOTE range
    rAw = grid.rAw.at[i, j].set(0.5*(rA[i, j]+rA[i-1, j]))         # by averaging

    j = loop_j(1-OLy, sNy+OLy)                                      # :213-226  v cells
    i = loop_i(1-OLx, sNx+OLx)
    lat = yC[i, j]
    dlon = delXloc[i]
    dlat = 0.5*(delYloc[j] + delYloc[j-1])
    rAs = grid.rAs.at[i, j].set(rSphere*rSphere*dlon*deg2rad
                                * jnp.abs(jnp.sin(lat*deg2rad)-jnp.sin((lat-dlat)*deg2rad)))
    rAs = rAs.at[i, j].set(jnp.where((jnp.abs(lat) > 90.) | (jnp.abs(lat-dlat) > 90.), 0., rAs[i, j]))

    j = loop_j(1-OLy, sNy+OLy)                                      # :229-239  vorticity points
    i = loop_i(1-OLx, sNx+OLx)
    lat = 0.5*(yGloc[i, j]+yGloc[i, j+1])
    dlon = 0.5*(delXloc[i] + delXloc[i-1])
    dlat = 0.5*(delYloc[j] + delYloc[j-1])
    rAz = grid.rAz.at[i, j].set(rSphere*rSphere*dlon*deg2rad
                                * jnp.abs(jnp.sin(lat*deg2rad)-jnp.sin((lat-dlat)*deg2rad)))
    rAz = rAz.at[i, j].set(jnp.where((jnp.abs(lat) > 90.) | (jnp.abs(lat-dlat) > 90.), 0., rAz[i, j]))

    j = loop_j(1-OLy, sNy+OLy)                                      # :242-252  trigonometric terms
    i = loop_i(1-OLx, sNx+OLx)
    lat = 0.5*(yGloc[i, j]+yGloc[i, j+1])
    tanPhiAtU = grid.tanPhiAtU.at[i, j].set(jnp.tan(lat*deg2rad))
    lat = 0.5*(yGloc[i, j]+yGloc[i+1, j])
    tanPhiAtV = grid.tanPhiAtV.at[i, j].set(jnp.tan(lat*deg2rad))

    j = loop_j(1-OLy, sNy+OLy)                                      # :255-270  Cosine(lat) scaling
    if params.cosPower != 0.:                                       # :257-263
        raise NotImplementedError("INI_SPHERICAL_POLAR_GRID: cosPower /= 0 is not ported")
    else:                                                           # :264-269
        cosFacU = grid.cosFacU.at[j].set(1.)
        cosFacV = grid.cosFacV.at[j].set(1.)
        sqCosFacU = grid.sqCosFacU.at[j].set(1.)
        sqCosFacV = grid.sqCosFacV.at[j].set(1.)

    if params.rotateGrid:                                           # :276-281
        raise NotImplementedError("INI_SPHERICAL_POLAR_GRID: rotateGrid (ROTATE_SPHERICAL_POLAR_GRID, "
                                  "CALC_GRID_ANGLES) is not ported")

    return grid.replace(xG=xG, yG=yG, xC=xC, yC=yC, dxF=dxF, dyF=dyF, dxG=dxG, dyG=dyG, dxC=dxC, dyC=dyC,
                        dxV=dxV, dyU=dyU, rA=rA, rAw=rAw, rAs=rAs, rAz=rAz, tanPhiAtU=tanPhiAtU,
                        tanPhiAtV=tanPhiAtV, cosFacU=cosFacU, cosFacV=cosFacV, sqCosFacU=sqCosFacU,
                        sqCosFacV=sqCosFacV)
