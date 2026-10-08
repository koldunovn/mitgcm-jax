"""CALC_GRID_ANGLES: model/src/calc_grid_angles.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.model.grid import deg2rad
from mitjax.ops.safe import safe_div, safe_sqrt

halfRL = 0.5           # EEPARAMS.h  PARAMETER ( halfRL = 0.5D0 )


def calc_grid_angles(grid, skipCalcAngleC, *, cfg, params):
    """CALC_GRID_ANGLES( skipCalcAngleC, myThid )   @63cdc0b model/src/calc_grid_angles.F:7-133

    C     | o calculate the angle between geographical north and model grid
    C     |   north, assuming that yG holds the geographical coordinates

    Returns `grid` with u2zonDir, v2zonDir and (unless skipCalcAngleC) angleCosC, angleSinC. Every loop is pointwise
    over its (i,j) range and runs as one vectorised statement; points outside a range keep their prior values (the
    INI_GRID initial 1./0.). uPseudo, vPseudo are local arrays (NaN where the Fortran never writes them; never read
    there). Divisions under `IF ( x.GT.0. )` / `IF ( uNorm.NE.0. )` are guarded before dividing (mitjax/ops/safe.py);
    SQRT of a sum of squares through safe_sqrt (its value at +0 is +0, as SQRT's). Association as written:
    `-(a - b)*deg2rad/dyG` is -(((a-b)*deg2rad)/dyG) (negation exact), `rSphere*(s1 - s0)*dxC/tmpVal` is
    ((rSphere*(s1-s0))*dxC)/tmpVal, `COS(deg2rad*(y0 + y1)*halfRL)` is COS((deg2rad*(y0+y1))*halfRL). `0.5` is a
    REAL*4 literal (exact). COS, SIN: XLA:CPU's are glibc 2.28's bitwise on the measured ranges (mitjax/ops/libm.py).
    """
    sz = cfg.size
    sNx, sNy, OLx, OLy = sz.sNx, sz.sNy, sz.OLx, sz.OLy
    rSphere = params.rSphere
    yG, dxG, dyG, dxC, dyC, rAw, rAs = grid.yG, grid.dxG, grid.dyG, grid.dxC, grid.dyC, grid.rAw, grid.rAs
    uPseudo = yG.local("uPseudo")
    vPseudo = yG.local("vPseudo")
    u2zonDir, v2zonDir = grid.u2zonDir, grid.v2zonDir
    angleCosC, angleSinC = grid.angleCosC, grid.angleSinC

    # -    compute pseudo velocities from stream function psi = -yG*deg2rad, that is, zonal flow (:51-74)
    j = loop_j(1-OLy, sNy+OLy-1)
    i = loop_i(1-OLx, sNx+OLx)
    m = dyG[i, j] > 0.
    uPseudo = uPseudo.at[i, j].set(jnp.where(m, -safe_div((yG[i, j] - yG[i, j+1])*deg2rad, dyG[i, j], m), 0.))
    u2zonDir = u2zonDir.at[i, j].set(rSphere*uPseudo[i, j])
    j = loop_j(1-OLy, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx-1)
    m = dxG[i, j] > 0.
    vPseudo = vPseudo.at[i, j].set(jnp.where(m, safe_div((yG[i, j] - yG[i+1, j])*deg2rad, dxG[i, j], m), 0.))
    v2zonDir = v2zonDir.at[i, j].set(rSphere*vPseudo[i, j])
    if not skipCalcAngleC:                                                    # :75-86
        j = loop_j(1-OLy, sNy+OLy-1)
        i = loop_i(1-OLx, sNx+OLx-1)
        uC = 0.5*(uPseudo[i, j] + uPseudo[i+1, j])
        vC = 0.5*(vPseudo[i, j] + vPseudo[i, j+1])
        x = uC*uC + vC*vC
        uNorm = safe_sqrt(x, x > 0., fill=0.)                                 # SQRT(uC*uC+vC*vC)
        uNorm = safe_div(1., uNorm, uNorm != 0., fill=uNorm)                  # IF (uNorm.NE.0.) uNorm = 1./uNorm
        angleCosC = angleCosC.at[i, j].set(uC*uNorm)
        angleSinC = angleSinC.at[i, j].set(-vC*uNorm)

    # -   alternative definition of grid-angles cosine (@ U pt) & sine (@ V pt) (:91-126)
    j = loop_j(1-OLy, sNy+OLy-1)
    i = loop_i(1-OLx, sNx+OLx)
    tmpVal = rAw[i, j]*jnp.cos(deg2rad*(yG[i, j] + yG[i, j+1])*halfRL)
    m = tmpVal > 0.
    u2zonDir = u2zonDir.at[i, j].set(jnp.where(
        m, safe_div(rSphere*(jnp.sin(yG[i, j+1]*deg2rad) - jnp.sin(yG[i, j]*deg2rad))*dxC[i, j], tmpVal, m), 1.))
    j = loop_j(1-OLy, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx-1)
    tmpVal = rAs[i, j]*jnp.cos(deg2rad*(yG[i, j] + yG[i+1, j])*halfRL)
    m = tmpVal > 0.
    v2zonDir = v2zonDir.at[i, j].set(jnp.where(
        m, -safe_div(rSphere*(jnp.sin(yG[i+1, j]*deg2rad) - jnp.sin(yG[i, j]*deg2rad))*dyC[i, j], tmpVal, m), 0.))
    return grid.replace(u2zonDir=u2zonDir, v2zonDir=v2zonDir, angleCosC=angleCosC, angleSinC=angleSinC)
