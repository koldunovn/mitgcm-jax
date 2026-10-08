"""INI_CARTESIAN_GRID: model/src/ini_cartesian_grid.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j
from mitjax.model.src.ini_local_grid import ini_local_grid


def ini_cartesian_grid(grid, delX, delY, *, cfg, params):
    """INI_CARTESIAN_GRID( myThid )   @63cdc0b model/src/ini_cartesian_grid.F:6-155

    C     | SUBROUTINE INI_CARTESIAN_GRID
    C     | o Initialise model coordinate system
    C     | The grid arrays, initialised here, are used throughout
    C     | the code in evaluating gradients, integrals and spatial
    C     | avarages. This routine is called separately by each
    C     | thread and initialises only the region of the domain
    C     | it is "responsible" for.
    C     | Under the cartesian grid mode primitive distances
    C     | in X and Y are in metres. Distance in Z are in m or Pa
    C     | depending on the vertical gridding mode.

    `delX`, `delY`: SET_GRID.h arrays (read by INI_LOCAL_GRID). Returns `grid` with xG, yG, xC, yC, dxF, dyF, dxG,
    dyG, dxC, dyC, dxV, dyU, rA, rAw, rAs, rAz. Every loop is pointwise (each point reads only INI_LOCAL_GRID's
    output or arrays written by an earlier loop), so each runs as one vectorised statement over its (i,j) range;
    points outside a loop's range (`NOTE range`, :110, :116, :123-124) keep the zero INI_GRID wrote. All tiles at
    once (the bi,bj loop is the tile axis). `0.25 _d 0` and `0.5 _d 0` are double literals.
    """
    sz = cfg.size
    sNx, sNy, OLx, OLy = sz.sNx, sz.sNy, sz.OLx, sz.OLy

    xGloc, yGloc, delXloc, delYloc, gridNx, gridNy = ini_local_grid(delX, delY, cfg=cfg, params=params)  # :63-67

    j = loop_j(1-OLy, sNy+OLy)                                      # :70-75
    i = loop_i(1-OLx, sNx+OLx)
    xG = grid.xG.at[i, j].set(xGloc[i, j])
    yG = grid.yG.at[i, j].set(yGloc[i, j])

    xC = grid.xC.at[i, j].set(0.25*(                                # :78-86
        xGloc[i, j]+xGloc[i+1, j]+xGloc[i, j+1]+xGloc[i+1, j+1]))
    yC = grid.yC.at[i, j].set(0.25*(
        yGloc[i, j]+yGloc[i+1, j]+yGloc[i, j+1]+yGloc[i+1, j+1]))

    dxF = grid.dxF.at[i, j].set(delXloc[i])                         # :89-94
    dyF = grid.dyF.at[i, j].set(delYloc[j])

    dxG = grid.dxG.at[i, j].set(delXloc[i])                         # :97-102
    dyG = grid.dyG.at[i, j].set(delYloc[j])

    j = loop_j(1-OLy, sNy+OLy)                                      # :109-113
    i = loop_i(1-OLx+1, sNx+OLx)                                    # NOTE range
    dxC = grid.dxC.at[i, j].set(0.5*(dxF[i, j]+dxF[i-1, j]))

    j = loop_j(1-OLy+1, sNy+OLy)                                    # :116-120  NOTE range
    i = loop_i(1-OLx, sNx+OLx)
    dyC = grid.dyC.at[i, j].set(0.5*(dyF[i, j]+dyF[i, j-1]))

    j = loop_j(1-OLy+1, sNy+OLy)                                    # :123-132  NOTE range
    i = loop_i(1-OLx+1, sNx+OLx)                                    # NOTE range
    dxV = grid.dxV.at[i, j].set(0.5*(dxG[i, j]+dxG[i-1, j]))       # by averaging (method I)
    dyU = grid.dyU.at[i, j].set(0.5*(dyG[i, j]+dyG[i, j-1]))

    j = loop_j(1-OLy, sNy+OLy)                                      # :135-148
    i = loop_i(1-OLx, sNx+OLx)
    rA = grid.rA.at[i, j].set(dxF[i, j]*dyF[i, j])
    rAw = grid.rAw.at[i, j].set(dxC[i, j]*dyG[i, j])
    rAs = grid.rAs.at[i, j].set(dxG[i, j]*dyC[i, j])
    rAz = grid.rAz.at[i, j].set(dxV[i, j]*dyU[i, j])

    return grid.replace(xG=xG, yG=yG, xC=xC, yC=yC, dxF=dxF, dyF=dyF, dxG=dxG, dyG=dyG, dxC=dxC, dyC=dyC,
                        dxV=dxV, dyU=dyU, rA=rA, rAw=rAw, rAs=rAs, rAz=rAz)
