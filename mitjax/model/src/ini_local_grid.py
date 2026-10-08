"""INI_LOCAL_GRID: model/src/ini_local_grid.F @63cdc0b."""

import jax.numpy as jnp
import numpy as np

from mitjax.farray import loop_i, loop_j
from mitjax.model.grid import local, n_tiles

# eesupp/src/ini_procs.F:75-76 (no MPI): myXGlobalLo = 1, myYGlobalLo = 1
myXGlobalLo = 1
myYGlobalLo = 1


def MOD(a, p):
    """Fortran MOD for integers: a - INT(a/p)*p (sign of a), elementwise on numpy integer arrays."""
    return np.fmod(a, p)


def ini_local_grid(delX, delY, *, cfg, params):
    """INI_LOCAL_GRID( xGloc, yGloc, delXloc, delYloc, gridNx, gridNy, bi, bj, myThid )
    @63cdc0b model/src/ini_local_grid.F:10-171

    C     | SUBROUTINE INI_LOCAL_GRID
    C     | o Initialise model tile-local horizontal grid
    C     | Set local grid-point location (xGloc & yGloc) and
    C     |  local grid-point spacing (delXloc,delYloc) keeping the
    C     |  same units as grid-spacing input parameter (delX,delY
    C     |  and xgOrigin,ygOrigin).
    C !OUTPUT: xGloc, yGloc   :: mesh corner-point location (local "Long" real array type)
    C         delXloc,delYloc :: mesh spacing in X / Y direction
    C         gridNx, gridNy  :: mesh total grid-point number in X / Y direction

    `delX`, `delY`: SET_GRID.h arrays (load_grid_spacing). Returns (xGloc, yGloc, delXloc, delYloc, gridNx, gridNy)
    for all tiles at once (the caller's bi,bj loop is the tile axis): xGloc/yGloc declared
    (1-OLx:sNx+OLx+1, 1-OLy:sNy+OLy+1), delXloc (0-OLx:sNx+OLx), delYloc (0-OLy:sNy+OLy), each with the tile axis.

    The tile base indices iG0, jG0 differ per tile, so the scalar tile-corner coordinates xG0, yG0 are arrays
    [tile, 1, 1]; the loops `DO i=1,iG0` (:127-129) run up to the largest iG0, each tile adding only while
    i <= its own iG0 (the same additions in the same order as the Fortran for every tile). The MOD-indexed reads of
    delX/delY (:131-150) are gathers with per-tile integer indices (storage index = Fortran index - 1, delX is
    declared from 1). The grid-line recursions (:153-164) run i (resp. j) in the Fortran order; the other index is
    vectorised (its iterations are independent).

    #ifdef ALLOW_EXCH2 (:100-106, :115-118): gridNx = exch2_mydNx(1), iG0 = exch2_tBasex(W2_myTileList(bi,bj));
    else gridNx = Nx, iG0 = myXGlobalLo-1+(bi-1)*sNx (:103-105, :119-121).
    """
    sz = cfg.size
    sNx, sNy, OLx, OLy = sz.sNx, sz.sNy, sz.OLx, sz.OLy
    T = n_tiles(sz)

    if cfg.cpp.ALLOW_EXCH2:                                         # :100-102
        gridNx = params.exch2.exch2_mydNx[0]
        gridNy = params.exch2.exch2_mydNy[0]
    else:                                                           # :103-105
        gridNx = sz.Nx
        gridNy = sz.Ny

    # :115-122  tile base X and Y indices (tile order bi + (bj-1)*nSx)
    bi = np.arange(T) % sz.nSx + 1
    bj = np.arange(T) // sz.nSx + 1
    if cfg.cpp.ALLOW_EXCH2:
        tN = np.asarray(params.exch2.W2_myTileList)
        iG0 = np.asarray(params.exch2.exch2_tBasex)[tN - 1]
        jG0 = np.asarray(params.exch2.exch2_tBasey)[tN - 1]
    else:
        iG0 = myXGlobalLo - 1 + (bi-1)*sNx
        jG0 = myYGlobalLo - 1 + (bj-1)*sNy
    iG0 = iG0.reshape(T, 1, 1)
    jG0 = jG0.reshape(T, 1, 1)

    # :124-129  coordinate of the tile corner: outer grid-line of the "real" tile
    xG0 = jnp.full((T, 1, 1), params.xgOrigin, dtype=jnp.float64)  # xG0 = xgOrigin
    for i in range(1, int(iG0.max()) + 1):
        xG0 = jnp.where(i <= iG0, xG0 + delX[i], xG0)
    # :130-133  back-step to the outer grid-line of the halo
    for i in range(1, OLx + 1):
        xG0 = xG0 - delX.data[(1+MOD(iG0-i+OLx*gridNx, gridNx)) - 1]
    # :134-138
    yG0 = jnp.full((T, 1, 1), params.ygOrigin, dtype=jnp.float64)  # yG0 = ygOrigin
    for j in range(1, int(jG0.max()) + 1):
        yG0 = jnp.where(j <= jG0, yG0 + delY[j], yG0)
    # :139-142
    for j in range(1, OLy + 1):
        yG0 = yG0 - delY.data[(1+MOD(jG0-j+OLy*gridNy, gridNy)) - 1]

    # :144-150  local copy of the tile grid spacing
    delXloc = local("delXloc", sz, i=(0-OLx, sNx+OLx))
    delYloc = local("delYloc", sz, j=(0-OLy, sNy+OLy))
    i = loop_i(0-OLx, sNx+OLx)
    i_val = np.arange(0-OLx, sNx+OLx+1).reshape(1, 1, -1)          # the values of i, [1, 1, ni]
    delXloc = delXloc.at[i].set(delX.data[(1+MOD(iG0+i_val-1+OLx*gridNx, gridNx)) - 1])
    j = loop_j(0-OLy, sNy+OLy)
    j_val = np.arange(0-OLy, sNy+OLy+1).reshape(1, -1, 1)          # the values of j, [1, nj, 1]
    delYloc = delYloc.at[j].set(delY.data[(1+MOD(jG0+j_val-1+OLy*gridNy, gridNy)) - 1])

    # :152-164  coordinates of cell corners for N+1 grid-lines
    xGloc = local("xGloc", sz, i=(1-OLx, sNx+OLx+1), j=(1-OLy, sNy+OLy+1))
    yGloc = local("yGloc", sz, i=(1-OLx, sNx+OLx+1), j=(1-OLy, sNy+OLy+1))
    j = loop_j(1-OLy, sNy+OLy+1)                                    # DO j=1-OLy,sNy+OLy +1
    xGloc = xGloc.at[1-OLx, j].set(xG0)                             #  xGloc(1-OLx,j) = xG0
    for ii in range(1-OLx, sNx+OLx+1):                              #  DO i=1-OLx,sNx+OLx
        i = loop_i(ii, ii)
        xGloc = xGloc.at[i+1, j].set(xGloc[i, j] + delXloc[i])
    i = loop_i(1-OLx, sNx+OLx+1)                                    # DO i=1-OLx,sNx+OLx +1
    yGloc = yGloc.at[i, 1-OLy].set(yG0)                             #  yGloc(i,1-OLy) = yG0
    for jj in range(1-OLy, sNy+OLy+1):                              #  DO j=1-OLy,sNy+OLy
        j = loop_j(jj, jj)
        yGloc = yGloc.at[i, j+1].set(yGloc[i, j] + delYloc[j])

    return xGloc, yGloc, delXloc, delYloc, gridNx, gridNy
