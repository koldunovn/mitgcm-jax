"""ADD_WALLS2MASKS: model/src/add_walls2masks.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j, loops_kji


def _blank(s):
    return str(s).strip() == ""


def add_walls2masks(rEmpty, hFacW, hFacS, rLowW, rLowS, rSurfW, rSurfS, kSurfW, kSurfS, maskInW, maskInS, *,
                    cfg, grid, params):
    """ADD_WALLS2MASKS( rEmpty, myThid )   @63cdc0b model/src/add_walls2masks.F:7-140

    C     | SUBROUTINE ADD_WALLS2MASKS
    C     | o Apply additional closing of Western and Southern edges
    C     |   grid-cell open-water factor
    C     | Reset to zero hFacW and/or hFacS grid factors at some
    C     | specific locations. In particular, allow to prevent fluid
    C     | transport (at any detph) between 2 adjacent vertical
    C     | column by adding a "thin wall" between the 2.
    C     | Location of "thin-wall" is reccorded as kSurfW/S = Nr+2
    C     rEmpty  :: empty column r-position

    The GRID.h arrays it updates (hFacW, hFacS, rLowW, rLowS, rSurfW, rSurfS, kSurfW, kSurfS, maskInW, maskInS) are
    arguments and are returned in that order; it reads dxG, dyG from `grid`.

    Ported: the closing of edges with dyG = 0 (W) or dxG = 0 (S) (:82-107, e.g. the halo rows beyond the poles of
    a lat-lon grid, where INI_SPHERICAL_POLAR_GRID set dxG = 0). The W and S blocks of the point loop write
    different arrays, so each runs as one vectorised `where` over all points (all k at once for hFacW/S: the k loop
    sets every level to the same value). `addWwallFile` / `addSwallFile` (:52-73, :112-139) are blank in every M1
    variant: raise.
    """
    if not _blank(params.addWwallFile) or not _blank(params.addSwallFile):         # :52-73, :112-139
        raise NotImplementedError("ADD_WALLS2MASKS: addWwallFile / addSwallFile are not ported")
    sz = cfg.size
    sNx, sNy, OLx, OLy, Nr = sz.sNx, sz.sNy, sz.OLx, sz.OLy, sz.Nr
    dxG, dyG = grid.dxG, grid.dyG
    zeroRS = 0.0

    j = loop_j(1-OLy, sNy+OLy)                                      # :82-107
    i = loop_i(1-OLx, sNx+OLx)
    wallW = dyG[i, j] == zeroRS                                     # IF ( dyG(i,j,bi,bj).EQ.zeroRS )
    wallS = dxG[i, j] == zeroRS                                     # IF ( dxG(i,j,bi,bj).EQ.zeroRS )
    kk, jk, ik = loops_kji((1, Nr), (1-OLy, sNy+OLy), (1-OLx, sNx+OLx))
    hFacW = hFacW.at[ik, jk, kk].set(jnp.where(dyG[ik, jk] == zeroRS, zeroRS, hFacW[ik, jk, kk]))
    rLowW = rLowW.at[i, j].set(jnp.where(wallW, rEmpty, rLowW[i, j]))
    rSurfW = rSurfW.at[i, j].set(jnp.where(wallW, rEmpty, rSurfW[i, j]))
    kSurfW = kSurfW.at[i, j].set(jnp.where(wallW, Nr+2, kSurfW[i, j]))
    maskInW = maskInW.at[i, j].set(jnp.where(wallW, zeroRS, maskInW[i, j]))
    hFacS = hFacS.at[ik, jk, kk].set(jnp.where(dxG[ik, jk] == zeroRS, zeroRS, hFacS[ik, jk, kk]))
    rLowS = rLowS.at[i, j].set(jnp.where(wallS, rEmpty, rLowS[i, j]))
    rSurfS = rSurfS.at[i, j].set(jnp.where(wallS, rEmpty, rSurfS[i, j]))
    kSurfS = kSurfS.at[i, j].set(jnp.where(wallS, Nr+2, kSurfS[i, j]))
    maskInS = maskInS.at[i, j].set(jnp.where(wallS, zeroRS, maskInS[i, j]))
    return hFacW, hFacS, rLowW, rLowS, rSurfW, rSurfS, kSurfW, kSurfS, maskInW, maskInS
