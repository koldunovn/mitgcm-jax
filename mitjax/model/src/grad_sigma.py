"""GRAD_SIGMA: model/src/grad_sigma.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j


def grad_sigma(iMin, iMax, jMin, jMax, k, rhoK, sigKm1, sigKp1, sigmaX, sigmaY, sigmaR, *, cfg, grid, params):
    """GRAD_SIGMA( bi, bj, iMin, iMax, jMin, jMax, k, rhoK, sigKm1, sigKp1, sigmaX, sigmaY, sigmaR, myThid )
    @63cdc0b model/src/grad_sigma.F:6-101

    C     | SUBROUTINE GRAD_SIGMA
    C     | o Calculate isoneutral gradients
    C     iMin,iMax  :: not used
    C     jMin,jMax  :: not used
    C     k          :: current level index
    C     rhoK       :: density at level k
    C     sigKm1     :: upper level density computed at current pressure
    C     sigKp1     :: lower level density computed at current pressure
    C     sigmaX,Y,R :: iso-neutral gradient of density in 3 directions X,Y,R

    Returns (sigmaX, sigmaY, sigmaR) (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr) with level k written; rhoK, sigKm1, sigKp1
    2-D. `k` a Python int (the caller's DO k), so the branch on k (:84) is static. The macros `_maskW`, `_maskS`,
    `_recip_dxC`, `_recip_dyC` are the plain GRID.h fields (ALLOW_DEPTH_CONTROL raises). Lane B (Task 25,
    global_ocean.cs32x15): useCubedSphereExchange, FILL_CS_CORNER_TR_RL( 1 / 2, .FALSE., rhoLoc ) before the X / Y
    gradients (:59-62, :72-75; the second fill acts on the first one's result, as the Fortran's in-place calls), the
    corner flags of every tile from the grid's W2 tile view (mitjax/eesupp/fill_cs_corner_tr_rl.py, lane B). The
    (i,j) nests are independent point by point. `0. _d 0` exact."""
    if cfg.cpp.ALLOW_DEPTH_CONTROL:
        raise NotImplementedError("GRAD_SIGMA: ALLOW_DEPTH_CONTROL (masked _maskW/_maskS) is not ported")
    sz = cfg.size
    g = grid
    OLx, OLy, sNx, sNy = sz.OLx, sz.OLy, sz.sNx, sz.sNy
    jA = loop_j(1-OLy, sNy+OLy)
    iA = loop_i(1-OLx, sNx+OLx)
    rhoLoc = rhoK.local("rhoLoc").at[iA, jA].set(rhoK[iA, jA])                    # :52-56
    if params.useCubedSphereExchange:                                           # :59-62 (lane B)
        rhoLoc = _fill_cs(1, rhoLoc, sz, g)

    i = loop_i(1-OLx+1, sNx+OLx)                                                # :63-69
    sigmaX = sigmaX.at[i, jA, k].set(g.maskW[i, jA, k]
                                     * g.recip_dxC[i, jA]*g.recip_deepFacC[k]
                                     * (rhoLoc[i, jA]-rhoLoc[i-1, jA]))

    if params.useCubedSphereExchange:                                           # :72-75 (lane B)
        rhoLoc = _fill_cs(2, rhoLoc, sz, g)
    j = loop_j(1-OLy+1, sNy+OLy)                                                # :76-82
    sigmaY = sigmaY.at[iA, j, k].set(g.maskS[iA, j, k]
                                     * g.recip_dyC[iA, j]*g.recip_deepFacC[k]
                                     * (rhoLoc[iA, j]-rhoLoc[iA, j-1]))

    if k == 1:                                                                  # :84-89
        sigmaR = sigmaR.at[iA, jA, k].set(0.)
    else:                                                                       # :90-98
        sigmaR = sigmaR.at[iA, jA, k].set(g.maskC[iA, jA, k]*g.maskC[iA, jA, k-1]
                                          * g.recip_drC[k]*g.rkSign
                                          * (sigKp1[iA, jA]-sigKm1[iA, jA]))
    return sigmaX, sigmaY, sigmaR


def _fill_cs(fill4dir, fld, sz, g):
    """CALL FILL_CS_CORNER_TR_RL( fill4dir, .FALSE., rhoLoc, bi, bj, myThid ) on every tile (lane B): the corner
    flags SW, SE, NW, NE of fill_cs_corner_tr_rl.F:75-82 from the grid's W2 tile view (exch2_is*edge)."""
    import jax.numpy as jnp
    from mitjax.eesupp.fill_cs_corner_tr_rl import fill_cs_corner_tr_rl
    from mitjax.farray import FArray
    W, S = g.exch2_isWedge.data[:, 0, 0] == 1, g.exch2_isSedge.data[:, 0, 0] == 1
    E, N = g.exch2_isEedge.data[:, 0, 0] == 1, g.exch2_isNedge.data[:, 0, 0] == 1
    corners = jnp.stack([W & S, E & S, W & N, E & N], axis=-1)
    new = fill_cs_corner_tr_rl(fill4dir, False, fld.data, corners, True, sNx=sz.sNx, sNy=sz.sNy, OLx=sz.OLx,
                               OLy=sz.OLy)
    return FArray(new, fld.name, tiled=fld.tiled, _dims=fld.dims)
