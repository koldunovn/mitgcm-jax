"""UPDATE_SURF_DR: model/src/update_surf_dr.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loops_kji
from mitjax.ops.safe import safe_div


def update_surf_dr(useLatest, myTime, myIter, *, cfg, grid, params, state=None):
    """UPDATE_SURF_DR( useLatest, myTime, myIter, myThid )   @63cdc0b model/src/update_surf_dr.F:6-145

    C     | SUBROUTINE UPDATE_SURF_DR
    C     | o Update the surface-level thickness fraction (hFacC,W,S)
    C     |   according to the surface r-position = Non-Linear FrSurf
    C     useLatest :: if true use hFac_surfC, else use hFac_surfNm1C
    C     myTime    :: Current time in simulation
    C     myIter    :: Current iteration number in simulation

    Called by FORWARD_STEP (forward_step.F:852, every step under NONLIN_FRSURF when select_rStar = 0 and
    selectSigmaCoord = 0; and :484 under doResetHFactors) and INITIALISE_VARIA (:321). Returns the Grid with hFacC
    and recip_hFacC (GRID.h) as the executed branch writes them.

    Ported: the ELSE branch :120-135 (nonlinFreeSurf <= 0: the linear free surface of a NONLIN_FRSURF build,
    executed by advect_xz/input and input.pqm, 200 calls each): hFacC = h0FacC and recip_hFacC = 1/h0FacC where
    h0FacC .NE. 0., else `0.` (REAL*4 zero, +0), every point of every level. `useLatest` and nonlinFreeSurf are static
    (a LOGICAL argument and an INTEGER namelist value select the branch). Lane B (Task 25, adjustment.cs
    input.nlfs): the nonlinFreeSurf > 0 branches :49-82 (useLatest: hFac_surf*) and :84-118 (hFac_surfNm1*) write
    hFacC/W/S and recip_hFacC/W/S of the surface level kSurfC/W/S from the State's hFac_surf fields (`state=`).

    Vectorisation: the DO k / DO j / DO i iterations of :122-133 are independent (each point reads only h0FacC at the
    same point): one k-vectorised nest; the IF is a `where` with the division guarded by its own condition."""
    if params.nonlinFreeSurf > 0:                                               # :49-118 (lane B, Task 25)
        return _nlfs(useLatest, cfg=cfg, grid=grid, state=state)
    sz = cfg.size
    g = grid
    k, j, i = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))     # :122-124
    hFacC = g.hFacC.at[i, j, k].set(g.h0FacC[i, j, k])                                      # :125
    wet = g.h0FacC[i, j, k] != 0.                                                           # :126
    recip_hFacC = g.recip_hFacC.at[i, j, k].set(
        jnp.where(wet, safe_div(1., g.h0FacC[i, j, k], wet), 0.))                           # :127, :129
    return g.replace(hFacC=hFacC, recip_hFacC=recip_hFacC)


def _nlfs(useLatest, *, cfg, grid, state):
    """update_surf_dr.F:49-118 (nonlinFreeSurf > 0): hFac(ks) = hFac_surf (useLatest) or hFac_surfNm1, recip_hFac(ks) =
    1. _d 0 / that, at the points with kSurf <= Nr: C on j = 1-OLy..sNy+OLy, i = 1-OLx..sNx+OLx; W on i = 2-OLx..; S on
    j = 2-OLy.. (:66-80 / :100-114); one level per point, so a scatter into level ks."""
    if state is None:
        raise ValueError("UPDATE_SURF_DR: nonlinFreeSurf > 0 needs the State (hFac_surf*)")
    sz = cfg.size
    Nr, OLx, OLy = sz.Nr, sz.OLx, sz.OLy
    suf = "" if useLatest else "Nm1"
    g = grid
    out = {}
    for pt, kname, j0, i0 in (("C", "kSurfC", 1, 1), ("W", "kSurfW", 1, 2), ("S", "kSurfS", 2, 1)):
        hs = jnp.asarray(getattr(state, f"hFac_surf{suf}{pt}").data)
        ks = jnp.asarray(getattr(g, kname).data)
        sel = jnp.zeros(ks.shape, bool).at[:, j0 - 1:, i0 - 1:].set(True) & (ks <= Nr)   # j >= j0-OLy, i >= i0-OLx
        h = getattr(g, f"hFac{pt}")
        r = getattr(g, f"recip_hFac{pt}")
        hd, rd = jnp.asarray(h.data), jnp.asarray(r.data)
        lev = jnp.arange(1, Nr + 1).reshape(1, Nr, 1, 1)
        at = sel[:, None] & (ks[:, None] == lev)
        hd = jnp.where(at, hs[:, None], hd)                                     # hFac(i,j,ks) = hFac_surf(i,j)
        rd = jnp.where(at, safe_div(1., hs, sel)[:, None], rd)                  # recip_hFac = 1. _d 0 / hFac_surf
        out[f"hFac{pt}"] = h.__class__(hd, h.name, tiled=h.tiled, _dims=h.dims)
        out[f"recip_hFac{pt}"] = r.__class__(rd, r.name, tiled=r.tiled, _dims=r.dims)
    return g.replace(**out)
