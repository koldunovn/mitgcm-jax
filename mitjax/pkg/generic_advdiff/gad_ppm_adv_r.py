"""pkg/generic_advdiff/gad_ppm_adv_r.F: PPM advective flux in R (GAD_PPM_ADV_R)."""

import jax.numpy as jnp

from mitjax.farray import FArray, loops_kji
from mitjax.pkg.generic_advdiff.gad_osc_hat_r import gad_osc_hat_r
from mitjax.pkg.generic_advdiff.gad_ppm_flx_r import gad_ppm_flx_r
from mitjax.pkg.generic_advdiff.gad_ppm_hat_r import gad_ppm_hat_r
from mitjax.pkg.generic_advdiff.gad_h import ENUM_PPM_WENO_LIMIT
from mitjax.pkg.generic_advdiff.gad_ppm_p3e_r import gad_ppm_p3e_r


def column_local(like, name, k, *, cfg):
    """A column local `_RL name(k[0]:k[1])` of the Fortran, for every column (ix,iy) at once: declared
    (1-OLx:sNx+OLx, 1-OLy:sNy+OLy, k[0]:k[1]) with the tiles of `like`, every point NaN (a read of a point the
    Fortran never wrote shows up)."""
    sNx, sNy, OLx, OLy = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy
    shape = (like.ntiles, k[1] - k[0] + 1, sNy + 2*OLy, sNx + 2*OLx)
    return FArray(jnp.full(shape, jnp.nan, like.dtype), name, i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy), k=k)


def gad_ppm_adv_r(meth, delT, velR, facR, fbar, flux, *, cfg, grid):
    """GAD_PPM_ADV_R(meth,bi,bj, delT,velR,facR,fbar, flux,myThid)   @63cdc0b pkg/generic_advdiff/gad_ppm_adv_r.F:3-161

    C     | PPM_ADV_R: evaluate grid-cell advective flux in R.             |
    C     | Lagrangian-type Piecewise Parabolic Method (PPM).              |
    C       meth     :: advection method.
    C       delT     :: level-wise time-steps.
    C       velR     :: vel. field in r-direction.
    C       facR     :: grid-areas in r-direction.
    C       fbar     :: grid-cell values.
    C       flux     :: trac.-flux in r-direction.

    Returns flux (every point is written). `meth` is static; delT is the traced `delT(1:Nr)` (an FArray, not
    tiled). The column loops `do iy / do ix` (:94-156) are vectorised: every column is independent (reads its own
    column of the inputs, writes its own column of flux), so the column locals floc, mloc, edge, fhat, ohat carry the
    column indices and the column routines get the column loops ix, iy (k-vectorised nests, `loops_kji`). The sum
    vsum (:97-102) runs in the Fortran order. The IF on vsum (:104-153) is a `where` per column between the flux of
    GAD_PPM_FLX_R and zero; the local copies (:106-119) are made for every column (columns with vsum = 0 never read
    them). The IF on meth (:122) is static.
    """
    sNx, sNy, OLx, OLy, Nr = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy, cfg.Nr
    maskC = grid.maskC

    floc = column_local(fbar, "floc", (1-3, Nr+3), cfg=cfg)             # :59-63
    mloc = column_local(fbar, "mloc", (1-3, Nr+3), cfg=cfg)
    fhat = {ii: column_local(fbar, f"fhat{ii}", (1-0, Nr+0), cfg=cfg) for ii in (1, 2, 3)}
    edge = column_local(fbar, "edge", (1-0, Nr+1), cfg=cfg)

    ir, iy, ix = loops_kji((1, Nr), (1-OLy+0, sNy+OLy-0), (1-OLx+0, sNx+OLx-0))

#     ======================================= mask boundary conditions
    mloc = mloc.at[ix, iy, -2].set(0.)                                  # :67-72
    mloc = mloc.at[ix, iy, -1].set(0.)
    mloc = mloc.at[ix, iy, +0].set(0.)
    mloc = mloc.at[ix, iy, Nr+1].set(0.)
    mloc = mloc.at[ix, iy, Nr+2].set(0.)
    mloc = mloc.at[ix, iy, Nr+3].set(0.)

#     ======================================= no flux through surf. bc
    flux = flux.at[ix, iy, +1].set(0.)                                  # :86-91

#     ==================== calculate transport for interior grid-cells    (:94-156)
    vsum = 0.0                                                          # :97-102
    for kr in range(2, Nr+1):
#     ================================== quick break on zero transport
        vsum = (vsum
                + jnp.abs(velR[ix, iy, kr]))

#     ================================== make local unit-stride copies
    floc = floc.at[ix, iy, ir].set(fbar[ix, iy, ir])                    # :106-111
    mloc = mloc.at[ix, iy, ir].set(maskC[ix, iy, ir])

#     ================================== make mask boundary conditions
    floc = floc.at[ix, iy, -2].set(floc[ix, iy, +1])                    # :114-119
    floc = floc.at[ix, iy, -1].set(floc[ix, iy, +1])
    floc = floc.at[ix, iy, +0].set(floc[ix, iy, +1])
    floc = floc.at[ix, iy, Nr+1].set(floc[ix, iy, Nr])
    floc = floc.at[ix, iy, Nr+2].set(floc[ix, iy, Nr])
    floc = floc.at[ix, iy, Nr+3].set(floc[ix, iy, Nr])

#     ==================== reconstruct derivatives for WENO indicators
    ohat = None
    if meth == ENUM_PPM_WENO_LIMIT:                                     # :122-126
        ohat = {1: column_local(fbar, "ohat1", (1-3, Nr+3), cfg=cfg),
                2: column_local(fbar, "ohat2", (1-3, Nr+3), cfg=cfg)}
        ohat = gad_osc_hat_r(ix, iy,
                             mloc, floc,
                             ohat, cfg=cfg)

#     ==================== reconstruct 3rd--order accurate edge values
    edge = gad_ppm_p3e_r(ix, iy,                                        # :129-131
                         mloc, floc,
                         edge, cfg=cfg)

#     ==================== reconstruct coeff. for grid-cell poynomials
    fhat = gad_ppm_hat_r(ix, iy,                                        # :134-138
                         meth,
                         mloc, floc,
                         edge, ohat,
                         fhat, cfg=cfg)

#     ==================== evaluate integral fluxes on grid-cell edges
    flux_flx = gad_ppm_flx_r(ix, iy,                                    # :141-144
                             delT, velR,
                             facR, fhat,
                             flux, cfg=cfg, grid=grid)

#     ================================== "null" flux on zero transport
    ir, iy, ix = loops_kji((2, Nr), (1-OLy+0, sNy+OLy-0), (1-OLx+0, sNx+OLx-0))   # :104, :148-151
    flux = flux.at[ix, iy, ir].set(jnp.where(vsum > 0., flux_flx[ix, iy, ir], 0.))
    return flux
