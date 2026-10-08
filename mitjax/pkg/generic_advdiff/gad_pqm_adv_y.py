"""pkg/generic_advdiff/gad_pqm_adv_y.F: PQM advective flux in Y (GAD_PQM_ADV_Y)."""

import jax.numpy as jnp

from mitjax.farray import loop_j, loop_i
from mitjax.pkg.generic_advdiff.gad_osc_hat_y import gad_osc_hat_y
from mitjax.pkg.generic_advdiff.gad_pqm_flx_y import gad_pqm_flx_y
from mitjax.pkg.generic_advdiff.gad_h import ENUM_PQM_WENO_LIMIT
from mitjax.pkg.generic_advdiff.gad_pqm_hat_y import gad_pqm_hat_y
from mitjax.pkg.generic_advdiff.gad_pqm_p5e_y import gad_pqm_p5e_y


def gad_pqm_adv_y(meth, kk, calc_CFL, delT, vvel, vfac, fbar, flux, *, cfg, grid):
    """GAD_PQM_ADV_Y(meth,bi,bj,kk, calc_CFL,delT,vvel,vfac,fbar, flux,myThid)
    @63cdc0b pkg/generic_advdiff/gad_pqm_adv_y.F:3-146

    C     | PQM_ADV_Y: evaluate grid-cell advective flux in Y.             |
    C     | Lagrangian-type Piecewise Quartic Method (PQM).                |
    C       meth     :: advection method.
    C       kk       :: r-index.
    C       calc_CFL :: TRUE to calc. CFL from vel.
    C       delT     :: time-step.
    C       vvel     :: vel.-comp in y-direction.
    C       vfac     :: vel.-flux in y-direction.
    C       fbar     :: grid-cell values.
    C       flux     :: adv.-flux in y-direction.

    Returns flux (every point is written). As GAD_PPM_ADV_Y (gad_ppm_adv_y.py): the row loop `do ix` (:91-141) is
    vectorised (rows independent; row locals carry the row index), vsum (:93-98) is summed in the Fortran order, the
    IF on vsum (:104-139) is a `where` per row. Here the local copies (:94-102) are inside the vsum loop, as in the
    Fortran (no effect on outputs).
    """
    sNy, sNx, OLy, OLx = cfg.sNy, cfg.sNx, cfg.OLy, cfg.OLx
    maskC = grid.maskC

    ix = loop_i(1-OLx+0, sNx+OLx-0)                                     # :68-77
#     ==================== zero stencil "ghost" cells along boundaries
    flux = flux.at[ix, +1-OLy+0].set(0.)
    flux = flux.at[ix, +1-OLy+1].set(0.)
    flux = flux.at[ix, +1-OLy+2].set(0.)
    flux = flux.at[ix, +1-OLy+3].set(0.)
    flux = flux.at[ix, sNy+OLy-0].set(0.)
    flux = flux.at[ix, sNy+OLy-1].set(0.)
    flux = flux.at[ix, sNy+OLy-2].set(0.)

    ix = loop_i(1-OLx+0, sNx+OLx-0)                                     # :91

    vsum = 0.0                                                          # :93-102
    for iy in range(1-OLy+0, sNy+OLy-0+1):
#     ================================== quick break on zero transport
        vsum = (vsum
                + jnp.abs(vfac[ix, iy]))
    iy = loop_j(1-OLy+0, sNy+OLy-0)
#     ================================== make local unit-stride copies
    floc = fbar.local("floc").at[ix, iy].set(fbar[ix, iy])
    mloc = fbar.local("mloc").at[ix, iy].set(maskC[ix, iy, kk])

#     ==================== reconstruct derivatives for WENO indicators
    ohat = None
    if meth == ENUM_PQM_WENO_LIMIT:                                     # :107-111
        ohat = {1: floc.local("ohat1"), 2: floc.local("ohat2")}
        ohat = gad_osc_hat_y(kk, ix,
                             mloc, floc,
                             ohat, cfg=cfg)

#     ==================== reconstruct 5th--order accurate edge values
    edge = {1: floc.local("edge1"), 2: floc.local("edge2")}
    edge = gad_pqm_p5e_y(kk, ix,                                        # :114-116
                         mloc, floc,
                         edge, cfg=cfg, grid=grid)

#     ==================== reconstruct coeff. for grid-cell poynomials
    fhat = {ii: floc.local(f"fhat{ii}") for ii in (1, 2, 3, 4, 5)}
    fhat = gad_pqm_hat_y(kk, ix,                                        # :119-123
                         meth,
                         mloc, floc,
                         edge, ohat,
                         fhat, cfg=cfg, grid=grid)

#     ==================== evaluate integral fluxes on grid-cell edges
    flux_flx = gad_pqm_flx_y(kk, ix,                                    # :126-130
                             calc_CFL,
                             delT, vvel,
                             vfac, fhat,
                             flux, cfg=cfg, grid=grid)

#     ================================== "null" flux on zero transport
    iy = loop_j(1-OLy+4, sNy+OLy-3)                                     # :104, :134-137
    flux = flux.at[ix, iy].set(jnp.where(vsum > 0., flux_flx[ix, iy], 0.))
    return flux
