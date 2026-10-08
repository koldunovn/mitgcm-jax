"""pkg/generic_advdiff/gad_pqm_adv_x.F: PQM advective flux in X (GAD_PQM_ADV_X)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.pkg.generic_advdiff.gad_osc_hat_x import gad_osc_hat_x
from mitjax.pkg.generic_advdiff.gad_pqm_flx_x import gad_pqm_flx_x
from mitjax.pkg.generic_advdiff.gad_h import ENUM_PQM_WENO_LIMIT
from mitjax.pkg.generic_advdiff.gad_pqm_hat_x import gad_pqm_hat_x
from mitjax.pkg.generic_advdiff.gad_pqm_p5e_x import gad_pqm_p5e_x


def gad_pqm_adv_x(meth, kk, calc_CFL, delT, uvel, ufac, fbar, flux, *, cfg, grid):
    """GAD_PQM_ADV_X(meth,bi,bj,kk, calc_CFL,delT,uvel,ufac,fbar, flux,myThid)
    @63cdc0b pkg/generic_advdiff/gad_pqm_adv_x.F:3-146

    C     | PQM_ADV_X: evaluate grid-cell advective flux in X.             |
    C     | Lagrangian-type Piecewise Quartic Method (PQM).                |
    C       meth     :: advection method.
    C       kk       :: r-index.
    C       calc_CFL :: TRUE to calc. CFL from vel.
    C       delT     :: time-step.
    C       uvel     :: vel.-comp in x-direction.
    C       ufac     :: vel.-flux in x-direction.
    C       fbar     :: grid-cell values.
    C       flux     :: adv.-flux in x-direction.

    Returns flux (every point is written). As GAD_PPM_ADV_X (gad_ppm_adv_x.py): the row loop `do iy` (:91-141) is
    vectorised (rows independent; row locals carry the row index), vsum (:93-98) is summed in the Fortran order, the
    IF on vsum (:104-139) is a `where` per row. Here the local copies (:94-102) are inside the vsum loop, as in the
    Fortran (no effect on outputs).
    """
    sNx, sNy, OLx, OLy = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy
    maskC = grid.maskC

    iy = loop_j(1-OLy+0, sNy+OLy-0)                                     # :68-77
#     ==================== zero stencil "ghost" cells along boundaries
    flux = flux.at[+1-OLx+0, iy].set(0.)
    flux = flux.at[+1-OLx+1, iy].set(0.)
    flux = flux.at[+1-OLx+2, iy].set(0.)
    flux = flux.at[+1-OLx+3, iy].set(0.)
    flux = flux.at[sNx+OLx-0, iy].set(0.)
    flux = flux.at[sNx+OLx-1, iy].set(0.)
    flux = flux.at[sNx+OLx-2, iy].set(0.)

    iy = loop_j(1-OLy+0, sNy+OLy-0)                                     # :91

    vsum = 0.0                                                          # :93-102
    for ix in range(1-OLx+0, sNx+OLx-0+1):
#     ================================== quick break on zero transport
        vsum = (vsum
                + jnp.abs(ufac[ix, iy]))
    ix = loop_i(1-OLx+0, sNx+OLx-0)
#     ================================== make local unit-stride copies
    floc = fbar.local("floc").at[ix, iy].set(fbar[ix, iy])
    mloc = fbar.local("mloc").at[ix, iy].set(maskC[ix, iy, kk])

#     ==================== reconstruct derivatives for WENO indicators
    ohat = None
    if meth == ENUM_PQM_WENO_LIMIT:                                     # :107-111
        ohat = {1: floc.local("ohat1"), 2: floc.local("ohat2")}
        ohat = gad_osc_hat_x(kk, iy,
                             mloc, floc,
                             ohat, cfg=cfg)

#     ==================== reconstruct 5th--order accurate edge values
    edge = {1: floc.local("edge1"), 2: floc.local("edge2")}
    edge = gad_pqm_p5e_x(kk, iy,                                        # :114-116
                         mloc, floc,
                         edge, cfg=cfg, grid=grid)

#     ==================== reconstruct coeff. for grid-cell poynomials
    fhat = {ii: floc.local(f"fhat{ii}") for ii in (1, 2, 3, 4, 5)}
    fhat = gad_pqm_hat_x(kk, iy,                                        # :119-123
                         meth,
                         mloc, floc,
                         edge, ohat,
                         fhat, cfg=cfg, grid=grid)

#     ==================== evaluate integral fluxes on grid-cell edges
    flux_flx = gad_pqm_flx_x(kk, iy,                                    # :126-130
                             calc_CFL,
                             delT, uvel,
                             ufac, fhat,
                             flux, cfg=cfg, grid=grid)

#     ================================== "null" flux on zero transport
    ix = loop_i(1-OLx+4, sNx+OLx-3)                                     # :104, :134-137
    flux = flux.at[ix, iy].set(jnp.where(vsum > 0., flux_flx[ix, iy], 0.))
    return flux
