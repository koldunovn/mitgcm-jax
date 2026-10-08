"""pkg/generic_advdiff/gad_ppm_adv_x.F: PPM advective flux in X (GAD_PPM_ADV_X)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.pkg.generic_advdiff.gad_osc_hat_x import gad_osc_hat_x
from mitjax.pkg.generic_advdiff.gad_ppm_flx_x import gad_ppm_flx_x
from mitjax.pkg.generic_advdiff.gad_h import ENUM_PPM_WENO_LIMIT
from mitjax.pkg.generic_advdiff.gad_ppm_hat_x import gad_ppm_hat_x
from mitjax.pkg.generic_advdiff.gad_ppm_p3e_x import gad_ppm_p3e_x


def gad_ppm_adv_x(meth, kk, calc_CFL, delT, uvel, ufac, fbar, flux, *, cfg, grid):
    """GAD_PPM_ADV_X(meth,bi,bj,kk, calc_CFL,delT,uvel,ufac,fbar, flux,myThid)
    @63cdc0b pkg/generic_advdiff/gad_ppm_adv_x.F:3-148

    C     | PPM_ADV_X: evaluate grid-cell advective flux in X.             |
    C     | Lagrangian-type Piecewise Parabolic Method (PPM).              |
    C       meth     :: advection method.
    C       kk       :: r-index.
    C       calc_CFL :: TRUE to calc. CFL from vel.
    C       delT     :: time-step.
    C       uvel     :: vel.-comp in x-direction.
    C       ufac     :: vel.-flux in x-direction.
    C       fbar     :: grid-cell values.
    C       flux     :: adv.-flux in x-direction.

    Returns flux (every point is written). `meth`, `kk` and `calc_CFL` are static, delT a traced float.
    The row loop `do iy` (:90-143) is vectorised: every row is independent (reads its own row of the inputs, writes
    its own row of flux), so the row locals floc, mloc, edge, fhat, ohat carry the row index (declared like fbar) and
    the row routines get the row loop `iy`. The sum vsum (:92-97) runs in the Fortran order. The IF on vsum (:99-141)
    is a `where` on each row between the flux of GAD_PPM_FLX_X and zero; the local copies (:101-106) are made for
    every row (rows with vsum = 0 never read them). The IF on meth (:109) is static.
    """
    sNx, sNy, OLx, OLy = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy
    maskC = grid.maskC

    iy = loop_j(1-OLy+0, sNy+OLy-0)                                     # :67-76
#     ==================== zero stencil "ghost" cells along boundaries
    flux = flux.at[+1-OLx+0, iy].set(0.)
    flux = flux.at[+1-OLx+1, iy].set(0.)
    flux = flux.at[+1-OLx+2, iy].set(0.)
    flux = flux.at[+1-OLx+3, iy].set(0.)
    flux = flux.at[sNx+OLx-0, iy].set(0.)
    flux = flux.at[sNx+OLx-1, iy].set(0.)
    flux = flux.at[sNx+OLx-2, iy].set(0.)

    iy = loop_j(1-OLy+0, sNy+OLy-0)                                     # :90

    vsum = 0.0                                                          # :92-97
    for ix in range(1-OLx+0, sNx+OLx-0+1):
#     ================================== quick break on zero transport
        vsum = (vsum
                + jnp.abs(ufac[ix, iy]))

    ix = loop_i(1-OLx+0, sNx+OLx-0)                                     # :101-106
#     ================================== make local unit-stride copies
    floc = fbar.local("floc").at[ix, iy].set(fbar[ix, iy])
    mloc = fbar.local("mloc").at[ix, iy].set(maskC[ix, iy, kk])

#     ==================== reconstruct derivatives for WENO indicators
    ohat = None
    if meth == ENUM_PPM_WENO_LIMIT:                                     # :109-113
        ohat = {1: floc.local("ohat1"), 2: floc.local("ohat2")}
        ohat = gad_osc_hat_x(kk, iy,
                             mloc, floc,
                             ohat, cfg=cfg)

#     ==================== reconstruct 3rd--order accurate edge values
    edge = gad_ppm_p3e_x(kk, iy,                                        # :116-118
                         mloc, floc,
                         floc.local("edge"), cfg=cfg)

#     ==================== reconstruct coeff. for grid-cell poynomials
    fhat = {ii: floc.local(f"fhat{ii}") for ii in (1, 2, 3)}
    fhat = gad_ppm_hat_x(kk, iy,                                        # :121-125
                         meth,
                         mloc, floc,
                         edge, ohat,
                         fhat, cfg=cfg)

#     ==================== evaluate integral fluxes on grid-cell edges
    flux_flx = gad_ppm_flx_x(kk, iy,                                    # :128-132
                             calc_CFL,
                             delT, uvel,
                             ufac, fhat,
                             flux, cfg=cfg, grid=grid)

#     ================================== "null" flux on zero transport
    ix = loop_i(1-OLx+3, sNx+OLx-2)                                     # :99, :136-139
    flux = flux.at[ix, iy].set(jnp.where(vsum > 0., flux_flx[ix, iy], 0.))
    return flux
