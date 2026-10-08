"""pkg/generic_advdiff/gad_ppm_adv_y.F: PPM advective flux in Y (GAD_PPM_ADV_Y)."""

import jax.numpy as jnp

from mitjax.farray import loop_j, loop_i
from mitjax.pkg.generic_advdiff.gad_osc_hat_y import gad_osc_hat_y
from mitjax.pkg.generic_advdiff.gad_ppm_flx_y import gad_ppm_flx_y
from mitjax.pkg.generic_advdiff.gad_h import ENUM_PPM_WENO_LIMIT
from mitjax.pkg.generic_advdiff.gad_ppm_hat_y import gad_ppm_hat_y
from mitjax.pkg.generic_advdiff.gad_ppm_p3e_y import gad_ppm_p3e_y


def gad_ppm_adv_y(meth, kk, calc_CFL, delT, vvel, vfac, fbar, flux, *, cfg, grid):
    """GAD_PPM_ADV_Y(meth,bi,bj,kk, calc_CFL,delT,vvel,vfac,fbar, flux,myThid)
    @63cdc0b pkg/generic_advdiff/gad_ppm_adv_y.F:3-148

    C     | PPM_ADV_Y: evaluate grid-cell advective flux in Y.             |
    C     | Lagrangian-type Piecewise Parabolic Method (PPM).              |
    C       meth     :: advection method.
    C       kk       :: r-index.
    C       calc_CFL :: TRUE to calc. CFL from vel.
    C       delT     :: time-step.
    C       vvel     :: vel.-comp in y-direction.
    C       vfac     :: vel.-flux in y-direction.
    C       fbar     :: grid-cell values.
    C       flux     :: adv.-flux in y-direction.

    Returns flux (every point is written). `meth`, `kk` and `calc_CFL` are static, delT a traced float.
    The row loop `do ix` (:90-143) is vectorised: every row is independent (reads its own row of the inputs, writes
    its own row of flux), so the row locals floc, mloc, edge, fhat, ohat carry the row index (declared like fbar) and
    the row routines get the row loop `ix`. The sum vsum (:92-97) runs in the Fortran order. The IF on vsum (:99-141)
    is a `where` on each row between the flux of GAD_PPM_FLX_Y and zero; the local copies (:101-106) are made for
    every row (rows with vsum = 0 never read them). The IF on meth (:109) is static.
    """
    sNy, sNx, OLy, OLx = cfg.sNy, cfg.sNx, cfg.OLy, cfg.OLx
    maskC = grid.maskC

    ix = loop_i(1-OLx+0, sNx+OLx-0)                                     # :67-76
#     ==================== zero stencil "ghost" cells along boundaries
    flux = flux.at[ix, +1-OLy+0].set(0.)
    flux = flux.at[ix, +1-OLy+1].set(0.)
    flux = flux.at[ix, +1-OLy+2].set(0.)
    flux = flux.at[ix, +1-OLy+3].set(0.)
    flux = flux.at[ix, sNy+OLy-0].set(0.)
    flux = flux.at[ix, sNy+OLy-1].set(0.)
    flux = flux.at[ix, sNy+OLy-2].set(0.)

    ix = loop_i(1-OLx+0, sNx+OLx-0)                                     # :90

    vsum = 0.0                                                          # :92-97
    for iy in range(1-OLy+0, sNy+OLy-0+1):
#     ================================== quick break on zero transport
        vsum = (vsum
                + jnp.abs(vfac[ix, iy]))

    iy = loop_j(1-OLy+0, sNy+OLy-0)                                     # :101-106
#     ================================== make local unit-stride copies
    floc = fbar.local("floc").at[ix, iy].set(fbar[ix, iy])
    mloc = fbar.local("mloc").at[ix, iy].set(maskC[ix, iy, kk])

#     ==================== reconstruct derivatives for WENO indicators
    ohat = None
    if meth == ENUM_PPM_WENO_LIMIT:                                     # :109-113
        ohat = {1: floc.local("ohat1"), 2: floc.local("ohat2")}
        ohat = gad_osc_hat_y(kk, ix,
                             mloc, floc,
                             ohat, cfg=cfg)

#     ==================== reconstruct 3rd--order accurate edge values
    edge = gad_ppm_p3e_y(kk, ix,                                        # :116-118
                         mloc, floc,
                         floc.local("edge"), cfg=cfg)

#     ==================== reconstruct coeff. for grid-cell poynomials
    fhat = {ii: floc.local(f"fhat{ii}") for ii in (1, 2, 3)}
    fhat = gad_ppm_hat_y(kk, ix,                                        # :121-125
                         meth,
                         mloc, floc,
                         edge, ohat,
                         fhat, cfg=cfg)

#     ==================== evaluate integral fluxes on grid-cell edges
    flux_flx = gad_ppm_flx_y(kk, ix,                                    # :128-132
                             calc_CFL,
                             delT, vvel,
                             vfac, fhat,
                             flux, cfg=cfg, grid=grid)

#     ================================== "null" flux on zero transport
    iy = loop_j(1-OLy+3, sNy+OLy-2)                                     # :99, :136-139
    flux = flux.at[ix, iy].set(jnp.where(vsum > 0., flux_flx[ix, iy], 0.))
    return flux
