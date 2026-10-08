"""pkg/generic_advdiff/gad_pqm_adv_r.F: PQM advective flux in R (GAD_PQM_ADV_R)."""

import jax.numpy as jnp

from mitjax.farray import loops_kji
from mitjax.pkg.generic_advdiff.gad_osc_hat_r import gad_osc_hat_r
from mitjax.pkg.generic_advdiff.gad_ppm_adv_r import column_local
from mitjax.pkg.generic_advdiff.gad_pqm_flx_r import gad_pqm_flx_r
from mitjax.pkg.generic_advdiff.gad_pqm_hat_r import gad_pqm_hat_r
from mitjax.pkg.generic_advdiff.gad_h import ENUM_PQM_WENO_LIMIT
from mitjax.pkg.generic_advdiff.gad_pqm_p5e_r import gad_pqm_p5e_r


def gad_pqm_adv_r(meth, delT, wvel, wfac, fbar, flux, *, cfg, grid):
    """GAD_PQM_ADV_R(meth,bi,bj, delT,wvel,wfac,fbar, flux,myThid)   @63cdc0b pkg/generic_advdiff/gad_pqm_adv_r.F:3-158

    C     | PQM_ADV_R: evaluate grid-cell advective flux in R.             |
    C     | Lagrangian-type Piecewise Quartic Method (PQM).                |
    C       meth     :: advection method.
    C       delT     :: level-wise time-steps.
    C       wvel     :: vel.-comp in r-direction.
    C       wfac     :: vel.-flux in r-direction.
    C       fbar     :: grid-cell values.
    C       flux     :: adv.-flux in r-direction.

    Returns flux (every point is written). As GAD_PPM_ADV_R (gad_ppm_adv_r.py), with these differences of the
    Fortran: vsum sums |wfac| over ir = 1..Nr (:97-102; GAD_PPM_ADV_R sums |velR| over 2..Nr), and the local
    copies (:103-105) are inside that loop (no effect on outputs).
    """
    sNx, sNy, OLx, OLy, Nr = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy, cfg.Nr
    maskC = grid.maskC

    floc = column_local(fbar, "floc", (1-3, Nr+3), cfg=cfg)             # :59-63
    mloc = column_local(fbar, "mloc", (1-3, Nr+3), cfg=cfg)
    fhat = {ii: column_local(fbar, f"fhat{ii}", (1-0, Nr+0), cfg=cfg) for ii in (1, 2, 3, 4, 5)}
    edge = {1: column_local(fbar, "edge1", (1-0, Nr+1), cfg=cfg),
            2: column_local(fbar, "edge2", (1-0, Nr+1), cfg=cfg)}

    ir, iy, ix = loops_kji((+1, Nr), (1-OLy+0, sNy+OLy-0), (1-OLx+0, sNx+OLx-0))

#     ======================================= mask boundary conditions
    mloc = mloc.at[ix, iy, -2].set(0.)                                  # :67-72
    mloc = mloc.at[ix, iy, -1].set(0.)
    mloc = mloc.at[ix, iy, +0].set(0.)
    mloc = mloc.at[ix, iy, Nr+1].set(0.)
    mloc = mloc.at[ix, iy, Nr+2].set(0.)
    mloc = mloc.at[ix, iy, Nr+3].set(0.)

#     ======================================= no flux through surf. bc
    flux = flux.at[ix, iy, +1].set(0.)                                  # :86-91

#     ==================== calculate transport for interior grid-cells    (:94-153)
    vsum = 0.0                                                          # :97-106
    for kr in range(+1, Nr+1):
#     ================================== quick break on zero transport
        vsum = (vsum
                + jnp.abs(wfac[ix, iy, kr]))
#     ================================== make local unit-stride copies
    floc = floc.at[ix, iy, ir].set(fbar[ix, iy, ir])
    mloc = mloc.at[ix, iy, ir].set(maskC[ix, iy, ir])

#     ================================== make mask boundary conditions
    floc = floc.at[ix, iy, -2].set(floc[ix, iy, +1])                    # :111-116
    floc = floc.at[ix, iy, -1].set(floc[ix, iy, +1])
    floc = floc.at[ix, iy, +0].set(floc[ix, iy, +1])
    floc = floc.at[ix, iy, Nr+1].set(floc[ix, iy, Nr])
    floc = floc.at[ix, iy, Nr+2].set(floc[ix, iy, Nr])
    floc = floc.at[ix, iy, Nr+3].set(floc[ix, iy, Nr])

#     ==================== reconstruct derivatives for WENO indicators
    ohat = None
    if meth == ENUM_PQM_WENO_LIMIT:                                     # :119-123
        ohat = {1: column_local(fbar, "ohat1", (1-3, Nr+3), cfg=cfg),
                2: column_local(fbar, "ohat2", (1-3, Nr+3), cfg=cfg)}
        ohat = gad_osc_hat_r(ix, iy,
                             mloc, floc,
                             ohat, cfg=cfg)

#     ==================== reconstruct 5th--order accurate edge values
    edge = gad_pqm_p5e_r(ix, iy,                                        # :126-128
                         mloc, floc,
                         edge, cfg=cfg, grid=grid)

#     ==================== reconstruct coeff. for grid-cell poynomials
    fhat = gad_pqm_hat_r(ix, iy,                                        # :131-135
                         meth,
                         mloc, floc,
                         edge, ohat,
                         fhat, cfg=cfg, grid=grid)

#     ==================== evaluate integral fluxes on grid-cell edges
    flux_flx = gad_pqm_flx_r(ix, iy,                                    # :138-141
                             delT, wvel,
                             wfac, fhat,
                             flux, cfg=cfg, grid=grid)

#     ================================== "null" flux on zero transport
    ir, iy, ix = loops_kji((2, Nr), (1-OLy+0, sNy+OLy-0), (1-OLx+0, sNx+OLx-0))   # :108, :145-148
    flux = flux.at[ix, iy, ir].set(jnp.where(vsum > 0., flux_flx[ix, iy, ir], 0.))
    return flux
