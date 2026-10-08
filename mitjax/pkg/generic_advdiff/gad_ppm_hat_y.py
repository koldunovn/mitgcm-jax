"""pkg/generic_advdiff/gad_ppm_hat_y.F: PPM grid-cell polynomials in Y (GAD_PPM_HAT_Y).

Row routine (see gad_osc_hat_y.py): the row arrays (along j) carry the index i of the caller's `do ix` loop, and `ix` is that loop.
"""

import jax.numpy as jnp

from mitjax.farray import loop_j
from mitjax.pkg.generic_advdiff.gad_osc_mul_y import gad_osc_mul_y
from mitjax.pkg.generic_advdiff.gad_plm_fun import gad_plm_fun_u
from mitjax.pkg.generic_advdiff.gad_ppm_fun import gad_ppm_fun_mono, gad_ppm_fun_null
from mitjax.pkg.generic_advdiff.gad_h import ENUM_PPM_MONO_LIMIT, ENUM_PPM_NULL_LIMIT, ENUM_PPM_WENO_LIMIT


def gad_ppm_hat_y(kk, ix, method, mask, fbar, edge, ohat, fhat, *, cfg):
    """GAD_PPM_HAT_Y(bi,bj,kk,ix, method, mask, fbar, edge, ohat, fhat, myThid)
    @63cdc0b pkg/generic_advdiff/gad_ppm_hat_y.F:3-138

    C     | PPM_HAT_Y: reconstruct grid-cell PPM polynomials.              |
    C       method    :: advection scheme.
    C       mask      :: row of cell-wise masking values.
    C       fbar      :: row of cell-wise values.
    C       edge      :: row of edge-wise values.
    C       ohat      :: row of oscl. coeff.
    C       fhat      :: row of poly. coeff.

    Returns fhat ({1,2,3: FArray}). `method` is static (the caller's advection scheme); ohat is read only for
    the WENO limiter (None otherwise). The point loop (:65-133) runs on all its points at once (each point reads
    only inputs); the IFs on mono and fdel (:103-123) are `where`s. The Fortran's select has no default branch
    (lhat would be undefined): another method raises.
    """
    sNy, OLy = cfg.sNy, cfg.OLy
    fhat = dict(fhat)

    iy = loop_j(1-OLy+2, sNy+OLy-2)                                     # :65

#     =============================== assemble cell mean + edge values
    ff00 = fbar[ix, iy+0]                                               # :68-72
    ffll = (ff00
            + mask[ix, iy-1]*(fbar[ix, iy-1]-ff00))
    ffrr = (ff00
            + mask[ix, iy+1]*(fbar[ix, iy+1]-ff00))

    fell = edge[ix, iy-0]                                               # :74-75
    ferr = edge[ix, iy+1]

    if method == ENUM_PPM_NULL_LIMIT:                                   # :79
#     =============================== "NULL" limited grid-cell profile
        lhat, mono = gad_ppm_fun_null(ff00, fell, ferr)                 # :81-82

    elif method == ENUM_PPM_MONO_LIMIT:                                 # :85
#     =============================== "MONO" limited grid-cell profile
        dfds = gad_plm_fun_u(ffll, ff00, ffrr)                          # :87

        fell, ferr, lhat, mono = gad_ppm_fun_mono(ff00, ffll, ffrr,     # :89-90
                                                  fell, ferr, dfds)

    elif method == ENUM_PPM_WENO_LIMIT:                                 # :93
#     =============================== "WENO" limited grid-cell profile
        dfds = gad_plm_fun_u(ffll, ff00, ffrr)                          # :95

        uhat, mono = gad_ppm_fun_null(ff00, fell, ferr)                 # :97-98

        fell, ferr, lhat, mono = gad_ppm_fun_mono(ff00, ffll, ffrr,     # :100-101
                                                  fell, ferr, dfds)

        weno = mono > 0                                                 # :103

#     =============================== only apply WENO if it is worth it
        fdel = jnp.abs(ffrr-ff00)+jnp.abs(ff00-ffll)                    # :106-107
        fmag = jnp.abs(ffll)+jnp.abs(ff00)+jnp.abs(ffrr)

        weno = weno & (fdel > 1.e-6 * fmag)                             # :109

#     =============================== calc. WENO oscillation weighting
        scal = gad_osc_mul_y(iy, +2, mask,                              # :112-113
                             ohat, ix=ix)

        for ii in (+1, +2, +3):                                         # :115-119
#     =============================== blend limited/un-limited profile
            lhat[ii] = jnp.where(weno, scal[1] * uhat[ii]
                                 + scal[2] * lhat[ii], lhat[ii])
    else:
        raise ValueError(f"GAD_PPM_HAT_Y: method {method} is not a PPM limiter (40, 41, 42)")

    for ii in (+1, +2, +3):                                             # :128-131
#     =============================== copy polynomial onto output data
        fhat[ii] = fhat[ii].at[ix, iy].set(lhat[ii])
    return fhat
