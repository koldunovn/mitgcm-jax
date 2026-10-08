"""pkg/generic_advdiff/gad_ppm_hat_x.F: PPM grid-cell polynomials in X (GAD_PPM_HAT_X).

Row routine (see gad_osc_hat_x.py): the row arrays carry the row index j and `iy` is the caller's row loop.
"""

import jax.numpy as jnp

from mitjax.farray import loop_i
from mitjax.pkg.generic_advdiff.gad_osc_mul_x import gad_osc_mul_x
from mitjax.pkg.generic_advdiff.gad_plm_fun import gad_plm_fun_u
from mitjax.pkg.generic_advdiff.gad_ppm_fun import gad_ppm_fun_mono, gad_ppm_fun_null
from mitjax.pkg.generic_advdiff.gad_h import ENUM_PPM_MONO_LIMIT, ENUM_PPM_NULL_LIMIT, ENUM_PPM_WENO_LIMIT


def gad_ppm_hat_x(kk, iy, method, mask, fbar, edge, ohat, fhat, *, cfg):
    """GAD_PPM_HAT_X(bi,bj,kk,iy, method, mask, fbar, edge, ohat, fhat, myThid)
    @63cdc0b pkg/generic_advdiff/gad_ppm_hat_x.F:3-137

    C     | PPM_HAT_X: reconstruct grid-cell PPM polynomials.              |
    C       method    :: advection scheme.
    C       mask      :: row of cell-wise masking values.
    C       fbar      :: row of cell-wise values.
    C       edge      :: row of edge-wise values.
    C       ohat      :: row of oscl. coeff.
    C       fhat      :: row of poly. coeff.

    Returns fhat ({1,2,3: FArray}). `method` is static (the caller's advection scheme); ohat is read only for
    the WENO limiter (None otherwise). The point loop (:64-132) runs on all its points at once (each point reads
    only inputs); the IFs on mono and fdel (:102-122) are `where`s. The Fortran's select has no default branch
    (lhat would be undefined): another method raises.
    """
    sNx, OLx = cfg.sNx, cfg.OLx
    fhat = dict(fhat)

    ix = loop_i(1-OLx+2, sNx+OLx-2)                                     # :64

#     =============================== assemble cell mean + edge values
    ff00 = fbar[ix+0, iy]                                               # :67-71
    ffll = (ff00
            + mask[ix-1, iy]*(fbar[ix-1, iy]-ff00))
    ffrr = (ff00
            + mask[ix+1, iy]*(fbar[ix+1, iy]-ff00))

    fell = edge[ix-0, iy]                                               # :73-74
    ferr = edge[ix+1, iy]

    if method == ENUM_PPM_NULL_LIMIT:                                   # :78
#     =============================== "NULL" limited grid-cell profile
        lhat, mono = gad_ppm_fun_null(ff00, fell, ferr)                 # :80-81

    elif method == ENUM_PPM_MONO_LIMIT:                                 # :84
#     =============================== "MONO" limited grid-cell profile
        dfds = gad_plm_fun_u(ffll, ff00, ffrr)                          # :86

        fell, ferr, lhat, mono = gad_ppm_fun_mono(ff00, ffll, ffrr,     # :88-89
                                                  fell, ferr, dfds)

    elif method == ENUM_PPM_WENO_LIMIT:                                 # :92
#     =============================== "WENO" limited grid-cell profile
        dfds = gad_plm_fun_u(ffll, ff00, ffrr)                          # :94

        uhat, mono = gad_ppm_fun_null(ff00, fell, ferr)                 # :96-97

        fell, ferr, lhat, mono = gad_ppm_fun_mono(ff00, ffll, ffrr,     # :99-100
                                                  fell, ferr, dfds)

        weno = mono > 0                                                 # :102

#     =============================== only apply WENO if it is worth it
        fdel = jnp.abs(ffrr-ff00)+jnp.abs(ff00-ffll)                    # :105-106
        fmag = jnp.abs(ffll)+jnp.abs(ff00)+jnp.abs(ffrr)

        weno = weno & (fdel > 1.e-6 * fmag)                             # :108

#     =============================== calc. WENO oscillation weighting
        scal = gad_osc_mul_x(ix, +2, mask,                              # :111-112
                             ohat, iy=iy)

        for ii in (+1, +2, +3):                                         # :114-118
#     =============================== blend limited/un-limited profile
            lhat[ii] = jnp.where(weno, scal[1] * uhat[ii]
                                 + scal[2] * lhat[ii], lhat[ii])
    else:
        raise ValueError(f"GAD_PPM_HAT_X: method {method} is not a PPM limiter (40, 41, 42)")

    for ii in (+1, +2, +3):                                             # :127-130
#     =============================== copy polynomial onto output data
        fhat[ii] = fhat[ii].at[ix, iy].set(lhat[ii])
    return fhat
