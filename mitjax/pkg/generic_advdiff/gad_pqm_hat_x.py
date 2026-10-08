"""pkg/generic_advdiff/gad_pqm_hat_x.F: PQM grid-cell polynomials in X (GAD_PQM_HAT_X).

Row routine (see gad_osc_hat_x.py): the row arrays carry the row index j and `iy` is the caller's row loop.
"""

import jax.numpy as jnp

from mitjax.farray import loop_i
from mitjax.pkg.generic_advdiff.gad_osc_mul_x import gad_osc_mul_x
from mitjax.pkg.generic_advdiff.gad_plm_fun import gad_plm_fun_u
from mitjax.pkg.generic_advdiff.gad_pqm_fun import gad_pqm_fun_mono, gad_pqm_fun_null
from mitjax.pkg.generic_advdiff.gad_h import ENUM_PQM_MONO_LIMIT, ENUM_PQM_NULL_LIMIT, ENUM_PQM_WENO_LIMIT


def gad_pqm_hat_x(kk, iy, method, mask, fbar, edge, ohat, fhat, *, cfg, grid):
    """GAD_PQM_HAT_X(bi,bj,kk,iy, method, mask, fbar, edge, ohat, fhat, myThid)
    @63cdc0b pkg/generic_advdiff/gad_pqm_hat_x.F:3-166

    C     | PQM_HAT_X: reconstruct grid-cell PQM polynomials.              |
    C       method    :: advection scheme.
    C       mask      :: row of cell-wise mask values.
    C       fbar      :: row of cell-wise values.
    C       edge      :: row of edge-wise values/slopes.
    C       ohat      :: row of oscl. coeff.
    C       fhat      :: row of poly. coeff.

    Returns fhat ({1..5: FArray}). `method` is static; ohat is read only for the WENO limiter (None otherwise). The
    point loop (:72-161) runs on all its points at once (each point reads only inputs); the IF on mask(ix) (:74,
    :153-159) and the IFs on mono and fdel (:123-143) are `where`s. Another method raises (the Fortran's select has
    no default branch).
    """
    sNx, OLx = cfg.sNx, cfg.OLx
    dxF = grid.dxF
    fhat = dict(fhat)

    ix = loop_i(1-OLx+3, sNx+OLx-3)                                     # :72

    wet = mask[ix, iy] > 0.                                             # :74

#     =============================== scale to local grid-cell co-ords
    xhat = dxF[ix, iy] * 0.5                                            # :77

#     =============================== assemble cell mean + edge values
    ff00 = fbar[ix+0, iy]                                               # :80-84
    ffll = (ff00
            + mask[ix-1, iy]*(fbar[ix-1, iy]-ff00))
    ffrr = (ff00
            + mask[ix+1, iy]*(fbar[ix+1, iy]-ff00))

    fell = edge[+1][ix-0, iy]                                           # :86-87
    ferr = edge[+1][ix+1, iy]

    dell = edge[+2][ix-0, iy]                                           # :89-90
    derr = edge[+2][ix+1, iy]

    dell = dell * xhat                                                  # :92-93
    derr = derr * xhat

    if method == ENUM_PQM_NULL_LIMIT:                                   # :97
#     =============================== "NULL" limited grid-cell profile
        lhat, mono = gad_pqm_fun_null(ff00,                             # :99-100
                                      fell, ferr, dell, derr)

    elif method == ENUM_PQM_MONO_LIMIT:                                 # :103
#     =============================== "MONO" limited grid-cell profile
        dfds = gad_plm_fun_u(ffll, ff00, ffrr)                          # :105

        fell, ferr, dell, derr, lhat, mono = gad_pqm_fun_mono(          # :107-109
            ff00, ffll, ffrr,
            fell, ferr, dell, derr, dfds)

    elif method == ENUM_PQM_WENO_LIMIT:                                 # :112
#     =============================== "WENO" limited grid-cell profile
        dfds = gad_plm_fun_u(ffll, ff00, ffrr)                          # :114

        uhat, mono = gad_pqm_fun_null(ff00,                             # :116-117
                                      fell, ferr, dell, derr)

        fell, ferr, dell, derr, lhat, mono = gad_pqm_fun_mono(          # :119-121
            ff00, ffll, ffrr,
            fell, ferr, dell, derr, dfds)

        weno = mono > 0                                                 # :123

#     =============================== only apply WENO if it is worth it
        fdel = jnp.abs(ffrr-ff00)+jnp.abs(ff00-ffll)                    # :126-127
        fmag = jnp.abs(ffll)+jnp.abs(ff00)+jnp.abs(ffrr)

        weno = weno & (fdel > 1.e-6 * fmag)                             # :129

#     =============================== calc. WENO oscillation weighting
        scal = gad_osc_mul_x(ix, +2, mask,                              # :132-133
                             ohat, iy=iy)

        for ii in (+1, +2, +3, +4, +5):                                 # :135-139
#     =============================== blend limited/un-limited profile
            lhat[ii] = jnp.where(weno, scal[1] * uhat[ii]
                                 + scal[2] * lhat[ii], lhat[ii])
    else:
        raise ValueError(f"GAD_PQM_HAT_X: method {method} is not a PQM limiter (50, 51, 52)")

    for ii in (+1, +2, +3, +4, +5):                                     # :148-151, :155-157
#     =============================== copy polynomial onto output data
        fhat[ii] = fhat[ii].at[ix, iy].set(jnp.where(wet, lhat[ii], 0.0))
    return fhat
