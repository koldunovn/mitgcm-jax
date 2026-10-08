"""pkg/generic_advdiff/gad_ppm_hat_r.F: PPM grid-cell polynomials in R (GAD_PPM_HAT_R).

Column routine (see gad_osc_hat_r.py): the column arrays carry the column indices and `ix, iy` are the caller's
column loops.
"""

import jax.numpy as jnp

from mitjax.farray import loops_kji
from mitjax.pkg.generic_advdiff.gad_osc_mul_r import gad_osc_mul_r
from mitjax.pkg.generic_advdiff.gad_plm_fun import gad_plm_fun_u
from mitjax.pkg.generic_advdiff.gad_ppm_fun import gad_ppm_fun_mono, gad_ppm_fun_null
from mitjax.pkg.generic_advdiff.gad_h import ENUM_PPM_MONO_LIMIT, ENUM_PPM_NULL_LIMIT, ENUM_PPM_WENO_LIMIT


def gad_ppm_hat_r(ix, iy, method, mask, fbar, edge, ohat, fhat, *, cfg):
    """GAD_PPM_HAT_R(bi,bj,ix,iy, method, mask, fbar, edge, ohat, fhat, myThid)
    @63cdc0b pkg/generic_advdiff/gad_ppm_hat_r.F:3-136

    C     | PPM_HAT_R: reconstruct grid-cell PPM polynomials.              |

    Returns fhat ({1,2,3: FArray}, declared (1-0:Nr+0) per column). As GAD_PPM_HAT_X (gad_ppm_hat_x.py): `method`
    static, ohat read only for WENO, the level loop (:63-131) vectorised with the column loops, the IFs on mono and
    fdel (:101-121) as `where`s; another method raises.
    """
    Nr = cfg.Nr
    fhat = dict(fhat)

    ir, iy, ix = loops_kji((+1, Nr), (iy.first, iy.last), (ix.first, ix.last))   # :63

#     =============================== assemble cell mean + edge values
    ff00 = fbar[ix, iy, ir+0]                                           # :66-70
    ffll = (ff00
            + mask[ix, iy, ir-1]*(fbar[ix, iy, ir-1]-ff00))
    ffrr = (ff00
            + mask[ix, iy, ir+1]*(fbar[ix, iy, ir+1]-ff00))

    fell = edge[ix, iy, ir-0]                                           # :72-73
    ferr = edge[ix, iy, ir+1]

    if method == ENUM_PPM_NULL_LIMIT:                                   # :77
#     =============================== "NULL" limited grid-cell profile
        lhat, mono = gad_ppm_fun_null(ff00, fell, ferr)                 # :79-80

    elif method == ENUM_PPM_MONO_LIMIT:                                 # :83
#     =============================== "MONO" limited grid-cell profile
        dfds = gad_plm_fun_u(ffll, ff00, ffrr)                          # :85

        fell, ferr, lhat, mono = gad_ppm_fun_mono(ff00, ffll, ffrr,     # :87-88
                                                  fell, ferr, dfds)

    elif method == ENUM_PPM_WENO_LIMIT:                                 # :91
#     =============================== "WENO" limited grid-cell profile
        dfds = gad_plm_fun_u(ffll, ff00, ffrr)                          # :93

        uhat, mono = gad_ppm_fun_null(ff00, fell, ferr)                 # :95-96

        fell, ferr, lhat, mono = gad_ppm_fun_mono(ff00, ffll, ffrr,     # :98-99
                                                  fell, ferr, dfds)

        weno = mono > 0                                                 # :101

#     =============================== only apply WENO if it is worth it
        fdel = jnp.abs(ffrr-ff00)+jnp.abs(ff00-ffll)                    # :104-105
        fmag = jnp.abs(ffll)+jnp.abs(ff00)+jnp.abs(ffrr)

        weno = weno & (fdel > 1.e-6 * fmag)                             # :107

#     =============================== calc. WENO oscillation weighting
        scal = gad_osc_mul_r(ir, +2, mask,                              # :110-111
                             ohat, ix=ix, iy=iy)

        for ii in (+1, +2, +3):                                         # :113-117
#     =============================== blend limited/un-limited profile
            lhat[ii] = jnp.where(weno, scal[1] * uhat[ii]
                                 + scal[2] * lhat[ii], lhat[ii])
    else:
        raise ValueError(f"GAD_PPM_HAT_R: method {method} is not a PPM limiter (40, 41, 42)")

    for ii in (+1, +2, +3):                                             # :126-129
#     =============================== copy polynomial onto output data
        fhat[ii] = fhat[ii].at[ix, iy, ir].set(lhat[ii])
    return fhat
