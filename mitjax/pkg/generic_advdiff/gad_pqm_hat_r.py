"""pkg/generic_advdiff/gad_pqm_hat_r.F: PQM grid-cell polynomials in R (GAD_PQM_HAT_R).

Column routine (see gad_osc_hat_r.py): the column arrays carry the column indices and `ix, iy` are the caller's
column loops.
"""

import jax.numpy as jnp

from mitjax.farray import loops_kji
from mitjax.pkg.generic_advdiff.gad_osc_mul_r import gad_osc_mul_r
from mitjax.pkg.generic_advdiff.gad_plm_fun import gad_plm_fun_u
from mitjax.pkg.generic_advdiff.gad_pqm_fun import gad_pqm_fun_mono, gad_pqm_fun_null
from mitjax.pkg.generic_advdiff.gad_h import ENUM_PQM_MONO_LIMIT, ENUM_PQM_NULL_LIMIT, ENUM_PQM_WENO_LIMIT


def gad_pqm_hat_r(ix, iy, method, mask, fbar, edge, ohat, fhat, *, cfg, grid):
    """GAD_PQM_HAT_R(bi,bj,ix,iy, method, mask, fbar, edge, ohat, fhat, myThid)
    @63cdc0b pkg/generic_advdiff/gad_pqm_hat_r.F:3-156

    C     | PQM_HAT_R: reconstruct grid-cell PQM polynomials.              |

    Returns fhat ({1..5: FArray}, declared (1-0:Nr+0) per column). As GAD_PQM_HAT_X (gad_pqm_hat_x.py) without
    its IF on mask (the R variant has none): `method` static, ohat read only for WENO, the level loop (:72-151)
    vectorised with the column loops, the IFs on mono and fdel (:121-141) as `where`s; another method raises.
    """
    Nr = cfg.Nr
    drF = grid.drF
    fhat = dict(fhat)

    ir, iy, ix = loops_kji((+1, Nr), (iy.first, iy.last), (ix.first, ix.last))   # :72

#     =============================== scale to local grid-cell co-ords
    rhat = drF[ir] * .5                                                 # :75

#     =============================== assemble cell mean + edge values
    ff00 = fbar[ix, iy, ir+0]                                           # :78-82
    ffll = (ff00
            + mask[ix, iy, ir-1]*(fbar[ix, iy, ir-1]-ff00))
    ffrr = (ff00
            + mask[ix, iy, ir+1]*(fbar[ix, iy, ir+1]-ff00))

    fell = edge[+1][ix, iy, ir-0]                                       # :84-85
    ferr = edge[+1][ix, iy, ir+1]

    dell = edge[+2][ix, iy, ir-0]                                       # :87-88
    derr = edge[+2][ix, iy, ir+1]

    dell = dell * rhat                                                  # :90-91
    derr = derr * rhat

    if method == ENUM_PQM_NULL_LIMIT:                                   # :95
#     =============================== "NULL" limited grid-cell profile
        lhat, mono = gad_pqm_fun_null(ff00,                             # :97-98
                                      fell, ferr, dell, derr)

    elif method == ENUM_PQM_MONO_LIMIT:                                 # :101
#     =============================== "MONO" limited grid-cell profile
        dfds = gad_plm_fun_u(ffll, ff00, ffrr)                          # :103

        fell, ferr, dell, derr, lhat, mono = gad_pqm_fun_mono(          # :105-107
            ff00, ffll, ffrr,
            fell, ferr, dell, derr, dfds)

    elif method == ENUM_PQM_WENO_LIMIT:                                 # :110
#     =============================== "WENO" limited grid-cell profile
        dfds = gad_plm_fun_u(ffll, ff00, ffrr)                          # :112

        uhat, mono = gad_pqm_fun_null(ff00,                             # :114-115
                                      fell, ferr, dell, derr)

        fell, ferr, dell, derr, lhat, mono = gad_pqm_fun_mono(          # :117-119
            ff00, ffll, ffrr,
            fell, ferr, dell, derr, dfds)

        weno = mono > 0                                                 # :121

#     =============================== only apply WENO if it is worth it
        fdel = jnp.abs(ffrr-ff00)+jnp.abs(ff00-ffll)                    # :124-125
        fmag = jnp.abs(ffll)+jnp.abs(ff00)+jnp.abs(ffrr)

        weno = weno & (fdel > 1.e-6 * fmag)                             # :127

#     =============================== calc. WENO oscillation weighting
        scal = gad_osc_mul_r(ir, +2, mask,                              # :130-131
                             ohat, ix=ix, iy=iy)

        for ii in (+1, +2, +3, +4, +5):                                 # :133-137
#     =============================== blend limited/un-limited profile
            lhat[ii] = jnp.where(weno, scal[1] * uhat[ii]
                                 + scal[2] * lhat[ii], lhat[ii])
    else:
        raise ValueError(f"GAD_PQM_HAT_R: method {method} is not a PQM limiter (50, 51, 52)")

    for ii in (+1, +2, +3, +4, +5):                                     # :146-149
#     =============================== copy polynomial onto output data
        fhat[ii] = fhat[ii].at[ix, iy, ir].set(lhat[ii])
    return fhat
