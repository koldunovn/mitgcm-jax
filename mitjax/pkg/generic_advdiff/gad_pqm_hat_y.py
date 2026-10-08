"""pkg/generic_advdiff/gad_pqm_hat_y.F: PQM grid-cell polynomials in Y (GAD_PQM_HAT_Y).

Row routine (see gad_osc_hat_y.py): the row arrays (along j) carry the index i of the caller's `do ix` loop, and `ix` is that loop.
"""

import jax.numpy as jnp

from mitjax.farray import loop_j
from mitjax.pkg.generic_advdiff.gad_osc_mul_y import gad_osc_mul_y
from mitjax.pkg.generic_advdiff.gad_plm_fun import gad_plm_fun_u
from mitjax.pkg.generic_advdiff.gad_pqm_fun import gad_pqm_fun_mono, gad_pqm_fun_null
from mitjax.pkg.generic_advdiff.gad_h import ENUM_PQM_MONO_LIMIT, ENUM_PQM_NULL_LIMIT, ENUM_PQM_WENO_LIMIT


def gad_pqm_hat_y(kk, ix, method, mask, fbar, edge, ohat, fhat, *, cfg, grid):
    """GAD_PQM_HAT_Y(bi,bj,kk,ix, method, mask, fbar, edge, ohat, fhat, myThid)
    @63cdc0b pkg/generic_advdiff/gad_pqm_hat_y.F:3-167

    C     | PQM_HAT_Y: reconstruct grid-cell PQM polynomials.              |
    C       method    :: advection scheme.
    C       mask      :: row of cell-wise mask values.
    C       fbar      :: row of cell-wise values.
    C       edge      :: row of edge-wise values/slopes.
    C       ohat      :: row of oscl. coeff.
    C       fhat      :: row of poly. coeff.

    Returns fhat ({1..5: FArray}). `method` is static; ohat is read only for the WENO limiter (None otherwise). The
    point loop (:73-162) runs on all its points at once (each point reads only inputs); the IF on mask(iy) (:75,
    :153-159) and the IFs on mono and fdel (:124-144) are `where`s. Another method raises (the Fortran's select has
    no default branch).
    """
    sNy, OLy = cfg.sNy, cfg.OLy
    dyF = grid.dyF
    fhat = dict(fhat)

    iy = loop_j(1-OLy+3, sNy+OLy-3)                                     # :73

    wet = mask[ix, iy] > 0.                                             # :75

#     =============================== scale to local grid-cell co-ords
    yhat = dyF[ix, iy] * 0.5                                            # :78

#     =============================== assemble cell mean + edge values
    ff00 = fbar[ix, iy+0]                                               # :81-85
    ffll = (ff00
            + mask[ix, iy-1]*(fbar[ix, iy-1]-ff00))
    ffrr = (ff00
            + mask[ix, iy+1]*(fbar[ix, iy+1]-ff00))

    fell = edge[+1][ix, iy-0]                                           # :87-88
    ferr = edge[+1][ix, iy+1]

    dell = edge[+2][ix, iy-0]                                           # :90-91
    derr = edge[+2][ix, iy+1]

    dell = dell * yhat                                                  # :93-94
    derr = derr * yhat

    if method == ENUM_PQM_NULL_LIMIT:                                   # :98
#     =============================== "NULL" limited grid-cell profile
        lhat, mono = gad_pqm_fun_null(ff00,                             # :100-101
                                      fell, ferr, dell, derr)

    elif method == ENUM_PQM_MONO_LIMIT:                                 # :104
#     =============================== "MONO" limited grid-cell profile
        dfds = gad_plm_fun_u(ffll, ff00, ffrr)                          # :106

        fell, ferr, dell, derr, lhat, mono = gad_pqm_fun_mono(          # :108-110
            ff00, ffll, ffrr,
            fell, ferr, dell, derr, dfds)

    elif method == ENUM_PQM_WENO_LIMIT:                                 # :113
#     =============================== "WENO" limited grid-cell profile
        dfds = gad_plm_fun_u(ffll, ff00, ffrr)                          # :115

        uhat, mono = gad_pqm_fun_null(ff00,                             # :117-118
                                      fell, ferr, dell, derr)

        fell, ferr, dell, derr, lhat, mono = gad_pqm_fun_mono(          # :120-122
            ff00, ffll, ffrr,
            fell, ferr, dell, derr, dfds)

        weno = mono > 0                                                 # :124

#     =============================== only apply WENO if it is worth it
        fdel = jnp.abs(ffrr-ff00)+jnp.abs(ff00-ffll)                    # :127-128
        fmag = jnp.abs(ffll)+jnp.abs(ff00)+jnp.abs(ffrr)

        weno = weno & (fdel > 1.e-6 * fmag)                             # :130

#     =============================== calc. WENO oscillation weighting
        scal = gad_osc_mul_y(iy, +2, mask,                              # :133-134
                             ohat, ix=ix)

        for ii in (+1, +2, +3, +4, +5):                                 # :136-140
#     =============================== blend limited/un-limited profile
            lhat[ii] = jnp.where(weno, scal[1] * uhat[ii]
                                 + scal[2] * lhat[ii], lhat[ii])
    else:
        raise ValueError(f"GAD_PQM_HAT_Y: method {method} is not a PQM limiter (50, 51, 52)")

    for ii in (+1, +2, +3, +4, +5):                                     # :149-152, :156-158
#     =============================== copy polynomial onto output data
        fhat[ii] = fhat[ii].at[ix, iy].set(jnp.where(wet, lhat[ii], 0.0))
    return fhat
