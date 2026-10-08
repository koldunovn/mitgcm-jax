"""pkg/generic_advdiff/gad_osc_mul_r.F: WENO oscillation weights in R (GAD_OSC_MUL_R).

Column routine (see gad_osc_hat_r.py): the column arrays carry the column indices and `ix, iy` are the caller's
column loops.
"""

import jax.numpy as jnp


def gad_osc_mul_r(ir, hh, mask, ohat, *, ix, iy):
    """GAD_OSC_MUL_R(ir, hh, mask, ohat, scal)   @63cdc0b pkg/generic_advdiff/gad_osc_mul_r.F:3-78

    C     | OSC_MUL_R: evaluate WENO oscillation weights in R.             |

    `ir` is the caller's (k-vectorised) level loop, `hh` a static int, `ix, iy` (keywords; not Fortran arguments) the
    caller's column loops. Returns scal = {1: scal(1), 2: scal(2)}. As GAD_OSC_MUL_X (gad_osc_mul_x.py): the
    stencil loop (:33-53) in Python in the Fortran order, the min/max IFs as `where`s, integer powers as
    lax.integer_pow.
    """
#     =============================== calc. WENO oscillation weighting
    omin = +1.e99                                                       # :27-28
    omax = -1.e99

    zero = 1.e-20                                                       # :30-31
    mval = 1.e+0

    for m in range(-hh, hh+1):                                          # :33  do ii = ir-hh, ir+hh
        ii = ir + m

#     =============================== calc. derivatives centred on II.
        dels = m * 2.                                                   # :36  (ii - ir) * 2. _d 0

        dfs1 = ohat[1][ix, iy, ii]                                      # :38-39
        dfs2 = ohat[2][ix, iy, ii]

        dfs1 = dfs1 + dfs2 * dels                                       # :41

#     =============================== oscl. = NORM(H^N * D^N/DR^N(F)).
        oval = ((2. * dfs1)**2                                          # :44-45
                + (4. * dfs2)**2)

        omin = jnp.where(oval < omin, oval, omin)                       # :47
        omax = jnp.where(oval > omax, oval, omax)                       # :48

#     =============================== any mask across oscil. stencil
        mval = mval * mask[ix, iy, ii]                                  # :51

    weno = mval > 0.                                                    # :55

#     =============================== calc. WENO-style profile weights
    scal1 = (1.e5                                                       # :58-61
             / (omax + zero)**3)
    scal2 = (1.e0
             / (omin + zero)**3)

    osum = scal1 + scal2                                                # :63-65
    scal1 = scal1 / osum
    scal2 = scal2 / osum

#     =============================== default to MONO. profile weights
    scal = {1: jnp.where(weno, scal1, 0.),                              # :70-71
            2: jnp.where(weno, scal2, 1.)}
    return scal
