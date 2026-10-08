"""pkg/generic_advdiff/gad_osc_mul_x.F: WENO oscillation weights in X (GAD_OSC_MUL_X).

Row routine (see gad_osc_hat_x.py): the row arrays carry the row index j and `iy` is the caller's row loop.
"""

import jax.numpy as jnp


def gad_osc_mul_x(ix, hh, mask, ohat, *, iy):
    """GAD_OSC_MUL_X(ix, hh, mask, ohat, scal)   @63cdc0b pkg/generic_advdiff/gad_osc_mul_x.F:3-78

    C     | OSC_MUL_X: evaluate WENO oscillation weights in X.             |

    `ix` is the caller's point loop (all its points at once: each reads only inputs), `hh` a static int, `iy`
    (keyword; not a Fortran argument) the caller's row loop. Returns scal = {1: scal(1), 2: scal(2)}. The stencil
    loop `do ii = ix-hh, ix+hh` (:33-53) runs in Python in the Fortran order (omin, omax and mval are recurrences);
    `if (oval.lt.omin) omin = oval` is a `where` (the same selection, also at ties). `(...)**2` and `(...)**3` are
    integer powers (lax.integer_pow: x*x, and x*(x*x) as libgcc's __powidf2). All literals are D literals.
    """
#     =============================== calc. WENO oscillation weighting
    omin = +1.e99                                                       # :27-28
    omax = -1.e99

    zero = 1.e-20                                                       # :30-31
    mval = 1.e+0

    for m in range(-hh, hh+1):                                          # :33  do ii = ix-hh, ix+hh
        ii = ix + m

#     =============================== calc. derivatives centred on II.
        dels = m * 2.                                                   # :36  (ii - ix) * 2. _d 0

        dfs1 = ohat[1][ii, iy]                                          # :38-39
        dfs2 = ohat[2][ii, iy]

        dfs1 = dfs1 + dfs2 * dels                                       # :41

#     =============================== oscl. = NORM(H^N * D^N/DX^N(F)).
        oval = ((2. * dfs1)**2                                          # :44-45
                + (4. * dfs2)**2)

        omin = jnp.where(oval < omin, oval, omin)                       # :47
        omax = jnp.where(oval > omax, oval, omax)                       # :48

#     =============================== any mask across oscil. stencil
        mval = mval * mask[ii, iy]                                      # :51

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
