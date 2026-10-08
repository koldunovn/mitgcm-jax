"""pkg/generic_advdiff/gad_ppm_fun.F: routines to form the PPM grid-cell polynomial.

    o GAD_PPM_FUN_NULL   ported
    o GAD_PPM_FUN_MONO   ported

Point routines: the Fortran works on scalars, one grid cell per call; here every argument is an array of all the
cells of the caller's loop at once (each cell reads only its own arguments). Array arguments with a Fortran index
range (fhat(+1:+3), dfds(-1:+1)) are dicts keyed by the Fortran index.
"""

import jax.numpy as jnp

from mitjax.ops.safe import safe_div


def _profile(ff00, fell, ferr):
    """The three statements that form the PPM coefficients, written out at gad_ppm_fun.F:30-37 (FUN_NULL) and
    :107-114, :149-156 (FUN_MONO): one function so that the three copies cannot drift apart. `3/2`, `1/4`, `1/2`,
    `3/4` are D literals (exact in binary)."""
    fhat = {}
    fhat[1] = (+(3. / 2.) * ff00
               - (1. / 4.) * (ferr + fell))
    fhat[2] = (+(1. / 2.) * (ferr - fell))
    fhat[3] = (-(3. / 2.) * ff00
               + (3. / 4.) * (ferr + fell))
    return fhat


def gad_ppm_fun_null(ff00, fell, ferr):
    """GAD_PPM_FUN_NULL(ff00, fell, ferr, fhat, mono)   @63cdc0b pkg/generic_advdiff/gad_ppm_fun.F:10-42

    C     | PPM_FUN_NULL: form PPM grid-cell polynomial.                   |
    C     | Piecewise Parabolic Method (PPM), unlimited variant.           |
    I  ff00, fell, ferr
    O  fhat(+1:+3), mono

    Returns (fhat, mono), fhat = {1: fhat(1), 2: fhat(2), 3: fhat(3)}.
    """
    mono = jnp.zeros(jnp.broadcast_shapes(jnp.shape(ff00), jnp.shape(fell), jnp.shape(ferr)), jnp.int32)  # :27
    fhat = _profile(ff00, fell, ferr)                                   # :30-37
    return fhat, mono


def gad_ppm_fun_mono(ff00, ffll, ffrr, fell, ferr, dfds):
    """GAD_PPM_FUN_MONO(ff00, ffll, ffrr, fell, ferr, dfds, fhat, mono)   @63cdc0b pkg/generic_advdiff/gad_ppm_fun.F:46-163

    C     | PPM_FUN_MONO: form PPM grid-cell polynomial.                   |
    C     | Piecewise Parabolic Method (PPM) - monotonic variant.          |
    I  ff00, ffll, ffrr, fell, ferr, dfds(-1:+1)
    O  fhat(+1:+3), mono

    The routine also overwrites its arguments fell and ferr (Fortran passes them by reference): returns
    (fell, ferr, fhat, mono). Control flow on every lane at once: each IF (:88-104, :117-142, :146-158) becomes a
    `where` that keeps the prior value where the condition is false, in the Fortran's statement order; the early
    RETURN of the extremum test (:73-85) is applied last, by selecting the flattened profile (and the unchanged
    fell, ferr) on its lanes. The division of :119-120 runs only where the Fortran runs it (|fhat(3)| >
    |fhat(2)|*.5, so fhat(3) /= 0); elsewhere its divisor is replaced by 1 before dividing (safe_div) and `turn` is
    not used there. Limiter switches are differentiated as written (the derivative of the selected branch).
    """
    fell_in, ferr_in = fell, ferr
    mono = jnp.zeros(jnp.broadcast_shapes(jnp.shape(ff00), jnp.shape(ffll), jnp.shape(ffrr)), jnp.int32)  # :70

#     ============================================== "flatten" extrema
    flat = ((ffrr - ff00) *                                             # :73-74 (the RETURN, applied below)
            (ff00 - ffll) <= 0.)

#     ============================================== limit edge values
    lim = ((ffll - fell) *                                              # :88-95
           (fell - ff00) <= 0.)
    mono = jnp.where(lim, +1, mono)
    fell = jnp.where(lim, ff00 - dfds[0], fell)

    lim = ((ffrr - ferr) *                                              # :97-104
           (ferr - ff00) <= 0.)
    mono = jnp.where(lim, +1, mono)
    ferr = jnp.where(lim, ff00 + dfds[0], ferr)

#     ============================================== limit cell values
    fhat = _profile(ff00, fell, ferr)                                   # :107-114

    big = (jnp.abs(fhat[3]) >                                           # :116-117
           jnp.abs(fhat[2]) * .5)

    turn = safe_div(-0.5 * fhat[2],                                     # :119-120
                    fhat[3], big)

    lo = big & ((turn >= -1.)                                           # :122-131
                & (turn <= +0.))
    mono = jnp.where(lo, +2, mono)
#     ====================================== push TURN onto lower edge
    ferr = jnp.where(lo, +3. * ff00
                         - 2. * fell, ferr)

    hi = big & ((turn > +0.)                                            # :133-142
                & (turn <= +1.))
    mono = jnp.where(hi, +2, mono)
#     ====================================== push TURN onto upper edge
    fell = jnp.where(hi, +3. * ff00
                         - 2. * ferr, fell)

    redo = mono > +1                                                    # :146-158
#     ====================================== re-calc. coeff. on demand
    fnew = _profile(ff00, fell, ferr)
    fhat = {ii: jnp.where(redo, fnew[ii], fhat[ii]) for ii in (1, 2, 3)}

    # the RETURN of :83 on the extremum lanes (:76-81): flattened profile, fell and ferr untouched
    fhat = {1: jnp.where(flat, ff00, fhat[1]),
            2: jnp.where(flat, 0., fhat[2]),
            3: jnp.where(flat, 0., fhat[3])}
    mono = jnp.where(flat, +1, mono)
    fell = jnp.where(flat, fell_in, fell)
    ferr = jnp.where(flat, ferr_in, ferr)
    return fell, ferr, fhat, mono
