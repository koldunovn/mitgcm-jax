"""pkg/generic_advdiff/gad_pqm_fun.F: routines to form the PQM grid-cell polynomial.

    o QUADROOT           ported (LOGICAL FUNCTION)
    o GAD_PQM_FUN_NULL   ported
    o GAD_PQM_FUN_MONO   ported

Point routines: the Fortran works on scalars, one grid cell per call; here every argument is an array of all the
cells of the caller's loop at once (each cell reads only its own arguments). Array arguments with a Fortran index
range (fhat(+1:+5), dfds(-1:+1), xx(1:2)) are dicts keyed by the Fortran index. Each IF becomes a `where` that keeps
the prior value where the condition is false, in the Fortran's statement order; divisions and the square root run
only where the Fortran runs them (their operands are replaced before the operation elsewhere, mitjax/ops/safe.py).
Limiter switches are differentiated as written (the derivative of the selected branch).
"""

import jax.numpy as jnp

from mitjax.ops.safe import safe_div, safe_sqrt


def quadroot(aa, bb, cc, xx=None):
    """LOGICAL FUNCTION QUADROOT(aa, bb, cc, xx)   @63cdc0b pkg/generic_advdiff/gad_pqm_fun.F:11-72

    C     | QUADROOT: find roots of quadratic ax**2 + bx + c = 0.          |
    _RL aa, bb, cc;  _RL xx(1:2)

    The function also overwrites its argument aa (:41, Fortran passes it by reference) and writes xx only where it
    returns .TRUE.: returns (QUADROOT, aa, xx) with xx = {1: xx(1), 2: xx(2)}; where QUADROOT is .FALSE., xx keeps
    the given prior (0 when xx is None: the caller's array is not initialised there and never read).
    A lane with sq = 0 exactly in the sqrt branch has the infinite derivative of sqrt at 0 (as written).
    """
    if xx is None:
        xx = {1: jnp.zeros_like(aa), 2: jnp.zeros_like(aa)}
    a0 = jnp.abs(aa)                                                    # :25-26
    b0 = jnp.abs(bb)

    sq = bb * bb - 4. * aa * cc                                         # :28

    quad = a0 > 0.                                                      # :30
    real = quad & (sq >= 0.)                                            # :32 (:34 QUADROOT = .TRUE.)

    sq = safe_sqrt(sq, real, sq)                                        # :36

    x1 = - bb + sq                                                      # :38-39
    x2 = - bb - sq

    aa = safe_div(0.5, aa, real, aa)                                    # :41 (a0 > 0: aa /= 0)

    x1 = x1 * aa                                                        # :43-44
    x2 = x2 * aa

    lin = (~quad) & (b0 > 0.)                                           # :54 (:56 QUADROOT = .TRUE.)

    xl = - safe_div(cc, bb, lin)                                        # :58-59

    QUADROOT = real | lin
    xx = {1: jnp.where(real, x1, jnp.where(lin, xl, xx[1])),
          2: jnp.where(real, x2, jnp.where(lin, xl, xx[2]))}
    return QUADROOT, aa, xx


def _profile(ff00, fell, ferr, dell, derr):
    """The five statements that form the PQM coefficients, written out at gad_pqm_fun.F:98-114 (FUN_NULL) and
    :210-226, :380-396 (FUN_MONO): one function so that the three copies cannot drift apart. All fractions are D
    literals (`30/16` etc. are exact in binary)."""
    fhat = {}
    fhat[1] = (+ (30. / 16.) * ff00
               - (7. / 16.) * (ferr + fell)
               + (1. / 16.) * (derr - dell))
    fhat[2] = (+ (3. / 4.) * (ferr - fell)
               - (1. / 4.) * (derr + dell))
    fhat[3] = (- (30. / 8.) * ff00
               + (15. / 8.) * (ferr + fell)
               - (3. / 8.) * (derr - dell))
    fhat[4] = (- (1. / 4.) * (ferr - fell - derr - dell))
    fhat[5] = (+ (30. / 16.) * ff00
               - (15. / 16.) * (ferr + fell)
               + (5. / 16.) * (derr - dell))
    return fhat


def gad_pqm_fun_null(ff00, fell, ferr, dell, derr):
    """GAD_PQM_FUN_NULL(ff00, fell, ferr, dell, derr, fhat, mono)   @63cdc0b pkg/generic_advdiff/gad_pqm_fun.F:76-119

    C     | PQM_FUN_NULL: form PQM grid-cell polynomial.                   |
    C     | Piecewise Quartic Method (PQM) - unlimited variant.            |
    I  ff00, fell, ferr, dell, derr
    O  fhat(+1:+5), mono

    Returns (fhat, mono), fhat = {1: fhat(1), ..., 5: fhat(5)}.
    """
    mono = jnp.zeros(jnp.broadcast_shapes(jnp.shape(ff00), jnp.shape(fell), jnp.shape(ferr)), jnp.int32)  # :95
    fhat = _profile(ff00, fell, ferr, dell, derr)                       # :98-114
    return fhat, mono


def gad_pqm_fun_mono(ff00, ffll, ffrr, fell, ferr, dell, derr, dfds):
    """GAD_PQM_FUN_MONO(ff00, ffll, ffrr, fell, ferr, dell, derr, dfds, fhat, mono)
    @63cdc0b pkg/generic_advdiff/gad_pqm_fun.F:123-403

    C     | PQM_FUN_MONO: form PQM grid-cell polynomial.                   |
    C     | Piecewise Quartic Method (PQM) - monotonic variant.            |
    I  ff00, ffll, ffrr, fell, ferr, dell, derr, dfds(-1:+1)
    O  fhat(+1:+5), mono

    The routine also overwrites its arguments fell, ferr, dell, derr (Fortran passes them by reference): returns
    (fell, ferr, dell, derr, fhat, mono). The early RETURN of the extremum test (:157-171) is applied last, by
    selecting the flattened profile (and the unchanged fell, ferr, dell, derr) on its lanes. `bind` changes only
    inside IF (QUADROOT(...)) (:233-375), so the two "pop" blocks (:291-331, :333-373) run only there.
    """
    fell_in, ferr_in, dell_in, derr_in = fell, ferr, dell, derr
    mono = jnp.zeros(jnp.broadcast_shapes(jnp.shape(ff00), jnp.shape(ffll), jnp.shape(ffrr)), jnp.int32)  # :154

#     ============================================== "flatten" extrema
    flat = ((ffrr - ff00) *                                             # :157-158 (the RETURN, applied below)
            (ff00 - ffll) <= 0.)

#     ============================================== limit edge values
    lim = ((ffll - fell) *                                              # :174-181
           (fell - ff00) <= 0.)
    mono = jnp.where(lim, +1, mono)
    fell = jnp.where(lim, ff00 - dfds[0], fell)

    lim = ((ffrr - ferr) *                                              # :183-190
           (ferr - ff00) <= 0.)
    mono = jnp.where(lim, +1, mono)
    ferr = jnp.where(lim, ff00 + dfds[0], ferr)

#     ============================================== limit edge slopes
    lim = (dell * dfds[-1]) < 0.                                        # :193-199
    mono = jnp.where(lim, +1, mono)
    dell = jnp.where(lim, dfds[-1], dell)

    lim = (derr * dfds[+1]) < 0.                                        # :201-207
    mono = jnp.where(lim, +1, mono)
    derr = jnp.where(lim, dfds[+1], derr)

#     ============================================== limit cell values
    fhat = _profile(ff00, fell, ferr, dell, derr)                       # :210-226

#     ============================= calc. inflexion via 2nd-derivative
    aval = 12. * fhat[5]                                                # :229-231
    bval = 6. * fhat[4]
    cval = 2. * fhat[3]

    root, aval, iflx = quadroot(aval, bval, cval)                       # :233

    bind = jnp.zeros_like(mono)                                         # :235

    inside = root & ((iflx[1] > -1.)                                    # :237-238
                     & (iflx[1] < +1.))
#     ============================= check for non-monotonic inflection
    dflx1 = (fhat[2]                                                    # :241-244
             + iflx[1] * fhat[3] * 2.
             + (iflx[1] ** 2) * fhat[4] * 3.
             + (iflx[1] ** 3) * fhat[5] * 4.)
    turn = inside & (dflx1 * dfds[+0] < 0.)                             # :246
    bind = jnp.where(turn, jnp.where(jnp.abs(dell)                      # :248-257
                                     < jnp.abs(derr), -1, +1), bind)

    inside = root & ((iflx[2] > -1.)                                    # :263-264
                     & (iflx[2] < +1.))
#     ============================= check for non-monotonic inflection
    dflx2 = (fhat[2]                                                    # :267-270
             + iflx[2] * fhat[3] * 2.
             + (iflx[2] ** 2) * fhat[4] * 3.
             + (iflx[2] ** 3) * fhat[5] * 4.)
    turn = inside & (dflx2 * dfds[+0] < 0.)                             # :272
    bind = jnp.where(turn, jnp.where(jnp.abs(dell)                      # :274-283
                                     < jnp.abs(derr), -1, +1), bind)

#     ============================= pop non-monotone inflexion to edge
    pop = bind == -1                                                    # :291
#     ============================= pop inflection points onto -1 edge
    mono = jnp.where(pop, +2, mono)                                     # :294
    derr = jnp.where(pop, (- (5. / 1.) * ff00                           # :296-299
                           + (3. / 1.) * ferr
                           + (2. / 1.) * fell), derr)
    dell = jnp.where(pop, (+ (5. / 3.) * ff00                           # :300-303
                           - (1. / 3.) * ferr
                           - (4. / 3.) * fell), dell)

    lim = pop & (dell * dfds[-1] < 0.)                                  # :305
    dell = jnp.where(lim, 0., dell)                                     # :307
    ferr = jnp.where(lim, (+ (5. / 1.) * ff00                           # :309-311
                           - (4. / 1.) * fell), ferr)
    derr = jnp.where(lim, (+ (10. / 1.) * ff00                          # :312-314
                           - (10. / 1.) * fell), derr)

    lim = pop & (derr * dfds[+1] < 0.)                                  # :318
    derr = jnp.where(lim, 0., derr)                                     # :320
    fell = jnp.where(lim, (+ (5. / 2.) * ff00                           # :322-324
                           - (3. / 2.) * ferr), fell)
    dell = jnp.where(lim, (- (5. / 3.) * ff00                           # :325-327
                           + (5. / 3.) * ferr), dell)

    pop = bind == +1                                                    # :333
#     ============================= pop inflection points onto +1 edge
    mono = jnp.where(pop, +2, mono)                                     # :336
    derr = jnp.where(pop, (- (5. / 3.) * ff00                           # :338-341
                           + (4. / 3.) * ferr
                           + (1. / 3.) * fell), derr)
    dell = jnp.where(pop, (+ (5. / 1.) * ff00                           # :342-345
                           - (2. / 1.) * ferr
                           - (3. / 1.) * fell), dell)

    lim = pop & (dell * dfds[-1] < 0.)                                  # :347
    dell = jnp.where(lim, 0., dell)                                     # :349
    ferr = jnp.where(lim, (+ (5. / 2.) * ff00                           # :351-353
                           - (3. / 2.) * fell), ferr)
    derr = jnp.where(lim, (+ (5. / 3.) * ff00                           # :354-356
                           - (5. / 3.) * fell), derr)

    lim = pop & (derr * dfds[+1] < 0.)                                  # :360
    derr = jnp.where(lim, 0., derr)                                     # :362
    fell = jnp.where(lim, (+ (5. / 1.) * ff00                           # :364-366
                           - (4. / 1.) * ferr), fell)
    dell = jnp.where(lim, (- (10. / 1.) * ff00                          # :367-369
                           + (10. / 1.) * ferr), dell)

#     ============================= re-assemble coefficients on demand
    redo = mono == +2                                                   # :378-398
    fnew = _profile(ff00, fell, ferr, dell, derr)
    fhat = {ii: jnp.where(redo, fnew[ii], fhat[ii]) for ii in (1, 2, 3, 4, 5)}

    # the RETURN of :169 on the extremum lanes (:160-167): flattened profile, fell..derr untouched
    fhat = {1: jnp.where(flat, ff00, fhat[1]),
            2: jnp.where(flat, 0., fhat[2]),
            3: jnp.where(flat, 0., fhat[3]),
            4: jnp.where(flat, 0., fhat[4]),
            5: jnp.where(flat, 0., fhat[5])}
    mono = jnp.where(flat, +1, mono)
    fell = jnp.where(flat, fell_in, fell)
    ferr = jnp.where(flat, ferr_in, ferr)
    dell = jnp.where(flat, dell_in, dell)
    derr = jnp.where(flat, derr_in, derr)
    return fell, ferr, dell, derr, fhat, mono
