"""The LSR sweep loop of an AD build (plan decision 14, option A1; lane M4ADLAB session 4).

pkg/seaice/seaice_lsr.F:782-990 runs `DO m = 1, linearIterLoc` and does work only while doIterate4u .OR. doIterate4v
(:796); with SEAICE_LSR_ADJOINT_ITER (lab_sea/code_ad/SEAICE_OPTIONS.h:180) TAF tapes every executed sweep under the
key lnkey = (nlkey-1)*SOLV_MAX_FIXED + MIN(m,SOLV_MAX_FIXED) (:785-789, :804-808; SEAICE_SIZE.h:37 SOLV_MAX_FIXED =
500) and its backward pass replays the executed sweeps in reverse with the stored convergence flags: the exact
transpose of the sweeps the forward executed (seaice_lsr.F at 63cdc0b, the lines above). Decision 14 gives
the port the same derivative: the loop is a fixed-length scan of SOLV_MAX_FIXED steps whose step applies the sweep
only while the Fortran loop would (`lax.cond` on the loop condition: the flags are the forward's own, so the
backward pass follows them), each step rematerialised (`jax.checkpoint`: the tape holds the scan carry per step, as
TAF's per-sweep store of uIce, vIce, WFAU/WFAV and the flags). The values are the while_loop's bit for bit (the
same body on the same carry; the skipped steps return the carry unchanged); the loop bound linearIterLoc <=
SOLV_MAX_FIXED is SEAICE_CHECK's (seaice_check.F:518-526 under ALLOW_AUTODIFF_TAMC).

The transforms live here (mitjax/ad/), not in the physics module (test_banned_transforms.py); seaice_lsr.py calls
`taped_sweeps` in place of `lax.while_loop` only for a build that defines SEAICE_LSR_ADJOINT_ITER.
"""

import jax
from jax import lax

SOLV_MAX_FIXED = 500                    # pkg/seaice/SEAICE_SIZE.h:36-37 PARAMETER ( SOLV_MAX_FIXED=500 )


def taped_sweeps(cond, body, init, n=SOLV_MAX_FIXED):
    """`lax.while_loop(cond, body, init)` as a scan of `n` steps (`DO m = 1, ...` with the tape length
    SOLV_MAX_FIXED): step = cond(c) ? body(c) : c, rematerialised per step. Exact (bitwise) as long as the
    while_loop would stop within n steps, which SEAICE_CHECK guarantees for the LSR (linearIterLoc <= n)."""
    def step(c, _):
        return lax.cond(cond(c), body, lambda c_: c_, c), None
    out, _ = lax.scan(jax.checkpoint(step), init, None, length=n)
    return out
