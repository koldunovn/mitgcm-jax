"""Guarded elementary operations: finite derivatives on masked lanes, the plain operation's value elsewhere.

Why (docs/LESSONS_CARRIED.md [L-AD-1], [L-AD-3]; plan Task 7c):
- A forward `jnp.where(mask, a / b, 0.0)` selects the right VALUE on a masked lane, but the backward pass still
  multiplies the zero cotangent of the discarded branch by the derivative of `a / b` there, and 0 * inf = NaN (b = 0)
  or 0 * NaN = NaN (0/0). The guard therefore goes BEFORE the operation: the masked lane gets a harmless operand (1.0),
  so the discarded branch has a finite derivative. Measured in env `mitjax` (jax 0.10.1): `grad` of
  `where(x > 0, sqrt(x), 0)` and of `where(x > 0, log(x), 0)` at x = 0 is NaN; the guarded versions give 0.
- JAX's derivative of `a / b` with respect to b is `-(da) * a * b**-2` (`integer_pow[y=-2]` in the jaxpr). For
  |b| < ~1e-154, b**2 underflows to 0 and b**-2 is inf; a zero cotangent then gives 0 * inf = NaN. In the ECCO port one
  snow thickness of -1.2e-240 made 197 lanes of d/d(uIce) NaN through stacked fields. `div` keeps the IEEE value a / b
  and writes the derivative as the quotient rule (da - q db) / b with q = a / b, which stays finite. Measured here:
  b = -1.2e-240, zero cotangent: jnp division gives NaN, `div` gives 0.

Contract of every guarded op `op(x, mask, fill)`:
- on lanes where `mask` is True the value is bitwise the value of the plain jax.numpy operation (same operands, same
  single IEEE operation; nothing is rewritten), NaN and signed zeros included;
- on lanes where `mask` is False the value is `fill` (default 0.0) and the derivative with respect to every input is
  exactly 0;
- the derivative on mask-True lanes is the plain operation's derivative (with `div`'s quotient rule for divisions).
The mask states where the Fortran computes the expression (e.g. `hFacC .NE. 0`, `maskC .EQ. 1`); a guard never changes
a value the Fortran computes. Where the Fortran itself guards (`IF (x .NE. 0.) y = 1/x`), the kernel writes that guard
as the mask.

These helpers are allowed in physics code (`mitjax/model`, `mitjax/pkg`); only this module uses `jax.custom_jvp`.
"""

import jax
import jax.numpy as jnp

__all__ = ["div", "safe_div", "safe_sqrt", "safe_log", "safe_pow"]


@jax.custom_jvp
def div(a, b):
    """a / b: the IEEE division (value bitwise jnp's), derivative by the quotient rule (da - (a/b) db) / b.

    Use it wherever a denominator can be tiny but nonzero on a lane whose cotangent may be zero (JAX's own rule forms
    b**-2, which underflows below |b| ~ 1e-154)."""
    return a / b


@div.defjvp
def _div_jvp(primals, tangents):
    a, b = primals
    da, db = tangents
    q = a / b
    return q, (da - q * db) / b


def _guard(x, mask, ok):
    """x on mask lanes, the harmless operand `ok` elsewhere (the guard before the operation)."""
    return jnp.where(mask, x, ok)


def safe_div(num, den, mask, fill=0.0):
    """num / den on `mask` lanes (quotient-rule derivative, `div`), `fill` elsewhere; finite derivatives everywhere.

    The denominator is replaced by 1.0 on masked lanes before dividing, so 0/0 and x/0 never occur there."""
    q = div(num, _guard(den, mask, 1.0))
    return jnp.where(mask, q, fill)


def safe_sqrt(x, mask, fill=0.0):
    """sqrt(x) on `mask` lanes, `fill` elsewhere. The derivative 1/(2 sqrt(x)) is infinite at x = 0, so a lane that
    may hold 0 must be masked out (e.g. `mask = x > 0`, which returns `fill` = +0 at x = +-0)."""
    y = jnp.sqrt(_guard(x, mask, 1.0))
    return jnp.where(mask, y, fill)


def safe_log(x, mask, fill=0.0):
    """log(x) on `mask` lanes, `fill` elsewhere (log(0) = -inf and its derivative 1/0 never reach the backward pass)."""
    y = jnp.log(_guard(x, mask, 1.0))
    return jnp.where(mask, y, fill)


def safe_pow(x, p, mask, fill=0.0):
    """x**p (real exponent p, the Fortran `x**p` with a REAL p) on `mask` lanes, `fill` elsewhere. For p < 1 the
    derivative p x**(p-1) is infinite at x = 0, and a traced p differentiates through log(x); both are guarded."""
    y = jnp.power(_guard(x, mask, 1.0), p)
    return jnp.where(mask, y, fill)
