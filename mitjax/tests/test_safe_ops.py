"""mitjax/ops/safe.py (plan Task 7c): guarded division, sqrt, log, pow and the quotient-rule division.

Each test checks the contract (values bitwise equal to the plain jax.numpy operation on unmasked lanes, under jit with
the gate XLA flags; finite gradients everywhere, exactly 0 on masked lanes) and carries its negative control: the naive
forward-`where` version on the same input gives NaN gradients (measured, asserted), so the input really exercises the
backward 0*inf the guard exists for ([L-AD-1], [L-AD-3]). Seconds; smoke.
"""

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.ops import safe

# Lanes: ordinary values, a 0/0 lane, an x/0 lane, a negative lane, signed zeros, tiny and huge magnitudes.
NUM = np.array([1.5, 0.0, 2.0, -3.0, 7.0, -0.0, 1e-200, 1e300, 4.0, 0.25])
DEN = np.array([3.0, 0.0, 0.0, -0.5, 1e-3, 2.0, -1.2e-240, 10.0, 0.0, 1e-160])
# mask: where the "Fortran" computes the quotient (denominator nonzero); lanes 1, 2, 8 are masked
DMASK = DEN != 0.0

ARG = np.array([2.0, 0.0, -0.0, -1.0, 1e-300, 1e300, 0.5, 3.0])
AMASK = ARG > 0.0


def _bits(a):
    return np.asarray(a, np.float64).view(np.uint64)


def _grads(f, *args):
    """Gradient of sum(w * f(*args)) with a cotangent that is 0 on half the lanes (as downstream masks make it)."""
    w = np.where(np.arange(args[0].size) % 2 == 0, 1.0, 0.0)

    def loss(*a):
        return jnp.sum(w * f(*a))
    return jax.jit(jax.grad(loss, argnums=tuple(range(len(args)))))(*args)


def test_safe_div():
    """safe_div == num/den bitwise on unmasked lanes (incl. -0, tiny and huge), fill on masked lanes; finite
    gradients w.r.t. both arguments on every lane (0 on masked lanes). Negative control: the naive
    where(mask, num/den, 0) has NaN gradients on the 0/0 lane and on the tiny-denominator lane."""
    out = np.asarray(jax.jit(safe.safe_div)(NUM, DEN, DMASK))
    plain = np.asarray(jax.jit(lambda a, b: a / b)(NUM, DEN))
    assert np.array_equal(_bits(out[DMASK]), _bits(plain[DMASK]))
    assert np.all(out[~DMASK] == 0.0)
    gn, gd = _grads(lambda a, b: safe.safe_div(a, b, DMASK), NUM, DEN)
    assert np.all(np.isfinite(gn)) and np.all(np.isfinite(gd)), (gn, gd)
    assert np.all(np.asarray(gn)[~DMASK] == 0.0) and np.all(np.asarray(gd)[~DMASK] == 0.0)

    # negative control: guard after the operation (forward where only)
    nn, nd = _grads(lambda a, b: jnp.where(DMASK, a / b, 0.0), NUM, DEN)
    bad = ~np.isfinite(np.asarray(nd))
    assert bad[1] and bad[2] and bad[8], nd          # 0/0 and x/0 lanes: 0 * inf / NaN
    assert bad[6], nd                                # b = -1.2e-240, zero cotangent: b**-2 = inf -> NaN


def test_div_quotient_rule():
    """div(a, b): value bitwise a/b; derivative finite where JAX's rule (b**-2) underflows; equal to JAX's rule on
    ordinary lanes up to rounding. Negative control: plain division on the tiny lane with a zero cotangent is NaN."""
    a = np.array([1e-200, 2.0, -3.0, 1.0])
    b = np.array([-1.2e-240, 4.0, 0.7, 3e-155])
    assert np.array_equal(_bits(jax.jit(safe.div)(a, b)), _bits(jax.jit(lambda a, b: a / b)(a, b)))
    w = np.array([0.0, 1.0, 1.0, 0.0])
    g_q = jax.jit(jax.grad(lambda a, b: jnp.sum(w * safe.div(a, b)), argnums=(0, 1)))(a, b)
    g_p = jax.jit(jax.grad(lambda a, b: jnp.sum(w * (a / b)), argnums=(0, 1)))(a, b)
    assert all(np.all(np.isfinite(g)) for g in g_q), g_q
    np.testing.assert_allclose(np.asarray(g_q[0])[1:3], np.asarray(g_p[0])[1:3], rtol=1e-15)
    np.testing.assert_allclose(np.asarray(g_q[1])[1:3], np.asarray(g_p[1])[1:3], rtol=1e-15)
    # negative control: JAX's own rule is NaN on the zero-cotangent tiny lanes (0 and 3)
    assert np.isnan(np.asarray(g_p[1])[0]) and np.isnan(np.asarray(g_p[1])[3]), g_p


def test_safe_sqrt_log_pow():
    """sqrt/log/pow: bitwise the plain op on unmasked lanes, fill elsewhere, finite gradients on every lane incl. the
    zeros and negatives. Negative control: the naive forward-where versions give NaN gradients at x = 0 (and pow with
    p = 0.5 too)."""
    cases = {
        "sqrt": (lambda x, m: safe.safe_sqrt(x, m), jnp.sqrt),
        "log": (lambda x, m: safe.safe_log(x, m), jnp.log),
        "pow": (lambda x, m: safe.safe_pow(x, 0.5, m), lambda x: jnp.power(x, 0.5)),
        "pow15": (lambda x, m: safe.safe_pow(x, 1.5, m), lambda x: jnp.power(x, 1.5)),
    }
    for name, (sf, pf) in cases.items():
        out = np.asarray(jax.jit(sf)(ARG, AMASK))
        plain = np.asarray(jax.jit(pf)(ARG))
        assert np.array_equal(_bits(out[AMASK]), _bits(plain[AMASK])), name
        assert np.all(out[~AMASK] == 0.0), name
        (g,) = _grads(lambda x: sf(x, AMASK), ARG)
        g = np.asarray(g)
        assert np.all(np.isfinite(g)), (name, g)
        assert np.all(g[~AMASK] == 0.0), (name, g)
        (gn,) = _grads(lambda x: jnp.where(AMASK, pf(x), 0.0), ARG)
        gn = np.asarray(gn)
        if name != "pow15":   # x**1.5 has the finite derivative 0 at x = 0: no NaN to show
            assert np.isnan(gn[1]) or np.isnan(gn[2]), (name, gn)   # x = +0 / -0 lanes (cotangent 0 on lane 1)


def test_safe_div_traced_mask_under_jit():
    """The mask may be traced (computed from a traced hFac inside jit) and the fill a traced scalar."""
    f = jax.jit(lambda n, d, fill: safe.safe_div(n, d, d != 0.0, fill))
    out = np.asarray(f(NUM, DEN, -1.0))
    assert np.all(out[~DMASK] == -1.0)
    plain = np.asarray(jax.jit(lambda a, b: a / b)(NUM, DEN))
    assert np.array_equal(_bits(out[DMASK]), _bits(plain[DMASK]))
