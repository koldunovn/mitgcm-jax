"""mitjax/ops/fortran_minmax.py: Fortran MAX/MIN with gfortran's values (GAD-A lane).

The oracle is the probe run named by reference/replay_gad_a/CURRENT_MINMAX_PROBE (relative to $MJX_REFERENCE; job
27828773: reference/replay_gad_a/probe/minmax_probe.F compiled with gfortran 11.2.0 -O0 -ffp-contract=off): REAL*8
MAX(a, b) and MIN(a, b) for a, b in {+0, -0, 1, -1, x} with x = +0 and x = NaN, as bit patterns. The probe's
statement `r = MAX(v(i), v(j))` keeps the second argument on ties and NaN (p="b"); other sites keep the first
(mitjax/tests/test_minmax_sites.py). Fails (not skips) without the probe output. Seconds; numpy + jax on CPU.
"""

import functools

import numpy as np

import jax
import jax.numpy as jnp

from mitjax import paths
from mitjax.ops.fortran_minmax import MAX, MIN
from mitjax.tests.gad_a_replay import REPO


def _probe():
    rel = (REPO / "reference" / "replay_gad_a" / "CURRENT_MINMAX_PROBE").read_text().split()[0]
    rows = [ln.split() for ln in (paths.REFERENCE / rel).read_text().splitlines()]
    assert len(rows) == 100
    ops = np.array([r[0] for r in rows])
    bits = np.array([[int(x, 16) for x in r[1:]] for r in rows], dtype=np.uint64).view(np.float64)
    return ops, bits[:, 0], bits[:, 1], bits[:, 2]


def _bits(x):
    return np.asarray(x, np.float64).view(np.int64)


def test_fortran_minmax_values_match_gfortran():
    """MAX/MIN equal gfortran bit for bit on every probed pair (ties of +-0 and NaN in either position), eagerly and
    under jit; jnp.maximum/jnp.minimum do not (negative control: they differ on the +-0 ties and on NaN)."""
    ops, a, b, want = _probe()
    for op, f, ref in (("MAX", functools.partial(MAX, p="b"), jnp.maximum),
                       ("MIN", functools.partial(MIN, p="b"), jnp.minimum)):
        s = ops == op
        aa, bb = jnp.asarray(a[s]), jnp.asarray(b[s])
        assert np.array_equal(_bits(f(aa, bb)), _bits(want[s])), op
        assert np.array_equal(_bits(jax.jit(f)(aa, bb)), _bits(want[s])), op
        assert np.sum(_bits(jax.jit(ref)(aa, bb)) != _bits(want[s])) > 0, op
        flipped = functools.partial(MAX if op == "MAX" else MIN, p="a")      # negative control: the other winner
        assert np.sum(_bits(jax.jit(flipped)(aa, bb)) != _bits(want[s])) > 0, op
    # a Python float first argument, as the kernels write MAX(0., x)
    x = jnp.asarray([-0.0, 0.0, -1.0, 2.0])
    assert np.array_equal(_bits(jax.jit(lambda v: MAX(0., v, p="b"))(x)), _bits([-0.0, 0.0, 0.0, 2.0]))
    assert np.array_equal(_bits(jax.jit(lambda v: MAX(0., v, p="a"))(x)), _bits([0.0, 0.0, 0.0, 2.0]))


def test_fortran_minmax_derivative_is_jax_own():
    """The derivative is jnp.maximum's / jnp.minimum's (JAX's own, [L-CONF-12]): the selected argument's tangent
    away from a tie, half of each at an exact tie; reverse mode agrees with forward mode."""
    a = jnp.asarray([1.0, 0.0, -2.0, 0.0])
    b = jnp.asarray([0.5, 0.0, 3.0, -0.0])
    da = jnp.asarray([1.0, 1.0, 1.0, 1.0])
    db = jnp.asarray([10.0, 10.0, 10.0, 10.0])
    for f, ref in ((functools.partial(MAX, p="a"), jnp.maximum), (functools.partial(MAX, p="b"), jnp.maximum),
                   (functools.partial(MIN, p="a"), jnp.minimum), (functools.partial(MIN, p="b"), jnp.minimum)):
        t = jax.jvp(f, (a, b), (da, db))[1]
        assert np.array_equal(np.asarray(t), np.asarray(jax.jvp(ref, (a, b), (da, db))[1]))
        ga, gb = jax.grad(lambda x, y: jnp.sum(f(x, y) * jnp.arange(1.0, 5.0)), argnums=(0, 1))(a, b)
        ra, rb = jax.grad(lambda x, y: jnp.sum(ref(x, y) * jnp.arange(1.0, 5.0)), argnums=(0, 1))(a, b)
        assert np.array_equal(np.asarray(ga), np.asarray(ra)) and np.array_equal(np.asarray(gb), np.asarray(rb))
    for p in ("a", "b"):
        assert np.allclose(np.asarray(jax.jvp(functools.partial(MAX, p=p), (a, b), (da, db))[1]), [1.0, 5.5, 10.0, 5.5])
