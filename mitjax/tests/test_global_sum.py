"""mitjax/eesupp/global_sum.py (plan Task 7b): Fortran-order global sums against gfortran.

Reference: scripts/global_sum_ref/gsumref.F (the per-tile `DO j; DO i` partials of cg2d.F:157, 161-175 and the tile sum
of global_sum_tile.F:199-204), compiled with the oracle's compiler and flags (-O0 -ffp-contract=off ...) and run by
job GSUM_JOB on the six M1 tile shapes plus an all -0 case: random signs, magnitudes 1e-8..1e8 (the order of the
additions changes the result), 15 % +0 and 10 % -0. Data: $MJX_REFERENCE/exch_maps/global_sum_ref-job<ID>/ (fails, not
skips, when missing). Sharded sums (P=4 == P=1): test_sharded_exchange.py.
"""

import hashlib
import importlib.util

import jax
import jax.numpy as jnp
import numpy as np

from mitjax import paths
from mitjax.eesupp import global_sum as gs

GSUM_JOB = "27827321"
REF = paths.REFERENCE / "exch_maps" / f"global_sum_ref-job{GSUM_JOB}"


def _load_make_input():
    spec = importlib.util.spec_from_file_location("_gsumref_make_input", REF / "make_input.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bits(x):
    return np.asarray(x, np.float64).view(np.int64)


def _py_partials(a):
    """Python float loop in the Fortran order (IEEE double adds, one at a time): the second reference."""
    out = []
    for t in range(a.shape[0]):
        s = 0.0
        for j in range(a.shape[1]):
            for i in range(a.shape[2]):
                s = s + float(a[t, j, i])
        out.append(s)
    return np.array(out)


def test_global_sum_equals_fortran():
    """tile_sum_fortran and global_sum_tile (jit, gate XLA flags) == gfortran bitwise (sign of zero included) for the
    plain sums and the sums of products; a Python float loop in the same order agrees too. Negative controls on the
    same data: jnp.sum per tile, the tiles in reverse order and the loops swapped (i outer) each differ somewhere."""
    for line in (REF / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((REF / name).read_bytes()).hexdigest() == digest, name
    mk = _load_make_input()
    cases = mk.cases()
    assert mk.encode(cases) == (REF / "input.bin").read_bytes()      # the seeded data is what gfortran read
    ref = mk.decode_output((REF / "output.bin").read_bytes(), cases)

    part_fn = jax.jit(gs.tile_sum_fortran)
    sum_fn = jax.jit(gs.global_sum_tile)
    prod_fn = jax.jit(lambda a, b: gs.tile_sum_fortran(a * b))
    differs = {"jnp.sum": 0, "reverse tiles": 0, "i outer": 0}
    for (a, b), (pA, sA, pAB, sAB) in zip(cases, ref):
        assert np.array_equal(_bits(part_fn(a)), _bits(pA)), a.shape
        assert np.array_equal(_bits(sum_fn(jnp.asarray(pA))), _bits(sA)), a.shape
        assert np.array_equal(_bits(prod_fn(a, b)), _bits(pAB)), a.shape
        assert np.array_equal(_bits(sum_fn(jnp.asarray(pAB))), _bits(sAB)), a.shape
        assert np.array_equal(_bits(_py_partials(a)), _bits(pA))
        assert np.array_equal(_bits(gs.global_sum_rl(jnp.asarray(a))), _bits(sA))
        # negative controls (counted over the cases)
        differs["jnp.sum"] += not np.array_equal(_bits(jax.jit(lambda x: jnp.sum(x, axis=(1, 2)))(a)), _bits(pA))
        differs["reverse tiles"] += not np.array_equal(_bits(sum_fn(jnp.asarray(pA[::-1]))), _bits(sA))
        differs["i outer"] += not np.array_equal(_bits(part_fn(np.swapaxes(a, 1, 2))), _bits(pA))
    print(f"order controls: cases (of {len(cases)}) where the wrong order differs from gfortran: {differs}")
    assert all(v > 0 for v in differs.values()), differs
    # the all -0 case: partials and sum are +0 (0. + -0. = +0.), as in the Fortran; also for -0 partials given
    # directly (XLA folds a constant `0. +` away, global_sum._zero_plus), and the first addition keeps derivative 1
    assert np.all(_bits(ref[-1][0]) == 0) and _bits(ref[-1][1]) == 0
    for p in ([-0.0], [-0.0, -0.0], [-0.0, 1.5]):
        assert _bits(sum_fn(jnp.asarray(p))) == _bits(0.0 + sum(p)), p
    assert np.array_equal(np.asarray(jax.grad(lambda p: gs.global_sum_tile(p))(jnp.zeros(3))), np.ones(3))
    assert np.array_equal(np.asarray(jax.grad(lambda a: gs.global_sum_rl(a))(jnp.zeros((2, 2, 3)))), np.ones((2, 2, 3)))
