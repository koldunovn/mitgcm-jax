"""mitjax.ops.libm.glibc_asin against the oracle's glibc asin (ggl90_idemix.F:587, :595; M3 sub-lane GGL90).

The reference is the libm the oracle binary loads (ctypes; test_libm.oracle_library). Bit for bit on a dense sweep of
[2^-26, 2^-3) (uniform and log-uniform, both signs; every stage of the path taken, stage 3 included), on |x| < 2^-26,
on the IDEMIX arguments 1/MAX(3,fxa) and 1/MAX(1.01,fxa) of the dumped input.idemix states and on the dry-column
constants 1/3 and 1/1.01 (the jnp.arcsin FALLBACK there is measured equal); the constants equal the bytes of the
oracle's libm. Negative controls: XLA's arcsin and stage 1 always accepted both differ. Seconds (tier1x).
"""

import ctypes

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.ops import libm
from mitjax.tests.test_libm import oracle_library


def _ref():
    m = ctypes.CDLL(oracle_library("libm.so.6"))
    m.asin.restype, m.asin.argtypes = ctypes.c_double, [ctypes.c_double]
    return np.vectorize(m.asin, otypes=[np.float64])


def _ndiff(got, want):
    return int(np.count_nonzero(np.asarray(got, np.float64).view(np.int64) != np.asarray(want).view(np.int64)))


def _sweep(n=200000, seed=7):
    rng = np.random.default_rng(seed)
    lo, hi = 2.0**-26, 0.125
    x = np.concatenate([rng.uniform(lo, hi, n), 10**rng.uniform(np.log10(lo), np.log10(hi), n),
                        rng.uniform(0.09, 0.11, n // 2)])
    x = np.concatenate([x, -x[: n // 2]])
    return x[np.abs(x) < hi]


def test_constants_equal_oracle_libm():
    assert libm.glibc_asin_constants_vs_libm(oracle_library("libm.so.6")) == []


def test_glibc_asin_bitwise_on_the_idemix_range():
    ref = _ref()
    x = _sweep()
    f = jax.jit(libm.glibc_asin)
    st = np.asarray(jax.jit(libm.glibc_asin_stage)(x))
    counts = {s: int(np.count_nonzero(st == s)) for s in range(6)}
    print("stages", counts)
    assert counts[1] > 0 and counts[2] > 0 and counts[3] > 0 and counts[4] == 0 and counts[5] == 0
    assert _ndiff(f(x), ref(x)) == 0
    tiny = np.concatenate([10**np.random.default_rng(1).uniform(-300, np.log10(2.0**-26), 20000), [0.0, -0.0]])
    assert _ndiff(f(tiny), ref(tiny)) == 0
    c = np.array([1.0/3.0, 1.0/1.01, 1.0/3.0000000000000004, 1.0/1.0100000000000002])
    assert _ndiff(f(c), ref(c)) == 0
    # negative controls: XLA's arcsin; stage 1 always accepted (r1 = 0: the later stages dropped). A 1-ulp change of
    # a stage-1 coefficient does NOT bite (measured): the acceptance tests reject the perturbed result and the next
    # stage recomputes it (Ziv's strategy), so the planted error removes the acceptance test instead.
    assert _ndiff(jax.jit(jnp.arcsin)(x), ref(x)) > 1000
    keep = libm._A["r1"]
    try:
        libm._A["r1"] = 0.0
        assert _ndiff(libm._glibc_asin_impl(jnp.asarray(x[:40000])), ref(x[:40000])) > 0
    finally:
        libm._A["r1"] = keep


def test_glibc_asin_on_idemix_dump_arguments():
    """The arguments IDEMIX_gofx2 / IDEMIX_hofx1 take at the dumped input.idemix states (fxa from P02 sigmaR as
    ggl90_idemix.F:164-202 forms it, on the host in numpy: the same IEEE operations)."""
    from mitjax.tests import ggl90_gate as gg
    ref = _ref()
    ds = gg.dumpset("global_ocean.90x40x15", "input.idemix")
    g = gg.replay("global_ocean.90x40x15").g
    xs = []
    for it in ds.iterations():
        sig = ds.field(it, "P02_rho_sigma_ivdc", "sigmaR")
        mC = ds.field(it, "G00_geometry", "maskC")
        f = ds.field(it, "G00_geometry", "fCori")
        fxb = np.maximum(1e-6, np.abs(f))                       # MINMAX-RAW: test-side numpy (finite, no tie)
        n2 = g["gravity"]*float(ds.field(it, "G00_geometry", "gravitySign")[0, 0, 0, 0])*g["recip_rhoConst"]*sig[:, 1:]
        n2 = np.maximum(100.*fxb*fxb, n2)*mC[:, 1:]*mC[:, :-1]  # MINMAX-RAW: test-side numpy
        fxa = np.sqrt(n2)/fxb
        xs += [1.0/np.maximum(3.0, fxa), 1.0/np.maximum(1.01, fxa)]     # MINMAX-RAW: test-side numpy
    x = np.concatenate([a.ravel() for a in xs])
    assert _ndiff(jax.jit(libm.glibc_asin)(x), ref(x)) == 0
    assert np.count_nonzero(np.abs(x) < 0.125) > 10000
