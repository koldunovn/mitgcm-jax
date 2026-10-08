"""mitjax/ops/libm.py (plan Task 7c) against the master oracle's own math libraries, on sampled arguments.

The reference is the library the oracle binary loads: `libm.so.6` and `libgcc_s.so.1` as recorded in the oracle's
`ldd.txt` ($MJX_REFERENCE/bin/<exp>-<code>-63cdc0b-*/), called through ctypes element by element (Python's math module
also calls glibc, but numpy's SIMD trig is not glibc [E§5], so no numpy reference). The JAX side runs jitted under the
gate XLA flags (conftest.py). The oracle files are required: a missing oracle fails the test (no skip, [L-CONF-3]).

Gates:
- glibc_exp == glibc exp bitwise on every sampled range (R's transcription re-verified on this glibc), and its tables
  and constants equal the bytes of the oracle's libm; glibc_tanh == glibc tanh bitwise;
- powi == libgcc __powidf2 bitwise (and == x ** n, lax.integer_pow);
- for every libm function the oracle imports, the measured number of mismatches of XLA's function equals what
  libm.MEASURED records as bitwise (0) or not (> 0); M1_LIBM uses only bitwise functions;
- negative controls: jnp.exp in place of glibc_exp and jnp.tanh in place of glibc_tanh fail (asserted).
Run as a script for the table: `python mitjax/tests/test_libm.py` (sets the gate flags itself).
"""

import ctypes
import math
import sys

import numpy as np

if __name__ == "__main__":
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from mitjax.xla_flags import set_gate_xla_flags
    set_gate_xla_flags()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import mitjax  # noqa: E402,F401  (x64)
from mitjax import paths  # noqa: E402
from mitjax.ops import libm  # noqa: E402

PI = math.pi


def oracle_library(soname):
    """Path of `soname` as the master oracle binary resolved it (its ldd.txt)."""
    bins = sorted((paths.REFERENCE / "bin").glob("*-63cdc0b-*"))
    if not bins:
        raise FileNotFoundError(f"no master oracle binaries under {paths.REFERENCE / 'bin'}")
    for b in bins:
        for line in (b / "ldd.txt").read_text().splitlines():
            parts = line.split()
            if parts and parts[0] == soname and "=>" in parts:
                return parts[parts.index("=>") + 1]
    raise FileNotFoundError(f"{soname} not in the oracle's ldd.txt")


def _cfun(lib, name, nargs, argtypes=None):
    f = getattr(lib, name)
    f.restype = ctypes.c_double
    f.argtypes = argtypes or [ctypes.c_double] * nargs
    return f


def glibc_reference():
    """{name: elementwise numpy function calling the oracle's libm (or libgcc)}."""
    m = ctypes.CDLL(oracle_library("libm.so.6"))
    g = ctypes.CDLL(oracle_library("libgcc_s.so.1"))
    out = {}
    for name in ("exp", "log", "sin", "cos", "tan", "atan", "asin", "tanh", "sqrt"):
        f = _cfun(m, name, 1)
        out[name] = (lambda f: lambda x: np.array([f(v) for v in x.tolist()]))(f)
    for name in ("pow", "atan2", "fmod"):
        f = _cfun(m, name, 2)
        out[name] = (lambda f: lambda x, y: np.array([f(a, b) for a, b in zip(x.tolist(), y.tolist())]))(f)
    p = _cfun(g, "__powidf2", 2, [ctypes.c_double, ctypes.c_int])
    out["__powidf2"] = lambda x, n: np.array([p(v, int(n)) for v in x.tolist()])
    return out


def samples(n=20000, seed=0):
    """{function: [(range label, args...)]}: the M1 argument ranges (libm.py docstring) plus generic ones."""
    rng = np.random.default_rng(seed)
    U = lambda a, b: rng.uniform(a, b, n)                                    # noqa: E731
    LU = lambda a, b: np.exp(rng.uniform(np.log(a), np.log(b), n)) * rng.choice([-1.0, 1.0], n)  # noqa: E731
    lat = np.arange(-90.0, 90.0001, 0.25) * (PI / 180.0)                   # grid latitudes, deg2rad = pi/180
    trig = [("lat*deg2rad", lat), ("[-pi/2-0.2, pi/2+0.2]", U(-PI / 2 - 0.2, PI / 2 + 0.2)),
            ("[-2pi, 2pi]", U(-2 * PI, 2 * PI)), ("[-1e3, 1e3]", U(-1e3, 1e3)), ("|x| in [1e-8, 1]", LU(1e-8, 1.0))]
    return {
        "sin": trig, "cos": trig,
        "tan": [("lat*deg2rad, |lat|<90", lat[1:-1]), ("[-1.55, 1.55]", U(-1.55, 1.55)), ("[-1e3, 1e3]", U(-1e3, 1e3))],
        "tanh": [("dm95 [-6, 4]", U(-6.0, 4.0)), ("[-30, 30]", U(-30.0, 30.0)), ("|x| in [1e-10, 1]", LU(1e-10, 1.0))],
        "exp": [("advect_xy [-50, 0]", U(-50.0, 0.0)), ("[-708, 708]", U(-708.0, 708.0)),
                ("|x| in [1e-12, 1]", LU(1e-12, 1.0))],
        "log": [("x in [1e-300, 1e300]", np.abs(LU(1e-300, 1e300))), ("[0.5, 2]", U(0.5, 2.0))],
        "atan": [("[-1e3, 1e3]", U(-1e3, 1e3)), ("|x| in [1e-8, 1e8]", LU(1e-8, 1e8))],
        "asin": [("[-1, 1]", U(-1.0, 1.0))],
        "sqrt": [("[0, 1e10]", U(0.0, 1e10)), ("x in [1e-300, 1e300]", np.abs(LU(1e-300, 1e300)))],
        "pow": [("cosFac: |cos|**0", np.abs(np.cos(U(-PI / 2, PI / 2))), np.zeros(n)),
                ("x in [1e-3, 1e3], y in [-3, 3]", np.abs(LU(1e-3, 1e3)), U(-3.0, 3.0))],
        "atan2": [("pairs [-1e3, 1e3]^2", U(-1e3, 1e3), U(-1e3, 1e3))],
        "fmod": [("x in [-1e9, 1e9], y in [1, 1e6]", U(-1e9, 1e9), U(1.0, 1e6))],
    }


JNP = {"exp": jnp.exp, "log": jnp.log, "sin": jnp.sin, "cos": jnp.cos, "tan": jnp.tan, "atan": jnp.arctan,
       "asin": jnp.arcsin, "tanh": jnp.tanh, "sqrt": jnp.sqrt, "pow": jnp.power, "atan2": jnp.arctan2,
       "fmod": jnp.fmod}


def mismatches(a, b):
    """(count of bitwise-different elements, max |difference| in units of the reference's ulp)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    bad = a.view(np.uint64) != b.view(np.uint64)
    if not bad.any():
        return 0, 0.0
    ulp = np.spacing(np.abs(b[bad]))
    return int(bad.sum()), float(np.max(np.abs(a[bad] - b[bad]) / ulp))


def measure(fn_table, ref, n=20000):
    """{name: [(range, n, mismatches, max ulp)]} of fn_table[name] (jitted) against the glibc reference."""
    out = {}
    for name, cases in samples(n).items():
        if name not in fn_table:
            continue
        f = jax.jit(fn_table[name])
        rows = []
        for label, *args in cases:
            got = np.asarray(f(*[jnp.asarray(a) for a in args]))
            rows.append((label, len(args[0]), *mismatches(got, ref[name](*args))))
        out[name] = rows
    return out


def test_glibc_exp_tables_equal_oracle_libm():
    """The regenerated exp tables (coar, fine, accurate) and the 13 constants equal the bytes of the oracle's libm."""
    assert libm.glibc_exp_tables_vs_libm(oracle_library("libm.so.6")) == []


def test_glibc_exp_bitwise():
    """glibc_exp == the oracle's exp bitwise on every sampled range. Negative control: jnp.exp does not."""
    ref = glibc_reference()
    res = measure({"exp": libm.glibc_exp}, ref)["exp"]
    assert all(r[2] == 0 for r in res), res
    neg = measure({"exp": jnp.exp}, ref)["exp"]
    assert sum(r[2] for r in neg) > 0, neg
    print("glibc_exp:", res, "\njnp.exp:", neg)


def test_glibc_tanh_bitwise():
    """glibc_tanh == the oracle's tanh bitwise on every sampled range (incl. the dm95 taper range [-6, 4], the paths
    |x| < 2^-55, |x| < 1, 1 <= |x| < 22, |x| >= 22). Negative control: jnp.tanh does not."""
    ref = glibc_reference()
    res = measure({"tanh": libm.glibc_tanh}, ref)["tanh"]
    assert all(r[2] == 0 for r in res), res
    x = np.concatenate([np.array([0.0, -0.0, 1e-300, -2e-17, 0.5, 1.0, -1.0, 21.99, 22.0, -40.0]),
                        np.random.default_rng(2).uniform(-25.0, 25.0, 5000)])
    assert mismatches(jax.jit(libm.glibc_tanh)(x), ref["tanh"](x))[0] == 0
    neg = measure({"tanh": jnp.tanh}, ref)["tanh"]
    assert sum(r[2] for r in neg) > 0, neg
    print("glibc_tanh:", res, "\njnp.tanh:", neg)


def test_powi_equals_libgcc():
    """powi == libgcc __powidf2 and == x ** n (lax.integer_pow) bitwise for n = -4..9."""
    ref = glibc_reference()["__powidf2"]
    x = np.random.default_rng(1).uniform(-2.0, 2.0, 5000)
    for n in range(-4, 10):
        want = ref(x, n)
        assert mismatches(jax.jit(lambda v: libm.powi(v, n))(x), want)[0] == 0, n
        assert mismatches(jax.jit(lambda v: v ** n)(x), want)[0] == 0, n


def test_measured_table_and_m1_bundle():
    """XLA's functions vs the oracle's libm: the measured bitwise status equals libm.MEASURED (0 = bitwise, else not)
    for every function the oracle imports; the M1 bundle uses only bitwise functions."""
    ref = glibc_reference()
    res = measure(JNP, ref)
    status = {k: sum(r[2] for r in rows) for k, rows in res.items()}
    print({k: [(r[0], r[2], round(r[3], 2)) for r in rows] for k, rows in res.items()})
    assert {k: v == 0 for k, v in status.items()} == {k: v == 0 for k, v in libm.MEASURED.items()}, status
    m1 = measure({k: getattr(libm.M1_LIBM, k) for k in ("exp", "sin", "cos", "tan", "tanh", "pow", "sqrt")}, ref)
    assert all(r[2] == 0 for rows in m1.values() for r in rows), m1


if __name__ == "__main__":
    ref = glibc_reference()
    table = measure(JNP, ref)
    table["glibc_exp"] = measure({"exp": libm.glibc_exp}, ref)["exp"]
    table["glibc_tanh"] = measure({"tanh": libm.glibc_tanh}, ref)["tanh"]
    for name, rows in table.items():
        for label, n, bad, ulp in rows:
            print(f"{name:10s} {label:34s} n={n:6d} mismatches={bad:6d} ({100.0 * bad / n:6.2f} %)  max {ulp:.2f} ulp")
