"""The environment the rest of the suite silently relies on: this checkout being the package under test, the pinned
package set, float64, four fake devices, and the gate XLA flags actually changing the arithmetic (each flag with a
negative control run in a subprocess without it)."""

import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P

import mitjax
from mitjax.xla_flags import FAKE_DEVICES, GATE_FLAGS, gate_xla_flags

REPO_ROOT = Path(__file__).resolve().parents[2]

# The packages whose version changes numerics or AD; the whole set is in constraints.txt.
PINNED = ("jax", "jaxlib", "numpy", "scipy")


def _constraints():
    pins = {}
    for line in (REPO_ROOT / "constraints.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, ver = line.split("==")
            pins[name.strip().lower()] = ver.strip()
    return pins


def test_imports_this_checkout():
    assert Path(mitjax.__file__).resolve().parent == REPO_ROOT / "mitjax"


def test_pinned_versions_match_constraints():
    pins = _constraints()
    got = {name: importlib.metadata.version(name) for name in PINNED}
    want = {name: pins[name] for name in PINNED}
    assert got == want, "env drifted from constraints.txt; a JAX upgrade goes through the canary (docs/ENV.md)"


def test_x64_float64_under_jit():
    assert jax.config.jax_enable_x64
    assert jnp.arange(4.0).dtype == jnp.float64

    @jax.jit
    def f(a):
        return a * (1.0 + 1e-12) - a

    y = f(jnp.ones(3))
    assert y.dtype == jnp.float64
    # 1e-12 is below float32 resolution (6e-8): a silent float32 fallback returns exactly 0.
    np.testing.assert_allclose(np.asarray(y), 1e-12, rtol=1e-3)


def test_gate_flags_set_once_and_four_devices():
    flags = os.environ["XLA_FLAGS"].split()
    for flag in GATE_FLAGS:
        name = flag.split("=")[0]
        assert [f for f in flags if f.split("=")[0] == name] == [flag], os.environ["XLA_FLAGS"]
    devs = jax.devices("cpu")
    assert len(devs) == FAKE_DEVICES, "conftest's XLA_FLAGS arrived after the CPU backend was initialised"
    mesh = Mesh(np.array(devs), ("tile",))

    @jax.jit
    def total(a):
        return jax.shard_map(lambda b: jax.lax.psum(b.sum(), "tile"),
                             mesh=mesh, in_specs=P("tile"), out_specs=P())(a)

    assert float(total(jnp.arange(8.0))) == 28.0
    # composing refuses a conflicting value instead of silently keeping one of two
    with pytest.raises(ValueError):
        gate_xla_flags("--xla_cpu_max_isa=AVX2")
    assert gate_xla_flags(" ".join(GATE_FLAGS)) == " ".join(GATE_FLAGS)


def measure():
    """The two roundings the gate flags control, computed under jit with traced arguments (run in-process with the
    gate flags and in subprocesses without one of them)."""
    n = 1024
    # a*b = 1 - 2**-60 exactly: two roundings give a*b+c = 0, a fused multiply-add gives -2**-60
    a, b, c = np.full(n, 1.0 + 2.0**-30), np.full(n, 1.0 - 2.0**-30), np.full(n, -1.0)
    fma = np.asarray(jax.jit(lambda a, b, c: a * b + c)(a, b, c))
    # x/d vs x*(1/d) for d = 3: they differ in the last bit for many of these x
    x, d = np.arange(1.0, n + 1.0) * 1.1, 3.0
    div = np.asarray(jax.jit(lambda x, d: x / d)(x, d))
    return {"fma_fused": int(np.sum(fma != 0.0)), "div_mismatch": int(np.sum(div != x / d)),
            "probe_bites": int(np.sum(x / d != x * (1.0 / d)))}


def _measure_without(dropped):
    flags = " ".join(f for f in GATE_FLAGS if f != dropped)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("XLA_", "JAX_"))}
    env.update(XLA_FLAGS=flags, JAX_PLATFORMS="cpu", PYTHONPATH=str(REPO_ROOT))
    code = "import json; from mitjax.tests.test_env import measure; print(json.dumps(measure()))"
    out = subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_gate_flags_bite():
    """Under the gate flags XLA:CPU computes a*b+c with two roundings and x/d as a division, as gfortran does with
    -ffp-contract=off. Negative controls (subprocesses): without --xla_cpu_max_isa=AVX the multiply-add is fused,
    without --xla_disable_hlo_passes=algsimp x/d becomes x*(1/d). If a control stops biting, that flag is no longer
    what keeps the gates bitwise, and this test says so."""
    m = measure()
    assert m["probe_bites"] > 100, m  # the probe can tell a division from a reciprocal multiply
    assert m["fma_fused"] == 0 and m["div_mismatch"] == 0, m
    no_avx = _measure_without(GATE_FLAGS[0])
    assert no_avx["fma_fused"] > 0, f"negative control did not bite: without {GATE_FLAGS[0]} got {no_avx}"
    no_algsimp = _measure_without(GATE_FLAGS[1])
    assert no_algsimp["div_mismatch"] > 0, f"negative control did not bite: without {GATE_FLAGS[1]} got {no_algsimp}"
