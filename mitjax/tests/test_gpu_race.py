"""Tier 2, one GPU: the GPU race of 2026-10-09 stays fixed (mitjax/xla_flags.py docstring, `multi_output_fusion`).
XLA:GPU's multi-output fusion built a kernel that wrote UPDATE_ETAH's etaH := etaN in place into etaH's buffer while
it read etaH at other points for etaN's halo exchange; every call of the vjp forward of
tutorial_global_oce_optim/input_ad gave another cost and gradient (jobs 27987414, 27994893). The program is
mitjax/tests/gpu_race_gate.py (the 3-step vjp forward, the smallest one that raced).

* test_gpu_vjp_forward_no_race: compiled with the gate flags (multi_output_fusion disabled), the program holds no
  kernel that mitjax/tests/hlo_race.py flags, and three calls of the executable give bitwise the same outputs (the
  cost and every residual; with the race 136 of 5534 outputs differed in every call).
* test_gpu_race_control: the same program compiled in a subprocess with the flags of before 2026-10-09
  (--xla_disable_hlo_passes=algsimp only) holds the racing kernel: an output written in place by a
  dynamic-update-slice f64[4,24,49] whose buffer a slice reads (job 27994893: output 3 of
  loop_dynamic_update_slice_select_fusion). A scanner or a program that stops showing it fails here.

Cost (job 28005413, one A100): model set-up under a minute and about 130 s of compile per program; a few minutes for
both. Run by scripts/run_tier2.sbatch; skips itself without a GPU."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from mitjax.xla_flags import GATE_FLAGS, set_gate_xla_flags

set_gate_xla_flags()

import jax  # noqa: E402

from mitjax.tests import hlo_race as H  # noqa: E402

BEFORE = "--xla_disable_hlo_passes=algsimp"      # the gate set's pass list before 2026-10-09


def _need_gpu():
    try:
        gpus = jax.devices("gpu")
    except RuntimeError:
        gpus = []
    if not gpus:
        pytest.skip("tier 2: needs a GPU (sbatch --gpus=1 scripts/run_tier2.sbatch)")


def _save(name, obj):
    d = os.environ.get("MJX_TIER2_OUT")
    if d:
        with open(os.path.join(d, name), "w") as f:
            json.dump(obj, f, indent=1)


def test_gpu_vjp_forward_no_race(tmp_path):
    _need_gpu()
    from mitjax.tests import gpu_race_gate as R
    fn, args = R.program(tmp_path)
    compiled = jax.jit(fn).lower(*args).compile()
    races = H.find(compiled.as_text())
    assert not races, f"{len(races)} kernel(s) write a buffer in place while reading it:\n{H.report(races)}"
    first = [np.asarray(x) for x in jax.tree_util.tree_leaves(compiled(*args))]
    differ = []
    for call in (1, 2):
        again = jax.tree_util.tree_leaves(compiled(*args))
        differ.append(sum(not np.array_equal(a, np.asarray(b), equal_nan=a.dtype.kind in "fc")
                          for a, b in zip(first, again)))
    _save("gpu_race.json", {"fc": float(first[0]), "outputs": len(first), "differ_calls_1_2": differ,
                            "xla_flags": os.environ.get("XLA_FLAGS")})
    assert differ == [0, 0], f"outputs differing from call 0 in calls 1, 2: {differ} of {len(first)}"


def test_gpu_race_control(tmp_path):
    _need_gpu()
    flags = " ".join(BEFORE if f.startswith("--xla_disable_hlo_passes=") else f for f in GATE_FLAGS)
    assert flags != " ".join(GATE_FLAGS)
    env = dict(os.environ, XLA_FLAGS=flags, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    p = subprocess.run([sys.executable, "-m", "mitjax.tests.gpu_race_gate", "--out", str(tmp_path / "control")],
                       env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    got = json.loads(p.stdout.strip().splitlines()[-1])
    _save("gpu_race_control.json", got)
    assert got["xla_flags"] == flags, got["xla_flags"]
    assert any("cuda" in d.lower() or "gpu" in d.lower() for d in got["devices"]), got["devices"]
    hits = [r for r in got["races"] if r["op"] == "dynamic-update-slice" and r["shape"] == "f64[4,24,49]"
            and any(x.startswith("slice ") for x in r["reads"])]
    assert hits, f"control did not bite: the program compiled with {flags!r} shows no racing kernel: {got['races']}"
