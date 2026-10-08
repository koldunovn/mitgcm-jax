"""R5 optim adjoint on A100s (plan Task 17 for R5, tier 2): the sharded gradient program on P A100s vs the
single-device program on one A100 (scripts/r5_shardgrad/gpu.py, gpu.sbatch: each run twice with seed 1 -- the
repeat floor -- and with seeds 0 and 2). The runs are separate GPU jobs; this test compares their saved arrays
(MJX_R5GPU_P1_DIR, MJX_R5GPU_PN_DIR: the `res` directories of the two jobs) and needs no GPU itself.

Criterion (Nikolay 2026-10-02, plan "Decisions 2026-10-02"): P=N vs P=1 gradient, max |g_N - g_1| / max |g_1|,
within floor(P=1) + floor(P=N) + the CPU-measured deterministic order class between the same two programs
(gpu.ORDER_CLASS = 7.66e-15, test_r5_sharded_grad_cpu.py, dev job 27838618). Measured (jobs 27838614 / 27838899):
1.2994e-14 vs 7.15e-15 + 5.76e-15 + 7.66e-15 = 2.06e-14. Also: fc identical across runs and P, every point finite,
seed 0 -> exactly 0.

Seed-2 linearity on the GPU, max |g(seed 2) - 2 g(seed 1)| / max |2 g(seed 1)| (gpu.py's `seed2_rel`), against a
sanity bound of 1e-12 at P=1 and at P=N (Nikolay 2026-10-02: GPU linearity = gross-bug guard; CPU is the bitwise
check -- the bitwise CPU linearity checks are in test_r5_sharded_grad_cpu.py and stay as they are). Measured
(jobs 27838614 / 27838899): 6.97e-15 at P=1, 1.17e-14 at P=4.
"""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest


def _gpu_mod():
    from mitjax import paths
    spec = importlib.util.spec_from_file_location("_r5gpu", paths.REPO / "scripts" / "r5_shardgrad" / "gpu.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dirs():
    d1, dn = os.environ.get("MJX_R5GPU_P1_DIR", ""), os.environ.get("MJX_R5GPU_PN_DIR", "")
    if not (d1 and dn):
        pytest.skip("tier 2: set MJX_R5GPU_P1_DIR / MJX_R5GPU_PN_DIR to the res directories of the GPU jobs")
    return Path(d1), Path(dn)


def test_r5_sharded_vs_p1_gpu():
    d1, dn = _dirs()
    res = _gpu_mod()._compare(d1, dn)
    i1, iN = (json.loads((d / "run.json").read_text()) for d in (d1, dn))
    assert len(set(i1["J"] + iN["J"])) == 1, (i1["J"], iN["J"])
    for info in (i1, iN):
        assert info["finite"] and info["seed0_zero"]
        assert info["seed2_rel"] <= 1e-12, info["seed2_rel"]        # gross-bug guard (Nikolay 2026-10-02)
    assert res["within"], res
    a = np.load(d1 / "run.npz")
    assert np.all(np.isfinite(a["g0"]))
