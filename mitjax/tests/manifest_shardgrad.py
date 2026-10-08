"""Manifest fragment of the SHARDGRAD lane (plan Task 17: the gradient/checkpoint drivers on the sharded model)."""

MANIFEST = {
    # xla_flags.flag_set: the GPU-only flag set of tier-2 runs (pure Python)
    "mitjax/tests/test_xla_flag_set.py": "smoke",
    # grad.value_and_grad / objective keep their jitted function per configuration (toy step, seconds)
    "mitjax/tests/test_grad_cache.py": "smoke",
    # R2 sharded gradients on 4 real A100s vs 1 (scripts/run_tier2.sbatch --gpus=4 / --gpus=1): repeat floors,
    # finiteness, linearity guard, P=4 vs P=1 gradient and forward; skips itself without GPUs
    "mitjax/tests/test_sharded_grad_gpu.py": "tier2",
    # R2 sharded gradients on fake CPU devices (P = 1, 4, 3 vs the single-device driver), linearity guard, negative
    # controls (wrong ppermute transpose, missing vary); ~45 min: eight programs, up to ~7 min each to trace and compile
    "mitjax/tests/test_sharded_grad_cpu.py": "tier1x",
}
