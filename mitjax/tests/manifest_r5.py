"""Manifest fragment of R5 tutorial_global_oce_optim/input_ad (plan Task 16; owner: lane R5).
One tier-1 test (the R5 forward smoke); the 10-step stage gates and negative controls are tier1x."""

MANIFEST = {
    "mitjax/tests/test_r5_optim_ad.py": "tier1x",
    "mitjax/tests/test_r5_optim_ad_tier1.py": "tier1",
    "mitjax/tests/test_r5_adjoint.py": "tier1x",          # one gradient compile ~11 min, jvp, FD sweep
    "mitjax/tests/test_r5_sharded_grad_cpu.py": "tier1x",  # single + P=1 mesh + P=2, 3, 4 gradient programs, 18 min (job 27953498)
    "mitjax/tests/test_r5_sharded_grad_gpu.py": "tier2",   # compares the saved A100 runs (scripts/r5_shardgrad)
}
