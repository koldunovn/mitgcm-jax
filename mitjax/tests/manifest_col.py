"""Manifest fragment of the M1 sub-lane COL (equation of state and column physics; owner: COL lane). All tier1x
(the tier-1 budget is reserved for the per-variant smoke tests of M1)."""

MANIFEST = {
    # scan_k: bitwise vs a Python loop in Fortran order, planted reversed order, gradients (seconds)
    "mitjax/tests/test_scan_k.py": "tier1x",
    # replay gates of the EOS and column-physics kernels on the three COL harness runs (reference/replay_col/CURRENT),
    # INI_EOS, negative controls, gradient finiteness and FD (a few minutes)
    "mitjax/tests/test_column_physics.py": "tier1x",
}
