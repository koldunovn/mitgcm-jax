"""Manifest fragment of the M1 tracer lane (plan Task 13, R2 tutorial_baroclinic_gyre)."""

MANIFEST = {
    # R2: initial state with exactConserv, substeps of steps 1-3 (P02/P03/T11-T13/T02 incl.), negative controls,
    # P=4 == P=1 for 10 steps on fake CPU devices, gradient finiteness and tile-edge FD (~6 tests, a few minutes)
    "mitjax/tests/test_r2_baroclinic_gyre.py": "tier1x",
    # R2: the variant's ONE tier-1 smoke test (whole 10-step run: PS/cg2d lines, %MON blocks, testreport digits)
    "mitjax/tests/test_r2_baroclinic_gyre_tier1.py": "tier1",
}
