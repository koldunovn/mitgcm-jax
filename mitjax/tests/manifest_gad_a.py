"""Manifest fragment of the M1 sub-lane GAD-A (simple advective and diffusive flux kernels of pkg/generic_advdiff;
owner: lane GAD-A). All tier 1x (coordinator 2026-10-01: tier 1 is at its budget)."""

MANIFEST = {
    # mitjax/ops/fortran_minmax.py (Fortran MAX/MIN) against the gfortran probe output (seconds)
    "mitjax/tests/test_fortran_minmax.py": "tier1x",
    # one compact replay gate (advect_xz/input.nlfs harness run, every GAD-A routine, bitwise on all points, ~10 s);
    # needs the run named by reference/replay_gad_a/CURRENT (fails while it is missing)
    "mitjax/tests/test_gad_simple_compact.py": "tier1x",
    # replay gates on all four harness grids (bitwise, P=4), negative controls, FD / finite gradients, unported
    # branches; needs the runs named by reference/replay_gad_a/CURRENT
    "mitjax/tests/test_gad_simple.py": "tier1x",
}
