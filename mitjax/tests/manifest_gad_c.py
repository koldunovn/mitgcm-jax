"""Manifest fragment of lane GAD-C (PPM/PQM advection kernels of pkg/generic_advdiff)."""

MANIFEST = {
    # replay gates against the gfortran harness reference/replay_gad_c (the run named by its CURRENT; fails while it
    # is missing): bitwise on every case and level, negative controls, limiter-branch coverage, finite gradients,
    # tangent/adjoint and FD
    "mitjax/tests/test_gad_pqm.py": "tier1x",
    # the one compact tier-1 test: the schemes advect_xz runs, three levels, bitwise (same harness run)
    "mitjax/tests/test_gad_pqm_compact.py": "tier1",
}
