"""Manifest fragment of M1 sub-lane GAD-B (the SOM advection path of pkg/generic_advdiff; owner: lane GAD-B)."""

MANIFEST = {
    # one test: all SOM routines element-equal to gfortran on the advect_xy replay run (about 10 s)
    "mitjax/tests/test_gad_som_quick.py": "tier1",
    # replay gates on advect_xy and advect_xz, run-switch cross-check, negative controls, finite gradients on all lanes,
    # dot test and FD sweep; needs the runs named by reference/replay_gad_b/CURRENT (fails while one is missing)
    "mitjax/tests/test_gad_som.py": "tier1x",
}
