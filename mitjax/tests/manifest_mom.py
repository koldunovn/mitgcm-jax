"""Manifest fragment of the MOM lane (M1 sub-lane MOM: pkg/mom_common and pkg/mom_fluxform leaf kernels)."""

MANIFEST = {
    # replay gates against the gfortran harness reference/replay_mom (runs named by reference/replay_mom/CURRENT);
    # fail, do not skip, without the runs. All tier1x (tier-1 budget, coordinator 2026-10-01).
    # tutorial_barotropic_gyre (1 tile x 1 level, 79 outputs; ~10 s)
    "mitjax/tests/test_mom_kernels_quick.py": "tier1x",
    # the other three experiments bitwise, the setup checks, negative controls, finite gradients, FD and dot tests
    "mitjax/tests/test_mom_kernels.py": "tier1x",
}
