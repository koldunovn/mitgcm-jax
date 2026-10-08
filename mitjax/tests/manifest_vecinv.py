"""Manifest fragment of the VECINV lane (M2 sub-lane VECINV, plan Task 23: pkg/mom_vecinv and the pkg/mom_common
routines MOM_VECINV calls)."""

MANIFEST = {
    # replay gates against the gfortran harness reference/replay_vecinv (runs named by reference/replay_vecinv/CURRENT):
    # global_ocean.90x40x15/code_ad, global_ocean.cs32x15/code, solid-body.cs-32x32x1/code; bitwise, setup checks,
    # negative controls, finite gradients, FD and dot tests. Fail, do not skip, without the runs. tier1x (tier 1 is
    # at ~95 of its 100 tests, coordinator 2026-10-01).
    "mitjax/tests/test_vecinv.py": "tier1x",
}
