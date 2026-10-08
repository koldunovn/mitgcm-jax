"""Manifest fragment of the r* / nonlinear free-surface routines (M1 lane RSTAR, plan Task 15a: CALC_R_STAR,
UPDATE_R_STAR, RESET_NLFS_VARS, UPDATE_SURF_DR). No tier-1 test (lane rule: the tier-1 budget is kept for one smoke
test per M1 variant)."""

MANIFEST = {
    # dump gates of every dumped step (global_ocean.90x40x15, advect_xz/input.nlfs, advect_xz/input + input.pqm),
    # INITIALISE_VARIA's r* sequence, UPDATE_CG2D on our hFac, P=4 == P=1, gradient finiteness / FD / dot test,
    # negative controls (~1-2 min on a compute node)
    "mitjax/tests/test_rstar.py": "tier1x",
}
