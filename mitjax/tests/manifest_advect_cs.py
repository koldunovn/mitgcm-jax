"""Manifest fragment of the ADVECT lane's cubed-sphere work (M2 Task 25 part: advect_cs)."""

MANIFEST = {
    # advect_cs: initial state, steps 1-3 at every dumped stage, negative controls, P=6 == P=1 (subprocess), whole
    # run (%MON, digits): tier 1x only (no tier-1 test)
    "mitjax/tests/test_advect_cs.py": "tier1x",
}
