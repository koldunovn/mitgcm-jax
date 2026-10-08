"""Manifest fragment of the cubed-sphere W2 topology, exchanges and corner fills (plan Task 22; owner: lane B).
Tier-1 budget (~95 of 100): nothing in tier 1."""

MANIFEST = {
    # W2 print-out, every exchange kind, FILL_CS_CORNER_*, P=4/P=6 == P=1, transpose; oracle: reference/replay_cube
    "mitjax/tests/test_cube.py": "tier1x",
    # cube dynamics end to end (Task 25): adjustment.cs steps 1-3 at every stage, whole run, P=4/P=6
    "mitjax/tests/test_cube_run.py": "tier1x",
}
