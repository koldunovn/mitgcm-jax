"""Manifest fragment of the GM/Redi sub-lane (M1 sub-lane GMREDI; owner: lane GMREDI)."""

MANIFEST = {
    # GMREDI_CALC_TENSOR bitwise vs the dumps-on runs (global_ocean.90x40x15, tutorial_global_oce_optim), the replay
    # harness reference/replay_gmredi (every ported routine; needs the runs named by
    # reference/replay_gmredi/CURRENT_<exp>), negative controls, gradient finiteness / FD / dot test, unported options
    "mitjax/tests/test_gmredi.py": "tier1x",
}
