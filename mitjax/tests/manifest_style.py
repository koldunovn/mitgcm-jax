"""Manifest fragment of the code-style prototype (plan Task 8; owner: lane C)."""

MANIFEST = {
    # farray semantics vs a numpy reference, error messages, pytree round trip incl. shard_map at P=4 (seconds)
    "mitjax/tests/test_farray.py": "smoke",
    # replay gates against the gfortran harness (bitwise, identical HLO, FD, finite gradients, divide probe) and their
    # negative controls; needs the replay run named by reference/replay/CURRENT (fails while it is missing)
    "mitjax/tests/test_style_prototype.py": "tier1x",
}
