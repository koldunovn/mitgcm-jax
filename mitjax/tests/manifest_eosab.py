"""Manifest fragment of the M3 lane EOSAB (FIND_ALPHA / FIND_BETA 'MDJWF'; owner: EOSAB lane). All tier1x (no tier-1
test: the tier-1 budget is reserved for the per-variant smoke tests)."""

MANIFEST = {
    # replay gates of FIND_ALPHA / FIND_BETA 'MDJWF' on the EOSAB harness run (reference/replay_eosab/CURRENT): three
    # passes bitwise (Python level loop and scan_levels), INI_EOS MDJWF coefficients, negative controls, gradients
    # (finite, dot test, FD); about a minute
    "mitjax/tests/test_eosab.py": "tier1x",
}
