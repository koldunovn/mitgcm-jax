"""Manifest fragment of the M3 sub-lane GGL90 (pkg/ggl90 incl. IDEMIX and Langmuir; owner: GGL90 lane). Tier 1x only
(the tier-1 budget is reserved)."""

MANIFEST = {
    # replay gates of every pkg/ggl90 routine on the two harness runs (reference/replay_ggl90/CURRENT), dump gates of
    # GGL90_CALC on the vermix .ggl90/.gglLC and global_ocean.90x40x15 .idemix dumps-on runs, negative controls,
    # gradient finiteness, dot test and FD (a few minutes on a CPU node)
    "mitjax/tests/test_ggl90.py": "tier1x",
    # glibc_asin (mitjax/ops/libm.py, IDEMIX's ASIN) bit for bit against the oracle's libm on the IDEMIX range, the
    # dump arguments and the dry-column constants, constants vs the binary, negative controls (seconds)
    "mitjax/tests/test_glibc_asin.py": "tier1x",
}
