"""Manifest fragment of the lane MINMAX (gfortran MAX/MIN tie and NaN winners per call site). Tier 1x (tier 1 is at
its budget)."""

MANIFEST = {
    # site tables of the M1 builds, the audit of every MAX/MIN in mitjax/model and mitjax/pkg against them (with
    # negative controls), and the helper's values on the probe sites and the oracle's own object (seconds; needs
    # $MJX_REFERENCE/minmax_sites, fails while it is missing)
    "mitjax/tests/test_minmax_sites.py": "tier1x",
}
