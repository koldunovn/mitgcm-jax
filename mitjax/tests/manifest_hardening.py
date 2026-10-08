"""Manifest fragment of the M0 hardening lane (REVIEW_M0 fixes; owner: hardening lane)."""

MANIFEST = {
    # REAL*4 literals against gfortran's bits and the REAL*4 namelist READ (seconds; reads the global_ocean and
    # barotropic configurations): tier 1x, the tier-1 budget is kept for the gates
    "mitjax/tests/test_real4.py": "tier1x",
    # the run verdict of the oracle job scripts (expected grdchk stop vs real failures; seconds, reads one oracle run)
    "scripts/tests/test_run_verdict.py": "tier1x",
}
