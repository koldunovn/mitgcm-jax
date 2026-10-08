"""Manifest fragment of pkg/monitor and the Fortran formatter (plan Task 11 monitor item; owner: lane MON).
Tier-1 budget (2026-10-01: tier 1 at 85 of 100): one compact test in tier 1, the rest in tier 1x."""

MANIFEST = {
    "mitjax/tests/test_monitor_tier1.py": "tier1",
    "mitjax/tests/test_monitor.py": "tier1x",
    "mitjax/tests/test_fortran_format.py": "tier1x",
}
