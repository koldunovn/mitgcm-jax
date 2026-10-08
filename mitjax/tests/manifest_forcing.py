"""Manifest fragment of the forcing routines and pkg/sbo (M1 sub-lane FORCING; owner: lane FORCING).
Tier-1 budget (2026-10-01: tier 1 at 88 of 100): no tier-1 tests; every gate is tier 1x (oracle dumps, grids)."""

MANIFEST = {
    "mitjax/tests/test_forcing.py": "tier1x",
    "mitjax/tests/test_sbo.py": "tier1x",
}
