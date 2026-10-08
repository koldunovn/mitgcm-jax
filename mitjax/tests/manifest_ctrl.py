"""Manifest fragment of pkg/ctrl, pkg/cost, pkg/grdchk (plan Task 16 preparation; owner: lane ctrl).
No tier-1 tests (tier 1 at 88 of 100, 2026-10-01): the oracle gates read the 10-step dumps (tier1x)."""

MANIFEST = {
    "mitjax/tests/test_ctrl_cost.py": "tier1x",
}
