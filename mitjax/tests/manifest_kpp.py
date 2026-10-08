"""Manifest fragment of the KPP sub-lane (M3 sub-lane KPP, plan Task 29; owner: lane KPP)."""

MANIFEST = {
    # pkg/kpp bitwise vs the replay harness reference/replay_kpp (runs named by reference/replay_kpp/CURRENT) and
    # vs the dumps-on runs of vermix/input(.dd) (P07_kpp, P10_kpp_exch), negative controls, gradient finiteness /
    # dot test / FD, unported options. No tier-1 test (tier 1 is at its budget).
    "mitjax/tests/test_kpp.py": "tier1x",
}
