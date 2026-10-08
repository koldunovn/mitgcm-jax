"""Manifest fragment of the gradient machinery, cg2d derivative rule, safe ops and libm (plan Task 7c; owner: lane B)."""

MANIFEST = {
    "mitjax/tests/test_safe_ops.py": "smoke",
    "mitjax/tests/test_cg2d_rule.py": "smoke",
    "mitjax/tests/test_checkpoint.py": "smoke",
    "mitjax/tests/test_libm.py": "tier1",
}
