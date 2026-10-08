"""Manifest fragment of the two-dimensional pressure solver (M1 sub-lane CG2D: INI_CG2D, UPDATE_CG2D, CG2D and its
derivative wiring). No tier-1 test (coordinator, 2026-10-01: the tier-1 budget is kept for one smoke test per M1
variant)."""

MANIFEST = {
    # barotropic gyre: INI_CG2D operator bitwise vs C01 on all points, CG2D replay C01 -> C02 bitwise, the rule's
    # forward value == the literal solve on every lane (one test, ~10 s; the candidate for a tier-1 slot)
    "mitjax/tests/test_cg2d_tier1.py": "tier1x",
    # the four solver-calling M1 variants: operator (incl. the global_ocean UPDATE_CG2D chain), replay at every dumped
    # step, rule == literal, P=4, STDOUT lines, parameters vs the printout, dot tests and FD, negative controls
    # (~2 min on a compute node)
    "mitjax/tests/test_cg2d.py": "tier1x",
}
