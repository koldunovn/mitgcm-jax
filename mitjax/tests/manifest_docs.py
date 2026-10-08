"""Manifest fragment of the docs lane (plan Task 9: docs/READING_GUIDE.md; Task 18: the M1 side-by-side page)."""

MANIFEST = {
    # every file:line citation and every quoted block of docs/READING_GUIDE.md against the repository and the upstream
    # clone at `pinned`, plus planted-error controls (seconds; needs $MJX_UPSTREAM)
    "mitjax/tests/test_docs_examples.py": "tier1x",
    # every quoted line of docs/review/M1_SIDE_BY_SIDE.html (plan Task 18 readability review) against the repository
    # and the upstream clone at `pinned`, no external resources, planted-error controls (seconds; needs $MJX_UPSTREAM)
    "mitjax/tests/test_m1_side_by_side.py": "tier1x",
    # docs plan 20261006 Task 7 (lane FMAP): the generated pages of tools/gen_docs.py (docs/fortran_map.md) are fresh,
    # their port citations exist upstream at 63cdc0b, own_code.PORTED is on the map; planted controls (stale page,
    # new citation, missing file, line past the end, machine path). Seconds on the login node, but tier1x, not smoke:
    # smoke counts in tier 1, which stays at 98 (LANE_RULES).
    "mitjax/tests/test_docs_generated.py": "tier1x",
    # docs plan 20261006 Task 10 (lane DOCS5): the public pages' relative links and anchors resolve, no private text
    # (machine paths, accounts, job IDs, host names, assistant markers, session links), every `python -m mitjax`
    # command line shown parses; planted controls for each. Seconds; tier1x (tier 1 stays at 98).
    "mitjax/tests/test_docs_pages.py": "tier1x",
}
