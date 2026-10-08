"""Manifest fragment of branch coverage and the CPP live-code viewer (plan Task 5: tools/cpp_live.py,
tools/coverage.py, reference/coverage/; owner: lane E).

test_cpp_live.py: two tests, seconds (cpp in lane D's link farms; needs the oracle build records): tier 1.
test_coverage.py: reads the gcov runs and reports under $MJX_REFERENCE/coverage (tens of seconds): tier 1x.
"""

MANIFEST = {
    "scripts/tests/test_cpp_live.py": "tier1",
    "scripts/tests/test_coverage.py": "tier1x",
}
