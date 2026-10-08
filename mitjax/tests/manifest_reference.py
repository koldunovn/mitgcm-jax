"""Manifest fragment of the Fortran oracle infrastructure (plan Tasks 3a, 3b, 4, 5; owner: lane A)."""

MANIFEST = {
    # seconds; test_m1_binaries needs the frozen M1 binaries (fails while one is missing). tier1x, not tier1: it checks
    # frozen oracle infrastructure that changes only with reference/ (tier-1 budget of 100 reached at the 2026-10-01
    # merge of Tasks 4 and 7c: 105 tests)
    "scripts/tests/test_reference_build.py": "tier1x",
    # seconds; synthetic files only
    "mitjax/tests/test_dump_reader.py": "smoke",
    # seconds; reads the jaxdump builds (JD_COMMIT) and runs (RUN_JOB): their tests fail while data is missing.
    # tier1x for the same budget reason (17 tests); run it whenever reference/jaxdump changes
    "scripts/tests/test_jaxdump.py": "tier1x",
    # minutes (I/O: renders docs/YARDSTICK.md and docs/REFERENCE_RUNS.md from ~60 oracle runs and compares them with
    # the committed files); oracle-dependent, fails while a registered run is missing (plan Task 3b)
    "scripts/tests/test_reference_runs.py": "tier1x",
    # ~15 s (reads the dumps of the Fortran restart pair, job 27831497); oracle-dependent, fails while the registered
    # restart_a/restart_b runs are missing (lane A session 5)
    "scripts/tests/test_restart_oracle.py": "tier1x",
    # ~1 min (reads the probe and ptracer dumps of the M2 jdon runs); oracle-dependent, fails while an M2 build or run
    # is missing (lane A session 6)
    "scripts/tests/test_m2_oracle.py": "tier1x",
    # ~30 s (reads the M3 jdon dumps' mixing stages and the coverage reports); oracle-dependent, fails while an M3
    # build, run or report is missing (lane A session 9, plan Task 28)
    "scripts/tests/test_m3_oracle.py": "tier1x",
    # ~1 min (reads the M4 jdon dumps' forcing / sea-ice stages, the FD runs and the coverage reports); oracle-dependent,
    # fails while an M4 build, run or report is missing (lane A session 11)
    "scripts/tests/test_m4_oracle.py": "tier1x",
    # ~1-2 min (hashes the FTZ builds' ~1000 objects, recomputes the FTZ-vs-standard band from the plain and dumps-on
    # runs, 2 GB of dumps); oracle-dependent, fails while an FTZ build, run or band report is missing (plan decision 13,
    # lane A session 12)
    "scripts/tests/test_ftz_oracle.py": "tier1x",
}
