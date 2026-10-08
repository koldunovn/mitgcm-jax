"""Manifest fragment of the core infrastructure (plan Task 1; owner: main session)."""

MANIFEST = {
    "mitjax/tests/test_env.py": "smoke",
    "mitjax/tests/test_manifest.py": "smoke",
    "mitjax/tests/test_paths.py": "smoke",
    "mitjax/tests/test_banned_transforms.py": "smoke",
    "scripts/tests/test_check_pytest_report.py": "smoke",
    "scripts/tests/test_tier1x_shards.py": "smoke",                 # parallel tier-1x split (run_tier1x.sbatch)
}
