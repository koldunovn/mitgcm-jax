"""Manifest fragment of the notebooks (docs plan 20261006 S3, Task 8: the laptop notebooks, lane NB; S4, Task 9: the
cluster notebooks, lane NBC). No tier-1 test (tier 1 stays at 98)."""

MANIFEST = {
    # Task 8: notebooks/01-04 executed with nbclient (cell asserts are the checks; wall time and peak RSS recorded,
    # RSS < 8 GiB), the planted wrong assert, the committed notebooks fresh / executed / clean of machine paths and
    # markers. Needs the `notebooks` extra (MJX_NB_PYTHON; scripts/nb_gate.sbatch). About 40 min on 8 cores.
    "mitjax/tests/test_notebooks_laptop.py": "tier1x",
    # Task 9 (lane NBC): notebooks/05-07 (cluster tier) executed with nbclient, wall time and peak RSS recorded, the
    # planted wrong assert. Run before a release by scripts/run_docs_notebooks.sbatch (~1.5 h on 8 cores), not by
    # tier 1x. The committed-notebook check for 05-07 stays in test_notebooks_laptop.py (tier1x).
    "mitjax/tests/test_notebooks_cluster.py": "tier3",
}
