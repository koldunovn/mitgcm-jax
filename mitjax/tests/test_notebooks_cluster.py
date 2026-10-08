"""The cluster notebooks notebooks/05-07 (docs plan 20261006 Task 9): they run and their cell asserts hold, within the
cluster tier's budget.

1. Execution (one test per notebook): nbclient runs every cell in a fresh kernel in a new directory (the machinery of
   test_notebooks_laptop.py: mitjax/tests/nb_exec.py in a subprocess); the cell asserts are the checks (testreport
   pass, gradient digits against TAF's tangent-linear model / adjoint / Tapenade >= a stated minimum, finite
   gradients, sharded forward bitwise on the same machine, sharded gradient within the R5 bar at devices = 4; platform-tolerant
   except the same-machine sharded-forward statement, plan decision 10). Recorded per notebook: wall time and the
   kernel's peak resident memory (measures.jsonl); asserted: peak RSS < RSS_LIMIT and wall < WALL_BUDGET_S (plan
   decision 5, cluster tier: 10-30 min, tens of GB). Run by scripts/run_docs_notebooks.sbatch pinned to 8 cores.
2. Negative control: notebook 07 with the expected number of differing values of the sharded forward planted to 1
   (the forward is bitwise: 0) must fail in nbclient with an AssertionError; it fails at the first assert after the
   sharded forward runs (a few minutes), before any gradient is compiled.
The committed-notebook check (cells == sources, executed, no error output, header keys, forbidden strings) covers
05-07 in test_notebooks_laptop.py::test_committed_notebooks_fresh_executed_and_clean.

Group tier3 (manifest_nb.py): about 1.5 h on 8 cores, run before a release by the docs job, not by tier 1x.
"""

import pytest

from mitjax.tests.test_notebooks_laptop import CLUSTER_NOTEBOOKS, NB_DIR, execute

RSS_LIMIT = 64 * 2**30                   # plan decision 5: cluster tier "tens of GB"
WALL_BUDGET_S = 40 * 60                  # plan decision 5: 10-30 min; each notebook's header states its own measure


@pytest.mark.parametrize("name", CLUSTER_NOTEBOOKS)
def test_notebook_runs_within_cluster_budget(name):
    rc, rec, executed, err = execute(name)
    print(f"{name}: wall {rec.get('wall_s')} s, peak RSS {rec.get('peak_rss_gib')} GiB -> {executed}")
    assert rc == 0, (rec.get("error"), err)
    assert rec["peak_rss_bytes"] > 100 * 2**20, rec
    assert rec["peak_rss_bytes"] < RSS_LIMIT, rec
    assert rec["wall_s"] < WALL_BUDGET_S, rec


def test_planted_wrong_assert_fails_nbclient():
    text = (NB_DIR / "src" / "07_parallel.py").read_text()
    assert text.count("FORWARD_BITS_EXPECTED = 0 ") == 1
    rc, rec, _, err = execute("07_parallel", text.replace("FORWARD_BITS_EXPECTED = 0 ", "FORWARD_BITS_EXPECTED = 1 "),
                              tag="07_parallel-planted")
    print(f"planted: exit {rc}, error {rec.get('error')}, wall {rec.get('wall_s')} s")
    assert rc != 0 and rec["error"]["ename"] == "AssertionError", (rec, err)
