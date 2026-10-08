"""The laptop notebooks notebooks/01-04 (docs plan 20261006 Task 8): they run, their cell asserts hold, they stay inside
the laptop budget, and what is committed of them is clean.

1. Execution (one test per notebook): nbclient runs every cell in a fresh kernel (mitjax/tests/nb_exec.py) in a new
   directory; the cell asserts are the checks (testreport digits >= a stated minimum, grdchk agreement, a finite
   gradient, a dot test; platform-tolerant, plan decision 10: no bitwise assert). Recorded per notebook: wall time
   and the kernel's peak resident memory (measures.jsonl in the output directory); asserted: peak RSS < 8 GiB and wall
   time < WALL_BUDGET_S (plan decision 5, laptop tier). The gate runs it pinned to 8 cores (`taskset -c 0-7`), a
   laptop's count. The executed notebooks are written to `<out>/executed/` (MJX_NB_OUT names <out>); those of a
   compute-node run are what notebooks/*.ipynb commits.
2. Negative control: notebook 01 with its minimum testreport digits planted to 17 (more than testreport can report)
   must fail in nbclient with an AssertionError.
3. The committed notebooks (01-04 and the cluster notebooks 05-07 of test_notebooks_cluster.py): the cells equal their sources notebooks/src/*.py (tools/build_notebooks.py), every code cell
   has been executed and none holds an error, the header cell states tier, run time, peak memory, experiment and
   what the notebook shows, and no cell or output holds a machine path, a user or project name of this machine, a
   batch-job reference, a host name or an assistant/session marker (FORBIDDEN); negative control: each forbidden
   sample planted into a committed notebook's text is found.

The notebooks need nbclient / nbformat / ipykernel / matplotlib (pyproject extra `notebooks`). The python that runs
them: MJX_NB_PYTHON, else `$MJX_WORK/envs/mitjax-nb/bin/python` where it exists (this project's overlay venv,
docs/ENV.md), else the running interpreter. Costs (CPU node, 8 cores): see the notebooks' header cells; about 30 min.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mitjax import paths

NOTEBOOKS = ("01_quickstart", "02_fortran_to_jax", "03_first_gradient", "04_own_configuration")
CLUSTER_NOTEBOOKS = ("05_lab_sea", "06_adjoint_modes_cs32x15", "07_parallel")    # executed by test_notebooks_cluster.py
NB_DIR = paths.REPO / "notebooks"
RSS_LIMIT = 8 * 2**30                    # plan decision 5: laptop tier < 8 GB
WALL_BUDGET_S = 20 * 60                  # plan decision 5: laptop tier "minutes"; each notebook's header states its own
HEADER_KEYS = ("Tier:", "Run time:", "Peak memory:", "Experiment:", "Shows:")
# The literals the public-tree scan also forbids carry a one-letter character class, and the planted samples are split
# into adjacent strings, so that this file does not hold what it forbids (the patterns match what they matched before).
FORBIDDEN = {
    "machine path": r"/(?:work|home|scratch|sw|pf|mnt|Users|private/var)/",
    "user or project": r"a27[0]088|ab0995|koldunov",
    "batch job": r"(?i)slurm|sbatch|\bjob ?id\b",
    "host name": r"(?i)levante|\bl\d{5}\b|dkrz",
    "assistant marker": r"(?i)c[l]aude|anthropic|session[ _-]?(?:id|link)|generated (?:by|with)",
}
PLANTED = ("/work/x/y", "/home/u/z", "a27" "0088", "slurm-123.out", "levante", "node l40011", "Cl" "aude",
           "session id", "/scr" "atch/a/b")


def _build_tool():
    spec = importlib.util.spec_from_file_location("_mjx_build_notebooks", paths.REPO / "tools" / "build_notebooks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _nb_python():
    p = os.environ.get("MJX_NB_PYTHON", "").strip()
    if p:
        return p
    work = getattr(paths, "WORK", None)
    if work is not None and (work / "envs" / "mitjax-nb" / "bin" / "python").exists():
        return str(work / "envs" / "mitjax-nb" / "bin" / "python")
    return sys.executable


def _out_root():
    d = os.environ.get("MJX_NB_OUT", "").strip()
    root = Path(d) if d else paths.RUNS / "tests_nb" / f"nb-{os.getpid()}-{time.time_ns()}"
    (root / "executed").mkdir(parents=True, exist_ok=True)
    return root


_OUT = None


def out_root():
    global _OUT
    if _OUT is None:
        _OUT = _out_root()
    return _OUT


def execute(name, text=None, tag=None):
    """Run notebook `name` (its committed cells, or the notebook built from source `text`) in a new directory;
    returns (exit code, measures dict, executed path, stderr tail)."""
    tag = tag or name
    work = out_root() / "work" / tag
    work.mkdir(parents=True)
    nb = work / f"{name}.ipynb"
    if text is None:
        src = json.loads((NB_DIR / f"{name}.ipynb").read_text())
        for c in src["cells"]:                       # a clean start: the committed outputs are not inputs
            if c["cell_type"] == "code":
                c["outputs"], c["execution_count"] = [], None
        nb.write_text(json.dumps(src, indent=1))
    else:
        nb.write_text(json.dumps(_build_tool().notebook(name, text), indent=1))
    executed = out_root() / "executed" / f"{tag}.ipynb"
    meas = out_root() / "work" / f"{tag}.json"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(paths.REPO)] + [p for p in os.environ.get(
        "PYTHONPATH", "").split(os.pathsep) if p]))
    py = _nb_python()
    probe = subprocess.run([py, "-c", "import nbclient, nbformat, ipykernel, matplotlib"], capture_output=True,
                           text=True, env=env)
    if probe.returncode != 0:
        pytest.fail(f"{py} cannot run the notebooks (pip install -e '.[notebooks]', or set MJX_NB_PYTHON): "
                    f"{probe.stderr[-300:]}")
    r = subprocess.run([py, "-m", "mitjax.tests.nb_exec", str(nb), str(executed), str(meas)], capture_output=True,
                       text=True, env=env, cwd=work)
    rec = json.loads(meas.read_text()) if meas.exists() else {}
    with open(out_root() / "measures.jsonl", "a") as fh:
        fh.write(json.dumps(dict(rec, tag=tag, exit=r.returncode)) + "\n")
    return r.returncode, rec, executed, r.stderr[-2000:]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_runs_within_laptop_budget(name):
    rc, rec, executed, err = execute(name)
    print(f"{name}: wall {rec.get('wall_s')} s, peak RSS {rec.get('peak_rss_gib')} GiB -> {executed}")
    assert rc == 0, (rec.get("error"), err)
    assert rec["peak_rss_bytes"] > 100 * 2**20, rec           # the kernel was measured (a python with jax > 100 MiB)
    assert rec["peak_rss_bytes"] < RSS_LIMIT, rec
    assert rec["wall_s"] < WALL_BUDGET_S, rec


def test_planted_wrong_assert_fails_nbclient():
    text = (NB_DIR / "src" / "01_quickstart.py").read_text()
    assert text.count("MIN_DIGITS = 10") == 1
    rc, rec, _, err = execute("01_quickstart", text.replace("MIN_DIGITS = 10", "MIN_DIGITS = 17"),
                              tag="01_quickstart-planted")
    print(f"planted: exit {rc}, error {rec.get('error')}, wall {rec.get('wall_s')} s")
    assert rc != 0 and rec["error"]["ename"] == "AssertionError", (rec, err)


def _scan(text):
    """The FORBIDDEN classes found in a notebook's JSON text (image data left out: base64 is not text)."""
    nb = json.loads(text)
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            for k in [k for k in o.get("data", {}) if k.startswith("image/")]:
                o["data"][k] = ""
    text = json.dumps(nb)
    return sorted({what for what, pat in FORBIDDEN.items() if re.search(pat, text)})


def test_committed_notebooks_fresh_executed_and_clean():
    tool = _build_tool()
    assert tool.check() == []
    assert sorted(p.stem for p in tool.sources()) == sorted(NOTEBOOKS + CLUSTER_NOTEBOOKS)
    for name in NOTEBOOKS + CLUSTER_NOTEBOOKS:
        text = (NB_DIR / f"{name}.ipynb").read_text()
        nb = json.loads(text)
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert code and all(c["execution_count"] is not None for c in code), f"{name}: not executed"
        assert not [o for c in code for o in c["outputs"] if o["output_type"] == "error"], f"{name}: an error output"
        head = "".join(nb["cells"][0]["source"])
        assert nb["cells"][0]["cell_type"] == "markdown" and all(k in head for k in HEADER_KEYS), (name, head[:400])
        assert _scan(text) == [], (name, _scan(text))
    # negative controls: every planted sample is found in a committed notebook's text
    base = (NB_DIR / f"{NOTEBOOKS[0]}.ipynb").read_text()
    for s in PLANTED:
        nb = json.loads(base)
        nb["cells"][-1]["source"] = ["x = 1\n", f"# {s}\n"]
        assert _scan(json.dumps(nb)), s
