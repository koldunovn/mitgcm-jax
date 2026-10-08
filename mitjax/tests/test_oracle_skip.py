"""Oracle-dependent tests skip cleanly without the oracle (docs plan 20261006 Task 11; mitjax/tests/oracle.py,
conftest.py item 5).

1. The static reading: known files classified as they are (tier-1 oracle gates need the oracle; pure unit tests do
   not), and on a planted tree each way of reaching the data is found -- a direct read, a helper function the test
   calls, a helper's read at import, a function passed on uncalled, a replay pointer file -- while a docstring that
   only names $MJX_REFERENCE is not (negative controls both ways).
2. A user's whole suite: `pytest --collect-only` over every test directory with only MJX_UPSTREAM, MJX_RUNS and
   MJX_CACHE set collects with no error, every oracle file as one `needs_oracle_data` item; and pointing
   MJX_REFERENCE / MJX_LESSONS at an empty directory gives the same set.
3. Running such a file: it skips with a reason naming the variable; with MJX_REQUIRE_ORACLE=1 it fails (control:
   the strict mode bites), and an invalid value is refused.
Costs: four pytest subprocesses (collection only, plus two one-file runs), about a minute.
"""

import os
import subprocess
import sys

import pytest

from mitjax import paths
from mitjax.tests import oracle

REPO = paths.REPO


def _user_env(tmp_path, **extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MJX_")}
    env.update(MJX_UPSTREAM=str(paths.UPSTREAM), MJX_RUNS=str(tmp_path / "runs"), MJX_CACHE=str(tmp_path / "cache"),
               JAX_PLATFORMS="cpu", PYTHONPATH=str(REPO), **extra)
    return env


def _pytest(env, *args):
    return subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args], cwd=str(REPO), env=env,
                          capture_output=True, text=True)


def test_classification_of_known_files():
    assert set(oracle.classify("mitjax/tests/test_r1_barotropic_gyre_tier1.py")) >= {"REFERENCE_RUNS"}
    assert "REFERENCE" in oracle.classify("mitjax/tests/test_testreport_jax.py")      # read at import
    assert "LESSONS" in oracle.classify("mitjax/tests/test_lessons_carried.py")
    assert any(k.startswith("REPLAY:") for k in oracle.classify("mitjax/tests/test_colmix.py"))
    for clean in ("mitjax/tests/test_farray.py", "mitjax/tests/test_manifest.py", "mitjax/tests/test_safe_ops.py",
                  "mitjax/tests/test_docs_pages.py", "mitjax/tests/test_paths.py",
                  "mitjax/tests/test_public_tree.py"):
        assert oracle.classify(clean) == {}, (clean, oracle.classify(clean))


PLANTED = {
    "mitjax/tests/helper_a.py": (
        "from mitjax import paths\n"
        "ROOT = paths.REFERENCE_RUNS / 'x'\n"),                          # read at import
    "mitjax/tests/helper_b.py": (
        "from mitjax import paths as P\n"
        "def dumps(exp):\n"
        "    return P.REFERENCE / 'dumps' / exp\n"
        "def harmless():\n"
        "    return 1\n"),
    "mitjax/tests/test_direct.py": (
        "from mitjax import paths\n"
        "def test_x():\n"
        "    assert getattr(paths, 'REFERENCE').exists()\n"),
    "mitjax/tests/test_via_call.py": (
        "from mitjax.tests import helper_b as hb\n"
        "def test_x():\n"
        "    assert hb.dumps('e')\n"),
    "mitjax/tests/test_via_name.py": (
        "from mitjax.tests.helper_b import dumps\n"
        "def run(fn):\n"
        "    return fn('e')\n"
        "def test_x():\n"
        "    assert run(dumps)\n"),                                      # passed on, called elsewhere
    "mitjax/tests/test_via_import.py": (
        "from mitjax.tests import helper_a\n"
        "def test_x():\n"
        "    assert True\n"),
    "mitjax/tests/test_replay.py": (
        "from pathlib import Path\n"
        "def test_x():\n"
        "    assert (Path('.') / 'reference' / 'replay_zz' / 'CURRENT').read_text()\n"),
    "reference/replay_zz/CURRENT": "# runs (relative to $MJX_REFERENCE)\nexp input replay_zz/run1/rundir\n",
    "mitjax/tests/test_docstring_only.py": (
        '"""Needs no $MJX_REFERENCE and never reads paths.REFERENCE (a docstring only)."""\n'
        "from mitjax.tests import helper_b as hb\n"
        "def test_x():\n"
        "    # paths.REFERENCE in a comment\n"
        "    assert hb.harmless() == 1\n"),
}


def test_planted_tree_each_route_found(tmp_path, monkeypatch):
    for rel, text in PLANTED.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    monkeypatch.setattr(oracle, "REPO", tmp_path)
    oracle.clear()
    try:
        got = {rel: set(oracle.classify(rel)) for rel in PLANTED if "/test_" in rel}
    finally:
        monkeypatch.undo()
        oracle.clear()
    assert got == {
        "mitjax/tests/test_direct.py": {"REFERENCE"},
        "mitjax/tests/test_via_call.py": {"REFERENCE"},
        "mitjax/tests/test_via_name.py": {"REFERENCE"},
        "mitjax/tests/test_via_import.py": {"REFERENCE_RUNS"},
        "mitjax/tests/test_replay.py": {"REPLAY:replay_zz"},
        "mitjax/tests/test_docstring_only.py": set(),
    }, got


def _collected(out):
    lines = [ln for ln in out.splitlines() if "::" in ln]
    stubs = sorted(ln.split("::")[0] for ln in lines if ln.endswith("::needs_oracle_data"))
    return lines, stubs


def test_user_collection_has_no_errors(tmp_path):
    from mitjax.tests.manifest import MANIFEST

    expected = sorted(f for f in MANIFEST if oracle.classify(f))
    r = _pytest(_user_env(tmp_path), "--collect-only", "-q")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert "error" not in r.stdout.splitlines()[-1].lower(), r.stdout[-500:]
    _, stubs = _collected(r.stdout)
    assert stubs == expected, sorted(set(stubs) ^ set(expected))
    empty = tmp_path / "empty"
    empty.mkdir()
    r2 = _pytest(_user_env(tmp_path, MJX_REFERENCE=str(empty), MJX_LESSONS=str(empty)), "--collect-only", "-q")
    assert r2.returncode == 0, r2.stdout[-3000:]
    assert _collected(r2.stdout)[1] == expected


def test_skip_names_the_variable_and_strict_mode_fails(tmp_path, monkeypatch):
    target = "mitjax/tests/test_r1_barotropic_gyre_tier1.py"
    r = _pytest(_user_env(tmp_path), "-q", "-rs", target)
    assert r.returncode == 0, r.stdout[-2000:]
    assert "1 skipped" in r.stdout and "$MJX_REFERENCE_RUNS is not set" in r.stdout, r.stdout[-2000:]
    r = _pytest(_user_env(tmp_path, MJX_REQUIRE_ORACLE="1"), "-q", target)          # control: strict mode bites
    assert r.returncode != 0 and "1 failed" in r.stdout and "MJX_REQUIRE_ORACLE=1" in r.stdout, r.stdout[-2000:]
    monkeypatch.setenv("MJX_REQUIRE_ORACLE", "yes")
    with pytest.raises(ValueError, match="MJX_REQUIRE_ORACLE"):
        oracle.required()
