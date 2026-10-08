"""Every test file sits in exactly one cost group, the fragments merge without overlap, every test directory is
collected by the runner, and the audits that say so can fail."""

import tomllib
from pathlib import Path

import pytest

from mitjax.tests.manifest import (FRAGMENTS, GROUPS, MANIFEST, TEST_DIRS, TIER1_EXPRESSION, audit, load_fragments,
                                   merge_fragments)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_manifest_matches_files_on_disk():
    unlisted, missing, bad_group, outside = audit(REPO_ROOT)
    assert unlisted == [], f"test files in no manifest fragment: {unlisted}"
    assert missing == [], f"manifest lists files that do not exist: {missing}"
    assert bad_group == [], f"unknown cost group (allowed {GROUPS}): {bad_group}"
    assert outside == [], f"test files outside the collected directories {TEST_DIRS}: {outside}"


def test_runner_collects_every_test_dir():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert tuple(pyproject["tool"]["pytest"]["ini_options"]["testpaths"]) == TEST_DIRS
    # the tier-1 runner selects by marker over the default testpaths, never by a hand-kept directory list
    runner = (REPO_ROOT / "scripts/run_tier1.sbatch").read_text()
    assert '-m "smoke or tier1"' in runner and not any(d in runner for d in TEST_DIRS)
    for d in TEST_DIRS:
        assert any(p.startswith(d + "/") for p in MANIFEST), f"no listed test under {d}"
    # a group that nothing runs is no guard (fesom_jax lessons §7): every CPU group in use has its runner
    tier1x = (REPO_ROOT / "scripts/run_tier1x.sbatch").read_text()
    assert '-m "tier1x"' in tier1x and not any(d in tier1x for d in TEST_DIRS)
    assert f'-m "{TIER1_EXPRESSION}"' in runner


def runner_problems(text):
    """A tier runner must run the tests from a frozen `git archive` snapshot of HEAD in its OUT_DIR (jobs run from a
    frozen snapshot: the milestone-0 review found the tier runners testing the live checkout), say loudly when the
    checkout is dirty, and (tier 1) take its test budget from manifest.TIER1_MAX_TESTS, never a literal (the same
    review found the constant unused and the budget a literal `--max-tests 100` in the runner)."""
    probs = []
    if 'git -C "$REPO" archive --format=tar "$COMMIT" | tar -x -C "$SRC"' not in text:
        probs.append("no git archive snapshot of HEAD")
    if 'cd "$SRC"' not in text or "export PYTHONPATH=$SRC" not in text or 'cd "$REPO"' in text:
        probs.append("tests do not run in the snapshot")
    if "NOT tested" not in text:
        probs.append("no loud note that uncommitted changes are not tested")
    if "--max-tests" in text and ("TIER1_MAX_TESTS" not in text or '--max-tests "$MAX_TESTS"' not in text):
        probs.append("--max-tests not read from manifest.TIER1_MAX_TESTS")
    if "rm " in text or "rm -" in text:
        probs.append("deletes something")
    return probs


def test_tier_runners_test_a_frozen_snapshot():
    for name in ("run_tier1.sbatch", "run_tier1x.sbatch"):
        assert runner_problems((REPO_ROOT / "scripts" / name).read_text()) == [], name
    assert "--max-tests" in (REPO_ROOT / "scripts/run_tier1.sbatch").read_text()
    # negative controls: the pre-hardening runner (live checkout, literal budget) is caught on every count
    old = ('cd "$REPO"\nexport PYTHONPATH=$REPO\n"$PY" -m pytest -m "smoke or tier1"\n'
           '"$PY" scripts/check_pytest_report.py "$OUT_DIR/report.xml" --max-tests 100\n')
    assert runner_problems(old) == ["no git archive snapshot of HEAD", "tests do not run in the snapshot",
                                    "no loud note that uncommitted changes are not tested",
                                    "--max-tests not read from manifest.TIER1_MAX_TESTS"]


def test_fragments_merged():
    assert "manifest_core" in FRAGMENTS
    assert MANIFEST == merge_fragments(FRAGMENTS)
    assert sum(len(f) for f in FRAGMENTS.values()) == len(MANIFEST)


def test_audit_negative_controls(tmp_path):
    (tmp_path / "mitjax/tests").mkdir(parents=True)
    (tmp_path / "scripts/tests").mkdir(parents=True)
    (tmp_path / "tools/tests").mkdir(parents=True)
    (tmp_path / "mitjax/tests/test_listed.py").touch()
    (tmp_path / "scripts/tests/test_planted.py").touch()
    (tmp_path / "tools/tests/test_stray.py").touch()
    manifest = {
        "mitjax/tests/test_listed.py": "tier1",
        "mitjax/tests/test_gone.py": "smoke",
        "mitjax/tests/test_typo.py": "tier9",
    }
    unlisted, missing, bad_group, outside = audit(tmp_path, manifest)
    assert unlisted == ["scripts/tests/test_planted.py"]
    assert missing == ["mitjax/tests/test_gone.py", "mitjax/tests/test_typo.py"]
    assert bad_group == ["mitjax/tests/test_typo.py"]
    assert outside == ["tools/tests/test_stray.py"]


def test_fragment_negative_controls(tmp_path):
    (tmp_path / "manifest_a.py").write_text('MANIFEST = {"mitjax/tests/test_x.py": "smoke"}\n')
    (tmp_path / "manifest_b.py").write_text('MANIFEST = {"mitjax/tests/test_x.py": "tier1"}\n')
    with pytest.raises(ValueError, match="listed by both"):
        merge_fragments(load_fragments(tmp_path))
    (tmp_path / "manifest_b.py").write_text('OTHER = {}\n')
    with pytest.raises(ValueError, match="no MANIFEST"):
        load_fragments(tmp_path)


def test_every_collected_test_carries_its_group_marker(request):
    group = MANIFEST["mitjax/tests/test_manifest.py"]
    assert request.node.get_closest_marker(group) is not None
