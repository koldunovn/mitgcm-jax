"""Repository-wide pytest setup.

1. The gate `XLA_FLAGS` (mitjax/xla_flags.py: no FMA, no algsimp, four fake CPU devices), set in one assignment
   before any test module runs jax code. XLA reads the variable once, when the first backend initialises; importing
   jax does not initialise one. test_env.py fails if the flags arrived too late or are not exactly the gate set.
2. Every collected test gets the marker of its file's cost group from mitjax/tests/manifest.py (merged from the
   per-area fragments mitjax/tests/manifest_<area>.py); a test file missing from the manifest is a collection error,
   not a silent default.
3. JAX's compilation caches are dropped at each module boundary (fesom_jax lessons §7: XLA's CPU JIT holds several
   memory mappings per executable and never releases them; a long suite hit ENOMEM at 86 % of vm.max_map_count,
   reported as "Cannot allocate memory", which reads as a numerical failure and is not).
4. MJX_MINMAX_STRICT=1 (mitjax/ops/fortran_minmax.strict): in our tests a verification build without a measured
   MAX/MIN winner at a build-dependent statement is refused, not given the users' default (plan decision 7, revised
   2026-10-07). Child processes inherit it, except the no-reference test's user environment (MJX_* stripped).
5. Tests that need data a user does not have -- the Fortran oracle ($MJX_REFERENCE, $MJX_REFERENCE_RUNS, the replay
   runs under $MJX_RUNS) or the lessons digests ($MJX_LESSONS) -- are found by static reading
   (mitjax/tests/oracle.py) and carry the `oracle` marker. Where that data is absent, such a file is not imported:
   it gets one test, `needs_oracle_data`, that skips naming the variable and why (a user's `pytest -m smoke` then
   reports skips, not collection errors). With MJX_REQUIRE_ORACLE=1 (levante.env) it fails instead: our tiers never
   turn a lost oracle into a green run of skips (docs plan 20261006 Task 11).

This file's directory is put first on sys.path by pytest, so `mitjax` is the package of this checkout (a worktree
tests its own code; env `mitjax` never has mitjax installed).
"""

import os
from pathlib import Path

import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()
os.environ["MJX_MINMAX_STRICT"] = "1"

REPO_ROOT = Path(__file__).resolve().parent


class NeedsOracleData(pytest.Item):
    """Stands for a test file whose data is absent (conftest item 5): skips, or fails under MJX_REQUIRE_ORACLE=1."""

    def __init__(self, *, reason, **kw):
        super().__init__(**kw)
        self.reason = reason

    def runtest(self):
        from mitjax.tests import oracle

        if oracle.required():
            pytest.fail(f"{oracle.REQUIRE_VAR}=1 and {self.reason}", pytrace=False)
        pytest.skip(self.reason)

    def reportinfo(self):
        return self.path, None, f"{self.path.name}::{self.name}"


class NeedsOracleFile(pytest.File):
    def __init__(self, *, reason, **kw):
        super().__init__(**kw)
        self.reason = reason

    def collect(self):
        yield NeedsOracleData.from_parent(self, name="needs_oracle_data", reason=self.reason)


@pytest.hookimpl(tryfirst=True)
def pytest_pycollect_makemodule(module_path, parent):
    from mitjax.tests import oracle

    try:
        rel = Path(module_path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None
    reasons = oracle.missing(rel)
    if reasons:
        return NeedsOracleFile.from_parent(parent, path=Path(module_path), reason=oracle.skip_reason(rel, reasons))
    return None


def pytest_configure(config):
    config.addinivalue_line("markers", "oracle: needs the Fortran oracle's data or the lessons digests "
                                       "(mitjax/tests/oracle.py); skipped where they are absent")


def pytest_collection_modifyitems(config, items):
    from mitjax.tests import oracle
    from mitjax.tests.manifest import MANIFEST

    unlisted = set()
    for item in items:
        rel = Path(str(item.fspath)).resolve().relative_to(REPO_ROOT).as_posix()
        group = MANIFEST.get(rel)
        if group is None:
            unlisted.add(rel)
            continue
        item.add_marker(getattr(pytest.mark, group))
        if oracle.classify(rel):
            item.add_marker(pytest.mark.oracle)
    if unlisted:
        raise pytest.UsageError(
            "test files missing from the test manifest (add each to its area's mitjax/tests/manifest_<area>.py): "
            + ", ".join(sorted(unlisted)))


@pytest.fixture(scope="module", autouse=True)
def _clear_jax_caches_per_module():
    yield
    import jax

    jax.clear_caches()
