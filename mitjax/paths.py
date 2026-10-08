"""Where the upstream MITgcm clone, the Fortran oracle, run output and caches live: the one place for machine paths.

Every location has an environment override. The defaults are neutral (docs plan 20261006 Task 5): no machine's path
is written here; this project's Levante values are in `levante.env` at the repository root, sourced by every batch
script and tier runner (`. levante.env` before running anything by hand on Levante).

    MJX_UPSTREAM        MITgcm clone at commit 63cdc0b (branch `pinned`)   REQUIRED (or $MJX_WORK/upstream/MITgcm)
    MJX_RUNS            job output (runs, tests, benchmarks)              $MJX_WORK/runs, else ./runs (at import)
    MJX_CACHE           derived tables (cpp link farms, exchange maps)    $MJX_WORK/cache, else $XDG_CACHE_HOME/mitjax
                                                                          (~/.cache/mitjax)
    MJX_REFERENCE       Fortran oracle (bin/, runs/, dumps/, ...)         $MJX_WORK/reference; optional (only our tests
                                                                          and the oracle tools read it)
    MJX_REFERENCE_RUNS  oracle run directories (mitjax/make_rundir.py)    $MJX_REFERENCE/runs
    MJX_DEV             lane worktrees (wt_<lane>)                        $MJX_WORK/dev
    MJX_LESSONS         lessons digests of the earlier ports              $MJX_WORK/lessons
    MJX_PYTHON          python of the environment                         the running interpreter
    MJX_WORK            optional work root the defaults above derive from (this project: levante.env)

A location without a value (MJX_UPSTREAM, or MJX_REFERENCE, MJX_DEV, MJX_LESSONS without MJX_WORK) is not an
attribute: reading it raises MissingPath naming the variable (an AttributeError, so `getattr(paths, name, None)`
tests for it).

Never reads `MITJAX_*`: that is the ECCO port's prefix, and a value left in the shell for it must not steer
this project. Values are read once, at import. Stdlib only and free of jax: tools that run in other environments load
this file by path, and batch scripts evaluate `python mitjax/paths.py --sh` (shell `export` lines; an unset location is
printed as a comment). Run without arguments it prints the resolved values.
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PREFIX = "MJX_"


class MissingPath(AttributeError):
    pass


def _env(name):
    assert name.startswith(PREFIX), name
    v = os.environ.get(name, "").strip()
    return Path(v).expanduser() if v else None


def _first(*cands):
    return next((c for c in cands if c is not None), None)


WORK = _env("MJX_WORK")
_W = (lambda *p: WORK.joinpath(*p)) if WORK is not None else (lambda *p: None)       # noqa: E731
_xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
_values = {
    "MJX_WORK": WORK,
    "MJX_UPSTREAM": _first(_env("MJX_UPSTREAM"), _W("upstream", "MITgcm")),
    "MJX_REFERENCE": _first(_env("MJX_REFERENCE"), _W("reference")),
    "MJX_RUNS": _first(_env("MJX_RUNS"), _W("runs"), Path.cwd() / "runs"),
    "MJX_CACHE": _first(_env("MJX_CACHE"), _W("cache"),
                        (Path(_xdg).expanduser() if _xdg else Path.home() / ".cache") / "mitjax"),
    "MJX_DEV": _first(_env("MJX_DEV"), _W("dev")),
    "MJX_LESSONS": _first(_env("MJX_LESSONS"), _W("lessons")),
    "MJX_PYTHON": _first(_env("MJX_PYTHON"), Path(sys.executable)),
}
_values["MJX_REFERENCE_RUNS"] = _first(_env("MJX_REFERENCE_RUNS"),
                                       _values["MJX_REFERENCE"] / "runs" if _values["MJX_REFERENCE"] else None)
_ORDER = ("MJX_WORK", "MJX_UPSTREAM", "MJX_REFERENCE", "MJX_REFERENCE_RUNS", "MJX_RUNS", "MJX_CACHE", "MJX_DEV",
          "MJX_LESSONS", "MJX_PYTHON")
ALL = {k: _values[k] for k in _ORDER if _values[k] is not None}
_HOW = {"MJX_UPSTREAM": "the MITgcm checkout at 63cdc0b, e.g. `git clone https://github.com/MITgcm/MITgcm && "
                        "git -C MITgcm checkout 63cdc0b && export MJX_UPSTREAM=$PWD/MITgcm`",
        "MJX_REFERENCE": "the Fortran oracle's data; optional: only the oracle comparisons need it"}
for _k, _v in ALL.items():
    globals()[_k[len(PREFIX):]] = _v


def __getattr__(name):
    key = PREFIX + name
    if key in _ORDER:
        raise MissingPath(f"{key} is not set ({_HOW.get(key, 'no default without MJX_WORK')}); on Levante: "
                          f"`. {REPO / 'levante.env'}` (mitjax/paths.py)")
    raise AttributeError(f"module 'mitjax.paths' has no attribute {name!r}")


if __name__ == "__main__":
    import shlex

    sh = "--sh" in sys.argv[1:]
    for k in _ORDER:
        v = _values[k]
        if sh:
            print(f"export {k}={shlex.quote(str(v))}" if v is not None else f"# {k} unset")
        else:
            print(f"{k:20s} {v if v is not None else '(unset)'}")
