"""Which test files need data a user does not have, and the skip that names it (docs plan 20261006 Task 11).

Most tests compare with the project's Fortran oracle ($MJX_REFERENCE: builds, runs, dumps, exchange maps, MAX/MIN
sites) or read the earlier ports' lessons digests ($MJX_LESSONS). A user has neither. conftest.py asks `missing()`
for every test file before importing it: when the data a file needs is absent, the file is not imported (several
read the oracle at import) and gets one test, `needs_oracle_data`, that skips with a reason naming the variable and
why it is unusable. Every test of such a file carries the `oracle` marker, so `pytest -m "not oracle"` selects the
tests that need neither.

Which files need the data is decided by static reading (`classify`), so a new test needs no registration:
  * a direct read in the file: `paths.REFERENCE`, `paths.REFERENCE_RUNS`, `paths.LESSONS` (also through
    `getattr(paths, "...")`), or a call of `load_maps` / `map_dir` (the oracle's exchange maps,
    mitjax/eesupp/exch_maps.py:213) or `oracle_toolchain` (the oracle's build records, mitjax/config/cpp_options.py);
  * through a helper module of the repository it imports (mitjax/tests/*, scripts/*, reference/*, tools/*, also
    when loaded by path): a read at the helper's import, or a call of a helper function that reads (a fixpoint over
    the functions of all helpers; by name, so conservative).
Docstrings and comments do not count (the scan walks the syntax tree). The package itself (mitjax/ outside tests)
does not read the oracle at run time (test_no_reference.py), so it is not followed. NOT_ORACLE lists the files the
reading flags wrongly, with the reason.

The data is absent when the variable is unset (mitjax/paths.py: MissingPath), or names a directory that does not
exist or is empty. With MJX_REQUIRE_ORACLE=1 (levante.env sets it: our tiers must never turn a lost oracle into
skips) such a file FAILS with the same message instead of skipping.

Stdlib only (conftest imports it before any test module).
"""

import ast
import os
import re
from functools import lru_cache
from pathlib import Path

from mitjax import paths

REPO = paths.REPO
REQUIRE_VAR = "MJX_REQUIRE_ORACLE"

# paths attribute -> the variable a user would set, and what it holds
DATA = {
    "REFERENCE": ("MJX_REFERENCE", "the Fortran oracle's data (builds, runs, dumps, exchange maps, MAX/MIN sites)"),
    "REFERENCE_RUNS": ("MJX_REFERENCE_RUNS", "the Fortran oracle's run directories"),
    "LESSONS": ("MJX_LESSONS", "the earlier ports' lessons digests (project data, not published)"),
}
ORACLE_CALLS = {"load_maps": "REFERENCE", "map_dir": "REFERENCE", "oracle_toolchain": "REFERENCE"}
HELPER_ROOTS = ("mitjax/tests", "scripts", "reference", "tools")
PATHS_ALIASES = ("paths", "P")      # `from mitjax import paths [as P]` (the two spellings in the repository)

# Files the static reading flags although they read no such data (path -> reason).
NOT_ORACLE = {
    "mitjax/tests/test_paths.py": "checks paths.py's own resolution in subprocesses with planted values; reads no "
                                  "oracle or lessons data",
    "mitjax/tests/test_public_tree.py": "loads tools/make_public_tree.py, whose content table names scripts as "
                                        "strings (read as by-path loads); builds throw-away repositories only "
                                        "(flagged wrongly in the empty-reference tier 1, job 27963596)",
}


def _facts(node):
    """(paths attributes read, references made) by an AST node, in one walk. Reads: paths.X, mitjax.paths.X,
    getattr(paths, "X"). References, called or not (a function passed on or stored may run later): ("name", f) for a
    loaded name, ("attr", base, f) for base.f with a plain name as base."""
    reads, refs = set(), set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            if isinstance(n.ctx, ast.Load):
                refs.add(("name", n.id))
        elif isinstance(n, ast.Attribute):
            v = n.value
            if isinstance(v, ast.Name):
                refs.add(("attr", v.id, n.attr))
            if n.attr in DATA and ((isinstance(v, ast.Name) and v.id in PATHS_ALIASES)
                                   or (isinstance(v, ast.Attribute) and v.attr == "paths")):
                reads.add(n.attr)
        elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "getattr"
              and len(n.args) >= 2 and isinstance(n.args[1], ast.Constant) and n.args[1].value in DATA):
            reads.add(n.args[1].value)
    return reads, refs


def _module_file(name):
    parts = name.split(".")
    for cand in (REPO.joinpath(*parts).with_suffix(".py"), REPO.joinpath(*parts, "__init__.py")):
        if cand.is_file():
            rel = cand.relative_to(REPO).as_posix()
            if rel.startswith(HELPER_ROOTS):
                return rel
    return None


_BY_PATH = re.compile(r"[\"']((?:scripts|reference|tools)(?:/[\w.-]+)*\.py)[\"']|"
                      r"[\"'](scripts|reference|tools)[\"']\s*/\s*[\"']([\w/.-]+\.py)[\"']")


@lru_cache(maxsize=None)
def _parse(rel):
    """Parsed facts of a file: (ok, {alias: (module, name or None)}, modules loaded by path, module-level reads,
    module-level calls, {function or class name: (reads, calls)})."""
    src = (REPO / rel).read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False, {}, (), frozenset(), frozenset(), {}
    imports, consts = {}, set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant):
            if isinstance(n.value, str):
                consts.add(n.value)
        elif isinstance(n, ast.Import):
            for a in n.names:
                f = _module_file(a.name)
                if f and f != rel:
                    imports[a.asname or a.name.split(".")[0]] = (f, None)
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            for a in n.names:
                sub = _module_file(f"{n.module}.{a.name}")
                if sub and sub != rel:
                    imports[a.asname or a.name] = (sub, None)
                    continue
                f = _module_file(n.module)
                if f and f != rel:
                    imports[a.asname or a.name] = (f, a.name)
    bypath = set()
    for m in _BY_PATH.finditer(src):
        f = m.group(1) or f"{m.group(2)}/{m.group(3)}"
        if (REPO / f).is_file() and f != rel:
            bypath.add(f)
    for c in consts:                      # a script run by name (e.g. in a subprocess): scripts/<c> or tools/<c>
        if re.fullmatch(r"[\w-]+\.py", c):
            for root in ("scripts", "tools"):
                if (REPO / root / c).is_file() and f"{root}/{c}" != rel:
                    bypath.add(f"{root}/{c}")
    replay = {c for c in consts if re.fullmatch(r"replay\w*", c) and (REPO / "reference" / c).is_dir()}
    replay |= {m.group(1) for c in consts for m in re.finditer(r"reference/(replay\w*)/", c)}
    funcs, top_reads, top_calls = {}, {f"REPLAY:{d}" for d in replay if _pointer_files(d)}, set()
    for stmt in tree.body:
        if (isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Compare)
                and isinstance(stmt.test.left, ast.Name) and stmt.test.left.id == "__name__"):
            continue                       # `if __name__ == "__main__":` does not run at import
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            reads, refs = _facts(stmt)
            funcs[stmt.name] = (frozenset(reads), frozenset(refs))
            for d in stmt.decorator_list:          # decorators run at import
                reads, refs = _facts(d)
                top_reads |= reads
                top_calls |= refs
        else:
            reads, refs = _facts(stmt)
            top_reads |= reads
            top_calls |= refs
    return True, imports, tuple(sorted(bypath)), frozenset(top_reads), frozenset(top_calls), funcs


def _resolve(rel, call):
    """The (module, function) targets a call can reach: same module, explicit imports, by-path modules."""
    ok, imports, bypath, _, _, funcs = _parse(rel)
    if call[0] == "name":
        f = call[1]
        if f in funcs:
            return [(rel, f)]
        if f in imports and imports[f][1] is not None:
            return [imports[f]]
        return []
    _, base, f = call
    if base in imports and imports[base][1] is None:
        return [(imports[base][0], f)]
    # base is a local value (e.g. a module returned by a loader): any helper this file reaches that defines f
    return [(m, f) for m in _defined_below(rel).get(f, ())]


@lru_cache(maxsize=None)
def _defined_below(rel):
    """{function name: helper modules defining it} over the helpers a file reaches."""
    out = {}
    for m in _closure(rel)[1:]:
        for f in _parse(m)[5]:
            out.setdefault(f, []).append(m)
    return out


def _deps(rel):
    _, imports, bypath, *_ = _parse(rel)
    return sorted({m for m, _ in imports.values()} | set(bypath))


@lru_cache(maxsize=None)
def _closure(rel):
    seen, todo = [], [rel]
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.append(f)
        todo += [d for d in _deps(f) if d not in seen]
    return tuple(seen)


_READS, _ACTIVE = {}, set()


def _reads_of(rel, func):
    """Data keys a function (or class) of a module reads, directly or through what it references (resolved; a
    recursion in progress counts as nothing)."""
    key = (rel, func)
    if key in _READS:
        return _READS[key]
    if key in _ACTIVE:
        return frozenset()
    _ACTIVE.add(key)
    funcs = _parse(rel)[5]
    if func not in funcs:
        out = {ORACLE_CALLS[func]} if func in ORACLE_CALLS else set()
    else:
        reads, calls = funcs[func]
        out = set(reads)
        for c in calls:
            if c[-1] in ORACLE_CALLS:
                out.add(ORACLE_CALLS[c[-1]])
            for m, f in _resolve(rel, c):
                if (m, f) != key:
                    out |= _reads_of(m, f)
    _ACTIVE.discard(key)
    _READS[key] = frozenset(out)
    return _READS[key]


def _import_reads(rel):
    """Data read when a module is imported (its module-level code and the calls it makes)."""
    _, _, _, top, calls, _ = _parse(rel)
    out = set(top)
    for c in calls:
        if c[-1] in ORACLE_CALLS:
            out.add(ORACLE_CALLS[c[-1]])
        for m, f in _resolve(rel, c):
            out |= _reads_of(m, f)
    return out


@lru_cache(maxsize=None)
def classify(rel):
    """{data key: [evidence]} a test file needs (empty = none). Evidence names the file and the read."""
    if rel in NOT_ORACLE:
        return {}
    need = {}
    for f in _closure(rel):
        for k in _import_reads(f):
            need.setdefault(k, []).append(f"{f} (at import)")
    for name in _parse(rel)[5]:
        for k in _reads_of(rel, name):
            need.setdefault(k, []).append(f"{rel}::{name}")
    return {k: sorted(set(v)) for k, v in need.items()}


def _pointer_files(replay_dir):
    """reference/<replay_dir>/CURRENT* and JDALL: the replay runs a harness test reads (relative to $MJX_RUNS)."""
    d = REPO / "reference" / replay_dir
    return sorted(list(d.glob("CURRENT*")) + list(d.glob("JDALL")))


def _replay_unusable(replay_dir):
    """The first run a pointer file lists, when it is under neither $MJX_REFERENCE nor $MJX_RUNS (the files say
    which root in their header; some do not, so both are tried)."""
    roots = [getattr(paths, k, None) for k in ("REFERENCE", "RUNS")]
    for f in _pointer_files(replay_dir):
        lines = [ln.split() for ln in f.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        if not lines:
            continue
        run = lines[0][-1]
        if not any(r is not None and (Path(r) / run).exists() for r in roots):
            return (f"the Fortran oracle's replay run {run} (listed in {f.relative_to(REPO).as_posix()}) is under "
                    f"neither $MJX_REFERENCE nor $MJX_RUNS")
    return None


def _unusable(key):
    """Why the data behind paths.<key> (or a replay harness's runs) is unusable, or None when it is there."""
    if key.startswith("REPLAY:"):
        return _replay_unusable(key.split(":", 1)[1])
    var, what = DATA[key]
    try:
        p = getattr(paths, key)
    except AttributeError:
        return f"${var} is not set ({what})"
    if not p.is_dir():
        return f"${var}={p} does not exist ({what})"
    try:
        next(p.iterdir())
    except StopIteration:
        return f"${var}={p} is empty ({what})"
    return None


def missing(rel):
    """[] when the file needs nothing absent; else one reason per absent variable."""
    return [r for r in (_unusable(k) for k in sorted(classify(rel))) if r]


def required():
    v = os.environ.get(REQUIRE_VAR, "").strip()
    if v not in ("", "0", "1"):
        raise ValueError(f"{REQUIRE_VAR}={v!r}: use 1 (missing oracle data is an error) or 0 / unset (a skip)")
    return v == "1"


def skip_reason(rel, reasons):
    return (f"needs data this machine does not have: {'; '.join(reasons)} -- {rel} was not imported "
            f"(mitjax/tests/oracle.py)")


def clear():
    """Drop the caches (a test that points REPO at a planted tree)."""
    for f in (_parse, _closure, _defined_below, classify):
        f.cache_clear()
    _READS.clear()
    _ACTIVE.clear()
