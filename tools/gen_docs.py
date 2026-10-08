#!/usr/bin/env python3
"""Generated reference pages of docs/ (docs plan 20261006: Task 7 `docs/fortran_map.md`; Task 10 adds its pages to
PAGES).

    python tools/gen_docs.py              write every page (needs $MJX_UPSTREAM; on Levante `. ./levante.env` first)
    python tools/gen_docs.py --check      write nothing; exit 1 when a committed page differs from its regeneration
    python tools/gen_docs.py --stats      the numbers behind the map (citations, rows, coverage) on stderr

Stdlib only, no jax and no `mitjax` import (mitjax/paths.py is loaded by file path): the pages are reproducible on a
user's machine from this repository and an MITgcm checkout at 63cdc0b. The Fortran oracle ($MJX_REFERENCE) is never
read. A page is a pure function of the repository's tracked sources and the upstream files, so the freshness test
(mitjax/tests/test_docs_generated.py) regenerates it in memory and compares byte for byte.

docs/fortran_map.md is built from the citations the code carries (docs/KERNEL_GUIDE.md §1):
  * a PORT CITATION is an upstream path next to the pin tag in the head (the text before the first blank line) of a
    module, top-level function or class docstring: `@63cdc0b <path>[:<a>[-<b>]]` or `<path>[:<a>[-<b>]] @63cdc0b`.
    `<path>` is `model/src/x.F`, `pkg/<p>/x.F`, `eesupp/src/x.F`, `verification/<exp>/<code dir>/x.F` (also `.h`,
    `.F90`), or a bare `x.F` resolved in the Python file's mirror directory, else by a unique file name upstream.
    A cited eesupp/exch2 file that genmake2 generates from a template (`exch_xy_rl.F` from `exch_xy_rx.template`) is
    mapped to the template;
  * every port citation is VALIDATED against the upstream checkout: the file must exist and the lines must be inside
    it, else generation stops (CitationError) -- the map never shows a citation the Fortran does not have;
  * the Fortran routine of a cited span is read from the upstream file (the SUBROUTINE / FUNCTION statements inside
    the span; none inside: the routine the span is part of);
  * header modules (`*_h.py`, or a module docstring that starts with a header name) map the `.h` files named in the
    head of their docstring.
Body citations (`# :NN`) are not read here: they point inside the routine the docstring cites.
"""

import argparse
import ast
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PIN = "63cdc0b"
PIN_FULL = "63cdc0b9602b46bda69df37c5eb9e17396116f67"
MITGCM_BLOB = f"https://github.com/MITgcm/MITgcm/blob/{PIN_FULL}/"
REGENERATE = "python tools/gen_docs.py"

# Python sources scanned for citations (the repository's package; tests are not ports)
CODE_ROOT = "mitjax"
CODE_SKIP = ("mitjax/tests/",)

# upstream directories whose files the map lists (model/src, model/inc, eesupp, every package, experiment code dirs)
SRC_EXT = (".F", ".F90", ".template")
HDR_EXT = (".h",)

_DIR = (r"(?:model/(?:src|inc)|eesupp/(?:src|inc)|pkg/[A-Za-z0-9_]+"
        r"|verification/[A-Za-z0-9_.+-]+/[A-Za-z0-9_.]+)/")
_FILE = r"[A-Za-z0-9_]+\.(?:F90|F|h|template)(?![A-Za-z0-9_])"
_LINES = r"(?::[ \t]*\n?[ \t]*(\d+)(?:[ \t]*-[ \t]*(\d+))?(?![\d.]))?"
PATH_RE = r"(?<![A-Za-z0-9_./-])((?:" + _DIR + r")?" + _FILE + r")" + _LINES
TAG = "@" + PIN
_WS = r"[ \t]*\n?[ \t]*"
_PATH_NC = re.sub(r"\((?!\?)", "(?:", PATH_RE)                      # the same pattern without capturing groups
_PATH_LIST = _PATH_NC + r"(?:" + _WS + r"(?:,|and|/)" + _WS + _PATH_NC + r")*"
# `@63cdc0b <path>[:a-b][, <path>[:a-b] ...]` and `<path>[:a-b][, ...] @63cdc0b`
CIT_AFTER = re.compile(re.escape(TAG) + _WS + r"(" + _PATH_LIST + r")")
CIT_BEFORE = re.compile(r"(" + _PATH_LIST + r")" + _WS + re.escape(TAG) + r"(?![A-Za-z0-9])")
ONE_PATH = re.compile(PATH_RE)
HEADER_TOKEN = re.compile(r"(?<![A-Za-z0-9_./-])((?:" + _DIR + r")?[A-Za-z0-9_]+\.h)(?![A-Za-z0-9_])")
_HDR = r"(?:" + _DIR + r")?[A-Za-z0-9_]+\.h(?![A-Za-z0-9_])"
HEADER_TITLE = re.compile(r"\s*(" + _HDR + r"(?:\s*(?:,|and|/)\s*" + _HDR + r")*)")
UNIT = re.compile(r"^[ \t]+(?:[\w*() ]+?\s)?(SUBROUTINE|FUNCTION|PROGRAM)\s+(\w+)", re.I)
LEAD_NAME = re.compile(r"\s*(?:[A-Za-z_*0-9]+\s+)*?(?:SUBROUTINE\s+|FUNCTION\s+)?([A-Z][A-Z0-9_]*)\s*\(")
TEMPLATE_SUFFIX = re.compile(r"_(rl|rs|r4|r8|rx)(?=(?:_[a-z0-9]+)*\.F$)")


class CitationError(ValueError):
    """A port citation names a file or a line the upstream checkout at 63cdc0b does not have."""


# Port citations measured wrong on 2026-10-07 (lane FMAP): the span ends past the end of the file at 63cdc0b
# ({(Python file, upstream file:span): the file's line count}). They are mapped to the whole file, without the span,
# until the docstring is corrected; an entry that no longer occurs is an error, so this list only shrinks.
# Lane DOCS5 (2026-10-07) corrected the spans whose module's body citations (`# :NN`) it checked against the Fortran at
# 63cdc0b (do_stagger_fields_exchanges, gad_dst2u1_impl_r, salt_plume_readparms, diagnostics_is_on in
# ini_parms_tracer, cost_averagesgeneric in cost_averagesfields). The entries below need more than a docstring edit:
KNOWN_BAD = {
    # body citations past the end of the file too (the module was cited from an older file version)
    ("mitjax/model/src/freesurf_rescale_g.py", "model/src/freesurf_rescale_g.F:6-87"): 76,
    ("mitjax/pkg/cd_code/cd_code_ini_vars.py", "pkg/cd_code/cd_code_ini_vars.F:3-67"): 59,
    ("mitjax/pkg/cd_code/cd_code_write_pickup.py", "pkg/cd_code/cd_code_write_pickup.F:8-99"): 85,
    # body citations inside the file but on other statements (shifted by 1-3 lines or more: checked line by line)
    ("mitjax/model/src/read_pickup.py", "model/src/check_pickup.F:8-268"): 253,
    ("mitjax/pkg/autodiff/adjoint_monitor.py", "pkg/monitor/monitor_ad.F:9-305"): 260,
    ("mitjax/pkg/cd_code/cd_code_scheme.py", "pkg/cd_code/cd_code_scheme.F:7-252"): 246,
    ("mitjax/pkg/cost/cost_tile.py", "pkg/cost/cost_tile.F:57-155"): 152,
    ("mitjax/pkg/ecco/cost_averagesfields.py", "pkg/ecco/cost_gencost_assignperiod.F:3-110"): 100,
    ("mitjax/pkg/gmredi/gmredi_write_pickup.py", "pkg/gmredi/gmredi_write_pickup.F:7-221"): 220,
    # ctrl_toolbox: the body's :182-184 are one line early, and the port's loops (no k loop) differ from :181-185
    ("mitjax/pkg/ctrl/ctrl_toolbox.py", "pkg/ctrl/ctrl_toolbox.F:154-197"): 195,
    # three of eleven body citations point at blank lines (:69, :88, :117); not settled
    ("mitjax/pkg/ptracers/ptracers_monitor_ad.py", "pkg/ptracers/ptracers_monitor_ad.F:7-148"): 147,
    # the module ports both files; its body citations were not attributed to one file and not checked
    ("mitjax/pkg/ecco/ecco_cost_init_fixed.py", "pkg/ecco/ecco_init_fixed.F:8-52"): 42,
    ("mitjax/pkg/ecco/ecco_cost_init_fixed.py", "pkg/ecco/ecco_cost_init_fixed.F:7-250"): 249,
}
KNOWN_BAD_SEEN = set()


# ---------------------------------------------------------------------------------------------------------------
# upstream checkout


def _paths_module():
    spec = importlib.util.spec_from_file_location("_mjx_paths_gen_docs", REPO / "mitjax" / "paths.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Upstream:
    """The MITgcm checkout at 63cdc0b ($MJX_UPSTREAM), read only."""

    def __init__(self, root=None):
        self.root = Path(root) if root is not None else Path(_paths_module().UPSTREAM)
        if not (self.root / "model" / "src").is_dir():
            raise FileNotFoundError(f"{self.root} is not an MITgcm checkout (no model/src)")
        if (self.root / ".git").exists():
            r = subprocess.run(["git", "-C", str(self.root), "rev-parse", "HEAD"], capture_output=True, text=True)
            if r.returncode == 0 and not r.stdout.strip().startswith(PIN):
                raise CitationError(f"{self.root} is at {r.stdout.strip()[:12]}, not {PIN}: the map cites {PIN}")
        self._lines, self._units = {}, {}
        self.files = self._index()
        self.verification_dirs = {}       # (python package name, code dir) -> upstream verification/<exp>/<code dir>
        for d in sorted((self.root / "verification").glob("*/code*")):
            self.verification_dirs[(re.sub(r"[^A-Za-z0-9_]", "_", d.parent.name), d.name)] = \
                d.relative_to(self.root).as_posix()
        self.by_name = {}
        for rel in self.files:
            self.by_name.setdefault(rel.rsplit("/", 1)[-1], []).append(rel)

    def _index(self):
        out = set()
        dirs = [self.root / "model" / "src", self.root / "model" / "inc", self.root / "eesupp" / "src",
                self.root / "eesupp" / "inc"]
        dirs += sorted(p for p in (self.root / "pkg").iterdir() if p.is_dir())
        dirs += sorted(p for p in (self.root / "verification").glob("*/code*") if p.is_dir())
        for d in dirs:
            for f in d.iterdir():
                if f.is_file() and f.suffix in SRC_EXT + HDR_EXT:
                    out.add(f.relative_to(self.root).as_posix())
        return out

    def lines(self, rel):
        if rel not in self._lines:
            self._lines[rel] = (self.root / rel).read_text(encoding="latin-1").split("\n")
        return self._lines[rel]

    def nlines(self, rel):
        ln = self.lines(rel)
        return len(ln) - 1 if ln and ln[-1] == "" else len(ln)

    def units(self, rel):
        """[(line, NAME)] of the SUBROUTINE / FUNCTION / PROGRAM statements (fixed form, comments skipped)."""
        return [(n, name) for n, name, _ in self.unit_spans(rel)]

    def unit_spans(self, rel):
        """[(line, NAME, END line)] of the program units of `rel` (END line: the first bare `END` statement after the
        unit statement, else the last line)."""
        if rel not in self._units:
            out, ends = [], []
            for n, ln in enumerate(self.lines(rel), 1):
                if not ln or ln[0] in "cC*!#":
                    continue
                if re.match(r"^[ \t]+END[ \t]*(?:(?:SUBROUTINE|FUNCTION|PROGRAM)\b.*)?$", ln, re.I):
                    ends.append(n)
                    continue
                m = UNIT.match(ln)
                if m:
                    out.append((n, m.group(2).upper()))
            nl = self.nlines(rel)
            self._units[rel] = [(n, name, next((e for e in ends if e > n), nl)) for n, name in out]
        return self._units[rel]


# ---------------------------------------------------------------------------------------------------------------
# citations in the code


def _head(doc):
    """The docstring's first paragraph (up to the first blank line); a path wrapped after a `/` is joined."""
    head = re.split(r"\n[ \t]*\n", doc, maxsplit=1)[0] if doc else ""
    return re.sub(r"(?<=[A-Za-z0-9_])/[ \t]*\n[ \t]*(?=[A-Za-z0-9_])", "/", head)


def mirror_dir(py, up):
    """The upstream directory a Python file mirrors (`mitjax/pkg/seaice/x.py` -> `pkg/seaice`,
    `mitjax/verification/solid_body_cs_32x32x1/code/x.py` -> `verification/solid-body.cs-32x32x1/code`), or None."""
    parts = py.split("/")[1:-1]
    if parts[:2] == ["model", "src"] or parts[:1] == ["pkg"] and len(parts) == 2:
        return "/".join(parts)
    if parts[:1] == ["eesupp"]:
        return "eesupp/src"
    if parts[:1] == ["verification"] and len(parts) == 3:
        return up.verification_dirs.get((parts[1], parts[2]))
    return None


def resolve(up, token, py):
    """The upstream file a cited `token` names (relative path), following templates; CitationError if none."""
    cands = []
    if "/" in token:
        cands.append(token)
    else:
        md = mirror_dir(py, up)
        if md:
            cands.append(f"{md}/{token}")
        hits = up.by_name.get(token, [])
        if len(hits) == 1:
            cands.append(hits[0])
        if token.endswith(".h"):
            cands += [f"model/inc/{token}", f"eesupp/inc/{token}"]
    for c in cands:
        if c in up.files:
            return c
    for c in cands:      # a file genmake2 generates from a template (eesupp/src, pkg/exch2)
        if c.endswith(".F"):
            t = TEMPLATE_SUFFIX.sub("_rx", c.rsplit("/", 1)[-1], count=1)[:-2] + ".template"
            d = c.rsplit("/", 1)[0] + "/" if "/" in c else ""
            for tc in ([d + t] if d else []) + up.by_name.get(t, []):
                if tc in up.files:
                    return tc
    raise CitationError(f"{py}: `{token}` is not a file of MITgcm at {PIN}")


def _routine(up, rel, a, b, doc):
    """(routine names, part_of) for the span a-b of `rel` (a = None: the whole file / the docstring's own name)."""
    units = up.units(rel)
    if a is None:
        m = LEAD_NAME.match(doc)
        names = [n for _, n in units]
        if m and m.group(1).upper() in names:
            return [m.group(1).upper()], False
        return ([names[0]], False) if len(names) == 1 else ([], False)
    inside = [n for ln, n in units if a <= ln <= b]
    if inside:
        return inside, False
    before = [(ln, n, end) for ln, n, end in up.unit_spans(rel) if ln < a]
    if not before:
        return [], True
    ln, n, end = before[-1]
    # a span that starts on the continuation lines of the unit statement and runs to its END is the whole routine
    return [n], not (a <= ln + 3 and b >= end)


def port_citations(doc, py, up, errors):
    """[(upstream rel, a, b, cited token)] of the port citations in the head of one docstring; a citation the
    upstream checkout does not have is appended to `errors` instead."""
    head = _head(doc)
    if TAG not in head:
        return []
    found = {}
    for rx in (CIT_AFTER, CIT_BEFORE):
        for m in rx.finditer(head):
            for c in ONE_PATH.finditer(m.group(1)):
                found.setdefault(m.start(1) + c.start(1), (c.group(1), c.group(2), c.group(3)))
    out = []
    for _, (token, a, b) in sorted(found.items()):
        try:
            rel = resolve(up, token, py)
        except CitationError as e:
            errors.append(str(e))
            continue
        a = int(a) if a else None
        b = int(b) if b else a
        if a is not None:
            n = up.nlines(rel)
            if not (1 <= a <= b <= n):
                if (py, f"{rel}:{a}-{b}") in KNOWN_BAD:
                    KNOWN_BAD_SEEN.add((py, f"{rel}:{a}-{b}"))
                    a = b = None          # mapped as a citation of the whole file, without the wrong span
                else:
                    errors.append(f"{py}: `{token}:{a}-{b}` is outside {rel} ({n} lines at {PIN})")
                    continue
        out.append((rel, a, b, token))
    return out


def header_citations(doc, py, up, errors):
    """[(upstream rel, None, None, token)] of the header files a module's docstring starts with, its title
    (`CG2D.h: ...`, `SEAICE.h and SEAICE_GRID.h of pkg/seaice ...`, `W2_EXCH2_SIZE.h, W2_EXCH2_TOPOLOGY.h and ...`)."""
    m = HEADER_TITLE.match(doc or "")
    if not m:
        return []
    out = []
    for m in HEADER_TOKEN.finditer(m.group(1)):
        try:
            out.append((resolve(up, m.group(1), py), None, None, m.group(1)))
        except CitationError as e:
            errors.append(str(e))
    return out


def python_files():
    root = REPO / CODE_ROOT
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(REPO).as_posix()
        if not rel.startswith(CODE_SKIP) and "__pycache__" not in rel:
            yield rel


def read_sources():
    """{repository path: text} of the Python files the map is built from (sorted)."""
    return {py: (REPO / py).read_text(encoding="utf-8") for py in python_files()}


def collect(up, errors=None, sources=None):
    """[{fortran, a, b, routines, part, py, func}] for every port citation of the code, in a fixed order. With
    `errors` (a list) a bad citation is recorded there; without, CitationError lists every bad citation."""
    strict, errors = errors is None, ([] if errors is None else errors)
    KNOWN_BAD_SEEN.clear()
    entries = []
    sources = read_sources() if sources is None else sources
    for py in sorted(sources):
        tree = ast.parse(sources[py], filename=py)
        mdoc = ast.get_docstring(tree, clean=False) or ""
        for rel, a, b, _ in port_citations(mdoc, py, up, errors) + header_citations(mdoc, py, up, errors):
            names, part = _routine(up, rel, a, b, mdoc) if rel.endswith(SRC_EXT) else ([], False)
            entries.append(dict(fortran=rel, a=a, b=b, routines=names, part=part, py=py, func=None))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                continue
            doc = ast.get_docstring(node, clean=False) or ""
            for rel, a, b, _ in port_citations(doc, py, up, errors):
                names, part = _routine(up, rel, a, b, doc) if rel.endswith(SRC_EXT) else ([], False)
                entries.append(dict(fortran=rel, a=a, b=b, routines=names, part=part, py=py, func=node.name))
    errors += [f"{py}: `{tok}` is listed in KNOWN_BAD but no longer cited that way: remove the entry"
               for py, tok in sorted(set(KNOWN_BAD) - KNOWN_BAD_SEEN)]
    if strict and errors:
        raise CitationError(f"{len(errors)} port citation(s) name a file or line MITgcm at {PIN} does not have:\n  "
                            + "\n  ".join(errors))
    return entries


def mentions(sources):
    """{token: set(py)} of every Fortran file name written anywhere in the code (docstrings, comments, strings)."""
    tok = re.compile(r"(?<![A-Za-z0-9_])((?:" + _DIR + r")?" + _FILE + r")")
    out = {}
    for py in sorted(sources):
        for m in tok.finditer(sources[py]):
            out.setdefault(m.group(1), set()).add(py)
    return out


# ---------------------------------------------------------------------------------------------------------------
# docs/fortran_map.md


def section_of(rel):
    """Section key of an upstream file: `model/src`, `model/inc`, `eesupp/src`, `pkg/<p>`, `verification/<e>/<c>`."""
    p = rel.split("/")
    return "/".join(p[:3]) if p[0] == "verification" else "/".join(p[:2])


def _section_order(key):
    rank = {"model/src": 0, "model/inc": 1, "eesupp/src": 2, "eesupp/inc": 3}
    return (rank.get(key, 4 if key.startswith("pkg/") else 5), key)


def _md_link(text, target):
    return f"[{text}]({target})"


def _span(a, b):
    return "" if a is None else (f":{a}" if a == b else f":{a}-{b}")


def _routine_cell(items, single_py):
    """`ROUTINE :a-b -> func` entries, by Fortran line, one per (routines, span, Python function); a module-level
    citation of the whole file is left out where functions of the same Python file cite the file."""
    seen, out = set(), []
    with_funcs = {e["py"] for e in items if e["func"] is not None}
    with_span = {(e["py"], e["func"]) for e in items if e["a"] is not None}
    for e in sorted(items, key=lambda e: (e["a"] or 0, e["b"] or 0, e["py"], e["func"] or "")):
        if e["func"] is None and e["py"] in with_funcs or e["a"] is None and (e["py"], e["func"]) in with_span:
            continue
        hdr = e["fortran"].endswith(HDR_EXT)
        name = "header" if hdr else (", ".join(e["routines"]) or "(file)")
        if e["part"]:
            name = f"part of {name}"
        tgt = e["func"] or "(module)"
        if not single_py:
            tgt = f"{e['py'].rsplit('/', 1)[-1]}::{tgt}"
        key = (name, e["a"], e["b"], tgt)
        if key in seen:
            continue
        seen.add(key)
        out.append(f"{name} `{_span(e['a'], e['b'])}` → `{tgt}`" if e["a"] is not None else f"{name} → `{tgt}`")
    return "<br>".join(out)


def build_map(up, errors=None, sources=None):
    entries = collect(up, errors, sources)
    by_file = {}
    for e in entries:
        by_file.setdefault(e["fortran"], []).append(e)
    sections = {}
    for rel in by_file:
        sections.setdefault(section_of(rel), []).append(rel)
    return entries, by_file, sections


# packages whose "not cited" list is printed (see COVERAGE_NOTE); the rest show only their mapped files
def coverage(up, by_file, ment, words):
    """{section: (mapped, named elsewhere, not cited)} over the section's compiled sources: `named elsewhere` = the
    file's name (`ment`) or one of its routine names (`words`, the upper-case words of the code) is written somewhere
    in the code without a port citation; `not cited` = neither."""
    names = {}
    for tok, pys in ment.items():
        names.setdefault(tok.rsplit("/", 1)[-1], set()).update(pys)
    out = {}
    for key in {section_of(r) for r in by_file}:
        if key.endswith("/inc"):
            continue
        srcs = sorted(r for r in up.files if section_of(r) == key and r.endswith(SRC_EXT))
        mapped = [r for r in srcs if r in by_file]
        rest = [r for r in srcs if r not in by_file]
        mentioned, notcited = [], []
        for r in rest:
            base = r.rsplit("/", 1)[-1]
            hit = (r in ment or (len(up.by_name.get(base, [])) == 1 and base in names)
                   or any(u in words for _, u in up.units(r)))
            (mentioned if hit else notcited).append(r)
        out[key] = (mapped, mentioned, notcited)
    return out


def routine_words(sources):
    """Every identifier-like word written anywhere in the code, upper-cased (Fortran names are case-insensitive:
    `exf_GetYearlyFieldName` in a comment names EXF_GETYEARLYFIELDNAME)."""
    out = set()
    for py in sorted(sources):
        out.update(w.upper() for w in re.findall(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_]{2,}(?![A-Za-z0-9_])",
                                                 sources[py]))
    return out


CONVENTIONS = """\
## How to read a mitjax file next to its `.F`

The short version for a Fortran MITgcm developer; the long version with worked examples is
[READING_GUIDE.md](READING_GUIDE.md) (section numbers below are its sections), the rules for writing a kernel are
[KERNEL_GUIDE.md](KERNEL_GUIDE.md).

- **Files and names.** One `.py` per `.F`, same path under `mitjax/`, file and function names in lower case:
  `pkg/seaice/seaice_lsr.F` is `mitjax/pkg/seaice/seaice_lsr.py`, `SUBROUTINE SEAICE_LSR` is `seaice_lsr()`. The
  arguments are the Fortran's in the Fortran order minus `bi, bj, myThid`; outputs are returned (an output is also an
  input: the points the Fortran does not write keep their values). Common blocks are objects passed by keyword:
  `GRID.h` is `mitjax/model/grid.py`, `DYNVARS.h` and `SURFACE.h` are `mitjax/model/state.py`, `PARAMS.h` is the
  `Params` of `mitjax/model/src/ini_parms.py`, other headers `<name>_h.py` next to their routines (§1). An
  experiment's own `code*/` routine is ported under `mitjax/verification/<experiment>/<code dir>/`.
- **Citations.** The docstring of a ported routine starts with the Fortran call line and the routine's span,
  `@63cdc0b pkg/seaice/seaice_lsr.F:24-1167` (MITgcm at commit 63cdc0b; your checkout is `$MJX_UPSTREAM`). In the
  body, `# :NN` or `# :NN-MM` gives the line(s) of that same `.F` file the statement ports; `# other.F:NN` points
  into another file. To follow one, open the `.F` at that line; the map below gives every routine's span. Where a
  ported routine reaches a branch or an option that is not ported, it raises `NotImplementedError` naming the
  routine and the option when the model is traced (§5; see also the note at the end of this page).
- **Fortran-index arrays** (`mitjax/farray.py`, §3-4). Every array is an `FArray` that keeps its Fortran
  declaration: `uFld[i+1, j]` reads `uFld(i+1,j)` with Fortran index values, `j = loop_j(1-OLy, sNy+OLy-1)` is the
  `DO` statement, `KE = KE.at[i, j].set(expr)` is the assignment inside the loops. Storage is `[tile, k, j, i]`;
  the `DO bj / DO bi` loop is implicit (every statement acts on all tiles), so `bi, bj` disappear. An index outside
  the declaration is an error when the code is traced.
- **Configuration** (§5). `cfg.cpp.NAME` is `#ifdef NAME` (the experiment's preprocessed `*_OPTIONS.h`),
  `cfg.size.sNx` is `SIZE.h`, `params.viscAh` a namelist value by its Fortran name; integer, logical and character
  values are static (a Python `if`), REAL values are traced (a run-time value, so they can be differentiated).
- **Pointwise IF** (§6). An `IF` on array values is `jnp.where(cond, a, b)`: both branches are computed everywhere,
  so a division on the branch not taken is guarded (`safe_div` and friends in `mitjax/ops/safe.py`); where the
  Fortran divides, the value is bit for bit `num/den`. The guard keeps the adjoint finite (0·inf is NaN).
- **Vertical loops** (§7). Independent levels may be vectorised (`loops_kji`); a routine the Fortran calls once per
  level (`DO k ... CALL MOM_FLUXFORM(..k..)`) is written for one level and its caller runs the loop body with
  `scan_levels`; a recursion in `k` (tridiagonal solves, integrations from the surface) is `scan_k`
  (`mitjax/ops/scan_k.py`). Levels where the Fortran branches on `k` run as separate calls next to the scan.
- **MAX and MIN** (§9). Every Fortran `MAX`/`MIN` of REAL values is `MAX(a, b, p="a")` or `p="b"`
  (`mitjax/ops/fortran_minmax.py`): `p` is the argument gfortran returns at that statement on a tie (+0 vs -0) or a
  NaN, measured from our gfortran builds of the verification experiments. At a few statements the winner depends on
  the build; there the kernel looks your build up by a hash of its preprocessed options and `SIZE.h`
  (`mitjax/config/build_identity.py`). A build our Fortran reference never compiled (your own `code/` options, an
  edited `code_ad`, a new tiling), or one of its builds where that statement was never measured (a namelist option
  of your run switches it on), takes the documented default winner of that statement and prints one
  `MinMaxDefaultWarning` naming it: bitwise agreement with your own gfortran build is then not guaranteed at that
  statement (the two can differ only where the arguments tie, +0 vs -0, or one is NaN).
- **REAL\\*4 literals.** The reference build has no `-fdefault-real-8`, so a Fortran literal without `_d`/`D` is
  REAL*4: `0.1` in Fortran is written `real4("0.1")` (`mitjax/config/fortran.py`), exact literals such as `0.5` stay
  plain. `0.1 _d 0` is the double `0.1`.
- **Exchanges and sums** (§10). `EXCH_*` calls stand where the Fortran calls them (`mitjax/eesupp`); global sums keep
  the Fortran order of additions (`tile_sum_fortran`, then the tiles in tile order), never `jnp.sum`.
- **Derivatives** (§11). There is no hand-written adjoint per routine: JAX derives it from the same code. The
  exceptions are rules in `mitjax/ad/` (the CG2D solve's implicit derivative, `mitjax/ad/cg2d_rule.py`). TAF's
  adjoint-mode switches of `data.autodiff` (`useApproxAdvectionInAdMode`, `SEAICEuseFREEDRIFTswitchInAd`, ...) are
  backward-only hooks set by `mitjax/drivers/ad_switches.py`; the forward run is unchanged by them.
"""

MAP_INTRO = """\
## The map

One row per MITgcm file that the code cites as the source of a Python module, function or class (a *port
citation*: `@63cdc0b <file>[:<lines>]` in the first paragraph of a docstring). *Routines* lists the Fortran
routine(s) of each cited span, read from the `.F` file at 63cdc0b, and the Python function that ports them
(`part of X`: the span is a block inside routine X; `(module)`: the citation is in the module docstring). Every
citation on this page was checked against the MITgcm checkout when the page was generated: the file exists and the
lines are inside it. Fortran files link to MITgcm on GitHub at 63cdc0b, Python files into this repository.
"""

COVERAGE_NOTE = """\
*Coverage* (per directory, compiled sources `.F`/`.F90`/`.template` only): *mapped* files have a port citation;
*named elsewhere* files are named somewhere in the code by file or routine name (a comment, a docstring note, a call
written inline in its caller, or a "not ported" remark) without a port citation of their own, so they may be partly
ported; *not cited* files are named nowhere in `mitjax/`: nothing of them is ported. The lists are given for
`model/src` and the packages mitjax ports; how reliable they are is said at the end of the page.
"""


def render_fortran_map(up, sources=None):
    sources = read_sources() if sources is None else sources
    entries, by_file, sections = build_map(up, sources=sources)
    cov = coverage(up, by_file, mentions(sources), routine_words(sources))
    n_rows = len(by_file)
    n_py = len({e["py"] for e in entries})
    head = [
        "# Fortran map: where each MITgcm file is in mitjax",
        "",
        f"<!-- GENERATED by `{REGENERATE}` from the code's citations and MITgcm at {PIN}; do not edit by hand. -->",
        "",
        f"**Generated page, do not edit by hand.** Regenerate with `{REGENERATE}` (needs `$MJX_UPSTREAM`, an MITgcm",
        f"checkout at {PIN}); `mitjax/tests/test_docs_generated.py` fails when this page is stale.",
        "",
        f"MITgcm master at commit `{PIN}` → mitjax: {n_rows} MITgcm files mapped to {n_py} Python files.",
        "Find your `.F` file under its directory below; the table names the Python file, the routines and their",
        "Fortran line spans. The section after this one explains how to read the Python next to the Fortran.",
        "",
        "Directories: " + ", ".join(f"[`{k}`](#{_anchor(k)})" for k in sorted(sections, key=_section_order)),
        "",
    ]
    body = [CONVENTIONS, MAP_INTRO, COVERAGE_NOTE]
    for key in sorted(sections, key=_section_order):
        files = sorted(sections[key])
        body.append(f"### {key}\n")
        body.append("| Fortran | Python | Routines (Fortran lines → Python) |")
        body.append("|---|---|---|")
        for rel in files:
            items = by_file[rel]
            pys = sorted({e["py"] for e in items})
            fcell = _md_link(f"`{rel.rsplit('/', 1)[-1]}`", MITGCM_BLOB + rel)
            pcell = "<br>".join(_md_link(f"`{p}`", "../" + p) for p in pys)
            body.append(f"| {fcell} | {pcell} | {_routine_cell(items, len(pys) == 1)} |")
        body.append("")
        if key in cov and key in COVERAGE_SECTIONS(cov):
            mapped, mentioned, notcited = cov[key]
            tot = len(mapped) + len(mentioned) + len(notcited)
            body.append(f"Coverage of `{key}`: {len(mapped)} of {tot} compiled sources mapped, "
                        f"{len(mentioned)} named elsewhere, {len(notcited)} not cited.")
            body.append("")
            if mentioned:
                body.append("- named elsewhere: " + ", ".join(f"`{r.rsplit('/', 1)[-1]}`" for r in mentioned))
            if notcited:
                body.append("- not cited (not ported): " + ", ".join(f"`{r.rsplit('/', 1)[-1]}`" for r in notcited))
            body.append("")
    body.append(RELIABILITY)
    return "\n".join(head) + "\n" + "\n".join(body).rstrip("\n") + "\n"


def COVERAGE_SECTIONS(cov):
    """Directories whose coverage lists are printed (see RELIABILITY): model/src and every pkg/<p> that mitjax has
    as a package (mitjax/pkg/<p>/), except pkg/autodiff (TAF's run-time support; JAX differentiates the code)."""
    return {k for k in cov if k == "model/src"
            or k.startswith("pkg/") and k != "pkg/autodiff" and (REPO / "mitjax" / k).is_dir()}


RELIABILITY = """\
## About the coverage lists

A *not cited* file is one whose file name and routine names (in any letter case) appear nowhere in `mitjax/`.
Every ported routine carries a port citation (a project rule, `docs/KERNEL_GUIDE.md` §1), and a port that skipped
it would still name its routine somewhere, so a not-cited file has no port. The reverse does not hold: a file
*named elsewhere* may be ported inline (a small routine written into its caller), partly ported, or only mentioned
as not ported; read the code that names it. Lists are printed for `model/src` and the packages
that have a directory under `mitjax/pkg/`. No list for `eesupp/src` (threads, MPI and I/O set-up, which the JAX
drivers replace), `pkg/autodiff` (TAF's run-time support; JAX differentiates the code itself), packages mitjax does
not port at all, and the experiments' code directories: an experiment `.F` without a port stops the model with an
error naming the routine (`mitjax/config/own_code.py`).

Not ported does not always mean "refused": most unported branches raise `NotImplementedError` when an experiment
reaches them, but the check is per branch, made by the routine that calls into the unported code. A routine listed
here that your configuration needs is worth a look at its caller's port before you trust a run.
"""


def _anchor(key):
    return re.sub(r"[^a-z0-9 _-]", "", key.lower()).replace(" ", "-")


# ---------------------------------------------------------------------------------------------------------------
# docs/status.md: verification variants x kind of test (docs plan 20261006 Task 10)

MILESTONES = (("M1", "VARIANTS"), ("M2", "M2_VARIANTS"), ("M2", "M2_MIN"), ("M3", "M3_VARIANTS"), ("M4", "M4_VARIANTS"))
# the kind of a test function, read from its name (the project names its gates this way); controls never prove
CONTROL_RE = re.compile(r"negative_control|_control_|control_|planted")
KINDS = (        # first match wins: a P=N whole-run comparison is a sharded test, not a forward one
    ("sharded", re.compile(r"shard|pn_equals|p\d_equals|equals_p1|equals_single")),
    ("gradient", re.compile(r"admgrd|gradient|adjoint|grdchk|dot_test|vs_taf|tlm")),
    ("forward", re.compile(r"whole_run")),
)
GPU_GROUP = "tier2"


def registry_variants():
    """[(milestone, experiment, input dir, code dir)] from reference/reference_runs.py (the oracle runs' registry)."""
    rr = _load_by_path("_mjx_reference_runs_gen_docs", REPO / "reference" / "reference_runs.py")
    out, seen = [], set()
    for ms, name in MILESTONES:
        for e, i, c in getattr(rr, name):
            if (e, i) not in seen:
                seen.add((e, i))
                out.append((ms, e, i, c))
    return out


def test_manifest():
    """{test file: group} merged from mitjax/tests/manifest_*.py (loaded by path, as mitjax/tests/manifest.py does)."""
    return _load_by_path("_mjx_manifest_gen_docs", REPO / "mitjax" / "tests" / "manifest.py").MANIFEST


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Module:
    """The string constants a test function reaches: its own (decorators, defaults, body), the module-level
    assignments it names (followed through further assignments), the fixtures of its module it takes as arguments,
    and the module-level assignments of the mitjax.tests helper modules it names as attributes (`G.VARIANTS`).
    Function bodies of helpers are not followed (they serve many experiments)."""

    def __init__(self, rel, text, load):
        self.tree = ast.parse(text)
        self.doc = ast.get_docstring(self.tree) or ""
        self.assign, self.funcs, self.helpers = {}, {}, {}
        self.load = load
        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef):
                self.funcs[node.name] = node
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            self.assign.setdefault(n.id, []).append(node.value)
        for n in ast.walk(self.tree):
            if isinstance(n, ast.ImportFrom) and n.module == "mitjax.tests":
                for a in n.names:
                    self.helpers.setdefault(a.asname or a.name, f"mitjax/tests/{a.name}.py")

    def strings(self, node, seen, fixtures=True):
        out = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                out.add(n.value)
            elif isinstance(n, ast.Name):
                out |= self.name(n.id, seen)
            elif isinstance(n, ast.arg) and fixtures and n.arg in self.funcs and n.arg not in seen:
                seen.add(n.arg)
                out |= self.strings(self.funcs[n.arg], seen, fixtures=False)
            elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in self.helpers:
                h = self.load(self.helpers[n.value.id])
                if h is not None:
                    out |= h.name(n.attr, seen)
        return out

    def name(self, ident, seen):
        key = (id(self), ident)
        if key in seen or ident not in self.assign:
            return set()
        seen.add(key)
        return set().union(*(self.strings(v, seen, fixtures=False) for v in self.assign[ident]))


def _variants_named(strings, variants):
    """Variants a set of string constants names: `<exp>/<input>` inside one string, the experiment and the input
    directory each as a whole string, or the experiment (as a word of a string) when it has one registered variant."""
    out = set()
    single = {e for e in {e for e, _ in variants} if sum(1 for x, _ in variants if x == e) == 1}
    for e, i in variants:
        word = re.compile(r"(?<![\w.-])" + re.escape(e) + r"(?![\w-]|\.[\w])")
        if (e in strings and i in strings
                or any(re.search(r"(?<![\w.-])" + re.escape(f"{e}/{i}") + r"(?![\w.-])", s) for s in strings)
                or e in single and any(word.search(s) for s in strings)):
            out.add((e, i))
    return out


def function_variants(mod, node, pairs):
    """The variants test function `node` names (_Module.strings, _variants_named); when it names none, the inputs it
    names combined with the experiments its module docstring names, else the variants the docstring names."""
    found = _variants_named(strings := mod.strings(node, set()), pairs)
    if found:
        return found
    doc = _variants_named({mod.doc}, pairs)
    exps = {e for e, _ in pairs if re.search(r"(?<![\w.-])" + re.escape(e) + r"(?![\w-]|\.[\w])", mod.doc)}
    with_inputs = {(e, i) for e, i in pairs if e in exps and i in strings}
    return with_inputs or doc


def status_cells(variants, manifest=None, sources=None):
    """{(exp, input): {kind: [(test file, function)]}} for the kinds forward / gradient / sharded / gpu."""
    manifest = test_manifest() if manifest is None else manifest
    cache = {}

    def load(rel):
        if rel not in cache:
            text = sources.get(rel) if sources and rel in sources else (
                (REPO / rel).read_text(encoding="utf-8") if (REPO / rel).is_file() else None)
            cache[rel] = None if text is None else _Module(rel, text, load)
        return cache[rel]

    pairs = [(e, i) for _, e, i, _ in variants]
    cells = {}
    for rel, group in sorted(manifest.items()):
        mod = load(rel)
        if mod is None:
            continue
        for node in mod.tree.body:
            if not (isinstance(node, ast.FunctionDef) and node.name.startswith("test_")) or CONTROL_RE.search(node.name):
                continue
            kinds = [k for k, rx in KINDS if rx.search(node.name)][:1]
            if group == GPU_GROUP:
                kinds = ["gpu"]
            if not kinds:
                continue
            for v in function_variants(mod, node, pairs):
                for k in kinds:
                    cells.setdefault(v, {}).setdefault(k, []).append((rel, node.name))
    return cells


STATUS_INTRO = """\
One row per verification experiment and input directory that this project checks against a gfortran build of
MITgcm at 63cdc0b (the registry `reference/reference_runs.py`: milestones M1-M4). A cell names the test files that
check that kind of result for that variant; an empty cell means no test does. The tests run in the tiers given in
the last table (`mitjax/tests/manifest_*.py`): tier 1x runs before a change goes into the main branch, tier 2 (GPU) at
milestones. The page says what is checked; the tier runs say whether it passes.

- **Forward**: a whole run (a test function named `*whole_run*`): every MONITOR record, the testreport digits
  against `results/` and the pickups compared with the Fortran run.
- **Gradient**: a test function named for an adjoint check (`admgrd`, `gradient`, `adjoint`, `grdchk`, `dot_test`,
  `vs_taf`, `tlm`): the gradient against TAF's `results/output_adm*.txt` / `output_tlm*.txt`, finite differences or
  a dot test, for the whole model or for the kernels of that variant at its own state.
- **Sharded**: P devices against one device (`shard`, `pn_equals`, `p4_equals`, `equals_p1`, `equals_single`),
  forward bit for bit or gradients within the bars of [parallel.md](parallel.md).
- **GPU**: a test of the GPU group (tier 2, NVIDIA A100).

How the page is made: a test function counts for a variant when the strings it uses name the variant (the
experiment and the input directory, `<experiment>/<input>`, or an experiment with one registered variant) through
its own code, the module-level values and fixtures it uses, and the module-level values of the `mitjax.tests`
helpers it imports; failing that, through the experiments its module docstring names. Negative controls (functions
named `*control*` or `*planted*`) never count. The rule reads test code, so a cell can miss a test that reaches a
variant in a way the rule does not follow; it does not invent one.
"""


def render_status(up, sources=None):
    variants = registry_variants()
    manifest = test_manifest()
    cells = status_cells(variants, manifest, sources)
    cols = ("forward", "gradient", "sharded", "gpu")
    head = ["# Status: what is checked, per verification experiment", "",
            f"<!-- GENERATED by `{REGENERATE}` from the test manifest and reference/reference_runs.py; do not edit "
            "by hand. -->", "",
            f"**Generated page, do not edit by hand.** Regenerate with `{REGENERATE}`; "
            "`mitjax/tests/test_docs_generated.py` fails when this page is stale.", "", STATUS_INTRO]
    n_any = sum(1 for _, e, i, _ in variants if cells.get((e, i)))
    counts = {k: sum(1 for _, e, i, _ in variants if cells.get((e, i), {}).get(k)) for k in cols}
    head += [f"{len(variants)} registered variants; {n_any} with at least one test named below: "
             + ", ".join(f"{counts[k]} {k}" for k in cols) + ".", ""]
    body = ["| milestone | experiment | input | code | forward | gradient | sharded | GPU |",
            "|---|---|---|---|---|---|---|---|"]
    used = set()
    for ms, e, i, c in variants:
        row = [ms, f"`{e}`", f"`{i}`", f"`{c}`"]
        for k in cols:
            files = sorted({r for r, _ in cells.get((e, i), {}).get(k, [])})
            used.update(files)
            row.append("<br>".join(f"[{Path(r).stem}](../{r})" for r in files))
        body.append("| " + " | ".join(row) + " |")
    body += ["", "## The tests named above", "",
             "Group (`smoke`, `tier1`: every commit; `tier1x`: before a merge, CPU; `tier2`: GPU) and the test",
             "functions that put the file in a column.", "",
             "| test file | group | functions |", "|---|---|---|"]
    per_file = {}
    for v, d in cells.items():
        for k, lst in d.items():
            for r, f in lst:
                per_file.setdefault(r, {}).setdefault(k, set()).add(f)
    for r in sorted(used):
        fn = "; ".join(f"{k}: " + ", ".join(f"`{f}`" for f in sorted(per_file[r][k]))
                       for k in cols if k in per_file.get(r, {}))
        body.append(f"| [`{r}`](../{r}) | {manifest[r]} | {fn} |")
    return "\n".join(head) + "\n" + "\n".join(body) + "\n"


# ---------------------------------------------------------------------------------------------------------------
# the supported-options part of docs/configurations.md (docs plan 20261006 Task 10): read from the code's own tables
# and refusal statements, never from a list kept for the page

REFUSALS = ("NotImplementedError", "UnsupportedOption", "UnportedRoutine", "UnportedPackage")
MACRO = re.compile(r"^[A-Z][A-Z0-9]*_[A-Z0-9_]+$")
_UPSTREAM_SCAN = ("model/src", "model/inc", "eesupp/src", "eesupp/inc", "pkg")
# output-only packages the Model accepts without porting them (mitjax/drivers/model.py, INTEGRATED_PACKAGES comments:
# diagnostics and mnc "output only", layers "output only, as diagnostics"); the kernels see their switches .FALSE.,
# so a kernel's raise on such a switch is not reached by a run (forward_step.py, DO_STATEVARS_DIAGS note)
OUTPUT_ONLY = ("diagnostics", "mnc", "layers")
OUTPUT_ONLY_SWITCHES = ("useDiagnostics", "useMNC", "useLayers")


def upstream_macros_and_namelists(up):
    """({CPP macro names used in #define/#undef/#ifdef/#ifndef/defined()}, {namelist variable (lower case): [GROUP]})
    of the MITgcm checkout (model, eesupp, pkg)."""
    macros, nml = set(), {}
    cpp = re.compile(r"^#\s*(?:define|undef|ifdef|ifndef)\s+([A-Za-z_]\w*)|defined\s*\(?\s*([A-Za-z_]\w*)", re.M)
    for d in _UPSTREAM_SCAN:
        for f in sorted((up.root / d).rglob("*")):
            if f.suffix not in (".F", ".h", ".F90"):
                continue
            text = f.read_text(encoding="latin-1")
            for a, b in cpp.findall(text):
                macros.add(a or b)
            lines = text.split("\n")
            for n, ln in enumerate(lines):
                m = re.match(r"^\s+NAMELIST\s*/\s*(\w+)\s*/(.*)$", ln, re.I)
                if not m:
                    continue
                body, k = m.group(2), n + 1
                while k < len(lines) and re.match(r"^     \S", lines[k]):
                    body += " " + lines[k][6:]
                    k += 1
                for v in re.findall(r"[A-Za-z_]\w*", re.sub(r"!.*", "", body)):
                    nml.setdefault(v.lower(), set()).add(m.group(1).upper())
    return macros, nml


def _parents(tree):
    par = {}
    for node in ast.walk(tree):
        for ch in ast.iter_child_nodes(node):
            par[ch] = node
    return par


def _where(node, par):
    names = []
    while node in par:
        node = par[node]
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.append(node.name)
    return ".".join(reversed(names)) or "(module)"


def _raises(stmts):
    for s in stmts:
        if isinstance(s, ast.Raise) and isinstance(s.exc, ast.Call):
            f = s.exc.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if name in REFUSALS:
                return name
    return None


def refusal_sites(sources=None):
    """[(python file, function, condition text, exception, {identifiers})]: every `if <condition>: raise X(...)` in
    mitjax/ (tests excluded) with X a refusal exception; identifiers = attribute names and string constants of the
    condition, with a loop variable over a module-level tuple of strings expanded to the tuple's entries."""
    sources = read_sources() if sources is None else sources
    out = []
    for py in sorted(sources):
        if not py.startswith("mitjax/") or py.startswith(CODE_SKIP):
            continue
        tree = ast.parse(sources[py])
        tables = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Tuple, ast.List)) and all(
                    isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.value.elts):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        tables[t.id] = [e.value for e in node.value.elts]
        par = _parents(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            exc = _raises(node.body)
            if exc is None:
                continue
            idents = set()
            for n in ast.walk(node.test):
                if isinstance(n, ast.Attribute):
                    idents.add(n.attr)
                elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                    idents.add(n.value)
                elif isinstance(n, ast.Name):
                    p = node
                    while p in par:
                        p = par[p]
                        if isinstance(p, ast.For) and isinstance(p.target, ast.Name) and p.target.id == n.id \
                                and isinstance(p.iter, ast.Name) and p.iter.id in tables:
                            idents.update(tables[p.iter.id])
                            break
            cond = " ".join(ast.get_source_segment(sources[py], node.test).split())   # the source text: the same
            out.append((py, _where(node, par), cond, exc, idents))                   # on every Python version
    return out


def integrated_packages(sources):
    """The packages the Model runs when switched on: every set literal of INTEGRATED_PACKAGES in drivers/model.py."""
    tree = ast.parse(sources["mitjax/drivers/model.py"])
    pk = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "INTEGRATED_PACKAGES"
                                                for t in node.targets):
            for n in ast.walk(node.value):
                if isinstance(n, ast.Constant) and isinstance(n.value, str):
                    pk.add(n.value)
    return pk


def module_table(sources, py, name):
    """The literal value of a module-level assignment (tuples / dicts of constants) of `py`."""
    for node in ast.parse(sources[py]).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{py}: no module-level {name}")


def _code(text, width=110):
    text = re.sub(r"\s+", " ", text).replace("|", "\\|").replace("`", "'")
    return f"`{text if len(text) <= width else text[:width - 1] + '…'}`"


def _pyref(py, where):
    return f"[`{py[len('mitjax/'):]}`](../{py}) `{where}`"


BEGIN_OPTIONS = "<!-- BEGIN GENERATED: supported options (python tools/gen_docs.py) -->"
END_OPTIONS = "<!-- END GENERATED: supported options -->"


def render_options_block(up, sources=None):
    sources = read_sources() if sources is None else sources
    macros, nml = upstream_macros_and_namelists(up)
    sites = refusal_sites(sources)
    pk = sorted(integrated_packages(sources))
    out = [BEGIN_OPTIONS, "",
           "## Supported options (generated)", "",
           "*Generated from the code by `python tools/gen_docs.py`; do not edit between the GENERATED markers.*", "",
           "### Packages", "",
           "A package whose `use<Package>` switch is `.TRUE.` in your `data.pkg` must be one of these, or the set-up",
           "stops with `UnportedPackage` naming the switch (`require_ported` in `mitjax/config/params.py`, called by",
           "`check_packages` in `mitjax/drivers/model.py`). Compiled but switched-off packages are fine.", "",
           "| package | how |", "|---|---|"]
    for p in pk:
        how = ("accepted, output only: not ported, its output is not written" if p in OUTPUT_ONLY else "ported")
        out.append(f"| `pkg/{p}` | {how} |")
    mnc = module_table(sources, "mitjax/drivers/model.py", "MNC_READ_SWITCHES")
    out += ["", "Reading input through `pkg/mnc` is refused (`MNC_READ_SWITCHES` in `mitjax/drivers/model.py`): with",
            "`useMNC`, each of these switches set to `.TRUE.` stops the set-up with `UnsupportedOption`.", "",
            "| file | namelist group | switch | read at (MITgcm) |", "|---|---|---|---|"]
    out += [f"| `{f}` | `{g}` | `{k}` | {c} |" for f, g, k, c in mnc]
    own = module_table(sources, "mitjax/config/own_code.py", "PORTED")
    out += ["", "### Experiment code files", "",
            "Every `.F` in an experiment's `code*/` directory must be the upstream routine or one of these ported",
            "experiment routines (compared byte for byte); any other stops the set-up with `UnportedRoutine` naming it",
            "(`mitjax/config/own_code.py`, plan decision 11).", "",
            "| experiment | code dir | file | ported as |", "|---|---|---|---|"]
    for (exp, cdir, fname), target in sorted(own.items()):
        out.append(f"| `{exp}` | `{cdir}` | `{fname}` | `{target}` |")

    def rows(select):
        got = []
        for py, where, cond, exc, idents in sites:
            names = sorted(select(idents))
            if names:
                got.append((names, py, where, cond, exc))
        return got

    cpp_rows = rows(lambda ids: {i for i in ids if MACRO.match(i) and i in macros})
    nml_rows = rows(lambda ids: {i for i in ids if i.lower() in nml and i not in OUTPUT_ONLY_SWITCHES
                                 and not MACRO.match(i)})
    used = {(py, where, cond) for _, py, where, cond, _ in cpp_rows + nml_rows}
    out_only = [(py, where, cond) for py, where, cond, exc, ids in sites
                if (py, where, cond) not in used and ids & set(OUTPUT_ONLY_SWITCHES)]
    used |= set(out_only)
    other = [(py, where, cond, exc) for py, where, cond, exc, _ in sites if (py, where, cond) not in used]
    out += ["", "### CPP options with an explicit refusal", "",
            f"{len(cpp_rows)} checks in the code stop a run with an error when the condition holds; the options are the",
            "macros of your preprocessed `*_OPTIONS.h` (`cfg.cpp.NAME` is `#ifdef NAME`). A check sits in the routine",
            "that reaches the unported code, so it fires when your run gets there (at set-up or when the step is",
            "traced). An option that appears in no row is either ported or not read by any ported routine.", "",
            "| option | where (mitjax module, function) | stops when |", "|---|---|---|"]
    for names, py, where, cond, exc in sorted(cpp_rows, key=lambda r: (r[0], r[1], r[2])):
        out.append(f"| {', '.join(f'`{n}`' for n in names)} | {_pyref(py, where)} | {_code(cond)} |")
    out += ["", "### Namelist parameters with an explicit refusal", "",
            f"{len(nml_rows)} checks name a parameter that MITgcm reads from a namelist (any `NAMELIST` statement",
            "upstream); the group is given for orientation. Values that a check refuses stop the run with an error",
            "naming the routine.", "",
            "| parameter (group) | where (mitjax module, function) | stops when |", "|---|---|---|"]
    for names, py, where, cond, exc in sorted(nml_rows, key=lambda r: ([n.lower() for n in r[0]], r[1], r[2])):
        cell = ", ".join(f"`{n}` ({'/'.join(sorted(nml[n.lower()]))})" for n in names)
        out.append(f"| {cell} | {_pyref(py, where)} | {_code(cond)} |")
    out += ["", "### Other checks", "",
            f"{len(out_only)} checks on the output-only switches (`{'`, `'.join(OUTPUT_ONLY_SWITCHES)}`) are not",
            "reached by a run: the Model runs the kernels with these switches off. The remaining",
            f"{len(other)} checks guard grid, tiling, input-file and internal conditions; they are listed so that",
            "every refusal in the code is on this page.", "",
            "| where (mitjax module, function) | stops when |", "|---|---|"]
    for py, where, cond, exc in sorted(other):
        out.append(f"| {_pyref(py, where)} | {_code(cond)} |")
    out += ["", END_OPTIONS]
    return "\n".join(out) + "\n"


def render_configurations(up, sources=None):
    """docs/configurations.md: the committed hand-written text with the block between the GENERATED markers
    regenerated."""
    path = REPO / "docs" / "configurations.md"
    text = path.read_text(encoding="utf-8") if path.is_file() else f"{BEGIN_OPTIONS}\n{END_OPTIONS}\n"
    a, b = text.find(BEGIN_OPTIONS), text.find(END_OPTIONS)
    if a < 0 or b < a:
        raise CitationError("docs/configurations.md: the GENERATED markers are missing or out of order")
    return text[:a] + render_options_block(up, sources) + text[b + len(END_OPTIONS) + 1:]


BEGIN_SUMMARY = "<!-- BEGIN GENERATED: status summary (python tools/gen_docs.py) -->"
END_SUMMARY = "<!-- END GENERATED: status summary -->"


def render_status_summary(up, sources=None):
    """README.md's excerpt of docs/status.md: per experiment, how many registered variants have a test of each kind."""
    variants = registry_variants()
    cells = status_cells(variants, test_manifest(), sources)
    cols = ("forward", "gradient", "sharded", "gpu")
    exps = []
    for _, e, _, _ in variants:
        if e not in exps:
            exps.append(e)
    out = [BEGIN_SUMMARY, "",
           "| experiment | variants | forward | gradient | sharded | GPU |", "|---|---|---|---|---|---|"]
    for e in exps:
        vs = [(x, i) for _, x, i, _ in variants if x == e]
        n = {k: sum(1 for v in vs if cells.get(v, {}).get(k)) for k in cols}
        out.append(f"| `{e}` | {len(vs)} | " + " | ".join(str(n[k]) if n[k] else "" for k in cols) + " |")
    out += ["", "Numbers of input directories with at least one test of that kind; which tests, and how the table is made:",
            "[docs/status.md](docs/status.md).", "", END_SUMMARY]
    return "\n".join(out) + "\n"


def render_readme(up, sources=None):
    """README.md: the committed text with the status summary between its GENERATED markers regenerated."""
    path = REPO / "README.md"
    text = path.read_text(encoding="utf-8")
    a, b = text.find(BEGIN_SUMMARY), text.find(END_SUMMARY)
    if a < 0 or b < a:
        raise CitationError("README.md: the GENERATED markers are missing or out of order")
    return text[:a] + render_status_summary(up, sources) + text[b + len(END_SUMMARY) + 1:]


# ---------------------------------------------------------------------------------------------------------------
# pages and command line


PAGES = {
    # page name: (path in the repository, renderer(Upstream, sources) -> text)
    "fortran_map": ("docs/fortran_map.md", render_fortran_map),
    "status": ("docs/status.md", render_status),
    "configurations": ("docs/configurations.md", render_configurations),
    "readme": ("README.md", render_readme),
}

# repository paths written in backticks or links on a page must exist (the template is kept true)
REPO_PATH = re.compile(r"(?:`|\]\(\.\./)((?:mitjax|tools|scripts|docs|notebooks)/[A-Za-z0-9_./-]+?)(?:`|\))")
DOC_LINK = re.compile(r"\]\(([A-Za-z0-9_.-]+\.md)(?:#[^)]*)?\)")


def check_paths(name, text, known=()):
    """Every repository path a page names exists (mitjax/, tools/, ...; links relative to docs/); `known` are paths
    of the sources the page was built from (a test's planted file)."""
    bad = [p for p in REPO_PATH.findall(text) if "<" not in p and p not in known and not (REPO / p).exists()]
    bad += [f"docs/{p}" for p in DOC_LINK.findall(text) if not (REPO / "docs" / p).exists()]
    if bad:
        raise CitationError(f"{name}: names repository paths that do not exist: {sorted(set(bad))}")


# two literals are split into adjacent strings so that this published file does not hold them (public-tree scan)
FORBIDDEN = ("/work/", "/home/", "/scr" "atch/", "a27" "0088", "ab0995", "levante")


def check_neutral(name, text):
    """No machine path on a page: it is published."""
    low = text.lower()
    hits = [w for w in FORBIDDEN if w in low]
    if hits:
        raise CitationError(f"{name}: contains machine-specific text {hits}")


def render(name, up=None, sources=None):
    """The text of page `name`; `sources` ({path: text}, default the repository's) lets a test plant a change."""
    up = up if up is not None else Upstream()
    text = PAGES[name][1](up, sources)
    check_paths(name, text, set(sources or ()))
    check_neutral(name, text)
    return text


def stats(up):
    errors, sources = [], read_sources()
    entries, by_file, sections = build_map(up, errors, sources)
    for e in errors:
        print("BAD CITATION", e, file=sys.stderr)
    cov = coverage(up, by_file, mentions(sources), routine_words(sources))
    print(f"port citations: {len(entries)}; Fortran files mapped: {len(by_file)}; Python files: "
          f"{len({e['py'] for e in entries})}; functions/classes: {len({(e['py'], e['func']) for e in entries})}",
          file=sys.stderr)
    for k in sorted(cov, key=_section_order):
        m, c, n = cov[k]
        print(f"  {k:40s} mapped {len(m):4d}  named elsewhere {len(c):4d}  not cited {len(n):4d}", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pages", nargs="*", default=sorted(PAGES), help=f"pages ({', '.join(sorted(PAGES))})")
    ap.add_argument("--check", action="store_true", help="compare with the committed pages; write nothing")
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args(argv)
    up = Upstream()
    if a.stats:
        stats(up)
        return 0
    stale = []
    for name in a.pages:
        rel, _ = PAGES[name]
        text = render(name, up)
        path = REPO / rel
        old = path.read_text(encoding="utf-8") if path.is_file() else None
        if a.check:
            if old != text:
                stale.append(rel)
        elif old != text:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {rel}")
    if stale:
        print(f"stale: {', '.join(stale)} (regenerate with `{REGENERATE}`)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
