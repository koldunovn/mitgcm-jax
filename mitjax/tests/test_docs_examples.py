"""docs/READING_GUIDE.md quotes the code it explains: every citation and every quoted block is checked (plan Task 9).

What is checked, outside fenced code blocks:
  * an upstream citation `@63cdc0b <path>:<a>[-<b>]` names a file of the MITgcm clone at branch `pinned` (= 63cdc0b,
    checked) with at least `b` lines;
  * a repository citation `<path>:<a>[-<b>]` (path under mitjax/, tools/, scripts/, reference/, docs/, dev/) names a
    file of this repository with at least `b` lines;
  * any other `<file>.<F|h|py|...>:<n>` is an error (an unqualified citation would go unchecked);
  * a backticked repository path without a line (`mitjax/ops/safe.py`) exists (paths with `<`, `*` or `$` are
    patterns and are skipped).
Every fenced block whose info string is `python` or `fortran` is a quote: the last citation on the non-blank line
before the opening fence is its source (a `python` block needs a repository `.py` citation, a `fortran` block an
upstream one), and the block equals the cited lines verbatim (after removing the indentation common to all lines and
trailing blanks). A `python`/`fortran` block without such a line is an error; `text`/`bash` blocks are not quotes.

Negative controls (measured on the real guide, not assumed): a quote whose cited range is shifted by one line, a
citation past the end of a file, a changed character inside a quote, an unqualified citation, a missing path and an
unlabelled quote each give an error.
"""

import re
import subprocess
import textwrap
from functools import lru_cache
from pathlib import Path

import pytest

from mitjax import paths

REPO = paths.REPO
GUIDE = REPO / "docs" / "READING_GUIDE.md"
PIN_BRANCH, PIN = "pinned", "63cdc0b"

_EXT = r"(?:py|F|F90|h|flow|template|md|sbatch|sh|json|txt|html|toml)"
_REPO_DIRS = r"(?:mitjax|tools|scripts|reference|docs|dev)"
UPSTREAM_CIT = re.compile(r"@" + PIN + r" ([A-Za-z0-9_./-]+\." + _EXT + r"):(\d+)(?:-(\d+))?")
REPO_CIT = re.compile(r"(?<![\w./@-])(" + _REPO_DIRS + r"/[A-Za-z0-9_./-]+\." + _EXT + r"):(\d+)(?:-(\d+))?")
ANY_CIT = re.compile(r"[A-Za-z0-9_./-]+\." + _EXT + r":\d+")
REPO_PATH = re.compile(r"`(" + _REPO_DIRS + r"/[^`\s]*)`")
FENCE = re.compile(r"^```(\w*)\s*$")


@lru_cache(maxsize=None)
def upstream_lines(path):
    """The lines of `path` at branch `pinned` of the upstream clone (None if the file does not exist there)."""
    r = subprocess.run(["git", "-C", str(paths.UPSTREAM), "show", f"{PIN_BRANCH}:{path}"], capture_output=True,
                       text=True, encoding="latin-1")
    return r.stdout.split("\n")[:-1] if r.returncode == 0 else None


@lru_cache(maxsize=None)
def repo_lines(path):
    f = REPO / path
    return f.read_text(encoding="utf-8").split("\n")[:-1] if f.is_file() else None


def _citations(line):
    """[(kind, path, a, b, start offset)] of one line: upstream citations first claim their text."""
    out, taken = [], []
    for m in UPSTREAM_CIT.finditer(line):
        out.append(("upstream", m.group(1), int(m.group(2)), int(m.group(3) or m.group(2)), m.start()))
        taken.append((m.start(), m.end()))
    for m in REPO_CIT.finditer(line):
        if not any(s <= m.start() < e for s, e in taken):
            out.append(("repo", m.group(1), int(m.group(2)), int(m.group(3) or m.group(2)), m.start()))
            taken.append((m.start(), m.end()))
    for m in ANY_CIT.finditer(line):
        if not any(s <= m.start() < e for s, e in taken):
            out.append(("unqualified", m.group(0), 0, 0, m.start()))
    return sorted(out, key=lambda c: c[4])


def _lines_of(kind, path):
    return upstream_lines(path) if kind == "upstream" else repo_lines(path)


def _check_citation(c, where):
    kind, path, a, b, _ = c
    if kind == "unqualified":
        return [f"{where}: unqualified citation {path!r} (write `@{PIN} <path>:<n>` or a repository path)"]
    lines = _lines_of(kind, path)
    if lines is None:
        return [f"{where}: {kind} file {path} does not exist"]
    if not (1 <= a <= b <= len(lines)):
        return [f"{where}: {kind} citation {path}:{a}-{b} outside the file's {len(lines)} lines"]
    return []


def _norm(block):
    return [s.rstrip() for s in textwrap.dedent("\n".join(block)).split("\n")]


def check_guide(text):
    """Every error of the guide `text` (empty list: all citations and quotes hold)."""
    errors = []
    lines = text.split("\n")
    n, prev = 0, None          # prev: (line number, text) of the last non-blank line outside code blocks
    while n < len(lines):
        m = FENCE.match(lines[n])
        if m:
            lang, start = m.group(1), n
            n += 1
            body = []
            while n < len(lines) and not FENCE.match(lines[n]):
                body.append(lines[n])
                n += 1
            if n == len(lines):
                errors.append(f"line {start + 1}: unclosed code block")
                break
            if lang in ("python", "fortran"):
                errors += _check_quote(lang, body, prev, start + 1)
            n += 1
            prev = None
            continue
        where = f"line {n + 1}"
        for c in _citations(lines[n]):
            errors += _check_citation(c, where)
        for p in REPO_PATH.findall(lines[n]):
            p = p.split(":")[0].rstrip(".,;")
            if any(ch in p for ch in "<*$") or not p:
                continue
            if not (REPO / p).exists():
                errors.append(f"{where}: path {p} does not exist in the repository")
        if lines[n].strip():
            prev = (n + 1, lines[n])
        n += 1
    return errors


def _check_quote(lang, body, prev, fence_line):
    where = f"quote at line {fence_line}"
    if prev is None:
        return [f"{where}: no label line before the {lang} block"]
    cits = [c for c in _citations(prev[1]) if c[0] != "unqualified"]
    if not cits:
        return [f"{where}: the line before the {lang} block (line {prev[0]}) cites no source"]
    kind, path, a, b, _ = cits[-1]
    want = "repo" if lang == "python" else "upstream"
    if kind != want or (lang == "python" and not path.endswith(".py")):
        return [f"{where}: a {lang} quote needs a {want} citation, got {kind} {path}"]
    lines = _lines_of(kind, path)
    if lines is None or not (1 <= a <= b <= len(lines)):
        return [f"{where}: source {path}:{a}-{b} does not exist"]
    got, src = _norm(body), _norm(lines[a - 1:b])
    if got != src:
        bad = next((k for k, (x, y) in enumerate(zip(got, src)) if x != y), min(len(got), len(src)))
        return [f"{where}: not verbatim {path}:{a}-{b} (first difference at quoted line {bad + 1}: "
                f"{got[bad] if bad < len(got) else '<end>'!r} vs {src[bad] if bad < len(src) else '<end>'!r})"]
    return []


# ---------------------------------------------------------------------------------------------------------------
# the guide as written

def test_upstream_pin():
    r = subprocess.run(["git", "-C", str(paths.UPSTREAM), "rev-parse", PIN_BRANCH], capture_output=True, text=True)
    assert r.returncode == 0, f"no branch {PIN_BRANCH} in {paths.UPSTREAM}: {r.stderr}"
    assert r.stdout.startswith(PIN), r.stdout


def test_reading_guide_citations_and_quotes():
    text = GUIDE.read_text(encoding="utf-8")
    # a quote whose code moved: `python tools/fix_guide_citations.py --write` re-points its citation
    assert check_guide(text) == []
    quotes = sum(1 for ln in text.split("\n") if ln.strip() in ("```python", "```fortran"))
    assert quotes >= 12, quotes            # 6 examples, each with Fortran and Python


# ---------------------------------------------------------------------------------------------------------------
# negative controls: each planted error is found

def _guide():
    return GUIDE.read_text(encoding="utf-8")


def _first_python_label(text):
    lines = text.split("\n")
    for n, ln in enumerate(lines):
        if ln.strip() == "```python":
            k = n - 1
            while not lines[k].strip():
                k -= 1
            return lines[k]
    raise AssertionError("no python quote in the guide")


def test_planted_shifted_line_numbers_fail():
    text = _guide()
    label = _first_python_label(text)
    kind, path, a, b, _ = [c for c in _citations(label) if c[0] == "repo"][-1]
    planted = text.replace(label, label.replace(f"{path}:{a}-{b}", f"{path}:{a + 1}-{b + 1}"), 1)
    assert planted != text
    errs = check_guide(planted)
    assert any("not verbatim" in e and f"{path}:{a + 1}-{b + 1}" in e for e in errs), errs


def test_planted_citation_past_end_of_file_fails():
    text, n = re.subn(r"`mitjax/model/src/do_oceanic_phys\.py:\d+`", "`mitjax/model/src/do_oceanic_phys.py:99999`",
                      _guide(), count=1)                       # the guide's single-line citation, wherever it now points
    assert n == 1
    errs = check_guide(text)
    assert any("do_oceanic_phys.py:99999-99999 outside" in e for e in errs), errs
    text = _guide().replace("`@63cdc0b model/src/do_oceanic_phys.F:803`", "`@63cdc0b model/src/do_oceanic_phys.F:99999`",
                            1)
    errs = check_guide(text)
    assert any("do_oceanic_phys.F:99999-99999 outside" in e for e in errs), errs


def test_planted_changed_quote_fails():
    text = _guide()
    a = "    rhsMax = MAX_CHAIN(0.0, jnp.abs(cg2d_b[i, j]), p=\"a\", acc=\"b\", ex=ex)"
    assert text.count(a) == 1
    errs = check_guide(text.replace(a, a.replace('p="a"', 'p="b"')))
    assert any("not verbatim mitjax/model/src/cg2d.py:" in e for e in errs), errs   # wherever the quote now points


def test_planted_unqualified_missing_and_unlabelled_fail():
    base = _guide()
    errs = check_guide(base + "\nSee cg2d.F:163 for the stencil.\n")
    assert any("unqualified citation 'cg2d.F:163'" in e for e in errs), errs
    errs = check_guide(base + "\nSee `mitjax/ops/no_such_module.py`.\n")
    assert any("path mitjax/ops/no_such_module.py does not exist" in e for e in errs), errs
    errs = check_guide(base + "\n\n```python\nx = 1\n```\n")
    assert any("cites no source" in e or "no label line" in e for e in errs), errs
