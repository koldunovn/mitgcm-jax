"""docs/review/M1_SIDE_BY_SIDE.html (plan Task 18, readability review) quotes the code verbatim and is self-contained.

Every code cell of the page carries `data-src="up:<path>:<line>"` (MITgcm at branch `pinned` of $MJX_UPSTREAM, checked
to be 63cdc0b) or `data-src="repo:<path>:<line>"` (this repository); its text must equal that line exactly. The page
must load nothing from outside (no loading tag such as `<img>`, `<link>`, `<script>`; no `src`/`href` other
than an in-page `#anchor`; no `@import`, `url(` or URL in its stylesheet), and it shows ten routines.

When this test fails because quoted code moved, regenerate the page: `python docs/review/make_m1_side_by_side.py`
(the generator finds each Python excerpt by its first and last line, so it follows the code) and look at the diff.

Negative controls (measured on the real page): one changed character in a quoted line, a line number shifted by one,
and a planted external resource are each reported.
"""

import re
import subprocess
from functools import lru_cache
from html.parser import HTMLParser

from mitjax import paths

PAGE = paths.REPO / "docs" / "review" / "M1_SIDE_BY_SIDE.html"
PIN_BRANCH, PIN = "pinned", "63cdc0b"
N_ROUTINES = 10
EXTERNAL_CSS = re.compile(r"https?://|@import|url\(", re.I)
LOADING_TAGS = {"link", "script", "img", "iframe", "object", "embed", "source", "video", "audio", "base"}


@lru_cache(maxsize=None)
def _upstream(path):
    r = subprocess.run(["git", "-C", str(paths.UPSTREAM), "show", f"{PIN_BRANCH}:{path}"], capture_output=True)
    return r.stdout.decode("latin-1").split("\n") if r.returncode == 0 else None


@lru_cache(maxsize=None)
def _repo(path):
    f = paths.REPO / path
    return f.read_text(encoding="utf-8").split("\n") if f.is_file() else None


class _Cells(HTMLParser):
    """(data-src, text) of every <td data-src=...>; count of <section class="routine">; anything that would load a
    resource: a loading tag, a `src` / `href` attribute other than an in-page `#anchor`, a URL in a <style>."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.cells, self.sections, self.external, self._cur, self._style = [], 0, [], None, None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "section" and a.get("class") == "routine":
            self.sections += 1
        if tag == "td" and "data-src" in a:
            self._cur = [a["data-src"], []]
        if tag in LOADING_TAGS:
            self.external.append(f"<{tag}> tag")
        for k in ("src", "href", "srcset", "data", "action", "poster"):
            v = a.get(k)
            if v is not None and not (k == "href" and v.startswith("#")):
                self.external.append(f"<{tag} {k}={v!r}>")
        if tag == "style":
            self._style = []

    def handle_endtag(self, tag):
        if tag == "td" and self._cur is not None:
            self.cells.append((self._cur[0], "".join(self._cur[1])))
            self._cur = None
        if tag == "style" and self._style is not None:
            css = "".join(self._style)
            self.external += [f"stylesheet loads {m.group(0)!r}" for m in EXTERNAL_CSS.finditer(css)]
            self._style = None

    def handle_data(self, data):
        if self._cur is not None:
            self._cur[1].append(data)
        if self._style is not None:
            self._style.append(data)


def check_page(text):
    """Every error of the page `text` (empty: every quoted line verbatim, nothing external, ten routines)."""
    errors = []
    p = _Cells()
    p.feed(text)
    p.close()
    errors += [f"external resource: {e}" for e in p.external]
    if p.sections != N_ROUTINES:
        errors.append(f"{p.sections} routine sections, expected {N_ROUTINES}")
    if not p.cells:
        errors.append("no quoted code cells")
    for src, got in p.cells:
        kind, _, rest = src.partition(":")
        path, _, n = rest.rpartition(":")
        if kind not in ("up", "repo") or not n.isdigit():
            errors.append(f"bad data-src {src!r}")
            continue
        lines = _upstream(path) if kind == "up" else _repo(path)
        if lines is None:
            errors.append(f"{src}: no such file")
        elif not 1 <= int(n) <= len(lines):
            errors.append(f"{src}: outside the file's {len(lines)} lines")
        elif lines[int(n) - 1] != got:
            errors.append(f"{src}: not verbatim: {got!r} vs {lines[int(n) - 1]!r}")
    return errors


def test_upstream_pin():
    r = subprocess.run(["git", "-C", str(paths.UPSTREAM), "rev-parse", PIN_BRANCH], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith(PIN), (r.stdout, r.stderr)


def test_page_quotes_verbatim_and_self_contained():
    text = PAGE.read_text(encoding="utf-8")
    assert check_page(text) == []
    p = _Cells()
    p.feed(text)
    kinds = {src.partition(":")[0] for src, _ in p.cells}
    assert kinds == {"up", "repo"} and len(p.cells) > 1000, (kinds, len(p.cells))


def _first_cell(text, kind):
    m = re.search(r'<td class="c" data-src="(' + kind + r':[^"]+:(\d+))">([^<]+)</td>', text)
    assert m, kind
    return m


def test_planted_changed_line_fails():
    text = PAGE.read_text(encoding="utf-8")
    for kind in ("up", "repo"):
        m = _first_cell(text, kind)
        planted = text[:m.start(3)] + m.group(3) + "x" + text[m.end(3):]
        errs = check_page(planted)
        assert any(e.startswith(m.group(1) + ": not verbatim") for e in errs), errs


def test_planted_shifted_line_number_fails():
    text = PAGE.read_text(encoding="utf-8")
    for kind in ("up", "repo"):
        m = _first_cell(text, kind)
        src, n = m.group(1), int(m.group(2))
        shifted = src[:-len(str(n))] + str(n + 1)
        errs = check_page(text[:m.start(1)] + shifted + text[m.end(1):])
        assert any(e.startswith(shifted + ": not verbatim") for e in errs), errs


def test_planted_external_resource_fails():
    text = PAGE.read_text(encoding="utf-8")
    errs = check_page(text.replace("</main>", '<img src="https://example.org/x.png"></main>', 1))
    assert any("external resource: <img> tag" in e for e in errs), errs
    errs = check_page(text.replace("</style>", "@import 'x.css';</style>", 1))
    assert any("stylesheet loads" in e for e in errs), errs
