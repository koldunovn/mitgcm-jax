#!/usr/bin/env python3
"""The notebooks of notebooks/ from their plain-text sources notebooks/src/<name>.py (docs plan 20261006 Task 8).

    python tools/build_notebooks.py [NAME ...] write notebooks/<name>.ipynb, WITHOUT outputs, for every source (or
                                               only the NAMEs: "05", "05_lab_sea", "05_lab_sea.ipynb" or the source
                                               path) whose cells differ from the notebook's; a notebook whose cells
                                               equal its source is left as it is, committed outputs included
    python tools/build_notebooks.py --check    write nothing; exit 1 when a committed notebook's cells differ from
                                               its source (outputs are not compared)

A source is a Python file in the "percent" cell format that Jupytext, VS Code and Spyder read: a line `# %%` starts a
code cell, `# %% [markdown]` a Markdown cell whose lines are written as comments (`# text`, a bare `#` for an empty
line). Text before the first marker is ignored. The committed .ipynb files hold the outputs of an execution on a
compute node (mitjax/tests/test_notebooks_laptop.py executes them; its cell asserts are the checks); this tool makes
the cells, the test checks that the committed cells still equal the sources.

Stdlib only (the notebook JSON is written directly, nbformat 4.5), so it runs in any Python 3.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_DIR = REPO / "notebooks"
SRC_DIR = NB_DIR / "src"
METADATA = {
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}


def parse(text):
    """[(cell_type, source)] of a percent-format text."""
    cells, kind, lines = [], None, []

    def flush():
        if kind is not None:
            while lines and not lines[-1].strip():
                lines.pop()
            while lines and not lines[0].strip():
                lines.pop(0)
            cells.append((kind, "\n".join(lines)))

    for ln in text.splitlines():
        if ln.startswith("# %%"):
            flush()
            kind, lines = ("markdown" if "[markdown]" in ln else "code"), []
            continue
        if kind == "markdown":
            if ln in ("#", ""):                  # an empty markdown line; blank lines at a cell's ends are dropped
                ln = ""
            elif ln.startswith("# "):
                ln = ln[2:]
            else:
                raise ValueError(f"markdown line not written as a comment: {ln!r}")
        if kind is not None:
            lines.append(ln)
    flush()
    return cells


def cell_id(name, n):
    return f"{name[:2]}-{n:02d}"


def notebook(name, text):
    """The notebook dict (no outputs) of source `text`."""
    cells = []
    for n, (kind, src) in enumerate(parse(text), start=1):
        c = {"cell_type": kind, "id": cell_id(name, n), "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        cells.append(c)
    return {"cells": cells, "metadata": METADATA, "nbformat": 4, "nbformat_minor": 5}


def sources():
    return sorted(SRC_DIR.glob("[0-9][0-9]_*.py"))


def cells_of(nb):
    """[(cell_type, source)] of a notebook dict (outputs ignored)."""
    return [(c["cell_type"], "".join(c["source"]) if isinstance(c["source"], list) else c["source"])
            for c in nb["cells"]]


def check():
    """Messages for every notebook whose cells differ from its source ([] = all fresh)."""
    errs = []
    for src in sources():
        ipynb = NB_DIR / (src.stem + ".ipynb")
        if not ipynb.is_file():
            errs.append(f"{ipynb.name}: missing (python tools/build_notebooks.py)")
            continue
        want = parse(src.read_text())
        got = cells_of(json.loads(ipynb.read_text()))
        if got != want:
            errs.append(f"{ipynb.name}: cells differ from notebooks/src/{src.name}")
    return errs


def select(names):
    """The sources named by `names` (all for none): a two-digit prefix, a stem, a notebook or source file name or
    path. An unknown or ambiguous name is an error."""
    srcs = sources()
    if not names:
        return srcs
    out = []
    for n in names:
        stem = Path(n).name
        for suffix in (".ipynb", ".py"):
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
        hits = [s for s in srcs if s.stem == stem or (len(stem) == 2 and s.stem[:2] == stem)]
        if len(hits) != 1:
            raise SystemExit(f"build_notebooks: {n!r} names {len(hits)} sources of {[s.stem for s in srcs]}")
        if hits[0] not in out:
            out.append(hits[0])
    return out


def build(src):
    """Write notebooks/<stem>.ipynb from `src` unless its cells already equal the source (its outputs are then kept).
    Returns "wrote" or "unchanged"."""
    out = NB_DIR / (src.stem + ".ipynb")
    text = src.read_text()
    if out.is_file() and cells_of(json.loads(out.read_text())) == parse(text):
        return "unchanged"
    out.write_text(json.dumps(notebook(src.stem, text), indent=1, ensure_ascii=False) + "\n")
    return "wrote"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help="notebooks to build (default: all)")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        errs = check()
        print("\n".join(errs) or "notebooks fresh")
        return 1 if errs else 0
    for src in select(a.names):
        what = build(src)
        print(f"{what} notebooks/{src.stem}.ipynb" + (" (cells equal the source; outputs kept)"
                                                      if what == "unchanged" else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
