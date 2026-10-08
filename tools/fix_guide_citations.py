#!/usr/bin/env python3
"""Re-point the repository citations of docs/READING_GUIDE.md whose quoted code moved.

    python tools/fix_guide_citations.py [--write]

For every fenced python block, the nearest repository citation `path:a-b` above it (outside fences) names the lines
the block quotes (mitjax/tests/test_docs_examples.py checks them verbatim). When the file no longer has the block at
a-b, the block is searched for in the file: exactly one verbatim occurrence re-points the citation to it; none or
several are reported and left alone (the quote itself must then be updated by hand). Without --write only reports.
The quotes themselves are never edited.
"""

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GUIDE = REPO / "docs" / "READING_GUIDE.md"
CIT = re.compile(r"(?<![\w./@-])((?:mitjax|tools|scripts|reference)/[A-Za-z0-9_./-]+\.py):(\d+)(?:-(\d+))?")
FENCE = re.compile(r"^```(\w*)\s*$")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    lines = GUIDE.read_text(encoding="utf-8").split("\n")
    last, in_fence, label, block, start, changes, problems = None, False, "", [], 0, [], []
    for n, ln in enumerate(lines):
        m = FENCE.match(ln)
        if m and not in_fence:
            in_fence, label, block, start = True, m.group(1), [], n
            continue
        if m and in_fence:
            in_fence = False
            if label == "python" and last is not None:
                cl, path, x, y, span = last
                src = (REPO / path).read_text(encoding="utf-8").split("\n")
                if src[x - 1:y] != block:
                    hits = [i for i in range(len(src) - len(block) + 1) if src[i:i + len(block)] == block]
                    if len(hits) == 1:
                        nx, ny = hits[0] + 1, hits[0] + len(block)
                        new = f"{path}:{nx}" + (f"-{ny}" if ny != nx else "")
                        changes.append((cl, span, new, f"{path}:{x}-{y} -> {new}"))
                    else:
                        problems.append(f"guide line {start + 1}: block quoted from {path}:{x}-{y} found "
                                        f"{len(hits)} times")
            continue
        if in_fence:
            block.append(ln)
            continue
        for c in CIT.finditer(ln):
            x = int(c.group(2))
            y = int(c.group(3)) if c.group(3) else x
            last = (n, c.group(1), x, y, c.span())
    for cl, (s0, s1), new, msg in sorted(changes, key=lambda t: (t[0], -t[1][0])):
        print("re-point:", msg)
        lines[cl] = lines[cl][:s0] + new + lines[cl][s1:]
    for p in problems:
        print("PROBLEM:", p)
    if a.write and changes:
        GUIDE.write_text("\n".join(lines), encoding="utf-8")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
