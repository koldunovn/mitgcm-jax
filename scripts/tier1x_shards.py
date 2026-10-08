#!/usr/bin/env python3
"""Split the tier1x test files into N shards of about equal wall time, for the parallel pytest processes of
scripts/run_tier1x.sbatch.

    tier1x_shards.py --shards N --index K [--reports-dir DIR] [--group tier1x]

Prints the test files of shard K (0-based), one per line. Files come from the test manifest
(mitjax/tests/manifest.py, group `tier1x`); a whole file is the unit, so each file runs in exactly one shard and
its tests keep their order and their module-level state. A file's weight is the sum of its testcase times in the
newest run directory under DIR (default $MJX_RUNS/tier1x) whose JUnit reports cover it, so a run cut off by its
time limit still measures the files it finished (tier 1x 27951742: four new files, each weighted as the heaviest,
pushed the real heavy files into two shards that timed out; the earlier rule, one report set, the one covering the
most files, would have ignored that run and weighted the four as unmeasured again). A file without a measured time
gets the largest measured time (or 60 s): a new file is scheduled first and alone, so an expensive new test cannot
stretch a shard past the job's time limit before its first complete run measures it (tier 1x 27911209: a 95-min
new file weighted as the median timed out its shard). Assignment is greedy longest-first, ties broken by path, so the
split is a pure function of (manifest, measured times, N): every shard of one job sees the same split.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def tier_files(group):
    from mitjax.tests.manifest import MANIFEST
    return sorted(f for f, g in MANIFEST.items() if g == group)


def file_of(classname):
    """JUnit classname -> repository path of its test file: `mitjax.tests.test_x` or `mitjax.tests.test_x.Class`."""
    parts = classname.split(".")
    for n in range(len(parts), 0, -1):
        p = "/".join(parts[:n]) + ".py"
        if (ROOT / p).is_file():
            return p
    return None


def run_order(run):
    """Oldest first: Slurm job ids compare as numbers, any other run directory after them by name."""
    return (0, int(run.name), "") if run.name.isdigit() else (1, 0, run.name)


def measured_times(reports_dir):
    """{file: seconds}, each file from the newest run directory (report*.xml) that measured it."""
    best = {}
    d = Path(reports_dir)
    if not d.is_dir():
        return best
    for run in sorted((r for r in d.iterdir() if r.is_dir()), key=run_order):
        times = {}
        for r in sorted(run.glob("report*.xml")):
            try:
                root = ET.parse(r).getroot()
            except (OSError, ET.ParseError):
                continue
            for case in root.iter("testcase"):
                f = file_of(case.get("classname", ""))
                if f:
                    times[f] = times.get(f, 0.0) + float(case.get("time", 0) or 0)
        best.update(times)                               # a newer run overrides the files it measured
    return best


def split(files, times, n):
    known = [times[f] for f in files if f in times]
    default = max(known) if known else 60.0          # unmeasured = the heaviest (see the module docstring)
    weight = {f: times.get(f, default) for f in files}
    shards = [[] for _ in range(n)]
    load = [0.0] * n
    for f in sorted(files, key=lambda f: (-weight[f], f)):
        k = min(range(n), key=lambda i: (load[i], i))
        shards[k].append(f)
        load[k] += weight[f]
    return shards, load


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--group", default="tier1x")
    ap.add_argument("--reports-dir", default=None)
    ap.add_argument("--show", action="store_true", help="print every shard with its estimated seconds instead")
    a = ap.parse_args(argv)
    if not 0 <= a.index < a.shards:
        ap.error("--index must be in [0, --shards)")
    rd = a.reports_dir or (Path(os.environ["MJX_RUNS"]) / "tier1x" if "MJX_RUNS" in os.environ else None)
    times = measured_times(rd) if rd else {}
    shards, load = split(tier_files(a.group), times, a.shards)
    if a.show:
        for k, (s, w) in enumerate(zip(shards, load)):
            print(f"# shard {k}: {len(s)} files, ~{w:.0f} s")
            for f in s:
                print(f"  {f}")
        return 0
    print("\n".join(shards[a.index]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
