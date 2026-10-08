#!/usr/bin/env python3
"""Fortran restart pair: run A (N steps, permanent pickup at nIter0_B) vs run B (restart from A's pickup) (lane A
session 5).

    restart_compare.py A_TOP B_TOP --steps 5:6:7:8:9 [--out FILE] [--json FILE]

A_TOP, B_TOP: run directories made by reference/make_rundir.py and run by reference/run.sh, both with dumps on
(<top>/dumps, the same jaxdump binary and JAXDUMP_STEPS). Compared, and reported (nothing is judged here; the tier-1x
test scripts/tests/test_restart_oracle.py asserts the measured pattern):
  * dumps: every key (iteration, stage, field) of the given iterations present in both dump sets, bit patterns at
    every point incl. halos: number of differing points, how many of them in the interior, the points
    (tile, k, j, i in the record's 0-based array index, halos included) and their bit patterns; keys of only one run;
  * STDOUT (rundir/output.txt): the MONITOR blocks (`%MON` lines between `// Begin MONITOR dynamic field statistics`
    and `// End MONITOR`) of B's iterations against A's blocks of the same `time_tsnumber`, line by line without the
    `(PID.TID ...)` prefix; the solver lines (` cg2d: Sum(rhs),rhsMax`, `cg2d_init_res`, `cg2d_iters(min,last)`,
    `cg2d_last_res`, printed once per step by SOLVE_FOR_PRESSURE) of B's steps against A's steps of the same end
    iteration; `%CHECKPOINT` lines;
  * files: every regular file B wrote into its run directory (output.txt, run.tr_log and the copied pickup excluded)
    against A's file of the same name: byte-identical, different, or absent in A.
Dump reader: mitjax/io/dump.py loaded by path (numpy only; no jax).
"""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_mjx_dump", REPO / "mitjax" / "io" / "dump.py")
dump = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dump)

PID = re.compile(r"^\(PID\.TID \d+\.\d+\) ?")
SOLVER = (re.compile(r"^ cg2d: Sum\(rhs\),rhsMax"), re.compile(r"cg2d_init_res ="),
          re.compile(r"cg2d_iters\(min,last\) ="), re.compile(r"cg2d_last_res ="))
MAX_POINTS = 4000   # points listed per key in the JSON (a 66x66x1 R1 field has 4356)


def bits(a):
    return np.ascontiguousarray(a, dtype=np.float64).view(np.uint64)


def diff_record(ra, rb):
    """(points, n_interior): the differing points of two records of one tile (bit patterns), as
    [(k, j, i, a_bits_hex, b_bits_hex)]."""
    a, b = bits(ra.data), bits(rb.data)
    if a.shape != b.shape:
        raise ValueError(f"{ra.stage}/{ra.field}: shapes {a.shape} != {b.shape}")
    m = a != b
    inner = np.zeros_like(m)
    inner[:, ra.oly:ra.oly + ra.sny, ra.olx:ra.olx + ra.snx] = True
    pts = [(int(k), int(j), int(i), f"{int(a[k, j, i]):016x}", f"{int(b[k, j, i]):016x}")
           for k, j, i in zip(*np.nonzero(m))]
    return pts, int(np.count_nonzero(m & inner))


def compare_dumps(da, db, iters):
    """[{iter, stage, field, kind, n_points, n_diff, n_interior, points}] for every key of `iters` in both sets
    (call order of A), and the keys found in one set only."""
    rows, only = [], {"A": [], "B": []}
    ka = [k for k in da.keys() if k[0] in iters]
    kb = [k for k in db.keys() if k[0] in iters]
    only["A"] = [list(k) for k in ka if k not in db.index]
    only["B"] = [list(k) for k in kb if k not in da.index]
    for key in ka:
        if key not in db.index:
            continue
        if da.n_occ(key) != db.n_occ(key):
            only["A"].append(list(key) + [f"occurrences {da.n_occ(key)} vs {db.n_occ(key)}"])
            continue
        for occ in range(da.n_occ(key)):
            ta, tb = da.tiles(*key, occ=occ), db.tiles(*key, occ=occ)
            if sorted(ta) != sorted(tb):
                raise ValueError(f"{key}: tiles {sorted(ta)} vs {sorted(tb)}")
            pts, n_in, n_all = [], 0, 0
            for t in sorted(ta):
                p, ni = diff_record(ta[t], tb[t])
                pts += [(t, *x) for x in p]
                n_in += ni
                n_all += ta[t].data.size
            kind = next(iter(ta.values())).kind
            rows.append({"iter": key[0], "stage": key[1], "field": key[2], "occ": occ, "kind": kind,
                         "n_points": n_all, "n_diff": len(pts), "n_interior": n_in, "points": pts[:MAX_POINTS]})
    return rows, only


def stdout_events(path):
    """(blocks {tsnumber: [lines]}, solver [[4 lines] per step in order], checkpoint lines) of a STDOUT."""
    blocks, solver, ckpt, cur, inblock = {}, [], [], None, False
    group = []
    for raw in Path(path).read_text(errors="replace").splitlines():
        line = PID.sub("", raw)
        if "// Begin MONITOR dynamic field statistics" in line:
            inblock, cur = True, []
            continue
        if inblock and "// End MONITOR" in line:
            ts = next(int(x.split("=")[1]) for x in cur if x.startswith("%MON time_tsnumber"))
            blocks[ts] = cur
            inblock = False
            continue
        if inblock and line.startswith("%MON"):
            cur.append(line)
            continue
        if line.startswith("%CHECKPOINT"):
            ckpt.append(line)
        for n, pat in enumerate(SOLVER):
            if pat.search(raw):
                if n == 0:
                    group = [line]
                else:
                    group.append(line)
                if n == 3:
                    solver.append(group)
                    group = []
    return blocks, solver, ckpt


def compare_stdout(a_out, b_out, niter0_a, niter0_b):
    ba, sa, ca = stdout_events(a_out)
    bb, sb, cb = stdout_events(b_out)
    blocks = {}
    for ts in sorted(bb):
        if ts not in ba:
            blocks[ts] = {"missing_in_A": True}
            continue
        la, lb = ba[ts], bb[ts]
        diffs = [[x, y] for x, y in zip(la, lb) if x != y]
        blocks[ts] = {"n_lines": [len(la), len(lb)], "diff": diffs}
    solver = {}
    for n, g in enumerate(sb):
        it = niter0_b + n + 1                        # the step that ends at iteration it
        k = it - niter0_a - 1
        solver[it] = {"equal": 0 <= k < len(sa) and sa[k] == g, "A": sa[k] if 0 <= k < len(sa) else None, "B": g}
    return {"blocks": blocks, "solver": solver, "checkpoint": {"A": ca, "B": cb}}


def compare_files(a_rundir, b_rundir, exclude):
    out = {}
    for p in sorted(Path(b_rundir).iterdir()):
        if p.is_symlink() or not p.is_file() or p.name in exclude:
            continue
        q = Path(a_rundir) / p.name
        if not q.is_file() or q.is_symlink():
            out[p.name] = "absent in A"
        else:
            out[p.name] = "identical" if q.read_bytes() == p.read_bytes() else "DIFFERENT"
    return out


def niter0(rundir):
    m = re.search(r"^\s*nIter0\s*=\s*(\d+)", Path(rundir, "data").read_text(), re.M | re.I)
    return int(m.group(1)) if m else 0


def compare(a_top, b_top, iters):
    a_top, b_top = Path(a_top), Path(b_top)
    da, db = dump.DumpSet(a_top / "dumps"), dump.DumpSet(b_top / "dumps")
    rows, only = compare_dumps(da, db, set(iters))
    man_b = json.loads((b_top / "MANIFEST.json").read_text())
    copied = {c["name"] for c in man_b.get("copies") or []}
    na, nb = niter0(a_top / "rundir"), niter0(b_top / "rundir")
    return {"A": str(a_top), "B": str(b_top), "iters": list(iters), "nIter0": [na, nb], "dumps": rows,
            "only": only,
            "stdout": compare_stdout(a_top / "rundir" / "output.txt", b_top / "rundir" / "output.txt", na, nb),
            "files": compare_files(a_top / "rundir", b_top / "rundir", {"output.txt", "run.tr_log"} | copied)}


def report(res):
    out = [f"restart pair  A {res['A']}", f"              B {res['B']}",
           f"nIter0 A {res['nIter0'][0]}  B {res['nIter0'][1]}; dumped iterations compared: {res['iters']}", ""]
    nd = [r for r in res["dumps"] if r["n_diff"]]
    out.append(f"dumps: {len(res['dumps'])} records compared, {len(nd)} differ "
               f"(bit patterns, all points incl. halos)")
    for r in nd:
        ex = ", ".join(f"t{p[0]}(k{p[1]},j{p[2]},i{p[3]}) {p[4]}/{p[5]}" for p in r["points"][:3])
        out.append(f"  DIFF {r['iter']:>6} {r['stage']:<26} {r['field']:<10} {r['n_diff']:>5} of {r['n_points']} "
                   f"points ({r['n_interior']} interior)  e.g. {ex}")
    for side in ("A", "B"):
        if res["only"][side]:
            out.append(f"  keys only in {side} (or with other occurrences): {len(res['only'][side])}: "
                       + "; ".join("/".join(map(str, k)) for k in res["only"][side][:12]))
    out.append("")
    so = res["stdout"]
    for ts, b in so["blocks"].items():
        if b.get("missing_in_A"):
            out.append(f"stdout MONITOR block {ts}: not in A")
        else:
            out.append(f"stdout MONITOR block {ts}: {b['n_lines'][1]} lines, {len(b['diff'])} differ"
                       + ("" if b["n_lines"][0] == b["n_lines"][1] else f" (A has {b['n_lines'][0]} lines)"))
            for x, y in b["diff"]:
                out.append(f"    A: {x}\n    B: {y}")
    for it, s in so["solver"].items():
        out.append(f"stdout solver lines of the step ending at {it}: {'equal' if s['equal'] else 'DIFFERENT'}")
    out.append(f"stdout %CHECKPOINT A: {so['checkpoint']['A']}")
    out.append(f"stdout %CHECKPOINT B: {so['checkpoint']['B']}")
    out.append("")
    for f, v in res["files"].items():
        out.append(f"file {f}: {v}")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a_top")
    ap.add_argument("b_top")
    ap.add_argument("--steps", required=True, help="dumped iterations to compare, e.g. 5:6:7:8:9")
    ap.add_argument("--out")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    iters = [int(x) for x in re.split(r"[:,]", a.steps)]
    res = compare(a.a_top, a.b_top, iters)
    text = report(res)
    if a.out:
        Path(a.out).write_text(text)
    if a.json:
        Path(a.json).write_text(json.dumps(res) + "\n")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
