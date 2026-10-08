#!/usr/bin/env python3
"""Compare a test run of the Fortran oracle with its base run (plan Task 3b: invariance checks, debugMode runs).

    compare_runs.py BASE_TOP TEST_TOP [--kind tiling|output|debug] [--json FILE]

BASE_TOP and TEST_TOP are run directories made by reference/make_rundir.py and run by reference/run.sh (the STDOUT is
<TOP>/rundir/output.txt). What is compared:

  * Numeric records of the STDOUT, in order, as text: every `%MON <name> = <value>` and `%SBO <name> = <value>` line
    (pkg/monitor, pkg/sbo), every `cg2d_init_res`, `cg2d_iters(min,last)` and `cg2d: Sum(rhs),rhsMax` line
    (solve_for_pressure.F / cg2d.F), every `fc =` line of the cost (cost_final.F) and every `grdchk ... fc` line. Per
    record name: number of values, how many differ, the largest relative difference |a-b|/max(|a|,|b|), the largest
    difference relative to the field's scale (`scale_rel`: for `%MON <f>_mean|_sd|_del2` the larger of |<f>_max| and
    |<f>_min| of the same record in the base run, the tolerance-class definition of docs/PORTING_RULES.md §5; else
    max(|a|,|b|)), and the digits testreport's comparison program gives for that series (tools/testreport_jax.tr_cmpnum:
    16 = every value equal, 22 = all zero; 99 = series of different length or under 2 values). A name in only one run
    is listed as such.
  * With --exp/--variant: the testreport check list of the variant (tools/testreport_jax: tr_checklist or the default
    list) with the base run's STDOUT as the reference, i.e. the digits testreport would report test-vs-base.
  * The whole STDOUT line by line after removing, in both files, the lines of the named exemptions of --kind:
      all kinds   BUILD_STAMP (`// Build user|host|date:`), TIMER (`User time:` ... `Wall clock time:` and the
                  timer table lines of TIMER_PRINTALL: lines inside the `Seconds in section` blocks)
      debug       DEBUG_MSG (lines printed by pkg/debug/debug_msg.F: `DEBUG_MSG: `)
    The remaining difference is reported as a unified-diff hunk count and the first differing lines; it is not
    exempted (a parameter echo of an overlaid namelist, a package's own setup print-out, a tile layout print-out are
    all listed, never hidden).
  * Written files (regular files in the run directory, symlinked inputs and output.txt/run.tr_log excluded): names
    only in one run; byte comparison of the files both runs wrote, where a netCDF file equal after blanking the
    build_host/build_date attributes (reference/jaxdump/invisibility.py NC_BUILD_STAMP: two binaries built at
    different times) is counted as `nc_stamp_only`, not as different.
Verdict line (stdout): `NUMERIC-IDENTICAL <label>` when every numeric record is equal as text, else
`NUMERIC-DIFFERENT <label>: <names>`. Exit 0 either way (this is a measurement, not a gate); 2 on a usage error or
a missing STDOUT. --json writes the full result (never overwrites an existing file). Stdlib only.
"""

import argparse
import difflib
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_mjx_testreport_jax", REPO / "tools" / "testreport_jax.py")
trj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trj)

PID = r"^\(PID\.TID \d+\.\d+\)"
RECORD_PATTERNS = (
    ("tag", re.compile(PID + r" %(MON|SBO) (\S+)\s*=\s*(\S+)\s*$")),
    ("cg2d_init_res", re.compile(PID + r"\s+cg2d_init_res\s*=\s*(\S+)\s*$")),
    ("cg2d_iters", re.compile(PID + r"\s+cg2d_iters\(min,last\)\s*=\s*(.+?)\s*$")),
    ("cg2d_sum_rhs", re.compile(r"^ cg2d: Sum\(rhs\),rhsMax\s*=\s*(.+?)\s*$")),
    ("fc", re.compile(PID + r"\s+(early|local|global) fc\s*=\s*(\S+)\s*$")),
    ("grdchk_fc", re.compile(PID + r" grdchk (reference fc: fcref|perturb\(\+\)fc: fcpertplus|"
                                   r"perturb\(-\)fc: fcpertminus)\s*=\s*(\S+)\s*$")),
)
EXEMPT = {
    "BUILD_STAMP": re.compile(PID + r" // Build (user|host|date): "),
    "TIMER": re.compile(PID + r"\s+(User time|System time|Wall clock time|No\. starts|No\. stops):"),
    "TIMER_SECTION": re.compile(PID + r"\s+Seconds in section "),
    "DEBUG_MSG": re.compile(PID + r" DEBUG_MSG: "),
}
KIND_EXEMPT = {"tiling": ("BUILD_STAMP", "TIMER", "TIMER_SECTION"),
               "output": ("BUILD_STAMP", "TIMER", "TIMER_SECTION"),
               "debug": ("BUILD_STAMP", "TIMER", "TIMER_SECTION", "DEBUG_MSG")}
SKIP = {"output.txt", "run.tr_log"}


def read_stdout(top):
    p = Path(top) / "rundir" / "output.txt"
    if not p.is_file():
        raise FileNotFoundError(f"no STDOUT {p}")
    return p.read_text(errors="replace").split("\n")


def records(lines):
    """{record name: [value text, ...]} in file order (a line with several values is one text value)."""
    out = {}
    for ln in lines:
        for kind, rx in RECORD_PATTERNS:
            m = rx.match(ln)
            if not m:
                continue
            if kind == "tag":
                name, val = f"%{m[1]} {m[2]}", m[3]
            elif kind == "fc":
                name, val = f"{m[1]} fc", m[2]
            elif kind == "grdchk_fc":
                name, val = f"grdchk {m[1]}", m[2]
            else:
                name, val = kind, m[1]
            out.setdefault(name, []).append(val)
            break
    return out


def _float(t):
    try:
        return float(t.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


def compare_series(a, b, scale=None):
    """dict(n, n_diff, max_rel, scale_rel, digits) for two lists of value texts; `scale`: per-record field scale."""
    res = {"n": [len(a), len(b)], "n_diff": sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))}
    rel = srel = 0.0
    numeric = True
    for r, (x, y) in enumerate(zip(a, b)):
        xs, ys = x.split(), y.split()
        if len(xs) != len(ys):
            numeric = False
            continue
        for u, v in zip(xs, ys):
            fu, fv = _float(u), _float(v)
            if fu is None or fv is None:
                numeric = numeric and u == v
                continue
            m = max(abs(fu), abs(fv))
            if m > 0 and fu != fv:
                rel = max(rel, abs(fu - fv) / m)
                sc = scale[r] if scale is not None and r < len(scale) and scale[r] else m
                srel = max(srel, abs(fu - fv) / sc)
    res["max_rel"] = rel if numeric else None
    res["scale_rel"] = srel if numeric else None
    if len(a) == len(b) and len(a) >= 2 and all(len(x.split()) == 1 for x in a + b):
        rows = "".join(f"{i + 1} {x} {y}\n" for i, (x, y) in enumerate(zip(a, b)))
        res["digits"] = trj.tr_cmpnum(rows + "-1\n")
    else:
        res["digits"] = 99
    return res


def stdout_diff(a, b, exempt):
    """Unified diff of the two STDOUTs after removing the exempted lines; counts of removed lines per exemption."""
    counts = {k: [0, 0] for k in exempt}

    def keep(lines, side):
        out = []
        for ln in lines:
            hit = next((k for k in exempt if EXEMPT[k].match(ln)), None)
            if hit:
                counts[hit][side] += 1
            else:
                out.append(ln)
        return out

    ka, kb = keep(a, 0), keep(b, 1)
    diff = list(difflib.unified_diff(ka, kb, "base", "test", n=0, lineterm=""))
    hunks = sum(1 for d in diff if d.startswith("@@"))
    changed = [d for d in diff if d[:1] in "+-" and not d.startswith(("+++", "---"))]
    return {"exempted": counts, "hunks": hunks, "changed_lines": len(changed), "first": changed[:40]}


def written_files(top):
    rundir = Path(top) / "rundir"
    out = {}
    for p in sorted(rundir.resolve().rglob("*")):
        if p.is_symlink() or not p.is_file():
            continue
        rel = p.relative_to(rundir.resolve()).as_posix()
        if rel not in SKIP:
            out[rel] = p
    return out


_inv_spec = importlib.util.spec_from_file_location("_mjx_invisibility", REPO / "reference" / "jaxdump" / "invisibility.py")
inv = importlib.util.module_from_spec(_inv_spec)
_inv_spec.loader.exec_module(inv)


def compare_files(base, test):
    fa, fb = written_files(base), written_files(test)
    common = sorted(set(fa) & set(fb))
    differ, stamp = [], []
    for r in common:
        x, y = fa[r].read_bytes(), fb[r].read_bytes()
        if x == y:
            continue
        if r.endswith(".nc") and inv.blank_nc_stamps(x) is not None and inv.blank_nc_stamps(x) == inv.blank_nc_stamps(y):
            stamp.append(r)
        else:
            differ.append(r)
    return {"only_base": sorted(set(fa) - set(fb)), "only_test": sorted(set(fb) - set(fa)),
            "common": len(common), "differ": differ, "nc_stamp_only": stamp}


def field_scale(ra, name):
    """Per-record scale of `%MON <f>_mean|_sd|_del2`: max(|<f>_max|, |<f>_min|) in the base run, else None."""
    m = re.fullmatch(r"(%MON \S+)_(mean|sd|del2)", name)
    if not m or f"{m[1]}_max" not in ra or f"{m[1]}_min" not in ra:
        return None
    mx, mn = ra[f"{m[1]}_max"], ra[f"{m[1]}_min"]
    if len(mx) != len(ra[name]) or len(mn) != len(ra[name]):
        return None
    return [max(abs(_float(x) or 0.0), abs(_float(y) or 0.0)) for x, y in zip(mx, mn)]


def checklist_digits(a, b, exp, variant, kind=None):
    """[(name, digits)] of testoutput_run on test STDOUT `b` with base STDOUT `a` as the reference."""
    upstream = trj._upstream()
    kind = trj.kind_of_variant(variant) if kind is None else kind
    files = trj.resolve_checklists(upstream / "verification" / exp, variant, kind)
    run = trj.testoutput_run(b, a, kind, {k: p.read_text() for k, p in files.items()})
    return run.s_var, [(v.name, v.digits) for v in run.variables]


def compare(base, test, kind, exp=None, variant=None, tr_kind=None):
    a, b = read_stdout(base), read_stdout(test)
    ra, rb = records(a), records(b)
    rec = {}
    for name in list(ra) + [n for n in rb if n not in ra]:
        if name not in ra or name not in rb:
            rec[name] = {"only_in": "base" if name in ra else "test", "n": len(ra.get(name, rb.get(name)))}
        else:
            rec[name] = compare_series(ra[name], rb[name], field_scale(ra, name))
    different = [n for n, r in rec.items() if "only_in" in r or r["n_diff"]]
    res = {"base": str(base), "test": str(test), "kind": kind, "records": rec, "different": different,
           "stdout": stdout_diff(a, b, KIND_EXEMPT[kind]), "files": compare_files(base, test)}
    if exp:
        res["checklist_first"], res["checklist"] = checklist_digits(
            trj.read_lines(Path(base) / "rundir" / "output.txt"), trj.read_lines(Path(test) / "rundir" / "output.txt"),
            exp, variant, tr_kind)
    return res


def label_of(top):
    return "/".join(Path(top).resolve().parts[-3:])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base")
    ap.add_argument("test")
    ap.add_argument("--kind", choices=sorted(KIND_EXEMPT), default="tiling")
    ap.add_argument("--exp", help="experiment (with --variant: check-list digits test vs base)")
    ap.add_argument("--variant")
    ap.add_argument("--tr-kind", choices=sorted(trj.KIND_NAMES), help="testreport kind (default: from the variant)")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    try:
        res = compare(a.base, a.test, a.kind, a.exp, a.variant or "input",
                      trj.KIND_NAMES[a.tr_kind] if a.tr_kind else None)
    except FileNotFoundError as e:
        print(f"FAIL: {e}")
        return 2
    lab = label_of(a.test)
    if res["different"]:
        print(f"NUMERIC-DIFFERENT {lab}: {len(res['different'])} of {len(res['records'])} record names: "
              + " ".join(res["different"][:30]))
        for n in res["different"]:
            r = res["records"][n]
            print(f"  {n:40s} " + (f"only in {r['only_in']}" if "only_in" in r else
                                    f"n={r['n']} differ={r['n_diff']} max_rel={r['max_rel']:.3g} "
                                    f"scale_rel={r['scale_rel']:.3g} digits={r['digits']}"
                                    if r["max_rel"] is not None else f"n={r['n']} differ={r['n_diff']} (text)"))
    else:
        print(f"NUMERIC-IDENTICAL {lab}: {len(res['records'])} record names, "
              f"{sum(r['n'][0] for r in res['records'].values())} values")
    if "checklist" in res:
        print("  check list (test vs base): " + " ".join(f"{n}{'*' if n == res['checklist_first'] else ''}="
                                                        f"{'--' if d == 99 else d}" for n, d in res["checklist"]))
    s = res["stdout"]
    print(f"  STDOUT: {s['changed_lines']} changed lines in {s['hunks']} hunks after exemptions {s['exempted']}")
    for d in s["first"][:12]:
        print(f"    {d[:150]}")
    f = res["files"]
    print(f"  files: {f['common']} common, {len(f['differ'])} differ ({len(f['nc_stamp_only'])} netCDF equal but "
          f"for the build stamp), only base {len(f['only_base'])}, "
          f"only test {len(f['only_test'])}")
    if a.json:
        if Path(a.json).exists():
            raise SystemExit(f"{a.json} exists (nothing is overwritten)")
        Path(a.json).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
