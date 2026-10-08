#!/usr/bin/env python3
"""Invisibility of the jaxdump shim (plan Task 4, lesson [E§2]): the instrumented binary, with dumps off and with
dumps on, must write the same state files and the same STDOUT as the plain binary.

    invisibility.py PLAIN_TOP JDOFF_TOP JDON_TOP [--out FILE]

Each *_TOP is a run directory made by reference/make_rundir.py and run by reference/run.sh (RUN_TOP/rundir/...).
Compared:
  * every regular file the run wrote into the run directory (recursively; symlinked inputs and the run's own
    output.txt/run.tr_log excluded): the same set of names, byte-identical contents (exemption NC_BUILD_STAMP);
  * STDOUT (rundir/output.txt) line by line: byte-identical except for lines covered by a NAMED exemption below,
    each with its cause; any other differing line, or a different number of lines, fails;
  * stderr (rundir/run.tr_log): byte-identical.
Exemptions (lines may differ; every exempted line is counted in the report):
  BUILD_STAMP   `// Build user|host|date:` -- two binaries built at different times (printed by
                eesupp/src/eeintro_msg.F:57-70 from BUILD_INFO.h, which genmake2 writes at build time).
  TIMER_VALUE   `User time:`, `System time:`, `Wall clock time:` -- measured run times (TIMER_PRINTALL).
  NC_BUILD_STAMP  netCDF (classic) files of pkg/mnc: the character values of the global attributes build_host
                and build_date (pkg/mnc/mnc_cw_model_attr.F:60-68, from BUILD_INFO.h) are blanked in both files
                before the byte comparison; every other byte must be equal.
  COMM_STATS    with dumps ON only: the `Tile <-> Tile communication statistics` block (exchange, spin and
                barrier counters, printed by eesupp/src/comm_stats.F) -- the exchange probe of stage
                X00 calls the exchange routines, which count. Not exempt with dumps off.
  DIAGSTATS_UNSET_REGIONS  (lane A session 6, global_ocean.cs32x15/input) netCDF statistics-diagnostics files of
                pkg/diagnostics (a dimension named `region`): DIAGSTATS_MNC_OUT writes statGlob(.,.,j) for every
                region j = 0..nRegions (pkg/diagnostics/diagstats_mnc_out.F:255-257, 298-301), but DIAGSTATS_OUTPUT
                fills only the selected regions, diagSt_region(j) > 0 (diagstats_output.F:77-84; with no
                `stat_region` in data.diagnostics only region 0, diagnostics_readparms.F:483): the other regions
                hold uninitialised values of the local array statGlob, which differ between binaries (and runs).
                When the run's data.diagnostics sets no `stat_region`, the values of region indices >= 1 of every
                variable with dimensions (record, region, level) are blanked in both files (plus NC_BUILD_STAMP)
                before the byte comparison; region 0 and every other byte must be equal.
Verdict line: `INVISIBLE <label> ...` or `VISIBLE <label>: <first problem>`; exit 0 / 1.
Stdlib only (runs in the job without the mitjax environment).
"""

import argparse
import re
import sys
from pathlib import Path

EXEMPT = {
    "BUILD_STAMP": re.compile(r"^\(PID\.TID \d+\.\d+\) // Build (user|host|date): "),
    "TIMER_VALUE": re.compile(r"^\(PID\.TID \d+\.\d+\)\s+(User time|System time|Wall clock time):"),
}
COMM_BEGIN = re.compile(r"^\(PID\.TID \d+\.\d+\) // Tile <-> Tile communication statistics")
SKIP = {"output.txt", "run.tr_log"}


def written_files(rundir):
    """{relative path: Path} of the regular (non-symlink) files under rundir, except STDOUT/stderr."""
    out = {}
    for p in sorted(rundir.rglob("*")):
        if p.is_symlink() or not p.is_file():
            continue
        rel = p.relative_to(rundir).as_posix()
        if rel in SKIP:
            continue
        out[rel] = p
    return out


NC_STAMP_ATTRS = (b"build_host", b"build_date")


def blank_nc_stamps(data):
    """Copy of a netCDF classic file's bytes with the values of the NC_STAMP_ATTRS character attributes zeroed
    (header layout: name length, name padded to 4 bytes, nc_type, nelems, values padded to 4 bytes)."""
    b = bytearray(data)
    if not b.startswith(b"CDF"):
        return None
    for name in NC_STAMP_ATTRS:
        i = b.find(name)
        while i >= 0:
            if int.from_bytes(b[i - 4:i], "big") == len(name):
                j = i + (len(name) + 3) // 4 * 4
                typ, n = int.from_bytes(b[j:j + 4], "big"), int.from_bytes(b[j + 4:j + 8], "big")
                if typ == 2:  # NC_CHAR
                    b[j + 8:j + 8 + (n + 3) // 4 * 4] = bytes((n + 3) // 4 * 4)
            i = b.find(name, i + 1)
    return bytes(b)


def _nc_header(b):
    """Minimal parser of a netCDF classic (CDF-1/CDF-2) header: (dims [(name, len)], vars [(name, dimids, nc_type,
    vsize, begin)], recsize) or None. Layout: the netCDF classic format specification (magic, numrecs, dim_list,
    gatt_list, var_list; names and values padded to 4 bytes)."""
    if b[:3] != b"CDF" or b[3] not in (1, 2):
        return None
    off = 8 if b[3] == 2 else 4
    pos = 8

    def i4():
        nonlocal pos
        v = int.from_bytes(b[pos:pos + 4], "big")
        pos += 4
        return v

    def name():
        nonlocal pos
        n = i4()
        s = bytes(b[pos:pos + n]).decode("latin-1")
        pos += (n + 3) // 4 * 4
        return s

    size = {1: 1, 2: 1, 3: 2, 4: 4, 5: 4, 6: 8}

    def atts():
        nonlocal pos
        tag, n = i4(), i4()
        for _ in range(n if tag else 0):
            name()
            typ, nel = i4(), i4()
            pos += (nel * size[typ] + 3) // 4 * 4

    tag, n = i4(), i4()
    dims = [(name(), i4()) for _ in range(n if tag else 0)]
    atts()
    tag, n = i4(), i4()
    var = []
    for _ in range(n if tag else 0):
        vname = name()
        nd = i4()
        ids = [i4() for _ in range(nd)]
        atts()
        typ, vsize = i4(), i4()
        begin = int.from_bytes(b[pos:pos + off], "big")
        pos += off
        var.append((vname, ids, typ, vsize, begin))
    rec = [v for v in var if v[1] and dims[v[1][0]][1] == 0]
    recsize = sum(v[3] for v in rec)
    return dims, var, recsize


def blank_unset_regions(data):
    """DIAGSTATS_UNSET_REGIONS (module docstring): a copy with the values of region indices >= 1 of every record
    variable (record, region, level) of type double zeroed; None if the file has no `region` dimension."""
    b = bytearray(data)
    h = _nc_header(b)
    if h is None:
        return None
    dims, var, recsize = h
    rid = next((k for k, (n, _) in enumerate(dims) if n == "region"), None)
    if rid is None:
        return None
    nrec = int.from_bytes(b[4:8], "big")
    for vname, ids, typ, vsize, begin in var:
        if len(ids) != 3 or ids[1] != rid or dims[ids[0]][1] != 0 or typ != 6:
            continue
        nreg, nlev = dims[rid][1], dims[ids[2]][1]
        for r in range(nrec):
            start = begin + r * recsize + nlev * 8          # region 1, level 1 of record r
            b[start:start + (nreg - 1) * nlev * 8] = bytes((nreg - 1) * nlev * 8)
    return bytes(b)


def stat_region_set(rundir):
    """True if the run's data.diagnostics assigns stat_region (then no region is exempted)."""
    f = Path(rundir) / "data.diagnostics"
    if not f.exists():
        return False
    text = "\n".join(ln.split("#", 1)[0] for ln in f.read_text(errors="replace").split("\n"))
    return re.search(r"(?i)\bstat_region\s*\(", text) is not None


def comm_block(lines):
    """Index range [i0, i1) of the communication statistics block (header up to the normal-end line), or None."""
    for i, ln in enumerate(lines):
        if COMM_BEGIN.match(ln):
            i0 = i - 1 if i > 0 and lines[i - 1].endswith("// ======================================================") \
                else i
            i1 = next((k for k in range(i, len(lines)) if lines[k].startswith("PROGRAM MAIN:")), len(lines))
            return i0, i1
    return None


def compare_stdout(a, b, allow_comm):
    """-> (problems, exempted counts). a, b: lists of lines."""
    problems, counts = [], {k: 0 for k in (*EXEMPT, "COMM_STATS")}
    if allow_comm:
        ca, cb = comm_block(a), comm_block(b)
        if (ca is None) != (cb is None):
            problems.append("communication statistics block in only one STDOUT")
        elif ca is not None:  # (absent in both: a run that stopped before printing it, e.g. optim's grdchk STOP)
            counts["COMM_STATS"] = sum(1 for x, y in zip(a[ca[0]:ca[1]], b[cb[0]:cb[1]]) if x != y) + \
                abs((ca[1] - ca[0]) - (cb[1] - cb[0]))
            a = a[:ca[0]] + a[ca[1]:]
            b = b[:cb[0]] + b[cb[1]:]
    if len(a) != len(b):
        problems.append(f"STDOUT has {len(a)} vs {len(b)} lines")
    for n, (x, y) in enumerate(zip(a, b)):
        if x == y:
            continue
        name = next((k for k, rx in EXEMPT.items() if rx.match(x) and rx.match(y)), None)
        if name is None:
            problems.append(f"STDOUT line {n + 1} differs: {x!r} vs {y!r}")
        else:
            counts[name] += 1
    return problems, counts


def check_pair(plain_top, test_top, allow_comm):
    pr, tr = Path(plain_top) / "rundir", Path(test_top) / "rundir"
    problems = []
    fa, fb = written_files(pr), written_files(tr)
    if set(fa) != set(fb):
        problems.append(f"written files differ: only plain {sorted(set(fa) - set(fb))[:5]}, "
                        f"only test {sorted(set(fb) - set(fa))[:5]}")
    same, nc_exempt, st_exempt = 0, 0, 0
    regions_ok = not stat_region_set(pr) and not stat_region_set(tr)
    for rel in sorted(set(fa) & set(fb)):
        x, y = fa[rel].read_bytes(), fb[rel].read_bytes()
        if x == y:
            same += 1
        elif rel.endswith(".nc") and blank_nc_stamps(x) is not None and blank_nc_stamps(x) == blank_nc_stamps(y):
            nc_exempt += 1
        elif (rel.endswith(".nc") and regions_ok and blank_nc_stamps(x) is not None
              and blank_unset_regions(blank_nc_stamps(x)) is not None
              and blank_unset_regions(blank_nc_stamps(x)) == blank_unset_regions(blank_nc_stamps(y))):
            st_exempt += 1
        else:
            problems.append(f"file {rel} differs")
    a = (pr / "output.txt").read_text(errors="replace").split("\n")
    b = (tr / "output.txt").read_text(errors="replace").split("\n")
    p2, counts = compare_stdout(a, b, allow_comm)
    problems += p2
    ea, eb = pr / "run.tr_log", tr / "run.tr_log"
    if (ea.read_bytes() if ea.exists() else b"") != (eb.read_bytes() if eb.exists() else b""):
        problems.append("stderr (run.tr_log) differs")
    counts["NC_BUILD_STAMP"] = nc_exempt
    counts["DIAGSTATS_UNSET_REGIONS"] = st_exempt
    return problems, same, counts, len(a)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("plain")
    ap.add_argument("jdoff")
    ap.add_argument("jdon")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    label = "/".join(Path(a.plain).resolve().parts[-3:-1])
    lines, ok = [], True
    for name, top, allow in (("dumps-off", a.jdoff, False), ("dumps-on", a.jdon, True)):
        problems, same, counts, nl = check_pair(a.plain, top, allow)
        ex = ", ".join(f"{k} {v}" for k, v in counts.items())
        if problems:
            ok = False
            lines.append(f"VISIBLE {label} {name}: {problems[0]} ({len(problems)} problems)")
            lines += [f"  {p}" for p in problems[:20]]
        else:
            lines.append(f"INVISIBLE {label} {name}: {same} written files byte-identical (+{counts['NC_BUILD_STAMP']} "
                         f"netCDF equal except build stamp), STDOUT {nl} lines "
                         f"identical except exempted ({ex}), stderr identical")
    dumps = Path(a.jdon) / "dumps"
    nfiles = len(list(dumps.glob("jd_*_t*.bin"))) if dumps.is_dir() else 0
    lines.append(f"  dumps-on wrote {nfiles} dump files in {dumps}")
    if nfiles == 0:
        ok = False
        lines.append(f"VISIBLE {label} dumps-on: no dump files written (the dumps-on run did not dump)")
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if a.out:
        if Path(a.out).exists():
            raise SystemExit(f"{a.out} exists (nothing is overwritten)")
        Path(a.out).write_text(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
