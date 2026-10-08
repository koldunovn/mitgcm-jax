#!/usr/bin/env python3
"""The FTZ oracle vs the standard oracle (plan decision 13): every difference, and whether it is confined to the
subnormal band.

    ftz_band.py STD_TOP FTZ_TOP [--out FILE] [--json FILE]

STD_TOP / FTZ_TOP: two run directories of the same variant (reference/make_rundir.py + reference/run.sh), the
standard binary's and its FTZ/DAZ variant's (reference/relink_ftz.sh), both plain or both dumps-on. Compared:
  * every regular file the run wrote into its run directory (as reference/jaxdump/invisibility.py): MITgcm binary
    files with a .meta (dataprec float64/float32, big-endian) value by value, bit for bit (so -0. vs 0. counts); the
    records of a pickup are named from fldList (3-D fields Nr levels, Nr from the run's 3-D output metas); text
    files line by line, token by token;
  * STDOUT (rundir/output.txt) line by line, token by token, except invisibility.py's BUILD_STAMP and TIMER_VALUE
    lines (both runs dumps on or both off: the communication statistics are compared);
  * the jaxdump records (RUN_TOP/dumps, mitjax/io/dump.py): the same record sequence and headers, values bit for bit
    (halos included), per (iteration, stage, field).
The band: |value| < 2**-1021 in BOTH runs at every differing point (numbers in text: both tokens parse as floats
below 2**-1021). That is the subnormal range (below 2**-1022) and the lowest normal binade, where an FTZ computation
stops: a value whose next update would be subnormal keeps its last normal value, or becomes 0. A differing point
outside it, a differing non-numeric token, a different record structure or a different file set is OUTSIDE the band.
Verdict line: `BAND CONFINED <label>: ...` (exit 0) or `OUTSIDE BAND <label>: <first problem>` (exit 1).
Runs in the mitjax environment (numpy; mitjax/io/dump.py loaded by path).
"""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BAND = 2.0 ** -1021


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


inv = _load("_mjx_invisibility", HERE / "jaxdump" / "invisibility.py")
dump = _load("_mjx_dump", HERE.parent / "mitjax" / "io" / "dump.py")

_FLOAT = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([EeDd][-+]?\d+)?$")


def _num(t):
    if not _FLOAT.match(t):
        return None
    return float(t.replace("D", "E").replace("d", "e"))


def compare_lines(a, b, label):
    """(n differing lines, band problems, examples) of two text files' lines, token by token."""
    if len(a) != len(b):
        return 0, [f"{label}: {len(a)} vs {len(b)} lines"], []
    ndiff, probs, ex = 0, [], []
    for i, (x, y) in enumerate(zip(a, b)):
        if x == y or any(p.match(x) and p.match(y) for p in inv.EXEMPT.values()):
            continue
        ndiff += 1
        tx, ty = x.split(), y.split()
        bad = len(tx) != len(ty)
        vals = []
        for s, t in zip(tx, ty) if not bad else ():
            if s == t:
                continue
            u, v = _num(s), _num(t)
            if u is None or v is None or not (abs(u) < BAND and abs(v) < BAND):
                bad = True
            vals.append((s, t))
        if len(ex) < 20:
            ex.append({"line": i + 1, "std": x, "ftz": y, "tokens": vals})
        if bad:
            probs.append(f"{label} line {i + 1}: outside the band: {x!r} vs {y!r}")
    return ndiff, probs, ex


def _meta(path):
    t = path.read_text()
    prec = re.search(r"dataprec = \[ '(float\d+)' \]", t).group(1)
    nrec = int(re.search(r"nrecords = \[\s*(\d+) \]", t).group(1))
    m = re.search(r"fldList = \{(.*?)\}", t, re.S)
    flds = re.findall(r"'([^']*)'", m.group(1)) if m else []
    return prec, nrec, [f.strip() for f in flds]


def record_names(nrec, flds, nr):
    """Field name per record of a multi-field file: the leading fields with L levels, the rest 2-D (L = Nr when that
    fits, else the smallest number of leading fields that fits)."""
    n = len(flds)
    if n == 0 or n == nrec:
        return [flds[r] if n else f"rec{r}" for r in range(nrec)]
    cands = []
    for k in range(1, n + 1):
        if (nrec - n) % k == 0:
            cands.append((k, (nrec - n) // k + 1))
    pick = next((c for c in cands if c[1] == nr), cands[0] if cands else None)
    if pick is None:
        return [f"rec{r}" for r in range(nrec)]
    k, lev = pick
    out = []
    for j, f in enumerate(flds):
        out += [f"{f}(k={q + 1})" for q in range(lev)] if j < k else [f]
    return out if len(out) == nrec else [f"rec{r}" for r in range(nrec)]


def _bits_diff(a, b):
    """Indices where two float arrays differ bit for bit."""
    return np.nonzero(a.view(np.int64 if a.dtype.itemsize == 8 else np.int32)
                      != b.view(np.int64 if b.dtype.itemsize == 8 else np.int32))[0]


def _summary(a, b, idx):
    va, vb = a[idx], b[idx]
    m = max(float(np.max(np.abs(va))), float(np.max(np.abs(vb))))
    tiny = np.finfo(np.float64).tiny
    return {"n": int(idx.size), "max_abs_std": float(np.max(np.abs(va))), "max_abs_ftz": float(np.max(np.abs(vb))),
            "std_subnormal": int(np.sum((va != 0) & (np.abs(va) < tiny))),
            "ftz_subnormal": int(np.sum((vb != 0) & (np.abs(vb) < tiny))),
            "ftz_zero": int(np.sum(vb == 0)), "std_zero": int(np.sum(va == 0)),
            "examples": [[float(x), float(y)] for x, y in zip(va[:4], vb[:4])], "in_band": m < BAND}


def compare_binary(pa, pb, rel, nr):
    """{record name: summary} of the differing values of two MITgcm .data files with the same .meta."""
    prec, nrec, flds = _meta(pa.with_suffix(".meta"))
    dt = ">f8" if prec == "float64" else ">f4"
    a = np.fromfile(pa, dt).astype(dt[1:]).reshape(nrec, -1)
    b = np.fromfile(pb, dt).astype(dt[1:]).reshape(nrec, -1)
    names = record_names(nrec, flds, nr)
    out = {}
    for r in range(nrec):
        idx = _bits_diff(a[r], b[r])
        if idx.size:
            out[names[r]] = _summary(a[r].astype(np.float64), b[r].astype(np.float64), idx)
    return out


def compare_dumps(da, db):
    """({iter/stage/field[#occ]: summary}, problems) of two dump directories."""
    fa = sorted(p.name for p in da.glob("jd_*.bin"))
    fb = sorted(p.name for p in db.glob("jd_*.bin"))
    if fa != fb:
        return {}, [f"dump file sets differ: {sorted(set(fa) ^ set(fb))[:5]}"]
    acc, probs = {}, []
    for name in fa:
        ra, rb = dump.read_file(da / name), dump.read_file(db / name)
        if len(ra) != len(rb):
            probs.append(f"{name}: {len(ra)} vs {len(rb)} records")
            continue
        for x, y in zip(ra, rb):
            hx = (x.iter, x.seq, x.stage, x.field, x.kind, x.shape, x.tile)
            if hx != (y.iter, y.seq, y.stage, y.field, y.kind, y.shape, y.tile):
                probs.append(f"{name}: record headers differ {hx} vs {(y.iter, y.seq, y.stage, y.field)}")
                break
            a, b = x.data.ravel(), y.data.ravel()
            idx = _bits_diff(a, b)
            if idx.size:
                key = f"{x.iter}/{x.stage}/{x.field}" + (f"#{x.occ}" if x.occ else "")
                s = _summary(a, b, idx)
                inner = _bits_diff(x.interior.ravel().copy(), y.interior.ravel().copy())
                s["n_interior"] = int(inner.size)
                s["tiles"] = [x.tile]
                if key in acc:
                    t = acc[key]
                    for f in ("n", "n_interior", "std_subnormal", "ftz_subnormal", "ftz_zero", "std_zero"):
                        t[f] += s[f]
                    t["max_abs_std"] = max(t["max_abs_std"], s["max_abs_std"])
                    t["max_abs_ftz"] = max(t["max_abs_ftz"], s["max_abs_ftz"])
                    t["in_band"] = t["in_band"] and s["in_band"]
                    t["tiles"].append(x.tile)
                else:
                    acc[key] = s
    return acc, probs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("std_top", type=Path)
    ap.add_argument("ftz_top", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    sa, sb = (a.std_top / "rundir").resolve(), (a.ftz_top / "rundir").resolve()
    label = f"{a.std_top.name} vs {a.ftz_top.name}"
    res = {"std": str(a.std_top), "ftz": str(a.ftz_top), "band": BAND, "files": {}, "text": {}, "dumps": {}}
    probs = []
    wa, wb = inv.written_files(sa), inv.written_files(sb)
    if set(wa) != set(wb):
        probs.append(f"written file sets differ: {sorted(set(wa) ^ set(wb))[:5]}")
    nr = None
    for m in sorted(sa.glob("*.meta")):
        d = re.search(r"dimList = \[(.*?)\]", m.read_text(), re.S).group(1).replace(",", " ").split()
        if len(d) == 9:
            nr = int(d[6])
            break
    nsame = 0
    for rel in sorted(set(wa) & set(wb)):
        pa, pb = wa[rel], wb[rel]
        if pa.read_bytes() == pb.read_bytes():
            nsame += 1
            continue
        if rel.endswith(".data") and pa.with_suffix(".meta").is_file():
            if pa.with_suffix(".meta").read_bytes() != pb.with_suffix(".meta").read_bytes():
                probs.append(f"{rel}: .meta files differ")
                continue
            d = compare_binary(pa, pb, rel, nr)
            res["files"][rel] = d
            probs += [f"{rel} {k}: outside the band (max |std| {v['max_abs_std']:.3e}, |ftz| {v['max_abs_ftz']:.3e})"
                      for k, v in d.items() if not v["in_band"]]
        else:
            try:
                la, lb = pa.read_text().splitlines(), pb.read_text().splitlines()
            except UnicodeDecodeError:
                probs.append(f"{rel}: binary file without .meta differs")
                continue
            n, p, ex = compare_lines(la, lb, rel)
            res["text"][rel] = {"lines": n, "examples": ex}
            probs += p
    jdon = (a.std_top / "dumps").is_dir()
    n, p, ex = compare_lines((sa / "output.txt").read_text().splitlines(),
                             (sb / "output.txt").read_text().splitlines(), "STDOUT")
    res["stdout"] = {"lines": n, "examples": ex}
    probs += p
    if jdon:
        if not (a.ftz_top / "dumps").is_dir():
            probs.append("no dumps in the FTZ run")
        else:
            d, p = compare_dumps(a.std_top / "dumps", a.ftz_top / "dumps")
            res["dumps"] = d
            probs += p
            probs += [f"dump {k}: outside the band (max |std| {v['max_abs_std']:.3e}, |ftz| {v['max_abs_ftz']:.3e})"
                      for k, v in d.items() if not v["in_band"]]
    lines = [f"FTZ vs standard: {label}", f"  band: |value| < 2**-1021 = {BAND:.6e} in both runs",
             f"  written files: {len(wa)} std, {len(wb)} ftz; {nsame} byte-identical"]
    for rel, d in res["files"].items():
        for k, v in d.items():
            lines.append(f"  file {rel} {k}: {v['n']} points; max |std| {v['max_abs_std']:.6e} |ftz| "
                         f"{v['max_abs_ftz']:.6e}; std subnormal {v['std_subnormal']}, ftz zero {v['ftz_zero']}; "
                         f"e.g. {v['examples'][:2]}; {'in band' if v['in_band'] else 'OUTSIDE'}")
    for rel, t in res["text"].items():
        lines.append(f"  text {rel}: {t['lines']} lines differ")
        lines += [f"    line {e['line']}: {e['tokens']}" for e in t["examples"][:5]]
    lines.append(f"  STDOUT: {res['stdout']['lines']} lines differ (exempt: BUILD_STAMP, TIMER_VALUE)")
    lines += [f"    line {e['line']}: {e['tokens']}" for e in res["stdout"]["examples"][:10]]
    if jdon:
        lines.append(f"  dumps: {len(res['dumps'])} (iteration, stage, field) keys differ")
        for k, v in res["dumps"].items():
            lines.append(f"  dump {k}: {v['n']} points ({v['n_interior']} interior) on tiles {sorted(set(v['tiles']))}; "
                         f"max |std| {v['max_abs_std']:.6e} |ftz| {v['max_abs_ftz']:.6e}; std subnormal "
                         f"{v['std_subnormal']}, ftz zero {v['ftz_zero']}, ftz subnormal {v['ftz_subnormal']}; "
                         f"{'in band' if v['in_band'] else 'OUTSIDE'}")
    verdict = (f"OUTSIDE BAND {label}: {probs[0]} ({len(probs)} problems)" if probs else
               f"BAND CONFINED {label}: {len(res['files'])} files, {res['stdout']['lines']} STDOUT lines, "
               f"{len(res['dumps'])} dump keys differ, all within |value| < 2**-1021")
    lines.append(verdict)
    lines += [f"  problem: {p}" for p in probs[:30]]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if a.out:
        a.out.write_text(text)
    if a.json:
        res["verdict"] = verdict
        res["problems"] = probs
        a.json.write_text(json.dumps(res, indent=1) + "\n")
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
