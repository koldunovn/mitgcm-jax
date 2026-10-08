"""Tier-3 twin comparison (plan Task 18; lane B): one JAX run against two Fortran runs that differ only at round-off.

    python scripts/twin_compare.py F1/output.txt F2/output.txt JAX/output.txt [--json out.json]

The criterion was fixed before the runs were looked at (coordinator, 2026-10-02): for every `%MON` statistic at every
monitor time (blocks matched by `time_tsnumber`), with f1, f2 the two Fortran values and j the JAX value,
    spread = |f2 - f1|,  envelope = [min(f1, f2) - spread, max(f1, f2) + spread],
and j passes when it lies inside the envelope (a zero spread admits only j == f1 == f2). Reported: the fraction of
(statistic, time) pairs inside, per statistic, and the worst offenders ranked by excess / spread (the distance
outside the envelope in units of the Fortran spread; infinite for a zero spread), with |j - f1| / |f1|.
"""

import argparse
import json
import math
import re
import sys

MON = re.compile(r"%MON (\w+)\s*=\s*(\S+)")


def _value(text):
    v = text.replace("D", "E")
    e3 = re.fullmatch(r"([+-]?\d*\.\d+)([+-]\d{3})", v)      # 1PE with a 3-digit exponent prints no letter
    return float(f"{e3.group(1)}E{e3.group(2)}" if e3 else v)


def blocks(path):
    """{time_tsnumber: {statistic: value}} of a STDOUT file's %MON blocks."""
    out, cur = {}, None
    for ln in open(path, errors="replace"):
        m = MON.search(ln)
        if not m:
            continue
        name, val = m.group(1), _value(m.group(2))
        if name == "time_tsnumber":
            cur = out.setdefault(int(val), {})
        if cur is not None:
            cur[name] = val
    return out


def compare(f1, f2, jx):
    b1, b2, bj = blocks(f1), blocks(f2), blocks(jx)
    times = sorted(set(b1) & set(b2) & set(bj))
    rows = []
    for t in times:
        for name in sorted(set(b1[t]) & set(b2[t]) & set(bj[t])):
            a, b, j = b1[t][name], b2[t][name], bj[t][name]
            spread = abs(b - a)
            lo, hi = min(a, b) - spread, max(a, b) + spread
            inside = lo <= j <= hi
            excess = 0.0 if inside else (j - hi if j > hi else lo - j)
            ratio = 0.0 if inside else (excess / spread if spread > 0 else math.inf)
            rel = abs(j - a) / abs(a) if a != 0 else (0.0 if j == 0 else math.inf)
            rows.append(dict(time=t, stat=name, f1=a, f2=b, jax=j, spread=spread, inside=inside, ratio=ratio,
                             rel=rel))
    per = {}
    for r in rows:
        n, k = per.get(r["stat"], (0, 0))
        per[r["stat"]] = (n + 1, k + int(r["inside"]))
    worst = sorted((r for r in rows if not r["inside"]), key=lambda r: (-r["ratio"], -r["rel"]))[:20]
    return dict(times=len(times), times_f1=len(b1), times_f2=len(b2), times_jax=len(bj), pairs=len(rows),
                inside=sum(r["inside"] for r in rows),
                per_stat={k: dict(pairs=n, inside=i) for k, (n, i) in sorted(per.items())}, worst=worst)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("f1")
    ap.add_argument("f2")
    ap.add_argument("jax")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = compare(a.f1, a.f2, a.jax)
    if a.json:
        with open(a.json, "x") as fh:
            json.dump(res, fh, indent=1, default=str)
    print(f"monitor times compared {res['times']} (F1 {res['times_f1']}, F2 {res['times_f2']}, JAX {res['times_jax']});"
          f" pairs {res['pairs']}, inside the envelope {res['inside']} ({100.0*res['inside']/max(res['pairs'], 1):.2f} %)")
    bad = {k: v for k, v in res["per_stat"].items() if v["inside"] < v["pairs"]}
    print(f"statistics with a pair outside: {len(bad)} of {len(res['per_stat'])}")
    for k, v in bad.items():
        print(f"  {k}: {v['pairs'] - v['inside']} of {v['pairs']} outside")
    for r in res["worst"]:
        print(f"  t={r['time']} {r['stat']}: F1 {r['f1']:.13e} F2 {r['f2']:.13e} JAX {r['jax']:.13e} "
              f"excess/spread {r['ratio']:.3g} |JAX-F1|/|F1| {r['rel']:.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
