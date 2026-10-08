#!/usr/bin/env python3
"""Write the exchange gather maps of the M1 layouts from the oracle's exchange probe (plan Task 7b).

    make_exch_maps.py [EXP ...] [--out-dir DIR]
    make_exch_maps.py EXP --code CODE [--out-dir DIR]      (a build listed in exch_maps.VARIANT_PROBES)

For each experiment (default: all of mitjax.eesupp.exch_maps.M1_PROBES) read the dumps-on probe run
`$MJX_REFERENCE_RUNS/<exp>/<input>/<run job>-jdon/dumps` (lane A, plan Task 4), build the maps (exch_maps.build_maps:
every value decodes, interior points unchanged, scalar exchanges take no v-component and no sign), check that the
maps reproduce every probe output and the signed-zero probes bitwise (numpy, the same formula as Exchanger), and write
`<out-dir>/<exp>-<layout tag>.npz` + `.npz.sha256` + `.json` (default out-dir `$MJX_REFERENCE/exch_maps`). Refuses
to overwrite an existing file (nothing is ever deleted). Prints, per map, the halo points written, taken from the
other component, sign flips and not written, then the sha256 lines to register in exch_maps.MAP_SHA256.

GOADK lane (M2): `--code CODE` takes the probe run of exch_maps.VARIANT_PROBES[(EXP, CODE)] (a build whose tiling
differs from the experiment's code/ build, e.g. global_ocean.90x40x15/code_ad) and writes `<EXP>-<CODE>-<tag>.npz`,
to register in exch_maps.VARIANT_MAP_SHA256.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from mitjax import paths  # noqa: E402
from mitjax.eesupp import exch_maps as EM  # noqa: E402
from mitjax.io.dump import DumpSet  # noqa: E402


def apply_np(m, own, fa, fb):
    """numpy twin of exchange.apply_map (used here to check the maps against the probe before writing them)."""
    src, comp, sign = m
    out = np.where(comp == 1, fa[src], own)
    if fb is not None:
        out = np.where(comp == 2, fb[src], out)
    return out * sign.astype(np.float64)


def check_against_probe(m, ds, it, stage=EM.STAGE):
    """Every probe output and signed-zero probe equals the map applied to the probe input, bitwise (sign bit
    included). Returns {probe field: number of differing points}."""
    L = m.layout
    cbase = ds.scalar(it, stage, "xCbase")
    pu, pv = (x.reshape(-1) for x in EM.probe_inputs(L, cbase))
    bad = {}

    def cmp(fld, got):
        want = ds.field(it, stage, fld)[:, 0].reshape(-1)
        bad[fld] = int(np.sum((got.view(np.int64) != want.view(np.int64))))

    for name, f in EM.SCALAR.items():
        cmp(f, apply_np(m.maps[name], pu, pu, None))
    for name, (fu, fv) in EM.VECTOR.items():
        cmp(fu, apply_np(m.maps[name + "_u"], pu, pu, pv))
        cmp(fv, apply_np(m.maps[name + "_v"], pv, pu, pv))
    for name, outs in EM.ZERO_PROBES.items():
        for pm, z in (("p", 0.0), ("m", -0.0)):
            zu = np.full(L.npoints, z)
            for suf in outs:
                key = name + suf if name in EM.VECTOR else name
                cmp(f"z{pm}{name}{suf}", apply_np(m.maps[key], zu, zu, zu if name in EM.VECTOR else None))
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("exps", nargs="*", default=list(EM.M1_PROBES))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--code", default=None, help="code directory of a build in exch_maps.VARIANT_PROBES")
    a = ap.parse_args(argv)
    out_dir = a.out_dir or EM.map_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    shas, rc = {}, 0
    for exp in a.exps:
        if a.code is not None:                                  # GOADK lane: a variant build's own probe run
            inp, rid = EM.VARIANT_PROBES[(exp, a.code)]
            job = rid.split("-")[0]
            run = paths.REFERENCE_RUNS / exp / inp / rid
        else:
            inp, job = {**EM.M1_PROBES, **EM.M2_PROBES, **EM.M3_PROBES, **EM.M4_PROBES}[exp]   # M2/M3: the experiments of M2/M3_PROBES
            run = paths.REFERENCE_RUNS / exp / inp / f"{job}-jdon"
        ds = DumpSet(run / "dumps")
        it = ds.iterations()[0]
        manifest = run / "MANIFEST.json"
        meta = {"exp": exp, "input": inp, "probe_run": str(run), "run_job": job,
                "manifest": json.loads(manifest.read_text()) if manifest.exists() else None}
        m = EM.build_maps(ds, it, meta=meta)
        bad = check_against_probe(m, ds, it)
        nbad = sum(bad.values())
        print(f"== {exp} ({inp}, {job}) layout {m.layout.tag()}: probe reproduced "
              f"{'bitwise' if nbad == 0 else 'NOT: ' + str({k: v for k, v in bad.items() if v})}")
        if nbad:
            rc = 1
            continue
        nhalo = int((~np.broadcast_to(m.layout.interior(), m.layout.shape2d)).sum())
        print(f"   {'map':8s} {'written':>8s} {'other':>6s} {'neg':>5s} {'unwritten':>9s}   (halo points: {nhalo})")
        for k, (w, o, n, u) in EM.summary(m).items():
            print(f"   {k:8s} {w:8d} {o:6d} {n:5d} {u:9d}")
        for a1, a2 in (("XY", "3D"), ("UVs_u", "UV3s_u"), ("UVs_v", "UV3s_v")):
            same = all(np.array_equal(x, y) for x, y in zip(m.maps[a1], m.maps[a2]))
            print(f"   {a1} == {a2}: {same}")
        path = out_dir / (f"{exp}-{a.code}-{m.layout.tag()}.npz" if a.code is not None
                          else f"{exp}-{m.layout.tag()}.npz")
        shas[path.name] = m.save(path)
        print(f"   wrote {path}")
    print("\nMAP_SHA256 = {")
    for k, v in shas.items():
        print(f'    "{k}": "{v}",')
    print("}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
