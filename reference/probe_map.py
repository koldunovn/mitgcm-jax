#!/usr/bin/env python3
"""Exchange-probe maps of a jaxdump run (X00_exch_probe), for any tile layout including the exch2 cube (lane A
session 6, M2; for lane B, who builds the cube exchange maps from them).

    probe_map.py DUMPS [--npz OUT.npz] [--json OUT.json]

The probe (reference/jaxdump/jaxdump.F, JAXDUMP_EXCH_PROBE) fills every point of every tile, halos included, with
value = comp*cbase + idx (idx codes the padded point of the SOURCE tile; comp 1 for u-like and scalar fields, 2 for
v-like), runs one exchange routine and dumps the result with halos. Decoding a halo value gives the point it was
copied from, which component it came from (a u halo filled from a v value = a component swap at a rotated face edge)
and its sign (a negative value = the exchange negated it). A point still holding its own code was not written.

Per probe field (`xT`, `xUVs_u`, ...) the summary counts, over all tiles: halo points; points written; points left
unwritten, split into the corner blocks (|i| and |j| both outside the interior) and the edge strips; written points
whose source is a HALO point of the source tile (a copy of a copy); written points whose source tile lies on another
face; component swaps; sign flips. The signed-zero records (zp*, zm*: both components +0 / -0; zu*: u=-0, v=+0;
zv*: u=+0, v=-0) are summarised as counts of -0 in the interior and in the halo of each component.

--npz writes, per probe field, int arrays [tile, j, i] (padded, tiles in increasing W2/tile number): src_tile,
src_ja, src_ia (0-based padded indices of the source point in the source tile), comp, sign, written (0/1); plus
the layout (tile -> face, tBasex, tBasey, sNx, sNy, OLx, OLy). Stdlib + numpy; deletes nothing; OUT must not exist.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from mitjax.io.dump import DumpSet, probe_decode  # noqa: E402

STAGE = "X00_exch_probe"
PROBE_FIELDS = ("xT", "xUVs_u", "xUVs_v", "xUVn_u", "xUVn_v", "xZ", "xAs_u", "xAs_v", "xAn_u", "xAn_v", "xBs_u",
                "xBs_v", "xBn_u", "xBn_v", "x3D", "xSMs", "xDs_u", "xDs_v", "xUV3s_u", "xUV3s_v", "xS3D")


def _comp_of(fld):
    return 2 if fld.endswith("_v") else 1


def decode_field(ds, it, fld):
    """{tile: dict of int arrays [j, i] src_tile, src_ja, src_ia, comp, sign, written} of one probe field
    (padded indices, 0-based); raises ValueError if any value is not a probe code."""
    cbase = ds.scalar(it, STAGE, "xCbase")
    out = {}
    for t, r in sorted(ds.tiles(it, STAGE, fld).items()):
        nyp, nxp = r.shape[1:]
        d = probe_decode(r.data[0], cbase, nxp, nyp)
        ja, ia = np.indices((nyp, nxp))
        own = (d["tile"] == t) & (d["ja"] == ja) & (d["ia"] == ia) & (d["comp"] == _comp_of(fld)) & (d["sign"] > 0)
        out[t] = {"src_tile": d["tile"], "src_ja": d["ja"], "src_ia": d["ia"], "comp": d["comp"], "sign": d["sign"],
                  "written": (~own).astype(np.int64)}
    return out


def summarize_field(ds, it, fld, dec=None):
    """Counts of one probe field (module docstring); `dec` = decode_field's result (else decoded here)."""
    dec = decode_field(ds, it, fld) if dec is None else dec
    info = ds.tiles_info
    s = dict(halo=0, written=0, unwritten_corner=0, unwritten_edge=0, interior_written=0, halo_source=0,
             cross_face=0, comp_swap=0, sign_flip=0)
    for t, d in dec.items():
        face, _, _, snx, sny, olx, oly = info[t]
        nyp, nxp = d["written"].shape
        ja, ia = np.indices((nyp, nxp))
        out_i, out_j = (ia < olx) | (ia >= olx + snx), (ja < oly) | (ja >= oly + sny)
        halo = out_i | out_j
        w = d["written"].astype(bool)
        s["halo"] += int(halo.sum())
        s["written"] += int((w & halo).sum())
        s["interior_written"] += int((w & ~halo).sum())
        s["unwritten_corner"] += int((~w & out_i & out_j).sum())
        s["unwritten_edge"] += int((~w & halo & ~(out_i & out_j)).sum())
        if w.any():
            st = d["src_tile"][w]
            sinfo = np.array([info[int(x)] for x in st]).reshape(-1, 7)   # face, tbx, tby, snx, sny, olx, oly
            sja, sia = d["src_ja"][w], d["src_ia"][w]
            s_int = ((sia >= sinfo[:, 5]) & (sia < sinfo[:, 5] + sinfo[:, 3]) & (sja >= sinfo[:, 6])
                     & (sja < sinfo[:, 6] + sinfo[:, 4]))
            s["halo_source"] += int((~s_int).sum())
            s["cross_face"] += int((sinfo[:, 0] != face).sum())
            s["comp_swap"] += int((d["comp"][w] != _comp_of(fld)).sum())
            s["sign_flip"] += int((d["sign"][w] < 0).sum())
    return s


def zero_summary(ds, it):
    """{field: (-0 in interior, -0 in halo, nonzero values, points)} of every signed-zero probe record."""
    out = {}
    for k in ds.keys(it):
        if k[1] != STAGE or k[2][:2] not in ("zp", "zm", "zu", "zv"):
            continue
        neg_i = neg_h = nonzero = n = 0
        for t, r in ds.tiles(it, STAGE, k[2]).items():
            v = r.data[0]
            nyp, nxp = v.shape
            ja, ia = np.indices((nyp, nxp))
            halo = (ia < r.olx) | (ia >= r.olx + r.snx) | (ja < r.oly) | (ja >= r.oly + r.sny)
            neg = np.signbit(v) & (v == 0)
            neg_i += int((neg & ~halo).sum())
            neg_h += int((neg & halo).sum())
            nonzero += int(np.count_nonzero(v))
            n += v.size
        out[k[2]] = (neg_i, neg_h, nonzero, n)
    return out


def summary(dumps):
    ds = DumpSet(dumps)
    it = ds.iterations()[0]
    fields = {f: summarize_field(ds, it, f) for f in PROBE_FIELDS}
    return {"dumps": str(dumps), "iteration": it, "ntiles": int(ds.scalar(it, STAGE, "xNtiles")),
            "faces": {str(f): list(s) for f, s in sorted(ds.face_shapes().items())},
            "fields": fields, "zeros": {k: list(v) for k, v in zero_summary(ds, it).items()}}


def write_npz(dumps, out):
    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists (nothing is overwritten)")
    ds = DumpSet(dumps)
    it = ds.iterations()[0]
    arrays = {}
    for fld in PROBE_FIELDS:
        dec = decode_field(ds, it, fld)
        for name in ("src_tile", "src_ja", "src_ia", "comp", "sign", "written"):
            arrays[f"{fld}__{name}"] = np.stack([dec[t][name] for t in sorted(dec)]).astype(np.int32)
    tiles = sorted(ds.tiles_info)
    arrays["tiles"] = np.array(tiles, dtype=np.int32)
    arrays["layout"] = np.array([ds.tiles_info[t] for t in tiles], dtype=np.int32)  # face tbx tby snx sny olx oly
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dumps")
    ap.add_argument("--npz")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    s = summary(a.dumps)
    text = json.dumps(s, indent=1)
    if a.json:
        if Path(a.json).exists():
            raise SystemExit(f"{a.json} exists (nothing is overwritten)")
        Path(a.json).write_text(text + "\n")
    else:
        print(text)
    if a.npz:
        print(f"WROTE {write_npz(a.dumps, a.npz)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
