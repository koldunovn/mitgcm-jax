#!/usr/bin/env python3
"""Compare two jaxdump sets key by key in the reference's call order; report the FIRST (iteration, stage, field)
that differs beyond tolerance (plan Task 4; copied from the ECCO port's tools/diffdump.py and generalised).

    diffdump.py REF_DIR TEST_DIR [--rtol 1e-13] [--tol FIELD=RTOL ...] [--interior-only] [--all]

What is compared (docs/PORTING_RULES.md §1, §3, §5): EVERY point of every record, halos and dry points included,
by element equality against the reference (the Fortran oracle). The Fortran computes values on halo and land points
too, and the port must reproduce them; a comparison that skips them lets NaN or garbage there pass.
  * Non-finite values: a non-finite value anywhere in the test set is NAN (a failure), and so is a non-finite value in
    the reference at a compared point (the comparison needs a finite oracle; Python's max() never picks a NaN, so
    finiteness is checked explicitly, before any metric).
  * Metric: d = |test - ref| at every compared point; scale = max|ref| over the compared points (the tolerance class
    applies relative to the field's max abs); rel = max d / scale; a point is over when d > rtol * scale (over when
    d > 0 if the reference is zero at every compared point). The NUMBER of points over is gated, not only the
    maximum: any point over is FAIL.
  * Element equality is checked first: identical values are "same" (equality of values: -0 equals +0).
Masks are for REPORTING, never for hiding points: wet points come from the reference's own hFacC, hFacW, hFacS
records (the first stage that dumped all three, `G00_geometry` or `S00_begin`), level by level:
  * a record with nz = Nr uses the mask of each level;
  * nz = 1 uses level 1, except a per-level slice `<name>_k<kkk>` (JAXDUMP_TILEK), which uses level kkk;
  * nz = Nr+1 (interface fields, e.g. kappaRU) uses, at interface k, "level k-1 or level k wet" (bounded);
  * kind Z (vorticity points, i at the west and j at the south cell edge): wet if any of the four adjacent velocity
    points is wet (hFacW at j and j-1, hFacS at i and i-1);
  * kinds N (scalars) and V (vertical grid) have no horizontal position (no halo, every point "wet").
Every result carries the point counts and the points over tolerance split into wet interior, dry interior and halo.
If a record of kind C, W, S or Z has no mask (no hFac in the reference, or a level count none of the rules covers),
the comparison REFUSES to run (exit 2), naming the record (lesson [E§4]: a comparator refuses to run without masks).
Flags (always reported, never silently passed):
  ZERO    the field is identically zero on wet points in BOTH sets (allocated but never computed? -> check); a
          warning, given only when nothing failed
  MISSING key, occurrence or tile present in only one set
  NAN     non-finite values (see above)
Exit 1 if anything exceeds tolerance or is MISSING/NAN; ZERO alone is a warning; exit 2 on a refusal.
Tolerance classes (docs/PORTING_RULES.md §5): pointwise ~1e-15, stencils/reductions ~1e-13, solver fields at the
solver tolerance; per field with --tol (a looser value needs a reason in the gate that sets it).

--interior-only is an explicit opt-in that skips the halo points of C/W/S/Z records (e.g. while a routine's halo
update is not ported yet); it is never the default, every output line then carries "INTERIOR-ONLY", and the test
set must still be finite at every point.
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mitjax.io.dump import DumpSet  # noqa: E402

_KSLICE = re.compile(r"_k(\d{3})$")
MASK_FIELDS = {"C": "hFacC", "W": "hFacW", "S": "hFacS"}
INTERIOR_ONLY_LABEL = "INTERIOR-ONLY (halo points NOT compared: explicit opt-in --interior-only)"


class MaskError(RuntimeError):
    """A record cannot be masked: the comparison refuses to run."""


def masks_from(ds):
    """{kind: {tile: bool (Nr, ny+2oly, nx+2olx)}} for C, W, S (from the first stage that dumped hFacC, hFacW and
    hFacS together) and Z (derived from W and S); {} if the reference has no hFac records."""
    for it, stage, fld in ds.keys():
        if fld != "hFacC":
            continue
        if all((it, stage, f) in ds.index for f in MASK_FIELDS.values()):
            out = {k: {t: r.data > 0 for t, r in ds.tiles(it, stage, f).items()} for k, f in MASK_FIELDS.items()}
            out["Z"] = {}
            for t, w in out["W"].items():
                s = out["S"][t]
                z = w.copy()
                z[:, 1:, :] |= w[:, :-1, :]
                z |= s
                z[:, :, 1:] |= s[:, :, :-1]
                out["Z"][t] = z
            return out
    return {}


def record_mask(masks, rec, shape, fld):
    """Bool wet mask of the record's shape (nz, ny, nx); raises MaskError when none of the rules applies."""
    if rec.kind in ("N", "V"):
        return np.ones(shape, dtype=bool)
    m = masks.get(rec.kind, {}).get(rec.tile)
    if m is None:
        raise MaskError(f"{rec.stage}/{fld} kind {rec.kind} tile {rec.tile}: no {rec.kind} mask in the reference "
                        f"(needs hFacC/hFacW/hFacS records)")
    nr, nz = m.shape[0], shape[0]
    if nz == nr:
        return m
    if nz == 1:
        k = _KSLICE.search(fld)
        lev = int(k.group(1)) - 1 if k else 0
        if not 0 <= lev < nr:
            raise MaskError(f"{rec.stage}/{fld}: level {lev + 1} outside 1..{nr}")
        return m[lev:lev + 1]
    if nz == nr + 1:
        out = np.zeros(shape, dtype=bool)
        for k in range(nz):
            out[k] = m[min(k, nr - 1)] | m[max(k - 1, 0)]
        return out
    raise MaskError(f"{rec.stage}/{fld}: nz = {nz} matches no mask rule (Nr = {nr})")


def halo_mask(rec, shape):
    """True on the halo points of a record (none for kinds N and V, which have no horizontal position)."""
    h = np.ones(shape, dtype=bool)
    if rec.kind in ("N", "V"):
        h[:] = False
    else:
        h[:, rec.oly:shape[1] - rec.oly, rec.olx:shape[2] - rec.olx] = False
    return h


@dataclass
class Result:
    """One key and occurrence. Counts are over the compared points; `over` and `nonfinite` split into wet interior,
    dry interior and halo points."""
    status: str                     # same, ok, FAIL, ZERO, NAN, MISSING
    rel: float = np.inf
    n_over: int = 0
    n: int = 0
    points: dict = field(default_factory=dict)       # {"wet": n, "dry": n, "halo": n} compared
    over: dict = field(default_factory=dict)         # same split, points over tolerance
    nonfinite: dict = field(default_factory=dict)    # {"test": {...}, "ref": {...}} non-finite points
    interior_only: bool = False

    def where(self):
        """Short text: where the points over tolerance (or the non-finite values) are."""
        def split(d):
            return ", ".join(f"{k} {v}" for k, v in d.items() if v) or "none"
        txt = ""
        if self.status == "NAN":
            txt = "; ".join(f"non-finite in {s}: {split(d)}" for s, d in self.nonfinite.items() if sum(d.values()))
        elif self.n_over:
            txt = f"over: {split(self.over)}"
        if self.interior_only:
            txt = (txt + "; " if txt else "") + "INTERIOR-ONLY"
        return txt


def _split(sel, wet, halo):
    return {"wet": int((sel & wet & ~halo).sum()), "dry": int((sel & ~wet & ~halo).sum()),
            "halo": int((sel & halo).sum())}


def compare_key(ref, test, key, masks, rtol, interior_only=False, occ=0):
    """-> Result for one key and occurrence (status in same, ok, FAIL, ZERO, NAN, MISSING)."""
    if key not in test.index or occ >= test.n_occ(key) or occ >= ref.n_occ(key):
        return Result("MISSING", interior_only=interior_only)
    fld = key[2]
    rt, tt = ref.tiles(*key, occ=occ), test.tiles(*key, occ=occ)
    if set(rt) != set(tt):
        return Result("MISSING", interior_only=interior_only)
    parts = []
    for tile, r in rt.items():
        a, b = r.data, tt[tile].data
        if a.shape != b.shape:
            return Result("MISSING", interior_only=interior_only)
        wet = record_mask(masks, r, a.shape, fld)
        halo = halo_mask(r, a.shape)
        cmp = ~halo if interior_only else np.ones(a.shape, dtype=bool)
        parts.append((a, b, wet, halo, cmp))
    zero3 = {"wet": 0, "dry": 0, "halo": 0}
    res = Result("same", 0.0, 0, 0, dict(zero3), dict(zero3), {"test": dict(zero3), "ref": dict(zero3)},
                 interior_only)
    for a, b, wet, halo, cmp in parts:
        for k, v in _split(cmp, wet, halo).items():
            res.points[k] += v
        # the test set must be finite EVERYWHERE (also on points an interior-only comparison skips)
        for k, v in _split(~np.isfinite(b), wet, halo).items():
            res.nonfinite["test"][k] += v
        for k, v in _split(cmp & ~np.isfinite(a), wet, halo).items():
            res.nonfinite["ref"][k] += v
    res.n = sum(res.points.values())
    if sum(res.nonfinite["test"].values()) or sum(res.nonfinite["ref"].values()):
        res.status, res.rel = "NAN", np.inf
        return res
    scale = max((float(np.abs(a[cmp]).max()) for a, _, _, _, cmp in parts if cmp.any()), default=0.0)
    identical = all(np.array_equal(a[cmp], b[cmp]) for a, b, _, _, cmp in parts)
    if not identical:
        dmax = 0.0
        for a, b, wet, halo, cmp in parts:
            d = np.where(cmp, np.abs(b - a), 0.0)
            dmax = max(dmax, float(d.max()))
            over = cmp & (d > rtol * scale)
            for k, v in _split(over, wet, halo).items():
                res.over[k] += v
        res.n_over = sum(res.over.values())
        res.rel = dmax / scale if scale > 0 else np.inf
        res.status = "FAIL" if res.n_over else "ok"
        if res.n_over:
            return res
    wet_zero = all(not np.any(a[wet & cmp]) and not np.any(b[wet & cmp]) for a, b, wet, _, cmp in parts)
    if res.points["wet"] and wet_zero:
        res.status = "ZERO"
    return res


def compare(ref, test, rtol=1e-13, tols=None, interior_only=False):
    """Rows (key, occ, Result) in the reference's call order, then keys only in the test set."""
    tols = tols or {}
    masks = masks_from(ref)
    rows = []
    for key in ref.keys():
        for occ in range(max(ref.n_occ(key), test.n_occ(key) if key in test.index else 0)):
            rows.append((key, occ, compare_key(ref, test, key, masks, tols.get(key[2], rtol), interior_only, occ)))
    for key in test.keys():
        if key not in ref.index:
            rows.append((key, 0, Result("MISSING", interior_only=interior_only)))
    return rows


BAD = ("FAIL", "MISSING", "NAN")


def row_text(key, occ, r):
    it, stage, fld = key
    w = r.where()
    return (f"{r.status:7s} iter {it} {stage:28s} {fld:18s} occ {occ} rel {r.rel:.3e} over {r.n_over}/{r.n}"
            + (f" ({w})" if w else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--rtol", type=float, default=1e-13)
    ap.add_argument("--tol", nargs="*", default=[], help="FIELD=RTOL overrides")
    ap.add_argument("--interior-only", action="store_true",
                    help="explicit opt-in: skip halo points (labelled INTERIOR-ONLY in every line); never the default")
    ap.add_argument("--all", action="store_true", help="print every comparison, not just the first failure")
    a = ap.parse_args(argv)
    tols = {k: float(v) for k, v in (t.split("=") for t in a.tol)}
    if a.interior_only:
        print(INTERIOR_ONLY_LABEL)
    try:
        rows = compare(DumpSet(a.ref), DumpSet(a.test), a.rtol, tols, a.interior_only)
    except MaskError as e:
        print(f"REFUSED: {e}")
        return 2
    bad = [r for r in rows if r[2].status in BAD]
    zero = [r for r in rows if r[2].status == "ZERO"]
    for key, occ, r in (rows if a.all else bad[:1]):
        print(row_text(key, occ, r))
    for (it, stage, fld), occ, _ in zero:
        print(f"ZERO    iter {it} {stage:28s} {fld} occ {occ} (zero on wet points in both)")
    pts = {k: sum(r[2].points.get(k, 0) for r in rows) for k in ("wet", "dry", "halo")}
    print(f"{len(rows)} compared ({pts['wet']} wet, {pts['dry']} dry, {pts['halo']} halo points), "
          f"{sum(r[2].status == 'same' for r in rows)} identical, {len(bad)} bad, {len(zero)} zero"
          + (f"  [{INTERIOR_ONLY_LABEL}]" if a.interior_only else ""))
    if bad:
        (it, stage, fld), occ, r = bad[0]
        print(f"FIRST DIFFERENCE: iter {it}, stage {stage}, field {fld}, occ {occ} ({r.status}, rel {r.rel:.3e}"
              + (f", {r.where()}" if r.where() else "") + ")")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
