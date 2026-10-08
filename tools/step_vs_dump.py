#!/usr/bin/env python3
"""Compare one time step of a candidate with the Fortran oracle's substep dumps, stage by stage in the oracle's call
order, and report the FIRST stage that differs (plan Task 4, lesson [E§4]; the ECCO port's tools/step_vs_dump.py
ran its own JAX FORWARD_STEP on the LLC90 oracle; mitjax has no time step yet, so the candidate is pluggable).

    step_vs_dump.py ORACLE_DIR --it IT --candidate-dir DIR          a second dump set (e.g. another build)
    step_vs_dump.py ORACLE_DIR --it IT --candidate MODULE:FUNCTION  FUNCTION(oracle DumpSet, it) ->
                                                                    {(stage, field): array [tile, k, j, i]}
    [--rtol 1e-13] [--tol FIELD=RTOL ...] [--stage-prefix S] [--not-ported STAGE/FIELD ...] [--interior-only]

A candidate function returns the fields it computes, with halos, tiles in increasing tile number (the layout of
DumpSet.field); the JAX model's step wrapper (M1) plugs in here. Every (stage, field) the oracle dumped for IT in the
selected stages (--stage-prefix) is compared: all points, halos and dry points included, by tools/diffdump.py's
compare_key (masks for reporting only; it refuses to run without masks; --interior-only is diffdump's labelled
opt-in). An oracle key the candidate does not provide is a FAILURE ("NOT COMPUTED"), unless it matches a
--not-ported pattern (fnmatch on "stage/field", e.g. `S06_dynamics/*` or `G00_geometry/hFac?`): those are listed as
"not ported yet" and never compared. Candidate keys the oracle never dumped are an error. A run that compares nothing
fails. Output: one line per stage (worst field), then `FIRST DIFFERING STAGE: <stage> (<field>, <status>, ...)` or
`ALL STAGES AGREE` (only when every selected oracle key was compared or declared not ported, at least one was
compared, and none differs); exit 1 / 0, 2 on a refusal.
"""

import argparse
import copy
import fnmatch
import importlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mitjax.io.dump import DumpSet  # noqa: E402
from tools.diffdump import BAD, INTERIOR_ONLY_LABEL, MaskError, compare_key, masks_from  # noqa: E402


class CandidateSet:
    """DumpSet look-alike holding candidate arrays on the oracle's record metadata (occurrence 0 only)."""

    def __init__(self, oracle, it, fields):
        self.index = {}
        for (stage, fld), arr in fields.items():
            key = (it, stage, fld)
            if key not in oracle.index:
                raise KeyError(f"candidate field {stage}/{fld} was never dumped by the oracle at iteration {it}")
            recs = oracle.tiles(*key)
            arr = np.asarray(arr, dtype=np.float64)
            if arr.shape != (len(recs), *next(iter(recs.values())).shape):
                raise ValueError(f"{stage}/{fld}: candidate shape {arr.shape} != oracle "
                                 f"{(len(recs), *next(iter(recs.values())).shape)}")
            tiles = {}
            for n, t in enumerate(sorted(recs)):
                r = copy.copy(recs[t])
                r._data = arr[n]
                tiles[t] = r
            self.index[key] = [tiles]

    def n_occ(self, key):
        return len(self.index[key])

    def tiles(self, it, stage, fld, occ=0):
        return self.index[(it, stage, fld)][occ]

    def keys(self, it=None):
        return list(self.index)


NOT_COMPUTED = "NOT COMPUTED"


def run(oracle, cand, it, rtol=1e-13, tols=None, prefix="", not_ported=(), interior_only=False):
    """-> (per-stage rows [(stage, worst (status, field, Result))], first bad stage row or None, not-computed keys
    (failures), not-ported keys (declared), number of keys compared). A not-computed key makes its stage bad."""
    tols = tols or {}
    masks = masks_from(oracle)
    per_stage, first, missing, declared, n_cmp = [], None, [], [], 0
    for stage in oracle.stages(it):
        if not stage.startswith(prefix):
            continue
        worst = None
        for key in oracle.keys(it):
            if key[1] != stage:
                continue
            if any(fnmatch.fnmatchcase(f"{key[1]}/{key[2]}", p) for p in not_ported):
                declared.append(key)
                continue
            if key not in cand.index:
                missing.append(key)
                row = (NOT_COMPUTED, key[2], None)
            else:
                res = compare_key(oracle, cand, key, masks, tols.get(key[2], rtol), interior_only)
                n_cmp += 1
                row = (res.status, key[2], res)
            bad = row[0] in BAD or row[0] == NOT_COMPUTED
            rank = (bad, row[2].rel if row[2] is not None else np.inf)
            if worst is None or rank > (worst[0] in BAD or worst[0] == NOT_COMPUTED,
                                        worst[2].rel if worst[2] is not None else np.inf):
                worst = row
            if bad and first is None:
                first = (stage, row)
        if worst is not None:
            per_stage.append((stage, worst))
    return per_stage, first, missing, declared, n_cmp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("oracle")
    ap.add_argument("--it", type=int, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--candidate-dir")
    g.add_argument("--candidate", help="MODULE:FUNCTION")
    ap.add_argument("--rtol", type=float, default=1e-13)
    ap.add_argument("--tol", nargs="*", default=[])
    ap.add_argument("--stage-prefix", default="")
    ap.add_argument("--not-ported", nargs="*", default=[],
                    help="stage/field fnmatch patterns of oracle keys the candidate does not compute yet")
    ap.add_argument("--interior-only", action="store_true",
                    help="explicit opt-in: skip halo points (labelled INTERIOR-ONLY); never the default")
    a = ap.parse_args(argv)
    oracle = DumpSet(a.oracle)
    if a.candidate_dir:
        cand = DumpSet(a.candidate_dir)
    else:
        mod, fn = a.candidate.split(":")
        cand = CandidateSet(oracle, a.it, getattr(importlib.import_module(mod), fn)(oracle, a.it))
    tols = {k: float(v) for k, v in (t.split("=") for t in a.tol)}
    if a.interior_only:
        print(INTERIOR_ONLY_LABEL)
    try:
        per_stage, first, missing, declared, n_cmp = run(oracle, cand, a.it, a.rtol, tols, a.stage_prefix,
                                                         a.not_ported, a.interior_only)
    except MaskError as e:
        print(f"REFUSED: {e}")
        return 2
    for stage, (status, fld, res) in per_stage:
        if res is None:
            print(f"{stage:30s} {status:12s} {fld}")
        else:
            w = res.where()
            print(f"{stage:30s} {status:12s} worst {fld:18s} rel {res.rel:.3e} over {res.n_over}/{res.n}"
                  + (f" ({w})" if w else ""))
    if declared:
        print(f"not ported yet (declared, not compared): {len(declared)} oracle keys, e.g. "
              f"{', '.join(f'{s}/{f}' for _, s, f in declared[:5])}")
    if missing:
        print(f"NOT COMPUTED by the candidate (failure; declare with --not-ported if intended): {len(missing)} oracle "
              f"keys: {', '.join(f'{s}/{f}' for _, s, f in missing[:10])}")
    if first:
        stage, (status, fld, res) = first
        detail = "" if res is None else f", rel {res.rel:.3e}, {res.n_over} points over" + (
            f", {res.where()}" if res.where() else "")
        print(f"FIRST DIFFERING STAGE: {stage} ({fld}, {status}{detail})")
        return 1
    if n_cmp == 0:
        print("NOTHING COMPARED: the candidate provides none of the selected oracle keys")
        return 1
    print("ALL STAGES AGREE" + (f"  [{INTERIOR_ONLY_LABEL}]" if a.interior_only else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
