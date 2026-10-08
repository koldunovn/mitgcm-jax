"""Helpers of the lab_sea one-tile oracle check (docs plan 20261006 S4, lane LSTILE; test_lstile_run.py).

lab_sea/code's SIZE.h has 2 x 2 tiles of 10 x 8; reference/retile/lab_sea-code/SIZE.h is the same file with one tile
of 20 x 16 (sNx=20, sNy=16, nSx=nSy=1, nothing else changed). The Fortran oracle of that tiling is the retile build
`lab_sea-code-63cdc0b-07ce3bc-retile` (reference/build.sh step 1c) run on lab_sea/input by
reference/jobs/oracle_runs.sbatch (run id `job<ID>-retile` under $MJX_REFERENCE_RUNS/lab_sea/input/).

mitjax runs the same tiling the way a user does (mitjax.load on a copy of the experiment whose code/SIZE.h is the
retile file, Experiment.run), and its STDOUT records and pickups are compared with that oracle run's with the
criterion of lab_sea's four-tile forward gate (test_m4lab_run::test_whole_run_monitor_solver_digits_pickups): every
MONITOR record and solver line identical, the testreport digits, the three pickups at iteration 10 byte-identical.
"""

import shutil
import warnings
from pathlib import Path

import numpy as np

from mitjax import paths

EXP, INP = "lab_sea", "input"
RETILE_SIZE_H = paths.REPO / "reference" / "retile" / "lab_sea-code" / "SIZE.h"
RETILE_RUN = "job27944614-retile"          # oracle_runs.sbatch on lab_sea-code-63cdc0b-07ce3bc-retile
FOUR_TILE_RUN = "job27855987-plain"        # lane A's plain run of lab_sea-code-63cdc0b-704fd6b (2 x 2 tiles)
PICKUPS = ("pickup.0000000010", "pickup_cd.0000000010", "pickup_seaice.0000000010")


def oracle_dir(run_id=RETILE_RUN):
    """The oracle run directory (rundir/ with output.txt = STDOUT and the pickups); None if it is not there."""
    d = paths.REFERENCE_RUNS / EXP / INP / run_id / "rundir"
    return d if (d / "output.txt").is_file() else None


def make_experiment(dst):
    """A copy of verification/lab_sea (code, input, results) in the new directory `dst` with code/SIZE.h replaced by
    the retile SIZE.h (one 20 x 16 tile). Returns dst."""
    src = paths.UPSTREAM / "verification" / EXP
    dst = Path(dst)
    dst.mkdir(parents=True)
    for d in ("code", INP, "results"):
        shutil.copytree(src / d, dst / d, symlinks=False)
    shutil.copyfile(RETILE_SIZE_H, dst / "code" / "SIZE.h")
    return dst


def run_mitjax(work, plant=None):
    """mitjax's one-tile run of lab_sea/input in `work` (new): (Run, [(warning category, message)]). `plant`, if
    given, is called with the copied experiment directory before loading (negative controls edit an input there)."""
    import mitjax
    work = Path(work)
    exp_dir = make_experiment(work / "lab_sea")
    if plant is not None:
        plant(exp_dir)
    exp = mitjax.load(exp_dir, INP)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        run = exp.run(out=work / "run")
    return run, [(x.category.__name__, str(x.message)) for x in w], exp


def out_dir(tag):
    """A new directory for one run of this lane's tests (never reused, never removed)."""
    import os
    import time
    return paths.RUNS / "tests_lstile" / f"{tag}-{os.getpid()}-{time.time_ns()}"


def restore_four_tile_size_h(exp_dir):
    """Negative-control plant: put lab_sea/code's own SIZE.h (2 x 2 tiles of 10 x 8) back into the copy."""
    shutil.copyfile(paths.UPSTREAM / "verification" / EXP / "code" / "SIZE.h", Path(exp_dir) / "code" / "SIZE.h")


def lines(path):
    return Path(path).read_bytes().decode("latin-1").split("\n")


def monitor_records(records):
    """Every %MON line and MONITOR banner, in order (m4lab_gate.monitor_records)."""
    return [r for r in records if "%MON " in r or ("// " in r and " MONITOR " in r)]


def solver_records(records):
    """cg2d and SEAICE_LSR lines that are not %MON (m4lab_gate.solver_records)."""
    return [r for r in records if "%MON" not in r and ("cg2d" in r.lower() or "SEAICE_LSR" in r)]


def first_difference(a, b):
    """(index, ours, oracle) of the first differing record, or None; a length difference counts at the end."""
    for n, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return n, x, y
    return None if len(a) == len(b) else (min(len(a), len(b)), None, None)


def digits(output, reference, exp_dir):
    """{variable: testreport digits} of the output file `output` against the reference STDOUT `reference`
    (mitjax.compare with the experiment's check lists), and the verdict line."""
    import mitjax
    from mitjax.api import Results
    res = Results(Path(exp_dir), INP, output=Path(reference))
    rep = mitjax.compare(Path(output), res, kind="fwd")
    return {v.name: v.digits for v in rep.run.variables}, rep.summary


def _levels(name, meta_names, nrec, nr):
    if not meta_names:                                  # pickup_cd: no fldList; 3-D records of Nr levels, then 2-D
        n3 = nrec // nr
        return [(f"cd3d_{k}", nr) for k in range(n3)] + [(f"cd2d_{k}", 1) for k in range(nrec - n3 * nr)]
    if name.startswith("pickup_seaice"):                # siTICES has nITD levels, the rest are 2-D
        lev0 = nrec - len(meta_names) + 1
        return [(n, lev0 if n == "siTICES" else 1) for n in meta_names]
    n3 = (nrec - len(meta_names)) // (nr - 1)            # pickup: the 3-D fields first (write_pickup.F)
    return [(n, nr if k < n3 else 1) for k, n in enumerate(meta_names)]


def read_pickup(path, nr=23):
    """{field: float64 array [lev, ny, nx]} of a global MDS pickup file (big-endian float64)."""
    path = Path(path)
    meta = path.with_suffix(".meta").read_text()
    dims = [int(x) for x in meta[meta.index("dimList"):].split("[")[1].split("]")[0].split(",")]
    nx, ny = dims[0], dims[3]
    nrec = int(meta[meta.index("nrecords"):].split("[")[1].split("]")[0])
    names = meta[meta.index("fldList"):].split("{")[1].split("}")[0].split("'")[1::2] if "fldList" in meta else []
    names = [n.strip() for n in names]
    raw = np.fromfile(path, dtype=">f8").reshape(nrec, ny, nx)
    out, k = {}, 0
    for n, lev in _levels(path.name, names, nrec, nr):
        out[n] = raw[k:k + lev].astype(np.float64)
        k += lev
    assert k == nrec, (path, names, nrec)
    return out


def pickup_diffs(dir_a, dir_b):
    """{file: {field: (max |a-b|, max |a-b| / max |b|, identical bytes?)}} for the three pickups at iteration 10."""
    out = {}
    for pre in PICKUPS:
        a, b = read_pickup(Path(dir_a) / f"{pre}.data"), read_pickup(Path(dir_b) / f"{pre}.data")
        assert a.keys() == b.keys(), (pre, a.keys(), b.keys())
        rows = {}
        for n in b:
            d = float(np.abs(a[n] - b[n]).max())
            s = float(np.abs(b[n]).max())
            rows[n] = (d, d / s if s > 0 else (0.0 if d == 0 else np.inf), a[n].tobytes() == b[n].tobytes())
        out[pre] = rows
    return out


def report(output, rundir, oracle, exp_dir):
    """Everything the check decides on, for a mitjax run (output.txt, rundir) against an oracle run directory."""
    ours, theirs = lines(output), lines(Path(oracle) / "output.txt")
    k0 = next(n for n, r in enumerate(theirs) if "Begin MONITOR dynamic field statistics" in r) - 1
    ma, mb = monitor_records(ours), monitor_records(theirs[k0:])
    sa, sb = solver_records(ours), solver_records(theirs[k0:])
    dg, summary = digits(output, Path(oracle) / "output.txt", exp_dir)
    return {"mon": (len(ma), len(mb), sum(x != y for x, y in zip(ma, mb)), first_difference(ma, mb)),
            "solver": (len(sa), len(sb), sum(x != y for x, y in zip(sa, sb)), first_difference(sa, sb)),
            "digits": dg, "summary": summary, "pickups": pickup_diffs(rundir, oracle)}


def main(work):
    """`python -m mitjax.tests.lstile_gate WORK` (compute node): mitjax's one-tile run and the report against the
    one-tile oracle, results/ and (for reference) the four-tile oracle."""
    import time
    t0 = time.time()
    run, warns, exp = run_mitjax(work)
    print(f"[{time.time() - t0:7.1f}s] one-tile run done: {run.output}", flush=True)
    print("warnings:", warns)
    exp_dir = exp.exp_dir
    print("mitjax-1 vs results/:", digits(run.output, exp_dir / "results" / "output.txt", exp_dir)[1])
    for label, rid in (("oracle-1", RETILE_RUN), ("oracle-4", FOUR_TILE_RUN)):
        od = oracle_dir(rid)
        if od is None:
            print(f"{label}: no oracle run {rid}")
            continue
        r = report(run.output, run.rundir, od, exp_dir)
        print(f"mitjax-1 vs {label} ({rid}): MON (ours, oracle, differing, first) {r['mon']}")
        print(f"mitjax-1 vs {label}: solver {r['solver']}")
        print(f"mitjax-1 vs {label}: {r['summary']}")
        print(f"mitjax-1 vs {label}: digits {r['digits']}")
        for f, rows in r["pickups"].items():
            for n, (d, rel, same) in rows.items():
                print(f"mitjax-1 vs {label}: {f} {n:10s} max|d| {d:.3e} rel {rel:.3e} bitwise {same}")
    od1, od4 = oracle_dir(RETILE_RUN), oracle_dir(FOUR_TILE_RUN)
    if od1 is not None:
        print("oracle-1 vs results/:", digits(od1 / "output.txt", exp_dir / "results" / "output.txt", exp_dir)[1])
        print("oracle-1 vs oracle-4 digits:", digits(od1 / "output.txt", od4 / "output.txt", exp_dir)[0])
        for f, rows in pickup_diffs(od1, od4).items():
            for n, (d, rel, same) in rows.items():
                print(f"oracle-1 vs oracle-4: {f} {n:10s} max|d| {d:.3e} rel {rel:.3e} bitwise {same}")
    print(f"[{time.time() - t0:7.1f}s] done", flush=True)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
