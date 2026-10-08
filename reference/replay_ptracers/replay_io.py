#!/usr/bin/env python3
"""Inputs and outputs of the PTRACERS replay harness (M2 sub-lane PTRACERS; reference/replay_ptracers/code/
the_main_loop.F, a copy of the COL harness reference/replay_col with its named records).

    replay_io.py write-inputs RUNDIR --size SIZE.h [--seed N]     (run by reference/replay_ptracers/jobs/replay.sbatch)

The harness is a genmake2 build of tutorial_tracer_adjsens/code_ad in which THE_MAIN_LOOP is replaced: THE_MODEL_MAIN
runs INITIALISE_FIXED (grid, parameters, INI_CG2D, PTRACERS_INIT_FIXED), the replacement reads the synthetic inputs
written here, calls the lane's leaf routines for every tile as the model calls them (CONVECTIVE_ADJUSTMENT,
CONVECTIVE_ADJUSTMENT_INI, PTRACERS_APPLY_FORCING, the experiment's PTRACERS_FORCING_SURF, KPP_CALC_DUMMY,
COST_TRACER, CG2D_NSA twice, SWFRAC, GAD_DST3FL_ADV_R every level) and writes every output (all points incl. halos; a point a routine does not
write keeps the prior given here) plus the parameters and grid the routines read.

Files (big-endian stream of named records): CHARACTER*16 name (a longer Fortran name is cut to 16 characters), int32 rank, int32 n, float64 a(n). A Fortran array
A(i,j,k,bi,bj) is the numpy C-order array [bj, bi, k, j, i]; the tile index is t = (bj-1)*nSx + (bi-1) (bi fastest),
so 3-D fields are returned as [tile, k, j, i] and 2-D fields as [tile, j, i].
    pt_in.bin    IN3 (3-D), IN2 (2-D), then swdk (NSW), swFact (1), objfPrior (nSx*nSy), in that order
    pt_out.bin   OUT records (the_main_loop.F 3a)
    pt_grid.bin  GRID records (the_main_loop.F 3b)

Numpy only (no jax): it runs in the batch job before the model.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

SIZE_KEYS = ("sNx", "sNy", "OLx", "OLy", "nSx", "nSy", "Nr")
IN3 = ("tA", "sA", "pA", "tB", "sB", "pB", "gPrior", "ivdc", "dKr", "pTr", "wF", "rTr")
IN2 = ("cgB", "cgX0", "sfS", "sfP", "relaxS", "emp", "sfPrior", "kppPrior")
NSW = 64          # the_main_loop.F: PARAMETER ( nSw = 64 )


def parse_size_h(path):
    """sNx, sNy, OLx, OLy, nSx, nSy, Nr from a SIZE.h PARAMETER statement."""
    text = Path(path).read_text()
    out = {}
    for key in SIZE_KEYS:
        m = re.search(rf"^\s*&?\s*{key}\s*=\s*(\d+)\s*,?", text, re.M)
        if not m:
            raise ValueError(f"{key} not found in {path}")
        out[key] = int(m.group(1))
    return out


def _dims(size):
    T = size["nSx"] * size["nSy"]
    Ny, Nx = size["sNy"] + 2 * size["OLy"], size["sNx"] + 2 * size["OLx"]
    return T, Ny, Nx


def make_inputs(size, seed=20261002):
    """Synthetic inputs ({name: array}): 3-D [tile, k, j, i], 2-D [tile, j, i], 1-D.

    Convective adjustment (two independent sets A and B): theta 0..30 degC and salinity 33..37 drawn independently
    per level (the LINEAR EOS of tutorial_tracer_adjsens then makes about half of the interfaces statically unstable),
    with 5 % of the interfaces exactly neutral (theta and salt of level k equal those of level k-1: the strict
    `.LT. 0.` test), passive tracer 0..1; apply-forcing prior and surface forcing O(1e-3); IVDConvCount 0/1;
    diffKr 1e-5..1e-3; the passive tracer of PTRACERS_FORCING_SURF and COST_TRACER 0..2; surfaceForcingS,
    relaxForcingS, EmPmR O(1e-5); GAD_DST3FL_ADV_R wFld O(1e-3) m/s with 2 % exact zeros, rTrans O(1e5); cg2d
    right-hand side O(1) with zero-sum not enforced, first guess O(1); SWFRAC
    depths -300..+10 m with the exact values -200 (the `.LT. -200.` boundary), 0 and the model's rF levels absent
    (synthetic), swFact 1.0; objf_tracer prior distinct per tile."""
    rng = np.random.default_rng(seed)
    T, Ny, Nx = _dims(size)
    Nr = size["Nr"]
    s3, s2 = (T, Nr, Ny, Nx), (T, Ny, Nx)
    u = lambda lo, hi, shp=s3: rng.uniform(lo, hi, shp)
    n = lambda scale, shp=s3: scale * rng.standard_normal(shp)
    f = {}
    for tag in ("A", "B"):
        t, s = u(0.0, 30.0), u(33.0, 37.0)
        neutral = rng.uniform(size=s3) < 0.05
        neutral[:, 0] = False
        for k in range(1, Nr):
            t[:, k][neutral[:, k]] = t[:, k - 1][neutral[:, k]]
            s[:, k][neutral[:, k]] = s[:, k - 1][neutral[:, k]]
        f["t" + tag], f["s" + tag], f["p" + tag] = t, s, u(0.0, 1.0)
    f2 = {}
    f["gPrior"] = n(1.0e-3)
    f["ivdc"] = (rng.uniform(size=s3) < 0.3).astype(np.float64)
    f["dKr"] = 10.0 ** u(-5.0, -3.0)
    f["pTr"] = u(0.0, 2.0)
    f["wF"] = n(1.0e-3)              # GAD_DST3FL_ADV_R: |w|*dT*recip_drC from ~0 to beyond 1 on the real grid
    f["wF"][rng.uniform(size=s3) < 0.02] = 0.0
    f["rTr"] = n(1.0e5)
    f2["cgB"] = n(1.0, s2)
    f2["cgX0"] = n(1.0, s2)
    f2["sfS"] = n(1.0e-5, s2)
    f2["sfP"] = n(1.0e-3, s2)
    f2["relaxS"] = n(1.0e-5, s2)
    f2["emp"] = n(1.0e-5, s2)
    f2["sfPrior"] = 7.0 + n(1.0, s2)
    f2["kppPrior"] = 3.0 + n(1.0, s2)
    sw = rng.uniform(-300.0, 10.0, NSW)
    sw[0], sw[1], sw[2] = -200.0, 0.0, -200.0 + 1e-9
    one = {"swdk": sw, "swFact": np.array([1.0]), "objfPrior": 100.0 + np.arange(T, dtype=np.float64)}
    assert tuple(f) == IN3 and tuple(f2) == IN2
    return f, f2, one


def _to_fortran(a, size, rank):
    """[tile, ...] -> [nSy, nSx, ...] (C order = Fortran (..., bi, bj))."""
    if rank == 1:
        return a
    return a.reshape((size["nSy"], size["nSx"]) + a.shape[1:])


def _write_record(fh, name, a):
    a = np.ascontiguousarray(a, dtype=">f8").ravel()
    fh.write(name.ljust(16).encode("ascii"))
    fh.write(np.array([a.ndim, a.size], ">i4").tobytes())
    fh.write(a.tobytes())


def write_inputs(rundir, size, seed=20261002):
    rundir = Path(rundir)
    f, f2, one = make_inputs(size, seed)
    p_in = rundir / "pt_in.bin"
    if p_in.exists():
        raise SystemExit(f"{p_in} exists (nothing is overwritten)")
    with open(p_in, "xb") as fh:
        for name in IN3:
            _write_record(fh, name, _to_fortran(f[name], size, 3))
        for name in IN2:
            _write_record(fh, name, _to_fortran(f2[name], size, 2))
        for name in ("swdk", "swFact", "objfPrior"):
            _write_record(fh, name, one[name])
    meta = {"seed": seed, "size": size, "IN3": IN3, "IN2": IN2,
            "sha256": {p_in.name: hashlib.sha256(p_in.read_bytes()).hexdigest()}}
    (rundir / "pt_in.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def read_records(path):
    """{name: (rank, float64 array (flat))} of a named-record file, in file order."""
    buf = Path(path).read_bytes()
    pos, out = 0, {}
    while pos < len(buf):
        name = buf[pos:pos + 16].decode("ascii").strip()
        rank, n = np.frombuffer(buf, ">i4", 2, pos + 16)
        pos += 24
        a = np.frombuffer(buf, ">f8", int(n), pos).astype(np.float64)
        pos += 8 * int(n)
        if name in out:
            raise ValueError(f"{path}: record {name} twice")
        out[name] = (int(rank), a)
    return out


def _shape(a, rank, size, name):
    """Flat Fortran-order record -> [tile, k, j, i] (rank 3), [tile, j, i] (rank 2), 1-D, or scalar (rank 0, n=1)."""
    T, Ny, Nx = _dims(size)
    if rank == 3:
        nk = a.size // (T * Ny * Nx)
        if nk * T * Ny * Nx != a.size:
            raise ValueError(f"{name}: {a.size} values are not [tile, k, {Ny}, {Nx}]")
        return a.reshape(size["nSy"], size["nSx"], nk, Ny, Nx).reshape(T, nk, Ny, Nx)
    if rank == 2:
        return a.reshape(size["nSy"], size["nSx"], Ny, Nx).reshape(T, Ny, Nx)
    if rank == 0:
        return float(a[0]) if a.size == 1 else a
    return a


def read_inputs(rundir, size):
    recs = read_records(Path(rundir) / "pt_in.bin")
    out = {}
    for name in IN3:
        out[name] = _shape(recs[name][1], 3, size, name)
    for name in IN2:
        out[name] = _shape(recs[name][1], 2, size, name)
    for name in ("swdk", "swFact", "objfPrior"):
        out[name] = recs[name][1]
    return out


def read_outputs(rundir, size):
    return {name: _shape(a, rank, size, name) for name, (rank, a) in read_records(Path(rundir) / "pt_out.bin").items()}


def read_grid(rundir, size):
    return {name: _shape(a, rank, size, name)
            for name, (rank, a) in read_records(Path(rundir) / "pt_grid.bin").items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write-inputs")
    w.add_argument("rundir")
    w.add_argument("--size", required=True, help="the build's SIZE.h")
    w.add_argument("--seed", type=int, default=20261002)
    a = ap.parse_args(argv)
    meta = write_inputs(a.rundir, parse_size_h(a.size), a.seed)
    print(f"PTRACERS replay inputs written: {json.dumps(meta['size'])} seed {meta['seed']} {meta['sha256']}")


if __name__ == "__main__":
    sys.exit(main())
