#!/usr/bin/env python3
"""Inputs and outputs of the COL replay harness (M1 sub-lane COL; reference/replay_col/code/the_main_loop.F, a copy of
the Task 8 harness reference/replay with named records).

    replay_io.py write-inputs RUNDIR --size SIZE.h [--seed N]     (run by reference/replay_col/jobs/replay.sbatch)

The harness is a genmake2 build of a verification experiment's code directory in which THE_MAIN_LOOP is replaced
(R's reference/adx_harness pattern [E§2]): THE_MODEL_MAIN runs INITIALISE_FIXED (grid, parameters, INI_EOS,
SET_REF_STATE), the replacement reads the synthetic inputs written here, overrides the deepFac*/rhoFac*, viscArNr and
dTtracerLev vectors with input values (trivial or uniform in the M1 experiments, so that a missing or misplaced factor
shows), calls the lane's column-physics routines for every tile and level as their callers do, and writes every
output (all points incl. halos; a point a routine does not write keeps the prior given here) plus the parameters and
grid the routines read.

Files (big-endian stream of named records): CHARACTER*16 name, int32 rank, int32 n, float64 a(n). A Fortran array
A(i,j,k,bi,bj) is the numpy C-order array [bj, bi, k, j, i]; the tile index is t = (bj-1)*nSx + (bi-1) (bi fastest),
so 3-D fields are returned as [tile, k, j, i] and 2-D fields as [tile, j, i].
    col_in.bin    IN3 (3-D), IN2 (2-D), OVERRIDES (1-D, Nr or Nr+1), in that order (the order the Fortran reads)
    col_out.bin   OUT records (the_main_loop.F 4a)
    col_grid.bin  GRID records (the_main_loop.F 4b)

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
IN3 = ("tFld", "sFld", "totPhi", "pLoc", "aTri", "bTri", "cTri", "yTri", "kappaRX", "gTr", "sigmaR", "uFld", "vFld",
       "wPrior", "rhoPrior", "ivdcPrior", "phiPrior")
IN2 = ("rStarDhDt", "hMixPrior")
# (name, levels: "r" = Nr, "rp1" = Nr+1)
OVERRIDES = (("deepFacC", "r"), ("deepFac2F", "rp1"), ("recip_deepFac2C", "r"), ("recip_deepFac2F", "rp1"),
             ("rhoFacC", "r"), ("rhoFacF", "rp1"), ("recip_rhoFacC", "r"), ("recip_rhoFacF", "rp1"),
             ("viscArNr", "r"), ("dTtracerLev", "r"))


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


def make_inputs(size, seed=20261001):
    """Synthetic but physical inputs ({name: array}): 3-D [tile, k, j, i], 2-D [tile, j, i], 1-D [Nr or Nr+1].

    theta -2..30 degC; salinity 30..38 with 3 % exact zeros (land-like) and 0.02 % negative values (the s <= 0
    branches of the JMD95 code and FIND_RHO_SCALAR's warning); totPhiHyd anomalies of a few m2/s2; in-situ pressure
    0..6e7 Pa; diagonally dominant tridiagonal systems with 1 % of b = 0 (the singular branch of SOLVE_TRIDIAGONAL);
    diffusivities 1e-5..10 m2/s (ivdc_kappa range incl.); sigmaR with half of the points statically unstable and
    10 % exact zeros; priors at distinct values. Overrides: deepFac*/rhoFac* = 1 + O(1e-2) (each one distinct, a
    reciprocal is not 1/x on purpose), viscArNr 1e-4..1e-2 and dTtracerLev 0.5..2 x 86400 s varying with k."""
    rng = np.random.default_rng(seed)
    T, Ny, Nx = _dims(size)
    Nr = size["Nr"]
    s3, s2 = (T, Nr, Ny, Nx), (T, Ny, Nx)
    u = lambda lo, hi, shp=s3: rng.uniform(lo, hi, shp)
    n = lambda scale, shp=s3: scale * rng.standard_normal(shp)
    salt = u(30.0, 38.0)
    salt[rng.uniform(size=s3) < 0.03] = 0.0
    neg = rng.uniform(size=s3) < 0.0002
    salt[neg] = rng.uniform(-1.0, 0.0, int(neg.sum()))
    aT, cT = -u(0.0, 1.0), -u(0.0, 1.0)
    bT = 1.0 - aT - cT + u(0.0, 0.1)
    zb = rng.uniform(size=s3) < 0.01
    bT[zb] = 0.0
    sig = n(1.0e-3)
    sig[rng.uniform(size=s3) < 0.1] = 0.0
    f = {"tFld": u(-2.0, 30.0), "sFld": salt, "totPhi": n(5.0), "pLoc": u(0.0, 6.0e7), "aTri": aT, "bTri": bT,
         "cTri": cT, "yTri": 15.0 + n(5.0), "kappaRX": 10.0 ** u(-5.0, 1.0), "gTr": u(-2.0, 30.0), "sigmaR": sig,
         "uFld": n(0.2), "vFld": n(0.2), "wPrior": n(1.0e-4), "rhoPrior": 5.0 + n(1.0),
         "ivdcPrior": 0.5 + n(0.01), "phiPrior": n(1.0)}
    f2 = {"rStarDhDt": n(1.0e-7, s2), "hMixPrior": -100.0 + n(10.0, s2)}
    one = {}
    for name, lev in OVERRIDES:
        m = Nr + (lev == "rp1")
        if name == "viscArNr":
            one[name] = 10.0 ** rng.uniform(-4.0, -2.0, m)
        elif name == "dTtracerLev":
            one[name] = 86400.0 * rng.uniform(0.5, 2.0, m)
        else:
            one[name] = 1.0 + 0.01 * rng.uniform(-1.0, 1.0, m)
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


def write_inputs(rundir, size, seed=20261001):
    rundir = Path(rundir)
    f, f2, one = make_inputs(size, seed)
    p_in = rundir / "col_in.bin"
    if p_in.exists():
        raise SystemExit(f"{p_in} exists (nothing is overwritten)")
    with open(p_in, "xb") as fh:
        for name in IN3:
            _write_record(fh, name, _to_fortran(f[name], size, 3))
        for name in IN2:
            _write_record(fh, name, _to_fortran(f2[name], size, 2))
        for name, _ in OVERRIDES:
            _write_record(fh, name, one[name])
    meta = {"seed": seed, "size": size, "IN3": IN3, "IN2": IN2, "OVERRIDES": [n for n, _ in OVERRIDES],
            "sha256": {p_in.name: hashlib.sha256(p_in.read_bytes()).hexdigest()}}
    (rundir / "col_in.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def read_records(path):
    """{name: float64 array (flat)} of a named-record file, in file order."""
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
    """Flat Fortran-order record -> [tile, k, j, i] (rank 3/4), [tile, j, i] (rank 2), 1-D, or scalar (rank 0)."""
    T, Ny, Nx = _dims(size)
    if rank in (3, 4):
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
    recs = read_records(Path(rundir) / "col_in.bin")
    out = {}
    for name in IN3:
        out[name] = _shape(recs[name][1], 3, size, name)
    for name in IN2:
        out[name] = _shape(recs[name][1], 2, size, name)
    for name, _ in OVERRIDES:
        out[name] = recs[name][1]
    return out


def read_outputs(rundir, size):
    return {name: _shape(a, rank, size, name) for name, (rank, a) in read_records(Path(rundir) / "col_out.bin").items()}


def read_grid(rundir, size):
    return {name: _shape(a, rank, size, name)
            for name, (rank, a) in read_records(Path(rundir) / "col_grid.bin").items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write-inputs")
    w.add_argument("rundir")
    w.add_argument("--size", required=True, help="the build's SIZE.h")
    w.add_argument("--seed", type=int, default=20261001)
    a = ap.parse_args(argv)
    meta = write_inputs(a.rundir, parse_size_h(a.size), a.seed)
    print(f"COL replay inputs written: {json.dumps(meta['size'])} seed {meta['seed']} {meta['sha256']}")


if __name__ == "__main__":
    sys.exit(main())
