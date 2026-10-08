#!/usr/bin/env python3
"""Inputs and outputs of the EOSAB replay harness (M3 lane EOSAB: FIND_ALPHA / FIND_BETA 'MDJWF';
reference/replay_eosab/code/the_main_loop.F, a copy of lane COLMIX's harness reference/replay_colmix).

    replay_io.py write-inputs RUNDIR --size SIZE.h [--seed N]    (run by reference/replay_eosab/jobs/replay.sbatch)

The harness is a genmake2 build of vermix/code in which THE_MAIN_LOOP is replaced: THE_MODEL_MAIN runs
INITIALISE_FIXED (grid, parameters, INI_EOS); the replacement reads NSAMP synthetic samples written here (theta, salt,
totPhiHyd and the priors of alphaLoc, betaLoc), and for each pass (1: kRef = k, full tile; 2: kRef = 1, full tile;
3: kRef = Nr+1-k on the index range 2-OLx..sNx+OLx-1, 1-OLy..sNy), each sample and each level k calls FIND_ALPHA and
FIND_BETA (the namelist eosType, MDJWF in vermix) and writes every output (all points incl. halos) plus the
parameters the routines read.

Files (big-endian stream of named records): CHARACTER*16 name, int32 rank, int32 n, float64 a(n). A Fortran array
A(i,j,k,bi,bj,n) is the numpy C-order array [n, bj, bi, k, j, i]; with one tile (vermix: nSx = nSy = 1) 3-D sample
fields are returned as [n, tile, k, j, i].
    cm_orig.bin  eosMDJWFnum, eosMDJWFden, eosRefP0 (EOS.h as INI_EOS left it for the namelist eosType)
    cm_in.bin    IN3 (3-D x NSAMP), in the order the Fortran reads them
    cm_out.bin   per pass p: alpha_p<p>, beta_p<p>
    cm_grid.bin  parameters (names longer than 16 characters are truncated by the Fortran)

Synthetic columns (make_inputs): per column a bottom level kLow in 0..Nr; below it (land) theta = salt = totPhiHyd = 0
in half of the points, ocean-like values in the other half. Stratified theta (-2..31 degC) and salt profiles with
noise; a quarter of the columns fresh (salt 0..5, river plume or melt water); salt <= 0 at isolated points (exact
+0, -0, small negative values: FIND_ALPHA/FIND_BETA's `IF ( s1 .GT. 0. _d 0 )` ELSE branch) and tiny positive
salinities (1e-14..1e-6: SQRT near 0); totPhiHyd anomalies N(0, 5^2) m^2/s^2 (with selectP_inEOS_Zc = 2,
PRESSURE_FOR_EOS adds phiRef); priors of both outputs at distinct values (points a pass does not write keep them).
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
NSAMP = 64                     # the_main_loop.F  PARAMETER ( nSamp = 64, nPass = 3, nIn3 = 5 )
NPASS = 3
IN3 = ("theta", "salt", "totPhiHyd", "alphaPrior", "betaPrior")
OUT3 = ("alpha", "beta")


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


def make_inputs(size, seed=20261003):
    """{name: array [n, tile, k, j, i]} (see the module docstring)."""
    rng = np.random.default_rng(seed)
    T, Ny, Nx = _dims(size)
    Nr = size["Nr"]
    S = NSAMP
    s2, s3 = (S, T, Ny, Nx), (S, T, Nr, Ny, Nx)
    z = np.cumsum(np.full(Nr, 10.0)) - 5.0                             # nominal depths (m), only for profiles
    r = rng.uniform(size=s2)
    kLow = rng.integers(4, Nr + 1, s2)
    kLow = np.where(r < 0.08, 0, np.where(r < 0.18, rng.integers(1, 4, s2), np.where(r < 0.55, Nr, kLow)))
    lev = np.arange(1, Nr + 1)[None, None, :, None, None]
    wet = lev <= kLow[:, :, None]
    zz = z[None, None, :, None, None]
    T0 = rng.uniform(-1.9, 31.0, s2)[:, :, None]
    dT = rng.uniform(0.0, 15.0, s2)[:, :, None]
    H = rng.uniform(50.0, 400.0, s2)[:, :, None]
    theta = np.maximum(T0 - dT * (1.0 - np.exp(-zz / H)), -2.0) + rng.normal(0.0, 0.05, s3)
    S0 = np.where(rng.uniform(size=s2) < 0.25, rng.uniform(0.0, 5.0, s2), rng.uniform(30.0, 41.0, s2))
    salt = S0[:, :, None] + rng.uniform(0.0, 1.0, s2)[:, :, None] * (1.0 - np.exp(-zz / H))
    salt = salt + rng.normal(0.0, 0.01, s3)
    u = rng.uniform(size=s3)                                            # isolated special salinities
    salt = np.where(u < 0.02, 0.0, salt)
    salt = np.where((u >= 0.02) & (u < 0.03), -0.0, salt)
    salt = np.where((u >= 0.03) & (u < 0.05), -rng.uniform(1e-6, 0.5, s3), salt)
    salt = np.where((u >= 0.05) & (u < 0.07), 10.0 ** rng.uniform(-14.0, -6.0, s3), salt)
    phi = rng.normal(0.0, 5.0, s3)
    zero_land = ~wet & (rng.uniform(size=s3) < 0.5)                    # land: half 0, half ocean-like values
    theta = np.where(zero_land, 0.0, theta)
    salt = np.where(zero_land, 0.0, salt)
    phi = np.where(zero_land, 0.0, phi)
    f3 = {"theta": theta, "salt": salt, "totPhiHyd": phi,
          "alphaPrior": 1000.0 + rng.uniform(0.0, 1.0, s3), "betaPrior": 2000.0 + rng.uniform(0.0, 1.0, s3)}
    assert tuple(f3) == IN3
    return f3


def _write_record(fh, name, a):
    a = np.ascontiguousarray(a, dtype=">f8").ravel()
    fh.write(name.ljust(16).encode("ascii"))
    fh.write(np.array([a.ndim, a.size], ">i4").tobytes())
    fh.write(a.tobytes())


def _to_fortran(a, size):
    """[n, tile, ...] -> [n, nSy, nSx, ...] (C order = Fortran (..., bi, bj, n))."""
    return a.reshape((a.shape[0], size["nSy"], size["nSx"]) + a.shape[2:])


def write_inputs(rundir, size, seed=20261003):
    if size["nSx"] * size["nSy"] != 1:
        raise SystemExit("the EOSAB harness copies tile 1 only (vermix: nSx = nSy = 1)")
    rundir = Path(rundir)
    f3 = make_inputs(size, seed)
    p_in = rundir / "cm_in.bin"
    if p_in.exists():
        raise SystemExit(f"{p_in} exists (nothing is overwritten)")
    with open(p_in, "xb") as fh:
        for name in IN3:
            _write_record(fh, name, _to_fortran(f3[name], size))
    meta = {"seed": seed, "size": size, "nsamp": NSAMP, "IN3": IN3,
            "sha256": {p_in.name: hashlib.sha256(p_in.read_bytes()).hexdigest()}}
    (rundir / "cm_in.json").write_text(json.dumps(meta, indent=1) + "\n")
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


def _shape(a, size):
    T, Ny, Nx = _dims(size)
    return a.reshape((NSAMP, T, size["Nr"], Ny, Nx))


def read_inputs(rundir, size):
    """{name: array [n, tile, k, j, i]} of cm_in.bin."""
    recs = read_records(Path(rundir) / "cm_in.bin")
    return {name: _shape(recs[name][1], size) for name in IN3}


def read_outputs(rundir, size):
    """{'<alpha|beta>_p<pass>': array [n, tile, k, j, i]}."""
    return {name: _shape(a, size) for name, (rank, a) in read_records(Path(rundir) / "cm_out.bin").items()}


def read_grid(rundir):
    """{name: float or 1-D array} of cm_grid.bin (rank-0 records as Python floats)."""
    return {name: (float(a[0]) if rank == 0 else a) for name, (rank, a) in
            read_records(Path(rundir) / "cm_grid.bin").items()}


def read_orig(rundir):
    """{name: array or float} of cm_orig.bin: eosMDJWFnum [12], eosMDJWFden [13], eosRefP0."""
    return {name: (float(a[0]) if rank == 0 else a) for name, (rank, a) in
            read_records(Path(rundir) / "cm_orig.bin").items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write-inputs")
    w.add_argument("rundir")
    w.add_argument("--size", required=True, help="the build's SIZE.h")
    w.add_argument("--seed", type=int, default=20261003)
    a = ap.parse_args(argv)
    meta = write_inputs(a.rundir, parse_size_h(a.size), a.seed)
    print(f"EOSAB replay inputs written: {json.dumps(meta['size'])} nsamp {meta['nsamp']} seed {meta['seed']} "
          f"{meta['sha256']}")


if __name__ == "__main__":
    sys.exit(main())
