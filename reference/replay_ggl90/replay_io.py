#!/usr/bin/env python3
"""Inputs and outputs of the GGL90 replay harness (M3 sub-lane GGL90; reference/replay_ggl90/code/the_main_loop.F, a
copy of the COL harness reference/replay_col with named records).

    replay_io.py write-inputs RUNDIR --size SIZE.h --exp EXPERIMENT [--seed N]   (run by jobs/replay.sbatch)

The harness is a genmake2 build of a verification experiment's code directory in which THE_MAIN_LOOP is replaced:
THE_MODEL_MAIN runs INITIALISE_FIXED (grid, parameters, GGL90_READPARMS from the run's data.ggl90); the replacement
writes the GGL90.h parameters and the GGL90_INIT_VARIA fields (ggl_parm.bin), reads the synthetic inputs written
here, runs GGL90_CALC for every case of the case table (GGL90.h flags set per case), GGL90_EXCHANGES, then
GGL90_CALC_DIFF, GGL90_CALC_VISC and (Langmuir builds) GGL90_ADD_STOKESDRIFT on priors, and writes every output (all
points incl. halos; ggl_out.bin) and the grid and parameters the routines read (ggl_grid.bin).

Files (big-endian stream of named records): CHARACTER*16 name, int32 rank, int32 n, float64 a(n). A Fortran array
A(i,j,k,bi,bj) is the numpy C-order array [bj, bi, k, j, i]; tile t = (bj-1)*nSx + (bi-1) (bi fastest); 3-D fields
are returned as [tile, k, j, i], 2-D fields as [tile, j, i].

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
IN3 = ("TKE", "IDEMIX_E", "uVel", "vVel", "sigmaR", "viscArUPrior", "viscArVPrior", "diffKrPrior", "kappaRx",
       "kappaRU", "kappaRV")
IN2 = ("sfU", "sfV")
CASE_KEYS = ("mxlMaxFlag", "useLANGMUIR", "calcMeanVertShear", "GGL90_dirichlet", "mxlSurfFlag", "useIDEMIX")
# the cases per experiment build (code options decide what can run: LANGMUIR only in vermix/code, IDEMIX only in
# global_ocean.90x40x15/code); case 1 is the run's own setting (data.ggl90 of input.ggl90 / input.idemix)
CASES = {
    "vermix": ((3, 0, 0, 1, 0, 0),      # input.ggl90
               (2, 1, 0, 1, 0, 0),      # input.gglLC
               (1, 1, 1, 0, 1, 0),
               (0, 0, 1, 0, 0, 0),
               (3, 1, 0, 0, 0, 0),
               (1, 0, 0, 1, 1, 0)),
    "global_ocean.90x40x15": ((2, 0, 0, 0, 0, 1),     # input.idemix
                              (3, 0, 1, 1, 1, 1),
                              (0, 0, 0, 1, 0, 0),
                              (1, 0, 1, 0, 0, 0)),
}


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
    """Synthetic but physical inputs ({name: array}): 3-D [tile, k, j, i], 2-D [tile, j, i].

    TKE: log-uniform 1e-8..1e-2 m2/s2 with 10 % exact zeros and 10 % at 1e-7 (vermix's GGL90TKEmin), i.e. TKE near
    its minimum; IDEMIX_E log-uniform 1e-7..1e-3 with 5 % zeros; uVel, vVel 0.1 m/s noise with every fourth level
    strongly sheared (1 m/s); sigmaR: 70 % stable (-1e-6..-1e-2 kg/m4 log-uniform), 20 % unstable, 10 % exact zeros;
    surface stress / rho 1e-4 m2/s2 noise with 5 % exact zeros; priors of the output arrays at distinct values."""
    rng = np.random.default_rng(seed)
    T, Ny, Nx = _dims(size)
    Nr = size["Nr"]
    s3, s2 = (T, Nr, Ny, Nx), (T, Ny, Nx)
    u = lambda lo, hi, shp=s3: rng.uniform(lo, hi, shp)
    n = lambda scale, shp=s3: scale * rng.standard_normal(shp)
    tke = 10.0 ** u(-8.0, -2.0)
    r = rng.uniform(size=s3)
    tke[r < 0.1] = 0.0
    tke[(r >= 0.1) & (r < 0.2)] = 1.0e-7
    E = 10.0 ** u(-7.0, -3.0)
    E[rng.uniform(size=s3) < 0.05] = 0.0
    uv, vv = n(0.1), n(0.1)
    uv[:, ::4] *= 10.0
    vv[:, 1::4] *= 10.0
    sig = -(10.0 ** u(-6.0, -2.0))
    r = rng.uniform(size=s3)
    sig[r < 0.2] = -sig[r < 0.2]
    sig[(r >= 0.2) & (r < 0.3)] = 0.0
    sfU, sfV = n(1.0e-4, s2), n(1.0e-4, s2)
    sfU[rng.uniform(size=s2) < 0.05] = 0.0
    sfV[rng.uniform(size=s2) < 0.05] = 0.0
    f = {"TKE": tke, "IDEMIX_E": E, "uVel": uv, "vVel": vv, "sigmaR": sig,
         "viscArUPrior": -1.0 + n(0.01), "viscArVPrior": -2.0 + n(0.01), "diffKrPrior": -3.0 + n(0.01),
         "kappaRx": 10.0 ** u(-6.0, -3.0), "kappaRU": 10.0 ** u(-5.0, -2.0), "kappaRV": 10.0 ** u(-5.0, -2.0)}
    f2 = {"sfU": sfU, "sfV": sfV}
    assert tuple(f) == IN3 and tuple(f2) == IN2
    return f, f2


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


def write_inputs(rundir, size, exp, seed=20261002):
    rundir = Path(rundir)
    f, f2 = make_inputs(size, seed)
    cases = CASES[exp]
    p_in = rundir / "ggl_in.bin"
    if p_in.exists():
        raise SystemExit(f"{p_in} exists (nothing is overwritten)")
    with open(p_in, "xb") as fh:
        for name in IN3:
            _write_record(fh, name, _to_fortran(f[name], size, 3))
        for name in IN2:
            _write_record(fh, name, _to_fortran(f2[name], size, 2))
        _write_record(fh, "nCase", np.array([float(len(cases))]))
        _write_record(fh, "cases", np.array(cases, dtype=np.float64).ravel())
    meta = {"seed": seed, "size": size, "exp": exp, "IN3": IN3, "IN2": IN2, "cases": cases,
            "sha256": {p_in.name: hashlib.sha256(p_in.read_bytes()).hexdigest()}}
    (rundir / "ggl_in.json").write_text(json.dumps(meta, indent=1) + "\n")
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
    recs = read_records(Path(rundir) / "ggl_in.bin")
    out = {name: _shape(recs[name][1], 3, size, name) for name in IN3}
    out.update({name: _shape(recs[name][1], 2, size, name) for name in IN2})
    nc = int(recs["nCase"][1][0])
    out["cases"] = [dict(zip(CASE_KEYS, (int(v) for v in row)))
                    for row in recs["cases"][1].reshape(nc, len(CASE_KEYS))]
    return out


def read_file(rundir, fname, size):
    return {name: _shape(a, rank, size, name)
            for name, (rank, a) in read_records(Path(rundir) / fname).items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write-inputs")
    w.add_argument("rundir")
    w.add_argument("--size", required=True, help="the build's SIZE.h")
    w.add_argument("--exp", required=True, choices=sorted(CASES))
    w.add_argument("--seed", type=int, default=20261002)
    a = ap.parse_args(argv)
    meta = write_inputs(a.rundir, parse_size_h(a.size), a.exp, a.seed)
    print(f"GGL90 replay inputs written: {json.dumps(meta['size'])} {len(meta['cases'])} cases seed {meta['seed']} "
          f"{meta['sha256']}")


if __name__ == "__main__":
    sys.exit(main())
