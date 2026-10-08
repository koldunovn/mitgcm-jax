#!/usr/bin/env python3
"""Inputs and outputs of the RSTAR replay harness (M1 lane RSTAR; reference/replay_rstar/code/the_main_loop.F, a copy
of the Task 8 harness reference/replay with the COL lane's named records).

    replay_io.py write-inputs RUNDIR --size SIZE.h [--seed N]     (run by reference/replay_rstar/jobs/replay.sbatch)

The harness runs after INITIALISE_FIXED (real grid, hFac = h0Fac), writes the grid and parameters the r* routines read
(rstar_grid.bin), reads the inputs written here, sets the SURFACE.h r* fields from them and calls CALC_R_STAR (A),
UPDATE_R_STAR(.TRUE.), UPDATE_R_STAR(.FALSE.), CALC_R_STAR with the simple average (B: vectorInvariantMomentum,
selectKEscheme = 1 set in the harness) and RESET_NLFS_VARS, writing every output after each call (rstar_out.bin).
CALC_R_STAR's warnings (columns above hFacSup) go to errorMessageUnit = STDERR.0000 of the run directory.

Files (big-endian stream of named records): CHARACTER*16 name, int32 rank, int32 n, float64 a(n). A Fortran array
A(i,j,k,bi,bj) is the numpy C-order array [bj, bi, k, j, i]; tile t = (bj-1)*nSx + (bi-1) (bi fastest): 3-D fields
are returned as [tile, k, j, i], 2-D fields as [tile, j, i].
    rstar_in.bin    IN2 records, in that order (the order the Fortran reads)
    rstar_out.bin   etaA, etaB (the eta the calls get), A_*, T_*, F_*, B_* (the_main_loop.F 3./4.), R_pStarFacK
    rstar_grid.bin  grid, masks, h0Fac, the initial hFac / recip_hFac, params (hFacInf, hFacSup, deltaTFreeSurf,
                    selectKEscheme, vectorInvariantMomentum as 0/1)

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
IN2 = ("etaBase", "spikeA", "etaB", "spikeB", "rStarFacC", "rStarFacW", "rStarFacS", "prior")
N_SPIKES = 40


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
    """{name: [tile, j, i]}: etaBase, etaB uniform in (-1, 1) m on every point (halos included: the routine reads the
    halo eta at i = 0, sNx+1 and j = 0, sNy+1 without an exchange); spikeA / spikeB 1.5 / 1.6 at N_SPIKES random points
    with i in 1..sNx+1, j in 1..sNy+1 (the range of the counters, :183-184, including the halo column/row sNx+1,
    sNy+1 whose pre-exchange value the counters see), 0 elsewhere (the harness adds spike*(Ro_surf-R_low): the
    column rises above hFacSup = 2 where the point is wet); prior rStarFacC/W/S uniform in (0.98, 1.02); prior of
    the fields the routine overwrites uniform in (0.5, 1.5) (never 0)."""
    rng = np.random.default_rng(seed)
    T, Ny, Nx = _dims(size)
    s2 = (T, Ny, Nx)
    out = {"etaBase": rng.uniform(-1.0, 1.0, s2)}
    for name, val in (("spikeA", 1.5), ("spikeB", 1.6)):
        sp = np.zeros(s2)
        t = rng.integers(0, T, N_SPIKES)
        j = rng.integers(1, size["sNy"] + 2, N_SPIKES) - 1 + size["OLy"]
        i = rng.integers(1, size["sNx"] + 2, N_SPIKES) - 1 + size["OLx"]
        sp[t, j, i] = val
        out[name] = sp
        if name == "spikeA":
            out["etaB"] = rng.uniform(-1.0, 1.0, s2)
    for name in ("rStarFacC", "rStarFacW", "rStarFacS"):
        out[name] = rng.uniform(0.98, 1.02, s2)
    out["prior"] = rng.uniform(0.5, 1.5, s2)
    assert tuple(out) == IN2
    return out


def _to_fortran(a, size):
    """[tile, ...] -> [nSy, nSx, ...] (C order = Fortran (..., bi, bj))."""
    return a.reshape((size["nSy"], size["nSx"]) + a.shape[1:])


def _write_record(fh, name, a):
    a = np.ascontiguousarray(a, dtype=">f8").ravel()
    fh.write(name.ljust(16).encode("ascii"))
    fh.write(np.array([a.ndim, a.size], ">i4").tobytes())
    fh.write(a.tobytes())


def write_inputs(rundir, size, seed=20261001):
    rundir = Path(rundir)
    f = make_inputs(size, seed)
    p_in = rundir / "rstar_in.bin"
    if p_in.exists():
        raise SystemExit(f"{p_in} exists (nothing is overwritten)")
    with open(p_in, "xb") as fh:
        for name in IN2:
            _write_record(fh, name, _to_fortran(f[name], size))
    meta = {"seed": seed, "size": size, "IN2": IN2,
            "sha256": {p_in.name: hashlib.sha256(p_in.read_bytes()).hexdigest()}}
    (rundir / "rstar_in.json").write_text(json.dumps(meta, indent=1) + "\n")
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
    """Flat Fortran-order record -> [tile, k, j, i] (rank 3), [tile, j, i] (rank 2), or 1-D (rank 1)."""
    T, Ny, Nx = _dims(size)
    if rank == 3:
        nk = a.size // (T * Ny * Nx)
        if nk * T * Ny * Nx != a.size:
            raise ValueError(f"{name}: {a.size} values are not [tile, k, {Ny}, {Nx}]")
        return a.reshape(size["nSy"], size["nSx"], nk, Ny, Nx).reshape(T, nk, Ny, Nx)
    if rank == 2:
        return a.reshape(size["nSy"], size["nSx"], Ny, Nx).reshape(T, Ny, Nx)
    return a


def read_inputs(rundir, size):
    recs = read_records(Path(rundir) / "rstar_in.bin")
    return {name: _shape(recs[name][1], 2, size, name) for name in IN2}


def read_outputs(rundir, size):
    return {name: _shape(a, rank, size, name)
            for name, (rank, a) in read_records(Path(rundir) / "rstar_out.bin").items()}


def read_grid(rundir, size):
    return {name: _shape(a, rank, size, name)
            for name, (rank, a) in read_records(Path(rundir) / "rstar_grid.bin").items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write-inputs")
    w.add_argument("rundir")
    w.add_argument("--size", required=True, help="the build's SIZE.h")
    w.add_argument("--seed", type=int, default=20261001)
    a = ap.parse_args(argv)
    meta = write_inputs(a.rundir, parse_size_h(a.size), a.seed)
    print(f"RSTAR replay inputs written: {json.dumps(meta['size'])} seed {meta['seed']} {meta['sha256']}")


if __name__ == "__main__":
    sys.exit(main())
