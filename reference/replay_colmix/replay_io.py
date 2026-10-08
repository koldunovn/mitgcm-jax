#!/usr/bin/env python3
"""Inputs and outputs of the COLMIX replay harness (M3 sub-lane COLMIX; reference/replay_colmix/code/the_main_loop.F,
a copy of lane COL's harness reference/replay_col with named records).

    replay_io.py write-inputs RUNDIR --size SIZE.h [--seed N]    (run by reference/replay_colmix/jobs/replay.sbatch)

The harness is a genmake2 build of vermix/code in which THE_MAIN_LOOP is replaced: THE_MODEL_MAIN runs
INITIALISE_FIXED (grid, parameters, INI_EOS, the packages' READPARMS); the replacement reads NSAMP synthetic samples
written here (state, the grid fields the routines read, priors of every output), overrides viscArNr and diffKrNrS,
runs the active package's INIT_VARIA and then, for each pass (1: the namelist eosType, MDJWF in vermix; 2: JMD95Z;
3 (OPPS only): JMD95Z with useGCMwVel), the package's routines on every sample, and writes every output (all points
incl. halos) plus the parameters and grid the routines read.

Files (big-endian stream of named records): CHARACTER*16 name, int32 rank, int32 n, float64 a(n). A Fortran array
A(i,j,k,bi,bj,n) is the numpy C-order array [n, bj, bi, k, j, i]; with one tile (vermix: nSx = nSy = 1) 3-D sample
fields are returned as [n, tile, k, j, i] and 2-D as [n, tile, j, i].
    cm_orig.bin  what INITIALISE_FIXED left, before any override: maskC/W/S, kLowC, kSurfC, viscArNr, diffKrNrS,
                 eosMDJWFnum, eosMDJWFden, eosRefP0 (the namelist eosType's EOS.h)
    cm_in.bin    IN3 (3-D x NSAMP), IN2 (2-D x NSAMP), OVERRIDES (Nr), in the order the Fortran reads them
    cm_out.bin   ini_vis, ini_dif, ini_hbl; per pass p: vis, dif, kapX0, kapXk, kapU, kapV, theta, salt, cnt, hbl
    cm_grid.bin  parameters and grid (names longer than 16 characters are truncated by the Fortran)

Synthetic columns (make_inputs): per column a bottom level kLowC in 0..Nr (land columns with kLowC = 0, kSurfC =
Nr+1), maskC = 1 on 1..kLowC, maskW/maskS the products with the western/southern neighbour; stratified theta/salt
profiles with a dense (cold, salty) surface layer of 1-6 levels in half of the columns (statically unstable: OPPS
plumes, negative Richardson numbers) and level noise (unstable interior layers), land levels set to 0 or to ocean
values; velocities with sheared profiles and runs of identical levels (zero shear: the epsilon floor of the
Richardson number), so that Ri spans the PP81 (RiLimit) and MY82 (RiMax) branches; downward wVel (pass 3) with exact
zeros; totPhiHyd anomalies; priors of every output at distinct values (points a routine does not write keep them).
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
NSAMP = 48                     # the_main_loop.F  PARAMETER ( nSamp = 48 )
IN3 = ("theta", "salt", "uVel", "vVel", "wVel", "totPhiHyd", "maskC", "maskW", "maskS", "viscAr0", "diffKr0",
       "kappaRx0", "kappaRU0", "kappaRV0", "viscAr1", "diffKr1")
IN2 = ("kLowC", "kSurfC", "MYhbl0")
OVERRIDES = ("viscArNr", "diffKrNrS")
OUT_PASS3 = ("vis", "dif", "kapX0", "kapXk", "kapU", "kapV", "theta", "salt", "cnt")
OUT_PASS2 = ("hbl",)


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
    """{name: array}: 3-D [n, tile, k, j, i], 2-D [n, tile, j, i], overrides [Nr] (see the module docstring)."""
    rng = np.random.default_rng(seed)
    T, Ny, Nx = _dims(size)
    Nr = size["Nr"]
    S = NSAMP
    s2, s3 = (S, T, Ny, Nx), (S, T, Nr, Ny, Nx)
    z = np.cumsum(np.full(Nr, 10.0)) - 5.0                             # nominal depths (m), only for profiles
    # bottom level per column
    r = rng.uniform(size=s2)
    kLow = rng.integers(4, Nr + 1, s2)
    kLow = np.where(r < 0.08, 0, np.where(r < 0.18, rng.integers(1, 4, s2), np.where(r < 0.55, Nr, kLow)))
    kSurf = np.where(kLow >= 1, 1, Nr + 1)
    lev = np.arange(1, Nr + 1)[None, None, :, None, None]
    maskC = (lev <= kLow[:, :, None]).astype(float)
    maskW = maskC * np.concatenate([maskC[..., :1], maskC[..., :-1]], axis=-1)
    maskS = maskC * np.concatenate([maskC[..., :1, :], maskC[..., :-1, :]], axis=-2)
    # stratified profiles + noise + dense surface layer in half of the columns
    T0 = rng.uniform(2.0, 25.0, s2)[:, :, None]
    dT = rng.uniform(1.0, 15.0, s2)[:, :, None]
    H = rng.uniform(50.0, 400.0, s2)[:, :, None]
    zz = z[None, None, :, None, None]
    theta = (T0 - dT * (1.0 - np.exp(-zz / H)))
    salt = rng.uniform(33.0, 36.0, s2)[:, :, None] + rng.uniform(0.0, 1.0, s2)[:, :, None] * (1.0 - np.exp(-zz / H))
    theta = theta + rng.normal(0.0, 0.05, s3)
    salt = salt + rng.normal(0.0, 0.01, s3)
    conv = rng.uniform(size=s2) < 0.5
    nlay = rng.integers(1, 7, s2)
    cool = rng.uniform(0.1, 4.0, s2)
    top = conv[:, :, None] & (lev <= nlay[:, :, None])
    theta = np.where(top, theta - cool[:, :, None], theta)
    salt = np.where(top, salt + 0.1 * cool[:, :, None], salt)
    # land levels: half 0, half ocean-like values
    zero_land = (maskC == 0.0) & (rng.uniform(size=s3) < 0.5)
    theta = np.where(zero_land, 0.0, theta)
    salt = np.where(zero_land, 0.0, salt)
    # velocities: sheared profiles, runs of identical levels (zero shear)
    U = rng.normal(0.0, 0.3, s2)[:, :, None]
    V = rng.normal(0.0, 0.3, s2)[:, :, None]
    L = rng.uniform(20.0, 300.0, s2)[:, :, None]
    uVel = U * np.exp(-zz / L) + rng.normal(0.0, 0.01, s3)
    vVel = V * np.exp(-zz / L) + rng.normal(0.0, 0.01, s3)
    same = rng.uniform(size=s3) < 0.15                                  # level k equal to level k-1
    for k in range(1, Nr):
        uVel[:, :, k] = np.where(same[:, :, k], uVel[:, :, k - 1], uVel[:, :, k])
        vVel[:, :, k] = np.where(same[:, :, k], vVel[:, :, k - 1], vVel[:, :, k])
    wVel = -rng.uniform(0.001, 0.05, s3)
    wVel[rng.uniform(size=s3) < 0.05] = 0.0
    f3 = {"theta": theta, "salt": salt, "uVel": uVel, "vVel": vVel, "wVel": wVel,
          "totPhiHyd": rng.normal(0.0, 5.0, s3), "maskC": maskC, "maskW": maskW, "maskS": maskS,
          "viscAr0": 3.3e-3 + rng.normal(0.0, 1e-4, s3), "diffKr0": 4.4e-4 + rng.normal(0.0, 1e-5, s3),
          "kappaRx0": 10.0 ** rng.uniform(-5.0, -2.0, s3), "kappaRU0": 10.0 ** rng.uniform(-5.0, -2.0, s3),
          "kappaRV0": 10.0 ** rng.uniform(-5.0, -2.0, s3),
          "viscAr1": 5.5e-3 + rng.normal(0.0, 1e-4, s3), "diffKr1": 6.6e-4 + rng.normal(0.0, 1e-5, s3)}
    f2 = {"kLowC": kLow.astype(float), "kSurfC": kSurf.astype(float), "MYhbl0": -50.0 + rng.normal(0.0, 10.0, s2)}
    one = {"viscArNr": 10.0 ** rng.uniform(-5.0, -3.0, Nr), "diffKrNrS": 10.0 ** rng.uniform(-6.0, -4.0, Nr)}
    assert tuple(f3) == IN3 and tuple(f2) == IN2 and tuple(one) == OVERRIDES
    return f3, f2, one


def _write_record(fh, name, a):
    a = np.ascontiguousarray(a, dtype=">f8").ravel()
    fh.write(name.ljust(16).encode("ascii"))
    fh.write(np.array([a.ndim, a.size], ">i4").tobytes())
    fh.write(a.tobytes())


def _to_fortran(a, size):
    """[n, tile, ...] -> [n, nSy, nSx, ...] (C order = Fortran (..., bi, bj, n))."""
    return a.reshape((a.shape[0], size["nSy"], size["nSx"]) + a.shape[2:])


def write_inputs(rundir, size, seed=20261002):
    if size["nSx"] * size["nSy"] != 1:
        raise SystemExit("the COLMIX harness copies tile 1 only (vermix: nSx = nSy = 1)")
    rundir = Path(rundir)
    f3, f2, one = make_inputs(size, seed)
    p_in = rundir / "cm_in.bin"
    if p_in.exists():
        raise SystemExit(f"{p_in} exists (nothing is overwritten)")
    with open(p_in, "xb") as fh:
        for name in IN3:
            _write_record(fh, name, _to_fortran(f3[name], size))
        for name in IN2:
            _write_record(fh, name, _to_fortran(f2[name], size))
        for name in OVERRIDES:
            _write_record(fh, name, one[name])
    meta = {"seed": seed, "size": size, "nsamp": NSAMP, "IN3": IN3, "IN2": IN2, "OVERRIDES": OVERRIDES,
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


def _shape(a, size, nk, samples):
    T, Ny, Nx = _dims(size)
    lead = (NSAMP,) if samples else ()
    mid = (nk,) if nk else ()
    return a.reshape(lead + (T,) + mid + (Ny, Nx))


def read_inputs(rundir, size):
    """{name: array} of cm_in.bin: IN3 [n, tile, k, j, i], IN2 [n, tile, j, i], OVERRIDES [Nr]."""
    recs = read_records(Path(rundir) / "cm_in.bin")
    out = {name: _shape(recs[name][1], size, size["Nr"], True) for name in IN3}
    out.update({name: _shape(recs[name][1], size, 0, True) for name in IN2})
    out.update({name: recs[name][1] for name in OVERRIDES})
    return out


def read_outputs(rundir, size):
    """{name: array}: ini_vis, ini_dif [tile, k, j, i], ini_hbl [tile, j, i]; '<field>_p<pass>' [n, tile, k, j, i]
    (hbl_p<pass> [n, tile, j, i])."""
    recs = read_records(Path(rundir) / "cm_out.bin")
    out = {}
    for name, (rank, a) in recs.items():
        if name in ("ini_vis", "ini_dif"):
            out[name] = _shape(a, size, size["Nr"], False)
        elif name == "ini_hbl":
            out[name] = _shape(a, size, 0, False)
        elif name.startswith("hbl_"):
            out[name] = _shape(a, size, 0, True)
        else:
            out[name] = _shape(a, size, size["Nr"], True)
    return out


def read_grid(rundir):
    """{name: float or 1-D array} of cm_grid.bin (rank-0 records as Python floats)."""
    return {name: (float(a[0]) if rank == 0 else a) for name, (rank, a) in
            read_records(Path(rundir) / "cm_grid.bin").items()}


def read_orig(rundir, size):
    """{name: array} of cm_orig.bin (what INITIALISE_FIXED left): maskC/W/S [tile, k, j, i], kLowC, kSurfC [tile,
    j, i], viscArNr, diffKrNrS [Nr], eosMDJWFnum [12], eosMDJWFden [13], eosRefP0 (float)."""
    out = {}
    for name, (rank, a) in read_records(Path(rundir) / "cm_orig.bin").items():
        if rank == 3:
            out[name] = _shape(a, size, size["Nr"], False)
        elif rank == 2:
            out[name] = _shape(a, size, 0, False)
        elif rank == 0:
            out[name] = float(a[0])
        else:
            out[name] = a
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write-inputs")
    w.add_argument("rundir")
    w.add_argument("--size", required=True, help="the build's SIZE.h")
    w.add_argument("--seed", type=int, default=20261002)
    a = ap.parse_args(argv)
    meta = write_inputs(a.rundir, parse_size_h(a.size), a.seed)
    print(f"COLMIX replay inputs written: {json.dumps(meta['size'])} nsamp {meta['nsamp']} seed {meta['seed']} "
          f"{meta['sha256']}")


if __name__ == "__main__":
    sys.exit(main())
