#!/usr/bin/env python3
"""Inputs and outputs of the GO replay harness (M1 lane GO; reference/replay_go/code/the_main_loop.F; a copy of the
GAD-A harness reference/replay_gad_a/replay_io.py adapted to GAD_DST2U1_IMPL_R and SOLVE_PENTADIAGONAL).

    replay_io.py write-inputs RUNDIR --size SIZE.h --experiment EXP [--seed N]   (run by jobs/replay.sbatch)

The harness is a genmake2 build of a verification experiment's code/ directory in which THE_MAIN_LOOP is replaced:
THE_MODEL_MAIN runs INITIALISE_FIXED, so GRID.h holds the experiment's real grid; the replacement reads the synthetic
inputs written here, calls GAD_DST2U1_IMPL_R (ENUM_UPWIND_1RST and ENUM_DST2, k = 1..Nr, into prior-initialised
matrices) and SOLVE_PENTADIAGONAL (errCode = -1 on entry) for every tile, and writes every output array (all points
incl. halos) and the grid as the routines read it.

Files (all big-endian stream; a Fortran array A(i,j,k,bi,bj) is the numpy C-order array [bj, bi, k, j, i], tile
t = (bj-1)*nSx + (bi-1)):
    replay_in.bin    int32 MAGIC, VERSION, sNx, sNy, OLx, OLy, nSx, nSy, Nr, len(IN3);
                     float64 deltaTarg(Nr); recip_deepFacC, recip_deepFac2C, recip_rhoFacC (Nr);
                     deepFac2F, recip_deepFac2F, rhoFacF, recip_rhoFacF (Nr+1); IN3 fields
    replay_out.bin   int32 MAGIC, VERSION, len(OUT3), Nr; OUT3 fields
    replay_grid.bin  int32 MAGIC, VERSION, 1, 1; recip_hFacC (3-D), recip_rA (2-D), recip_deepFac2C, recip_rhoFacC,
                     recip_drF (Nr), rkSign

Numpy only (no jax): it runs in the batch job before the model.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

MAGIC, VERSION = 20261003, 1

# the order of the in3(..., f) fields in the_main_loop.F
IN3 = ("rTrans", "aPrior", "bPrior", "cPrior", "a5d", "b5d", "c5d", "d5d", "e5d", "y5d")
# the order of the out3(..., n) fields
OUT3 = ("up1_a3d", "up1_b3d", "up1_c3d", "dst2_a3d", "dst2_b3d", "dst2_c3d", "penta_y", "penta_err")
OVERRIDE_NR = ("recip_deepFacC", "recip_deepFac2C", "recip_rhoFacC")
OVERRIDE_NR1 = ("deepFac2F", "recip_deepFac2F", "rhoFacF", "recip_rhoFacF")
GRID_NR = ("recip_deepFac2C", "recip_rhoFacC", "recip_drF")
SIZE_KEYS = ("sNx", "sNy", "OLx", "OLy", "nSx", "nSy", "Nr")
# per-experiment magnitudes (as replay_gad_a: rTrans ~ 0.5 rA drC / deltaT, |w_CFL| ~ 0..1.5)
SCALES = {
    # input/data:57 deltaTtracer= 86400.; dx ~ 3e5 m, delR 50..690
    "global_ocean.90x40x15": {"deltaTloc": 86400.0, "rTrans": 5.0e7},
    # input.nlfs/data:41 deltaT=1200.; dx = 10.E3, delR = 100
    "advect_xz": {"deltaTloc": 1200.0, "rTrans": 4.0e6},
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


def _tile_to_fortran(a, size):
    return a.reshape((size["nSy"], size["nSx"]) + a.shape[1:])


def _fortran_to_tile(a, size):
    return a.reshape((size["nSy"] * size["nSx"],) + a.shape[2:])


def _with_zeros(rng, a, frac_pos, frac_neg=0.0):
    a = a.copy()
    r = rng.uniform(size=a.shape)
    a[r < frac_pos] = 0.0
    a[(r >= frac_pos) & (r < frac_pos + frac_neg)] = -0.0
    return a


def make_inputs(size, experiment, seed=20261003):
    """Synthetic inputs at physical magnitudes, storage [tile, k, j, i] incl. halos. rTrans with exact +0 / -0
    (ABS and the upwind split); the penta-diagonal systems moderately diagonally dominant (c5d ~ 1 + |offdiag|),
    with c5d = 0 at level 1 on ~1 % of the points (the zero-pivot branch: d', e', y' := 0, errCode = 1)."""
    sc = SCALES[experiment]
    rng = np.random.default_rng(seed)
    T = size["nSx"] * size["nSy"]
    Nr = size["Nr"]
    Ny, Nx = size["sNy"] + 2 * size["OLy"], size["sNx"] + 2 * size["OLx"]
    shp = (T, Nr, Ny, Nx)
    n = lambda scale: scale * rng.standard_normal(shp)   # noqa: E731
    off = {k: _with_zeros(rng, n(0.3), 0.02, 0.01) for k in ("a5d", "b5d", "d5d", "e5d")}
    c5d = 1.0 + np.abs(off["a5d"]) + np.abs(off["b5d"]) + np.abs(off["d5d"]) + np.abs(off["e5d"]) \
        + 0.5 * rng.uniform(size=shp)
    zero = rng.uniform(size=shp[:1] + shp[2:]) < 0.01
    c5d[:, 0][zero] = 0.0
    f = {
        "rTrans": _with_zeros(rng, n(sc["rTrans"]), 0.04, 0.01),
        "aPrior": n(0.5) - 0.2, "bPrior": n(0.5) + 1.0, "cPrior": n(0.5) - 0.3,
        "a5d": off["a5d"], "b5d": off["b5d"], "c5d": c5d, "d5d": off["d5d"], "e5d": off["e5d"],
        "y5d": _with_zeros(rng, 10.0 + 5.0 * rng.standard_normal(shp), 0.02, 0.01),
    }
    assert tuple(f) == IN3
    one_d = {"deltaTarg": sc["deltaTloc"] * (1.0 + 0.01 * rng.uniform(0.5, 1.5, Nr))}
    for name in OVERRIDE_NR:
        one_d[name] = 1.0 + 0.01 * rng.uniform(0.5, 1.5, Nr)
    for name in OVERRIDE_NR1:
        one_d[name] = 1.0 + 0.01 * rng.uniform(0.5, 1.5, Nr + 1)
    return f, one_d


def write_inputs(rundir, size, experiment, seed=20261003):
    rundir = Path(rundir)
    f, one_d = make_inputs(size, experiment, seed)
    p_in = rundir / "replay_in.bin"
    if p_in.exists():
        raise SystemExit(f"{p_in} exists (nothing is overwritten)")
    hdr = [MAGIC, VERSION] + [size[k] for k in SIZE_KEYS] + [len(IN3)]
    with open(p_in, "xb") as fh:
        fh.write(np.array(hdr, ">i4").tobytes())
        fh.write(one_d["deltaTarg"].astype(">f8").tobytes())
        for name in OVERRIDE_NR + OVERRIDE_NR1:
            fh.write(one_d[name].astype(">f8").tobytes())
        for name in IN3:
            fh.write(_tile_to_fortran(f[name], size).astype(">f8").tobytes())
    meta = {"seed": seed, "experiment": experiment, "size": size, "IN3": IN3, "OUT3": OUT3,
            "sha256": {p_in.name: hashlib.sha256(p_in.read_bytes()).hexdigest()}}
    (rundir / "replay_in.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


class _Reader:
    def __init__(self, path):
        self.buf = Path(path).read_bytes()
        self.pos = 0

    def take(self, dtype, count):
        dt = np.dtype(dtype)
        a = np.frombuffer(self.buf, dt, count, self.pos)
        self.pos += dt.itemsize * count
        return a.astype(dt.newbyteorder("="))

    def done(self):
        if self.pos != len(self.buf):
            raise ValueError(f"{len(self.buf) - self.pos} trailing bytes")


def _dims(size):
    T = size["nSx"] * size["nSy"]
    return T, size["Nr"], size["sNy"] + 2 * size["OLy"], size["sNx"] + 2 * size["OLx"]


def read_inputs(rundir):
    """(size, fields {name: [tile,k,j,i]}, one_d {deltaTarg [k], overrides [k] or [k+1]})."""
    r = _Reader(Path(rundir) / "replay_in.bin")
    hdr = r.take(">i4", 10)
    if hdr[0] != MAGIC or hdr[1] != VERSION or hdr[9] != len(IN3):
        raise ValueError(f"bad replay_in.bin header {hdr}")
    size = dict(zip(SIZE_KEYS, (int(x) for x in hdr[2:9])))
    T, Nr, Ny, Nx = _dims(size)
    one_d = {"deltaTarg": r.take(">f8", Nr)}
    for name in OVERRIDE_NR:
        one_d[name] = r.take(">f8", Nr)
    for name in OVERRIDE_NR1:
        one_d[name] = r.take(">f8", Nr + 1)
    fields = {}
    for name in IN3:
        fields[name] = _fortran_to_tile(r.take(">f8", T * Nr * Ny * Nx).reshape(size["nSy"], size["nSx"], Nr, Ny,
                                                                                Nx), size)
    r.done()
    return size, fields, one_d


def read_outputs(rundir, size):
    r = _Reader(Path(rundir) / "replay_out.bin")
    hdr = r.take(">i4", 4)
    if hdr[0] != MAGIC or hdr[1] != VERSION or hdr[2] != len(OUT3) or hdr[3] != size["Nr"]:
        raise ValueError(f"bad replay_out.bin header {hdr}")
    T, Nr, Ny, Nx = _dims(size)
    out = {}
    for name in OUT3:
        out[name] = _fortran_to_tile(r.take(">f8", T * Nr * Ny * Nx).reshape(size["nSy"], size["nSx"], Nr, Ny, Nx),
                                     size)
    r.done()
    return out


def read_grid(rundir, size):
    """{recip_hFacC [tile,k,j,i], recip_rA [tile,j,i], recip_deepFac2C, recip_rhoFacC, recip_drF [k], rkSign}."""
    r = _Reader(Path(rundir) / "replay_grid.bin")
    hdr = r.take(">i4", 4)
    if hdr[0] != MAGIC or hdr[1] != VERSION or hdr[2] != 1 or hdr[3] != 1:
        raise ValueError(f"bad replay_grid.bin header {hdr}")
    T, Nr, Ny, Nx = _dims(size)
    g = {"recip_hFacC": _fortran_to_tile(r.take(">f8", T * Nr * Ny * Nx).reshape(size["nSy"], size["nSx"], Nr, Ny,
                                                                                 Nx), size),
         "recip_rA": _fortran_to_tile(r.take(">f8", T * Ny * Nx).reshape(size["nSy"], size["nSx"], Ny, Nx), size)}
    for name in GRID_NR:
        g[name] = r.take(">f8", Nr)
    g["rkSign"] = float(r.take(">f8", 1)[0])
    r.done()
    return g


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write-inputs")
    w.add_argument("rundir")
    w.add_argument("--size", required=True, help="the build's SIZE.h")
    w.add_argument("--experiment", required=True, choices=sorted(SCALES))
    w.add_argument("--seed", type=int, default=20261003)
    a = ap.parse_args(argv)
    meta = write_inputs(a.rundir, parse_size_h(a.size), a.experiment, a.seed)
    print(f"replay inputs written: {a.experiment} {json.dumps(meta['size'])} seed {meta['seed']} {meta['sha256']}")


if __name__ == "__main__":
    sys.exit(main())
