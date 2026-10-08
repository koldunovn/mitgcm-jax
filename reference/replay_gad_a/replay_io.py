#!/usr/bin/env python3
"""Inputs and outputs of the GAD-A replay harness (M1 sub-lane GAD-A; reference/replay_gad_a/code/the_main_loop.F; a
copy of the Task 8 harness reference/replay/replay_io.py adapted to the simple flux routines of pkg/generic_advdiff).

    replay_io.py write-inputs RUNDIR --size SIZE.h --experiment EXP [--seed N]   (run by jobs/replay.sbatch)

The harness is a genmake2 build of a verification experiment's code/ directory in which THE_MAIN_LOOP is replaced
(R's reference/adx_harness pattern [E§2]): THE_MODEL_MAIN runs INITIALISE_FIXED, so GRID.h holds the experiment's
real grid (land, partial cells, exchanged halos); the replacement reads the synthetic inputs written here, calls the
GAD-A routines for every tile and level (each output first set to a prior given here: points a routine does not
write keep it) and writes every output array (all points incl. halos) and the grid as the routines read it.

Files (all big-endian stream; Fortran column-major, so a Fortran array A(i,j,k,bi,bj) is the numpy C-order array
[bj, bi, k, j, i], and the tile index is t = (bj-1)*nSx + (bi-1), bi fastest):
    replay_in.bin    int32 MAGIC, VERSION, sNx, sNy, OLx, OLy, nSx, nSy, Nr, len(IN3);
                     float64 deltaTloc, diffKh; deltaTarg(Nr); recip_deepFacC, recip_deepFac2C, recip_rhoFacC (Nr);
                     deepFac2F, recip_deepFac2F, rhoFacF, recip_rhoFacF (Nr+1); cosFacU, cosFacV (1-OLy:sNy+OLy,
                     nSx, nSy); IN3 fields (1-OLx:sNx+OLx, 1-OLy:sNy+OLy, Nr, nSx, nSy)
    replay_out.bin   int32 MAGIC, VERSION, len(OUT3), Nr; OUT3 fields
    replay_grid.bin  int32 MAGIC, VERSION, 5, 4; GRID3, GRID2, cosFacU, cosFacV, GRID_NR, GRID_NR1, rkSign

Numpy only (no jax): it runs in the batch job before the model.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

MAGIC, VERSION = 20261002, 1

# the order of the in3(..., f) fields in the_main_loop.F
IN3 = ("uTrans", "vTrans", "rTrans", "uVel", "vVel", "uCFL", "vCFL", "tracer", "KappaR", "xA", "yA", "fluxPrior",
       "aPrior", "bPrior", "cPrior", "dPrior", "ePrior")
# the order of the out3(..., n) fields: n = 1..20 one call per level (prior fluxPrior), 21..28 the implicit matrices
OUT3 = ("c2_x", "c2_y", "c2_r", "c4_x", "c4_y", "u3_x", "u3_y",
        "dst3fl_x_calcCFL", "dst3fl_x_givenCFL", "dst3fl_y_calcCFL", "dst3fl_y_givenCFL",
        "dst3_x_calcCFL", "dst3_x_givenCFL",
        "fluxlimit_x_calcCFL", "fluxlimit_x_givenCFL", "fluxlimit_y_calcCFL", "fluxlimit_y_givenCFL",
        "diff_x", "diff_y", "diff_r",
        "u3c4_a5d", "u3c4_b5d", "u3c4_c5d", "u3c4_d5d", "u3c4_e5d", "fluxlimit_a3d", "fluxlimit_b3d", "fluxlimit_c3d")
GRID3 = ("hFacC", "recip_hFacC", "maskC", "maskW", "maskS")
GRID2 = ("recip_dxC", "recip_dyC", "rA", "recip_rA")
GRID_NR = ("recip_deepFacC", "recip_deepFac2C", "recip_rhoFacC", "recip_drF")
GRID_NR1 = ("deepFac2F", "recip_deepFac2F", "rhoFacF", "recip_rhoFacF", "recip_drC")
OVERRIDE_NR = ("recip_deepFacC", "recip_deepFac2C", "recip_rhoFacC")
OVERRIDE_NR1 = ("deepFac2F", "recip_deepFac2F", "rhoFacF", "recip_rhoFacF")
SIZE_KEYS = ("sNx", "sNy", "OLx", "OLy", "nSx", "nSy", "Nr")

# Per-experiment magnitudes of the synthetic inputs, so that CFL numbers and the limiters' slope ratios span their
# physical range on each grid: deltaTloc is the experiment's tracer time step; uVel ~ 0.6 dx / deltaT (|CFL| ~ 0..2);
# rTrans ~ 0.5 rA drC / deltaT (|w_CFL| ~ 0..1.5). dx, drC from the grid parameters cited.
SCALES = {
    # input/data:57 deltaTtracer= 86400.; :82-83 dxSpacing = dySpacing = 4. (deg, ~3e5 m at mid latitude); delR 50..690
    "global_ocean.90x40x15": {"deltaTloc": 86400.0, "uVel": 2.0, "rTrans": 5.0e7},
    # input.nlfs/data:41 deltaT=1200.; :54-55 dXspacing = dYspacing = 10.E3; :53 delR=20*100.
    "advect_xz": {"deltaTloc": 1200.0, "uVel": 5.0, "rTrans": 4.0e6},
    # input/data:28 deltaT=2500.0; :42-43 dXspacing = dYspacing = 10.E3 (Nr = 1)
    "advect_xy": {"deltaTloc": 2500.0, "uVel": 2.4, "rTrans": 2.0e6},
    # input/data:35 deltaT=1200.; :50-51 delX = delY = 1 deg (~1e5 m); :54 delR 50..190
    "tutorial_baroclinic_gyre": {"deltaTloc": 1200.0, "uVel": 50.0, "rTrans": 4.0e8},
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
    """[tile, ...] -> [nSy, nSx, ...] (C order = Fortran (..., bi, bj))."""
    return a.reshape((size["nSy"], size["nSx"]) + a.shape[1:])


def _fortran_to_tile(a, size):
    return a.reshape((size["nSy"] * size["nSx"],) + a.shape[2:])


def _with_zeros(rng, a, frac_pos, frac_neg=0.0):
    """Exact +0.0 at a fraction of points and -0.0 at another (sign-of-zero paths: SIGN, ABS, the limiters)."""
    a = a.copy()
    r = rng.uniform(size=a.shape)
    a[r < frac_pos] = 0.0
    a[(r >= frac_pos) & (r < frac_pos + frac_neg)] = -0.0
    return a


def make_tracer(rng, shp):
    """10 + 5 N(0,1) with plateaus (a point equal to its i-1, j-1 or k-1 neighbour: zero differences Rj, Rjm, Rjp
    with an open mask, so the thetaMax / CrMax branches and the limiters' kinks are hit), exact +0 and -0 values."""
    t = 10.0 + 5.0 * rng.standard_normal(shp)
    for axis in (3, 2, 1):                                   # i, j, k (storage [tile, k, j, i])
        if shp[axis] < 2:
            continue
        sel = rng.uniform(size=shp) < 0.08
        lo = [slice(None)] * 4
        hi = [slice(None)] * 4
        lo[axis], hi[axis] = slice(0, -1), slice(1, None)
        dst = t[tuple(hi)]
        dst[sel[tuple(hi)]] = t[tuple(lo)][sel[tuple(hi)]]
        t[tuple(hi)] = dst
    return _with_zeros(rng, t, 0.03, 0.01)


def make_inputs(size, experiment, seed=20261002):
    """Synthetic inputs: random binary64 values at physical magnitudes (every rounding counts), storage
    [tile, k, j, i] incl. halos; per-experiment time step and velocity scales (SCALES). The vertical/metric factors
    that are 1 in the experiments (recip_deepFacC, recip_deepFac2C, recip_rhoFacC, deepFac2F, recip_deepFac2F,
    rhoFacF, recip_rhoFacF, cosFacU, cosFacV) get synthetic values != 1 (a missing or swapped factor must show)."""
    sc = SCALES[experiment]
    rng = np.random.default_rng(seed)
    T = size["nSx"] * size["nSy"]
    Nr = size["Nr"]
    Ny, Nx = size["sNy"] + 2 * size["OLy"], size["sNx"] + 2 * size["OLx"]
    shp = (T, Nr, Ny, Nx)
    n = lambda scale: scale * rng.standard_normal(shp)
    u = lambda lo, hi: rng.uniform(lo, hi, shp)
    f = {
        "uTrans": _with_zeros(rng, n(2.0e6), 0.04, 0.01),
        "vTrans": _with_zeros(rng, n(2.0e6), 0.04, 0.01),
        "rTrans": _with_zeros(rng, n(sc["rTrans"]), 0.04, 0.01),
        "uVel": _with_zeros(rng, n(sc["uVel"]), 0.02, 0.01),
        "vVel": _with_zeros(rng, n(sc["uVel"]), 0.02, 0.01),
        "uCFL": _with_zeros(rng, u(-0.9, 0.9), 0.02),
        "vCFL": _with_zeros(rng, u(-0.9, 0.9), 0.02),
        "tracer": make_tracer(rng, shp),
        "KappaR": u(1.0e-5, 1.0e-3),
        "xA": _with_zeros(rng, u(1.0e5, 5.0e6), 0.1),
        "yA": _with_zeros(rng, u(1.0e5, 5.0e6), 0.1),
        "fluxPrior": n(1.0e6) - 3.0e6,
        "aPrior": n(0.5) - 0.1, "bPrior": n(0.5) - 0.2, "cPrior": n(0.5) + 1.0, "dPrior": n(0.5) - 0.3,
        "ePrior": n(0.5) - 0.05,
    }
    assert tuple(f) == IN3
    scal = {"deltaTloc": sc["deltaTloc"], "diffKh": 1.0e3 * (1.0 + 0.1 * rng.uniform())}
    one_d = {"deltaTarg": sc["deltaTloc"] * (1.0 + 0.01 * rng.uniform(0.5, 1.5, Nr))}
    for name in OVERRIDE_NR:
        one_d[name] = 1.0 + 0.01 * rng.uniform(0.5, 1.5, Nr)
    for name in OVERRIDE_NR1:
        one_d[name] = 1.0 + 0.01 * rng.uniform(0.5, 1.5, Nr + 1)
    one_d["cosFacU"] = rng.uniform(0.2, 1.0, (T, Ny))
    one_d["cosFacV"] = rng.uniform(0.2, 1.0, (T, Ny))
    return f, scal, one_d


def write_inputs(rundir, size, experiment, seed=20261002):
    rundir = Path(rundir)
    f, scal, one_d = make_inputs(size, experiment, seed)
    p_in = rundir / "replay_in.bin"
    if p_in.exists():
        raise SystemExit(f"{p_in} exists (nothing is overwritten)")
    hdr = [MAGIC, VERSION] + [size[k] for k in SIZE_KEYS] + [len(IN3)]
    with open(p_in, "xb") as fh:
        fh.write(np.array(hdr, ">i4").tobytes())
        fh.write(np.array([scal["deltaTloc"], scal["diffKh"]], ">f8").tobytes())
        fh.write(one_d["deltaTarg"].astype(">f8").tobytes())
        for name in OVERRIDE_NR + OVERRIDE_NR1:
            fh.write(one_d[name].astype(">f8").tobytes())
        for name in ("cosFacU", "cosFacV"):
            fh.write(_tile_to_fortran(one_d[name], size).astype(">f8").tobytes())
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
    """(size, fields {name: [tile,k,j,i]}, scalars {deltaTloc, diffKh}, one_d {deltaTarg [k], overrides [k] or [k+1],
    cosFacU/cosFacV [tile,j]})."""
    r = _Reader(Path(rundir) / "replay_in.bin")
    hdr = r.take(">i4", 10)
    if hdr[0] != MAGIC or hdr[1] != VERSION or hdr[9] != len(IN3):
        raise ValueError(f"bad replay_in.bin header {hdr}")
    size = dict(zip(SIZE_KEYS, (int(x) for x in hdr[2:9])))
    T, Nr, Ny, Nx = _dims(size)
    s = r.take(">f8", 2)
    scal = {"deltaTloc": float(s[0]), "diffKh": float(s[1])}
    one_d = {"deltaTarg": r.take(">f8", Nr)}
    for name in OVERRIDE_NR:
        one_d[name] = r.take(">f8", Nr)
    for name in OVERRIDE_NR1:
        one_d[name] = r.take(">f8", Nr + 1)
    for name in ("cosFacU", "cosFacV"):
        one_d[name] = _fortran_to_tile(r.take(">f8", T * Ny).reshape(size["nSy"], size["nSx"], Ny), size)
    fields = {}
    for name in IN3:
        a = r.take(">f8", T * Nr * Ny * Nx).reshape(size["nSy"], size["nSx"], Nr, Ny, Nx)
        fields[name] = _fortran_to_tile(a, size)
    r.done()
    return size, fields, scal, one_d


def read_outputs(rundir, size):
    r = _Reader(Path(rundir) / "replay_out.bin")
    hdr = r.take(">i4", 4)
    if hdr[0] != MAGIC or hdr[1] != VERSION or hdr[2] != len(OUT3) or hdr[3] != size["Nr"]:
        raise ValueError(f"bad replay_out.bin header {hdr}")
    T, Nr, Ny, Nx = _dims(size)
    out = {}
    for name in OUT3:
        a = r.take(">f8", T * Nr * Ny * Nx).reshape(size["nSy"], size["nSx"], Nr, Ny, Nx)
        out[name] = _fortran_to_tile(a, size)
    r.done()
    return out


def read_grid(rundir, size):
    """{name: array}: GRID3 [tile,k,j,i], GRID2 [tile,j,i], cosFacU/cosFacV [tile,j], GRID_NR [k], GRID_NR1 [k+1],
    rkSign (float), as the routines read them (after the harness overrides)."""
    r = _Reader(Path(rundir) / "replay_grid.bin")
    hdr = r.take(">i4", 4)
    if hdr[0] != MAGIC or hdr[1] != VERSION or hdr[2] != len(GRID3) or hdr[3] != len(GRID2):
        raise ValueError(f"bad replay_grid.bin header {hdr}")
    T, Nr, Ny, Nx = _dims(size)
    g = {}
    for name in GRID3:
        g[name] = _fortran_to_tile(r.take(">f8", T * Nr * Ny * Nx).reshape(size["nSy"], size["nSx"], Nr, Ny, Nx),
                                   size)
    for name in GRID2:
        g[name] = _fortran_to_tile(r.take(">f8", T * Ny * Nx).reshape(size["nSy"], size["nSx"], Ny, Nx), size)
    for name in ("cosFacU", "cosFacV"):
        g[name] = _fortran_to_tile(r.take(">f8", T * Ny).reshape(size["nSy"], size["nSx"], Ny), size)
    for name in GRID_NR:
        g[name] = r.take(">f8", Nr)
    for name in GRID_NR1:
        g[name] = r.take(">f8", Nr + 1)
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
    w.add_argument("--seed", type=int, default=20261002)
    a = ap.parse_args(argv)
    meta = write_inputs(a.rundir, parse_size_h(a.size), a.experiment, a.seed)
    print(f"replay inputs written: {a.experiment} {json.dumps(meta['size'])} seed {meta['seed']} {meta['sha256']}")


if __name__ == "__main__":
    sys.exit(main())
