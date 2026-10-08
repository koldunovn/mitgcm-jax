#!/usr/bin/env python3
"""Inputs and outputs of the GAD-C replay harness (lane GAD-C; reference/replay_gad_c/code/the_main_loop.F, a copy of
the Task 8 harness reference/replay).

    replay_io.py write-inputs RUNDIR --size SIZE.h [--seed N]     (run by reference/replay_gad_c/jobs/replay.sbatch)

The harness is a genmake2 build of advect_xz/code in which THE_MAIN_LOOP is replaced: THE_MODEL_MAIN runs
INITIALISE_FIXED (real maskC with the slope's land, exchanged halos), the replacement overrides the metric factors
that are uniform in the experiment with the synthetic values written here, calls GAD_PPM_ADV_X/Y/R and
GAD_PQM_ADV_X/Y/R for every limiter (and calc_CFL T/F for X/Y), and writes every output array at all points.

Files (all big-endian stream; Fortran column-major, so a Fortran array A(i,j,k,bi,bj) is the numpy C-order array
[bj, bi, k, j, i], and the tile index is t = (bj-1)*nSx + (bi-1), bi fastest):
    replay_in.bin    int32 MAGIC, VERSION, sNx, sNy, OLx, OLy, nSx, nSy, Nr, len(IN3);
                     float64 deltaT, dtR(Nr); recip_deepFacC(Nr), drF(Nr), recip_drF(Nr), recip_drC(Nr+1);
                     GRID2 fields (1-OLx:sNx+OLx, 1-OLy:sNy+OLy, nSx, nSy); IN3 fields (..., Nr, nSx, nSy)
    replay_out.bin   int32 MAGIC, VERSION, len(OUT3), Nr; OUT3 fields
    replay_grid.bin  int32 MAGIC, VERSION, 1, 6; maskC; GRID2; recip_deepFacC, drF, recip_drF, recip_drC

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
IN3 = ("uVel", "uCFL", "uTrans", "vVel", "vCFL", "vTrans", "wVel", "wTrans", "tracer", "fluxPrior")
GRID2 = ("dxF", "dyF", "recip_dxF", "recip_dyF", "recip_dxC", "recip_dyC")
GRID1 = (("recip_deepFacC", 0), ("drF", 0), ("recip_drF", 0), ("recip_drC", 1))     # (name, extra length over Nr)
# out3 index n (1-based) = 1 + 2*(iD*3 + iM-1) + (iC-1) for the X/Y drivers, 25 + iD*3 + iM-1 for the R drivers
XY_DRIVERS = (("gad_ppm_adv_x", (40, 41, 42)), ("gad_pqm_adv_x", (50, 51, 52)),
              ("gad_ppm_adv_y", (40, 41, 42)), ("gad_pqm_adv_y", (50, 51, 52)))
R_DRIVERS = (("gad_ppm_adv_r", (40, 41, 42)), ("gad_pqm_adv_r", (50, 51, 52)))
CASES = (tuple((d, m, c) for d, ms in XY_DRIVERS for m in ms for c in (True, False))
         + tuple((d, m, None) for d, ms in R_DRIVERS for m in ms))
OUT3 = tuple(f"{d}_{m}" + ("" if c is None else ("_calcCFL" if c else "_givenCFL")) for d, m, c in CASES)
SIZE_KEYS = ("sNx", "sNy", "OLx", "OLy", "nSx", "nSy", "Nr")


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


def _runs(rng, shape, axis, frac, length=(4, 7)):
    """Boolean mask of plateaus: in a fraction `frac` of the lines along `axis`, one run of 4-6 consecutive points."""
    m = np.zeros(shape, bool)
    a = np.moveaxis(m, axis, -1)
    n = a.shape[-1]
    lines = np.argwhere(rng.uniform(size=a.shape[:-1]) < frac)
    for idx in lines:
        ln = int(rng.integers(*length))
        s = int(rng.integers(0, n - ln + 1))
        a[tuple(idx)][s:s + ln] = True
    return m


def make_inputs(size, seed=20261002):
    """Synthetic inputs at physical magnitudes, storage [tile, k, j, i] incl. halos.

    tracer: 10 + 2 sin(.) (smooth: monotone stretches and smooth extrema along i, j and k) + noise on 30 % of the
    points (sharp extrema), with plateaus (exactly equal values: the flat and tie branches of the limiters) on runs
    of 4-6 points along i (20 % of the rows), along k (20 % of the columns) and along j (10 %).
    Velocities: exact zeros on 15 % of the points (the zero-velocity IF); |CFL| < ~0.9; the transports have whole
    rows (x: along i; y: along j) or columns (r: along k) of zeros (the vsum = 0 IF), and wVel is zero on levels
    2..Nr (not 1) of some columns (GAD_PPM_ADV_R sums |velR| over 2..Nr only). Metric factors are synthetic and
    distinct (a missing or swapped factor shows)."""
    rng = np.random.default_rng(seed)
    T = size["nSx"] * size["nSy"]
    Nr, Ny, Nx = size["Nr"], size["sNy"] + 2 * size["OLy"], size["sNx"] + 2 * size["OLx"]
    shp = (T, Nr, Ny, Nx)
    k, j, i = np.meshgrid(np.arange(Nr), np.arange(Ny), np.arange(Nx), indexing="ij")
    tracer = np.broadcast_to(10.0 + 2.0 * np.sin(0.55 * i + 0.9 * j + 0.45 * k), shp).copy()
    tracer += 0.1 * np.arange(T)[:, None, None, None]
    tracer += np.where(rng.uniform(size=shp) < 0.3, 0.5 * rng.standard_normal(shp), 0.0)
    for axis, frac in ((3, 0.2), (1, 0.2), (2, 0.1)):
        m = _runs(rng, shp, axis, frac)
        a = np.moveaxis(tracer, axis, -1)
        am = np.moveaxis(m, axis, -1)
        for idx in np.argwhere(am.any(axis=-1)):
            line = a[tuple(idx)]
            sel = am[tuple(idx)]
            line[sel] = line[sel][0]                       # one constant over the run
    zero = lambda f: rng.uniform(size=shp) < f
    uVel = np.where(zero(0.15), 0.0, rng.uniform(-6.0, 6.0, shp))
    uCFL = np.where(zero(0.15), 0.0, rng.uniform(-0.9, 0.9, shp))
    uTrans = np.where(zero(0.1), 0.0, uVel * 1.0e6 + rng.standard_normal(shp) * 1.0e5)
    uTrans[rng.uniform(size=(T, Nr, Ny)) < 0.2, :] = 0.0                  # whole rows along i
    vVel = np.where(zero(0.15), 0.0, rng.uniform(-6.0, 6.0, shp))
    vCFL = np.where(zero(0.15), 0.0, rng.uniform(-0.9, 0.9, shp))
    vTrans = np.where(zero(0.1), 0.0, vVel * 1.0e6 + rng.standard_normal(shp) * 1.0e5)
    vt = np.moveaxis(vTrans, 2, -1)
    vt[rng.uniform(size=(T, Nr, Nx)) < 0.2, :] = 0.0                      # whole rows along j
    wVel = np.where(zero(0.15), 0.0, rng.uniform(-0.07, 0.07, shp))
    wTrans = np.where(zero(0.1), 0.0, wVel * 1.0e8 + rng.standard_normal(shp) * 1.0e6)
    cols = rng.uniform(size=(T, Ny, Nx)) < 0.2
    wt = np.moveaxis(wTrans, 1, -1)
    wt[cols, :] = 0.0                                                     # whole columns along k
    wv = np.moveaxis(wVel, 1, -1)
    cols2 = rng.uniform(size=(T, Ny, Nx)) < 0.2
    wv[cols2, 1:] = 0.0                                                   # levels 2..Nr only
    wv[cols2, 0] = rng.uniform(0.01, 0.05, cols2.sum())
    f = {"uVel": uVel, "uCFL": uCFL, "uTrans": uTrans, "vVel": vVel, "vCFL": vCFL, "vTrans": vTrans,
         "wVel": wVel, "wTrans": wTrans, "tracer": tracer, "fluxPrior": rng.standard_normal(shp) * 1.0e6 - 3.0e6}
    assert tuple(f) == IN3
    g2 = {n: (rng.uniform(0.8e4, 1.2e4, (T, Ny, Nx)) if n in ("dxF", "dyF")
              else rng.uniform(0.8e-4, 1.2e-4, (T, Ny, Nx))) for n in GRID2}
    g1 = {"recip_deepFacC": 1.0 + 0.01 * rng.uniform(0.5, 1.5, Nr), "drF": rng.uniform(90.0, 110.0, Nr),
          "recip_drF": rng.uniform(0.009, 0.011, Nr), "recip_drC": rng.uniform(0.009, 0.011, Nr + 1)}
    scal = {"deltaT": 1200.0, "dtR": rng.uniform(1100.0, 1300.0, Nr)}
    return f, g2, g1, scal


def write_inputs(rundir, size, seed=20261002):
    rundir = Path(rundir)
    f, g2, g1, scal = make_inputs(size, seed)
    p_in = rundir / "replay_in.bin"
    if p_in.exists():
        raise SystemExit(f"{p_in} exists (nothing is overwritten)")
    hdr = [MAGIC, VERSION] + [size[k] for k in SIZE_KEYS] + [len(IN3)]
    with open(p_in, "xb") as fh:
        fh.write(np.array(hdr, ">i4").tobytes())
        fh.write(np.array([scal["deltaT"]], ">f8").tobytes())
        fh.write(scal["dtR"].astype(">f8").tobytes())
        for name, _ in GRID1:
            fh.write(g1[name].astype(">f8").tobytes())
        for name in GRID2:
            fh.write(_tile_to_fortran(g2[name], size).astype(">f8").tobytes())
        for name in IN3:
            fh.write(_tile_to_fortran(f[name], size).astype(">f8").tobytes())
    meta = {"seed": seed, "size": size, "IN3": IN3, "OUT3": OUT3,
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
    """(size, fields {name: [tile,k,j,i]}, grid2 {name: [tile,j,i]}, grid1 {name: [k]}, scal {deltaT, dtR[k]})."""
    r = _Reader(Path(rundir) / "replay_in.bin")
    hdr = r.take(">i4", 10)
    if hdr[0] != MAGIC or hdr[1] != VERSION or hdr[9] != len(IN3):
        raise ValueError(f"bad replay_in.bin header {hdr}")
    size = dict(zip(SIZE_KEYS, (int(x) for x in hdr[2:9])))
    T, Nr, Ny, Nx = _dims(size)
    scal = {"deltaT": float(r.take(">f8", 1)[0]), "dtR": r.take(">f8", Nr)}
    g1 = {name: r.take(">f8", Nr + extra) for name, extra in GRID1}
    g2 = {name: _fortran_to_tile(r.take(">f8", T * Ny * Nx).reshape(size["nSy"], size["nSx"], Ny, Nx), size)
          for name in GRID2}
    fields = {name: _fortran_to_tile(r.take(">f8", T * Nr * Ny * Nx).reshape(size["nSy"], size["nSx"], Nr, Ny, Nx),
                                     size) for name in IN3}
    r.done()
    return size, fields, g2, g1, scal


def read_outputs(rundir, size):
    r = _Reader(Path(rundir) / "replay_out.bin")
    hdr = r.take(">i4", 4)
    if hdr[0] != MAGIC or hdr[1] != VERSION or hdr[2] != len(OUT3) or hdr[3] != size["Nr"]:
        raise ValueError(f"bad replay_out.bin header {hdr}")
    T, Nr, Ny, Nx = _dims(size)
    out = {name: _fortran_to_tile(r.take(">f8", T * Nr * Ny * Nx).reshape(size["nSy"], size["nSx"], Nr, Ny, Nx),
                                  size) for name in OUT3}
    r.done()
    return out


def read_grid(rundir, size):
    """{maskC: [tile,k,j,i], GRID2: [tile,j,i], GRID1: [k]} as the routines read them (after the harness overrides)."""
    r = _Reader(Path(rundir) / "replay_grid.bin")
    hdr = r.take(">i4", 4)
    if hdr[0] != MAGIC or hdr[1] != VERSION or hdr[2] != 1 or hdr[3] != len(GRID2):
        raise ValueError(f"bad replay_grid.bin header {hdr}")
    T, Nr, Ny, Nx = _dims(size)
    g = {"maskC": _fortran_to_tile(r.take(">f8", T * Nr * Ny * Nx).reshape(size["nSy"], size["nSx"], Nr, Ny, Nx),
                                   size)}
    for name in GRID2:
        g[name] = _fortran_to_tile(r.take(">f8", T * Ny * Nx).reshape(size["nSy"], size["nSx"], Ny, Nx), size)
    for name, extra in GRID1:
        g[name] = r.take(">f8", Nr + extra)
    r.done()
    return g


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write-inputs")
    w.add_argument("rundir")
    w.add_argument("--size", required=True, help="the build's SIZE.h")
    w.add_argument("--seed", type=int, default=20261002)
    a = ap.parse_args(argv)
    meta = write_inputs(a.rundir, parse_size_h(a.size), a.seed)
    print(f"replay inputs written: {json.dumps(meta['size'])} seed {meta['seed']} {meta['sha256']}")


if __name__ == "__main__":
    sys.exit(main())
