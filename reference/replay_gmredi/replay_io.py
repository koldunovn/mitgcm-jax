#!/usr/bin/env python3
"""Inputs and outputs of the GM/Redi replay harness (M1 sub-lane GMREDI; reference/replay_gmredi/code/the_main_loop.F,
a copy of the Task 8 harness reference/replay).

    replay_io.py write-inputs RUNDIR --size SIZE.h [--seed N]     (run by reference/replay_gmredi/jobs/replay.sbatch)

The harness is a genmake2 build of global_ocean.90x40x15/code or tutorial_global_oce_optim/code_ad (forward only) in
which THE_MAIN_LOOP is replaced: THE_MODEL_MAIN runs INITIALISE_FIXED (real grid, GMREDI_READPARMS,
GMREDI_INIT_FIXED); the replacement writes GMREDI_INIT_FIXED's factors and the GMREDI.h parameters, reads the
synthetic inputs written here, calls GMREDI_INIT_VARIA, overrides the trivial factors, then GMREDI_CALC_TENSOR,
GMREDI_SLOPE_LIMIT (kPos 1..3, every level), GMREDI_X/Y/RTRANSPORT, GMREDI_CALC_DIFF and GMREDI_RESIDUAL_FLOW for
every tile, and writes every output (all points, incl. those a routine does not write: they keep the prior given here)
and the grid/parameters the routines read.

Synthetic but physical inputs over the real grid: stable stratification sigmaR = dRho/dr < 0 (z-coords, kg/m^4,
N^2 ~ 1e-6..1e-4), with unstable points and whole unstable columns (sigmaR > 0), neutral points and columns
(sigmaR = 0 exactly), stratification below GM_Small_Number, horizontal gradients giving slopes from 1e-6 to far beyond
GM_maxSlope (incl. GM_slopeSqCutoff of the optim run), exact zeros (SlopeSqr = 0); the harness multiplies sigmaX/Y/R
by maskW/S/C (land zero, as GRAD_SIGMA leaves them). The direct GMREDI_SLOPE_LIMIT inputs also cover dSigmaDr = 0
with nonzero gradients of both signs (SIGN(GM_bigSlope, .)), and a few slopes beyond 1e24 (GM_slopeSqCutoff = 1e48
of global_ocean). Every value is a random binary64 (every rounding counts).

Files (big-endian stream; a Fortran array A(i,j,k,bi,bj) is the numpy C-order array [bj, bi, k, j, i]; tile
t = (bj-1)*nSx + (bi-1), bi fastest):
    replay_in.bin    int32 MAGIC, VERSION, sNx, sNy, OLx, OLy, nSx, nSy, Nr, len(IN3), len(IN2); float64 PAR;
                     wUnit2rVel, rVel2wUnit (Nr+1); rUnit2z, z2rUnit (Nr); GM_isoFac1d, GM_bolFac1d (Nr);
                     recip_deepFacC (Nr); deepFac2F (Nr+1); IN2 fields (xy tiles); IN3 fields (xyz tiles)
    replay_fixed.bin int32 MAGIC, VERSION, len(GP), Nr; GM_isoFac2d, GM_bolFac2d; GM_isoFac1d, GM_bolFac1d; GP
                     (GMREDI_INIT_FIXED's factors and the parameters as GMREDI_READPARMS left them)
    replay_out.bin   int32 MAGIC, VERSION, len(OUT3), Nr; OUT3 fields
    replay_grid.bin  int32 MAGIC, VERSION, len(GP), Nr; GRID3; GRID2; kLowC (float64); rF, recip_drC; rC,
                     recip_deepFacC; deepFac2F, wUnit2rVel, rVel2wUnit; rUnit2z, z2rUnit; GM_isoFac1d, GM_bolFac1d;
                     gravitySign; GP (after the overrides); usingZCoords (1./0.)

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

IN3 = ("sigmaX", "sigmaY", "sigmaR",
       "KwxP", "KwyP", "KwzP", "KuxP", "KvyP", "KuzP", "KvzP", "PsiXP", "PsiYP",
       "dSx", "dSy", "dSr", "SlopeXP", "SlopeYP", "SlopeSqrP", "taperFctP",
       "Tracer", "xA", "yA", "maskFk", "maskUp", "dfP", "KappaRxP", "uFld", "vFld", "wFld")
IN2 = ("GM_isoFac2d", "GM_bolFac2d", "hMixLayer")
PAR = ("GM_isopycK", "GM_skewflx", "unused")
TENSOR = ("Kwx", "Kwy", "Kwz", "Kux", "Kvy", "Kuz", "Kvz", "GM_PsiX", "GM_PsiY")
SLOPE_OUT = ("SlopeX", "SlopeY", "SlopeSqr", "taperFct", "dSigmaDr")
OUT3 = (tuple(f"initvaria_{n}" for n in TENSOR) + tuple(f"tensor_{n}" for n in TENSOR)
        + tuple(f"slope{kp}_{n}" for kp in (1, 2, 3) for n in SLOPE_OUT)
        + ("dfX", "dfY", "dfR", "kappa_k0_tr1", "kappa_k0_tr3", "kappa_kArg_tr2", "uRes", "vRes", "wRes"))
GP = ("GM_isopycK", "GM_background_K", "GM_maxSlope", "GM_Kmin_horiz", "GM_Small_Number", "GM_slopeSqCutoff",
      "GM_Scrit", "GM_Sd", "GM_rMaxSlope", "GM_skewflx", "GM_ExtraDiag")
GRID3 = ("maskC", "maskW", "maskS")
GRID2 = ("R_low", "recip_dxC", "recip_dyC", "rA", "maskInC")
SIZE_KEYS = ("sNx", "sNy", "OLx", "OLy", "nSx", "nSy", "Nr")
assert len(IN3) == 29 and len(IN2) == 3 and len(OUT3) == 42 and len(GP) == 11    # the_main_loop.F PARAMETERs


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


def make_inputs(size, seed=20261002):
    rng = np.random.default_rng(seed)
    T, Nr = size["nSx"] * size["nSy"], size["Nr"]
    Ny, Nx = size["sNy"] + 2 * size["OLy"], size["sNx"] + 2 * size["OLx"]
    shp, shp2 = (T, Nr, Ny, Nx), (T, Ny, Nx)
    n = lambda scale, s=shp: scale * rng.standard_normal(s)
    u = lambda lo, hi, s=shp: rng.uniform(lo, hi, s)
    sgn = lambda s=shp: np.where(rng.uniform(size=s) < 0.5, -1.0, 1.0)
    logu = lambda lo, hi, s=shp: 10.0 ** rng.uniform(lo, hi, s)

    # stratification: sigmaR = dRho/dr (r = z upward), stable < 0
    sigmaR = -logu(-4, -2)
    r = rng.uniform(size=shp)
    sigmaR = np.where(r < 0.08, logu(-5, -3), sigmaR)                       # unstable points
    sigmaR = np.where((r >= 0.08) & (r < 0.13), 0.0, sigmaR)                # neutral points
    sigmaR = np.where((r >= 0.13) & (r < 0.16), -logu(-30, -21), sigmaR)    # below GM_Small_Number
    col = rng.uniform(size=shp2)[:, None]
    sigmaR = np.where(col < 0.05, logu(-5, -3), sigmaR)                     # unstable columns
    sigmaR = np.where((col >= 0.05) & (col < 0.10), 0.0, sigmaR)            # neutral columns
    sigmaX, sigmaY = n(1e-7), n(1e-7)
    steep = rng.uniform(size=shp) < 0.15                                    # slopes beyond GM_maxSlope
    sigmaX = np.where(steep, sgn() * logu(-5, -3), sigmaX)
    sigmaY = np.where(steep & (rng.uniform(size=shp) < 0.5), sgn() * logu(-5, -3), sigmaY)
    flat = rng.uniform(size=shp) < 0.05                                     # SlopeSqr = 0
    sigmaX, sigmaY = np.where(flat, 0.0, sigmaX), np.where(flat, 0.0, sigmaY)

    # direct GMREDI_SLOPE_LIMIT inputs (dSigmaDr = downward gradient, stable > 0)
    dSr = logu(-4, -2)
    r = rng.uniform(size=shp)
    dSr = np.where(r < 0.10, -logu(-5, -3), dSr)                            # unstable
    dSr = np.where((r >= 0.10) & (r < 0.18), 0.0, dSr)                      # neutral: SIGN(GM_bigSlope, .)
    dSr = np.where((r >= 0.18) & (r < 0.22), logu(-30, -20.5), dSr)         # 0 < dSigmaDr <= GM_Small_Number
    dSx, dSy = n(1e-6), n(1e-6)
    steep = rng.uniform(size=shp) < 0.2
    dSx = np.where(steep, sgn() * logu(-5, -2), dSx)
    dSy = np.where(steep & (rng.uniform(size=shp) < 0.5), sgn() * logu(-5, -2), dSy)
    huge = rng.uniform(size=shp) < 0.003                                    # beyond the 1e48 cut-off
    dSx = np.where(huge, sgn() * logu(4.5, 6), dSx)
    r = rng.uniform(size=shp)
    dSx = np.where(r < 0.05, 0.0, dSx)                                      # zero gradients
    dSy = np.where(r < 0.08, 0.0, dSy)

    f = {"sigmaX": sigmaX, "sigmaY": sigmaY, "sigmaR": sigmaR,
         "KwxP": n(0.1), "KwyP": n(0.1), "KwzP": u(0.0, 1e-2), "KuxP": u(50.0, 2000.0), "KvyP": u(50.0, 2000.0),
         "KuzP": n(1.0) + 3.0, "KvzP": n(1.0) - 3.0, "PsiXP": n(1.0) + 5.0, "PsiYP": n(1.0) - 5.0,
         "dSx": dSx, "dSy": dSy, "dSr": dSr,
         "SlopeXP": n(1.0) + 7.0, "SlopeYP": n(1.0) - 7.0, "SlopeSqrP": n(1.0) + 9.0, "taperFctP": n(1.0) - 9.0,
         "Tracer": 10.0 + n(5.0), "xA": u(1e8, 1e10), "yA": u(1e8, 1e10),
         "maskFk": (rng.uniform(size=shp) < 0.8).astype(np.float64),
         "maskUp": (rng.uniform(size=shp) < 0.8).astype(np.float64),
         "dfP": n(1e6), "KappaRxP": u(1e-5, 1e-3), "uFld": n(0.1), "vFld": n(0.1), "wFld": n(1e-4)}
    assert tuple(f) == IN3
    f2 = {"GM_isoFac2d": u(0.5, 1.5, shp2), "GM_bolFac2d": u(0.5, 1.5, shp2), "hMixLayer": u(10.0, 300.0, shp2)}
    assert tuple(f2) == IN2
    par = {"GM_isopycK": 700.0, "GM_skewflx": 0.75, "unused": 0.0}
    one_d = {"wUnit2rVel": u(0.5, 2.0, Nr + 1), "rVel2wUnit": u(0.5, 2.0, Nr + 1), "rUnit2z": u(0.5, 2.0, Nr),
             "z2rUnit": u(0.5, 2.0, Nr), "GM_isoFac1d": u(0.5, 1.5, Nr), "GM_bolFac1d": u(0.5, 1.5, Nr),
             "recip_deepFacC": 1.0 + 0.01 * u(0.5, 1.5, Nr), "deepFac2F": 1.0 + 0.02 * u(0.5, 1.5, Nr + 1)}
    return f, f2, par, one_d


ONE_D = ("wUnit2rVel", "rVel2wUnit", "rUnit2z", "z2rUnit", "GM_isoFac1d", "GM_bolFac1d", "recip_deepFacC",
         "deepFac2F")


def write_inputs(rundir, size, seed=20261002):
    rundir = Path(rundir)
    f, f2, par, one_d = make_inputs(size, seed)
    p_in = rundir / "replay_in.bin"
    if p_in.exists():
        raise SystemExit(f"{p_in} exists (nothing is overwritten)")
    hdr = [MAGIC, VERSION] + [size[k] for k in SIZE_KEYS] + [len(IN3), len(IN2)]
    with open(p_in, "xb") as fh:
        fh.write(np.array(hdr, ">i4").tobytes())
        fh.write(np.array([par[k] for k in PAR], ">f8").tobytes())
        for name in ONE_D:
            fh.write(one_d[name].astype(">f8").tobytes())
        for name in IN2:
            fh.write(_tile_to_fortran(f2[name], size).astype(">f8").tobytes())
        for name in IN3:
            fh.write(_tile_to_fortran(f[name], size).astype(">f8").tobytes())
    meta = {"seed": seed, "size": size, "IN3": IN3, "IN2": IN2, "OUT3": OUT3,
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


def _xyz(r, size):
    T, Nr, Ny, Nx = _dims(size)
    return _fortran_to_tile(r.take(">f8", T * Nr * Ny * Nx).reshape(size["nSy"], size["nSx"], Nr, Ny, Nx), size)


def _xy(r, size):
    T, Nr, Ny, Nx = _dims(size)
    return _fortran_to_tile(r.take(">f8", T * Ny * Nx).reshape(size["nSy"], size["nSx"], Ny, Nx), size)


def read_inputs(rundir):
    """(size, IN3 {name: [tile,k,j,i]}, IN2 {name: [tile,j,i]}, PAR {name: float}, ONE_D {name: [k]})."""
    r = _Reader(Path(rundir) / "replay_in.bin")
    hdr = r.take(">i4", 11)
    if hdr[0] != MAGIC or hdr[1] != VERSION or hdr[9] != len(IN3) or hdr[10] != len(IN2):
        raise ValueError(f"bad replay_in.bin header {hdr}")
    size = dict(zip(SIZE_KEYS, (int(x) for x in hdr[2:9])))
    Nr = size["Nr"]
    par = dict(zip(PAR, (float(x) for x in r.take(">f8", len(PAR)))))
    lens = {"wUnit2rVel": Nr + 1, "rVel2wUnit": Nr + 1, "deepFac2F": Nr + 1}
    one_d = {name: r.take(">f8", lens.get(name, Nr)) for name in ONE_D}
    f2 = {name: _xy(r, size) for name in IN2}
    f3 = {name: _xyz(r, size) for name in IN3}
    r.done()
    return size, f3, f2, par, one_d


def read_outputs(rundir, size):
    r = _Reader(Path(rundir) / "replay_out.bin")
    hdr = r.take(">i4", 4)
    if hdr[0] != MAGIC or hdr[1] != VERSION or hdr[2] != len(OUT3) or hdr[3] != size["Nr"]:
        raise ValueError(f"bad replay_out.bin header {hdr}")
    out = {name: _xyz(r, size) for name in OUT3}
    r.done()
    return out


def read_fixed(rundir, size):
    """GMREDI_INIT_FIXED's factors and the GMREDI.h parameters as GMREDI_READPARMS left them."""
    r = _Reader(Path(rundir) / "replay_fixed.bin")
    hdr = r.take(">i4", 4)
    if hdr[0] != MAGIC or hdr[1] != VERSION or hdr[2] != len(GP) or hdr[3] != size["Nr"]:
        raise ValueError(f"bad replay_fixed.bin header {hdr}")
    out = {"GM_isoFac2d": _xy(r, size), "GM_bolFac2d": _xy(r, size),
           "GM_isoFac1d": r.take(">f8", size["Nr"]), "GM_bolFac1d": r.take(">f8", size["Nr"])}
    out.update(zip(GP, (float(x) for x in r.take(">f8", len(GP)))))
    r.done()
    return out


def read_grid(rundir, size):
    """{name: array} as the routines read them (after the harness overrides): GRID3 [tile,k,j,i], GRID2 and kLowC
    [tile,j,i], the vertical vectors [k], gravitySign, usingZCoords, and the GP parameters (after the overrides)."""
    r = _Reader(Path(rundir) / "replay_grid.bin")
    hdr = r.take(">i4", 4)
    if hdr[0] != MAGIC or hdr[1] != VERSION or hdr[2] != len(GP) or hdr[3] != size["Nr"]:
        raise ValueError(f"bad replay_grid.bin header {hdr}")
    Nr = size["Nr"]
    g = {name: _xyz(r, size) for name in GRID3}
    g.update({name: _xy(r, size) for name in GRID2})
    g["kLowC"] = _xy(r, size)
    for name, n in (("rF", Nr + 1), ("recip_drC", Nr + 1), ("rC", Nr), ("recip_deepFacC", Nr),
                    ("deepFac2F", Nr + 1), ("wUnit2rVel", Nr + 1), ("rVel2wUnit", Nr + 1), ("rUnit2z", Nr),
                    ("z2rUnit", Nr), ("GM_isoFac1d", Nr), ("GM_bolFac1d", Nr)):
        g[name] = r.take(">f8", n)
    g["gravitySign"] = float(r.take(">f8", 1)[0])
    g.update(zip(GP, (float(x) for x in r.take(">f8", len(GP)))))
    g["usingZCoords"] = bool(r.take(">f8", 1)[0] == 1.0)
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
