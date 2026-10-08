#!/usr/bin/env python3
"""Inputs and outputs of the one-routine replay harness (plan Task 8; reference/replay/code/the_main_loop.F).

    replay_io.py write-inputs RUNDIR --size SIZE.h [--seed N]     (run by reference/replay/jobs/replay.sbatch)

The harness is a genmake2 build of a verification experiment's code/ directory in which THE_MAIN_LOOP is replaced
(R's reference/adx_harness pattern [E§2]): THE_MODEL_MAIN runs INITIALISE_FIXED, so GRID.h holds the experiment's
real grid (land, partial cells, exchanged halos); the replacement reads the synthetic inputs written here, calls
MOM_CALC_KE, GAD_DST3_ADV_X and MOM_VI_HDISSIP for every tile and level, and writes every output array (all points,
incl. the ones a routine does not write: they keep the prior field given here), the grid as the routines read it, and
a float64 divide probe.

Files (all big-endian stream; Fortran column-major, so a Fortran array A(i,j,k,bi,bj) is the numpy C-order array
[bj, bi, k, j, i], and the tile index is t = (bj-1)*nSx + (bi-1), bi fastest):
    replay_in.bin    int32 MAGIC, VERSION, sNx, sNy, OLx, OLy, nSx, nSy, Nr, len(IN3);
                     float64 deltaTloc, viscAhD, viscAhZ, viscA4D, viscA4Z; recip_deepFacC(Nr);
                     cosFacU, cosFacV (1-OLy:sNy+OLy, nSx, nSy); IN3 fields (1-OLx:sNx+OLx, 1-OLy:sNy+OLy, Nr, nSx, nSy)
    divide_in.bin    int32 MAGIC, VERSION, n; n pairs (a, b) of float64
    replay_out.bin   int32 MAGIC, VERSION, len(OUT3), Nr; OUT3 fields
    replay_grid.bin  int32 MAGIC, VERSION, 8, 7; GRID3 (Nr levels), GRID2, cosFacU, cosFacV, recip_deepFacC, PARAMS
    divide_out.bin   int32 MAGIC, VERSION, 5; per pair: a/b, a/3.D0, a/1000.D0, a/7.D0, a/0.1D0

Numpy only (no jax): it runs in the batch job before the model.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

MAGIC, VERSION = 20261001, 1

# the order of the in3(..., f) fields in the_main_loop.F
IN3 = ("uFld", "vFld", "KEprior", "uTrans", "uVel", "uCFL", "tracer", "uTprior", "hDiv", "vort3", "dStar", "zStar",
       "hFacZ", "viscAh_Z", "viscAh_D", "viscA4_Z", "viscA4_D", "uDissipPrior", "vDissipPrior")
# MOM_VI_HDISSIP combinations: iCmb bit 0 = harmonic, bit 1 = biharmonic, bit 2 = useVariableViscosity
HDISSIP_CASES = tuple((bool(c & 1), bool(c & 2), bool(c & 4)) for c in range(8))
OUT3 = (("KE_m1", "KE_0", "KE_1", "KE_2", "KE_3", "uT_calcCFL", "uT_givenCFL")
        + tuple(f"{uv}Dissip_c{c}" for c in range(8) for uv in "uv"))
KESCHEMES = (-1, 0, 1, 2, 3)
GRID3 = ("hFacC", "hFacW", "hFacS", "recip_hFacC", "recip_hFacW", "recip_hFacS", "maskW", "maskS")
GRID2 = ("rAw", "rAs", "recip_rA", "recip_dxC", "recip_dyC", "recip_dxG", "recip_dyG")
PARAMS = ("viscAhD", "viscAhZ", "viscA4D", "viscA4Z")
DIVIDE_LITERALS = (3.0, 1000.0, 7.0, 0.1)          # the_main_loop.F: a/3.D0, a/1000.D0, a/7.D0, a/0.1D0
SIZE_KEYS = ("sNx", "sNy", "OLx", "OLy", "nSx", "nSy", "Nr")
DIVIDE_N = 1 << 16


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


def make_inputs(size, seed=20261001):
    """Synthetic inputs: random binary64 values at physical magnitudes (every rounding counts), storage
    [tile, k, j, i] incl. halos. hFacZ has exact zeros; deltaTloc is global_ocean.90x40x15's deltaTtracer
    (input/data: 86400.); viscAhD and viscA4D are its viscAh, viscA4 (input/data:10-11), viscAhZ and viscA4Z are
    synthetic and distinct (a D/Z swap must show); cosFacU, cosFacV and recip_deepFacC are 1 in that experiment and
    get synthetic values here (a missing factor must show)."""
    rng = np.random.default_rng(seed)
    T = size["nSx"] * size["nSy"]
    Ny, Nx = size["sNy"] + 2 * size["OLy"], size["sNx"] + 2 * size["OLx"]
    shp = (T, size["Nr"], Ny, Nx)
    n = lambda scale: scale * rng.standard_normal(shp)
    u = lambda lo, hi: rng.uniform(lo, hi, shp)
    hFacZ = u(0.0, 1.0)
    hFacZ[rng.uniform(size=shp) < 0.2] = 0.0
    f = {"uFld": n(0.3), "vFld": n(0.3), "KEprior": n(1.0) + 5.0, "uTrans": n(2.0e6), "uVel": n(0.3),
         "uCFL": u(-0.9, 0.9), "tracer": 10.0 + n(5.0), "uTprior": n(1.0e6) - 3.0e6, "hDiv": n(1.0e-6),
         "vort3": n(1.0e-6), "dStar": n(1.0e-17), "zStar": n(1.0e-17), "hFacZ": hFacZ,
         "viscAh_Z": u(1.0e5, 1.0e6), "viscAh_D": u(1.0e5, 1.0e6), "viscA4_Z": u(1.0e13, 2.0e14),
         "viscA4_D": u(1.0e13, 2.0e14), "uDissipPrior": n(1.0e-6) + 1.0e-5, "vDissipPrior": n(1.0e-6) - 1.0e-5}
    assert tuple(f) == IN3
    scal = {"deltaTloc": 86400.0, "viscAhD": 5.0e5, "viscAhZ": 2.5e5, "viscA4D": 1.0e14, "viscA4Z": 0.6e14}
    one_d = {"recip_deepFacC": 1.0 + 0.01 * rng.uniform(0.5, 1.5, size["Nr"]),
             "cosFacU": rng.uniform(0.2, 1.0, (T, Ny)), "cosFacV": rng.uniform(0.2, 1.0, (T, Ny))}
    return f, scal, one_d


def make_divide_pairs(seed=20261001, n=DIVIDE_N):
    """n (a, b) float64 pairs over many magnitudes: normal range, the full exponent range (quotients that overflow
    or underflow), subnormal-adjacent quotients, subnormal dividends, large ratios, and the literal-like divisors
    3, 7, 10, 1000, 0.1, 6, 1.5, 9.81, 86400."""
    rng = np.random.default_rng(seed + 1)
    q = n // 8
    sgn = lambda m: np.where(rng.uniform(size=m) < 0.5, -1.0, 1.0)
    mag = lambda lo, hi, m: 10.0 ** rng.uniform(lo, hi, m) * rng.uniform(1.0, 2.0, m)
    tiny = np.finfo(np.float64).tiny
    parts = [
        (sgn(2 * q) * mag(-10, 10, 2 * q), sgn(2 * q) * mag(-10, 10, 2 * q)),                    # normal range
        (sgn(2 * q) * mag(-300, 300, 2 * q), sgn(2 * q) * mag(-300, 300, 2 * q)),                # full range
        (sgn(q) * tiny * rng.uniform(0.5, 4.0, q), rng.uniform(1.0, 16.0, q)),                   # -> subnormal
        (sgn(q) * tiny * rng.uniform(0.0, 1.0, q), sgn(q) * rng.uniform(0.1, 10.0, q)),          # subnormal a
        (sgn(q) * mag(200, 300, q), sgn(q) * mag(-100, -5, q)),                                  # large ratios
    ]
    m = n - sum(len(a) for a, _ in parts)
    lit = np.array([3.0, 7.0, 10.0, 1000.0, 0.1, 6.0, 1.5, 9.81, 86400.0])
    parts.append((sgn(m) * mag(-5, 5, m), lit[rng.integers(0, len(lit), m)]))
    a = np.concatenate([p[0] for p in parts])
    b = np.concatenate([p[1] for p in parts])
    assert len(a) == n and np.all(b != 0) and np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    return a, b


def write_inputs(rundir, size, seed=20261001):
    rundir = Path(rundir)
    f, scal, one_d = make_inputs(size, seed)
    p_in, p_dv = rundir / "replay_in.bin", rundir / "divide_in.bin"
    for p in (p_in, p_dv):
        if p.exists():
            raise SystemExit(f"{p} exists (nothing is overwritten)")
    hdr = [MAGIC, VERSION] + [size[k] for k in SIZE_KEYS] + [len(IN3)]
    with open(p_in, "xb") as fh:
        fh.write(np.array(hdr, ">i4").tobytes())
        fh.write(np.array([scal[k] for k in ("deltaTloc",) + PARAMS], ">f8").tobytes())
        fh.write(one_d["recip_deepFacC"].astype(">f8").tobytes())
        for name in ("cosFacU", "cosFacV"):
            fh.write(_tile_to_fortran(one_d[name], size).astype(">f8").tobytes())
        for name in IN3:
            fh.write(_tile_to_fortran(f[name], size).astype(">f8").tobytes())
    a, b = make_divide_pairs(seed)
    with open(p_dv, "xb") as fh:
        fh.write(np.array([MAGIC, VERSION, len(a)], ">i4").tobytes())
        fh.write(np.stack([a, b], axis=1).astype(">f8").tobytes())
    meta = {"seed": seed, "size": size, "IN3": IN3, "OUT3": OUT3,
            "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (p_in, p_dv)}}
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


def read_inputs(rundir):
    """(size, fields {name: [tile,k,j,i]}, scalars {name: float}, one_d {recip_deepFacC: [k], cosFacU: [tile,j],
    cosFacV})."""
    r = _Reader(Path(rundir) / "replay_in.bin")
    hdr = r.take(">i4", 10)
    if hdr[0] != MAGIC or hdr[1] != VERSION or hdr[9] != len(IN3):
        raise ValueError(f"bad replay_in.bin header {hdr}")
    size = dict(zip(SIZE_KEYS, (int(x) for x in hdr[2:9])))
    T = size["nSx"] * size["nSy"]
    Ny, Nx = size["sNy"] + 2 * size["OLy"], size["sNx"] + 2 * size["OLx"]
    s = r.take(">f8", 5)
    scal = dict(zip(("deltaTloc",) + PARAMS, (float(x) for x in s)))
    one_d = {"recip_deepFacC": r.take(">f8", size["Nr"])}
    for name in ("cosFacU", "cosFacV"):
        one_d[name] = _fortran_to_tile(r.take(">f8", T * Ny).reshape(size["nSy"], size["nSx"], Ny), size)
    fields = {}
    for name in IN3:
        a = r.take(">f8", T * size["Nr"] * Ny * Nx).reshape(size["nSy"], size["nSx"], size["Nr"], Ny, Nx)
        fields[name] = _fortran_to_tile(a, size)
    r.done()
    return size, fields, scal, one_d


def read_outputs(rundir, size):
    r = _Reader(Path(rundir) / "replay_out.bin")
    hdr = r.take(">i4", 4)
    if hdr[0] != MAGIC or hdr[1] != VERSION or hdr[2] != len(OUT3) or hdr[3] != size["Nr"]:
        raise ValueError(f"bad replay_out.bin header {hdr}")
    T = size["nSx"] * size["nSy"]
    Ny, Nx = size["sNy"] + 2 * size["OLy"], size["sNx"] + 2 * size["OLx"]
    out = {}
    for name in OUT3:
        a = r.take(">f8", T * size["Nr"] * Ny * Nx).reshape(size["nSy"], size["nSx"], size["Nr"], Ny, Nx)
        out[name] = _fortran_to_tile(a, size)
    r.done()
    return out


def read_grid(rundir, size):
    """{name: array} with GRID3 [tile,k,j,i], GRID2 [tile,j,i], cosFacU/cosFacV [tile,j], recip_deepFacC [k] and
    the PARAMS as floats, as the routines read them (after the harness overrides)."""
    r = _Reader(Path(rundir) / "replay_grid.bin")
    hdr = r.take(">i4", 4)
    if hdr[0] != MAGIC or hdr[1] != VERSION or hdr[2] != len(GRID3) or hdr[3] != len(GRID2):
        raise ValueError(f"bad replay_grid.bin header {hdr}")
    T = size["nSx"] * size["nSy"]
    Nr, Ny, Nx = size["Nr"], size["sNy"] + 2 * size["OLy"], size["sNx"] + 2 * size["OLx"]
    g = {}
    for name in GRID3:
        g[name] = _fortran_to_tile(r.take(">f8", T * Nr * Ny * Nx).reshape(size["nSy"], size["nSx"], Nr, Ny, Nx),
                                   size)
    for name in GRID2:
        g[name] = _fortran_to_tile(r.take(">f8", T * Ny * Nx).reshape(size["nSy"], size["nSx"], Ny, Nx), size)
    for name in ("cosFacU", "cosFacV"):
        g[name] = _fortran_to_tile(r.take(">f8", T * Ny).reshape(size["nSy"], size["nSx"], Ny), size)
    g["recip_deepFacC"] = r.take(">f8", Nr)
    for name, v in zip(PARAMS, r.take(">f8", len(PARAMS))):
        g[name] = float(v)
    r.done()
    return g


def read_divide(rundir):
    """(a, b, q) with q[:, 0] = a/b and q[:, 1:] = a/DIVIDE_LITERALS, as gfortran computed them."""
    r = _Reader(Path(rundir) / "divide_in.bin")
    hdr = r.take(">i4", 3)
    if hdr[0] != MAGIC or hdr[1] != VERSION:
        raise ValueError(f"bad divide_in.bin header {hdr}")
    ab = r.take(">f8", 2 * int(hdr[2])).reshape(-1, 2)
    r.done()
    r = _Reader(Path(rundir) / "divide_out.bin")
    hdr = r.take(">i4", 3)
    if hdr[0] != MAGIC or hdr[1] != VERSION or hdr[2] != 1 + len(DIVIDE_LITERALS):
        raise ValueError(f"bad divide_out.bin header {hdr}")
    q = r.take(">f8", len(ab) * int(hdr[2])).reshape(len(ab), int(hdr[2]))
    r.done()
    return ab[:, 0], ab[:, 1], q


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write-inputs")
    w.add_argument("rundir")
    w.add_argument("--size", required=True, help="the build's SIZE.h")
    w.add_argument("--seed", type=int, default=20261001)
    a = ap.parse_args(argv)
    meta = write_inputs(a.rundir, parse_size_h(a.size), a.seed)
    print(f"replay inputs written: {json.dumps(meta['size'])} seed {meta['seed']} {meta['sha256']}")


if __name__ == "__main__":
    sys.exit(main())
