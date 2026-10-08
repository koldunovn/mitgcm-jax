#!/usr/bin/env python3
"""Inputs and outputs of the GAD-B replay harness (the SOM advection path; reference/replay_gad_b/code/the_main_loop.F).

    replay_io.py write-inputs RUNDIR --size SIZE.h [--seed N]     (run by reference/replay_gad_b/jobs/replay.sbatch)

A copy of lane C's reference/replay (plan Task 8) for M1 sub-lane GAD-B: a genmake2 build of advect_xy/code or
advect_xz/code in which THE_MAIN_LOOP is replaced. THE_MODEL_MAIN runs INITIALISE_FIXED, so GRID.h holds the
experiment's real grid and GAD.h the run's SOM switches; the replacement reads the synthetic inputs written here, calls
GAD_SOM_ADVECT (schemes 80 and 81), GAD_SOM_ADV_X/_Y (limiter 0 and 1), GAD_SOM_LIM_R (limiter 1), GAD_SOM_EXCHANGES
and GAD_EXCH_SOM, and writes every output array (all points incl. halos) and the grid as the routines read it.

Velocities of GAD_SOM_ADVECT are given as CFL numbers and turned into velocities by the harness on the real grid
(uFld = cflU*dxC/dTlev(k), vFld = cflV*dyC/dTlev(k), wFld = cflW*MIN(drF(k)*hFacC(k), drF(k-1)*hFacC(k-1))/dTlev(k)),
the moments as ratios times the real cell volume rA*drF*hFacC; the harness writes the values it built (out3 1..12),
which are the inputs the JAX side uses. |CFL| <= 0.15 keeps every cell volume positive through the three passes.
The leaf routines get fully synthetic volumes, contents, moments and transports (no grid).

Files (all big-endian stream; Fortran column-major, so A(i,j,k,bi,bj) is the numpy C-order array [bj, bi, k, j, i] and
the tile index is t = (bj-1)*nSx + (bi-1), bi fastest):
    replay_in.bin    int32 MAGIC, VERSION, sNx, sNy, OLx, OLy, nSx, nSy, Nr, len(IN3);
                     float64 dTlev(Nr); deepFacC, deepFac2C, recip_deepFac2C, rhoFacC, recip_rhoFacC (Nr),
                     deepFac2F, rhoFacF (Nr+1); IN3 fields (1-OLx:sNx+OLx, 1-OLy:sNy+OLy, Nr, nSx, nSy)
    replay_out.bin   int32 HEADER (MAGIC, VERSION, len(OUT3), Nr, the run's switches); OUT3 fields
    replay_grid.bin  int32 MAGIC, VERSION, 7, 5; GRID2, GRID3, GRID1 (Nr), GRID1F (Nr+1)

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

# moments n = 1..nSOM in the order of GAD_SOM_VARS.h:219-220 (1rst order x,y,z; 2nd order xx,yy,zz,xy,xz,yz)
SOM = ("x", "y", "z", "xx", "yy", "zz", "xy", "xz", "yz")
# the 11 U arguments of GAD_SOM_ADV_X/_Y/_R and GAD_SOM_LIM_R, in their argument order
SM11 = ("v", "o") + SOM

# the order of the in3(..., f) fields in the_main_loop.F
IN3 = (("tracer", "cflU", "cflV", "cflW") + tuple(f"ratio_{m}" for m in SOM)
       + ("uTransX", "vTransY") + tuple(f"sm_{m}" for m in SM11) + ("fluxPrior",)
       + tuple(f"som_T_{m}" for m in SOM) + tuple(f"som_S_{m}" for m in SOM) + tuple(f"smTr_{m}" for m in SOM))
OUT3 = (("uFld", "vFld", "wFld") + tuple(f"smTr0_{m}" for m in SOM)
        + ("gTracer_80",) + tuple(f"smTr80_{m}" for m in SOM)
        + ("gTracer_81",) + tuple(f"smTr81_{m}" for m in SOM)
        + tuple(f"advx_l0_{m}" for m in SM11) + ("advx_l0_uT",)
        + tuple(f"advx_l1_{m}" for m in SM11) + ("advx_l1_uT",)
        + tuple(f"advy_l0_{m}" for m in SM11) + ("advy_l0_vT",)
        + tuple(f"advy_l1_{m}" for m in SM11) + ("advy_l1_vT",)
        + tuple(f"limr_{m}" for m in SM11)
        + tuple(f"som_T_out_{m}" for m in SOM) + tuple(f"som_S_out_{m}" for m in SOM)
        + tuple(f"exch_{m}" for m in SOM))
HEADER = ("MAGIC", "VERSION", "nOut3", "Nr", "tempSOM_Advection", "saltSOM_Advection", "rigidLid", "nonlinFreeSurf",
          "select_rStar", "useCubedSphereExchange", "uniformFreeSurfLev", "tempAdvScheme", "saltAdvScheme",
          "tempVertAdvScheme", "saltVertAdvScheme")
FACTORS = ("deepFacC", "deepFac2C", "recip_deepFac2C", "rhoFacC", "recip_rhoFacC")      # (Nr)
FACTORS_F = ("deepFac2F", "rhoFacF")                                                    # (Nr+1)
GRID2 = ("dxC", "dyC", "dxG", "dyG", "rA", "recip_rA", "maskInC")
GRID3 = ("hFacC", "hFacW", "hFacS", "recip_hFacC", "maskC")
GRID1 = ("drF", "recip_drF") + FACTORS
SIZE_KEYS = ("sNx", "sNy", "OLx", "OLy", "nSx", "nSy", "Nr")
CFL_MAX = 0.15
assert len(IN3) == 54 and len(OUT3) == 118 and len(HEADER) == 15


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


def _upwind_min(v, axis):
    """min(v at the point, v at the point upstream along `axis`) (the point itself at the first index)."""
    prev = np.concatenate([np.take(v, [0], axis=axis), np.take(v, range(v.shape[axis] - 1), axis=axis)], axis=axis)
    return np.minimum(v, prev)


def make_inputs(size, seed=20261002):
    """Synthetic inputs, storage [tile, k, j, i] incl. halos (see the module docstring). Tracer values 10 +- 5 (about
    2 % negative: the limiter's sm_o <= 0 branch); moment ratios 8*N(0,1) (|S_x| beyond the limiter's 1.5*S_0 about a
    third of the time); leaf contents with 2 % exact zeros (the limiter's tie sm_o = 0)."""
    rng = np.random.default_rng(seed)
    T = size["nSx"] * size["nSy"]
    Nr = size["Nr"]
    Ny, Nx = size["sNy"] + 2 * size["OLy"], size["sNx"] + 2 * size["OLx"]
    shp = (T, Nr, Ny, Nx)
    n = lambda scale: scale * rng.standard_normal(shp)
    u = lambda lo, hi: rng.uniform(lo, hi, shp)
    dTlev = 1200.0 + 50.0 * rng.uniform(0.0, 1.0, Nr)
    f = {"tracer": 10.0 + n(5.0), "cflU": u(-CFL_MAX, CFL_MAX), "cflV": u(-CFL_MAX, CFL_MAX),
         "cflW": u(-CFL_MAX, CFL_MAX)}
    for m in SOM:
        f[f"ratio_{m}"] = n(8.0)
    sm_v = u(0.5, 1.5) * 1.0e9
    dt4 = dTlev[None, :, None, None]
    f["uTransX"] = u(-CFL_MAX, CFL_MAX) * _upwind_min(sm_v, 3) / dt4
    f["vTransY"] = u(-CFL_MAX, CFL_MAX) * _upwind_min(sm_v, 2) / dt4
    sm_o = (10.0 + n(5.0)) * sm_v
    sm_o[rng.uniform(size=shp) < 0.02] = 0.0
    f["sm_v"], f["sm_o"] = sm_v, sm_o
    for m in SOM:
        f[f"sm_{m}"] = n(8.0) * sm_v
    f["fluxPrior"] = n(1.0e6) - 3.0e6
    for pre, scale in (("som_T", 1.0), ("som_S", 2.0), ("smTr", 3.0)):
        for m in SOM:
            f[f"{pre}_{m}"] = n(scale)
    assert tuple(f) == IN3
    one_d = {k: 1.0 + 0.01 * rng.uniform(0.5, 1.5, Nr) for k in FACTORS}
    one_d.update({k: 1.0 + 0.01 * rng.uniform(0.5, 1.5, Nr + 1) for k in FACTORS_F})
    one_d["dTlev"] = dTlev
    return f, one_d


def write_inputs(rundir, size, seed=20261002):
    rundir = Path(rundir)
    f, one_d = make_inputs(size, seed)
    p_in = rundir / "replay_in.bin"
    if p_in.exists():
        raise SystemExit(f"{p_in} exists (nothing is overwritten)")
    hdr = [MAGIC, VERSION] + [size[k] for k in SIZE_KEYS] + [len(IN3)]
    with open(p_in, "xb") as fh:
        fh.write(np.array(hdr, ">i4").tobytes())
        fh.write(one_d["dTlev"].astype(">f8").tobytes())
        for name in FACTORS + FACTORS_F:
            fh.write(one_d[name].astype(">f8").tobytes())
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
    return size["nSx"] * size["nSy"], size["Nr"], size["sNy"] + 2 * size["OLy"], size["sNx"] + 2 * size["OLx"]


def _fields(r, names, size, nk):
    T, _, Ny, Nx = _dims(size)
    out = {}
    for name in names:
        a = r.take(">f8", T * nk * Ny * Nx).reshape(size["nSy"], size["nSx"], nk, Ny, Nx)
        out[name] = _fortran_to_tile(a, size)
    return out


def read_inputs(rundir):
    """(size, fields {name: [tile,k,j,i]}, one_d {dTlev, FACTORS: [Nr], FACTORS_F: [Nr+1]})."""
    r = _Reader(Path(rundir) / "replay_in.bin")
    hdr = r.take(">i4", 10)
    if hdr[0] != MAGIC or hdr[1] != VERSION or hdr[9] != len(IN3):
        raise ValueError(f"bad replay_in.bin header {hdr}")
    size = dict(zip(SIZE_KEYS, (int(x) for x in hdr[2:9])))
    Nr = size["Nr"]
    one_d = {"dTlev": r.take(">f8", Nr)}
    for name in FACTORS:
        one_d[name] = r.take(">f8", Nr)
    for name in FACTORS_F:
        one_d[name] = r.take(">f8", Nr + 1)
    fields = _fields(r, IN3, size, Nr)
    r.done()
    return size, fields, one_d


def read_outputs(rundir, size):
    """(header {name: int}, outputs {name: [tile,k,j,i]})."""
    r = _Reader(Path(rundir) / "replay_out.bin")
    hdr = dict(zip(HEADER, (int(x) for x in r.take(">i4", len(HEADER)))))
    if (hdr["MAGIC"] != MAGIC or hdr["VERSION"] != VERSION or hdr["nOut3"] != len(OUT3)
            or hdr["Nr"] != size["Nr"]):
        raise ValueError(f"bad replay_out.bin header {hdr}")
    out = _fields(r, OUT3, size, size["Nr"])
    r.done()
    return hdr, out


def read_grid(rundir, size):
    """{name: array}: GRID2 [tile,j,i], GRID3 [tile,k,j,i], GRID1 [Nr], deepFac2F/rhoFacF [Nr+1], as the routines read
    them (after the harness overrides)."""
    r = _Reader(Path(rundir) / "replay_grid.bin")
    hdr = r.take(">i4", 4)
    if hdr[0] != MAGIC or hdr[1] != VERSION or hdr[2] != len(GRID2) or hdr[3] != len(GRID3):
        raise ValueError(f"bad replay_grid.bin header {hdr}")
    T, Nr, Ny, Nx = _dims(size)
    g = {k: v[:, 0] for k, v in _fields(r, GRID2, size, 1).items()}
    g.update(_fields(r, GRID3, size, Nr))
    for name in GRID1:
        g[name] = r.take(">f8", Nr)
    for name in FACTORS_F:
        g[name] = r.take(">f8", Nr + 1)
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
