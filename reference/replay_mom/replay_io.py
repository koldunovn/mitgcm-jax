#!/usr/bin/env python3
"""Inputs and outputs of the MOM replay harness (M1 sub-lane MOM; reference/replay_mom/code/the_main_loop.F, a copy of
the Task 8 harness reference/replay).

    replay_io.py write-inputs RUNDIR --size SIZE.h [--seed N]     (run by reference/replay_mom/jobs/replay.sbatch)

The harness is a genmake2 build of a verification experiment's code directory in which THE_MAIN_LOOP is replaced:
THE_MODEL_MAIN runs INITIALISE_FIXED, so GRID.h holds the experiment's real grid (land, partial cells, exchanged
halos); the replacement reads the synthetic inputs written here, overrides the PARAMS.h/GRID.h values listed below,
calls the leaf kernels of pkg/mom_common and pkg/mom_fluxform for every tile and level with the branch selectors set
per case (OUT3), and writes every output (all points, incl. the ones a routine does not write: they keep the prior
field given here) and the grid and parameters as the routines read them.

Files (all big-endian stream; Fortran column-major, so a Fortran array A(i,j,k,bi,bj) is the numpy C-order array
[bj, bi, k, j, i], and the tile index is t = (bj-1)*nSx + (bi-1), bi fastest):
    replay_in.bin    int32 MAGIC, VERSION, sNx, sNy, OLx, OLy, nSx, nSy, Nr, len(IN3);
                     float64 SPAR; R_IN (Nr each); RP1_IN (Nr+1 each); Y_IN (1-OLy:sNy+OLy, nSx, nSy);
                     XY_IN (1-OLx:sNx+OLx, 1-OLy:sNy+OLy, nSx, nSy); IN3 fields (..., Nr, nSx, nSy)
    replay_out.bin   int32 MAGIC, VERSION, len(OUT3), Nr; OUT3 fields
    replay_grid.bin  int32 MAGIC, VERSION, len(GRID3), len(GRID2); GRID3; GRID2; GRIDY; GRIDR; GRIDRP1; SCALARS;
                     int32 FLAGS

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
IN3 = ("uFld", "vFld", "wFld", "uTrans", "vTrans", "rTrans", "hFacZ", "h0FacZ", "del2u", "del2v",
       "viscAh_D", "viscAh_Z", "viscA4_D", "viscA4_Z", "kappaRU", "kappaRV", "kappaRU_Nr1", "kappaRV_Nr1",
       "KEin", "prior")
# the_main_loop.F section 2: spar(1..12)
SPAR = ("sideDragFactor", "viscAh", "viscAhGrid", "viscAhMaxFrac", "viscA4", "viscA4Grid", "viscA4MaxFrac",
        "viscA4GridMax", "viscA4GridMin", "deltaTMom", "bottomDragLinear", "bottomDragQuadratic")
R_IN = ("recip_deepFacC", "recip_deepFac2C", "recip_gravFacC", "deepFacA")
RP1_IN = ("deepFac2F", "rhoFacF", "rVel2wUnit")
Y_IN = ("cosFacU", "cosFacV", "sqCosFacU", "sqCosFacV")
XY_IN = ("fCoriCos", "angleCosC", "angleSinC", "tanPhiAtU", "tanPhiAtV")

# outputs, in the order the_main_loop.F writes them (iO = 1..79)
OUT3 = (("hFacZ", "r_hFacZ")
        + tuple(f"KE_{s}" for s in ("m1", "0", "1", "2", "3"))
        + ("rTransU_k", "rTransV_k", "rTransU_kp1", "rTransV_kp1")
        + ("fZon_uu", "fMer_vu_mt0", "fMer_vu_mt3")
        + tuple(f"fVerU_c{c}" for c in range(4))
        + ("fZon_uv", "fMer_vv")
        + tuple(f"fVerV_c{c}" for c in range(4))
        + tuple(f"uCf_s{c}" for c in range(5)) + tuple(f"vCf_s{c}" for c in range(5))
        + ("uMT_sph1", "uMT_sph2", "vMT_sph1", "vMT_sph2")
        + ("xViscU", "yViscU", "xViscV", "yViscV")
        + ("rViscU_k", "rViscU_kp1", "rViscV_k", "rViscV_kp1")
        + ("del2u_ns0", "del2u_ns1", "del2v_ns0", "del2v_ns1")
        + tuple(f"uSD_c{c}" for c in range(3)) + tuple(f"vSD_c{c}" for c in range(3))
        + tuple(f"cDragU_b{c}" for c in range(7)) + ("KEU_b4",)
        + tuple(f"cDragV_b{c}" for c in range(7)) + ("KEV_b4",)
        + ("uCfNH_s1", "uCfNH_s2", "uMetNH", "vMetNH")
        + ("qhyd_c1", "qhyd_c2", "qhyd_c3"))
assert len(OUT3) == 79 and len(set(OUT3)) == 79

# MOM_U/V_ADV_WU/WV cases (MOMRP_WCASE): (k offset, rigidLid, select_rStar)
WCASES = ((0, False, 0), (0, True, 0), (0, False, 1), (1, False, 0))
# MOM_U/V_SIDEDRAG cases: (sideDragFactor from spar (True) or -1, viscA4GridMax from spar (True) or 0)
SDCASES = ((True, True), (False, True), (False, False))
# MOM_U/V_BOTDRAG_COEFF cases b0..b6: (selectBotDragQuadr, no_slip_bottom, bottomVisc_pCell, inp_KE)
BCASES = ((-1, True, False, True), (-1, True, True, True), (-1, False, False, True), (0, False, False, True),
          (0, False, False, False), (1, True, False, True), (2, True, False, True))
# MOM_QUASIHYDROSTATIC cases: (select3dCoriScheme, useNHMTerms)
QCASES = ((1, False), (0, True), (2, True))

GRID3 = ("hFacC", "hFacW", "hFacS", "recip_hFacC", "recip_hFacW", "recip_hFacS", "maskC", "maskW", "maskS",
         "h0FacW", "h0FacS")
GRID2 = ("rA", "rAw", "rAs", "recip_rAw", "recip_rAs", "recip_rAz", "dxC", "dxG", "dyG", "dxF", "dyF", "dxV", "dyU",
         "recip_dxC", "recip_dyC", "recip_dxF", "recip_dyF", "recip_dxV", "recip_dyU",
         "fCori", "fCoriCos", "angleCosC", "angleSinC", "tanPhiAtU", "tanPhiAtV", "recip_rA")
GRIDY = ("cosFacU", "cosFacV", "sqCosFacU", "sqCosFacV")
GRIDR = ("drF", "recip_drF", "recip_deepFacC", "recip_deepFac2C", "recip_gravFacC", "deepFacA")
GRIDRP1 = ("recip_drC", "deepFac2F", "rhoFacF", "rVel2wUnit")
SCALARS = ("rkSign", "gravitySign", "recip_rSphere", "rhoConst", "mass2rUnit", "recip_gravity", "sideDragFactor",
           "viscAh", "viscAhGrid", "viscAhMax", "viscA4", "viscA4Grid", "viscA4Max", "viscA4GridMax", "viscA4GridMin",
           "deltaTMom", "bottomDragLinear", "bottomDragQuadratic")
FLAGS = ("useRealFreshWaterFlux", "usingPCoords", "usingZCoords", "fluidIsWater", "NONLIN_FRSURF")
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
    """[tile, ...] -> [nSy, nSx, ...] (C order = Fortran (..., bi, bj))."""
    return a.reshape((size["nSy"], size["nSx"]) + a.shape[1:])


def _fortran_to_tile(a, size):
    return a.reshape((size["nSy"] * size["nSx"],) + a.shape[2:])


def make_inputs(size, seed=20261002):
    """Synthetic inputs: random binary64 values at physical magnitudes (every rounding counts), storage
    [tile, k, j, i] incl. halos. hFacZ, h0FacZ and KEin have exact zeros (the IF/where branches); the 1-D and 2-D
    factors that are 1 or 0 in the experiments get synthetic values (a missing factor must show)."""
    rng = np.random.default_rng(seed)
    T = size["nSx"] * size["nSy"]
    Nr = size["Nr"]
    Ny, Nx = size["sNy"] + 2 * size["OLy"], size["sNx"] + 2 * size["OLx"]
    shp = (T, Nr, Ny, Nx)
    n = lambda scale: scale * rng.standard_normal(shp)
    u = lambda lo, hi: rng.uniform(lo, hi, shp)

    def with_zeros(a, frac=0.2):
        a[rng.uniform(size=shp) < frac] = 0.0
        return a

    f = {"uFld": n(0.3), "vFld": n(0.3), "wFld": n(1.0e-4), "uTrans": n(2.0e6), "vTrans": n(2.0e6),
         "rTrans": n(1.0e6), "hFacZ": with_zeros(u(0.0, 1.0)), "h0FacZ": with_zeros(u(0.0, 1.0)),
         "del2u": n(1.0e-9), "del2v": n(1.0e-9),
         "viscAh_D": u(1.0e3, 1.0e4), "viscAh_Z": u(1.0e3, 1.0e4), "viscA4_D": u(1.0e10, 1.0e12),
         "viscA4_Z": u(1.0e10, 1.0e12), "kappaRU": u(1.0e-5, 1.0e-2), "kappaRV": u(1.0e-5, 1.0e-2),
         "kappaRU_Nr1": u(1.0e-5, 1.0e-2), "kappaRV_Nr1": u(1.0e-5, 1.0e-2),
         "KEin": with_zeros(np.abs(n(0.05))), "prior": n(1.0) + 3.0}
    assert tuple(f) == IN3
    spar = {"sideDragFactor": 1.75, "viscAh": 3.0e3, "viscAhGrid": 0.05, "viscAhMaxFrac": 0.5, "viscA4": 1.0e11,
            "viscA4Grid": 0.02, "viscA4MaxFrac": 0.3, "viscA4GridMax": 0.015, "viscA4GridMin": 0.004,
            "deltaTMom": 1800.0, "bottomDragLinear": 1.1e-3, "bottomDragQuadratic": 2.3e-3}
    assert tuple(spar) == SPAR
    one_d = {}
    for name in R_IN:
        one_d[name] = 1.0 + 0.01 * rng.uniform(0.5, 1.5, Nr)
    for name in RP1_IN:
        one_d[name] = 1.0 + 0.01 * rng.uniform(0.5, 1.5, Nr + 1)
    for name in Y_IN:
        one_d[name] = rng.uniform(0.2, 1.0, (T, Ny))
    xy = {"fCoriCos": 1.0e-4 * rng.uniform(-1.0, 1.0, (T, Ny, Nx)), "angleCosC": rng.uniform(0.5, 1.0, (T, Ny, Nx)),
          "angleSinC": rng.uniform(-0.5, 0.5, (T, Ny, Nx)), "tanPhiAtU": rng.uniform(-2.0, 2.0, (T, Ny, Nx)),
          "tanPhiAtV": rng.uniform(-2.0, 2.0, (T, Ny, Nx))}
    assert tuple(xy) == XY_IN
    return f, spar, one_d, xy


def write_inputs(rundir, size, seed=20261002):
    rundir = Path(rundir)
    f, spar, one_d, xy = make_inputs(size, seed)
    p_in = rundir / "replay_in.bin"
    if p_in.exists():
        raise SystemExit(f"{p_in} exists (nothing is overwritten)")
    hdr = [MAGIC, VERSION] + [size[k] for k in SIZE_KEYS] + [len(IN3)]
    with open(p_in, "xb") as fh:
        fh.write(np.array(hdr, ">i4").tobytes())
        fh.write(np.array([spar[k] for k in SPAR], ">f8").tobytes())
        for name in R_IN + RP1_IN:
            fh.write(one_d[name].astype(">f8").tobytes())
        for name in Y_IN:
            fh.write(_tile_to_fortran(one_d[name], size).astype(">f8").tobytes())
        for name in XY_IN:
            fh.write(_tile_to_fortran(xy[name], size).astype(">f8").tobytes())
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
    """(size, fields {name: [tile,k,j,i]}, spar {name: float}, one_d {name: [k] or [tile, j]},
    xy {name: [tile,j,i]})."""
    r = _Reader(Path(rundir) / "replay_in.bin")
    hdr = r.take(">i4", 10)
    if hdr[0] != MAGIC or hdr[1] != VERSION or hdr[9] != len(IN3):
        raise ValueError(f"bad replay_in.bin header {hdr}")
    size = dict(zip(SIZE_KEYS, (int(x) for x in hdr[2:9])))
    T, Nr, Ny, Nx = _dims(size)
    spar = dict(zip(SPAR, (float(x) for x in r.take(">f8", len(SPAR)))))
    one_d = {}
    for name in R_IN:
        one_d[name] = r.take(">f8", Nr)
    for name in RP1_IN:
        one_d[name] = r.take(">f8", Nr + 1)
    for name in Y_IN:
        one_d[name] = _fortran_to_tile(r.take(">f8", T * Ny).reshape(size["nSy"], size["nSx"], Ny), size)
    xy = {}
    for name in XY_IN:
        xy[name] = _fortran_to_tile(r.take(">f8", T * Ny * Nx).reshape(size["nSy"], size["nSx"], Ny, Nx), size)
    fields = {}
    for name in IN3:
        a = r.take(">f8", T * Nr * Ny * Nx).reshape(size["nSy"], size["nSx"], Nr, Ny, Nx)
        fields[name] = _fortran_to_tile(a, size)
    r.done()
    return size, fields, spar, one_d, xy


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
    """{name: array} with GRID3 [tile,k,j,i], GRID2 [tile,j,i], GRIDY [tile,j], GRIDR [k], GRIDRP1 [k], SCALARS as
    floats and FLAGS as bools, as the routines read them (after the harness overrides)."""
    r = _Reader(Path(rundir) / "replay_grid.bin")
    hdr = r.take(">i4", 4)
    if hdr[0] != MAGIC or hdr[1] != VERSION or hdr[2] != len(GRID3) or hdr[3] != len(GRID2):
        raise ValueError(f"bad replay_grid.bin header {hdr}")
    T, Nr, Ny, Nx = _dims(size)
    sy, sx = size["nSy"], size["nSx"]
    g = {}
    for name in GRID3:
        g[name] = _fortran_to_tile(r.take(">f8", T * Nr * Ny * Nx).reshape(sy, sx, Nr, Ny, Nx), size)
    for name in GRID2:
        g[name] = _fortran_to_tile(r.take(">f8", T * Ny * Nx).reshape(sy, sx, Ny, Nx), size)
    for name in GRIDY:
        g[name] = _fortran_to_tile(r.take(">f8", T * Ny).reshape(sy, sx, Ny), size)
    for name in GRIDR:
        g[name] = r.take(">f8", Nr)
    for name in GRIDRP1:
        g[name] = r.take(">f8", Nr + 1)
    for name, v in zip(SCALARS, r.take(">f8", len(SCALARS))):
        g[name] = float(v)
    for name, v in zip(FLAGS, r.take(">i4", len(FLAGS))):
        g[name] = bool(v)
    r.done()
    return g


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write-inputs")
    w.add_argument("rundir")
    w.add_argument("--size", required=True, help="the build's SIZE.h")
    w.add_argument("--seed", type=int, default=20261002)
    w.add_argument("--experiment", help="accepted for the GAD-A sbatch interface; unused")
    a = ap.parse_args(argv)
    meta = write_inputs(a.rundir, parse_size_h(a.size), a.seed)
    print(f"replay inputs written: {json.dumps(meta['size'])} seed {meta['seed']} {meta['sha256']}")


if __name__ == "__main__":
    sys.exit(main())
