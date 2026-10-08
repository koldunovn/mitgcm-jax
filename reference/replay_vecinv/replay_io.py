#!/usr/bin/env python3
"""Inputs and outputs of the VECINV replay harness (M2 sub-lane VECINV, plan Task 23;
reference/replay_vecinv/code/the_main_loop.F, a copy of reference/replay_mom).

    replay_io.py write-inputs RUNDIR --size SIZE.h [--seed N]     (run by reference/replay_vecinv/jobs/replay.sbatch)

The harness is a genmake2 build of a verification experiment's code directory in which THE_MAIN_LOOP is replaced:
THE_MODEL_MAIN runs INITIALISE_FIXED, so GRID.h holds the experiment's real grid (land, partial cells, cube corners,
exchanged halos) and MOM_VISC.h the viscosity lengths of MOM_INIT_FIXED; the replacement reads the synthetic inputs
written here, overrides the grid factors listed below, writes the grid and the run's PARAMS.h values (replay_grid.bin),
calls MOM_VECINV per tile and level with the run's own parameters (m0) and with the vorticity selectors flipped (m1),
then the leaf routines of pkg/mom_vecinv and pkg/mom_common with the selectors set per case (OUT3), and writes every
output (all points, incl. the ones a routine does not write: they keep the prior field given here).

Files (all big-endian stream; Fortran column-major, so a Fortran array A(i,j,k,bi,bj) is the numpy C-order array
[bj, bi, k, j, i], and the tile index is t = (bj-1)*nSx + (bi-1), bi fastest):
    replay_in.bin    int32 MAGIC, VERSION, sNx, sNy, OLx, OLy, nSx, nSy, Nr, len(IN3);
                     float64 SPAR; VPAR (20 x 4, Fortran order: parameter fastest); R_IN (Nr each); RP1_IN (Nr+1
                     each); Y_IN (1-OLy:sNy+OLy, nSx, nSy); XY_IN (1-OLx:sNx+OLx, 1-OLy:sNy+OLy, nSx, nSy);
                     IN3 fields (..., Nr, nSx, nSy)
    replay_out.bin   int32 MAGIC, VERSION, len(OUT3), Nr; OUT3 fields
    replay_grid.bin  int32 MAGIC, VERSION, len(GRID3), len(GRID2); GRID3; GRID2; GRIDY; GRIDR; GRIDRP1; SCALARS;
                     int32 FLAGS  -- as MOM_VECINV (m0) reads them, after the overrides
    replay_w2.bin    int32 MAGIC, VERSION, nTiles (0 without ALLOW_EXCH2); per tile (bi fastest) W2_TOPO:
                     myTile = W2_myTileList(bi,bj) and exch2_myFace, exch2_isWedge/Eedge/Sedge/Nedge of myTile

Numpy only (no jax): it runs in the batch job before the model.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

MAGIC, VERSION = 20261003, 2

# the order of the in3(..., f) fields in the_main_loop.F
IN3 = ("uFld", "vFld", "wFld", "hDiv", "vort3", "dStar", "zStar", "tension", "strain", "KE",
       "viscAh_Z", "viscAh_D", "viscA4_Z", "viscA4_D", "omega3", "kappaRU", "kappaRV", "kappaRU_Nr1", "kappaRV_Nr1",
       "fVerUkm", "fVerVkm", "prior")
# the_main_loop.F: spar(1..4) -> viscAhD, viscAhZ, viscA4D, viscA4Z of the leaf cases after MOM_CALC_VISC
SPAR = ("viscAhD", "viscAhZ", "viscA4D", "viscA4Z")
# VIRP_SETVISC: vpar(1..20, case)
VPAR = ("viscAhD", "viscAhZ", "viscA4D", "viscA4Z", "viscAhGrid", "viscA4Grid", "viscAhMax", "viscA4Max",
        "viscAhGridMax", "viscAhGridMin", "viscA4GridMax", "viscA4GridMin", "viscAhReMax", "viscA4ReMax",
        "viscC2leith", "viscC2leithD", "viscC4leith", "viscC4leithD", "viscC2smag", "viscC4smag")
# MOM_CALC_VISC cases v1..v4: DATA vFul, vHar, vBih (useFullLeith, useHarmonicVisc, useBiharmonicVisc)
VCASES = ((False, False, True), (True, True, True), (False, True, True), (False, True, False))
VPAR_VALUES = (
    # v1: the global_ocean.90x40x15/input_ad selection (data: viscC4Leith=1.5, viscC4Leithd=1.5, viscA4GridMax=0.5;
    #     everything else at its set_defaults.F value)
    (0., 0., 0., 0., 0., 0., 1.e21, 1.e21, 1.e21, 0., 0.5, 0., 0., 0., 0., 0., 1.5, 1.5, 0., 0.),
    # v2-v4: magnitudes set from the measured quantiles of each term on 90x40x15 (L2 ~ 1e10-1e11 m^2, deltaTMom =
    #     1200 s, the synthetic fields): Alin, the Reynolds-number limits and the four MIN/MAX bounds are comparable, so
    #     every limiter wins on a fraction of the points only (clipping everywhere would hide the Leith/Smagorinsky
    #     terms: the first values clipped viscAh/viscA4 to viscAhMax/viscA4Max on 81-93 % of the points)
    # v2: full Leith, harmonic + biharmonic, grid-scale terms, bounds and Reynolds-number limits
    (150., 120., 1.1e11, 0.9e11, 1.e-4, 1.e-5, 4.e4, 3.e13, 2.e-3, 3.e-4, 1.e-4, 1.e-5, 10., 300., 1.1, 0.9, 0.3,
     0.25, 0., 0.),
    # v3: Smagorinsky (harmonic + biharmonic) with Reynolds-number limits, no Leith
    (150., 120., 1.1e11, 0.9e11, 1.e-4, 1.e-5, 4.e4, 3.e13, 2.e-3, 3.e-4, 1.e-4, 1.e-5, 10., 300., 0., 0., 0., 0.,
     0.3, 0.05),
    # v4: Leith (not full) + Smagorinsky, harmonic only
    (150., 120., 1.1e11, 0.9e11, 1.e-4, 1.e-5, 4.e4, 3.e13, 2.e-3, 3.e-4, 1.e-4, 1.e-5, 0., 0., 1.1, 0.9, 0.3, 0.25,
     0.3, 0.05),
)
R_IN = ("recip_deepFacC", "recip_deepFac2C", "deepFacC", "deepFac2C", "recip_rhoFacC", "deepFacA")
RP1_IN = ("deepFac2F", "rhoFacF", "rVel2wUnit")
Y_IN = ("cosFacU", "cosFacV")
XY_IN = ("fCoriCos", "angleCosC", "angleSinC")

# MOM_VI_HDISSIP cases h0..h6: (harmonic, biharmonic, useVariableViscosity)
HCASES = ((True, True, True), (True, True, False), (True, False, True), (True, False, False),
          (False, True, True), (False, True, False), (False, False, False))
# MOM_VI_U/V_CORIOLIS cases c0..c5: (selectVortScheme, useJamartMomAdv)
CCASES = ((0, False), (1, False), (2, False), (3, False), (4, False), (1, True))
# MOM_VI_U/V_VERTSHEAR cases s0..s3: (selectKEscheme, upwindShear)
SCASES = ((0, False), (0, True), (1, False), (3, True))
# MOM_VECINV m1: the selectors set over the run's own (the_main_loop.F 3a)
M1_OVERRIDES = dict(useAbsVorticity=True, useCDscheme=False, upwindShear=True, selectVortScheme=2,
                    useJamartMomAdv=True)

# outputs, in the order the_main_loop.F writes them (iO = 1..91)
OUT3 = (("hFacZ", "r_hFacZ", "hDiv_s1", "hDiv_s2", "vort3")
        + tuple(f"omega3_c{c}" for c in range(3))
        + ("tension", "strain")
        + tuple(f"{n}_v{v}" for v in range(1, 5) for n in ("viscAh_Z", "viscAh_D", "viscA4_Z", "viscA4_D", "hDiv"))
        + ("del2u", "del2v", "hDiv_del2")
        + tuple(f"{n}_h{h}" for h in range(7) for n in ("uDissip", "vDissip"))
        + tuple(f"{n}_cs{s}" for s in range(4) for n in ("uCf", "vCf"))
        + tuple(f"uVort_c{c}" for c in range(6)) + tuple(f"vVort_c{c}" for c in range(6))
        + tuple(f"uShear_s{c}" for c in range(4)) + tuple(f"vShear_s{c}" for c in range(4))
        + ("dKEdx", "dKEdy", "vCfNH_s1", "vCfNH_s2")
        + tuple(f"{n}_m{m}" for m in range(2) for n in ("fVerUkp", "fVerVkp", "guDiss", "gvDiss", "gU", "gV")))
assert len(OUT3) == 91 and len(set(OUT3)) == 91

GRID3 = ("hFacC", "hFacW", "hFacS", "recip_hFacC", "recip_hFacW", "recip_hFacS", "maskC", "maskW", "maskS",
         "h0FacW", "h0FacS")
GRID2 = ("rA", "rAw", "rAs", "rAz", "recip_rA", "recip_rAw", "recip_rAs", "recip_rAz", "dxC", "dyC", "dxG", "dyG",
         "dxV", "dyU", "recip_dxC", "recip_dyC", "recip_dxG", "recip_dyG", "recip_dxV", "recip_dyU", "recip_dxF",
         "recip_dyF", "fCoriG", "fCoriCos", "angleCosC", "angleSinC", "L2_D", "L2_Z", "L3_D", "L3_Z", "L4rdt_D", "L4rdt_Z")
GRIDY = ("cosFacU", "cosFacV", "sqCosFacU", "sqCosFacV")
GRIDR = ("drF", "recip_drF", "recip_deepFacC", "recip_deepFac2C", "deepFacC", "deepFac2C", "recip_rhoFacC",
         "deepFacA", "deepFacAdv")
GRIDRP1 = ("recip_drC", "deepFac2F", "rhoFacF", "rVel2wUnit")
SCALARS = ("rkSign", "gravitySign", "pi", "recip_rSphere", "deltaTMom", "deltaTClock", "diagFreq", "vfFacMom",
           "sideDragFactor", "bottomDragLinear", "bottomDragQuadratic", "rUnit2mass",
           "viscAh", "viscAhD", "viscAhZ", "viscAhGrid", "viscAhMax", "viscAhGridMax", "viscAhGridMin", "viscAhReMax",
           "viscA4", "viscA4D", "viscA4Z", "viscA4Grid", "viscA4Max", "viscA4GridMax", "viscA4GridMin", "viscA4ReMax",
           "viscC2leith", "viscC2leithD", "viscC2LeithQG", "viscC4leith", "viscC4leithD", "viscC2smag", "viscC4smag",
           "viscAhW", "viscA4W")
FLAGS = ("momViscosity", "momAdvection", "useCoriolis", "useCDscheme", "useAbsVorticity", "highOrderVorticity",
         "upwindVorticity", "selectVortScheme", "useJamartMomAdv", "upwindShear", "selectKEscheme", "momImplVertAdv",
         "implicitViscosity", "no_slip_sides", "no_slip_bottom", "selectBotDragQuadr", "bottomVisc_pCell",
         "selectImplicitDrag", "select3dCoriScheme", "useNHMTerms", "deepAtmosphere", "usingCurvilinearGrid",
         "rotateGrid", "useStrainTensionVisc", "useHarmonicVisc", "useBiharmonicVisc", "useVariableVisc",
         "useFullLeith", "nonlinFreeSurf", "selectCoriScheme", "useCubedSphereExchange", "usingZCoords",
         "useAreaViscLength")
INT_FLAGS = ("selectVortScheme", "selectKEscheme", "selectBotDragQuadr", "selectImplicitDrag", "select3dCoriScheme",
             "nonlinFreeSurf", "selectCoriScheme")
assert len(FLAGS) == 33 and len(GRID2) == 32
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


def make_inputs(size, seed=20261003):
    """Synthetic inputs: random binary64 values at physical magnitudes (every rounding counts), storage
    [tile, k, j, i] incl. halos. KE has exact zeros (the KE > 0 branches of MOM_CALC_VISC); the 1-D and 2-D factors
    that are 1 or 0 in the experiments get synthetic values (a missing factor must show)."""
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

    f = {"uFld": n(0.3), "vFld": n(0.3), "wFld": n(1.0e-4), "hDiv": n(3.0e-6), "vort3": n(3.0e-6),
         "dStar": n(1.0e-16), "zStar": n(1.0e-16), "tension": n(3.0e-6), "strain": n(3.0e-6),
         "KE": with_zeros(np.abs(n(0.05))),
         "viscAh_Z": u(1.0e3, 1.0e4), "viscAh_D": u(1.0e3, 1.0e4), "viscA4_Z": u(1.0e10, 1.0e12),
         "viscA4_D": u(1.0e10, 1.0e12), "omega3": n(1.0e-4),
         "kappaRU": u(1.0e-5, 1.0e-2), "kappaRV": u(1.0e-5, 1.0e-2),
         "kappaRU_Nr1": u(1.0e-5, 1.0e-2), "kappaRV_Nr1": u(1.0e-5, 1.0e-2),
         "fVerUkm": n(1.0e4), "fVerVkm": n(1.0e4), "prior": n(1.0) + 3.0}
    assert tuple(f) == IN3
    spar = {"viscAhD": 2.1e3, "viscAhZ": 1.7e3, "viscA4D": 3.3e11, "viscA4Z": 2.9e11}
    assert tuple(spar) == SPAR
    vpar = np.array(VPAR_VALUES, np.float64)          # [case, parameter]
    assert vpar.shape == (len(VCASES), len(VPAR))
    one_d = {}
    for name in R_IN:
        one_d[name] = 1.0 + 0.01 * rng.uniform(0.5, 1.5, Nr)
    for name in RP1_IN:
        one_d[name] = 1.0 + 0.01 * rng.uniform(0.5, 1.5, Nr + 1)
    for name in Y_IN:
        one_d[name] = rng.uniform(0.2, 1.0, (T, Ny))
    xy = {"fCoriCos": 1.0e-4 * rng.uniform(-1.0, 1.0, (T, Ny, Nx)), "angleCosC": rng.uniform(0.5, 1.0, (T, Ny, Nx)),
          "angleSinC": rng.uniform(-0.5, 0.5, (T, Ny, Nx))}
    assert tuple(xy) == XY_IN
    return f, spar, vpar, one_d, xy


def write_inputs(rundir, size, seed=20261003):
    rundir = Path(rundir)
    f, spar, vpar, one_d, xy = make_inputs(size, seed)
    p_in = rundir / "replay_in.bin"
    if p_in.exists():
        raise SystemExit(f"{p_in} exists (nothing is overwritten)")
    hdr = [MAGIC, VERSION] + [size[k] for k in SIZE_KEYS] + [len(IN3)]
    with open(p_in, "xb") as fh:
        fh.write(np.array(hdr, ">i4").tobytes())
        fh.write(np.array([spar[k] for k in SPAR], ">f8").tobytes())
        fh.write(vpar.astype(">f8").tobytes())       # C order [case, parameter] = Fortran vpar(parameter, case)
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
    """(size, fields {name: [tile,k,j,i]}, spar {name: float}, vpar [case, parameter], one_d {name: [k] or
    [tile, j]}, xy {name: [tile,j,i]})."""
    r = _Reader(Path(rundir) / "replay_in.bin")
    hdr = r.take(">i4", 10)
    if hdr[0] != MAGIC or hdr[1] != VERSION or hdr[9] != len(IN3):
        raise ValueError(f"bad replay_in.bin header {hdr}")
    size = dict(zip(SIZE_KEYS, (int(x) for x in hdr[2:9])))
    T, Nr, Ny, Nx = _dims(size)
    spar = dict(zip(SPAR, (float(x) for x in r.take(">f8", len(SPAR)))))
    vpar = r.take(">f8", len(VCASES) * len(VPAR)).reshape(len(VCASES), len(VPAR))
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
    return size, fields, spar, vpar, one_d, xy


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
    floats and FLAGS as bools (ints for INT_FLAGS), as MOM_VECINV (m0) reads them (after the harness overrides)."""
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
        g[name] = int(v) if name in INT_FLAGS else bool(v)
    r.done()
    return g


W2_TOPO = ("myTile", "exch2_myFace", "exch2_isWedge", "exch2_isEedge", "exch2_isSedge", "exch2_isNedge")


def read_w2(rundir):
    """{name: int array [tile]} in tile storage order (t = (bj-1)*nSx + bi-1), or None without ALLOW_EXCH2."""
    r = _Reader(Path(rundir) / "replay_w2.bin")
    hdr = r.take(">i4", 3)
    if hdr[0] != MAGIC or hdr[1] != VERSION:
        raise ValueError(f"bad replay_w2.bin header {hdr}")
    n = int(hdr[2])
    a = r.take(">i4", n * len(W2_TOPO)).reshape(n, len(W2_TOPO))
    r.done()
    return None if n == 0 else {name: a[:, c].astype(np.int64) for c, name in enumerate(W2_TOPO)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write-inputs")
    w.add_argument("rundir")
    w.add_argument("--size", required=True, help="the build's SIZE.h")
    w.add_argument("--seed", type=int, default=20261003)
    w.add_argument("--experiment", help="accepted for the replay sbatch interface; unused")
    a = ap.parse_args(argv)
    meta = write_inputs(a.rundir, parse_size_h(a.size), a.seed)
    print(f"replay inputs written: {json.dumps(meta['size'])} seed {meta['seed']} {meta['sha256']}")


if __name__ == "__main__":
    sys.exit(main())
