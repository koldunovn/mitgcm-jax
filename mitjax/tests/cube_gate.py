"""Helpers of the cubed-sphere gates (test_cube.py; not a test file). Oracle: the gfortran cube replay harness
(reference/replay_cube: a genmake2 build of the experiment's code/ with THE_MAIN_LOOP replaced; its preprocessed
exch2, fill_cs_corner, W2 set-up and grid sources are byte-identical to lane A's plain oracle build), runs named in
reference/replay_cube/CURRENT. Each run directory holds the W2 print-out (w2_tile_topology.0000.log, output.txt),
cube_exch.bin (every exchange on index-coded and signed-zero fields) and cube_fill.bin (every FILL_CS_CORNER_*
option on every tile).
"""

import copy
import functools
import importlib.util

import jax.numpy as jnp
import numpy as np

from mitjax import paths
from mitjax.config.params import load
from mitjax.eesupp import exch_maps as EM
from mitjax.eesupp.exchange import Exchanger
from mitjax.eesupp.print import STANDARD_MESSAGE_UNIT
from mitjax.eesupp.tiles import TileLayout
from mitjax.pkg.exch2.w2_eeboot import w2_eeboot

EXPS = ("adjustment.cs-32x32x1", "solid-body.cs-32x32x1", "advect_cs", "global_ocean.cs32x15")
SCALAR_KINDS = ("XY", "3D", "Z", "S3D", "SMs", "SMn")
VECTOR_KINDS = ("As", "An", "Bs", "Bn", "UVs", "UVn", "UV3s", "UV3n", "Ds", "Dn")


def _rio():
    spec = importlib.util.spec_from_file_location("_mjx_replay_cube_io",
                                                  paths.REPO / "reference" / "replay_cube" / "replay_io.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def replay_dir(exp):
    for line in (paths.REPO / "reference" / "replay_cube" / "CURRENT").read_text().split("\n"):
        if line.strip() and line.split()[0] == exp:
            return paths.REFERENCE / line.split()[1]
    raise FileNotFoundError(f"{exp}: no cube replay run in reference/replay_cube/CURRENT")


@functools.lru_cache(maxsize=None)
def replay(exp, name):
    """The parsed cube_exch.bin / cube_fill.bin of the experiment's replay run."""
    return _rio().read(replay_dir(exp) / name, True)


@functools.lru_cache(maxsize=None)
def setup(exp):
    """(experiment, W2Common, MessageUnits) of our W2 set-up (callers that mutate deep-copy)."""
    e = load(exp, "input")
    w2, io = w2_eeboot(e)
    return e, w2, io


def printout_mismatches(exp, w2, io):
    """{'log': differing records, 'stdout': differing records, ...} vs the replay run, character for character."""
    rd = replay_dir(exp)
    log = (rd / "w2_tile_topology.0000.log").read_text().split("\n")
    assert log[-1] == ""
    log = log[:-1]
    ours = io.records(w2.W2_oUnit)
    nlog = abs(len(log) - len(ours)) + sum(a != b for a, b in zip(log, ours))
    out = (rd / "output.txt").read_text().split("\n")
    so = io.records(STANDARD_MESSAGE_UNIT)
    if so[0] not in out:
        return {"log": nlog, "stdout": len(so)}
    i0 = out.index(so[0])
    nout = sum(a != b for a, b in zip(out[i0:i0 + len(so)], so)) + max(0, i0 + len(so) - len(out))
    return {"log": nlog, "stdout": nout, "log_records": len(ours), "stdout_records": len(so)}


def layout(exp):
    e, w2, _ = setup(exp)
    sz = e.cfg.size
    return TileLayout(sz.sNx, sz.sNy, sz.OLx, sz.OLy, w2.exch2_nTiles)


def probe_inputs(exp, iz, cbase):
    """The harness's REPLAY_CUBE_PFILL fields [nTiles, ny, nx]: iz = 0 index codes, 1..4 signed zeros."""
    L = layout(exp)
    idx = (np.arange(L.npoints, dtype=np.float64) + 1.0).reshape(L.shape2d)
    if iz == 0:
        return cbase + idx, 2.0 * cbase + idx
    zu = -0.0 if iz in (2, 3) else 0.0
    zv = -0.0 if iz in (2, 4) else 0.0
    return np.full(L.shape2d, zu), np.full(L.shape2d, zv)


def bits(x):
    return np.asarray(x, np.float64).view(np.int64)


def call(ex, kind, u, v):
    """The Fortran-named exchange of `kind` (exch_maps names) on (u, v); scalar kinds use u only."""
    sc = {"XY": lambda: ex.EXCH_XY_RL(u), "3D": lambda: ex.EXCH_3D_RL(u), "Z": lambda: ex.EXCH_Z_3D_RL(u),
          "S3D": lambda: ex.EXCH_S3D_RL(u), "SMs": lambda: ex.EXCH_SM_3D_RL(u, True),
          "SMn": lambda: ex.EXCH_SM_3D_RL(u, False)}
    vc = {"As": lambda: ex.EXCH_UV_AGRID_3D_RL(u, v, True), "An": lambda: ex.EXCH_UV_AGRID_3D_RL(u, v, False),
          "Bs": lambda: ex.EXCH_UV_BGRID_3D_RL(u, v, True), "Bn": lambda: ex.EXCH_UV_BGRID_3D_RL(u, v, False),
          "UVs": lambda: ex.EXCH_UV_XY_RL(u, v, True), "UVn": lambda: ex.EXCH_UV_XY_RL(u, v, False),
          "UV3s": lambda: ex.EXCH_UV_3D_RL(u, v, True), "UV3n": lambda: ex.EXCH_UV_3D_RL(u, v, False),
          "Ds": lambda: ex.EXCH_UV_DGRID_3D_RL(u, v, True), "Dn": lambda: ex.EXCH_UV_DGRID_3D_RL(u, v, False)}
    if kind in sc:
        return (sc[kind](),)
    return vc[kind]()


def exchange_mismatches(exp, maps=None, apply=None):
    """{record: points that differ bitwise} of every exchange and probe vs the replay (empty = gate passes).
    apply(kind, u, v) -> outputs; default the single-device Exchanger of `maps` (default load_cube_maps)."""
    R = replay(exp, "cube_exch.bin")
    F, cb = R["fields"], R["cbase"]
    if apply is None:
        ex = Exchanger(EM.load_cube_maps(exp, "input") if maps is None else maps)

        def apply(kind, u, v):
            return call(ex, kind, jnp.asarray(u), jnp.asarray(v))
    bad = {}
    for iz in range(5):
        u, v = probe_inputs(exp, iz, cb)
        for kind in SCALAR_KINDS + VECTOR_KINDS:
            outs = apply(kind, u, v)
            for o, suf in zip(outs, ("_u", "_v")):
                key = f"z{iz}{kind}{suf}"
                n = int(np.count_nonzero(bits(o) != bits(F[key])))
                if n:
                    bad[key] = n
    return bad


def planted_maps(exp, mutate):
    """A deep copy of the cube ExchangeMaps with mutate(maps) applied (negative controls)."""
    m = copy.deepcopy(EM.load_cube_maps(exp, "input"))
    mutate(m)
    return m


# ---------------------------------------------------------------------------------------------------------------------
# lane A's registered oracle runs (reference/reference_runs.py, kind "jdon": dumps-on runs with the exchange probe)

ORACLE_EXPS = ("adjustment.cs-32x32x1", "solid-body.cs-32x32x1", "advect_cs", "global_ocean.cs32x15")


def oracle_top(exp):
    spec = importlib.util.spec_from_file_location("_mjx_reference_runs", paths.REPO / "reference" / "reference_runs.py")
    rr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rr)
    runs = [r for r in rr.RUNS if (r["exp"], r["input"], r["kind"]) == (exp, "input", "jdon")]
    if len(runs) != 1:
        raise FileNotFoundError(f"{exp}/input: {len(runs)} registered dumps-on runs")
    return rr.run_top(runs[0])


def oracle_printout_mismatches(exp, w2, io):
    """As printout_mismatches, against lane A's dumps-on run."""
    rd = oracle_top(exp) / "rundir"
    log = (rd / "w2_tile_topology.0000.log").read_text().split("\n")
    assert log[-1] == ""
    log = log[:-1]
    ours = io.records(w2.W2_oUnit)
    nlog = abs(len(log) - len(ours)) + sum(a != b for a, b in zip(log, ours))
    out = (rd / "output.txt").read_text().split("\n")
    so = io.records(STANDARD_MESSAGE_UNIT)
    i0 = out.index(so[0])
    nout = sum(a != b for a, b in zip(out[i0:i0 + len(so)], so)) + max(0, i0 + len(so) - len(out))
    return {"log": nlog, "stdout": nout, "log_records": len(ours), "stdout_records": len(so)}


def oracle_probe_mismatches(exp, maps=None):
    """({probe field: points that differ bitwise}, fields compared) of the jaxdump exchange probe (stage
    X00_exch_probe: the index-coded fields through every routine, and the zp/zm/zu/zv signed-zero probes) vs our
    exchanges built from the W2 topology alone."""
    from mitjax.io.dump import DumpSet
    ds = DumpSet(oracle_top(exp) / "dumps")
    it = ds.iterations()[0]
    st = EM.STAGE
    cb = ds.scalar(it, st, "xCbase")
    m = EM.load_cube_maps(exp, "input") if maps is None else maps
    ex = Exchanger(m)
    L = m.layout
    idx = (np.arange(L.npoints, dtype=np.float64) + 1.0).reshape(L.shape2d)
    bad, n = {}, 0

    def cmp(o, f):
        nonlocal n
        n += 1
        k = int(np.count_nonzero(bits(o) != bits(ds.field(it, st, f)[:, 0])))
        if k:
            bad[f] = k
    for kind, flds in list(EM.SCALAR.items()) + list(EM.VECTOR.items()):
        flds = (flds,) if isinstance(flds, str) else flds
        for o, f in zip(call(ex, kind, jnp.asarray(cb + idx), jnp.asarray(2.0 * cb + idx)), flds):
            cmp(o, f)
    for iz, tag in ((1, "zp"), (2, "zm"), (3, "zu"), (4, "zv")):
        zu = -0.0 if iz in (2, 3) else 0.0
        zv = -0.0 if iz in (2, 4) else 0.0
        u, v = jnp.full(L.shape2d, zu), jnp.full(L.shape2d, zv)
        for kind in ("UVs", "As", "Bs", "Ds", "UV3s", "SMs"):
            if kind == "SMs" and iz > 2:
                continue
            for o, suf in zip(call(ex, kind, u, v), ("_u", "_v") if kind != "SMs" else ("_u",)):
                cmp(o, f"{tag}{kind}{suf}")
    return bad, n


# ---------------------------------------------------------------------------------------------------------------------
# the curvilinear grid (INI_CURVILINEAR_GRID, plan Task 22)

GRID_FIELDS_NOT_INI_GRID = ("fCori", "fCoriG", "fCoriCos")      # INI_CORI's, dumped by the replay too


def horizontal_grid(exp, rundir=None, gp=None, ex=None):
    """The GRID.h fields of INITIALISE_FIXED for a cube experiment: the model's chain (drivers/model.py
    initialise_fixed_grid: INI_GRID, SET_GRID_FACTORS, INI_DEPTHS, INI_MASKS_ETC, INI_CORI) where the vertical grid is
    ported; for the p-coordinate solid-body (INI_VERTICAL_GRID raises: rF from the bottom is not ported) the
    horizontal part of INI_GRID exactly (init_horizontal, INI_CURVILINEAR_GRID, reciprocals) and INI_CORI. Files are
    read from `rundir` (default: lane A's dumps-on run directory, where prepare_run linked the grid files)."""
    from mitjax.model.grid import Grid
    from mitjax.model.src import ini_grid as IG
    from mitjax.model.src.ini_curvilinear_grid import ini_curvilinear_grid
    from mitjax.model.src.ini_parms import ini_parms_grid
    from mitjax.pkg.exch2.w2_eeboot import exch2_topology
    from mitjax.pkg.rw.read_rec import RW
    e, w2, _ = setup(exp)
    gp = ini_parms_grid(e, exch2_topology(w2)) if gp is None else gp
    ex = Exchanger(EM.load_cube_maps(exp, "input")) if ex is None else ex
    rw = RW(oracle_top(exp) / "rundir" if rundir is None else rundir, gp.readBinaryPrec, e.cfg.size)
    if gp.usingPCoords:
        from mitjax.model.src.ini_cori import ini_cori
        g = IG.init_horizontal(Grid(), e.cfg.size)
        g = ini_curvilinear_grid(g, cfg=e.cfg, params=gp, ex=ex, rw=rw)
        g = IG.reciprocals(g, e.cfg.size)
        return ini_cori(g, cfg=e.cfg, params=gp)
    from mitjax.drivers.model import initialise_fixed_grid
    return initialise_fixed_grid(e, gp, ex=ex, rw=rw)


def grid_replay_mismatches(exp, grid):
    """({field: points that differ bitwise}, fields compared) vs the replay's cube_grid.bin (all points)."""
    R = _rio().read(replay_dir(exp) / "cube_grid.bin", False)["fields"]
    bad, n = {}, 0
    for name, ref in R.items():
        n += 1
        k = int(np.count_nonzero(bits(getattr(grid, name).data) != bits(ref)))
        if k:
            bad[name] = k
    return bad, n


def grid_oracle_mismatches(exp, grid):
    """({field: points that differ bitwise}, fields compared) vs lane A's G00_geometry dumps of the dumps-on run, for
    every dumped field our horizontal grid holds (all points, halos and corners)."""
    from mitjax.io.dump import DumpSet
    ds = DumpSet(oracle_top(exp) / "dumps")
    it = ds.iterations()[0]
    names = {k[2] for k in ds.keys(it) if k[1] == "G00_geometry"}
    bad, n = {}, 0
    for name in GRID_ORACLE_FIELDS:
        if name not in names:
            continue
        ref = ds.field(it, "G00_geometry", name)
        ref = ref[:, 0] if ref.ndim == 4 else ref
        n += 1
        k = int(np.count_nonzero(bits(getattr(grid, name).data) != bits(ref)))
        if k:
            bad[name] = k
    return bad, n


GRID_ORACLE_FIELDS = ("xC", "yC", "xG", "yG", "dxC", "dyC", "dxF", "dyF", "dxG", "dyG", "dxV", "dyU", "rA", "rAw",
                      "rAs", "rAz", "angleCosC", "angleSinC", "u2zonDir", "v2zonDir", "recip_dxC", "recip_dyC",
                      "recip_dxF", "recip_dyF", "recip_dxG", "recip_dyG", "recip_dxV", "recip_dyU", "recip_rA",
                      "recip_rAw", "recip_rAs", "recip_rAz", "fCori", "fCoriG", "fCoriCos")


# ---------------------------------------------------------------------------------------------------------------------
# GAD_ADVECTION's cubed-sphere pass structure (gad_cs_passes.py)

def gad_pass_replay():
    """{(nCFace, iE, ipass): (flags (overlapOnly, interiorOnly, calc_fluxes_X, calc_fluxes_Y), events)} of the
    gfortran replay (reference/replay_cube/gad_passes; its run directory in reference/replay_cube/CURRENT as
    `gad_passes`)."""
    out, key = {}, None
    for line in (replay_dir("gad_passes") / "out.txt").read_text().split("\n"):
        if line.startswith("P"):
            f = line[1:].split()
            key = (int(f[0]), int(f[1]), int(f[2]))
            out[key] = (tuple(x == "T" for x in f[3:7]), [])
        elif line.startswith("E "):
            kind, what = line[2:].split()
            out[key][1].append((kind, int(what) if kind == "fill" else what))
    return out
