"""Gate helpers for pkg/ctrl, pkg/cost and pkg/grdchk on tutorial_global_oce_optim/input_ad (lane ctrl).

Oracle runs (docs/REFERENCE_RUNS.md conventions; made by mitjax/tests/ctrl_oracle.sbatch, jaxdump binary
tutorial_global_oce_optim-code_ad-63cdc0b-5f16129-jaxdump, dumps at all 10 steps):
  ZERO = job27829092-ctrlzero   the variant as testreport runs it (first-guess control zero)
  XX   = job27829159-ctrlxx     planted xx_qnet (seed 20261001, +-50 W/m^2), doInitXX=.FALSE., doMainUnpack=.FALSE.
  FD   = job27827478-fdzero     zero adxx_qnet: the full grdchk STDOUT (positions, FD costs)
Inputs to our routines come from the oracle (dumped forcing at S02, grid at G00, state at S17, the run directory's
control/weight files); outputs are compared with the dumps / files the oracle wrote, every point incl. halos.
"""

from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from mitjax import paths
from mitjax.farray import FArray
from mitjax.io import stdout as so
from mitjax.io.dump import DumpSet

EXP, INP = "tutorial_global_oce_optim", "input_ad"
RUNS = {"zero": "job27829092-ctrlzero", "xx": "job27829159-ctrlxx", "fd": "job27827478-fdzero"}


def run_top(kind):
    return paths.REFERENCE_RUNS / EXP / INP / RUNS[kind]


@lru_cache(maxsize=None)
def dumps(kind):
    return DumpSet(run_top(kind) / "dumps")


@lru_cache(maxsize=None)
def experiment():
    from mitjax.config import params
    return params.load(EXP, INP)


@lru_cache(maxsize=None)
def exchanger():
    from mitjax.eesupp.exch_maps import load_maps
    from mitjax.eesupp.exchange import Exchanger
    return Exchanger(load_maps(EXP))


def fa2(data, name, sz):
    return FArray(jnp.asarray(data), name, i=(1 - sz.OLx, sz.sNx + sz.OLx), j=(1 - sz.OLy, sz.sNy + sz.OLy))


def fa3(data, name, sz, nz):
    return FArray(jnp.asarray(data), name, i=(1 - sz.OLx, sz.sNx + sz.OLx), j=(1 - sz.OLy, sz.sNy + sz.OLy),
                  k=(1, nz))


def field2(kind, it, stage, name):
    f = dumps(kind).field(it, stage, name)
    assert f.shape[1] == 1, (stage, name, f.shape)
    return f[:, 0]


@lru_cache(maxsize=None)
def grid(kind="zero"):
    sz = experiment().cfg.size
    g = {}
    for n in ("angleCosC", "angleSinC", "rA"):
        g[n] = fa2(field2(kind, 0, "G00_geometry", n), n, sz)
    for n in ("maskC", "maskW", "maskS"):
        g[n] = fa3(dumps(kind).field(0, "G00_geometry", n), n, sz, sz.Nr)
    return g


def read_tiled_xy(top, stem, sz):
    """A tiled MDS 2-D file `<stem>.<bi>.<bj>.data` (big-endian float64, one record, no halos) -> [tile, j, i] with
    zero halos, tiles in bi + (bj-1)*nSx order."""
    out = np.zeros((sz.nSx * sz.nSy, sz.sNy + 2 * sz.OLy, sz.sNx + 2 * sz.OLx))
    for bj in range(1, sz.nSy + 1):
        for bi in range(1, sz.nSx + 1):
            raw = np.fromfile(Path(top) / "rundir" / f"{stem}.{bi:03d}.{bj:03d}.data", ">f8")
            out[bi - 1 + (bj - 1) * sz.nSx, sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] = \
                raw.reshape(sz.sNy, sz.sNx)
    return out


def read_global_xy_record(path, sz, rec=1):
    """Record `rec` of a global big-endian float64 file of Nx*Ny values (READ_REC_3D_RL, one level) -> [tile, j, i]
    with zero halos."""
    n = sz.Nx * sz.Ny
    raw = np.fromfile(path, ">f8", count=n, offset=8 * n * (rec - 1)).reshape(sz.Ny, sz.Nx)
    out = np.zeros((sz.nSx * sz.nSy, sz.sNy + 2 * sz.OLy, sz.sNx + 2 * sz.OLx))
    for bj in range(sz.nSy):
        for bi in range(sz.nSx):
            out[bi + bj * sz.nSx, sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] = \
                raw[bj * sz.sNy:(bj + 1) * sz.sNy, bi * sz.sNx:(bi + 1) * sz.sNx]
    return out


@lru_cache(maxsize=None)
def printed(kind):
    lines = so.read_stdout(run_top(kind) / "rundir" / "output.txt")
    return lines, so.parameter_dict(so.parameters(lines))


def printed_float(kind, name):
    return so.fortran_number(printed(kind)[1][name].values()[0])


def clock(kind="zero"):
    """The model clock as the oracle printed it (CONFIG_SUMMARY): host floats for the record bookkeeping."""
    exp = experiment()
    return {"useCAL": exp.cfg.use_flag("useCAL"), "startTime": printed_float(kind, "startTime"),
            "deltaTClock": printed_float(kind, "deltaTClock"),
            "externForcingCycle": printed_float(kind, "externForcingCycle")}


def bits_equal(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def n_bits_differ(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return int(np.count_nonzero(a.view(np.uint64) != b.view(np.uint64)))


# ---------------------------------------------------------------------------------------------------------------
# the gated chains

FORCING = ("fu", "fv", "Qnet", "EmPmR", "Qsw", "pLoad", "saltFlux")


def control_input(kind):
    """(xx records {1: [FArray]}, weight {1: FArray}) as CTRL_MAP_INI_GENTIM2D reads them in run `kind`."""
    sz = experiment().cfg.size
    top = run_top(kind)
    xx = fa2(read_tiled_xy(top, "xx_qnet.0000000000", sz), "xx", sz)
    w = fa2(read_global_xy_record(top / "rundir" / "ones_64b.bin", sz), "w", sz)
    return {1: [xx]}, {1: w}


def ctrl_init(kind, xx_in=None):
    """CTRL_MAP_INI_GENTIM2D on run `kind`'s inputs (or `xx_in`) -> (effective, wgentim2d, genarr0)."""
    from mitjax.pkg.ctrl.ctrl_map_ini_gentim2d import ctrl_map_ini_gentim2d
    from mitjax.pkg.ctrl.ctrl_readparms import ctrl_readparms_gentim2d
    exp = experiment()
    cfg, sz = exp.cfg, exp.cfg.size
    xx, w = control_input(kind)
    if xx_in is not None:
        xx = xx_in
    clk = clock(kind)
    eff, wg, _ = ctrl_map_ini_gentim2d(xx, w, cfg=cfg, gentim2d=ctrl_readparms_gentim2d(exp.run),
                                       maskC=grid(kind)["maskC"], ex=exchanger(), startTime=clk["startTime"],
                                       endTime=printed_float(kind, "endTime"))
    zero = jnp.zeros((sz.nSx * sz.nSy, sz.sNy + 2 * sz.OLy, sz.sNx + 2 * sz.OLx))
    genarr = {k: {1: fa2(zero, k, sz)} for k in ("xx_gentim2d", "xx_gentim2d0", "xx_gentim2d1")}
    genarr["wgentim2d"] = {1: wg[1]}
    return eff, wg, genarr


def ctrl_step(kind, it, eff, genarr, forcing=None):
    """CTRL_MAP_GENTIM2D + CTRL_MAP_FORCING at step `it` on the S02 forcing of run `kind` -> (out fields, genarr)."""
    from mitjax.pkg.ctrl.ctrl_map_forcing import ctrl_map_forcing
    from mitjax.pkg.ctrl.ctrl_map_gentim2d import ctrl_map_gentim2d
    from mitjax.pkg.ctrl.ctrl_readparms import ctrl_readparms_gentim2d
    exp = experiment()
    cfg, sz = exp.cfg, exp.cfg.size
    gt = ctrl_readparms_gentim2d(exp.run)
    clk = clock(kind)
    myTime = clk["startTime"] + clk["deltaTClock"] * it          # forward_step.F:430 (code_ad arm)
    genarr = ctrl_map_gentim2d(myTime, it, cfg=cfg, gentim2d=gt, maskC=grid(kind)["maskC"], genarr=genarr,
                               effective=eff, clock=clk)
    ff = forcing or {n: fa2(field2(kind, it, "S02_load_fields", n), n, sz) for n in FORCING}
    zero = jnp.zeros_like(ff["Qnet"].data)
    ff.setdefault("SST", fa2(zero, "SST", sz))                    # not dumped; exchanged only
    ff.setdefault("SSS", fa2(zero, "SSS", sz))
    out = ctrl_map_forcing(myTime, it, cfg=cfg, gentim2d=gt, xx_gentim2d=genarr["xx_gentim2d"], ffields=ff,
                           grid=grid(kind), ex=exchanger(), usingPCoords=False)
    return out, genarr


@lru_cache(maxsize=None)
def cost_inputs(kind):
    """(tmpwti, Err_hflux FArray, thetalev FArray) read as the Fortran reads them."""
    sz = experiment().cfg.size
    rd = run_top(kind) / "rundir"
    tmpwti = np.fromfile(rd / "Err_levitus_15layer.bin", ">f8", count=sz.Nr)
    errh = fa2(read_global_xy_record(rd / "Err_hflux.bin", sz), "whfluxm", sz)
    raw = np.fromfile(rd / "lev_t_an.bin", ">f4").astype(np.float64).reshape(sz.Nr, sz.Ny, sz.Nx)
    tl = np.zeros((sz.nSx * sz.nSy, sz.Nr, sz.sNy + 2 * sz.OLy, sz.sNx + 2 * sz.OLx))
    for bj in range(sz.nSy):
        for bi in range(sz.nSx):
            tl[bi + bj * sz.nSx, :, sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] = \
                raw[:, bj * sz.sNy:(bj + 1) * sz.sNy, bi * sz.sNx:(bi + 1) * sz.sNx]
    return tmpwti, errh, fa3(tl, "thetalev", sz, sz.Nr)


def cost_params():
    p = experiment().params
    return {"mult_temp_tut": p["data.cost:cost_nml:mult_temp_tut"],
            "mult_hflux_tut": p["data.cost:cost_nml:mult_hflux_tut"],
            "lastinterval": p["data.cost:cost_nml:lastinterval"]}


def cost_run(kind, xx_gentim2d, tmpwti=None, per_step=None):
    """COST_WEIGHTS, COST_INIT_VARIA/COST_DEPENDENT_INIT, COST_TILE at each of the 10 steps on the oracle's S17 state,
    COST_FINAL -> (cost, loc_fc, early_fc). per_step(it, cost) is called after each COST_TILE."""
    from mitjax.pkg.cost.cost_final import cost_final
    from mitjax.pkg.cost.cost_init_varia import cost_dependent_init, cost_init_varia
    from mitjax.pkg.cost.cost_tile import cost_tile
    from mitjax.pkg.cost.cost_weights import cost_weights
    exp = experiment()
    cfg, sz = exp.cfg, exp.cfg.size
    g, ds, clk = grid(kind), dumps(kind), clock(kind)
    tw, errh, thetalev = cost_inputs(kind)
    whfluxm, wtheta = cost_weights(cfg=cfg, tmpwti=tw if tmpwti is None else tmpwti, Err_hflux=errh,
                                   ex=exchanger(), xy=lambda d, n: fa2(d, n, sz))
    cost = cost_dependent_init(cost_init_varia(cfg=cfg, ntiles=sz.nSx * sz.nSy), cfg=cfg)
    prm = cost_params()
    endTime = printed_float(kind, "endTime")
    for it in range(10):
        myTime = clk["startTime"] + clk["deltaTClock"] * (it + 1)   # COST_TILE after the update (forward_step.F:807)
        st = {n: fa3(ds.field(it, "S17_monitor", n), n, sz, sz.Nr) for n in ("theta", "uVel", "vVel")}
        cost = cost_tile(cost, myTime, it + 1, cfg=cfg, endTime=endTime, lastinterval=float(prm["lastinterval"]),
                         maskC=g["maskC"], maskW=g["maskW"], maskS=g["maskS"], deltaTClock=clk["deltaTClock"],
                         lastinterval_traced=prm["lastinterval"], **st)
        if per_step:
            per_step(it, cost)
    early = cost.fc
    cost, loc = cost_final(cost, cfg=cfg, params=prm, maskC=g["maskC"], wtheta=wtheta, thetalev=thetalev,
                           whfluxm=whfluxm, xx_gentim2d=xx_gentim2d)
    return cost, loc, early


def dumped_tiles(kind, it, stage, name):
    recs = dumps(kind).tiles(it, stage, name)
    return np.array([recs[t].data.ravel()[0] for t in sorted(recs)])
