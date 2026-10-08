"""Helpers of the R2 (tutorial_baroclinic_gyre) gates, plan Task 13: the R1 set-up of mitjax/tests/r1_gate.py with
the tracer-step parameters (mitjax/model/src/ini_parms_tracer.py) added before INITIALISE_VARIA, the substep stages
the baroclinic gyre dumps, and the 4-tile P=4 run on fake CPU devices.

Diagnostics: pkg/diagnostics is not ported (decided 2026-10-01 by Nikolay); the kernels see useDiagnostics =
.FALSE., DIAGNOSTICS_IS_ON is evaluated on the host (ini_parms_tracer.diagnostics_is_on: 'MXLDEPTH' is on in this
run, so DO_OCEANIC_PHYS runs GRAD_SIGMA at k=1 and CALC_OCE_MXLAYER exactly as the oracle does).
"""

from functools import lru_cache
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.farray import loop_i, loop_j
from mitjax.model.grid import UNSET_RL, declare
from mitjax.model.src.cg2d_h import ini_parms_cg2d
from mitjax.model.src.ini_linear_phisurf import ini_linear_phisurf
from mitjax.model.src.ini_parms import ini_parms_dyn
from mitjax.model.src.ini_parms_tracer import ini_parms_tracer
from mitjax.tests import grid_gate as gg
from mitjax.tests import init_gate as ig
from mitjax.tests import r1_gate as rg

EXP = ("tutorial_baroclinic_gyre", "input")

# every stage of steps 1-3 the baroclinic gyre's oracle dumps on the tracer and density side, plus the R1 stages
# (tile-scope stages P02, P03, T11-T13, T02 compare all tiles at once: tiles are independent inside those routines)
STAGES = ("S02_load_fields", "P02_rho_sigma_ivdc", "P03_mxlayer", "S04_oceanic_phys", "T11_temp_gT", "T12_temp_step",
          "T13_temp_impl", "T02_temp_integrate", "S05_thermodynamics_sync", "S06_dynamics", "S09_solve_for_pressure",
          "S10_momentum_correction", "S11_integr_continuity", "S15_tracers_correction", "S16_blocking_exchanges")


class Model(rg.Model):
    """r1_gate.Model with the tracer parameters (the same construction, INI_PARMS -> INITIALISE_VARIA order)."""

    def __init__(self, exp, inp):        # noqa: C901 - mirrors r1_gate.Model.__init__ with one added line
        from mitjax.model.src.ini_eos import ini_eos
        from mitjax.model.src.ini_ffields import ini_ffields
        from mitjax.model.src.ini_forcing import ini_forcing
        from mitjax.model.src.ini_grid import ini_grid
        from mitjax.model.src.ini_parms_forcing import ini_parms_forcing
        from mitjax.model.src.initialise_varia import executed_pending, initialise_varia
        from mitjax.pkg.rw.read_rec import RW
        from mitjax.params_io import RunParams
        self.exp, self.inp = exp, inp
        e = self.e = gg.experiment(exp, inp)
        cfg = self.cfg = e.cfg
        self.prm = ig.params(exp, inp)
        params = ini_parms_dyn(e, self.prm.grid, self.prm.time, self.prm.init)
        self.params = ini_parms_tracer(e, params, self.prm.time, self.prm.init)          # the added line
        self.fp = ini_parms_forcing(e)
        self.cg2d_params = ini_parms_cg2d(e)
        self.ex = gg.exchanger(exp)
        self.ds, self.it0, self.rundir = gg.oracle(exp, inp)
        self.rw = RW(self.rundir, self.prm.init.readBinaryPrec, cfg.size)
        grid = gg.build_grid(exp, inp, params=self.prm.grid, ex=self.ex)
        self.grid = ini_linear_phisurf(grid, cfg=cfg, params=self.params)                # initialise_fixed.F:226
        rp = RunParams(e.run)
        eos_p = SimpleNamespace(fluidIsWater=self.params.fluidIsWater, usingPCoords=self.params.usingPCoords,
                                eosType=self.params.eosType,
                                tAlpha=rp.get("data", "PARM01", "tAlpha") if rp.has("data", "PARM01", "tAlpha")
                                else UNSET_RL,                                           # ini_parms.F:421
                                sBeta=rp.get("data", "PARM01", "sBeta") if rp.has("data", "PARM01", "sBeta")
                                else UNSET_RL)                                           # ini_parms.F:422
        self.eos = ini_eos(cfg=cfg, params=eos_p)                                        # initialise_fixed.F:176
        from mitjax.model.src.ini_cg2d import ini_cg2d
        self.cg2dh = jax.jit(lambda g, s, p, x: ini_cg2d(cfg=cfg, grid=g, surface=s, params=p, ex=x))(
            self.grid, self.grid, self.cg2d_params, self.ex)                            # initialise_fixed.F:246
        # GO lane: INITIALISE_VARIA's NONLIN_FRSURF sequence (CALC_R_STAR, UPDATE_R_STAR, UPDATE_CG2D) is ported
        pend = tuple(r for r in executed_pending(cfg, self.prm)
                     if r not in ("INTEGR_CONTINUITY", "CALC_R_STAR", "UPDATE_R_STAR", "UPDATE_CG2D"))
        st = initialise_varia(self.grid, cfg=cfg, params=self.prm, ex=self.ex, rw=self.rw, pending=pend,
                              dyn=self.params, cg2dh=self.cg2dh, cg2d_params=self.cg2d_params)   # GO: CG2D.h
        ff = ini_ffields(cfg=cfg)                                                        # initialise_varia.F:213
        _, _, _, lat = ini_grid(cfg=cfg, params=self.prm.grid)
        self.ff = ini_forcing(ff, cfg=cfg, grid=self.grid, fp=self.fp, rw=self.rw, ex=self.ex,
                              latBandClimRelax=lat)                                      # initialise_varia.F:242
        sz = cfg.size
        j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
        i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
        self.phi0surf = declare("Bo_surf", sz).at[i, j].set(0.)                          # ini_linear_phisurf.F:204
        self.state0 = st
        self.pending = pend


@lru_cache(maxsize=None)
def model(exp=EXP[0], inp=EXP[1]):
    return Model(exp, inp)


def stage_values(stage, v):
    """The probe values of a stage as {name: FArray} (compare_stage accepts States and dicts)."""
    if stage == "S04_oceanic_phys":
        s_, f_, p_ = v
        vals = rg.state_fields(s_)
        vals.update({n: getattr(f_, n) for n in f_.names()})
        vals["phi0surf"] = p_
        return vals
    return v


def run_steps(m, params, n, stages=STAGES, compare=True):
    """n steps from the initial state; per step {stage: compare_stage} (or the probe values with compare=False)."""
    fn = m.step_fn(stages)
    state, ff, phi0, t, it = m.state0, m.ff, m.phi0surf, m.prm.time.startTime, m.prm.time.nIter0
    res = []
    for k in range(n):
        state, ff, phi0, t, it, out, pr = fn(m.grid, params, m.eos, m.cg2dh, m.cg2d_params, m.ex, state, ff, phi0,
                                             jnp.int32(k + 1), jnp.float64(t), jnp.int32(it))
        if compare:
            res.append({st: rg.compare_stage(m.ds, k, st, stage_values(st, pr[st])) for st in stages})
        else:
            res.append(pr)
    return res, state


def bad(r):
    return {k: v for k, v in r.items() if v[0] == "shape" or any(v[1:])}


def ndiff(res, stages):
    """Number of differing points (==) per step and stage, summed over the stage's fields."""
    return [{st: sum(v[1] for v in r[st].values()) for st in stages} for r in res]


def whole_run(m, step_fn=None):
    """The whole run (nTimeSteps steps) as test_r1_barotropic_gyre.whole_run does it for R1: (oracle, CG2D Sum(rhs)
    lines, SOLVE_FOR_PRESSURE lines, STDOUT-like records, {iteration: %MON block}, final State). `step_fn(k, state,
    ff, phi0, t, it)` -> (state, ff, phi0, t, it, out): default the jit'ed single-device step."""
    from mitjax.model.src.cg2d import cg2d_sum_rhs_message, solve_for_pressure_cg2d_messages
    from mitjax.pkg.monitor.mon_calc_advcfl import mon_calc_advcfl_glob, mon_calc_advcfl_tile
    from mitjax.pkg.monitor.mon_init import mon_init
    from mitjax.pkg.monitor.monitor import monitor
    from mitjax.pkg.monitor.monitor_h import MonitorCommon
    from mitjax.tests import monitor_gate as mg
    o = mg.oracle(m.exp, m.inp)
    mcfg = mg.make_cfg(o)
    mprm = mg.make_params(o, mcfg)
    mon = MonitorCommon()
    mon_init(cfg=mcfg, params=mprm, mon=mon)

    def gns():
        ns = SimpleNamespace(**{n: getattr(m.grid, n) for n in m.grid.names()})
        ns.rhoFacC, ns.rhoFacF, ns.recip_rhoFacC = m.params.rhoFacC, m.params.rhoFacF, m.params.recip_rhoFacC
        return ns

    def sns(state, ff, phi0):
        ns = SimpleNamespace(**rg.state_fields(state))
        for n in ("Qnet", "Qsw", "EmPmR", "fu", "fv"):
            setattr(ns, n, getattr(ff, n))
        ns.phi0surf = phi0
        return ns

    blocks = {}

    def do_mon(tsn, t, state, ff, phi0):
        n0 = len(mon.units.get(mon.mon_ioUnit, []))
        monitor(t, tsn, cfg=mcfg, params=mprm, grid=gns(), state=sns(state, ff, phi0), mon=mon)
        blocks[tsn] = list(mon.units.get(mon.mon_ioUnit, []))[n0:]

    if step_fn is None:
        fn = m.step_fn(())

        def step_fn(k, state, ff, phi0, t, it):
            out = fn(m.grid, m.params, m.eos, m.cg2dh, m.cg2d_params, m.ex, state, ff, phi0,
                     jnp.int32(k + 1), jnp.float64(t), jnp.int32(it))
            return out[:6]
    state, ff, phi0, t, it = m.state0, m.ff, m.phi0surf, m.prm.time.startTime, m.prm.time.nIter0
    do_mon(it, t, state, ff, phi0)
    sums, mons, records = [], [], list(blocks[it])
    for k in range(m.prm.time.nTimeSteps):
        state, ff, phi0, t, it, out = step_fn(k, state, ff, phi0, t, it)
        c = {key: np.asarray(v) for key, v in out["cg2d"].items()}
        sums.append(cg2d_sum_rhs_message(c["sumRHS"], c["rhsMax"]))
        lines = solve_for_pressure_cg2d_messages(c["firstResidual"], c["minResidualSq"], c["lastResidual"],
                                                 c["numIters"], c["nIterMin"])
        mons += lines
        records.append(sums[-1])
        records += ["(PID.TID 0000.0001) " + ln for ln in lines]
        maxCFL = mon_calc_advcfl_tile(mcfg.Nr, *out["flow"], mprm.dTtracerLev, None, k, cfg=mcfg, grid=gns())
        mon_calc_advcfl_glob(maxCFL, k, mon=mon)
        do_mon(int(it), float(t), state, ff, phi0)
        records += blocks[int(it)]
    return o, sums, mons, records, blocks, state


def testreport_digits(m, o, records):
    """tools/testreport_jax.py: (ours vs results/, the oracle's own yardstick vs results/, ours vs the oracle)."""
    import importlib.util as iu
    import tempfile
    from pathlib import Path
    from mitjax import paths
    spec = iu.spec_from_file_location("_trj", paths.REPO / "tools" / "testreport_jax.py")
    trj = iu.module_from_spec(spec)
    spec.loader.exec_module(trj)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "output.txt"
        out.write_text("\n".join(records) + "\n")
        ours_res = trj.compare(str(out), m.exp, m.inp)
        yard = trj.compare(str(o.stdout_path), m.exp, m.inp)
        vs_oracle = trj.compare(str(out), m.exp, m.inp, reference=str(o.stdout_path))
    return [(a.name, a.digits, y.digits, s.digits) for a, y, s in
            zip(ours_res.run.variables, yard.run.variables, vs_oracle.run.variables)]


def specs(sh, tree):
    """PartitionSpec prefix tree of a model pytree: tiled FArrays sharded on the tile axis, everything else
    replicated (the placement TileSharding.put_tree makes)."""
    from mitjax.farray import FArray
    return jax.tree.map(lambda x: sh.TILES if isinstance(x, FArray) and x.tiled else sh.REP, tree,
                        is_leaf=lambda x: isinstance(x, FArray))


def sharded_step_fn(m, sh):
    """FORWARD_STEP under jit(shard_map(check_vma=True)) on the TileSharding `sh` (fake CPU devices of the gate
    flags), the ShardedExchanger as `ex`; the grid, cg2d operator, state and forcing placed by put_tree once.
    Returns step_fn(k, state4, ff4, phi04, t, it) -> (state4, ff4, phi04, t, it, out) on placed arrays, and the
    placed (state0, ff0, phi00)."""
    from mitjax.model.src.forward_step import forward_step
    cfg, fp = m.cfg, m.fp
    g4, eos4, cg2dh4 = sh.put_tree(m.grid), sh.put_tree(m.eos), sh.put_tree(m.cg2dh)
    prm4, cgp4 = sh.put_tree(m.params), sh.put_tree(m.cg2d_params)
    st0, ff0, ph0 = sh.put_tree(m.state0), sh.put_tree(m.ff), sh.put_tree(m.phi0surf)

    def body(grid, params, eos, cg2dh, cg2d_params, ex, state, ff, phi0surf, iloop, myTime, myIter):
        st, f, p, t, it, out = forward_step(iloop, myTime, myIter, cfg=cfg, grid=grid, params=params, fp=fp, eos=eos,
                                            cg2dh=cg2dh, cg2d_params=cg2d_params, state=state, ff=ff,
                                            phi0surf=phi0surf, ex=ex)
        return st, f, p, t, it, out

    T, R = sh.TILES, sh.REP
    flow_spec = specs(sh, m.state0.uVel)
    in_specs = (specs(sh, m.grid), specs(sh, m.params), specs(sh, m.eos), specs(sh, m.cg2dh),
                specs(sh, m.cg2d_params), T, specs(sh, m.state0), specs(sh, m.ff), specs(sh, m.phi0surf), R, R, R)
    out_specs = (specs(sh, m.state0), specs(sh, m.ff), specs(sh, m.phi0surf), R, R,
                 {"flow": (flow_spec, flow_spec, flow_spec), "cg2d": R})
    f = sh.shard_map(body, in_specs=in_specs, out_specs=out_specs)

    def step_fn(k, state, ff, phi0, t, it):
        return f(g4, prm4, eos4, cg2dh4, cgp4, sh.ex, state, ff, phi0, jnp.int32(k + 1), jnp.float64(t),
                 jnp.int32(it))
    return step_fn, (st0, ff0, ph0)


def tree_bits_differ(a, b):
    """{leaf path: number of points whose bit patterns differ} for two pytrees of the same structure (numpy)."""
    out = {}
    la = jax.tree_util.tree_leaves_with_path(a)
    lb = jax.tree.leaves(b)
    for (path, x), y in zip(la, lb):
        x64 = np.ascontiguousarray(np.asarray(x), np.float64)
        y64 = np.ascontiguousarray(np.asarray(y), np.float64)
        if x64.shape != y64.shape:
            out[jax.tree_util.keystr(path)] = ("shape", x64.shape, y64.shape)
            continue
        n = int(np.count_nonzero(x64.view(np.int64) != y64.view(np.int64)))
        if n:
            out[jax.tree_util.keystr(path)] = n
    return out


__all__ = ["Model", "model", "STAGES", "run_steps", "bad", "ndiff", "stage_values", "whole_run", "testreport_digits", "specs",
           "sharded_step_fn", "tree_bits_differ"]
