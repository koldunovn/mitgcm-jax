"""Helpers of the R4 global_ocean.90x40x15 gates (M1 lane GO, plan Tasks 15a/15c): the routines this lane wires into
FORWARD_STEP run under jit at the gate XLA flags (conftest.py), every float a traced argument, on inputs
teacher-forced from the oracle's substep dumps (registered dumps-on run of kind `jdon3`: steps 36000-36002 from the
pickup start nIter0 = 36000), and their outputs are compared with the dumped outputs on every point of every tile,
halos included (element equality, bit patterns, finite).

Which dumped stage feeds which input (reference/jaxdump/SUBSTEPS.md; forward_step.F order, global_ocean has
staggerTimeStep = .FALSE.):
  * S00_begin / G00_geometry: the State at the start of the step (groups d t a r f m k g c, the r* fields of G00 R,
    recip_hFacW/S of G00 G: the values UPDATE_R_STAR(.TRUE.) of the previous step left);
  * S01_update_rstar_F: after RESET_NLFS_VARS + UPDATE_R_STAR(.FALSE.) (forward_step.py nlfs_reset);
  * S04_oceanic_phys (groups f m k g r t): forcing (FFIELDS.h, phi0surf), rhoInSitu, totPhiHyd, ... as DYNAMICS
    reads them; S05_thermodynamics_sync (t a): theta, salt and gU, gV, guNm1, gvNm1 (THERMODYNAMICS does not write
    them) -> DYNAMICS -> S06_dynamics (a d m);
  * S06 -> forward_step.py nlfs_update_hfac (UPDATE_R_STAR(.TRUE.)) -> S07 (r); nlfs_update_cg2d -> S08 (c);
  * S11_integr_continuity (d: etaH) -> nlfs_calc_r_star -> S12 (r) and G00 R of the next step.
The CD_CODE_VARS.h state is not dumped: at the first step it is CD_CODE_INI_VARS' (pickup_cd.0000036000 of the
oracle run directory), at later steps the one our own DYNAMICS of the previous step produced (the chain in
`dynamics_chain`), with etaNm1 = etaN as SOLVE_FOR_PRESSURE sets it (solve_for_pressure.F:127) and the end-of-step
exchange of uVelD, vVelD (do_fields_blocking_exchanges.F:82-85).

Parameters: ini_parms_dyn + ini_parms_tracer of the experiment (the kernels see useDiagnostics = .FALSE.;
DIAGNOSTICS_IS_ON on the host, Nikolay 2026-10-01).
"""

import functools
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.farray import FArray
from mitjax.tests import grid_gate as gg
from mitjax.tests import init_gate as ig
from mitjax.tests import rstar_gate as rsg

EXP = ("global_ocean.90x40x15", "input")
GRID_R = ("hFacC", "hFacW", "hFacS", "recip_hFacC")


@functools.lru_cache(maxsize=None)
def setup(exp=EXP[0], inp=EXP[1], kind="jdon3"):
    """(e, prm, params, grid, ex, eos, cg2d_params, cg2dh, ds, its, rundir) of the variant; grid as INITIALISE_FIXED
    builds it (+ INI_LINEAR_PHISURF), params = ini_parms_dyn + ini_parms_tracer."""
    from mitjax.model.grid import UNSET_RL
    from mitjax.model.src.cg2d_h import ini_parms_cg2d
    from mitjax.model.src.ini_cg2d import ini_cg2d
    from mitjax.model.src.ini_eos import ini_eos
    from mitjax.model.src.ini_linear_phisurf import ini_linear_phisurf
    from mitjax.model.src.ini_parms import ini_parms_dyn
    from mitjax.model.src.ini_parms_tracer import ini_parms_tracer
    from mitjax.params_io import RunParams
    e = gg.experiment(exp, inp)
    cfg = e.cfg
    prm = ig.params(exp, inp)
    params = ini_parms_dyn(e, prm.grid, prm.time, prm.init)
    params = ini_parms_tracer(e, params, prm.time, prm.init)
    ds, its, rundir = rsg.oracle(exp, inp, kind)
    ex = gg.exchanger(exp)
    sz = cfg.size
    if ex.layout.nTiles != sz.nSx*sz.nSy*sz.nPx*sz.nPy or ex.layout.sNx != sz.sNx:
        # session 4: a build whose layout has no registered map file (global_ocean.90x40x15/code_ad: 4 tiles of
        # 45x20, exch1): the maps decoded in memory from the run's own exchange probe (X00_exch_probe), the same
        # decoder (exch_maps.build_maps) the registered map files are made with; nothing is written
        from mitjax.eesupp.exch_maps import build_maps
        from mitjax.eesupp.exchange import Exchanger
        ex = Exchanger(build_maps(ds))
    grid = gg.build_grid(exp, inp, params=prm.grid, ex=ex)
    grid = ini_linear_phisurf(grid, cfg=cfg, params=params)
    rp = RunParams(e.run)
    eos_p = SimpleNamespace(fluidIsWater=params.fluidIsWater, usingPCoords=params.usingPCoords,
                            eosType=params.eosType,
                            tAlpha=rp.get("data", "PARM01", "tAlpha") if rp.has("data", "PARM01", "tAlpha")
                            else UNSET_RL,
                            sBeta=rp.get("data", "PARM01", "sBeta") if rp.has("data", "PARM01", "sBeta")
                            else UNSET_RL)
    eos = ini_eos(cfg=cfg, params=eos_p)
    cg2d_params = ini_parms_cg2d(e)
    cg2dh = jax.jit(lambda g, s, p, x: ini_cg2d(cfg=cfg, grid=g, surface=s, params=p, ex=x))(
        grid, grid, cg2d_params, ex)
    from mitjax.model.src.ini_parms_forcing import ini_parms_forcing
    fp = ini_parms_forcing(e)
    return SimpleNamespace(e=e, cfg=cfg, prm=prm, params=params, grid=grid, ex=ex, eos=eos, fp=fp,
                           cg2d_params=cg2d_params, cg2dh=cg2dh, ds=ds, its=its, rundir=rundir)


def counters(s, it):
    """(myTime, myIter) at the start of step `it` (the_main_loop: myTime = startTime + deltaTClock*(it - nIter0))."""
    tp = s.prm.time
    return jnp.float64(tp.startTime + tp.deltaTClock*(it - tp.nIter0)), jnp.int32(it)


def _farr(a, like):
    a = np.asarray(a)
    if like.data.ndim == 3 and a.ndim == 4:
        assert a.shape[1] == 1, (like.name, a.shape)
        a = a[:, 0]
    assert a.shape == like.data.shape, (like.name, a.shape, like.data.shape)
    return FArray(jnp.asarray(a.astype(np.float64)), like.name, tiled=like.tiled, _dims=like.dims)


def dumped(s, it, stage):
    """{name: array} of every field the oracle dumps at (it, stage)."""
    return {n: s.ds.field(it, st, n) for (i, st, n) in s.ds.keys(it) if st == stage}


def teacher_state(s, it, sources):
    """A State of the build: every field NaN, then the fields of each (stage, names-or-None) in `sources` in order
    (later sources overwrite earlier ones; None = every State field the stage dumps)."""
    from mitjax.model.state import empty_state
    st = empty_state(s.cfg)
    for stage, names in sources:
        d = dumped(s, it, stage)
        upd = {}
        for n, a in d.items():
            if n in st and (names is None or n in names):
                upd[n] = _farr(a, getattr(st, n))
        st = st.replace(**upd)
    return st


def teacher_grid(s, it, stage, state=None):
    """The Grid with GRID_R from (it, stage) and recip_hFacW/S from G00 of `it` (or the State's NLFS fields)."""
    g = s.grid
    d = dumped(s, it, stage)
    upd = {n: _farr(d[n], getattr(g, n)) for n in GRID_R}
    g00 = dumped(s, it, "G00_geometry")
    upd["recip_hFacW"] = _farr(g00["recip_hFacW"], g.recip_hFacW)
    upd["recip_hFacS"] = _farr(g00["recip_hFacS"], g.recip_hFacS)
    return g.replace(**upd)


def teacher_ff(s, it, stage="S04_oceanic_phys"):
    """FFIELDS.h with the fields dumped at (it, stage) (group f), the rest zero (INI_FFIELDS); and phi0surf."""
    from mitjax.model.src.ini_ffields import ini_ffields
    ff = ini_ffields(cfg=s.cfg)
    d = dumped(s, it, stage)
    upd = {n: _farr(d[n], getattr(ff, n)) for n in ff.names() if n in d}
    ff = ff.replace(**upd)
    from mitjax.model.grid import declare
    phi0 = _farr(d["phi0surf"], declare("Bo_surf", s.cfg.size))
    return ff, phi0


def compare(values, ref):
    """(n points, n differing (==), n non-finite ours, n differing bit patterns) on every point incl. halos."""
    from mitjax.tests.r1_gate import compare_field
    return compare_field(values, ref)


def compare_stage(s, it, stage, values, names=None):
    """{field: compare} for every field dumped at (it, stage) that `values` (dict name -> FArray) holds."""
    d = dumped(s, it, stage)
    out = {}
    for n in sorted(d):
        if n in values and (names is None or n in names):
            out[n] = compare(values[n].data, d[n])
    return out


def bad(res):
    return {k: v for k, v in res.items() if v[0] == "shape" or any(v[1:])}


# ---------------------------------------------------------------------------------------------------------------
# stage programs (jitted; params traced)

def run_nlfs_reset(s, state, grid, it):
    from mitjax.model.src.forward_step import nlfs_reset
    cfg = s.cfg
    t, i = counters(s, it)
    f = jax.jit(lambda st, g, p, t, i: nlfs_reset(t, i, cfg=cfg, grid=g, params=p, state=st))
    return f(state, grid, s.params, t, i)


def run_dynamics(s, state, grid, ff, phi0surf, it):
    """DYNAMICS (forward_step.F:786-795) at the start counters of step `it` (myIter is updated after it)."""
    from mitjax.model.src.dynamics import dynamics
    cfg = s.cfg
    t, i = counters(s, it)
    f = jax.jit(lambda st, g, p, ff_, ph, t, i: dynamics(t, i, cfg=cfg, grid=g, params=p, state=st, ff=ff_,
                                                          fp=s.fp, phi0surf=ph))
    return f(state, grid, s.params, ff, phi0surf, t, i)


def cd_initial(s, state):
    """CD_CODE_INI_VARS on the oracle run directory (pickup_cd.0000036000)."""
    from mitjax.pkg.cd_code.cd_code_ini_vars import cd_code_ini_vars
    from mitjax.pkg.rw.read_rec import RW
    rw = RW(s.rundir, s.prm.init.readBinaryPrec, s.cfg.size)
    return cd_code_ini_vars(state, cfg=s.cfg, params=s.prm, ex=s.ex, rw=rw)


# ---------------------------------------------------------------------------------------------------------------
# session 2: S09-S11 (SOLVE_FOR_PRESSURE, MOMENTUM_CORRECTION_STEP, INTEGR_CONTINUITY) from S08 inputs

def start_state(s, it):
    """(State, Grid) at the start of step `it`: S00 + G00 R fields, the State's NLFS grid fields from S00 / G00."""
    from mitjax.model.state import NLFS_GRID
    st = teacher_state(s, it, [("S00_begin", None), ("G00_geometry", rsg.R_STATE)])
    g = teacher_grid(s, it, "S00_begin")
    return st.replace(**{n: getattr(g, n) for n in NLFS_GRID}), g


def after_s08(s, it):
    """(state, grid, cg2dh, ff) as SOLVE_FOR_PRESSURE of step `it` reads them: the dumped State up to S06 (S00, G00
    R, S01, S04, S05, S06), our UPDATE_R_STAR(.FALSE.) -> UPDATE_R_STAR(.TRUE.) grid and our UPDATE_CG2D operator
    (S01, S07, S08 bitwise in test_r4a_global_ocean.py), FFIELDS.h of S04, CD state of the State left NaN (etaNm1 is
    written by SOLVE_FOR_PRESSURE itself)."""
    from mitjax.model.src.forward_step import nlfs_reset, nlfs_update_cg2d, nlfs_update_hfac
    from mitjax.model.state import NLFS_CG2D, NLFS_GRID
    cfg = s.cfg
    st0, g0 = start_state(s, it)
    st0 = st0.replace(**{n: getattr(s.cg2dh, n) for n in NLFS_CG2D})
    t0, i0 = counters(s, it)
    t1, i1 = counters(s, it + 1)

    def f(st, g, p, cp, c, t0, i0, t1, i1):
        st, g = nlfs_reset(t0, i0, cfg=cfg, grid=g, params=p, state=st)
        st, g = nlfs_update_hfac(t1, i1, cfg=cfg, grid=g, params=p, state=st)
        st, c = nlfs_update_cg2d(t1, i1, cfg=cfg, grid=g, params=p, cg2d_params=cp, cg2dh=c, state=st, ex=s.ex)
        return st, g, c
    stn, g, c = jax.jit(f)(st0, g0, s.params, s.cg2d_params, s.cg2dh, t0, i0, t1, i1)
    st = teacher_state(s, it, [("S00_begin", None), ("G00_geometry", rsg.R_STATE), ("S01_update_rstar_F", None),
                               ("S04_oceanic_phys", None), ("S05_thermodynamics_sync", None),
                               ("S06_dynamics", None)])
    st = st.replace(**{n: getattr(stn, n) for n in NLFS_GRID + NLFS_CG2D})
    ff, _ = teacher_ff(s, it)
    return st, g, c, ff


def run_s09_s11(s, it, st, g, c, ff, params=None):
    """SOLVE_FOR_PRESSURE -> MOMENTUM_CORRECTION_STEP -> INTEGR_CONTINUITY (forward_step.F:897-930) at the updated
    counters of step `it`; returns {stage: State} for S09, S10, S11 and the solver scalars."""
    from mitjax.model.src.integr_continuity import integr_continuity
    from mitjax.model.src.momentum_correction_step import momentum_correction_step
    from mitjax.model.src.solve_for_pressure import solve_for_pressure
    cfg = s.cfg
    t1, i1 = counters(s, it + 1)

    def f(st, g, p, cp, c, ff_, t, i):
        st9, diag = solve_for_pressure(t, i, cfg=cfg, grid=g, params=p, state=st, ff=ff_, cg2dh=c, cg2d_params=cp,
                                       ex=s.ex)
        st10 = momentum_correction_step(t, i, cfg=cfg, grid=g, params=p, state=st9, ex=s.ex)
        st11 = integr_continuity(st10.uVel, st10.vVel, t, i, cfg=cfg, grid=g, params=p, state=st10, ex=s.ex,
                                 ff=ff_)
        return dict(S09_solve_for_pressure=st9, S10_momentum_correction=st10, S11_integr_continuity=st11), diag
    p = s.params if params is None else params
    return jax.jit(f)(st, g, p, s.cg2d_params, c, ff, t1, i1)


# ---------------------------------------------------------------------------------------------------------------
# session 2: the GO stages of one step under jit(shard_map(check_vma=True)) on P devices vs P = 1

def chain_body(s):
    """body(st_start, st_dyn, grid, params, cg2d_params, cg2dh, ff, phi0surf, ex, t0, i0, t1, i1): forward_step.F
    :461-494 nlfs_reset, DYNAMICS (inputs teacher-forced: st_dyn holds the S04/S05 values THERMODYNAMICS and
    DO_OCEANIC_PHYS leave; its NLFS grid fields are replaced by nlfs_reset's), :832-875 nlfs_update_hfac /
    nlfs_update_cg2d, :897-930 SOLVE_FOR_PRESSURE, MOMENTUM_CORRECTION_STEP, INTEGR_CONTINUITY, :939-961
    nlfs_calc_r_star. Returns (state after S01, state after S12, CALC_R_STAR counters, solver scalars)."""
    from mitjax.model.src.dynamics import dynamics
    from mitjax.model.src.forward_step import nlfs_calc_r_star, nlfs_reset, nlfs_update_cg2d, nlfs_update_hfac
    from mitjax.model.src.integr_continuity import integr_continuity
    from mitjax.model.src.momentum_correction_step import momentum_correction_step
    from mitjax.model.src.solve_for_pressure import solve_for_pressure
    from mitjax.model.state import NLFS_GRID
    cfg = s.cfg

    def body(st0, std, g, p, cp, c, ff, phi0, ex, t0, i0, t1, i1):
        st1, g1 = nlfs_reset(t0, i0, cfg=cfg, grid=g, params=p, state=st0)
        st = std.replace(**{n: getattr(st1, n) for n in NLFS_GRID})
        st = dynamics(t0, i0, cfg=cfg, grid=g1, params=p, state=st, ff=ff, fp=s.fp, phi0surf=phi0)
        st, g2 = nlfs_update_hfac(t1, i1, cfg=cfg, grid=g1, params=p, state=st)
        st, c2 = nlfs_update_cg2d(t1, i1, cfg=cfg, grid=g2, params=p, cg2d_params=cp, cg2dh=c, state=st, ex=ex)
        st, diag = solve_for_pressure(t1, i1, cfg=cfg, grid=g2, params=p, state=st, ff=ff, cg2dh=c2,
                                      cg2d_params=cp, ex=ex)
        st = momentum_correction_step(t1, i1, cfg=cfg, grid=g2, params=p, state=st, ex=ex)
        st = integr_continuity(st.uVel, st.vVel, t1, i1, cfg=cfg, grid=g2, params=p, state=st, ex=ex, ff=ff)
        st, cnt = nlfs_calc_r_star(t1, i1, cfg=cfg, grid=g2, params=p, state=st, ex=ex)
        return st1, st, cnt, diag
    return body


def chain_inputs(s, it):
    """Teacher-forced inputs of chain_body at step `it` (CD state of CD_CODE_INI_VARS at the first dumped step)."""
    from mitjax.model.state import NLFS_CG2D
    st0, g0 = start_state(s, it)
    st0 = st0.replace(**{n: getattr(s.cg2dh, n) for n in NLFS_CG2D})
    std = teacher_state(s, it, [("S00_begin", None), ("G00_geometry", rsg.R_STATE), ("S01_update_rstar_F", None),
                                ("S04_oceanic_phys", None), ("S05_thermodynamics_sync", None)])
    std = std.replace(**{n: getattr(s.cg2dh, n) for n in NLFS_CG2D})
    std = cd_initial(s, std)
    ff, phi0 = teacher_ff(s, it)
    t0, i0 = counters(s, it)
    t1, i1 = counters(s, it + 1)
    return (st0, std, g0, s.params, s.cg2d_params, s.cg2dh, ff, phi0), (t0, i0, t1, i1)


def run_chain(s, it, nproc=None):
    """chain_body at P = 1 (nproc None) or inside jit(shard_map(check_vma=True)) on nproc fake CPU devices (outputs
    gathered, padding tiles dropped)."""
    import numpy as np
    from mitjax.eesupp.exch_maps import load_maps
    from mitjax.eesupp.shard import TileSharding
    body = chain_body(s)
    args, clk = chain_inputs(s, it)
    if nproc is None:
        out = jax.jit(lambda *a: body(*a[:8], s.ex, *a[8:]))(*args, *clk)
        return jax.tree.map(np.asarray, out)
    sh = TileSharding(load_maps(EXP[0]), nproc)

    def spec(t):
        return jax.tree.map(lambda x: sh.TILES if (isinstance(x, FArray) and x.tiled) else sh.REP, t,
                            is_leaf=lambda x: isinstance(x, FArray))
    placed = tuple(sh.put_tree(a) for a in args)
    out1 = jax.eval_shape(lambda *a: body(*a[:8], s.ex, *a[8:]), *args, *clk)
    st_spec = spec(args[0])
    cnt_spec = jax.tree.map(lambda _: sh.REP, out1[2])
    diag_spec = jax.tree.map(lambda _: sh.REP, out1[3])
    f = sh.shard_map(lambda *a: body(*a), in_specs=tuple(spec(a) for a in args) + (sh.TILES,) + (sh.REP,) * 4,
                     out_specs=(st_spec, st_spec, cnt_spec, diag_spec))
    st1, st12, cnt, diag = f(*placed, sh.ex, *clk)
    return (sh.unpad_tree(st1), sh.unpad_tree(st12), jax.tree.map(np.asarray, cnt), jax.tree.map(np.asarray, diag))


# ---------------------------------------------------------------------------------------------------------------
# session 5: the driver's Model (no teacher forcing), its steps probed at every dumped stage

def driver_model(tag="go"):
    """The run driver's Model of global_ocean.90x40x15/input in a new run directory (as `python -m mitjax run`)."""
    from mitjax.drivers.model import Model as DriverModel
    from mitjax.drivers.run import load_experiment, make_rundir
    from mitjax.tests import advect_gate as ag
    exp_dir = paths_upstream() / "verification" / EXP[0]
    e = load_experiment(exp_dir, EXP[1])
    rundir = make_rundir(exp_dir, EXP[1], ag.out_dir(f"{EXP[0]}-{EXP[1]}-{tag}"))
    return DriverModel(e, rundir)


def paths_upstream():
    from mitjax import paths
    return paths.UPSTREAM


def probed_steps(m, n, stages):
    """n FORWARD_STEPs of the driver's Model from its initial carry (the scan driver's step, with forward_step's
    probe): [(start iteration, {stage: values})]."""
    from mitjax.farray import FArray
    from mitjax.model.src.forward_step import forward_step
    cfg, fp, pks = m.cfg, m.fp, m.pks

    def step(a, carry, iloop, myTime, myIter):
        probes = {}

        def probe(stage, values):
            if stage in stages:
                probes[stage] = values
        state, ff, phi0surf = carry[:3]
        pk = carry[4] if len(carry) > 4 else None
        state, ff, phi0surf, myTime, myIter, out = forward_step(
            iloop, myTime, myIter, cfg=cfg, grid=a.grid, params=a.params, fp=fp, eos=a.eos, cg2dh=a.cg2dh,
            cg2d_params=a.cg2d_params, state=state, ff=ff, phi0surf=phi0surf, ex=a.ex, probe=probe, pk=pk,
            pkc=a.pkc, pks=pks)
        flow = tuple(FArray(f.data, c.name, tiled=c.tiled, _dims=c.dims) for f, c in zip(out["flow"], carry[3]))
        return (state, ff, phi0surf, flow) + ((out["pk"],) if pk else ()), myTime, myIter, probes
    f = jax.jit(step)
    carry = m.initial_carry()
    t, it = m.start_counters()
    res = []
    for k in range(n):
        it0 = int(it)
        carry, t, it, probes = f(m.arrays, carry, jnp.int32(k + 1), t, it)
        res.append((it0, probes))
    return res, carry


def run_steps_p(m, n, nproc=None, maps=None):
    """n FORWARD_STEPs of the driver's Model (its own step, make_step) from its initial carry: at P = 1 (nproc None,
    jit) or inside jit(shard_map(check_vma=True)) on nproc fake CPU devices (Arrays and carry placed with
    sharded_grad.place_model / tile_specs; the per-step scalars replicated). Returns (final carry gathered and
    unpadded, [per-step solver / r* outputs])."""
    import numpy as np
    from mitjax.drivers.sharded_grad import place_model, tile_specs
    from mitjax.eesupp.exch_maps import load_maps
    from mitjax.eesupp.shard import TileSharding
    carry, (t, it) = m.initial_carry(), m.start_counters()
    outs = []
    if nproc is None:
        f = jax.jit(m.step)
        for k in range(n):
            carry, t, it, c = f(m.arrays, carry, jnp.int32(k + 1), t, it)
            outs.append(jax.tree.map(np.asarray, c))
        return jax.tree.map(np.asarray, carry), outs
    sh = TileSharding(load_maps(m.cfg.experiment) if maps is None else maps, nproc)   # maps: e.g. the cube's
    arrays = m.arrays
    # the periodic-forcing preload's record sets are tiled FArrays (R5 session 5), placed and sharded on their tile
    # axis by place_model / tile_specs like every tiled field
    a4, c4 = place_model(sh, arrays), place_model(sh, carry)
    a_s, c_s = tile_specs(sh, arrays), tile_specs(sh, carry)
    c_shape = jax.eval_shape(m.step, m.arrays, carry, jnp.int32(1), t, it)[3]
    f = sh.shard_map(m.step, in_specs=(a_s, c_s, sh.REP, sh.REP, sh.REP),
                     out_specs=(c_s, sh.REP, sh.REP, jax.tree.map(lambda _: sh.REP, c_shape)))
    for k in range(n):
        c4, t, it, c = f(a4, c4, jnp.int32(k + 1), t, it)
        outs.append(jax.tree.map(np.asarray, c))
    return sh.unpad_tree(c4), outs



def nonfinite_leaves(carry):
    """{leaf name: number of non-finite values} over the floating leaves of a driver carry (State fields by name,
    the other carry parts by tree path)."""
    import numpy as np

    def scan(prefix, tree, out):
        items = jax.tree_util.tree_flatten_with_path(tree)[0]
        for k, x in items:
            a = np.asarray(x)
            if np.issubdtype(a.dtype, np.floating) and not np.all(np.isfinite(a)):
                out[prefix if len(items) == 1 else prefix + jax.tree_util.keystr(k)] = int(
                    np.count_nonzero(~np.isfinite(a)))
    out = {}
    state = carry[0]
    for n in state.names():
        scan(n, getattr(state, n), out)
    for i, part in enumerate(carry[1:], 1):
        scan(f"carry[{i}]", part, out)
    return out


# The storage of the M1 variants that holds NaN (mitjax/model/state.py empty_state, pkg/cost/cost_init_varia.py) at
# the end of a whole run, with the Fortran's reason (read at `pinned`; docs/M1_ACCEPTANCE.md section 5). In every
# case the Fortran neither initialises nor reads the points during the run (the COMMON block holds the loader's
# zero), and no backward pass reads them (test_m1_pn.py::test_backward_does_not_read_nan_storage), so the port keeps
# its NaN: a read of one would reach %MON.
def _whole(sz, nr=1):
    return sz.nSx * sz.nSy * sz.nPx * sz.nPy * nr * (sz.sNx + 2 * sz.OLx) * (sz.sNy + 2 * sz.OLy)


def _rim(sz):
    return sz.nSx * sz.nSy * sz.nPx * sz.nPy * ((sz.sNx + 2 * sz.OLx) + (sz.sNy + 2 * sz.OLy) - 1)


NAN_STORAGE = {
    "advect_xz": [
        (f, _whole, "MOM_FLUXFORM.h dWtrans (r* code): MOM_CALC_RTRANS (mom_calc_rtrans.F:111-164) is a momentum "
         "routine and momStepping=.FALSE.; the ALLOW_AUTODIFF reset (dynamics.F:326-331) is not compiled")
        for f in ("dWtransC", "dWtransU", "dWtransV")],
    "global_ocean.90x40x15": [
        (f, _rim, "mom_calc_rtrans.F:123-129 and :140-148 write i >= 2-OLx, j >= 2-OLy only and :150-157 read the "
         "same range: row j = 1-OLy and column i = 1-OLx of every tile are never written nor read")
        for f in ("dWtransU", "dWtransV")],
    # lane M4LAB session 2 (drivers/model.py): the PTRACERS_FIELDS.h fields of the compiled but unused pkg/ptracers
    # (usePTRACERS = .FALSE.; PTRACERS_INIT_VARIA, packages_init_variables.F:329-334, and every reader skipped) hold
    # the never-written static common's zero, as the oracle dumps them, no longer NaN: not in this table
    "tutorial_global_oce_optim": [
        (f"carry[4]['cost'].{f}", lambda sz: sz.nSx * sz.nSy * sz.nPx * sz.nPy, "cost.h: not initialised (no "
         "COST_INIT_VARIA line); COST_FINAL writes it (cost_final.F:134-138, COST_HFLUX / COST_TEMP) before it "
         "reads it (:180-203), after the time loop")
        for f in ("objf_temp_tut", "objf_hflux_tut")],
}


def expected_nonfinite(exp, size):
    """{leaf name: count} that nonfinite_leaves must give for experiment `exp` after a whole run (NAN_STORAGE)."""
    return {f: n(size) for f, n, _ in NAN_STORAGE.get(exp, [])}


def whole_run_pn(exp, inp, nproc, tag="pn"):
    """The run driver's Model of exp/inp (as `python -m mitjax run`) and its whole run at P = 1 and, with nproc,
    at P = nproc (run_steps_p): (m, c1, o1, cN, oN, differing leaves, outputs equal)."""
    import numpy as np
    from mitjax.drivers.model import Model as DriverModel
    from mitjax.drivers.run import load_experiment, make_rundir
    from mitjax.tests import advect_gate as ag
    exp_dir = paths_upstream() / "verification" / exp
    m = DriverModel(load_experiment(exp_dir, inp), make_rundir(exp_dir, inp, ag.out_dir(f"{exp}-{inp}-{tag}")))
    n = m.prm.time.nTimeSteps
    c1, o1 = run_steps_p(m, n)
    if not nproc:
        return m, c1, o1, None, None, None, None
    cN, oN = run_steps_p(m, n, nproc)
    def bits(x):                                                   # bit patterns (NaN payloads included)
        a = np.asarray(x)
        return a.dtype, a.shape, np.ascontiguousarray(a).tobytes()
    la, lb = jax.tree.leaves(c1), jax.tree.leaves(cN)
    assert len(la) == len(lb)
    ndiff = sum(bits(a) != bits(b) for a, b in zip(la, lb))
    same = len(o1) == len(oN) and all(
        jax.tree.structure(p_) == jax.tree.structure(q_)
        and all(bits(x) == bits(y) for x, y in zip(jax.tree.leaves(p_), jax.tree.leaves(q_))) for p_, q_ in zip(o1, oN))
    return m, c1, o1, cN, oN, ndiff, same
