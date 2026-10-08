"""Helpers of the r* / nonlinear free-surface gates (M1 lane RSTAR, plan Task 15a): the ported routines run under jit
at the gate XLA flags (conftest.py) with every float a traced argument, on inputs teacher-forced from the oracle's
substep dumps, and their outputs compared with the dumped outputs on every point of every tile, halos included.

Oracle: the registered dumps-on run of kind `jdon3` of each variant (reference/reference_runs.py: build jaxdump3,
which dumps rStarDhCDt at S12_calc_rstar): global_ocean.90x40x15/input steps 36000-36002, advect_xz/input.nlfs and
advect_xz/input steps 0, 1, 2, 9. Stages (reference/jaxdump/SUBSTEPS.md): G00_geometry (groups G, V, R: grid, masks,
recip_hFacW/S, kSurf*, h0Fac*, and the r* fields rStarFacNm1*, rStarExp*, rStarDh*Dt, pStarFacK, as the previous step
left them), S00_begin / S01_update_rstar_F / S07_update_rstar_T / S12_calc_rstar (group r: hFacC/W/S, recip_hFacC,
rStarFacC/W/S, rStarDhCDt), S11_integr_continuity (group d: etaH), S08_update_cg2d (group c).

Which values each routine sees (the order of forward_step.F, [L-LIT-14]):
  * :465-494 (doResetHFactors; global_ocean): RESET_NLFS_VARS, UPDATE_R_STAR(.FALSE.) after G00 (anchor :434) ->
    S01; reads rStarFacNm1* = G00 R of the step;
  * :839 UPDATE_R_STAR(.TRUE.) -> S07; reads rStarFacC/W/S as S00 holds them (CALC_R_STAR runs only at :949);
  * :949 CALC_R_STAR(etaH) -> S12 (r group) and G00 R of the NEXT step (nothing writes the r* fields in between:
    G00 of step n+1 is dumped before :467);
  * recip_hFacW/S are dumped only in G00 (group G): the value UPDATE_R_STAR(.TRUE.) of the previous step left (or
    INITIALISE_VARIA's, initialise_varia.F:307, at the first step); at mask = 0 points every call keeps the prior
    value, so G00 of the step is the prior of both calls of the step.
The stand-in `params` holds the PARAMS.h values the routines read (ini_parms_dyn raises for these variants until the
core lane extends it: rhoConstFresh in global_ocean's data, no ALLOW_MOM_COMMON in advect_xz), each cited.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.farray import FArray
from mitjax.io.dump import DumpSet
from mitjax.model.grid import Grid
from mitjax.model.src.ini_parms import Params, _get
from mitjax.model.src.ini_parms_rstar import ini_parms_rstar
from mitjax.model.state import State
from mitjax.params_io import RunParams
from mitjax.tests import grid_gate as gg
from mitjax.tests import init_gate as ig

RSTAR_VARIANTS = [("global_ocean.90x40x15", "input"), ("advect_xz", "input.nlfs")]
SURFDR_VARIANTS = [("advect_xz", "input"), ("advect_xz", "input.pqm")]

GRID_XY = ("Ro_surf", "R_low", "recip_Rcol", "rSurfW", "rSurfS", "rLowW", "rLowS", "rA", "recip_rAw", "recip_rAs")
GRID_INT = ("kSurfC", "kSurfW", "kSurfS")
GRID_XYZ = ("maskC", "maskW", "maskS", "recip_hFacW", "recip_hFacS")
R_XYZ = ("h0FacC", "h0FacW", "h0FacS")
R_STATE = ("rStarFacNm1C", "rStarFacNm1W", "rStarFacNm1S", "rStarExpC", "rStarExpW", "rStarExpS",
           "rStarDhCDt", "rStarDhWDt", "rStarDhSDt", "pStarFacK")
R_FAC = ("rStarFacC", "rStarFacW", "rStarFacS")
HFAC = ("hFacC", "hFacW", "hFacS", "recip_hFacC")
CALC_OUT_S12 = ("rStarFacC", "rStarFacW", "rStarFacS", "rStarDhCDt")
CALC_OUT_G00 = ("rStarFacNm1C", "rStarFacNm1W", "rStarFacNm1S", "rStarExpC", "rStarExpW", "rStarExpS",
                "rStarDhCDt", "rStarDhWDt", "rStarDhSDt")

experiment = gg.experiment
exchanger = functools.lru_cache(maxsize=None)(gg.exchanger)


@functools.lru_cache(maxsize=None)
def oracle(exp, inp, kind="jdon3"):
    """(DumpSet, dumped iterations, run directory) of the registered dumps-on run of `kind`."""
    reg = gg._registry()
    runs = [r for r in reg.RUNS if (r["exp"], r["input"], r["kind"]) == (exp, inp, kind)]
    if len(runs) != 1:
        raise FileNotFoundError(f"{exp}/{inp}: {len(runs)} registered runs of kind {kind}")
    top = reg.run_top(runs[0])
    ds = DumpSet(top / "dumps")
    return ds, tuple(ds.iterations()), top / "rundir"


@functools.lru_cache(maxsize=None)
def params(exp, inp):
    """Stand-in Params (module docstring) + ini_parms_rstar: static vectorInvariantMomentum (set_defaults.F:193),
    nonlinFreeSurf (:257), select_rStar (:260), fluidIsAir (ini_parms.F:449-453: .TRUE. only for an atmospheric
    buoyancyRelation; the M1 variants are OCEANIC), traced deltaTFreeSurf (TimeParams: ini_parms.F:1068)."""
    e = experiment(exp, inp)
    rp = RunParams(e.run)
    prm = ig.params(exp, inp)
    if prm.grid.buoyancyRelation != "OCEANIC":
        raise NotImplementedError(f"rstar_gate.params: buoyancyRelation {prm.grid.buoyancyRelation}")
    static = dict(vectorInvariantMomentum=bool(_get(e, rp, "PARM01", "vectorInvariantMomentum",
                                                    "model/src/set_defaults.F:193")),
                  nonlinFreeSurf=prm.init.nonlinFreeSurf, select_rStar=prm.init.select_rStar, fluidIsAir=False)
    traced = dict(deltaTFreeSurf=np.float64(prm.time.deltaTFreeSurf))
    return ini_parms_rstar(e, Params(static, traced))


# ---------------------------------------------------------------------------------------------------------------
# dumped fields as FArrays

def _xy(a, name, sz, dtype=np.float64):
    a = np.asarray(a)
    assert a.shape[1] == 1, (name, a.shape)
    return FArray(jnp.asarray(a[:, 0].astype(dtype)), name, i=(1 - sz.OLx, sz.sNx + sz.OLx),
                  j=(1 - sz.OLy, sz.sNy + sz.OLy))


def _xyz(a, name, sz):
    return FArray(jnp.asarray(np.asarray(a)), name, i=(1 - sz.OLx, sz.sNx + sz.OLx), j=(1 - sz.OLy, sz.sNy + sz.OLy),
                  k=(1, sz.Nr))


def field(exp, inp, it, stage, name):
    ds, _, _ = oracle(exp, inp)
    return ds.field(it, stage, name)


def teacher_grid(exp, inp, it, hfac_stage):
    """Grid of the fields the r* routines read, from G00 of iteration `it`; hFacC/W/S and recip_hFacC (group r) from
    `hfac_stage` of the same iteration (the prior values of the call)."""
    e = experiment(exp, inp)
    sz = e.cfg.size
    f = {}
    for n in GRID_XY:
        f[n] = _xy(field(exp, inp, it, "G00_geometry", n), n, sz)
    for n in GRID_INT:
        a = field(exp, inp, it, "G00_geometry", n)
        assert np.array_equal(a, np.round(a)), n                    # integers dumped as exact reals
        f[n] = _xy(a, n, sz, dtype=np.int32)
    for n in GRID_XYZ + R_XYZ:
        f[n] = _xyz(field(exp, inp, it, "G00_geometry", n), n, sz)
    for n in HFAC:
        f[n] = _xyz(field(exp, inp, it, hfac_stage, n), n, sz)
    return Grid(f)


def teacher_state(exp, inp, it, fac_stage, rstate_it=None):
    """State of the r* fields: rStarFacC/W/S from `fac_stage` (group r) of `it`; the G00 R fields (R_STATE) of
    `rstate_it` (default `it`)."""
    e = experiment(exp, inp)
    sz = e.cfg.size
    it2 = it if rstate_it is None else rstate_it
    f = {n: _xy(field(exp, inp, it2, "G00_geometry", n), n, sz) for n in R_STATE}
    for n in R_FAC:
        f[n] = _xy(field(exp, inp, it, fac_stage, n), n, sz)
    return State(f)


# ---------------------------------------------------------------------------------------------------------------
# comparison

def compare(ours, ref):
    """(n compared, n differing, n non-finite ours, n non-finite oracle, n differing bit patterns): every point of
    every tile, halos included; "differing" is element inequality (+0 == -0), the bit count also sees the sign of a
    zero. A 2-D FArray is compared with the dump's k = 1 record."""
    a = np.asarray(ours.data if isinstance(ours, FArray) else ours)
    r = np.asarray(ref)
    if a.ndim == 3:
        a = a[:, None]
    if a.shape != r.shape:
        return ("shape", a.shape, r.shape)
    a64, r64 = np.ascontiguousarray(a, np.float64), np.ascontiguousarray(r, np.float64)
    return (a.size, int(np.count_nonzero(~(a == r))), int(np.count_nonzero(~np.isfinite(a))),
            int(np.count_nonzero(~np.isfinite(r))), int(np.count_nonzero(a64.view(np.int64) != r64.view(np.int64))))


def failures(result):
    return {k: v for k, v in result.items() if v[0] == "shape" or any(v[1:])}


# ---------------------------------------------------------------------------------------------------------------
# the routines under jit (floats traced: grid, state, params, etaFld are jit arguments)

def run_calc_r_star(exp, inp, etaFld, grid, state, prm=None, ex=None, fn=None):
    from mitjax.model.src.calc_r_star import calc_r_star
    e = experiment(exp, inp)
    fn = fn or calc_r_star
    prm = prm if prm is not None else params(exp, inp)
    if ex is not None and not hasattr(ex, "tree_flatten"):        # a planted-error wrapper: closed over, not traced
        f = jax.jit(lambda eta, g, s, p: fn(eta, None, None, cfg=e.cfg, grid=g, params=p, state=s, ex=ex))
        return f(etaFld, grid, state, prm)
    f = jax.jit(lambda eta, g, s, p, x: fn(eta, None, None, cfg=e.cfg, grid=g, params=p, state=s, ex=x))
    return f(etaFld, grid, state, prm, ex if ex is not None else exchanger(exp))


def run_update_r_star(exp, inp, useLatest, grid, state, fn=None):
    from mitjax.model.src.update_r_star import update_r_star
    e = experiment(exp, inp)
    fn = fn or update_r_star
    f = jax.jit(lambda g, s: fn(useLatest, None, None, cfg=e.cfg, grid=g, state=s))
    return f(grid, state)


def run_reset_nlfs_vars(exp, inp, state):
    from mitjax.model.src.reset_nlfs_vars import reset_nlfs_vars
    e = experiment(exp, inp)
    f = jax.jit(lambda s, p: reset_nlfs_vars(None, None, cfg=e.cfg, params=p, state=s))
    return f(state, params(exp, inp))


def run_update_surf_dr(exp, inp, grid):
    from mitjax.model.src.update_surf_dr import update_surf_dr
    e = experiment(exp, inp)
    f = jax.jit(lambda g, p: update_surf_dr(True, None, None, cfg=e.cfg, grid=g, params=p))
    return f(grid, params(exp, inp))


# ---------------------------------------------------------------------------------------------------------------
# per-step gates (results: {label: compare tuple})

def gate_calc_r_star(exp, inp, it, **kw):
    """CALC_R_STAR at forward_step.F:949 of step `it`: inputs etaH (S11), rStarFacC/W/S (S07), the G00 R fields
    (prior values, all overwritten) and G00 grid; outputs vs S12 (r group) and, if dumped, G00 R of the next step.
    Returns (results, counters, number of zero denominators of :306-311 on the inputs)."""
    sz = experiment(exp, inp).cfg.size
    _, its, _ = oracle(exp, inp)
    eta = _xy(field(exp, inp, it, "S11_integr_continuity", "etaH"), "etaH", sz)
    grid = teacher_grid(exp, inp, it, "S07_update_rstar_T")
    state = teacher_state(exp, inp, it, "S07_update_rstar_T")
    zero_den = sum(int(np.count_nonzero(np.asarray(getattr(state, n).data) == 0.)) for n in R_FAC)
    out, counters = run_calc_r_star(exp, inp, eta, grid, state, **kw)
    res = {f"S12/{n}": compare(getattr(out, n), field(exp, inp, it, "S12_calc_rstar", n)) for n in CALC_OUT_S12}
    if it + 1 in its:
        for n in CALC_OUT_G00 + ("pStarFacK",):
            res[f"G00+1/{n}"] = compare(getattr(out, n), field(exp, inp, it + 1, "G00_geometry", n))
    return res, counters, zero_den


def gate_update_r_star(exp, inp, it, useLatest, fn=None):
    """UPDATE_R_STAR(.TRUE.) at :839 (outputs S07, recip_hFacW/S vs G00 of the next step) or (.FALSE.) at :475
    (outputs S01; recip_hFacW/S vs G00 of the same step: the value the Fortran computed from the same rStarFacNm1 at
    the previous :839, module docstring)."""
    _, its, _ = oracle(exp, inp)
    if useLatest:
        prior = "S01_update_rstar_F" if ("S01_update_rstar_F" in stages(exp, inp, it)) else "S00_begin"
        grid = teacher_grid(exp, inp, it, prior)
        state = teacher_state(exp, inp, it, "S00_begin")
        out_stage, w_it = "S07_update_rstar_T", it + 1
    else:
        grid = teacher_grid(exp, inp, it, "S00_begin")
        state = teacher_state(exp, inp, it, "S00_begin")
        out_stage, w_it = "S01_update_rstar_F", it
    out = run_update_r_star(exp, inp, useLatest, grid, state, fn=fn)
    res = {f"{out_stage[:3]}/{n}": compare(getattr(out, n), field(exp, inp, it, out_stage, n)) for n in HFAC}
    if w_it in its:
        for n in ("recip_hFacW", "recip_hFacS"):
            res[f"G00{'+1' if useLatest else ''}/{n}"] = compare(getattr(out, n),
                                                                 field(exp, inp, w_it, "G00_geometry", n))
    return res


def stages(exp, inp, it):
    ds, _, _ = oracle(exp, inp)
    return {s for (_, s, _) in ds.keys(it)}


def gate_reset_nlfs_vars(exp, inp, it):
    """RESET_NLFS_VARS at :467 of step `it`: pStarFacK vs G00 R of the next step (CALC_R_STAR does not write it)."""
    state = teacher_state(exp, inp, it, "S00_begin")
    out = run_reset_nlfs_vars(exp, inp, state)
    return {"G00+1/pStarFacK": compare(out.pStarFacK, field(exp, inp, it + 1, "G00_geometry", "pStarFacK"))}


def gate_update_surf_dr(exp, inp, it):
    """UPDATE_SURF_DR(.TRUE.) at :852 of step `it` (linear free surface of a NONLIN_FRSURF build): prior hFac from
    S00 of `it`; hFacC, recip_hFacC (and the untouched hFacW/S) vs S00 of the next step."""
    grid = teacher_grid(exp, inp, it, "S00_begin")
    out = run_update_surf_dr(exp, inp, grid)
    return {f"S00+1/{n}": compare(getattr(out, n), field(exp, inp, it + 1, "S00_begin", n)) for n in HFAC}


# ---------------------------------------------------------------------------------------------------------------
# INITIALISE_VARIA's r* sequence (initialise_varia.F:297-346)

def init_chain(exp, inp, calc_fn=None):
    """INI_NLFS_VARS (core lane, through init_gate.build_state: the State INITIALISE_VARIA leaves before its pending
    r* calls), then :302 CALC_R_STAR(etaH), :307 UPDATE_R_STAR(.TRUE.), :341 CALC_R_STAR(etaH) with the etaH that
    INTEGR_CONTINUITY (:334, exactConserv: not ported yet) leaves = S00_begin's etaH of the first step (nothing
    writes etaH in between). Grid: G00 of the first step with the hFac priors of S00 (INI_MASKS_ETC's values at
    mask = 0 points, which no call writes). Returns (results vs S00 r / G00 R / G00 recip_hFacW/S of the first
    step, counters of both calls, etaH check of the first call vs I02_ini_fields)."""
    e = experiment(exp, inp)
    sz = e.cfg.size
    _, its, _ = oracle(exp, inp)
    it0 = its[0]
    st0, _ = ig.build_state(exp, inp)
    names = R_STATE + R_FAC
    state = State({n: getattr(st0, n) for n in names})
    eta1 = st0.etaH
    eta_ok = compare(eta1, field(exp, inp, it0, "I02_ini_fields", "etaH"))
    grid = teacher_grid(exp, inp, it0, "S00_begin")
    state, c1 = run_calc_r_star(exp, inp, eta1, grid, state, fn=calc_fn)                  # :302
    grid = run_update_r_star(exp, inp, True, grid, state)                                # :307
    eta2 = _xy(field(exp, inp, it0, "S00_begin", "etaH"), "etaH", sz)
    state, c2 = run_calc_r_star(exp, inp, eta2, grid, state, fn=calc_fn)                 # :341
    res = {}
    for n in R_FAC + ("rStarDhCDt",):
        res[f"S00/{n}"] = compare(getattr(state, n), field(exp, inp, it0, "S00_begin", n))
    for n in HFAC:
        res[f"S00/{n}"] = compare(getattr(grid, n), field(exp, inp, it0, "S00_begin", n))
    for n in R_STATE:
        res[f"G00/{n}"] = compare(getattr(state, n), field(exp, inp, it0, "G00_geometry", n))
    for n in ("recip_hFacW", "recip_hFacS"):
        res[f"G00/{n}"] = compare(getattr(grid, n), field(exp, inp, it0, "G00_geometry", n))
    return res, (c1, c2), eta_ok, grid


# ---------------------------------------------------------------------------------------------------------------
# P = N (fake CPU devices of the gate flags)

def run_sharded(exp, inp, it, nproc=4):
    """CALC_R_STAR then UPDATE_R_STAR(.TRUE.) of step `it` (gate_calc_r_star's inputs) inside
    jit(shard_map(check_vma=True)) on `nproc` devices: (state, counters, grid) gathered and unpadded, and the same
    two calls at P=1."""
    from mitjax.eesupp.exch_maps import load_maps
    from mitjax.eesupp.shard import TileSharding
    from mitjax.model.src.calc_r_star import calc_r_star
    from mitjax.model.src.update_r_star import update_r_star
    e = experiment(exp, inp)
    sz = e.cfg.size
    eta = _xy(field(exp, inp, it, "S11_integr_continuity", "etaH"), "etaH", sz)
    grid = teacher_grid(exp, inp, it, "S07_update_rstar_T")
    state = teacher_state(exp, inp, it, "S07_update_rstar_T")
    prm = params(exp, inp)

    def body(eta, g, s, p, x):
        s, c = calc_r_star(eta, None, None, cfg=e.cfg, grid=g, params=p, state=s, ex=x)
        g = update_r_star(True, None, None, cfg=e.cfg, grid=g, state=s)
        return s, c, g

    one = jax.jit(body)(eta, grid, state, prm, exchanger(exp))
    sh = TileSharding(load_maps(exp), nproc)
    spec = lambda t: jax.tree.map(lambda x: sh.TILES if (isinstance(x, FArray) and x.tiled) else sh.REP, t,  # noqa
                                  is_leaf=lambda x: isinstance(x, FArray))
    args = (sh.put_tree(eta), sh.put_tree(grid), sh.put_tree(state), sh.put_tree(prm))
    cspec = {k: sh.REP for k in ("icntc1", "icntw", "icnts", "icntc2", "maxhFacC",
                                 "nzeroC", "nzeroW", "nzeroS")}   # GO lane: + the zero-denominator counts
    f = sh.shard_map(body, in_specs=tuple(spec(a) for a in args) + (sh.TILES,),
                     out_specs=(spec(state), cspec, spec(grid)))
    s4, c4, g4 = f(*args, sh.ex)
    return (sh.unpad_tree(s4), jax.tree.map(np.asarray, c4), sh.unpad_tree(g4)), one


# ---------------------------------------------------------------------------------------------------------------
# gradients

def objective_inputs(exp, inp, it):
    sz = experiment(exp, inp).cfg.size
    eta = _xy(field(exp, inp, it, "S11_integr_continuity", "etaH"), "etaH", sz)
    return eta, teacher_grid(exp, inp, it, "S07_update_rstar_T"), teacher_state(exp, inp, it, "S07_update_rstar_T")


def objective(exp, inp, grid, state, weights):
    """J(etaH) = sum of every output of CALC_R_STAR and of UPDATE_R_STAR(.TRUE.) on its output, each times a fixed
    random weight field (all points incl. halos and land): the derivative reaches every lane."""
    from mitjax.model.src.calc_r_star import calc_r_star
    from mitjax.model.src.update_r_star import update_r_star
    e = experiment(exp, inp)
    prm = params(exp, inp)
    ex = exchanger(exp)

    def J(eta_data):
        eta = FArray(eta_data, "etaH", _dims=grid.Ro_surf.dims)
        s, _ = calc_r_star(eta, None, None, cfg=e.cfg, grid=grid, params=prm, state=state, ex=ex)
        g = update_r_star(True, None, None, cfg=e.cfg, grid=grid, state=s)
        tot = 0.
        for n in CALC_OUT_G00 + R_FAC:
            tot = tot + jnp.sum(getattr(s, n).data * weights[n])
        for n in HFAC + ("recip_hFacW", "recip_hFacS"):
            tot = tot + jnp.sum(getattr(g, n).data * weights[n])
        return tot
    return J


def weights(exp, inp, grid, state, seed=0):
    rng = np.random.default_rng(seed)
    w = {}
    for n in CALC_OUT_G00 + R_FAC:
        w[n] = jnp.asarray(rng.standard_normal(np.shape(getattr(state, n).data)))
    for n in HFAC + ("recip_hFacW", "recip_hFacS"):
        w[n] = jnp.asarray(rng.standard_normal(np.shape(getattr(grid, n).data)))
    # rStarDh*Dt carry 1/deltaTFreeSurf: scale so that every term is O(1) per unit eta change
    dt = float(params(exp, inp).deltaTFreeSurf)
    for n in ("rStarDhCDt", "rStarDhWDt", "rStarDhSDt"):
        w[n] = w[n] * dt
    return w


# ---------------------------------------------------------------------------------------------------------------
# the RSTAR replay harness (reference/replay_rstar: synthetic inputs on the real global_ocean grid)

def replay_runs():
    """[(experiment, input_dir, rundir)] from reference/replay_rstar/CURRENT (relative to $MJX_REFERENCE)."""
    from pathlib import Path
    from mitjax import paths
    repo = Path(__file__).resolve().parents[2]
    out = []
    for ln in (repo / "reference" / "replay_rstar" / "CURRENT").read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            exp, inp, rel = ln.split()
            out.append((exp, inp, Path(paths.REFERENCE) / rel))
    return out


def _replay_io():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "reference" / "replay_rstar" / "replay_io.py"
    spec = importlib.util.spec_from_file_location("_rstar_replay_io", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def replay_data(exp, rundir):
    """(inputs, outputs, grid records, STDERR lines) of a harness run."""
    io = _replay_io()
    sz = experiment(exp, "input").cfg.size
    size = dict(sNx=sz.sNx, sNy=sz.sNy, OLx=sz.OLx, OLy=sz.OLy, nSx=sz.nSx, nSy=sz.nSy, Nr=sz.Nr)
    stderr = (rundir / "STDERR.0000").read_text(errors="replace").splitlines()
    return io.read_inputs(rundir, size), io.read_outputs(rundir, size), io.read_grid(rundir, size), stderr


def replay_gate(exp, inp, rundir, ex=None, calc_fn=None, planted_area_weight=False):
    """The harness's call sequence on its own inputs and grid: CALC_R_STAR(etaA), UPDATE_R_STAR(.TRUE.),
    UPDATE_R_STAR(.FALSE.), CALC_R_STAR(etaB) with the simple average, RESET_NLFS_VARS; returns ({label: compare},
    our warning lines of both CALC_R_STAR calls (calc_r_star_host, numbWrite carried), the harness's STDERR lines
    written by CALC_R_STAR). planted_area_weight: call B with the area weighting (planted error)."""
    from mitjax.model.src.calc_r_star import calc_r_star_host
    e = experiment(exp, inp)
    sz = e.cfg.size
    inp_, out, grd, stderr = replay_data(exp, rundir)
    par = grd["params"]
    p0 = params(exp, inp)
    assert (par[0], par[1], par[2]) == (float(p0.hFacInf), float(p0.hFacSup), float(p0.deltaTFreeSurf)), par
    assert (int(par[3]), bool(par[4])) == (p0.selectKEscheme, p0.vectorInvariantMomentum), par
    f = {}
    for n in GRID_XY:
        f[n] = _xy(grd[n][:, None], n, sz)
    for n in GRID_INT:
        f[n] = _xy(grd[n][:, None], n, sz, dtype=np.int32)
    for n in GRID_XYZ + R_XYZ + HFAC:
        f[n] = _xyz(grd[n], n, sz)
    grid = Grid(f)
    st = {n: _xy(inp_["prior"][:, None], n, sz) for n in R_STATE}
    for n in R_FAC:
        st[n] = _xy(inp_[n][:, None], n, sz)
    state = State(st)
    etaA, etaB = (_xy(out[n][:, None], n, sz) for n in ("etaA", "etaB"))
    res = {}
    state, cA = run_calc_r_star(exp, inp, etaA, grid, state, ex=ex, fn=calc_fn)
    for n in R_STATE + R_FAC:
        res[f"A_{n}"] = compare(getattr(state, n), out[f"A_{n}"][:, None])
    grid = run_update_r_star(exp, inp, True, grid, state)
    for n in HFAC + ("recip_hFacW", "recip_hFacS"):
        res[f"T_{n}"] = compare(getattr(grid, n), out[f"T_{n}"])
    grid = run_update_r_star(exp, inp, False, grid, state)
    for n in HFAC + ("recip_hFacW", "recip_hFacS"):
        res[f"F_{n}"] = compare(getattr(grid, n), out[f"F_{n}"])
    pB = p0 if planted_area_weight else p0.replace(static=dict(vectorInvariantMomentum=True, selectKEscheme=1))
    state, cB = run_calc_r_star(exp, inp, etaB, grid, state, prm=pB, ex=ex, fn=calc_fn)
    for n in R_STATE + R_FAC:
        res[f"B_{n}"] = compare(getattr(state, n), out[f"B_{n}"][:, None])
    state = run_reset_nlfs_vars(exp, inp, state)
    res["R_pStarFacK"] = compare(state.pStarFacK, out["R_pStarFacK"][:, None])
    lines, nw = calc_r_star_host(jax.tree.map(np.asarray, cA), 1, 0, cfg=e.cfg)
    linesB, nw = calc_r_star_host(jax.tree.map(np.asarray, cB), 2, nw, cfg=e.cfg)
    ref = [ln for ln in stderr if ln.startswith("WARNING: r*Fac") or ln.startswith("WARNING: max(hFacC)")]
    return res, lines + linesB, ref, (cA, cB)
