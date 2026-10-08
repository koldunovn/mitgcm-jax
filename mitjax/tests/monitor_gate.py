"""The %MON gate of pkg/monitor (plan Task 11, lane MON): our MONITOR on the oracle's own state, compared with the
oracle's STDOUT character for character. Helper module of test_monitor.py (not a test file).

For each M1 variant (reference/reference_runs.py VARIANTS) the registered dumps-on run (`jdon`, docs/REFERENCE_RUNS.md)
gives the state and the STDOUT. Which dump stage holds the state the MONITOR call sees (reference/jaxdump/SUBSTEPS.md;
records carry the step's START iteration):

* block of iteration nIter0 (MONITOR(startTime, nIter0), initialise_varia.F:383): S00_begin of nIter0 (state 'd', 't',
  forcing 'f', hFac 'r'), G00_geometry of nIter0 (grid 'G', 'V', r* 'R'). Nothing between initialise_varia's MONITOR
  and the start of forward_step changes these fields (forward_step.F:427-435 resets only AD bookkeeping), and
  mon_trAdvCFL is MON_INIT's zero (mon_init.F:46-48).
* block of iteration n+1 (MONITOR at forward_step.F:1154 of the step starting at n): S17_monitor of n ('d', 't': the
  state at the call); forcing 'f', hFac 'r': the last stage of step n before S17 that dumps them (S04_oceanic_phys
  for 'f'; S12_calc_rstar under r*, after UPDATE_R_STAR(.TRUE.) at forward_step.F:839 and CALC_R_STAR, which does not
  change hFac, else S00_begin for 'r'); rStarDhCDt (r* only, set by CALC_R_STAR at forward_step.F:949 and dumped only
  in G00): G00_geometry of n+1, the next step's start, which no routine between the MONITOR call and that dump
  writes (RESET_NLFS_VARS / UPDATE_R_STAR(.FALSE.) run after it, forward_step.F:451-475); in the jaxdump3 runs
  (registry kind `jdon3`) rStarDhCDt is also dumped at S12_calc_rstar of n, right after CALC_R_STAR, and used there
  (lane A: equal to G00 of n+1 bit for bit). mon_trAdvCFL of the block
  comes from THERMODYNAMICS of step n (thermodynamics.F:263-283, 387-390), recomputed by mon_calc_advcfl.py from the
  flow it used: T01_residual_flow (uFld, vFld, wFld after GMREDI_RESIDUAL_FLOW) where dumped, else the 'd' fields
  at S00_begin (staggerTimeStep=.FALSE.: THERMODYNAMICS runs before DYNAMICS, forward_step.F:733), with the hFac of
  S01_update_rstar_F (r*) or S00_begin.
* later MONITOR steps: the `jdon3` runs (JAXDUMP_STEPS with step 9 or 15) give block 10 of advect_xz (input, nlfs,
  pqm) and advect_xy/input.ab3_c4 and block 16 of advect_xy/input (monitorFreq = 10 and 16 steps), and block 36003
  of global_ocean (S12 of 36002); `gated_blocks` lists every block per run.
* the loose grid-statistics blocks (ini_grid.F:209-226, ini_cori.F:207-210): G00_geometry of nIter0 (the grid is
  fixed after INI_GRID / INI_CORI).

Fields the dumps do not hold, computed literally: kSurfC (ini_masks_etc.F:173, 186-190, from h0FacC under r*, else
hFacC), deepFacC/deepFac2C/deepFac2F/recip_deepFac2C (set_grid_factors.F:49-58, deepAtmosphere=.FALSE., checked),
recip_rhoFacC (set_ref_state.F:74-75, rhoFacC = 1 checked), Bo_surf without r* (ini_linear_phisurf.F:84, gBaro with
usingZCoords, checked), myTime (forward_step.F:808: startTime + deltaTClock*iLoop). Run-time parameters come from the
oracle's own parameter printout (config_summary), SIZE and CPP options from mitjax/config (the build's own
preprocessor), MNC switches from data.pkg / data.mnc (defaults: packages_boot.F:161, mnc_readparms.F:101,
eeset_parms.F:106).
"""

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from mitjax.farray import FArray
from mitjax.eesupp.print import MessageUnits
from mitjax.io.dump import DumpSet
from mitjax.io.stdout import fortran_number, monitor_blocks, parameter_dict, parameters, read_stdout
from mitjax.pkg.monitor.mon_calc_advcfl import mon_calc_advcfl_glob, mon_calc_advcfl_tile
from mitjax.pkg.monitor.mon_init import mon_init
from mitjax.pkg.monitor.mon_printstats_rs import mon_printstats_rs
from mitjax.pkg.monitor.mon_set_pref import mon_set_pref
from mitjax.pkg.monitor.monitor import level_view, monitor
from mitjax.pkg.monitor.monitor_h import MonitorCommon, mon_string_none

REPO = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rr = _load("_mjx_reference_runs_mon", REPO / "reference" / "reference_runs.py")
VARIANTS = [(e, i) for e, i, _ in rr.VARIANTS]
BEGIN = "// Begin MONITOR dynamic field statistics"


# ---------------------------------------------------------------------------------------------------------------
# oracle side

class Oracle:
    """One variant's dumps-on run (registry kind `jdon`, or `jdon3` for the later MONITOR steps): raw STDOUT records,
    parameter printout, dump set."""

    def __init__(self, exp, inp, kind="jdon"):
        self.exp, self.inp, self.kind = exp, inp, kind
        self.run = rr.find_run(exp, inp, kind)
        top = rr.run_top(self.run)
        self.stdout_path = top / "rundir" / "output.txt"
        self.raw = self.stdout_path.read_bytes().decode("latin-1").split("\n")
        self.lines = read_stdout(self.stdout_path)
        self.prm = parameter_dict(parameters(self.lines))
        self.ds = DumpSet(top / "dumps")
        self.its = self.ds.iterations()
        self.blocks = monitor_blocks(self.lines)

    # -- parameter printout
    def p(self, name):
        """A scalar parameter: the first value line (selector printouts add explanation lines after it)."""
        return _value(self.prm[name].values()[0])

    def plist(self, name):
        """A 1-D parameter: every value (`n @ v` repeats expanded)."""
        return [_value(v) for v in self.prm[name].values()]

    def raw_lines(self, first, last):
        return self.raw[first - 1:last]

    def block_lines(self, tsnumber):
        """Raw records of the MONITOR block of iteration tsnumber: the banner line before Begin through the banner
        after End (monitor.F:78-86, 183-191)."""
        for b in self.blocks:
            if b.kind == "MONITOR dynamic field statistics" and int(b.text["time_tsnumber"]) == tsnumber:
                return self.raw_lines(b.first - 1, b.last + 1)
        raise KeyError(f"{self.exp}/{self.inp}: no MONITOR block of iteration {tsnumber}")

    def loose_blocks(self):
        return [self.raw_lines(b.first, b.last) for b in self.blocks if b.kind is None]


def _value(t):
    t = t.strip()
    if t in ("T", "F"):
        return t == "T"
    return fortran_number(t.split()[0])


@lru_cache(maxsize=None)
def oracle(exp, inp, kind="jdon"):
    return Oracle(exp, inp, kind)


def has_run(exp, inp, kind):
    return any((r["exp"], r["input"], r["kind"]) == (exp, inp, kind) for r in rr.RUNS)


@lru_cache(maxsize=None)
def experiment(exp, inp):
    from mitjax.config import params as P
    return P.load(exp, inp)


# ---------------------------------------------------------------------------------------------------------------
# inputs of our MONITOR

def make_cfg(o):
    """Static configuration of MONITOR (monitor.py docstring) for oracle o."""
    ex = experiment(o.exp, o.inp)
    c, sz = ex.cfg, ex.cfg.size
    st = {k: v for k, v in c.static}

    def static(fname, group, key, default):
        return st.get((fname, group, key.lower()), default)

    cpp = c.cpp
    flag = lambda name, header=None: (cpp.flag(name, header) if header else bool(getattr(cpp, name))) \
        if name in cpp.known else False
    use_mnc = c.use_flag("useMNC")
    cfg = SimpleNamespace(
        sNx=sz.sNx, sNy=sz.sNy, OLx=sz.OLx, OLy=sz.OLy, nSx=sz.nSx, nSy=sz.nSy, Nr=sz.Nr,
        monitorSelect=o.p("monitorSelect"), monitor_stdio=o.p("monitor_stdio"), usingPCoords=o.p("usingPCoords"),
        usingSphericalPolarGrid=o.p("usingSphericalPolarGrid"), fluidIsAir=o.p("fluidIsAir"),
        fluidIsWater=o.p("fluidIsWater"), useCoriolis=o.p("useCoriolis"), selectCoriMap=o.p("selectCoriMap"),
        nonHydrostatic=o.p("nonHydrostatic"),
        select_rStar=o.p("select_rStar") if "select_rStar" in o.prm else 0,
        useCubedSphereExchange=static("eedata", "eeparms", "useCubedSphereExchange", False),  # eeset_parms.F:106
        useMNC=use_mnc,                                                                       # packages_boot.F:161
        monitor_mnc=static("data.mnc", "mnc_01", "monitor_mnc", True),                       # mnc_readparms.F:101
        useAIM=c.use_flag("useAIM") if "useAIM" in dict(c.use) else False,                   # packages_boot.F:154
        ALLOW_MNC=flag("ALLOW_MNC"), ALLOW_NONHYDROSTATIC=flag("ALLOW_NONHYDROSTATIC"), ALLOW_AIM=flag("ALLOW_AIM"),
        NONLIN_FRSURF=flag("NONLIN_FRSURF"),
        MONITOR_TEST_HFACZ=flag("MONITOR_TEST_HFACZ", "MONITOR_OPTIONS.h"),
        staggerTimeStep=o.p("staggerTimeStep"),
    )
    return cfg


def make_params(o, cfg):
    dt = o.plist("dTtracerLev")
    if len(dt) != cfg.Nr:
        raise ValueError(f"dTtracerLev printout has {len(dt)} values, Nr = {cfg.Nr}")
    return SimpleNamespace(monitorFreq=o.p("monitorFreq"), deltaTClock=o.p("deltaTClock"),
                           deltaTMom=o.p("deltaTMom"), dTtracerLev=[float(x) for x in dt],
                           rUnit2mass=o.p("rUnit2mass"), startTime=o.p("startTime"), nIter0=o.p("nIter0"),
                           gBaro=o.p("gBaro"))


def _f2(cfg, a, name):
    return FArray(jnp.asarray(a[:, 0]), name, i=(1 - cfg.OLx, cfg.sNx + cfg.OLx), j=(1 - cfg.OLy, cfg.sNy + cfg.OLy))


def _f3(cfg, a, name):
    return FArray(jnp.asarray(a), name, i=(1 - cfg.OLx, cfg.sNx + cfg.OLx), j=(1 - cfg.OLy, cfg.sNy + cfg.OLy),
                  k=(1, a.shape[1]))


def _vec(o, it, name):
    recs = o.ds.tiles(it, "G00_geometry", name)
    r = recs[min(recs)]
    if r.kind != "V":
        raise ValueError(f"{name} is not a kind-V record")
    plane = r.data.reshape(r.nz, -1)
    if not np.all(plane == plane[:, :1]):
        raise ValueError(f"{name}: kind-V record not constant over its plane")
    return plane[:, 0].copy()


# fields declared with an Nr dimension (GRID.h, DYNVARS.h, SURFACE.h, thermodynamics.F uFld/vFld/wFld)
THREE_D = {"uVel", "vVel", "wVel", "theta", "salt", "hFacC", "hFacW", "hFacS", "recip_hFacC", "maskC", "h0FacC",
           "uFld", "vFld", "wFld"}


def _fld(o, it, stage, name, cfg):
    a = o.ds.field(it, stage, name)
    if name in THREE_D:
        if a.shape[1] != cfg.Nr:
            raise ValueError(f"{stage}/{name}: {a.shape[1]} levels, Nr = {cfg.Nr}")
        return _f3(cfg, a, name)
    if a.shape[1] != 1:
        raise ValueError(f"{stage}/{name}: 2-D field with {a.shape[1]} levels")
    return _f2(cfg, a, name)


def _last_stage_with(o, it, name, before="S17_monitor"):
    st = o.ds.stages(it)
    stop = st.index(before)
    found = [s for s in st[:stop] if (it, s, name) in o.ds.index]
    if not found:
        raise KeyError(f"{name} is dumped by no stage of iteration {it} before {before}")
    return found[-1]


GRID_2D = ("rA", "rAw", "rAs", "rAz", "recip_rA", "recip_rAz", "dxC", "dyC", "dxG", "dyG", "dxF", "dyF", "dxV",
           "dyU", "recip_dxC", "recip_dyC", "maskInC", "maskInW", "maskInS", "fCori", "fCoriG", "fCoriCos", "xC",
           "yC", "xG", "yG", "angleCosC", "angleSinC")
GRID_3D = ("maskC",)
GRID_V = ("drF", "drC", "recip_drF", "recip_drC", "rhoFacC", "rhoFacF")
HFAC = ("hFacC", "hFacW", "hFacS", "recip_hFacC")


def make_grid(o, cfg, params, it0, hfac_stage, hfac_it):
    """GRID.h / SURFACE.h inputs of MONITOR: G00 of it0 plus the hFac fields of (hfac_it, hfac_stage)."""
    g = SimpleNamespace()
    for n in GRID_2D:
        setattr(g, n, _fld(o, it0, "G00_geometry", n, cfg))
    for n in GRID_3D:
        setattr(g, n, _fld(o, it0, "G00_geometry", n, cfg))
    for n in GRID_V:
        v = _vec(o, it0, n)
        setattr(g, n, FArray(jnp.asarray(v), n, k=(1, v.size), tiled=False))
    for n in HFAC:
        setattr(g, n, _fld(o, hfac_it, hfac_stage, n, cfg))
    Nr = cfg.Nr
    if o.p("deepAtmosphere"):
        raise NotImplementedError("deepAtmosphere: deep-model grid factors (set_grid_factors.F:59-91)")
    one_c = jnp.ones(Nr)                                    # set_grid_factors.F:48-53 (1. _d 0)
    one_f = jnp.ones(Nr + 1)                                # set_grid_factors.F:54-59
    g.deepFacC = FArray(one_c, "deepFacC", k=(1, Nr), tiled=False)
    g.deepFac2C = FArray(one_c, "deepFac2C", k=(1, Nr), tiled=False)
    g.recip_deepFac2C = FArray(one_c, "recip_deepFac2C", k=(1, Nr), tiled=False)
    g.deepFac2F = FArray(one_f, "deepFac2F", k=(1, Nr + 1), tiled=False)
    if not np.all(np.asarray(g.rhoFacC.data) == 1.0):
        raise NotImplementedError("rhoFacC != 1: recip_rhoFacC of the anelastic set-up (set_ref_state.F:365-)")
    g.recip_rhoFacC = FArray(one_c, "recip_rhoFacC", k=(1, Nr), tiled=False)   # set_ref_state.F:74-75
    rstar = (it0, "G00_geometry", "h0FacC") in o.ds.index
    if rstar:
        g.h0FacC = _fld(o, it0, "G00_geometry", "h0FacC", cfg)
        g.Bo_surf = _fld(o, it0, "G00_geometry", "Bo_surf", cfg)
        hk = np.asarray(g.h0FacC.data)
    else:
        if not o.p("usingZCoords"):
            raise NotImplementedError("Bo_surf outside z coordinates (ini_linear_phisurf.F:89-)")
        g.Bo_surf = FArray(jnp.full(g.rA.data.shape, params.gBaro), "Bo_surf",      # ini_linear_phisurf.F:84
                           i=(1 - cfg.OLx, cfg.sNx + cfg.OLx), j=(1 - cfg.OLy, cfg.sNy + cfg.OLy))
        hk = np.asarray(o.ds.field(it0, "S00_begin", "hFacC"))
    ks = np.full(hk.shape[:1] + hk.shape[2:], Nr + 1, dtype=np.int64)              # ini_masks_etc.F:173
    for k in range(Nr, 0, -1):                                                     # ini_masks_etc.F:186-190
        ks = np.where(hk[:, k - 1] != 0.0, k, ks)
    g.kSurfC = FArray(jnp.asarray(ks), "kSurfC", i=(1 - cfg.OLx, cfg.sNx + cfg.OLx),
                      j=(1 - cfg.OLy, cfg.sNy + cfg.OLy))
    return g


STATE_D = ("uVel", "vVel", "wVel", "etaN")
STATE_T = ("theta", "salt")
STATE_F = ("Qnet", "Qsw", "EmPmR", "fu", "fv", "phi0surf")


def make_state(o, cfg, it, stage, f_stage, rstar=None):
    """rstar: (iteration, stage) of the rStarDhCDt record the MONITOR call sees, or None."""
    s = SimpleNamespace()
    for n in STATE_D + STATE_T:
        setattr(s, n, _fld(o, it, stage, n, cfg))
    for n in STATE_F:
        setattr(s, n, _fld(o, it, f_stage, n, cfg))
    if rstar is not None:
        s.rStarDhCDt = _fld(o, rstar[0], rstar[1], "rStarDhCDt", cfg)
    return s


def rstar_source(o, it):
    """Where rStarDhCDt as the MONITOR call of the step starting at `it` sees it: S12_calc_rstar of `it` (CALC_R_STAR,
    forward_step.F:949, dumped by the jaxdump3 builds) or else G00_geometry of it+1 (equal bit for bit in lane A's
    dumps); None when neither is dumped."""
    if (it, "S12_calc_rstar", "rStarDhCDt") in o.ds.index:
        return (it, "S12_calc_rstar")
    if (it + 1, "G00_geometry", "rStarDhCDt") in o.ds.index:
        return (it + 1, "G00_geometry")
    return None


def tr_adv_cfl(o, cfg, params, grid, it, mon):
    """mon_trAdvCFL as THERMODYNAMICS of the step starting at `it` leaves it (see the module docstring)."""
    if cfg.staggerTimeStep:
        raise NotImplementedError("trAdv_CFL of a staggered run (THERMODYNAMICS after the update) is not gated")
    if (it, "T01_residual_flow", "uFld") in o.ds.index:
        flow = [_fld(o, it, "T01_residual_flow", n, cfg) for n in ("uFld", "vFld", "wFld")]
    else:
        flow = [_fld(o, it, "S00_begin", n, cfg) for n in ("uVel", "vVel", "wVel")]
    hst = "S01_update_rstar_F" if (it, "S01_update_rstar_F", "hFacC") in o.ds.index else "S00_begin"
    gt = SimpleNamespace(**vars(grid))
    for n in HFAC:
        setattr(gt, n, _fld(o, it, hst, n, cfg))
    maxCFL = mon_calc_advcfl_tile(cfg.Nr, *flow, params.dTtracerLev, None, it, cfg=cfg, grid=gt)
    mon_calc_advcfl_glob(maxCFL, it, mon=mon)


# ---------------------------------------------------------------------------------------------------------------
# our MONITOR

def ours_block(o, tsnumber, *, cfg=None, params=None, mutate=None):
    """Records our MONITOR writes for the block of iteration tsnumber (nIter0: the initial block)."""
    cfg = cfg or make_cfg(o)
    params = params or make_params(o, cfg)
    mon = MonitorCommon()
    mon_init(cfg=cfg, params=params, mon=mon)
    it0 = o.its[0]
    if tsnumber == it0:
        grid = make_grid(o, cfg, params, it0, "S00_begin", it0)
        rst = (it0, "G00_geometry") if (it0, "G00_geometry", "rStarDhCDt") in o.ds.index else None
        state = make_state(o, cfg, it0, "S00_begin", "S00_begin", rst)
        myTime = params.startTime                                         # initialise_varia.F:383
    else:
        it = tsnumber - 1
        hst = _last_stage_with(o, it, "hFacC")
        grid = make_grid(o, cfg, params, it0, hst, it)
        rst = None
        if needs_rstar(o):
            rst = rstar_source(o, it)
            if rst is None:
                raise KeyError(f"rStarDhCDt at the MONITOR call of iteration {tsnumber} is not dumped")
        state = make_state(o, cfg, it, "S17_monitor", _last_stage_with(o, it, "Qnet"), rst)
        if cfg.monitorSelect >= 2:
            tr_adv_cfl(o, cfg, params, grid, it, mon)
        myTime = params.startTime + params.deltaTClock*float(tsnumber - params.nIter0)   # forward_step.F:808
    if mutate is not None:
        mutate(cfg=cfg, params=params, grid=grid, state=state, mon=mon)
    monitor(myTime, tsnumber, cfg=cfg, params=params, grid=grid, state=state, mon=mon)
    return mon.io.records(mon.mon_ioUnit)


def ours_grid_blocks(o, *, cfg=None):
    """The two loose grid-statistics blocks (INI_GRID, ini_grid.F:184-231; INI_CORI, ini_cori.F:186-215)."""
    cfg = cfg or make_cfg(o)
    params = make_params(o, cfg)
    mon = MonitorCommon()
    mon_init(cfg=cfg, params=params, mon=mon)        # INI_MODEL_IO (initialise_fixed.F:146) before INI_GRID (:156)
    g = make_grid(o, cfg, params, o.its[0], "S00_begin", o.its[0])
    out = []
    for names in (("xC", "XC"), ("xG", "XG"), ("dxC", "DXC"), ("dxF", "DXF"), ("dxG", "DXG"), ("dxV", "DXV"),
                  ("yC", "YC"), ("yG", "YG"), ("dyC", "DYC"), ("dyF", "DYF"), ("dyG", "DYG"), ("dyU", "DYU"),
                  ("rA", "RA"), ("rAw", "RAW"), ("rAs", "RAS"), ("rAz", "RAZ"), ("angleCosC", "AngleCS"),
                  ("angleSinC", "AngleSN")), (("fCori", "fCori"), ("fCoriG", "fCoriG"), ("fCoriCos", "fCoriCos")):
        mon.io = MessageUnits()
        mon.mon_write_stdout = bool(cfg.monitor_stdio)                  # ini_grid.F:189-193, ini_cori.F:190-194
        mon.mon_write_mnc = False
        if names[0][0] == "fCori":
            mon_set_pref(mon_string_none, mon=mon)                      # ini_cori.F:207
        for fld, label in names:
            mon_printstats_rs(1, level_view(getattr(g, fld)), label, cfg=cfg, mon=mon)
        mon.mon_write_stdout = False                                    # ini_grid.F:228-231
        out.append(mon.io.records(mon.mon_ioUnit))
    return out


def compare(ours, theirs):
    """[(n, ours, theirs)] of the records that differ, character for character (and the count difference)."""
    diffs = [(n, a, b) for n, (a, b) in enumerate(zip(ours, theirs)) if a != b]
    if len(ours) != len(theirs):
        diffs.append((min(len(ours), len(theirs)), f"<{len(ours)} records>", f"<{len(theirs)} records>"))
    return diffs


def needs_rstar(o):
    """MONITOR reads rStarDhCDt only in MON_SURFCOR's r* branch: monitorSelect >= 2 (monitor.F:172) and, under
    NONLIN_FRSURF, select_rStar /= 0 (mon_surfcor.F:138-140). advect_xz's build compiles NONLIN_FRSURF for every input
    (G00 holds group R), but only input.nlfs sets select_rStar, and its monitorSelect is 1."""
    sel = o.p("select_rStar") if "select_rStar" in o.prm else 0
    return o.p("monitorSelect") >= 2 and sel != 0 and (o.its[0], "G00_geometry", "h0FacC") in o.ds.index


def gated_iterations(o):
    """The blocks this gate can compare in run o: nIter0, and n+1 for every dumped n whose block was printed and whose
    inputs are dumped (r*: rStarDhCDt at S12_calc_rstar of n or G00 of n+1, `rstar_source`)."""
    printed = {int(b.text["time_tsnumber"]) for b in o.blocks if b.kind == "MONITOR dynamic field statistics"}
    out = [o.its[0]] if o.its[0] in printed else []
    for it in o.its:
        n = it + 1
        if n not in printed or (it, "S17_monitor", "uVel") not in o.ds.index:
            continue
        if needs_rstar(o) and rstar_source(o, it) is None:
            continue
        out.append(n)
    return out


def gated_blocks(exp, inp):
    """[(kind, tsnumber)]: every block of the `jdon` run, then the blocks only the `jdon3` run (later MONITOR steps,
    rStarDhCDt at S12) adds: advect_xz (input, nlfs, pqm) and advect_xy/input.ab3_c4 block 10, advect_xy/input block
    16, global_ocean block 36003 (reference/reference_runs.py)."""
    out = [("jdon", t) for t in gated_iterations(oracle(exp, inp, "jdon"))]
    if has_run(exp, inp, "jdon3"):
        seen = {t for _, t in out}
        out += [("jdon3", t) for t in gated_iterations(oracle(exp, inp, "jdon3")) if t not in seen]
    return out
