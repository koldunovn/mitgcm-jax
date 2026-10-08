"""Helpers of the GAD-B gates (test_gad_som.py, test_gad_som_quick.py): the static cfg of the SOM routines, the replay
harness data (reference/replay_gad_b, runs named by reference/replay_gad_b/CURRENT relative to $MJX_REFERENCE) as
FArrays, the jitted callers of every ported routine, and the comparison with gfortran.

The callers take plain arrays and build FArrays inside the traced program (KERNEL_GUIDE §5); float values (deltaTLev,
grid and PARAMS arrays, fields) are jit arguments, never closed over [E§5]. JAX transforms live here, not in
mitjax/pkg.
"""

import importlib.util
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from mitjax import paths
from mitjax.farray import FArray
from mitjax.pkg.generic_advdiff.gad_exch_som import gad_exch_som
from mitjax.pkg.generic_advdiff.gad_som_adv_x import gad_som_adv_x
from mitjax.pkg.generic_advdiff.gad_som_adv_y import gad_som_adv_y
from mitjax.pkg.generic_advdiff.gad_som_advect import gad_som_advect
from mitjax.pkg.generic_advdiff.gad_som_exchanges import gad_som_exchanges
from mitjax.pkg.generic_advdiff.gad_som_lim_r import gad_som_lim_r

REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS = ("advect_xy", "advect_xz")


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rio = _load_by_path("_mjx_replay_gad_b_io", REPO / "reference" / "replay_gad_b" / "replay_io.py")
SOM, SM11 = rio.SOM, rio.SM11


class SomCfg(NamedTuple):
    """Static configuration the SOM routines read (hashable; never a differentiated argument)."""
    sNx: int
    sNy: int
    OLx: int
    OLy: int
    Nr: int
    GAD_ALLOW_TS_SOM_ADV: bool          # GAD_OPTIONS.h
    PTRACERS_ALLOW_DYN_STATE: bool      # PTRACERS_OPTIONS.h (as seen by GAD_OPTIONS.h)
    ALLOW_AUTODIFF: bool
    ALLOW_DIAGNOSTICS: bool
    ALLOW_OBCS: bool
    useDiagnostics: bool                # data.pkg (the harness sets it .FALSE.)
    useCubedSphereExchange: bool        # eedata / EEPARAMS.h
    rigidLid: bool                      # PARAMS.h
    nonlinFreeSurf: int
    select_rStar: int
    uniformFreeSurfLev: bool            # derived in set_parms.F:162-166
    tempSOM_Advection: bool             # derived in gad_init_fixed.F:118-123
    saltSOM_Advection: bool


CPP_NAMES = ("GAD_ALLOW_TS_SOM_ADV", "PTRACERS_ALLOW_DYN_STATE", "ALLOW_AUTODIFF", "ALLOW_DIAGNOSTICS", "ALLOW_OBCS")
RUNTIME_NAMES = ("useCubedSphereExchange", "rigidLid", "nonlinFreeSurf", "select_rStar", "uniformFreeSurfLev",
                 "tempSOM_Advection", "saltSOM_Advection")


def som_cfg(exp, size, header):
    """SomCfg of an experiment: CPP options from mitjax.config (the build's cpp, as GAD_OPTIONS.h sees them), SIZE.h
    from the replay inputs (checked against mitjax.config), the run-time switches from the harness header (the values
    the Fortran used after INITIALISE_FIXED); test_gad_som checks the namelist-derivable ones against mitjax.config.
    useDiagnostics is .FALSE. (the harness sets it; advect_xz/input has it .FALSE. too)."""
    from mitjax.config.params import load
    e = load(exp, "input")
    sz = e.cfg.size
    assert (sz.sNx, sz.sNy, sz.OLx, sz.OLy, sz.Nr) == tuple(size[k] for k in ("sNx", "sNy", "OLx", "OLy", "Nr"))
    cpp = {n: e.cfg.cpp.flag(n, "GAD_OPTIONS.h") for n in CPP_NAMES}
    rt = {n: (bool(header[n]) if n not in ("nonlinFreeSurf", "select_rStar") else int(header[n]))
          for n in RUNTIME_NAMES}
    return SomCfg(sNx=sz.sNx, sNy=sz.sNy, OLx=sz.OLx, OLy=sz.OLy, Nr=sz.Nr, useDiagnostics=False, **cpp, **rt), e


@jax.tree_util.register_pytree_node_class
class Common:
    """A Fortran common block (GRID.h, PARAMS.h): fields by their Fortran names; a pytree (names static)."""

    def __init__(self, **fields):
        self.__dict__.update(fields)

    def tree_flatten(self):
        keys = tuple(sorted(self.__dict__))
        return tuple(self.__dict__[k] for k in keys), keys

    @classmethod
    def tree_unflatten(cls, keys, children):
        return cls(**dict(zip(keys, children)))


def bounds(cfg):
    return {"i": (1 - cfg.OLx, cfg.sNx + cfg.OLx), "j": (1 - cfg.OLy, cfg.sNy + cfg.OLy), "k": (1, cfg.Nr)}


# GRID.h (and PARAMS.h) fields read by GAD_SOM_ADVECT, with their declared dimensions (model/inc/GRID.h, PARAMS.h)
GRID_DECL = {"dxG": "ij", "dyG": "ij", "rA": "ij", "recip_rA": "ij", "maskInC": "ij",
             "hFacC": "ijk", "hFacW": "ijk", "hFacS": "ijk", "recip_hFacC": "ijk", "maskC": "ijk",
             "drF": "k", "recip_drF": "k", "deepFacC": "k", "deepFac2C": "k", "recip_deepFac2C": "k",
             "deepFac2F": "K"}
PARAMS_DECL = {"rhoFacC": "k", "recip_rhoFacC": "k", "rhoFacF": "K"}           # K = (1, Nr+1)


def farray(cfg, a, name, dims):
    b = dict(bounds(cfg))
    kw = {d: b[d] for d in dims if d in "ijk"}
    if "K" in dims:
        kw["k"] = (1, cfg.Nr + 1)
    return FArray(a, name, tiled=("i" in dims), **kw)


def f3(cfg, a, name="f"):
    return farray(cfg, a, name, "ijk")


def f2(cfg, a, name="f"):
    return farray(cfg, a, name, "ij")


def make_grid(cfg, g):
    return Common(**{n: farray(cfg, jnp.asarray(g[n], jnp.float64), n, d) for n, d in GRID_DECL.items()})


def make_params(cfg, g):
    return Common(**{n: farray(cfg, jnp.asarray(g[n], jnp.float64), n, d) for n, d in PARAMS_DECL.items()})


class Replay:
    """One harness run: inputs, Fortran outputs, grid, cfg."""

    def __init__(self, exp, rundir):
        self.exp, self.rundir = exp, Path(rundir)
        self.size, self.inp, self.one_d = rio.read_inputs(self.rundir)
        self.header, self.out = rio.read_outputs(self.rundir, self.size)
        self.grid_np = rio.read_grid(self.rundir, self.size)
        self.cfg, self.experiment = som_cfg(exp, self.size, self.header)

    def grid_args(self):
        """(grid arrays, params arrays, dTlev) as plain jax arrays (jit arguments)."""
        g = {n: jnp.asarray(self.grid_np[n]) for n in GRID_DECL}
        p = {n: jnp.asarray(self.grid_np[n]) for n in PARAMS_DECL}
        return g, p, jnp.asarray(self.one_d["dTlev"])


def current_runs():
    """{experiment: Replay} for the runs named by reference/replay_gad_b/CURRENT (relative to $MJX_REFERENCE);
    FileNotFoundError (a test failure) when a run is missing."""
    runs = {}
    for line in (REPO / "reference" / "replay_gad_b" / "CURRENT").read_text().splitlines():
        if line.strip():
            exp, rel = line.split()
            runs[exp] = Replay(exp, paths.REFERENCE / rel)
    assert tuple(sorted(runs)) == EXPERIMENTS, sorted(runs)
    return runs


# ---------------------------------------------------------------------------------------------------- callers
def advect_fn(cfg, scheme, mod_advect=None):
    """f(fields, g, p, dTlev) -> (gTracer, 9 moments) of GAD_SOM_ADVECT(.FALSE., scheme, scheme, GAD_TEMPERATURE, ...)
    with fields = {tracer, uFld, vFld, wFld, gPrior, smTr0_<m>} ([tile, k, j, i])."""
    advect = gad_som_advect if mod_advect is None else mod_advect.gad_som_advect

    def f(fields, g, p, dTlev):
        grid = Common(**{n: farray(cfg, g[n], n, d) for n, d in GRID_DECL.items()})
        params = Common(**{n: farray(cfg, p[n], n, d) for n, d in PARAMS_DECL.items()})
        dT = FArray(dTlev, "deltaTLev", k=(1, cfg.Nr), tiled=False)
        smTr = tuple(f3(cfg, fields[f"smTr0_{m}"], f"smTr_{m}") for m in SOM)
        smTr, gTr = advect(False, scheme, scheme, 1, dT,
                           f3(cfg, fields["uFld"], "uFld"), f3(cfg, fields["vFld"], "vFld"),
                           f3(cfg, fields["wFld"], "wFld"), f3(cfg, fields["tracer"], "tracer"),
                           smTr, f3(cfg, fields["gPrior"], "gTracer"), 0.0, 0, cfg=cfg, grid=grid, params=params)
        return (gTr.data,) + tuple(a.data for a in smTr)
    return f


def advect_inputs(R):
    """The GAD_SOM_ADVECT inputs as the harness built them (out3 1..12) plus the tracer and the gTracer prior."""
    fields = {"tracer": R.inp["tracer"], "uFld": R.out["uFld"], "vFld": R.out["vFld"], "wFld": R.out["wFld"],
              "gPrior": np.full_like(R.inp["tracer"], -999.0)}
    fields.update({f"smTr0_{m}": R.out[f"smTr0_{m}"] for m in SOM})
    return {n: jnp.asarray(a) for n, a in fields.items()}


def advect_out_names(scheme):
    return (f"gTracer_{scheme}",) + tuple(f"smTr{scheme}_{m}" for m in SOM)


def leaf_fn(cfg, direction, limiter, mod=None):
    """f(fields, dTlev) -> the 11 updated fields + flux of GAD_SOM_ADV_X (direction 'x') or _Y ('y') for every level
    (a Python loop over k, as the harness's DO k loop), each [tile, k, j, i]."""
    fn = (gad_som_adv_x if direction == "x" else gad_som_adv_y) if mod is None else getattr(mod, f"gad_som_adv_{direction}")
    trans = "uTransX" if direction == "x" else "vTransY"

    def f(fields, maskInC, dTlev):
        per_k = []
        for k in range(1, cfg.Nr + 1):
            sm = [f2(cfg, fields[f"sm_{m}"][:, k - 1], f"sm_{m}") for m in SM11]
            out = fn(k, limiter, False, False, False, False, False, False, dTlev[k - 1],
                     f2(cfg, fields[trans][:, k - 1], trans), f2(cfg, maskInC, "maskInC"), *sm,
                     f2(cfg, fields["fluxPrior"][:, k - 1], "uT"), cfg=cfg)
            per_k.append([o.data for o in out])
        return tuple(jnp.stack([o[n] for o in per_k], axis=1) for n in range(12))
    return f


def leaf_out_names(direction, limiter):
    flux = "uT" if direction == "x" else "vT"
    return tuple(f"adv{direction}_l{limiter}_{m}" for m in SM11) + (f"adv{direction}_l{limiter}_{flux}",)


def lim_r_fn(cfg, mod=None):
    fn = gad_som_lim_r if mod is None else mod.gad_som_lim_r

    def f(fields):
        out = fn(1, *(f3(cfg, fields[f"sm_{m}"], f"sm_{m}") for m in SM11), cfg=cfg)
        return tuple(o.data for o in out)
    return f


def exch_fn(cfg, mod=None):
    """f(fields, ex) -> (som_T after GAD_SOM_EXCHANGES, som_S after, smTr after GAD_EXCH_SOM), each 9 arrays."""
    exchanges = gad_som_exchanges if mod is None else mod.gad_som_exchanges
    exch = gad_exch_som if mod is None else mod.gad_exch_som

    def f(fields, ex):
        som_T = tuple(f3(cfg, fields[f"som_T_{m}"], f"som_T_{m}") for m in SOM)
        som_S = tuple(f3(cfg, fields[f"som_S_{m}"], f"som_S_{m}") for m in SOM)
        smTr = tuple(f3(cfg, fields[f"smTr_{m}"], f"smTr_{m}") for m in SOM)
        som_T, som_S = exchanges(cfg=cfg, ex=ex, som_T=som_T, som_S=som_S)
        smTr = exch(smTr, cfg.Nr, ex=ex)
        return tuple(a.data for a in som_T + som_S + smTr)
    return f


EXCH_OUT_NAMES = (tuple(f"som_T_out_{m}" for m in SOM) + tuple(f"som_S_out_{m}" for m in SOM)
                  + tuple(f"exch_{m}" for m in SOM))


def exchanger(exp):
    from mitjax.eesupp.exch_maps import load_maps
    from mitjax.eesupp.exchange import Exchanger
    return Exchanger(load_maps(exp))


# ---------------------------------------------------------------------------------------------------- comparison
def n_differ(jax_out, fortran_out):
    """Points where the JAX output is not element-equal to the (finite) Fortran output, or is not finite. Element
    equality (numpy ==; never Python max(), which skips NaN): -0 == +0 counts as equal."""
    a, b = np.asarray(jax_out), np.asarray(fortran_out)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.all(np.isfinite(b)), "Fortran output not finite"
    return int(np.sum(~(a == b)))                    # a non-finite JAX value never equals the finite oracle


GROUPS = ("advect", "adv_x", "adv_y", "lim_r", "exch")


def run_all(R, mods=None, ex=None, only=GROUPS, schemes=(80, 81)):
    """{output name: points differing from gfortran} for every ported routine and case of one harness run (the
    routine groups in `only`). `mods`: optional {group: module} replacing a routine (negative controls)."""
    mods = mods or {}
    cfg = R.cfg
    g, p, dTlev = R.grid_args()
    res = {}
    fields = advect_inputs(R)
    for scheme in (schemes if "advect" in only else ()):
        outs = jax.jit(advect_fn(cfg, scheme, mods.get("advect")))(fields, g, p, dTlev)
        for name, o in zip(advect_out_names(scheme), outs):
            res[name] = n_differ(o, R.out[name])
    leaf_in = {n: jnp.asarray(R.inp[n]) for n in ("uTransX", "vTransY", "fluxPrior") + tuple(f"sm_{m}" for m in SM11)}
    for direction in ("x", "y"):
        if f"adv_{direction}" not in only:
            continue
        for limiter in (0, 1):
            outs = jax.jit(leaf_fn(cfg, direction, limiter, mods.get(f"adv_{direction}")))(
                leaf_in, jnp.asarray(R.grid_np["maskInC"]), dTlev)
            for name, o in zip(leaf_out_names(direction, limiter), outs):
                res[name] = n_differ(o, R.out[name])
    if "lim_r" in only:
        outs = jax.jit(lim_r_fn(cfg, mods.get("lim_r")))(leaf_in)
        for m, o in zip(SM11, outs):
            res[f"limr_{m}"] = n_differ(o, R.out[f"limr_{m}"])
    if "exch" not in only:
        return res
    ex = exchanger(R.exp) if ex is None else ex
    ex_in = {n: jnp.asarray(R.inp[n]) for n in rio.IN3 if n.startswith(("som_T_", "som_S_", "smTr_"))}
    outs = jax.jit(exch_fn(cfg, mods.get("exch")))(ex_in, ex)
    for name, o in zip(EXCH_OUT_NAMES, outs):
        res[name] = n_differ(o, R.out[name])
    return res


def clone(mod, name, **globals_):
    """A copy of module `mod` (same source) whose module globals in `globals_` are replaced, e.g. a driver that calls
    a planted leaf routine."""
    import types
    m = types.ModuleType(f"_clone_{name}")
    m.__file__ = mod.__file__
    exec(compile(Path(mod.__file__).read_text(), f"<clone {name}>", "exec"), m.__dict__)
    m.__dict__.update(globals_)
    return m


def planted(mod, old, new, name):
    """A copy of module `mod` with `old` replaced by `new` (exactly one occurrence), loaded under a new name."""
    src = Path(mod.__file__).read_text()
    if src.count(old) != 1:
        raise ValueError(f"planted error {name}: {src.count(old)} occurrences of {old!r} in {mod.__file__}")
    import types
    m = types.ModuleType(f"_planted_{name}")
    m.__file__ = mod.__file__
    exec(compile(src.replace(old, new), f"<planted {name}>", "exec"), m.__dict__)
    return m
