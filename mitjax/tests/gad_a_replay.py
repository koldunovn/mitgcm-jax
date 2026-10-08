"""Helpers of the GAD-A gates (mitjax/tests/test_gad_simple*.py): the replay harness runs named by
reference/replay_gad_a/CURRENT (relative to $MJX_REFERENCE), the static kernel `cfg` of each experiment through
mitjax/config, the GRID.h / PARAMS.h common blocks as FArray pytrees, one driver per harness output that calls the
routine for every level as the harness (and GAD_CALC_RHS / GAD_IMPLICIT_R) calls it, planted-error copies of a kernel
module, and bit-pattern comparisons. JAX transforms (jit, grad) are used here, never in mitjax/pkg.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.farray import FArray

REPO = Path(__file__).resolve().parents[2]
PKG = "mitjax.pkg.generic_advdiff"


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


replay_io = _load_by_path("_mjx_replay_gad_a_io", REPO / "reference" / "replay_gad_a" / "replay_io.py")

ROUTINES = ("gad_c2_adv_x", "gad_c2_adv_y", "gad_c2_adv_r", "gad_c4_adv_x", "gad_c4_adv_y", "gad_u3_adv_x",
            "gad_u3_adv_y", "gad_u3c4_impl_r", "gad_dst3fl_adv_x", "gad_dst3fl_adv_y", "gad_dst3_adv_x",
            "gad_fluxlimit_adv_x", "gad_fluxlimit_adv_y", "gad_fluxlimit_impl_r", "gad_diff_x", "gad_diff_y",
            "gad_diff_r")

# CPP options the GAD-A kernels test (each .F includes GAD_OPTIONS.h first, which includes PACKAGES_CONFIG.h and
# CPP_OPTIONS.h, pkg/generic_advdiff/GAD_OPTIONS.h:9-10): the macros as seen after GAD_OPTIONS.h.
CPP_NAMES = ("ALLOW_AUTODIFF", "TARGET_NEC_SX", "OLD_DST3_FORMULATION", "ALLOW_SMAG_3D_DIFFUSIVITY",
             "ISOTROPIC_COS_SCALING")
CPP_HEADER = "GAD_OPTIONS.h"


class KernelCfg(NamedTuple):
    """Static configuration of the GAD-A kernels (hashable; never a differentiated argument)."""
    sNx: int
    sNy: int
    OLx: int
    OLy: int
    Nr: int
    ALLOW_AUTODIFF: bool
    TARGET_NEC_SX: bool
    OLD_DST3_FORMULATION: bool
    ALLOW_SMAG_3D_DIFFUSIVITY: bool
    ISOTROPIC_COS_SCALING: bool


_CFG_CACHE = {}


def kernel_cfg(experiment, input_dir):
    """KernelCfg of an experiment variant from mitjax/config (SIZE.h of the build; the CPP options after
    GAD_OPTIONS.h, by the build's own preprocessor)."""
    key = (experiment, input_dir)
    if key not in _CFG_CACHE:
        from mitjax.config import params as cp
        c = cp.load(experiment, input_dir).cfg
        s = c.size
        _CFG_CACHE[key] = KernelCfg(sNx=s.sNx, sNy=s.sNy, OLx=s.OLx, OLy=s.OLy, Nr=s.Nr,
                                    **{n: c.cpp.flag(n, CPP_HEADER) for n in CPP_NAMES})
    return _CFG_CACHE[key]


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
    return {"i": (1 - cfg.OLx, cfg.sNx + cfg.OLx), "j": (1 - cfg.OLy, cfg.sNy + cfg.OLy)}


# model/inc/GRID.h and PARAMS.h declarations of the fields the kernels read: dims, and k extent (Nr or Nr+1)
GRID_DECL = {"maskC": "ijk", "maskW": "ijk", "maskS": "ijk", "recip_hFacC": "ijk",          # GRID.h:468-470, ...
             "recip_dxC": "ij", "recip_dyC": "ij", "rA": "ij", "recip_rA": "ij",           # GRID.h:445, 449, ...
             "cosFacU": "j", "cosFacV": "j",                                               # GRID.h:320-321
             "recip_deepFacC": "k", "recip_deepFac2C": "k", "recip_drF": "k",              # (Nr)
             "deepFac2F": "k+", "recip_deepFac2F": "k+", "recip_drC": "k+",                # (Nr+1)
             "rkSign": ""}                                                                 # GRID.h:333 scalar
PARAMS_DECL = {"recip_rhoFacC": "k", "rhoFacF": "k+", "recip_rhoFacF": "k+"}               # PARAMS.h:960-961


def wrap(cfg, name, a, dims):
    """A plain array as the FArray of its Fortran declaration (dims "ijk", "ij", "j", "k" = (1,Nr), "k+" =
    (1,Nr+1), "" = scalar)."""
    if dims == "":
        return a
    b = bounds(cfg)
    if dims in ("k", "k+"):
        return FArray(a, name, k=(1, cfg.Nr + (dims == "k+")), tiled=False)
    kw = {d: b[d] for d in dims if d in "ij"}
    if "k" in dims:
        kw["k"] = (1, cfg.Nr)
    return FArray(a, name, **kw)


def make_common(cfg, values, decl):
    return Common(**{n: wrap(cfg, n, values[n], d) for n, d in decl.items()})


class Replay:
    """One harness run: inputs, Fortran outputs, the grid as the routines read it, and the kernel cfg."""

    def __init__(self, experiment, input_dir, rundir):
        self.experiment, self.input_dir, self.rundir = experiment, input_dir, Path(rundir)
        self.size, self.fields, self.scal, self.one_d = replay_io.read_inputs(self.rundir)
        self.out = replay_io.read_outputs(self.rundir, self.size)
        self.grid_np = replay_io.read_grid(self.rundir, self.size)
        self.cfg = kernel_cfg(experiment, input_dir)
        top = self.rundir
        while top.name != "runs" and top != top.parent:
            top = top.parent
        self.options_dir = top.parent / "options"

    def args(self, field_names=None):
        """(fields, grid, params, scal) as jax arrays: fields {name: [tile,k,j,i]}, grid/params plain-array dicts,
        scal {deltaTloc, diffKh, deltaTarg}: everything a kernel reads, all traced (never closed over [E§5])."""
        names = replay_io.IN3 if field_names is None else field_names
        fields = {n: jnp.asarray(self.fields[n]) for n in names}
        g = self.grid_np
        grid = {n: jnp.asarray(g[n], jnp.float64) for n in GRID_DECL}
        params = {n: jnp.asarray(g[n], jnp.float64) for n in PARAMS_DECL}
        scal = {"deltaTloc": jnp.asarray(self.scal["deltaTloc"], jnp.float64),
                "diffKh": jnp.asarray(self.scal["diffKh"], jnp.float64),
                "deltaTarg": jnp.asarray(self.one_d["deltaTarg"])}
        return fields, grid, params, scal


def current_replays():
    """{experiment: Replay} of the runs named by reference/replay_gad_a/CURRENT; a missing run is an error (the
    gates fail, not skip)."""
    from mitjax import paths
    out = {}
    for line in (REPO / "reference" / "replay_gad_a" / "CURRENT").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        exp, inp, rel = line.split()
        out[exp] = Replay(exp, inp, paths.REFERENCE / rel)
    return out


def module(name):
    return importlib.import_module(f"{PKG}.{name}")


def planted(name, old, new, tag="planted"):
    """A fresh copy of kernel module `name` with `old` replaced by `new` (exactly one occurrence)."""
    mod = module(name)
    src = Path(mod.__file__).read_text()
    if src.count(old) != 1:
        raise ValueError(f"plant target occurs {src.count(old)} times in {mod.__file__}: {old!r}")
    m = type(sys)(f"{mod.__name__}_{tag}")
    m.__file__ = mod.__file__
    exec(compile(src.replace(old, new), f"<{tag}>", "exec"), m.__dict__)
    return m


# Each harness output: (routine, how the harness calls it). Per-level outputs are written with the prior fluxPrior.
# the_main_loop.F passes maskLocW = maskW(:,:,k), maskLocS = maskS(:,:,k), maskUp = maskC(k-1)*maskC(k) (k-1 -> 1
# at k = 1), the 3-D tracer for the _R routines, and recip_hFacC as recip_hFac.
PER_LEVEL = {
    "c2_x": ("gad_c2_adv_x", None), "c2_y": ("gad_c2_adv_y", None), "c2_r": ("gad_c2_adv_r", None),
    "c4_x": ("gad_c4_adv_x", None), "c4_y": ("gad_c4_adv_y", None),
    "u3_x": ("gad_u3_adv_x", None), "u3_y": ("gad_u3_adv_y", None),
    "dst3fl_x_calcCFL": ("gad_dst3fl_adv_x", True), "dst3fl_x_givenCFL": ("gad_dst3fl_adv_x", False),
    "dst3fl_y_calcCFL": ("gad_dst3fl_adv_y", True), "dst3fl_y_givenCFL": ("gad_dst3fl_adv_y", False),
    "dst3_x_calcCFL": ("gad_dst3_adv_x", True), "dst3_x_givenCFL": ("gad_dst3_adv_x", False),
    "fluxlimit_x_calcCFL": ("gad_fluxlimit_adv_x", True), "fluxlimit_x_givenCFL": ("gad_fluxlimit_adv_x", False),
    "fluxlimit_y_calcCFL": ("gad_fluxlimit_adv_y", True), "fluxlimit_y_givenCFL": ("gad_fluxlimit_adv_y", False),
    "diff_x": ("gad_diff_x", None), "diff_y": ("gad_diff_y", None), "diff_r": ("gad_diff_r", None),
}
IMPL = {"gad_u3c4_impl_r": ("u3c4_a5d", "u3c4_b5d", "u3c4_c5d", "u3c4_d5d", "u3c4_e5d"),
        "gad_fluxlimit_impl_r": ("fluxlimit_a3d", "fluxlimit_b3d", "fluxlimit_c3d")}
IMPL_PRIORS = {"gad_u3c4_impl_r": ("aPrior", "bPrior", "cPrior", "dPrior", "ePrior"),
               "gad_fluxlimit_impl_r": ("aPrior", "bPrior", "cPrior")}
# the cases: per-level outputs one by one, the implicit routines with all their outputs
CASES = tuple(PER_LEVEL) + tuple(IMPL)


def case_routine(case):
    return PER_LEVEL[case][0] if case in PER_LEVEL else case


def case_outputs(case):
    return (case,) if case in PER_LEVEL else IMPL[case]


def case_fields(case):
    """The input fields a case reads."""
    r = case_routine(case)
    if r.startswith("gad_diff_"):
        return {"gad_diff_x": ("xA", "tracer", "fluxPrior"), "gad_diff_y": ("yA", "tracer", "fluxPrior"),
                "gad_diff_r": ("KappaR", "tracer", "fluxPrior")}[r]
    if case in IMPL:
        return ("rTrans", "tracer") + IMPL_PRIORS[case] if case == "gad_fluxlimit_impl_r" else \
            ("rTrans",) + IMPL_PRIORS[case]
    xy = "u" if r.endswith("_x") else "v"
    if r.endswith("_r"):
        return ("rTrans", "tracer", "fluxPrior")
    calc = PER_LEVEL[case][1]
    vel = () if calc is None else ((f"{xy}Vel",) if calc else (f"{xy}CFL",))
    return (f"{xy}Trans",) + vel + ("tracer", "fluxPrior")


def level_call(case, k, cfg, fields, grid, params, scal, mod=None):
    """The routine of `case` at level k as the harness calls it; fields {name: [tile,k,j,i]} plain arrays, grid and
    params Commons of FArrays; returns the output FArray(s)."""
    r = case_routine(case)
    m = mod or module(r)
    fn = getattr(m, r)
    b = bounds(cfg)

    def L(name):
        return FArray(fields[name][:, k - 1], name, i=b["i"], j=b["j"])

    def L3(name):
        return FArray(fields[name], name, i=b["i"], j=b["j"], k=(1, cfg.Nr))

    def mask2(name):
        return FArray(grid.__dict__[name].data[:, k - 1], name, i=b["i"], j=b["j"])

    if case in IMPL:
        recip_hFac = grid.recip_hFacC
        deltaTarg = FArray(scal["deltaTarg"], "deltaTarg", k=(1, cfg.Nr), tiled=False)
        prior = [L3(n) for n in IMPL_PRIORS[case]]
        if r == "gad_u3c4_impl_r":
            from mitjax.pkg.generic_advdiff.gad_h import ENUM_UPWIND_3RD
            return fn(k, 1, cfg.sNx, 1, cfg.sNy, ENUM_UPWIND_3RD, deltaTarg, L("rTrans"), recip_hFac,
                      *prior, cfg=cfg, grid=grid, params=params)
        return fn(k, 1, cfg.sNx, 1, cfg.sNy, deltaTarg, L("rTrans"), recip_hFac, L3("tracer"), *prior,
                  cfg=cfg, grid=grid, params=params)
    prior = L("fluxPrior")
    calc = PER_LEVEL[case][1]
    if r == "gad_c2_adv_x":
        return fn(k, L("uTrans"), L("tracer"), prior, cfg=cfg)
    if r == "gad_c2_adv_y":
        return fn(k, L("vTrans"), L("tracer"), prior, cfg=cfg)
    if r == "gad_c2_adv_r":
        return fn(k, L("rTrans"), L3("tracer"), prior, cfg=cfg, grid=grid)
    if r == "gad_c4_adv_x":
        return fn(k, L("uTrans"), mask2("maskW"), L("tracer"), prior, cfg=cfg, grid=grid)
    if r == "gad_c4_adv_y":
        return fn(k, L("vTrans"), mask2("maskS"), L("tracer"), prior, cfg=cfg, grid=grid)
    if r == "gad_u3_adv_x":
        return fn(k, L("uTrans"), mask2("maskW"), L("tracer"), prior, cfg=cfg)
    if r == "gad_u3_adv_y":
        return fn(k, L("vTrans"), mask2("maskS"), L("tracer"), prior, cfg=cfg)
    if r in ("gad_dst3fl_adv_x", "gad_dst3_adv_x", "gad_fluxlimit_adv_x"):
        return fn(k, calc, scal["deltaTloc"], L("uTrans"), L("uVel" if calc else "uCFL"), mask2("maskW"),
                  L("tracer"), prior, cfg=cfg, grid=grid)
    if r in ("gad_dst3fl_adv_y", "gad_fluxlimit_adv_y"):
        return fn(k, calc, scal["deltaTloc"], L("vTrans"), L("vVel" if calc else "vCFL"), mask2("maskS"),
                  L("tracer"), prior, cfg=cfg, grid=grid)
    if r == "gad_diff_x":
        return fn(k, L("xA"), scal["diffKh"], L("tracer"), prior, cfg=cfg, grid=grid)
    if r == "gad_diff_y":
        return fn(k, L("yA"), scal["diffKh"], L("tracer"), prior, cfg=cfg, grid=grid)
    if r == "gad_diff_r":
        km1 = max(1, k - 1)
        mc = grid.maskC.data
        maskUp = FArray(mc[:, km1 - 1] * mc[:, k - 1], "maskUp", i=b["i"], j=b["j"])
        return fn(k, maskUp, L("KappaR"), L3("tracer"), prior, cfg=cfg, grid=grid, params=params)
    raise KeyError(case)


def case_fn(case, cfg, mod=None, levels=None):
    """f(fields, grid, params, scal) -> tuple of outputs [tile, k, j, i] over `levels` (default 1..Nr): per-level
    cases stack the outputs of each level; the implicit routines accumulate over k = 1..Nr into their matrices, as
    GAD_IMPLICIT_R's k loop (here including k = 1, where the routine does nothing)."""
    ks = tuple(range(1, cfg.Nr + 1)) if levels is None else tuple(levels)

    def f(fields, grid, params, scal):
        g = make_common(cfg, grid, GRID_DECL)
        p = make_common(cfg, params, PARAMS_DECL)
        if case in IMPL:
            if levels is not None:
                raise ValueError("the implicit routines run over all levels")
            b = bounds(cfg)
            mats = [FArray(fields[n], n, i=b["i"], j=b["j"], k=(1, cfg.Nr)) for n in IMPL_PRIORS[case]]
            for k in ks:
                fk = dict(fields)
                for n, mat in zip(IMPL_PRIORS[case], mats):
                    fk[n] = mat.data
                mats = level_call(case, k, cfg, fk, g, p, scal, mod)
            return tuple(m.data for m in mats)
        outs = [level_call(case, k, cfg, fields, g, p, scal, mod).data for k in ks]
        return (jnp.stack(outs, axis=1),)
    return f


def run_case(R, case, mod=None, levels=None):
    fields, grid, params, scal = R.args(case_fields(case))
    f = jax.jit(case_fn(case, R.cfg, mod, levels))
    return [np.asarray(o) for o in f(fields, grid, params, scal)]


def bit_diff(a, b):
    """Number of elements whose float64 bit patterns differ (a NaN never equals anything here)."""
    a, b = np.ascontiguousarray(a, np.float64), np.ascontiguousarray(b, np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    return int(np.count_nonzero(a.view(np.int64) != b.view(np.int64)))


def diffs_vs_fortran(R, case, mod=None, levels=None):
    """{output name: points whose bit pattern differs from gfortran's}, all tiles, levels, halos."""
    outs = run_case(R, case, mod, levels)
    ks = slice(None) if levels is None else [k - 1 for k in levels]
    return {n: bit_diff(o, R.out[n][:, ks]) for n, o in zip(case_outputs(case), outs)}
