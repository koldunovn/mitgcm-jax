"""Helpers of the GAD-C gates (mitjax/tests/test_gad_pqm.py): load the gfortran replay run of
reference/replay_gad_c (named by reference/replay_gad_c/CURRENT, relative to $MJX_REFERENCE), call the PPM/PQM
drivers of mitjax/pkg/generic_advdiff as the harness calls them, plant errors for negative controls.
JAX transforms (jit, grad) live here and in the tests, never in mitjax/pkg.
"""

import contextlib
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from mitjax import paths
from mitjax.farray import FArray

REPO = Path(__file__).resolve().parents[2]
PKG = "mitjax.pkg.generic_advdiff"


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


replay_io = _load_by_path("_mjx_replay_gad_c_io", REPO / "reference" / "replay_gad_c" / "replay_io.py")
DRIVERS = {name: getattr(importlib.import_module(f"{PKG}.{name}"), name)
           for name in ("gad_ppm_adv_x", "gad_ppm_adv_y", "gad_ppm_adv_r",
                        "gad_pqm_adv_x", "gad_pqm_adv_y", "gad_pqm_adv_r")}
# the schemes advect_xz runs (input: tempAdvScheme=42; input.pqm: 51, 52; calc_CFL = .TRUE. at every call site,
# gad_advection.F:435, 441, 656, 662)
USED = tuple(c for c in replay_io.CASES if c[1] in (42, 51, 52) and c[2] in (True, None))


class Replay:
    pass


def current_replay():
    """The replay run named by reference/replay_gad_c/CURRENT; FileNotFoundError (a test failure) when missing."""
    cur = REPO / "reference" / "replay_gad_c" / "CURRENT"
    rundir = paths.REFERENCE / cur.read_text().strip()
    if not (rundir / "replay_out.bin").is_file():
        raise FileNotFoundError(f"no replay output in {rundir}")
    R = Replay()
    R.rundir = rundir
    R.size, R.fields, R.g2, R.g1, R.scal = replay_io.read_inputs(rundir)
    R.out = replay_io.read_outputs(rundir, R.size)
    R.grid = replay_io.read_grid(rundir, R.size)
    s = R.size
    R.cfg = types.SimpleNamespace(sNx=s["sNx"], sNy=s["sNy"], OLx=s["OLx"], OLy=s["OLy"], Nr=s["Nr"])
    return R


def bounds(cfg):
    return dict(i=(1 - cfg.OLx, cfg.sNx + cfg.OLx), j=(1 - cfg.OLy, cfg.sNy + cfg.OLy))


def grid_farrays(cfg, g):
    """GRID.h fields as the routines read them, from plain arrays (built inside the traced program)."""
    b, Nr = bounds(cfg), cfg.Nr
    out = {"maskC": FArray(g["maskC"], "maskC", k=(1, Nr), **b)}
    for n in replay_io.GRID2:
        out[n] = FArray(g[n], n, **b)
    out["recip_deepFacC"] = FArray(g["recip_deepFacC"], "recip_deepFacC", k=(1, Nr), tiled=False)
    out["drF"] = FArray(g["drF"], "drF", k=(1, Nr), tiled=False)
    out["recip_drF"] = FArray(g["recip_drF"], "recip_drF", k=(1, Nr), tiled=False)
    out["recip_drC"] = FArray(g["recip_drC"], "recip_drC", k=(1, Nr + 1), tiled=False)
    return types.SimpleNamespace(**out)


def jax_inputs(R):
    """(fields, grid, deltaT, dtR) as jax arrays: the traced arguments of case_fn's function."""
    grid = {n: jnp.asarray(R.grid[n]) for n in R.grid}
    return ({n: jnp.asarray(a) for n, a in R.fields.items()}, grid, jnp.asarray(R.scal["deltaT"]),
            jnp.asarray(R.scal["dtR"]))


def case_fn(cfg, case, levels=None, drivers=None):
    """f(fields, grid, deltaT, dtR) -> flux [tile, k, j, i] of one harness case (driver, meth, calc_CFL): the X/Y
    drivers called once per level as the harness does (the levels not in `levels` keep the prior, which the
    comparison then skips), the R drivers once. `drivers` overrides the driver functions (negative controls)."""
    d, meth, calc = case
    fn = (drivers or DRIVERS)[d]
    Nr = cfg.Nr
    levels = tuple(range(1, Nr + 1)) if levels is None else tuple(levels)
    b = bounds(cfg)

    def f(fields, grid, deltaT, dtR):
        G = grid_farrays(cfg, grid)
        if d.endswith("_r"):
            k3 = dict(k=(1, Nr), **b)
            out = fn(meth, FArray(dtR, "delT", k=(1, Nr), tiled=False), FArray(fields["wVel"], "wvel", **k3),
                     FArray(fields["wTrans"], "wfac", **k3), FArray(fields["tracer"], "fbar", **k3),
                     FArray(fields["fluxPrior"], "flux", **k3), cfg=cfg, grid=G)
            return out.data
        x = d.endswith("_x")
        vel = ("uVel" if calc else "uCFL") if x else ("vVel" if calc else "vCFL")
        fac = "uTrans" if x else "vTrans"
        levs = []
        for k in range(1, Nr + 1):
            prior = fields["fluxPrior"][:, k - 1]
            if k not in levels:
                levs.append(prior)
                continue
            out = fn(meth, k, calc, deltaT, FArray(fields[vel][:, k - 1], "vel", **b),
                     FArray(fields[fac][:, k - 1], "fac", **b), FArray(fields["tracer"][:, k - 1], "fbar", **b),
                     FArray(prior, "flux", **b), cfg=cfg, grid=G)
            levs.append(out.data)
        return jnp.stack(levs, axis=1)
    return f


def bit_diff(a, b):
    """Number of elements whose float64 bit patterns differ."""
    a, b = np.ascontiguousarray(a, np.float64), np.ascontiguousarray(b, np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    return int(np.count_nonzero(a.view(np.int64) != b.view(np.int64)))


@contextlib.contextmanager
def planted(module, old, new):
    """A negative control: a fresh copy of `mitjax.pkg.generic_advdiff.<module>` with `old` replaced by `new`
    (exactly one occurrence); every module of the package that imported one of its functions gets the planted
    function for the duration of the block."""
    mod = importlib.import_module(f"{PKG}.{module}")
    src = Path(mod.__file__).read_text()
    if src.count(old) != 1:
        raise ValueError(f"plant target occurs {src.count(old)} times in {mod.__file__}: {old!r}")
    m = types.ModuleType(f"{mod.__name__}_planted")
    m.__file__ = mod.__file__
    exec(compile(src.replace(old, new), f"<planted {module}>", "exec"), m.__dict__)
    saved = []
    for name, other in list(sys.modules.items()):
        if not name.startswith(PKG + ".") or other is None:
            continue
        for attr, val in list(vars(other).items()):
            if callable(val) and getattr(val, "__module__", None) == mod.__name__ and hasattr(m, attr):
                saved.append((other, attr, val))
                setattr(other, attr, getattr(m, attr))
    try:
        yield {d: getattr(sys.modules[f"{PKG}.{d}"], d) for d in DRIVERS}
    finally:
        for other, attr, val in saved:
            setattr(other, attr, val)
