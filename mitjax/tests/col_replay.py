"""Helpers of the COL gates (mitjax/tests/test_column_physics.py): the replay harness runs named by
reference/replay_col/CURRENT (relative to $MJX_REFERENCE), the common blocks (GRID.h, PARAMS.h, EOS.h, DYNVARS.h)
as FArray pytrees built from the harness's own dump of what the routines read, one driver per harness output that
calls the routine for every tile and level as the harness (and the model's caller) does, planted-error copies of a
kernel module, and bit-pattern comparisons. JAX transforms (jit, grad) are used here, never in mitjax/model.
"""

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mitjax import paths
from mitjax.farray import FArray
from mitjax.model.src.ini_eos import EOS, _VECTORS

REPO = Path(__file__).resolve().parents[2]


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


replay_io = _load_by_path("_mjx_replay_col_io", REPO / "reference" / "replay_col" / "replay_io.py")


def current_runs():
    """[(experiment, input_dir, rundir)] from reference/replay_col/CURRENT."""
    out = []
    for ln in (REPO / "reference" / "replay_col" / "CURRENT").read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            exp, inp, rel = ln.split()
            out.append((exp, inp, Path(paths.REFERENCE) / rel))
    return out


@jax.tree_util.register_pytree_node_class
class Common:
    """A Fortran common block: fields by their Fortran names. Arrays are pytree leaves (traced under jit); `static`
    holds the integer/logical/character values that select branches (aux data, Python values), and `static_float`
    the concrete host value of a REAL that selects a branch (hMixCriteria, hMixSmooth: the traced copy is the field)."""

    def __init__(self, static=None, floats=None, **fields):
        object.__setattr__(self, "_static", dict(static or {}))
        object.__setattr__(self, "_floats", dict(floats or {}))
        object.__setattr__(self, "_fields", dict(fields))

    def __getattr__(self, name):
        for d in (object.__getattribute__(self, "_fields"), object.__getattribute__(self, "_static")):
            if name in d:
                return d[name]
        raise AttributeError(name)

    def static_float(self, name):
        return self._floats[name]

    def replace(self, **kw):
        return Common(self._static, self._floats, **{**self._fields, **kw})

    def tree_flatten(self):
        keys = tuple(sorted(self._fields))
        return tuple(self._fields[k] for k in keys), (keys, tuple(sorted(self._static.items())),
                                                       tuple(sorted(self._floats.items())))

    @classmethod
    def tree_unflatten(cls, aux, children):
        keys, static, floats = aux
        return cls(dict(static), dict(floats), **dict(zip(keys, children)))


class Cfg:
    """The kernel cfg: the experiment's real ExperimentConfig (cfg.cpp, cfg.size, cfg.use_flag), hashable."""

    def __init__(self, c):
        self.cpp, self.size, self._c = c.cpp, c.size, c

    def use_flag(self, name):
        return self._c.use_flag(name)


_CFG = {}


def experiment(exp, inp):
    if (exp, inp) not in _CFG:
        from mitjax.config import params as cp
        _CFG[(exp, inp)] = cp.load(exp, inp)
    return _CFG[(exp, inp)]


def _ij(size):
    return {"i": (1 - size.OLx, size.sNx + size.OLx), "j": (1 - size.OLy, size.sNy + size.OLy)}


GRID3 = ("hFacC", "hFacW", "hFacS", "recip_hFacC", "recip_hFacW", "recip_hFacS", "maskC", "maskW", "maskS",
         "h0FacC")
GRID2 = ("dxG", "dyG", "dxC", "dyC", "recip_dxC", "recip_dyC", "rA", "recip_rA", "Ro_surf", "R_low", "etaH")
GRID_INT2 = ("kLowC", "kSurfC")
# (name, upper bound: "r" = Nr, "rp1" = Nr+1)
GRID1 = (("rC", "r"), ("rF", "rp1"), ("drC", "rp1"), ("drF", "r"), ("recip_drC", "rp1"), ("recip_drF", "r"),
         ("deepFacC", "r"), ("deepFac2F", "rp1"), ("recip_deepFac2C", "r"), ("recip_deepFac2F", "rp1"))
PARAMS1 = (("rhoFacC", "r"), ("rhoFacF", "rp1"), ("recip_rhoFacC", "r"), ("recip_rhoFacF", "rp1"),
           ("gravFacC", "r"), ("gravFacF", "rp1"), ("viscArNr", "r"), ("dTtracerLev", "r"), ("tRef", "r"),
           ("sRef", "r"), ("pRef4EOS", "r"))
PARAMS0 = ("rhoNil", "rhoConst", "recip_rhoConst", "gravity", "surf_pRef", "deltaTMom", "hMixCriteria", "dRhoSmall",
           "hMixSmooth", "ivdc_kappa")


class Replay:
    """One harness run: inputs, Fortran outputs, the dumped grid/parameters, the experiment config."""

    def __init__(self, exp, inp, rundir):
        self.exp, self.inp, self.rundir = exp, inp, Path(rundir)
        self.experiment = experiment(exp, inp)
        self.cfg = Cfg(self.experiment.cfg)
        self.size = self.experiment.cfg.size
        sz = {k: getattr(self.size, k) for k in replay_io.SIZE_KEYS}
        self.inputs = replay_io.read_inputs(self.rundir, sz)
        self.out = replay_io.read_outputs(self.rundir, sz)
        self.g = replay_io.read_grid(self.rundir, sz)

    # -- common blocks --------------------------------------------------------------------------------------------
    def eosType(self):
        v = [x.value for (f, g, k), x in self.experiment.run.vars.items() if f == "data" and k == "eostype"
             and x.value is not None]
        # set_defaults.F:175  eosType = 'LINEAR'
        return (v[0] if v else "LINEAR").ljust(6)[:6]

    def commons(self, overrides=None):
        """(grid, params, eos) Common/EOS pytrees of the dumped values (arrays traced when passed to jit)."""
        sz, g = self.size, self.g
        Nr = sz.Nr
        ij = _ij(sz)
        kb = {"r": (1, Nr), "rp1": (1, Nr + 1)}
        grid = {n: FArray(jnp.asarray(g[n]), n, k=(1, Nr), **ij) for n in GRID3 if n in g}
        grid.update({n: FArray(jnp.asarray(g[n]), n, **ij) for n in GRID2})
        grid.update({n: FArray(jnp.asarray(g[n].astype(np.int32)), n, **ij) for n in GRID_INT2})
        grid.update({n: FArray(jnp.asarray(g[n]), n, k=kb[b], tiled=False) for n, b in GRID1})
        grid["gravitySign"] = jnp.float64(g["gravitySign"])
        grid["rkSign"] = jnp.float64(g["rkSign"])
        par = {n: FArray(jnp.asarray(g[n]), n, k=kb[b], tiled=False) for n, b in PARAMS1}
        par["phiRef"] = FArray(jnp.asarray(g["phiRef"]), "phiRef", k=(1, 2 * Nr + 1), tiled=False)
        par.update({n: jnp.float64(g[n]) for n in PARAMS0})
        static = {"selectP_inEOS_Zc": int(g["selectP_inEOS_Zc"]), "select_rStar": int(g["select_rStar"]),
                  "selectSigmaCoord": int(g["selectSigmaCoord"]), "rigidLid": bool(g["rigidLid"]),
                  "usingPCoords": bool(g["usingPCoords"]), "usingZCoords": not bool(g["usingPCoords"]),
                  "useDiagnostics": bool(g["useDiagnostics"]),
                  # ini_parms.F:1214, 1216 (PARM04 defaults; no M1 data file sets them)
                  "interViscAr_pCell": False, "pCellMix_select": 0}
        floats = {"hMixCriteria": float(g["hMixCriteria"]), "hMixSmooth": float(g["hMixSmooth"])}
        params = Common(static, floats, **par)
        vec = {}
        for n, (lo, hi) in _VECTORS.items():
            data = g[n] if n in g else np.zeros(hi - lo + 1)
            vec[n] = FArray(jnp.asarray(data), n, k=(lo, hi), tiled=False)
        eos = EOS(equationOfState=self.eosType(), eosRefP0=jnp.float64(g["eosRefP0"]),
                  tAlpha=jnp.float64(g["tAlpha"]), sBeta=jnp.float64(g["sBeta"]), **vec)
        return Common(**grid), params, eos

    def fields3(self, name):
        return FArray(jnp.asarray(self.inputs[name]), name, k=(1, self.size.Nr), **_ij(self.size))

    def fields2(self, name):
        return FArray(jnp.asarray(self.inputs[name]), name, **_ij(self.size))


def bit_equal(got, want):
    """(n_differ, n_total, both_finite): element equality on all points by bit pattern (signed zeros count)."""
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    assert got.shape == want.shape, (got.shape, want.shape)
    finite = bool(np.all(np.isfinite(got)) and np.all(np.isfinite(want)))
    n = int(np.count_nonzero(got.view(np.int64) != want.view(np.int64)))
    return n, got.size, finite


def planted(module_name, old, new):
    """A copy of mitjax.model.src.<module_name> with the text `old` replaced by `new` (must occur exactly once)."""
    path = REPO / "mitjax" / "model" / "src" / f"{module_name}.py"
    src = path.read_text()
    if src.count(old) != 1:
        raise ValueError(f"planted: {old!r} occurs {src.count(old)} times in {path.name}")
    name = f"_mjx_planted_{module_name}_{abs(hash((old, new)))}"
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src.replace(old, new), str(path), "exec"), mod.__dict__)
    return mod
