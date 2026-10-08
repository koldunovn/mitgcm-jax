"""Helpers of the GGL90 gates (mitjax/tests/test_ggl90.py; M3 sub-lane GGL90).

Two oracles:
  * the GGL90 replay harness reference/replay_ggl90/ (runs named by reference/replay_ggl90/CURRENT, relative to
    $MJX_REFERENCE): synthetic columns over the real grids of vermix/code (Langmuir, GGL90_MISSING_HFAC_BUG) and
    global_ocean.90x40x15/code (IDEMIX), several GGL90.h flag cases, every routine of pkg/ggl90 the M3 runs call;
  * the registered dumps-on runs (reference/reference_runs.py, kind "jdon") of vermix/input.ggl90, input.gglLC and
    global_ocean.90x40x15/input.idemix: GGL90_CALC replayed per dumped iteration from the oracle's own inputs
    (GGL90TKE, IDEMIX_E, uVel, vVel, surfaceForcingU/V and the output priors at S00_begin, sigmaR at
    P02_rho_sigma_ivdc, the grid at G00_geometry / S00_begin) and compared with P04_ggl90. PARAMS.h from our own
    INI_PARMS (ini_parms, ini_parms_dyn, ini_parms_tracer; vermix's diffKzS / viscAz / diffKzT aliases).
Comparison by element equality on every point of every tile, halos included, both arrays required finite (numpy on
the whole array), plus the count of differing bit patterns (sign of zeros). JAX transforms (jit, grad) are used here,
never in mitjax/pkg. IDEMIX runs with glibc_asin (mitjax/ops/libm.py) as in the model; `with_xla_asin` is a control.
"""

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mitjax import paths
from mitjax.farray import FArray
from mitjax.pkg.ggl90 import ggl90_calc as calc_mod
from mitjax.pkg.ggl90.ggl90_h import wrap
from mitjax.pkg.ggl90.ggl90_readparms import ggl90_readparms

REPO = Path(__file__).resolve().parents[2]
CASE_KEYS = ("mxlMaxFlag", "useLANGMUIR", "calcMeanVertShear", "GGL90_dirichlet", "mxlSurfFlag", "useIDEMIX")
DUMP_VARIANTS = (("vermix", "input.ggl90"), ("vermix", "input.gglLC"), ("global_ocean.90x40x15", "input.idemix"))
REPLAY_INPUT = {"vermix": "input.gglLC", "global_ocean.90x40x15": "input.idemix"}
OUT_FIELDS = ("GGL90TKE", "GGL90viscArU", "GGL90viscArV", "GGL90diffKr")


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


replay_io = _load_by_path("_mjx_replay_ggl90_io", REPO / "reference" / "replay_ggl90" / "replay_io.py")


@jax.tree_util.register_pytree_node_class
class Common:
    """A Fortran common block by name: arrays are leaves, `static` values aux data."""

    def __init__(self, static=None, **fields):
        object.__setattr__(self, "_static", dict(static or {}))
        object.__setattr__(self, "_fields", dict(fields))

    def __getattr__(self, name):
        for d in (object.__getattribute__(self, "_fields"), object.__getattribute__(self, "_static")):
            if name in d:
                return d[name]
        raise AttributeError(name)

    def tree_flatten(self):
        keys = tuple(sorted(self._fields))
        return tuple(self._fields[k] for k in keys), (keys, tuple(sorted(self._static.items())))

    @classmethod
    def tree_unflatten(cls, aux, children):
        keys, static = aux
        return cls(dict(static), **dict(zip(keys, children)))


_EXP = {}


def experiment(exp, inp):
    if (exp, inp) not in _EXP:
        from mitjax.config import params as cp
        _EXP[(exp, inp)] = cp.load(exp, inp)
    return _EXP[(exp, inp)]


def _ij(sz):
    return {"i": (1 - sz.OLx, sz.sNx + sz.OLx), "j": (1 - sz.OLy, sz.sNy + sz.OLy)}


def f3(name, a, sz, nk=None):
    return FArray(jnp.asarray(a, jnp.float64), name, k=(1, nk or sz.Nr), **_ij(sz))


def f2(name, a, sz):
    return FArray(jnp.asarray(a, jnp.float64), name, **_ij(sz))


def vec(name, a):
    a = np.asarray(a, np.float64)
    return FArray(jnp.asarray(a), name, k=(1, a.size), tiled=False)


# ---------------------------------------------------------------------------------------------------------------
# the replay harness

def current_runs():
    """{experiment: run directory} from reference/replay_ggl90/CURRENT."""
    out = {}
    for ln in (REPO / "reference" / "replay_ggl90" / "CURRENT").read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            exp, inp, rel = ln.split()
            out[exp] = Path(paths.REFERENCE) / rel
    return out


class Replay:
    """One harness run: inputs, Fortran outputs, parameters and grid as the Fortran read them, the experiment."""

    def __init__(self, exp, rundir):
        self.exp, self.inp, self.rundir = exp, REPLAY_INPUT[exp], Path(rundir)
        self.e = experiment(exp, self.inp)
        self.cfg = self.e.cfg
        sz = self.size = self.cfg.size
        s = {k: getattr(sz, k) for k in replay_io.SIZE_KEYS}
        self.inputs = replay_io.read_inputs(self.rundir, s)
        self.parm = replay_io.read_file(self.rundir, "ggl_parm.bin", s)
        self.out = replay_io.read_file(self.rundir, "ggl_out.bin", s)
        self.g = replay_io.read_file(self.rundir, "ggl_grid.bin", s)

    def grid(self):
        return grid_from(self.g, self.size)

    def params(self):
        return params_from(self.g)

    def state(self):
        sz, x = self.size, self.inputs
        return Common(uVel=f3("uVel", x["uVel"], sz), vVel=f3("vVel", x["vVel"], sz),
                      surfaceForcingU=f2("surfaceForcingU", x["sfU"], sz),
                      surfaceForcingV=f2("surfaceForcingV", x["sfV"], sz))

    def ggl_case(self, case):
        """GGL90.h for one case: GGL90_READPARMS of the run with the case's flags, the input fields."""
        sz, x = self.size, self.inputs
        ggl = ggl90_readparms(self.e)
        flags = {k: (int(case[k]) if k == "mxlMaxFlag" else bool(case[k])) for k in CASE_KEYS}
        flags["adMxlMaxFlag"] = flags["mxlMaxFlag"]
        fields = {"GGL90TKE": x["TKE"], "GGL90viscArU": x["viscArUPrior"], "GGL90viscArV": x["viscArVPrior"],
                  "GGL90diffKr": x["diffKrPrior"]}
        if self.cfg.cpp.flag("ALLOW_GGL90_IDEMIX", "GGL90_OPTIONS.h"):
            fields["IDEMIX_E"] = x["IDEMIX_E"]
            fields["IDEMIX_F_B"] = self.parm["init_F_B"]
            fields["IDEMIX_F_S"] = self.parm["init_F_S"]
        return ggl.replace(**flags, **{k: wrap(k, v, sz) for k, v in fields.items()})


def grid_from(g, sz):
    """GRID.h fields GGL90 reads, from a named-record dict (harness ggl_grid.bin or dump-derived)."""
    grid = {n: f3(n, g[n], sz) for n in ("hFacC", "hFacW", "hFacS", "recip_hFacC", "maskC", "maskW", "maskS")}
    grid.update({n: f2(n, g[n], sz) for n in ("dxF", "dyF", "dxG", "dyG", "recip_dxC", "recip_dyC", "recip_rA",
                                               "Ro_surf", "R_low", "fCori")})
    grid["kLowC"] = FArray(jnp.asarray(np.asarray(g["kLowC"]).astype(np.int32)), "kLowC", **_ij(sz))
    grid.update({n: vec(n, g[n]) for n in ("rC", "rF", "drC", "drF", "recip_drC", "recip_drF")})
    grid["gravitySign"] = jnp.float64(g["gravitySign"])
    return Common(**grid)


def params_from(g):
    """PARAMS.h values GGL90 reads (harness dump of the same build and namelists); z coordinates."""
    return Common({"usingPCoords": False, "usingZCoords": True},
                  gravity=jnp.float64(g["gravity"]), recip_rhoConst=jnp.float64(g["recip_rhoConst"]),
                  dTtracerLev=vec("dTtracerLev", g["dTtracerLev"]), diffKrNrS=vec("diffKrNrS", g["diffKrNrS"]),
                  viscArNr=vec("viscArNr", g["viscArNr"]))


def calc_jit(cfg, module=None):
    """GGL90_CALC under jit: grid, params, ggl (REAL parameters traced), state and sigmaR are arguments; cfg closed
    over (static). `module`: a planted copy of mitjax.pkg.ggl90.ggl90_calc."""
    m = module or calc_mod

    def f(grid, params, ggl, state, sigmaR):
        return m.ggl90_calc(sigmaR, 0.0, 0, cfg=cfg, grid=grid, params=params, ggl=ggl, state=state)
    return jax.jit(f)


def compare(ours, ref):
    """(n points, n differing (==), n non-finite ours, n non-finite oracle, n differing bit patterns)."""
    o, r = np.ascontiguousarray(np.asarray(ours, np.float64)), np.ascontiguousarray(np.asarray(ref, np.float64))
    assert o.shape == r.shape, (o.shape, r.shape)
    return (o.size, int(np.count_nonzero(~(o == r))), int(np.count_nonzero(~np.isfinite(o))),
            int(np.count_nonzero(~np.isfinite(r))), int(np.count_nonzero(o.view(np.int64) != r.view(np.int64))))


def bad(res):
    return {k: v for k, v in res.items() if any(v[1:])}


@lru_cache(maxsize=None)
def replay(exp):
    return Replay(exp, current_runs()[exp])


def replay_case(exp, ic, module=None, ggl_override=None):
    """{field: compare} of GGL90_CALC case ic (1-based) of the harness run of `exp`."""
    rep = replay(exp)
    case = rep.inputs["cases"][ic - 1]
    ggl = rep.ggl_case(case)
    if ggl_override is not None:
        ggl = ggl_override(ggl)
    sig = f3("sigmaR", rep.inputs["sigmaR"], rep.size)
    out = calc_jit(rep.cfg, module)(rep.grid(), rep.params(), ggl, rep.state(), sig)
    names = {"GGL90TKE": "TKE", "GGL90viscArU": "viscArU", "GGL90viscArV": "viscArV", "GGL90diffKr": "diffKr"}
    res = {n: compare(getattr(out, n).data, rep.out[f"{s}_c{ic}"]) for n, s in names.items()}
    if f"E_c{ic}" in rep.out:
        res["IDEMIX_E"] = compare(out.IDEMIX_E.data, rep.out[f"E_c{ic}"])
    return res, out


def planted(module, old, new, rebind=None):
    """A copy of mitjax.pkg.ggl90.<module> with `old` replaced by `new` (exactly once); `rebind` {name: object}
    replaces module globals (e.g. a planted callee)."""
    path = REPO / "mitjax" / "pkg" / "ggl90" / f"{module}.py"
    src = path.read_text()
    if src.count(old) != 1:
        raise ValueError(f"planted: {old!r} occurs {src.count(old)} times in {path.name}")
    name = f"_mjx_planted_{module}_{abs(hash((old, new)))}"
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src.replace(old, new), str(path), "exec"), mod.__dict__)
    for k, v in (rebind or {}).items():
        mod.__dict__[k] = v
    return mod


# ---------------------------------------------------------------------------------------------------------------
# negative control: XLA's arcsin in IDEMIX (the physics code uses glibc_asin)

def with_xla_asin():
    """A GGL90_CALC module copy whose IDEMIX uses jnp.arcsin (measured: ~650-1800 points differ)."""
    idm = planted("ggl90_idemix", "ASIN = glibc_asin      # the oracle's libm asin (module docstring)",
                  "ASIN = jnp.arcsin")
    return planted("ggl90_calc", "from mitjax.pkg.ggl90.ggl90_idemix import ggl90_idemix",
                   "from mitjax.pkg.ggl90.ggl90_idemix import ggl90_idemix  # rebound",
                   rebind={"ggl90_idemix": idm.ggl90_idemix})


# ---------------------------------------------------------------------------------------------------------------
# the dumps-on runs

def _registry():
    return _load_by_path("_mjx_reference_runs_ggl90", REPO / "reference" / "reference_runs.py")


@lru_cache(maxsize=None)
def dumpset(exp, inp):
    from mitjax.io.dump import DumpSet
    reg = _registry()
    runs = [r for r in reg.RUNS if (r["exp"], r["input"], r["kind"]) == (exp, inp, "jdon")]
    if len(runs) != 1:
        raise FileNotFoundError(f"{exp}/{inp}: {len(runs)} registered dumps-on runs")
    return DumpSet(reg.run_top(runs[0]) / "dumps")


def dump_grid(ds, it, sz):
    """GRID.h fields from the dumps (G00_geometry; the hFac arrays at S00_begin); kind-V records -> vectors."""
    g = {}
    for n in ("maskC", "maskW", "maskS"):
        g[n] = ds.field(it, "G00_geometry", n)
    for n in ("hFacC", "hFacW", "hFacS", "recip_hFacC"):
        g[n] = ds.field(it, "S00_begin", n)
    for n in ("dxF", "dyF", "dxG", "dyG", "recip_dxC", "recip_dyC", "recip_rA", "Ro_surf", "R_low", "fCori",
              "kLowC"):
        g[n] = ds.field(it, "G00_geometry", n)[:, 0]
    for n in ("rC", "rF", "drC", "drF", "recip_drC", "recip_drF"):
        g[n] = ds.field(it, "G00_geometry", n)[0, :, 0, 0]
    g["gravitySign"] = float(ds.field(it, "G00_geometry", "gravitySign")[0, 0, 0, 0])
    return g


@lru_cache(maxsize=None)
def model_params(exp, inp):
    """PARAMS.h of the variant from our own INI_PARMS (ini_parms, ini_parms_dyn, ini_parms_tracer)."""
    from mitjax.model.src.ini_parms import ini_parms, ini_parms_dyn
    from mitjax.model.src.ini_parms_tracer import ini_parms_tracer
    from mitjax.tests import grid_gate
    e = experiment(exp, inp)
    ex2 = grid_gate.exch2_topology(dumpset(exp, inp)) if e.cfg.cpp.ALLOW_EXCH2 else None
    prm = ini_parms(e, ex2)
    p = ini_parms_dyn(e, prm.grid, prm.time, prm.init)
    return ini_parms_tracer(e, p, prm.time, prm.init)


def dump_case(exp, inp, it, module=None):
    """GGL90_CALC for dumped iteration `it` from the oracle's inputs: ({field: compare} vs P04_ggl90, ours)."""
    e = experiment(exp, inp)
    cfg, sz = e.cfg, e.cfg.size
    ds = dumpset(exp, inp)
    g = dump_grid(ds, it, sz)
    ggl = ggl90_readparms(e)
    names = list(OUT_FIELDS)
    if cfg.cpp.flag("ALLOW_GGL90_IDEMIX", "GGL90_OPTIONS.h"):
        names += ["IDEMIX_E", "IDEMIX_F_B", "IDEMIX_F_S"]
    prior = {}
    for n in names:
        a = ds.field(it, "S00_begin", n)
        prior[n] = wrap(n, a[:, 0] if n in ("IDEMIX_F_B", "IDEMIX_F_S") else a, sz)
    ggl = ggl.replace(**prior)
    st = Common(uVel=f3("uVel", ds.field(it, "S00_begin", "uVel"), sz),
                vVel=f3("vVel", ds.field(it, "S00_begin", "vVel"), sz),
                surfaceForcingU=f2("surfaceForcingU", ds.field(it, "P01_external_forcing_surf",
                                                               "surfaceForcingU")[:, 0], sz),
                surfaceForcingV=f2("surfaceForcingV", ds.field(it, "P01_external_forcing_surf",
                                                               "surfaceForcingV")[:, 0], sz))
    sig = f3("sigmaR", ds.field(it, "P02_rho_sigma_ivdc", "sigmaR"), sz)
    out = calc_jit(cfg, module)(grid_from(g, sz), model_params(exp, inp), ggl, st, sig)
    res = {}
    for n in names:
        ref = ds.field(it, "P04_ggl90", n)
        o = np.asarray(getattr(out, n).data)
        res[n] = compare(o[:, None] if o.ndim == 3 else o, ref)
    return res, out
