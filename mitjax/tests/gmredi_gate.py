"""Helpers of the GM/Redi gates (M1 sub-lane GMREDI; plan Task 15b's kernels).

Two oracles:
  * the registered dumps-on runs (reference/reference_runs.py, kind "jdon") of global_ocean.90x40x15/input (gkw91)
    and tutorial_global_oce_optim/input_ad (dm95, the forward-only code_ad build): GMREDI_CALC_TENSOR replayed per
    dumped iteration from the oracle's own inputs (sigmaX/Y/R at P02_rho_sigma_ivdc, hMixLayer at P03_mxlayer, the
    tensor at S00_begin as the prior of the points it does not write) and compared with P05_gmredi_tensor;
  * the GM/Redi replay harness reference/replay_gmredi/ (synthetic stratification over the real grids, every ported
    routine; read with reference/replay_gmredi/replay_io.py).
Comparison by element equality on every point of every tile, halos included, both arrays required finite (numpy on the
whole array), plus the count of differing bit patterns (sign of zeros).
"""

import dataclasses
import importlib.util
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from mitjax import paths
from mitjax.farray import FArray
from mitjax.pkg.gmredi.gmredi_calc_tensor import gmredi_calc_tensor
from mitjax.pkg.gmredi.gmredi_h import DECLARATIONS as GM_DECL
from mitjax.pkg.gmredi.gmredi_h import wrap
from mitjax.pkg.gmredi.gmredi_init_fixed import gmredi_init_fixed
from mitjax.pkg.gmredi.gmredi_init_varia import gmredi_init_varia
from mitjax.pkg.gmredi.gmredi_readparms import gmredi_readparms
from mitjax.tests import grid_gate

EXPERIMENTS = [("global_ocean.90x40x15", "input"), ("tutorial_global_oce_optim", "input_ad")]
TENSOR = ("Kwx", "Kwy", "Kwz", "Kux", "Kvy", "Kuz", "Kvz")


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ParamsH:
    """The PARAMS.h values the GM/Redi kernels read (until the core lane's PARAMS container exists)."""
    rVel2wUnit: FArray
    wUnit2rVel: FArray
    rUnit2z: FArray
    z2rUnit: FArray
    usingZCoords: bool = dataclasses.field(metadata=dict(static=True))


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StateH:
    """The DYNVARS.h field GMREDI_CALC_TENSOR reads."""
    hMixLayer: FArray


def vec(name, data, n):
    return FArray(jnp.asarray(data, jnp.float64), name, k=(1, n), tiled=False)


def set_ref_state_units(size):
    """rUnit2z, z2rUnit (Nr) and rVel2wUnit, wUnit2rVel (Nr+1) as SET_REF_STATE leaves them in z-coordinates
    (model/src/set_ref_state.F:56-70 @63cdc0b: `rUnit2z(k) = 1. _d 0`, `z2rUnit(k) = 1. _d 0`, `rVel2wUnit(k) =
    1. _d 0`, `wUnit2rVel(k) = 1. _d 0`; nothing later in the routine changes them for usingZCoords)."""
    Nr = size.Nr
    return {"rUnit2z": np.ones(Nr), "z2rUnit": np.ones(Nr), "rVel2wUnit": np.ones(Nr + 1),
            "wUnit2rVel": np.ones(Nr + 1)}


@lru_cache(maxsize=None)
def setup(exp, inp):
    """(Experiment, Gmredi after GMREDI_READPARMS + GMREDI_INIT_FIXED + GMREDI_INIT_VARIA, Grid (the Task 10 port),
    ParamsH)."""
    e = grid_gate.experiment(exp, inp)
    cfg = e.cfg
    gm = gmredi_readparms(e)
    gm = gmredi_init_fixed(gm, cfg=cfg)
    gm = gmredi_init_varia(gm, cfg=cfg)
    gp = grid_gate.grid_params(exp, inp)
    grid = grid_gate.build_grid(exp, inp, params=gp)
    u = set_ref_state_units(cfg.size)
    Nr = cfg.size.Nr
    params = ParamsH(rVel2wUnit=vec("rVel2wUnit", u["rVel2wUnit"], Nr + 1),
                     wUnit2rVel=vec("wUnit2rVel", u["wUnit2rVel"], Nr + 1),
                     rUnit2z=vec("rUnit2z", u["rUnit2z"], Nr), z2rUnit=vec("z2rUnit", u["z2rUnit"], Nr),
                     usingZCoords=bool(gp.usingZCoords))
    return e, gm, grid, params


def xyz(name, data, size):
    sNx, sNy, OLx, OLy, Nr = size.sNx, size.sNy, size.OLx, size.OLy, size.Nr
    return FArray(jnp.asarray(data, jnp.float64), name, i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy), k=(1, Nr))


def xy(name, data, size):
    sNx, sNy, OLx, OLy = size.sNx, size.sNy, size.OLx, size.OLy
    return FArray(jnp.asarray(data, jnp.float64), name, i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy))


def tensor_names(cfg):
    return TENSOR + (("GM_PsiX", "GM_PsiY") if cfg.cpp.GM_BOLUS_ADVEC else ())


def calc_tensor_jit(cfg, iMin, iMax, jMin, jMax):
    """GMREDI_CALC_TENSOR under jit: grid, params, gm (REAL parameters traced), state and the sigma fields are jit
    arguments; cfg and the loop bounds are static (closed over)."""
    def f(grid, params, gm, state, sigmaX, sigmaY, sigmaR, myTime):
        return gmredi_calc_tensor(iMin, iMax, jMin, jMax, sigmaX, sigmaY, sigmaR, myTime, 0,
                                  cfg=cfg, grid=grid, params=params, gm=gm, state=state)
    return jax.jit(f)


class CppOverride:
    """cfg.cpp with some flags replaced (the planted error of a negative control)."""

    def __init__(self, cpp, **flags):
        self._cpp, self._flags = cpp, flags

    def __getattr__(self, name):
        return self._flags[name] if name in self._flags else getattr(self._cpp, name)


class CfgOverride:
    """An ExperimentConfig whose CPP flags `flags` are replaced; everything else is the experiment's."""

    def __init__(self, cfg, **flags):
        self._cfg, self.cpp = cfg, CppOverride(cfg.cpp, **flags)

    def __getattr__(self, name):
        return getattr(self._cfg, name)


def dump_case(exp, inp, it, gm_override=None, prior_override=None, cfg_override=None):
    """Replay GMREDI_CALC_TENSOR for dumped iteration `it` from the oracle's inputs; returns ({name: ours}, {name:
    oracle}) as [tile, k, j, i] numpy arrays. `gm_override(gm) -> gm`, `prior_override(name, array) -> array` and
    `cfg_override(cfg) -> cfg` plant the errors of the negative controls."""
    e, gm, grid, params = setup(exp, inp)
    cfg, sz = e.cfg, e.cfg.size
    if cfg_override is not None:
        cfg = cfg_override(cfg)
    ds, _, _ = grid_gate.oracle(exp, inp)
    names = tensor_names(cfg)
    prior = {}
    for n in names:
        a = ds.field(it, "S00_begin", n)
        prior[n] = prior_override(n, a) if prior_override else a
    gm = gm.replace(**{n: wrap(n, jnp.asarray(prior[n]), sz) for n in names})
    if gm_override is not None:
        gm = gm_override(gm)
    state = StateH(hMixLayer=xy("hMixLayer", ds.field(it, "P03_mxlayer", "hMixLayer")[:, 0], sz))
    sig = [xyz(n, ds.field(it, "P02_rho_sigma_ivdc", n), sz) for n in ("sigmaX", "sigmaY", "sigmaR")]
    # do_oceanic_phys.F:560-563 (iMin = 1-OLx, iMax = sNx+OLx, jMin = 1-OLy, jMax = sNy+OLy)
    f = calc_tensor_jit(cfg, 1-sz.OLx, sz.sNx+sz.OLx, 1-sz.OLy, sz.sNy+sz.OLy)
    out = f(grid, params, gm, state, *sig, jnp.float64(0.0))
    ours = {n: np.asarray(getattr(out, n).data) for n in names}
    ref = {n: ds.field(it, "P05_gmredi_tensor", n) for n in names}
    return ours, ref


def compare_fields(ours, ref):
    """{name: (n points, n differing (==), n non-finite ours, n non-finite oracle, n differing bit patterns)}."""
    out = {}
    for n in ref:
        o, r = np.asarray(ours[n], np.float64), np.asarray(ref[n], np.float64)
        if o.shape != r.shape:
            out[n] = ("shape", o.shape, r.shape)
            continue
        o, r = np.ascontiguousarray(o), np.ascontiguousarray(r)
        out[n] = (o.size, int(np.count_nonzero(~(o == r))), int(np.count_nonzero(~np.isfinite(o))),
                  int(np.count_nonzero(~np.isfinite(r))), int(np.count_nonzero(o.view(np.int64) != r.view(np.int64))))
    return out


def failures(result):
    return {k: v for k, v in result.items() if v[0] == "shape" or any(v[1:])}


def n_differing(result):
    return sum(v[1] for v in result.values() if v[0] != "shape")


# ---------------------------------------------------------------------------------------------------------------
# the replay harness (reference/replay_gmredi)

def _replay_io():
    spec = importlib.util.spec_from_file_location("_mjx_replay_gmredi_io",
                                                  paths.REPO / "reference" / "replay_gmredi" / "replay_io.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def replay_rundir(exp):
    """The harness run directory of `exp` named by reference/replay_gmredi/CURRENT_<exp> (relative to
    $MJX_REFERENCE)."""
    p = paths.REPO / "reference" / "replay_gmredi" / f"CURRENT_{exp}"
    if not p.exists():
        raise FileNotFoundError(f"{p}: no replay run registered for {exp}")
    return paths.REFERENCE / p.read_text().strip()


def _bounds(kind, size):
    sNx, sNy, OLx, OLy, Nr = size.sNx, size.sNy, size.OLx, size.OLy, size.Nr
    return {"xy": dict(i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy)),
            "xyz": dict(i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy), k=(1, Nr)),
            "r": dict(k=(1, Nr), tiled=False), "rp1": dict(k=(1, Nr+1), tiled=False)}[kind]


def fa(name, data, kind, size, dtype=jnp.float64):
    b = _bounds(kind, size)
    tiled = b.pop("tiled", True)
    return FArray(jnp.asarray(data, dtype), name, **b, tiled=tiled)


@lru_cache(maxsize=None)
def replay_case(exp):
    """Everything of one harness run: (cfg, inputs, outputs, grid file, fixed file) with the inputs as numpy."""
    io = _replay_io()
    inp = dict(EXPERIMENTS)[exp]
    e = grid_gate.experiment(exp, inp)
    rd = replay_rundir(exp)
    size, f3, f2, par, one_d = io.read_inputs(rd)
    sz = e.cfg.size
    if (size["sNx"], size["sNy"], size["OLx"], size["OLy"], size["nSx"], size["nSy"], size["Nr"]) != \
            (sz.sNx, sz.sNy, sz.OLx, sz.OLy, sz.nSx, sz.nSy, sz.Nr):
        raise ValueError(f"{rd}: harness size {size} != {sz}")
    return (e, dict(f3=f3, f2=f2, par=par, one_d=one_d), io.read_outputs(rd, size), io.read_grid(rd, size),
            io.read_fixed(rd, size))


def replay_grid(g, size):
    """GRID.h of the harness (as the routines read it, after its overrides) as a mitjax Grid."""
    from mitjax.model.grid import Grid
    f = {n: fa(n, g[n], "xyz", size) for n in ("maskC", "maskW", "maskS")}
    f.update({n: fa(n, g[n], "xy", size) for n in ("R_low", "recip_dxC", "recip_dyC", "rA", "maskInC")})
    f["kLowC"] = fa("kLowC", g["kLowC"].astype(np.int32), "xy", size, dtype=jnp.int32)
    f.update({n: fa(n, g[n], "rp1", size) for n in ("rF", "recip_drC", "deepFac2F")})
    f.update({n: fa(n, g[n], "r", size) for n in ("rC", "recip_deepFacC")})
    f["gravitySign"] = jnp.float64(g["gravitySign"])
    return Grid(f)


def replay_params(g, size):
    return ParamsH(rVel2wUnit=fa("rVel2wUnit", g["rVel2wUnit"], "rp1", size),
                   wUnit2rVel=fa("wUnit2rVel", g["wUnit2rVel"], "rp1", size),
                   rUnit2z=fa("rUnit2z", g["rUnit2z"], "r", size), z2rUnit=fa("z2rUnit", g["z2rUnit"], "r", size),
                   usingZCoords=g["usingZCoords"])


def replay_gm(e, inp, g, size):
    """GMREDI.h after our GMREDI_READPARMS, GMREDI_INIT_FIXED, GMREDI_INIT_VARIA and the harness overrides."""
    cfg = e.cfg
    gm = gmredi_init_varia(gmredi_init_fixed(gmredi_readparms(e), cfg=cfg), cfg=cfg)
    return gm.replace(GM_isopycK=np.float64(inp["par"]["GM_isopycK"]), GM_skewflx=np.float64(inp["par"]["GM_skewflx"]),
                      GM_isoFac1d=fa("GM_isoFac1d", inp["one_d"]["GM_isoFac1d"], "r", size),
                      GM_bolFac1d=fa("GM_bolFac1d", inp["one_d"]["GM_bolFac1d"], "r", size),
                      GM_isoFac2d=fa("GM_isoFac2d", inp["f2"]["GM_isoFac2d"], "xy", size),
                      GM_bolFac2d=fa("GM_bolFac2d", inp["f2"]["GM_bolFac2d"], "xy", size))


PRIOR = {"Kwx": "KwxP", "Kwy": "KwyP", "Kwz": "KwzP", "Kux": "KuxP", "Kvy": "KvyP", "Kuz": "KuzP", "Kvz": "KvzP",
         "GM_PsiX": "PsiXP", "GM_PsiY": "PsiYP"}


def with_priors(gm, inp, cfg, names=None):
    size = cfg.size
    names = names or tensor_names(cfg)
    return gm.replace(**{n: wrap(n, jnp.asarray(inp["f3"][PRIOR[n]]), size) for n in names})


def replay_run(exp, patch=None):
    """Our port on the harness inputs: {output name: [tile,k,j,i]} for every OUT3 entry this build defines, and the
    matching oracle arrays. `patch(objs) -> objs` plants the error of a negative control in the dict
    {grid, params, gm, state, cfg}."""
    from mitjax.pkg.gmredi.gmredi_calc_diff import gmredi_calc_diff
    from mitjax.pkg.gmredi.gmredi_residual_flow import gmredi_residual_flow
    from mitjax.pkg.gmredi.gmredi_rtransport import gmredi_rtransport
    from mitjax.pkg.gmredi.gmredi_slope_limit import gmredi_slope_limit
    from mitjax.pkg.gmredi.gmredi_xtransport import gmredi_xtransport
    from mitjax.pkg.gmredi.gmredi_ytransport import gmredi_ytransport
    e, inp, out, g, _ = replay_case(exp)
    cfg = e.cfg
    sz = cfg.size
    sNx, sNy, OLx, OLy, Nr = sz.sNx, sz.sNy, sz.OLx, sz.OLy, sz.Nr
    f3 = inp["f3"]
    objs = dict(grid=replay_grid(g, sz), params=replay_params(g, sz), cfg=cfg,
                gm=replay_gm(e, inp, g, sz), state=StateH(hMixLayer=fa("hMixLayer", inp["f2"]["hMixLayer"], "xy", sz)))
    if patch is not None:
        objs = patch(objs)
    grid, params, gm, state, cfg = objs["grid"], objs["params"], objs["gm"], objs["state"], objs["cfg"]
    names = tensor_names(cfg)
    ours, ref = {}, {}

    # GMREDI_INIT_VARIA on prior-filled tensors (the harness's overrides come after it; the tensor is all it writes)
    gm0 = with_priors(gm, inp, cfg)
    r0 = jax.jit(lambda gm_: gmredi_init_varia(gm_, cfg=cfg))(gm0)
    for n in names:
        ours[f"initvaria_{n}"] = np.asarray(getattr(r0, n).data)

    # GMREDI_CALC_TENSOR: sigma = input times maskW/S/C, as the harness passes them
    sig = [xyz(n, f3[n] * np.asarray(g[m]), sz) for n, m in (("sigmaX", "maskW"), ("sigmaY", "maskS"),
                                                             ("sigmaR", "maskC"))]
    f = calc_tensor_jit(cfg, 1-OLx, sNx+OLx, 1-OLy, sNy+OLy)
    r1 = f(grid, params, gm0, state, *sig, jnp.float64(0.0))
    for n in names:
        ours[f"tensor_{n}"] = np.asarray(getattr(r1, n).data)

    # GMREDI_SLOPE_LIMIT, kPos 1..3, every level
    def slopes(kPos, grid, params, gm, state, dSx, dSy, dSr, sX, sY, sSq, tF):
        res = {n: [] for n in ("SlopeX", "SlopeY", "SlopeSqr", "taperFct", "dSigmaDr")}
        for k in range(1, Nr + 1):
            two = lambda name, a: xy(name, a[:, k-1], sz)
            depthZ, rDepth = (grid.rF, grid.rF[1] - grid.rF[k]) if kPos == 3 else (grid.rC, grid.rF[1] - grid.rC[k])
            o = gmredi_slope_limit(two("SlopeX", sX), two("SlopeY", sY), two("SlopeSqr", sSq), two("taperFct", tF),
                                   xy("hTransLay", grid.R_low.data, sz), xy("baseSlope", jnp.zeros_like(grid.R_low.data), sz),
                                   xy("recipLambda", jnp.zeros_like(grid.R_low.data), sz), two("dSigmaDr", dSr),
                                   two("dSigmaDx", dSx), two("dSigmaDy", dSy), xy("Lrho", jnp.zeros_like(grid.R_low.data), sz),
                                   state.hMixLayer, rDepth, depthZ, grid.kLowC, kPos, k, 0.0, 0,
                                   cfg=cfg, params=params, gm=gm)
            for n, a in zip(("SlopeX", "SlopeY", "SlopeSqr", "taperFct"), o[:4]):
                res[n].append(a.data)
            res["dSigmaDr"].append(o[7].data)
        return {n: jnp.stack(v, axis=1) for n, v in res.items()}
    for kPos in (1, 2, 3):
        fs = jax.jit(lambda *a, kPos=kPos: slopes(kPos, *a))
        r2 = fs(grid, params, gm, state, *(jnp.asarray(f3[n]) for n in ("dSx", "dSy", "dSr", "SlopeXP", "SlopeYP",
                                                                           "SlopeSqrP", "taperFctP")))
        for n, a in r2.items():
            ours[f"slope{kPos}_{n}"] = np.asarray(a)

    # GMREDI_X/Y/RTRANSPORT (trIdentity 1) with the priors as tensor; TEMP_INTEGRATE bounds
    iMin, iMax, jMin, jMax = 0, sNx+1, 0, sNy+1
    gmT = with_priors(gm, inp, cfg, ("Kwx", "Kwy", "Kwz", "Kux", "Kvy"))

    def transports(grid, gm, Tr, xA, yA, mFk, mUp, dfP):
        T = xyz("Tracer", Tr, sz)
        res = {"dfX": [], "dfY": [], "dfR": []}
        for k in range(1, Nr + 1):
            two = lambda name, a: xy(name, a[:, k-1], sz)
            res["dfX"].append(gmredi_xtransport(1, k, iMin, iMax+1, jMin, jMax, two("xA", xA), two("maskFk", mFk), T,
                                                two("df", dfP), cfg=cfg, grid=grid, gm=gm).data)
            res["dfY"].append(gmredi_ytransport(1, k, iMin, iMax, jMin, jMax+1, two("yA", yA), two("maskFk", mFk), T,
                                                two("df", dfP), cfg=cfg, grid=grid, gm=gm).data)
            res["dfR"].append(gmredi_rtransport(1, k, iMin, iMax, jMin, jMax, two("maskUp", mUp), T,
                                                two("df", dfP), cfg=cfg, grid=grid, gm=gm).data)
        return {n: jnp.stack(v, axis=1) for n, v in res.items()}
    r3 = jax.jit(transports)(grid, gmT, *(jnp.asarray(f3[n]) for n in ("Tracer", "xA", "yA", "maskFk", "maskUp",
                                                                        "dfP")))
    ours.update({n: np.asarray(a) for n, a in r3.items()})

    # GMREDI_CALC_DIFF: kArg = 0 (trIdentity 1, 3), kArg = k (trIdentity 2)
    def diffs(grid, gm, kap):
        K = xyz("KappaRx", kap, sz)
        a = gmredi_calc_diff(iMin, iMax, jMin, jMax, 0, Nr, K, 1, cfg=cfg, grid=grid, gm=gm)
        b = gmredi_calc_diff(iMin, iMax, jMin, jMax, 0, Nr, K, 3, cfg=cfg, grid=grid, gm=gm)
        c = K
        for k in range(1, Nr + 1):
            c = gmredi_calc_diff(iMin, iMax, jMin, jMax, k, Nr, c, 2, cfg=cfg, grid=grid, gm=gm)
        return a.data, b.data, c.data
    r4 = jax.jit(diffs)(grid, gmT, jnp.asarray(f3["KappaRxP"]))
    for n, a in zip(("kappa_k0_tr1", "kappa_k0_tr3", "kappa_kArg_tr2"), r4):
        ours[n] = np.asarray(a)

    # GMREDI_RESIDUAL_FLOW
    u, v, w = gmredi_residual_flow(*(xyz(n, f3[n], sz) for n in ("uFld", "vFld", "wFld")), 0, cfg=cfg, gm=gm)
    ours.update(uRes=np.asarray(u.data), vRes=np.asarray(v.data), wRes=np.asarray(w.data))

    ref = {n: out[n] for n in ours}
    return ours, ref


def fixed_checks(exp):
    """Our GMREDI_READPARMS and GMREDI_INIT_FIXED vs the harness's replay_fixed.bin: {name: (ours, fortran)}."""
    e, _, _, _, fixed = replay_case(exp)
    cfg = e.cfg
    gm = gmredi_init_fixed(gmredi_readparms(e), cfg=cfg)
    out = {}
    for n in ("GM_isopycK", "GM_background_K", "GM_maxSlope", "GM_Kmin_horiz", "GM_Small_Number", "GM_slopeSqCutoff",
              "GM_Scrit", "GM_Sd", "GM_rMaxSlope", "GM_skewflx"):
        out[n] = (np.float64(getattr(gm, n)), np.float64(fixed[n]))
    out["GM_ExtraDiag"] = (float(gm.GM_ExtraDiag), fixed["GM_ExtraDiag"])
    for n in ("GM_isoFac2d", "GM_bolFac2d", "GM_isoFac1d", "GM_bolFac1d"):
        out[n] = (np.asarray(getattr(gm, n).data), fixed[n])
    return out


# ---------------------------------------------------------------------------------------------------------------
# gradients

FLOATS = ("GM_maxSlope", "GM_Scrit", "GM_Sd", "GM_isopycK", "GM_background_K", "GM_Kmin_horiz", "GM_Small_Number",
          "GM_skewflx")


def inputs_dump(exp, inp, it):
    """(x0, gm with the S00 priors, state) of a dumped iteration: x0 = {sigmaX, sigmaY, sigmaR: [tile,k,j,i],
    the REAL GMREDI.h parameters} (the differentiated inputs)."""
    e, gm, grid, params = setup(exp, inp)
    ds, _, _ = grid_gate.oracle(exp, inp)
    sz = e.cfg.size
    gm = gm.replace(**{n: wrap(n, jnp.asarray(ds.field(it, "S00_begin", n)), sz) for n in tensor_names(e.cfg)})
    state = StateH(hMixLayer=xy("hMixLayer", ds.field(it, "P03_mxlayer", "hMixLayer")[:, 0], sz))
    x0 = {n: jnp.asarray(ds.field(it, "P02_rho_sigma_ivdc", n)) for n in ("sigmaX", "sigmaY", "sigmaR")}
    x0.update({n: jnp.float64(getattr(gm, n)) for n in FLOATS})
    return x0, gm, state


def inputs_synthetic(exp, inp, seed=7):
    """Like inputs_dump, with the harness's synthetic stratification (unstable, neutral, below GM_Small_Number,
    steep and flat lanes) masked by the real grid's maskW/S/C, and synthetic priors."""
    e, gm, grid, params = setup(exp, inp)
    sz = e.cfg.size
    io = _replay_io()
    size = {k: getattr(sz, k) for k in io.SIZE_KEYS}
    f3, _, _, _ = io.make_inputs(size, seed)
    gm = gm.replace(**{n: wrap(n, jnp.asarray(f3[PRIOR[n]]), sz) for n in tensor_names(e.cfg)})
    state = StateH(hMixLayer=xy("hMixLayer", np.full((sz.nSx*sz.nSy, sz.sNy+2*sz.OLy, sz.sNx+2*sz.OLx), 50.0), sz))
    x0 = {n: jnp.asarray(f3[n] * np.asarray(getattr(grid, m).data)) for n, m in
          (("sigmaX", "maskW"), ("sigmaY", "maskS"), ("sigmaR", "maskC"))}
    x0.update({n: jnp.float64(getattr(gm, n)) for n in FLOATS})
    return x0, gm, state


def tensor_fn(exp, inp, gm, state):
    """x -> {tensor name: [tile,k,j,i]} through GMREDI_CALC_TENSOR (x as inputs_dump returns it)."""
    e, _, grid, params = setup(exp, inp)
    cfg, sz = e.cfg, e.cfg.size
    names = tensor_names(cfg)

    def f(x):
        g = gm.replace(**{n: x[n] for n in FLOATS})
        sig = [xyz(n, x[n], sz) for n in ("sigmaX", "sigmaY", "sigmaR")]
        out = gmredi_calc_tensor(1-sz.OLx, sz.sNx+sz.OLx, 1-sz.OLy, sz.sNy+sz.OLy, *sig, 0.0, 0,
                                 cfg=cfg, grid=grid, params=params, gm=g, state=state)
        return {n: getattr(out, n).data for n in names}
    return f


def cost_fn(f, weights):
    def J(x):
        out = f(x)
        return sum(jnp.sum(weights[n] * out[n]) for n in sorted(weights))
    return J


def random_like(tree, seed):
    leaves, tdef = jax.tree_util.tree_flatten(tree)
    rng = np.random.default_rng(seed)
    return jax.tree_util.tree_unflatten(tdef, [jnp.asarray(rng.standard_normal(np.shape(a))) for a in leaves])
