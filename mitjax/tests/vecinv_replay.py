"""Helpers of the VECINV-lane gates (mitjax/tests/test_vecinv.py): the gfortran replay runs of reference/replay_vecinv
(named by reference/replay_vecinv/CURRENT, relative to $MJX_REFERENCE) loaded as FArrays, and every replay output
recomputed by the ported kernels of mitjax/pkg/mom_vecinv and mitjax/pkg/mom_common.

The PARAMS.h / MOM_VISC.h values the kernels read come in one mitjax.model.src.ini_parms.Params (`params.NAME`):
LOGICAL, INTEGER and the host flags of REAL comparisons are static (pytree aux data), REAL values and arrays are leaves
(traced when the pytree is a jit argument). The values are the ones the harness wrote (replay_grid.bin: the run's own
PARAMS.h as MOM_VECINV case m0 reads them); the leaf cases replace them as the_main_loop.F does (replay_io.py VPAR,
SPAR, the *CASES tables).
"""

import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

from mitjax.farray import FArray
from mitjax.model.src.ini_parms import Params

REPO = Path(__file__).resolve().parents[2]
_RIO = REPO / "reference" / "replay_vecinv"


@jax.tree_util.register_pytree_node_class
class CommonBlock:
    """A common block by Fortran name (`visc.L2_D`): a pytree whose arrays are the leaves (here MOM_VISC.h)."""

    def __init__(self, **fields):
        object.__setattr__(self, "_f", dict(fields))

    def __getattr__(self, name):
        f = object.__getattribute__(self, "_f")
        if name in f:
            return f[name]
        raise AttributeError(name)

    def tree_flatten(self):
        keys = tuple(sorted(self._f))
        return tuple(self._f[k] for k in keys), keys

    @classmethod
    def tree_unflatten(cls, keys, leaves):
        return cls(**dict(zip(keys, leaves)))



def ctrlf_for(cfg, like):
    """CTRL_FIELDS.h for MOM_U/V_BOTDRAG_COEFF in builds with ALLOW_BOTTOMDRAG_CONTROL (90x40x15/code_ad): bottomDragFld
    = 0, as CTRL_INIT_VARIABLES leaves it without an xx_bottomdrag control (input_ad controls xx_theta only); None
    elsewhere. `like`: any xy FArray of the case (zeros with its bounds)."""
    import jax
    from mitjax.tests.goadk_gate import CtrlFields
    if not (cfg.cpp.ALLOW_CTRL and cfg.cpp.flag("ALLOW_BOTTOMDRAG_CONTROL", "CTRL_OPTIONS.h")):
        return None
    return CtrlFields(bottomDragFld=jax.tree.map(jnp.zeros_like, like))

def _load_rio():
    name = "_mjx_replay_vecinv_io"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _RIO / "replay_io.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rio = _load_rio()

KERNELS = {
    "mom_vecinv": ("mom_vecinv", "mom_vi_coriolis", "mom_vi_del2uv", "mom_vi_hdissip", "mom_vi_u_coriolis",
                   "mom_vi_v_coriolis", "mom_vi_u_vertshear", "mom_vi_v_vertshear", "mom_vi_u_grad_ke",
                   "mom_vi_v_grad_ke"),
    "mom_common": ("mom_calc_hdiv", "mom_calc_relvort3", "mom_calc_absvort3", "mom_calc_tension", "mom_calc_strain",
                   "mom_calc_visc", "mom_v_coriolis_nh", "mom_calc_hfacz"),
}


def kernel_modules():
    """{routine name: module} of the kernels the gates call."""
    return {r: importlib.import_module(f"mitjax.pkg.{pkg}.{r}") for pkg, rs in KERNELS.items() for r in rs}


@dataclass
class Replay:
    experiment: str
    input_dir: str
    rundir: Path
    exp: object                 # mitjax.config.params.Experiment
    size: dict
    fields: dict                # IN3 name -> [tile, k, j, i]
    spar: dict
    vpar: np.ndarray            # [case, VPAR]
    one_d: dict
    out: dict                   # OUT3 name -> [tile, k, j, i]
    g: dict                     # replay_grid.bin

    @property
    def cfg(self):
        return self.exp.cfg

    @property
    def Nr(self):
        return self.size["Nr"]

    @property
    def cube(self):
        return bool(self.g["useCubedSphereExchange"])


def current_runs():
    """[(experiment, input dir, run dir)] from reference/replay_vecinv/CURRENT."""
    from mitjax import paths
    out = []
    p = _RIO / "CURRENT"
    for line in p.read_text().splitlines():
        if line.strip():
            exp, inp, rel = line.split()
            out.append((exp, inp, paths.REFERENCE / rel))
    return out


_EXP = {}


def load_exp(experiment, input_dir):
    if (experiment, input_dir) not in _EXP:
        from mitjax.config import params as cp
        _EXP[experiment, input_dir] = cp.load(experiment, input_dir)
    return _EXP[experiment, input_dir]


def load(experiment, input_dir, rundir):
    size, fields, spar, vpar, one_d, _ = rio.read_inputs(rundir)
    out = rio.read_outputs(rundir, size)
    g = rio.read_grid(rundir, size)
    exp = load_exp(experiment, input_dir)
    s = exp.cfg.size
    assert (s.sNx, s.sNy, s.OLx, s.OLy, s.nSx, s.nSy, s.Nr) == tuple(size[k] for k in rio.SIZE_KEYS), (s, size)
    return Replay(experiment, input_dir, Path(rundir), exp, size, fields, spar, vpar, one_d, out, g)


def _b(R):
    sNx, sNy, OLx, OLy = (R.size[k] for k in ("sNx", "sNy", "OLx", "OLy"))
    return dict(i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy))


def w2_view(R):
    """The W2_EXCH2_TOPOLOGY.h values of each tile at myTile = W2_myTileList(bi,bj), [tile] arrays in tile storage
    order (t = (bj-1)*nSx + bi-1), from lane B's W2 set-up (mitjax/pkg/exch2/w2_eeboot.py; equal to the harness's
    replay_w2.bin, test_vecinv_params_from_the_run); None off the cube. A pytree (CommonBlock): sharded on the tile
    axis like any tiled field."""
    if not R.cube:
        return None
    from mitjax.pkg.exch2.w2_eeboot import w2_eeboot
    w2, _ = w2_eeboot(R.exp)
    nSx, nSy = R.size["nSx"], R.size["nSy"]
    tiles = [w2.W2_myTileList[bi, bj] for bj in range(1, nSy + 1) for bi in range(1, nSx + 1)]
    return CommonBlock(**{n: jnp.asarray(np.array([getattr(w2, n)[t] for t in tiles], np.int64))
                          for n in rio.W2_TOPO if n != "myTile"})


def run_params(R):
    """(Params, source) from the run's own files: ini_parms_dyn (its VECINV arm included) where it is ported, else
    (the cube runs: SET_REF_STATE under ALLOW_NONHYDROSTATIC and usingPCoords raise there, core lanes) the harness's
    record of the run's PARAMS.h with the VECINV arm's values (ini_parms_vecinv) put over it."""
    from mitjax.model.src.ini_parms import ini_parms, ini_parms_dyn, ini_parms_vecinv
    try:
        m = ini_parms(R.exp, _exch2_topology(R))
        # the kernels see useDiagnostics = .FALSE. (pkg/diagnostics not ported, brainstorm §3; Model's params carry
        # it from ini_parms_tracer); ini_parms_dyn alone keeps the run's switch (solid-body has it on)
        return (ini_parms_dyn(R.exp, m.grid, m.time, m.init).replace(static={"useDiagnostics": False}),
                "ini_parms_dyn")
    except NotImplementedError as e:
        _, params, _, _ = grid_params_visc(R)
        vs, vt = ini_parms_vecinv(R.exp)
        return params.replace(static=vs, traced=vt), f"harness record + ini_parms_vecinv ({e})"


def _exch2_topology(R):
    if not R.cfg.cpp.ALLOW_EXCH2:
        return None
    from mitjax.pkg.exch2.w2_eeboot import exch2_topology, w2_eeboot
    w2, _ = w2_eeboot(R.exp)
    return exch2_topology(w2)


def grid_params_visc(R):
    """(Grid, Params, visc, deepFacA): the grid, PARAMS.h and MOM_VISC.h as MOM_VECINV (case m0) reads them."""
    from mitjax.model.grid import Grid
    Nr = R.Nr
    b2 = _b(R)
    b3 = dict(b2, k=(1, Nr))
    g = R.g
    f = {}
    for n in rio.GRID3:
        f[n] = FArray(jnp.asarray(g[n]), n, **b3)
    for n in rio.GRID2:
        if n.startswith("L"):
            continue
        f[n] = FArray(jnp.asarray(g[n]), n, **b2)
    for n in rio.GRIDY:
        f[n] = FArray(jnp.asarray(g[n]), n, j=b2["j"])
    for n in ("drF", "recip_drF", "recip_deepFacC", "recip_deepFac2C", "deepFacC", "deepFac2C"):
        f[n] = FArray(jnp.asarray(g[n]), n, k=(1, Nr), tiled=False)
    for n in ("recip_drC", "deepFac2F"):
        f[n] = FArray(jnp.asarray(g[n]), n, k=(1, Nr+1), tiled=False)
    f["rkSign"] = jnp.float64(g["rkSign"])
    f["gravitySign"] = jnp.float64(g["gravitySign"])
    grid = Grid(f)
    visc = CommonBlock(**{n: FArray(jnp.asarray(g[n]), n, **b2)
                              for n in ("L2_D", "L2_Z", "L3_D", "L3_Z", "L4rdt_D", "L4rdt_Z")},
                           deepFacAdv=FArray(jnp.asarray(g["deepFacAdv"]), "deepFacAdv", k=(1, Nr), tiled=False))
    reals = [n for n in rio.SCALARS if n not in ("rkSign", "gravitySign", "pi", "diagFreq")]
    traced = {n: jnp.float64(g[n]) for n in reals}
    traced["recip_rhoFacC"] = FArray(jnp.asarray(g["recip_rhoFacC"]), "recip_rhoFacC", k=(1, Nr), tiled=False)
    traced["rhoFacF"] = FArray(jnp.asarray(g["rhoFacF"]), "rhoFacF", k=(1, Nr+1), tiled=False)
    traced["rVel2wUnit"] = FArray(jnp.asarray(g["rVel2wUnit"]), "rVel2wUnit", k=(1, Nr+1), tiled=False)
    traced["recip_rSphere"] = jnp.float64(g["recip_rSphere"])
    static = {n: g[n] for n in rio.FLAGS}
    static.update(_ne0_flags({n: g[n] for n in rio.SCALARS}))
    static.update(useDiagnostics=False, nonHydrostatic=False, useShelfIce=R.cfg.use_flag("useShelfIce"),
                  diagFreq=float(g["diagFreq"]), debugLevel=0, usingPCoords=not g["usingZCoords"],
                  useLANGMUIR=False)
    deepFacA = FArray(jnp.asarray(g["deepFacA"]), "deepFacA", k=(1, Nr), tiled=False)
    return grid, Params(static, traced), visc, deepFacA


def _ne0_flags(v):
    """The host flags of the REAL comparisons that select code (KERNEL_GUIDE REAL-IF rule)."""
    return {f"{n}_ne_0": bool(v[n] != 0.) for n in ("viscC2leith", "viscC2leithD", "viscC2LeithQG", "viscC4leith",
                                                  "viscC4leithD", "viscC2smag", "viscC4smag", "bottomDragLinear")}


def case_arrays(R):
    """The REAL parameters of the leaf cases as arrays, passed to the gates as jit arguments (traced, never closed
    over): vpar [case, VPAR] (VIRP_SETVISC) and spar [SPAR] (viscAhD/Z, viscA4D/Z after MOM_CALC_VISC)."""
    return {"vpar": jnp.asarray(R.vpar), "spar": jnp.asarray([R.spar[n] for n in rio.SPAR])}


def visc_case_params(params, R, case, cp):
    """Params of MOM_CALC_VISC case v<case> (1..4): VIRP_SETVISC, the_main_loop.F. The REAL values are the traced
    `cp["vpar"]` entries; the static flags of the REAL comparisons come from the host copy R.vpar."""
    host = dict(zip(rio.VPAR, (float(x) for x in R.vpar[case - 1])))
    ful, har, bih = rio.VCASES[case - 1]
    static = dict(useFullLeith=ful, useHarmonicVisc=har, useBiharmonicVisc=bih,
                  **_ne0_flags({**{n: R.g[n] for n in rio.SCALARS}, **host}))
    traced = {n: cp["vpar"][case - 1, c] for c, n in enumerate(rio.VPAR)}
    return params.replace(static=static, traced=traced)


def leaf_params(params, R, cp):
    """Params of the leaf cases after MOM_CALC_VISC (the_main_loop.F: viscAhD/Z, viscA4D/Z = spar; the last visc case
    and nonHydrostatic = F stay set)."""
    p = visc_case_params(params, R, len(rio.VCASES), cp)
    return p.replace(traced={n: cp["spar"][c] for c, n in enumerate(rio.SPAR)})


# ---------------------------------------------------------------------------------------------------------------
# one replay output per level k

def level_fn(name, R, mods, w2):
    """f(k, F, grid, params, visc, deepFacA, cp) -> the output's level-k data [tile, j, i]; cp: case_arrays."""
    m = mods
    Nr = R.Nr

    def hz(k, F, grid):
        P = F["prior"]
        return m["mom_calc_hfacz"].mom_calc_hfacz(k, P, P, cfg=R.cfg, grid=grid)

    def f(k, F, grid, params, visc, deepFacA, cp):
        cfg = R.cfg
        P = F["prior"]
        kg = dict(cfg=cfg, grid=grid)
        if name.endswith("_m0") or name.endswith("_m1"):
            mm = int(name[-1])
            p = params if mm == 0 else params.replace(static=rio.M1_OVERRIDES)
            from mitjax.model.state import State
            state = State({"uVel": F["uFld3"], "vVel": F["vFld3"], "wVel": F["wFld3"],
                           "gU": F["prior3"], "gV": F["prior3"]})
            fUp, fVp, guD, gvD, st = m["mom_vecinv"].mom_vecinv(
                k, 0, R.size["sNx"]+1, 0, R.size["sNy"]+1, F["kapU"], F["kapV"], F["fVerUkm"], F["fVerVkm"],
                P, P, P, P, 0., 0, cfg=cfg, grid=grid, params=p, state=state, visc=visc, w2=w2,
                ctrlf=ctrlf_for(cfg, P))
            base = name[:-3]
            return {"fVerUkp": fUp, "fVerVkp": fVp, "guDiss": guD, "gvDiss": gvD}[base].data if base not in (
                "gU", "gV") else getattr(st, base).data[:, k - 1]
        hFacZ, r_hFacZ = hz(k, F, grid)
        if name in ("hFacZ", "r_hFacZ"):
            return (hFacZ if name == "hFacZ" else r_hFacZ).data
        lp = leaf_params(params, R, cp)
        if name.startswith("hDiv_s"):
            return m["mom_calc_hdiv"].mom_calc_hdiv(k, int(name[-1]), F["uFld"], F["vFld"], P, **kg).data
        if name == "vort3":
            return m["mom_calc_relvort3"].mom_calc_relvort3(k, F["uFld"], F["vFld"], hFacZ, P, **kg, params=lp,
                                                            w2=w2).data
        if name.startswith("omega3_c"):
            c = int(name[-1])
            p = lp.replace(static=dict(momAdvection=c != 1, useCoriolis=c != 2))
            return m["mom_calc_absvort3"].mom_calc_absvort3(k, F["vort3"], P, **kg, params=p).data
        if name == "tension":
            return m["mom_calc_tension"].mom_calc_tension(k, F["uFld"], F["vFld"], P, **kg).data
        if name == "strain":
            return m["mom_calc_strain"].mom_calc_strain(k, F["uFld"], F["vFld"], hFacZ, P, **kg).data
        if "_v" in name and name.split("_v")[-1].isdigit():
            case = int(name.split("_v")[-1])
            p = visc_case_params(params, R, case, cp)
            outs = m["mom_calc_visc"].mom_calc_visc(k, P, P, P, P, F["hDiv"], F["vort3"], F["tension"], F["strain"],
                                                    F["zStar"], F["KE"], hFacZ, **kg, params=p, visc=visc, w2=w2)
            base = name.rsplit("_v", 1)[0]
            return outs[("viscAh_Z", "viscAh_D", "viscA4_Z", "viscA4_D", "hDiv").index(base)].data
        if name in ("del2u", "del2v", "hDiv_del2"):
            outs = m["mom_vi_del2uv"].mom_vi_del2uv(k, F["hDiv"], F["vort3"], hFacZ, P, P, **kg, params=lp, w2=w2)
            return outs[("del2u", "del2v", "hDiv_del2").index(name)].data
        if name[:9] in ("uDissip_h", "vDissip_h"):
            h, b, v = rio.HCASES[int(name[-1])]
            outs = m["mom_vi_hdissip"].mom_vi_hdissip(k, F["hDiv"], F["vort3"], F["dStar"], F["zStar"], hFacZ,
                                                      F["viscAh_Z"], F["viscAh_D"], F["viscA4_Z"], F["viscA4_D"],
                                                      h, b, v, P, P, **kg, params=lp)
            return outs[0 if name[0] == "u" else 1].data
        if name[:6] in ("uCf_cs", "vCf_cs"):
            p = lp.replace(static=dict(selectCoriScheme=int(name[-1])))
            outs = m["mom_vi_coriolis"].mom_vi_coriolis(k, F["uFld"], F["vFld"], hFacZ, r_hFacZ, P, P, **kg,
                                                        params=p)
            return outs[0 if name[0] == "u" else 1].data
        if name[:7] in ("uVort_c", "vVort_c"):
            s, jam = rio.CCASES[int(name[-1])]
            if name[0] == "u":
                return m["mom_vi_u_coriolis"].mom_vi_u_coriolis(k, s, jam, F["vFld"], F["omega3"], hFacZ, r_hFacZ, P,
                                                                **kg).data
            return m["mom_vi_v_coriolis"].mom_vi_v_coriolis(k, s, jam, F["uFld"], F["omega3"], hFacZ, r_hFacZ, P,
                                                            **kg).data
        if name[:8] in ("uShear_s", "vShear_s"):
            ke, up = rio.SCASES[int(name[-1])]
            p = lp.replace(static=dict(selectKEscheme=ke, upwindShear=up))
            if name[0] == "u":
                return m["mom_vi_u_vertshear"].mom_vi_u_vertshear(k, deepFacA, F["uFld3"], F["wFld3"], P, **kg,
                                                                  params=p).data
            return m["mom_vi_v_vertshear"].mom_vi_v_vertshear(k, deepFacA, F["vFld3"], F["wFld3"], P, **kg,
                                                              params=p).data
        if name == "dKEdx":
            return m["mom_vi_u_grad_ke"].mom_vi_u_grad_ke(k, F["KE"], P, **kg).data
        if name == "dKEdy":
            return m["mom_vi_v_grad_ke"].mom_vi_v_grad_ke(k, F["KE"], P, **kg).data
        if name.startswith("vCfNH_s"):
            p = lp.replace(static=dict(select3dCoriScheme=int(name[-1])))
            return m["mom_v_coriolis_nh"].mom_v_coriolis_nh(k, F["wFld3"], P, **kg, params=p).data
        raise KeyError(name)

    return f


LEVEL_INPUTS = ("uFld", "vFld", "hDiv", "vort3", "dStar", "zStar", "tension", "strain", "KE", "viscAh_Z", "viscAh_D",
                "viscA4_Z", "viscA4_D", "omega3", "fVerUkm", "fVerVkm", "prior")


def all_levels_fn(name, R, mods, w2, levels=None):
    """f(fields, grid, params, visc, deepFacA, cp) -> the output [tile, k, j, i] over all levels (Python loop over k,
    as the harness's DO k), or over `levels` only; fields are plain arrays, FArrays are built inside the program."""
    lf = level_fn(name, R, mods, w2)
    OLx, OLy, sNx, sNy, Nr = (R.size[k] for k in ("OLx", "OLy", "sNx", "sNy", "Nr"))
    b2 = dict(i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy))
    b3 = dict(b2, k=(1, Nr))
    bk1 = dict(b2, k=(1, Nr+1))

    def f(fields, grid, params, visc, deepFacA, cp):
        F3 = {"uFld3": FArray(fields["uFld"], "uFld", **b3), "vFld3": FArray(fields["vFld"], "vFld", **b3),
              "wFld3": FArray(fields["wFld"], "wFld", **b3), "prior3": FArray(fields["prior"], "prior", **b3),
              "kapU": FArray(jnp.concatenate([fields["kappaRU"], fields["kappaRU_Nr1"][:, :1]], axis=1), "kapU",
                             **bk1),
              "kapV": FArray(jnp.concatenate([fields["kappaRV"], fields["kappaRV_Nr1"][:, :1]], axis=1), "kapV",
                             **bk1)}
        outs = []
        for k in (range(1, Nr + 1) if levels is None else levels):
            F = dict(F3)
            for n in LEVEL_INPUTS:
                F[n] = FArray(fields[n][:, k - 1], n, **b2)
            outs.append(lf(k, F, grid, params, visc, deepFacA, cp))
        return jnp.stack(outs, axis=1)

    return f


COMPOSITION = ("fVerUkp", "fVerVkp", "guDiss", "gvDiss", "gU", "gV")


def composition_fn(R, mods, w2, m, levels=None):
    """f(fields, grid, params, visc, deepFacA, cp) -> {output name: [tile, k, j, i]} for the six MOM_VECINV outputs of
    case m<m> (one program for the six; the per-level calls in the harness's order)."""
    fns = {n: level_fn(f"{n}_m{m}", R, mods, w2) for n in COMPOSITION}
    OLx, OLy, sNx, sNy, Nr = (R.size[k] for k in ("OLx", "OLy", "sNx", "sNy", "Nr"))
    b2 = dict(i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy))
    b3 = dict(b2, k=(1, Nr))
    bk1 = dict(b2, k=(1, Nr+1))
    from mitjax.model.state import State

    def f(fields, grid, params, visc, deepFacA, cp):
        F3 = {"uFld3": FArray(fields["uFld"], "uFld", **b3), "vFld3": FArray(fields["vFld"], "vFld", **b3),
              "wFld3": FArray(fields["wFld"], "wFld", **b3), "prior3": FArray(fields["prior"], "prior", **b3),
              "kapU": FArray(jnp.concatenate([fields["kappaRU"], fields["kappaRU_Nr1"][:, :1]], axis=1), "kapU",
                             **bk1),
              "kapV": FArray(jnp.concatenate([fields["kappaRV"], fields["kappaRV_Nr1"][:, :1]], axis=1), "kapV",
                             **bk1)}
        p = params if m == 0 else params.replace(static=rio.M1_OVERRIDES)
        outs = {n: [] for n in COMPOSITION}
        for k in (range(1, Nr + 1) if levels is None else levels):
            P = FArray(fields["prior"][:, k - 1], "prior", **b2)
            fUm = FArray(fields["fVerUkm"][:, k - 1], "fVerUkm", **b2)
            fVm = FArray(fields["fVerVkm"][:, k - 1], "fVerVkm", **b2)
            state = State({"uVel": F3["uFld3"], "vVel": F3["vFld3"], "wVel": F3["wFld3"],
                           "gU": F3["prior3"], "gV": F3["prior3"]})
            fUp, fVp, guD, gvD, st = mods["mom_vecinv"].mom_vecinv(
                k, 0, sNx+1, 0, sNy+1, F3["kapU"], F3["kapV"], fUm, fVm, P, P, P, P, 0., 0,
                cfg=R.cfg, grid=grid, params=p, state=state, visc=visc, w2=w2, ctrlf=ctrlf_for(R.cfg, P))
            for n, a in zip(COMPOSITION, (fUp.data, fVp.data, guD.data, gvD.data, st.gU.data[:, k - 1],
                                          st.gV.data[:, k - 1])):
                outs[n].append(a)
        return {n: jnp.stack(a, axis=1) for n, a in outs.items()}

    return f


def field_arrays(R):
    return {n: jnp.asarray(a) for n, a in R.fields.items()}


def compare(ours, ref):
    """(points with ours != ref, non-finite points of ours, points whose bit patterns differ); numpy element-wise
    on every point incl. halos and the points a routine does not write."""
    a = np.ascontiguousarray(np.asarray(ours), np.float64)
    b = np.ascontiguousarray(ref, np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    return (int(np.count_nonzero(a != b)), int(np.count_nonzero(~np.isfinite(a))),
            int(np.count_nonzero(a.view(np.int64) != b.view(np.int64))))


def planted(mod, old, new, name="planted"):
    """A fresh copy of a kernel module with `old` replaced by `new` (exactly one occurrence): a negative control."""
    src = Path(mod.__file__).read_text()
    if src.count(old) != 1:
        raise ValueError(f"plant target occurs {src.count(old)} times in {mod.__file__}: {old!r}")
    mm = type(sys)(f"{mod.__name__}_{name}")
    mm.__file__ = mod.__file__
    exec(compile(src.replace(old, new), f"<{name}>", "exec"), mm.__dict__)
    return mm
