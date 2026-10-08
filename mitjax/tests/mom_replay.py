"""Helpers of the MOM-lane gates (mitjax/tests/test_mom_kernels.py): the gfortran replay runs of
reference/replay_mom (named by reference/replay_mom/CURRENT, relative to $MJX_REFERENCE) loaded as FArrays, and every
replay output recomputed by the ported kernels of mitjax/pkg/mom_common and mitjax/pkg/mom_fluxform.

The PARAMS.h values the kernels read come in one `Params` pytree (`params.NAME`): logical, integer and string values
are static (pytree aux data: they select branches at trace time), REAL values and arrays are leaves (traced when
the pytree is a jit argument). The Fortran case list (selectors per output) is replay_io.py's OUT3/WCASES/...
"""

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import jax
import jax.numpy as jnp

from mitjax.farray import FArray

REPO = Path(__file__).resolve().parents[2]
_RIO = REPO / "reference" / "replay_mom"
if str(_RIO) not in sys.path:
    sys.path.insert(0, str(_RIO))
import replay_io as rio  # noqa: E402

KERNELS = {
    "mom_common": ("mom_calc_ke", "mom_calc_hfacz", "mom_u_rviscflux", "mom_v_rviscflux", "mom_u_sidedrag",
                   "mom_v_sidedrag", "mom_u_botdrag_coeff", "mom_v_botdrag_coeff", "mom_u_coriolis_nh",
                   "mom_u_metric_nh", "mom_v_metric_nh", "mom_quasihydrostatic"),
    "mom_fluxform": ("mom_calc_rtrans", "mom_u_adv_uu", "mom_u_adv_vu", "mom_u_adv_wu", "mom_v_adv_uv",
                     "mom_v_adv_vv", "mom_v_adv_wv", "mom_u_coriolis", "mom_v_coriolis", "mom_u_metric_sphere",
                     "mom_v_metric_sphere", "mom_u_xviscflux", "mom_u_yviscflux", "mom_v_xviscflux",
                     "mom_v_yviscflux", "mom_u_del2u", "mom_v_del2v"),
}


def kernel_modules():
    """{routine name: module} of the ported kernels."""
    return {r: importlib.import_module(f"mitjax.pkg.{pkg}.{r}") for pkg, rs in KERNELS.items() for r in rs}


@jax.tree_util.register_pytree_node_class
class Params:
    """PARAMS.h by Fortran name: `static` values (logical/integer: branch selectors) are aux data, `traced` values
    (REAL scalars and FArrays) are the leaves."""

    def __init__(self, static, traced):
        object.__setattr__(self, "_s", dict(static))
        object.__setattr__(self, "_t", dict(traced))

    def __getattr__(self, name):
        s, t = object.__getattribute__(self, "_s"), object.__getattribute__(self, "_t")
        if name in t:
            return t[name]
        if name in s:
            v = s[name]
            if v is None:
                raise AttributeError(f"PARAMS.h {name} is not set for this case")
            return v
        raise AttributeError(f"PARAMS.h {name} is not provided")

    def replace(self, static=None, traced=None):
        return Params({**self._s, **(static or {})}, {**self._t, **(traced or {})})

    def tree_flatten(self):
        keys = tuple(sorted(self._t))
        return tuple(self._t[k] for k in keys), (keys, tuple(sorted(self._s.items())))

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        keys, static = aux
        return cls(dict(static), dict(zip(keys, leaves)))


@dataclass
class Replay:
    experiment: str
    input_dir: str
    rundir: Path
    cfg: object                 # mitjax.config.params.ExperimentConfig
    size: dict
    fields: dict                # IN3 name -> [tile, k, j, i]
    out: dict                   # OUT3 name -> [tile, k, j, i]
    g: dict                     # replay_grid.bin
    spar: dict

    @property
    def Nr(self):
        return self.size["Nr"]


def current_runs():
    """[(experiment, input dir, run dir)] from reference/replay_mom/CURRENT."""
    from mitjax import paths
    out = []
    for line in (_RIO / "CURRENT").read_text().splitlines():
        if line.strip():
            exp, inp, rel = line.split()
            out.append((exp, inp, paths.REFERENCE / rel))
    return out


_CFG = {}


def load_cfg(experiment, input_dir):
    if (experiment, input_dir) not in _CFG:
        from mitjax.config import params as cp
        _CFG[experiment, input_dir] = cp.load(experiment, input_dir).cfg
    return _CFG[experiment, input_dir]


def load(experiment, input_dir, rundir):
    size, fields, spar, _, _ = rio.read_inputs(rundir)
    out = rio.read_outputs(rundir, size)
    g = rio.read_grid(rundir, size)
    cfg = load_cfg(experiment, input_dir)
    s = cfg.size
    assert (s.sNx, s.sNy, s.OLx, s.OLy, s.nSx, s.nSy, s.Nr) == tuple(size[k] for k in rio.SIZE_KEYS), (s, size)
    return Replay(experiment, input_dir, Path(rundir), cfg, size, fields, out, g, spar)


def _b(R):
    sNx, sNy, OLx, OLy = (R.size[k] for k in ("sNx", "sNy", "OLx", "OLy"))
    return dict(i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy))


def grid_and_params(R):
    """(Grid, Params, deepFacA) as the harness's routines read them (numpy -> jnp)."""
    from mitjax.model.grid import Grid
    Nr = R.Nr
    b2 = _b(R)
    b3 = dict(b2, k=(1, Nr))
    by = dict(j=b2["j"])
    g = R.g
    f = {}
    for n in rio.GRID3:
        f[n] = FArray(jnp.asarray(g[n]), n, **b3)
    for n in rio.GRID2:
        f[n] = FArray(jnp.asarray(g[n]), n, **b2)
    for n in rio.GRIDY:
        f[n] = FArray(jnp.asarray(g[n]), n, **by)
    for n in ("drF", "recip_drF", "recip_deepFacC", "recip_deepFac2C"):
        f[n] = FArray(jnp.asarray(g[n]), n, k=(1, Nr), tiled=False)
    for n in ("recip_drC", "deepFac2F"):
        f[n] = FArray(jnp.asarray(g[n]), n, k=(1, Nr+1), tiled=False)
    f["rkSign"] = jnp.float64(g["rkSign"])
    f["gravitySign"] = jnp.float64(g["gravitySign"])
    grid = Grid(f)
    traced = {n: jnp.float64(g[n]) for n in ("recip_rSphere", "rhoConst", "mass2rUnit", "recip_gravity",
                                             "sideDragFactor", "viscAh", "viscAhGrid", "viscAhMax", "viscA4",
                                             "viscA4Grid", "viscA4Max", "viscA4GridMax", "viscA4GridMin",
                                             "deltaTMom", "bottomDragLinear", "bottomDragQuadratic")}
    traced["recip_gravFacC"] = FArray(jnp.asarray(g["recip_gravFacC"]), "recip_gravFacC", k=(1, Nr), tiled=False)
    traced["rhoFacF"] = FArray(jnp.asarray(g["rhoFacF"]), "rhoFacF", k=(1, Nr+1), tiled=False)
    traced["rVel2wUnit"] = FArray(jnp.asarray(g["rVel2wUnit"]), "rVel2wUnit", k=(1, Nr+1), tiled=False)
    static = {n: g[n] for n in ("useRealFreshWaterFlux", "usingPCoords", "usingZCoords", "fluidIsWater")}
    static.update(useDiagnostics=False, staggerTimeStep=False, rigidLid=None, select_rStar=None,
                  selectMetricTerms=None, selectCoriScheme=None, no_slip_sides=None, no_slip_bottom=None,
                  bottomVisc_pCell=None, selectBotDragQuadr=None, select3dCoriScheme=None, useNHMTerms=None)
    deepFacA = FArray(jnp.asarray(g["deepFacA"]), "deepFacA", k=(1, Nr), tiled=False)
    return grid, Params(static, traced), deepFacA


def field_arrays(R):
    """The IN3 inputs as jnp arrays [tile, k, j, i] (plain arrays: FArrays are built inside the traced program)."""
    return {n: jnp.asarray(a) for n, a in R.fields.items()}


# ---------------------------------------------------------------------------------------------------------------
# one replay output: case selectors and the kernel call, per level k

def case_of(name):
    """(routine group, static PARAMS.h selectors, extra) of a replay output (mirrors the_main_loop.F)."""
    if name in ("hFacZ", "r_hFacZ"):
        return "hfacz", {}, {}
    if name.startswith("KE_"):
        return "ke", {}, {"scheme": {"m1": -1, "0": 0, "1": 1, "2": 2, "3": 3}[name[3:]]}
    if name.startswith("rTrans"):
        return "rtrans", {"select_rStar": 0}, {"off": 1 if name.endswith("kp1") else 0}
    if name == "fZon_uu":
        return "uu", {}, {}
    if name.startswith("fMer_vu_mt"):
        return "vu", {"selectMetricTerms": int(name[-1])}, {}
    if name.startswith("fVerU_c") or name.startswith("fVerV_c"):
        off, rl, rs = rio.WCASES[int(name[-1])]
        return ("wu" if name[4] == "U" else "wv"), {"rigidLid": rl, "select_rStar": rs}, {"off": off}
    if name == "fZon_uv":
        return "uv", {}, {}
    if name == "fMer_vv":
        return "vv", {}, {}
    if name[:5] in ("uCf_s", "vCf_s"):
        return name[0] + "cor", {"selectCoriScheme": int(name[-1])}, {}
    if name[:7] in ("uMT_sph", "vMT_sph"):
        return name[0] + "msph", {"selectMetricTerms": int(name[-1])}, {}
    if name in ("xViscU", "yViscU", "xViscV", "yViscV"):
        return name, {}, {}
    if name.startswith("rVisc"):
        return "rvisc" + name[5], {}, {"off": 1 if name.endswith("kp1") else 0}
    if name.startswith("del2"):
        return name[:5], {"no_slip_sides": name.endswith("1")}, {}
    if name[:5] in ("uSD_c", "vSD_c"):
        return name[0] + "sd", {}, {"sd": rio.SDCASES[int(name[-1])]}
    if name.startswith("cDrag") or name.startswith("KE") and name.endswith("_b4"):
        uv = "u" if name in ("KEU_b4",) or name.startswith("cDragU") else "v"
        b = 4 if name.startswith("KE") else int(name[-1])
        sel, nsb, pce, inp = rio.BCASES[b]
        return uv + "bot", {"selectBotDragQuadr": sel, "no_slip_bottom": nsb, "bottomVisc_pCell": pce}, {
            "inp": inp, "ke": name.startswith("KE")}
    if name.startswith("uCfNH_s"):
        return "corNH", {"select3dCoriScheme": int(name[-1])}, {}
    if name in ("uMetNH", "vMetNH"):
        return name, {}, {}
    if name.startswith("qhyd_c"):
        s3, nhm = rio.QCASES[int(name[-1]) - 1]
        return "qhyd", {"select3dCoriScheme": s3, "useNHMTerms": nhm}, {}
    raise KeyError(name)


def level_fn(name, cfg, size, mods):
    """f(k, F, grid, params, deepFacA) -> the output's level-k data [tile, j, i]; F holds the level FArrays and the
    3-D inputs (see all_levels_fn)."""
    group, static, x = case_of(name)
    m = mods

    def f(k, F, grid, params, deepFacA):
        params = params.replace(static=static)
        P = F["prior"]
        kw = dict(cfg=cfg, grid=grid, params=params)
        kwg = dict(cfg=cfg, grid=grid)
        if group == "hfacz":
            h, r = m["mom_calc_hfacz"].mom_calc_hfacz(k, P, P, **kwg)
            return (h if name == "hFacZ" else r).data
        if group == "ke":
            return m["mom_calc_ke"].mom_calc_ke(k, x["scheme"], F["uFld"], F["vFld"], P, **kwg).data
        if group == "rtrans":
            u, v = m["mom_calc_rtrans"].mom_calc_rtrans(k + x["off"], P, P, 0., 0, state=SimpleNamespace(
                wVel=F["wFld3"]), **kw)
            return (u if name.startswith("rTransU") else v).data
        if group == "uu":
            return m["mom_u_adv_uu"].mom_u_adv_uu(k, F["uTrans"], F["uFld"], P, **kwg).data
        if group == "vu":
            return m["mom_u_adv_vu"].mom_u_adv_vu(k, F["vTrans"], F["uFld"], P, **kw).data
        if group == "wu":
            return m["mom_u_adv_wu"].mom_u_adv_wu(k + x["off"], deepFacA, F["uFld3"], F["wFld3"], F["rTrans"], P,
                                                  **kw).data
        if group == "uv":
            return m["mom_v_adv_uv"].mom_v_adv_uv(k, F["uTrans"], F["vFld"], P, **kwg).data
        if group == "vv":
            return m["mom_v_adv_vv"].mom_v_adv_vv(k, F["vTrans"], F["vFld"], P, **kwg).data
        if group == "wv":
            return m["mom_v_adv_wv"].mom_v_adv_wv(k + x["off"], deepFacA, F["vFld3"], F["wFld3"], F["rTrans"], P,
                                                  **kw).data
        if group == "ucor":
            return m["mom_u_coriolis"].mom_u_coriolis(k, F["vFld"], P, **kw).data
        if group == "vcor":
            return m["mom_v_coriolis"].mom_v_coriolis(k, F["uFld"], P, **kw).data
        if group == "umsph":
            return m["mom_u_metric_sphere"].mom_u_metric_sphere(k, F["uFld"], F["vFld"], P, **kw).data
        if group == "vmsph":
            return m["mom_v_metric_sphere"].mom_v_metric_sphere(k, F["uFld"], P, **kw).data
        if group == "xViscU":
            return m["mom_u_xviscflux"].mom_u_xviscflux(k, F["uFld"], F["del2u"], P, F["viscAh_D"], F["viscA4_D"],
                                                        **kwg).data
        if group == "yViscU":
            return m["mom_u_yviscflux"].mom_u_yviscflux(k, F["uFld"], F["del2u"], F["hFacZ"], P, F["viscAh_Z"],
                                                        F["viscA4_Z"], **kwg).data
        if group == "xViscV":
            return m["mom_v_xviscflux"].mom_v_xviscflux(k, F["vFld"], F["del2v"], F["hFacZ"], P, F["viscAh_Z"],
                                                        F["viscA4_Z"], **kwg).data
        if group == "yViscV":
            return m["mom_v_yviscflux"].mom_v_yviscflux(k, F["vFld"], F["del2v"], P, F["viscAh_D"], F["viscA4_D"],
                                                        **kwg).data
        if group == "rviscU":
            return m["mom_u_rviscflux"].mom_u_rviscflux(k + x["off"], F["uFld3"], F["kapU"], P, **kw).data
        if group == "rviscV":
            return m["mom_v_rviscflux"].mom_v_rviscflux(k + x["off"], F["vFld3"], F["kapV"], P, **kw).data
        if group == "del2u":
            return m["mom_u_del2u"].mom_u_del2u(k, F["uFld"], F["hFacZ"], F["h0FacZ"], P, **kw).data
        if group == "del2v":
            return m["mom_v_del2v"].mom_v_del2v(k, F["vFld"], F["hFacZ"], F["h0FacZ"], P, **kw).data
        if group in ("usd", "vsd"):
            sdf_spar, gmx_spar = x["sd"]
            p2 = params.replace(traced={
                "sideDragFactor": params.sideDragFactor if sdf_spar else jnp.float64(-1.0),
                "viscA4GridMax": params.viscA4GridMax if gmx_spar else jnp.float64(0.0)})
            kw2 = dict(cfg=cfg, grid=grid, params=p2)
            if group == "usd":
                return m["mom_u_sidedrag"].mom_u_sidedrag(k, F["uFld"], F["del2u"], F["hFacZ"], F["viscAh_Z"],
                                                          F["viscA4_Z"], True, True, False, P, **kw2).data
            return m["mom_v_sidedrag"].mom_v_sidedrag(k, F["vFld"], F["del2v"], F["hFacZ"], F["viscAh_Z"],
                                                      F["viscA4_Z"], True, True, False, P, **kw2).data
        if group in ("ubot", "vbot"):
            if group == "ubot":
                KE, cD = m["mom_u_botdrag_coeff"].mom_u_botdrag_coeff(k, x["inp"], F["uFld"], F["vFld"], F["kapU"],
                                                                      F["KEin"], P, 0, **kw)
            else:
                KE, cD = m["mom_v_botdrag_coeff"].mom_v_botdrag_coeff(k, x["inp"], F["uFld"], F["vFld"], F["kapV"],
                                                                      F["KEin"], P, 0, **kw)
            return (KE if x["ke"] else cD).data
        if group == "corNH":
            return m["mom_u_coriolis_nh"].mom_u_coriolis_nh(k, F["wFld3"], P, **kw).data
        if group == "uMetNH":
            return m["mom_u_metric_nh"].mom_u_metric_nh(k, F["uFld"], F["wFld3"], P, **kw).data
        if group == "vMetNH":
            return m["mom_v_metric_nh"].mom_v_metric_nh(k, F["vFld"], F["wFld3"], P, **kw).data
        if group == "qhyd":
            return m["mom_quasihydrostatic"].mom_quasihydrostatic(k, F["uFld3"], F["vFld3"], P, 0., 0, **kw).data
        raise KeyError(group)

    return f


LEVEL_INPUTS = ("uFld", "vFld", "uTrans", "vTrans", "rTrans", "hFacZ", "h0FacZ", "del2u", "del2v", "viscAh_D",
                "viscAh_Z", "viscA4_D", "viscA4_Z", "KEin", "prior")


def all_levels_fn(name, cfg, size, mods, levels=None):
    """f(fields, grid, params, deepFacA) -> the output [tile, k, j, i] over all levels (Python loop over k, as the
    harness's DO k), or over `levels` only; fields are plain arrays, FArrays are built inside the traced program."""
    lf = level_fn(name, cfg, size, mods)
    OLx, OLy, sNx, sNy, Nr = size["OLx"], size["OLy"], size["sNx"], size["sNy"], size["Nr"]
    b2 = dict(i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy))
    b3 = dict(b2, k=(1, Nr))
    bk1 = dict(b2, k=(1, Nr+1))

    def f(fields, grid, params, deepFacA):
        F3 = {"uFld3": FArray(fields["uFld"], "uFld", **b3), "vFld3": FArray(fields["vFld"], "vFld", **b3),
              "wFld3": FArray(fields["wFld"], "wFld", **b3),
              "kapU": FArray(jnp.concatenate([fields["kappaRU"], fields["kappaRU_Nr1"][:, :1]], axis=1), "kapU",
                             **bk1),
              "kapV": FArray(jnp.concatenate([fields["kappaRV"], fields["kappaRV_Nr1"][:, :1]], axis=1), "kapV",
                             **bk1)}
        outs = []
        for k in (range(1, Nr + 1) if levels is None else levels):
            F = dict(F3)
            for n in LEVEL_INPUTS:
                F[n] = FArray(fields[n][:, k - 1], n, **b2)
            outs.append(lf(k, F, grid, params, deepFacA))
        return jnp.stack(outs, axis=1)

    return f


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
    m = type(sys)(f"{mod.__name__}_{name}")
    m.__file__ = mod.__file__
    exec(compile(src.replace(old, new), f"<{name}>", "exec"), m.__dict__)
    return m
