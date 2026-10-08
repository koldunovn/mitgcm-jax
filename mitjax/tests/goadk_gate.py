"""Helpers of the GOADK gates (M2 sub-lane GOADK, plan Task 24: leaf kernels of global_ocean.90x40x15/input_ad*).

Oracle: lane A's registered dumps-on runs of the four code_ad variants (reference/reference_runs.py, kind "jdon",
build jaxdump_m2, job 27832229; dumps of iterations 0, 1, 2). The kernels run on the oracle's own inputs of the
dumped stage (teacher forcing) and the oracle's own grid of that iteration (stage G00_geometry: the grid routines of
the code_ad build are not part of this lane, and its exch1 2x2-tile layout has no registered exchange map yet), and
are compared with the dumped outputs on every point of every tile (halos and land included): element equality, both
sides finite, equal bit patterns.

  * GMREDI_CALC_TENSOR (with GMREDI_CALC_PSI_BOLUS, GMREDI_SLOPE_PSI, the K3D fields and GM_ExtraDiag) from
    P02_rho_sigma_ivdc (sigmaX/Y/R), P03_mxlayer (hMixLayer) and the S00_begin tensor (prior of the points it does
    not write) -> P05_gmredi_tensor (Kwx .. Kvz, GM_PsiX, GM_PsiY);
  * GMREDI_RESIDUAL_FLOW (bolus velocity) from S13_stagger_exchanges (uVel, vVel, wVel: THERMODYNAMICS copies them,
    thermodynamics.F:258-268) and P06_gmredi_exch (GM_PsiX/Y after GMREDI_DO_EXCH) -> T01_residual_flow.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.farray import FArray
from mitjax.model.grid import Grid, bounds
from mitjax.pkg.gmredi.gmredi_calc_tensor import gmredi_calc_tensor
from mitjax.pkg.gmredi.gmredi_h import wrap
from mitjax.pkg.gmredi.gmredi_init_fixed import gmredi_init_fixed
from mitjax.pkg.gmredi.gmredi_init_varia import gmredi_init_varia
from mitjax.pkg.gmredi.gmredi_readparms import gmredi_readparms
from mitjax.pkg.gmredi.gmredi_residual_flow import gmredi_residual_flow
from mitjax.tests import grid_gate
from mitjax.tests.gmredi_gate import ParamsH, StateH, compare_fields, failures, n_differing  # noqa: F401

EXP = "global_ocean.90x40x15"
VARIANTS = ("input_ad", "input_ad.kapgm", "input_ad.kapredi", "input_ad.bottomdrag")
TENSOR = ("Kwx", "Kwy", "Kwz", "Kux", "Kvy", "Kuz", "Kvz", "GM_PsiX", "GM_PsiY")

# GRID.h fields the kernels read: name -> declaration kind (mitjax/model/grid.py bounds)
GRID3 = ("maskC", "maskW", "maskS", "recip_hFacW", "recip_hFacS")
GRID2 = ("R_low", "dxG", "dyG", "recip_rA")
GRIDV = {"rF": "rp1", "rC": "r", "recip_drF": "r", "deepFacF": "rp1", "recip_deepFacF": "rp1",
         "recip_deepFacC": "r", "deepFac2F": "rp1", "recip_drC": "rp1"}


def with_data(fa, data):
    """The FArray `fa` (same name and declaration) holding `data`."""
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(fa), [jnp.asarray(data)])


def oracle(inp):
    return grid_gate.oracle(EXP, inp)[0]


def _vec(a, n):
    """A kind-V record (one record of n levels, written on the first tile) as a 1-D array."""
    a = np.asarray(a)
    v = a[0, :, 0, 0]
    if a.shape[1] != n or not np.array_equal(a[0], np.broadcast_to(v[:, None, None], a.shape[1:]), equal_nan=True):
        raise ValueError(f"kind-V record of shape {a.shape} is not one value per level")
    return v


def dump_grid(inp, it, ds=None):
    """GRID.h of dumped iteration `it` (stage G00_geometry) as a mitjax Grid (only the fields the gated kernels
    read; kLowC as int32; gravitySign a float64 scalar)."""
    ds = ds or oracle(inp)
    e = grid_gate.experiment(EXP, inp)
    sz = e.cfg.size
    f = {}
    for n in GRID3:
        f[n] = FArray(jnp.asarray(ds.field(it, "G00_geometry", n)), n, **bounds("xyz", sz))
    for n in GRID2:
        f[n] = FArray(jnp.asarray(ds.field(it, "G00_geometry", n)[:, 0]), n, **bounds("xy", sz))
    kl = ds.field(it, "G00_geometry", "kLowC")[:, 0]
    if not np.array_equal(kl, np.round(kl)):
        raise ValueError("kLowC dump is not integer-valued")
    f["kLowC"] = FArray(jnp.asarray(kl.astype(np.int32)), "kLowC", **bounds("xy", sz))
    for n, kind in GRIDV.items():
        b = dict(bounds(kind, sz))
        b.pop("tiled")
        nlev = b["k"][1]
        f[n] = FArray(jnp.asarray(_vec(ds.field(it, "G00_geometry", n), nlev)), n, **b, tiled=False)
    gs = np.asarray(ds.field(it, "G00_geometry", "gravitySign"))
    f["gravitySign"] = jnp.float64(gs.flat[0])
    return Grid(f)


def dump_params(inp, it, ds=None):
    """PARAMS.h values of GMREDI_SLOPE_LIMIT / _SLOPE_PSI: rVel2wUnit, wUnit2rVel (dumped, Nr+1), rUnit2z, z2rUnit
    (Nr; set_ref_state.F:56-59 sets 1. _d 0 in z-coordinates, gmredi_gate.set_ref_state_units)."""
    from mitjax.tests.gmredi_gate import set_ref_state_units
    ds = ds or oracle(inp)
    e = grid_gate.experiment(EXP, inp)
    sz = e.cfg.size
    Nr = sz.Nr
    u = set_ref_state_units(sz)

    def vec(name, data, n):
        return FArray(jnp.asarray(data, jnp.float64), name, k=(1, n), tiled=False)
    return ParamsH(rVel2wUnit=vec("rVel2wUnit", _vec(ds.field(it, "G00_geometry", "rVel2wUnit"), Nr + 1), Nr + 1),
                   wUnit2rVel=vec("wUnit2rVel", _vec(ds.field(it, "G00_geometry", "wUnit2rVel"), Nr + 1), Nr + 1),
                   rUnit2z=vec("rUnit2z", u["rUnit2z"], Nr), z2rUnit=vec("z2rUnit", u["z2rUnit"], Nr),
                   usingZCoords=bool(grid_gate.grid_params(EXP, inp).usingZCoords))


def setup_gm(inp):
    """GMREDI.h after our GMREDI_READPARMS, GMREDI_INIT_FIXED and GMREDI_INIT_VARIA of the variant (the K3D fields
    included; the controls of the jdon runs are zero)."""
    e = grid_gate.experiment(EXP, inp)
    cfg = e.cfg
    return gmredi_init_varia(gmredi_init_fixed(gmredi_readparms(e), cfg=cfg), cfg=cfg)


def xyz(name, data, size):
    return FArray(jnp.asarray(data, jnp.float64), name, **bounds("xyz", size))


def xy(name, data, size):
    return FArray(jnp.asarray(data, jnp.float64), name, **bounds("xy", size))


@functools.lru_cache(maxsize=None)
def tensor_jit(cfg, iMin, iMax, jMin, jMax):
    """GMREDI_CALC_TENSOR under jit; grid, params, gm (REAL parameters traced), state and sigma are arguments. Cached
    per configuration: one compilation per static structure (a planted change of a static GMREDI.h value retraces)."""
    def f(grid, params, gm, state, sigmaX, sigmaY, sigmaR, myTime):
        return gmredi_calc_tensor(iMin, iMax, jMin, jMax, sigmaX, sigmaY, sigmaR, myTime, 0,
                                  cfg=cfg, grid=grid, params=params, gm=gm, state=state)
    return jax.jit(f)


def tensor_case(inp, it, gm_override=None, grid_override=None, cfg_override=None, sigma_override=None, ds=None):
    """GMREDI_CALC_TENSOR on dumped iteration `it`: ({name: ours}, {name: oracle}) as [tile,k,j,i] numpy arrays.
    The *_override callables plant the error of a negative control; `ds`: another dump set of the variant (default
    the jdon run)."""
    ds = ds or oracle(inp)
    e = grid_gate.experiment(EXP, inp)
    cfg, sz = e.cfg, e.cfg.size
    if cfg_override is not None:
        cfg = cfg_override(cfg)
    grid, params = dump_grid(inp, it, ds), dump_params(inp, it, ds)
    if grid_override is not None:
        grid = grid_override(grid)
    gm = setup_gm(inp)
    gm = gm.replace(**{n: wrap(n, jnp.asarray(ds.field(it, "S00_begin", n)), sz) for n in TENSOR})
    if gm_override is not None:
        gm = gm_override(gm)
    state = StateH(hMixLayer=xy("hMixLayer", ds.field(it, "P03_mxlayer", "hMixLayer")[:, 0], sz))
    sig = {n: ds.field(it, "P02_rho_sigma_ivdc", n) for n in ("sigmaX", "sigmaY", "sigmaR")}
    if sigma_override is not None:
        sig = sigma_override(sig)
    # do_oceanic_phys.F:560-563 (iMin = 1-OLx, iMax = sNx+OLx, jMin = 1-OLy, jMax = sNy+OLy)
    f = tensor_jit(cfg, 1-sz.OLx, sz.sNx+sz.OLx, 1-sz.OLy, sz.sNy+sz.OLy)
    out = f(grid, params, gm, state, *(xyz(n, sig[n], sz) for n in ("sigmaX", "sigmaY", "sigmaR")),
            jnp.float64(0.0))
    ours = {n: np.asarray(getattr(out, n).data) for n in TENSOR}
    ref = {n: ds.field(it, "P05_gmredi_tensor", n) for n in TENSOR}
    return ours, ref


def residual_case(inp, it, grid_it=None, gm_override=None, grid_override=None):
    """GMREDI_RESIDUAL_FLOW on dumped iteration `it`: inputs uVel/vVel/wVel at S13_stagger_exchanges, GM_PsiX/Y at
    P06_gmredi_exch, grid of dumped iteration `grid_it` (default `it`); outputs vs T01_residual_flow."""
    ds = oracle(inp)
    e = grid_gate.experiment(EXP, inp)
    cfg, sz = e.cfg, e.cfg.size
    grid = dump_grid(inp, it if grid_it is None else grid_it, ds)
    if grid_override is not None:
        grid = grid_override(grid)
    gm = setup_gm(inp)
    gm = gm.replace(**{n: wrap(n, jnp.asarray(ds.field(it, "P06_gmredi_exch", n)), sz)
                       for n in ("GM_PsiX", "GM_PsiY")})
    if gm_override is not None:
        gm = gm_override(gm)

    def f(grid, gm, u, v, w):
        return gmredi_residual_flow(u, v, w, 0, cfg=cfg, gm=gm, grid=grid)
    u, v, w = jax.jit(f)(grid, gm, *(xyz(n, ds.field(it, "S13_stagger_exchanges", n), sz)
                                     for n in ("uVel", "vVel", "wVel")))
    ours = {"uFld": np.asarray(u.data), "vFld": np.asarray(v.data), "wFld": np.asarray(w.data)}
    ref = {n: ds.field(it, "T01_residual_flow", n) for n in ours}
    return ours, ref


@dataclasses.dataclass
class Counts:
    """Differing points of a planted error, per name."""
    by_name: dict

    @property
    def total(self):
        return sum(self.by_name.values())


# ---------------------------------------------------------------------------------------------------------------
# the GOADK replay harness (reference/replay_goadk: global_ocean.90x40x15/code_ad, synthetic inputs over the real grid)

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BotDragParams:
    """The PARAMS.h values MOM_U/V_BOTDRAG_COEFF reads: REAL traced, the rest static. input_ad: no_slip_bottom
    (set_defaults.F:133 `.TRUE.`), bottomVisc_pCell (:134 `.FALSE.`), selectBotDragQuadr (:139 -1, then
    ini_parms.F:550-551 0 since bottomDragQuadratic = 0.0021 in data), usingZCoords."""
    bottomDragLinear: object
    bottomDragQuadratic: object
    no_slip_bottom: bool = dataclasses.field(metadata=dict(static=True))
    bottomVisc_pCell: bool = dataclasses.field(metadata=dict(static=True))
    selectBotDragQuadr: int = dataclasses.field(metadata=dict(static=True))
    usingZCoords: bool = dataclasses.field(metadata=dict(static=True))


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class CostParams:
    HeatCapacity_Cp: object
    rhoConst: object


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class CostMean:
    cMeanVVel: object
    cMeanThetaVVel: object


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class CtrlFields:
    bottomDragFld: object


def _io():
    import importlib.util
    from mitjax import paths
    spec = importlib.util.spec_from_file_location("_mjx_replay_goadk_io",
                                                  paths.REPO / "reference" / "replay_goadk" / "replay_io.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def replay_rundir():
    """The harness run directory named by reference/replay_goadk/CURRENT (relative to $MJX_REFERENCE)."""
    from mitjax import paths
    p = paths.REPO / "reference" / "replay_goadk" / "CURRENT"
    if not p.exists():
        raise FileNotFoundError(f"{p}: no GOADK replay run registered")
    return paths.REFERENCE / p.read_text().strip()


def replay_run(patch=None):
    """Our port on the GOADK harness inputs: ({name: ours}, {name: oracle}) for every output (OUT3 of the GM/Redi
    part, OUT2 and objf_atl). `patch(objs) -> objs` plants the error of a negative control."""
    from mitjax.pkg.cost.cost_atlantic_heat import cost_atlantic_heat
    from mitjax.pkg.generic_advdiff.gad_calc_rhs import gad_kernel_cfg
    from mitjax.pkg.generic_advdiff.gad_dst3_adv_x import gad_dst3_adv_x
    from mitjax.pkg.generic_advdiff.gad_dst3_adv_y import gad_dst3_adv_y
    from mitjax.pkg.gmredi.gmredi_calc_diff import gmredi_calc_diff
    from mitjax.pkg.gmredi.gmredi_rtransport import gmredi_rtransport
    from mitjax.pkg.gmredi.gmredi_slope_limit import gmredi_slope_limit
    from mitjax.pkg.gmredi.gmredi_xtransport import gmredi_xtransport
    from mitjax.pkg.gmredi.gmredi_ytransport import gmredi_ytransport
    from mitjax.pkg.mom_common.mom_u_botdrag_coeff import mom_u_botdrag_coeff
    from mitjax.pkg.mom_common.mom_v_botdrag_coeff import mom_v_botdrag_coeff
    from mitjax.tests import gmredi_gate as GG
    io = _io()
    inp = "input_ad"
    e = grid_gate.experiment(EXP, inp)
    cfg = e.cfg
    sz = cfg.size
    sNx, sNy, OLx, OLy, Nr = sz.sNx, sz.sNy, sz.OLx, sz.OLy, sz.Nr
    rd = replay_rundir()
    size, f3, f2, par, one_d = io.read_inputs(rd)
    out, out2 = io.read_outputs(rd, size), io.read_outputs2(rd, size)
    g, g2 = io.read_grid(rd, size), io.read_grid2(rd, size)
    grid = GG.replay_grid(g, sz)
    gf = {n: GG.fa(n, g2[n], "xyz", sz) for n in io.GRID2_3}
    gf.update({n: GG.fa(n, g2[n], "xy", sz) for n in io.GRID2_2})
    gf.update({n: GG.fa(n, g2[n], "r", sz) for n in ("recip_drF", "drF")})
    gf.update({n: GG.fa(n, g2[n], "rp1", sz) for n in ("deepFacF", "recip_deepFacF")})
    grid = grid.replace(**gf)
    # GMREDI.h as the harness leaves it after step 4b
    gm = setup_gm(inp).replace(
        GM_isopycK=np.float64(par["GM_isopycK"]), GM_skewflx=np.float64(par["GM_skewflx"]),
        GM_isoFac1d=GG.fa("GM_isoFac1d", one_d["GM_isoFac1d"], "r", sz),
        GM_bolFac1d=GG.fa("GM_bolFac1d", one_d["GM_bolFac1d"], "r", sz),
        GM_isoFac2d=GG.fa("GM_isoFac2d", f2["GM_isoFac2d"], "xy", sz),
        GM_bolFac2d=GG.fa("GM_bolFac2d", f2["GM_bolFac2d"], "xy", sz),
        GM_inpK3dRedi=wrap("GM_inpK3dRedi", jnp.asarray(f3["K3dRediP"]), sz),
        GM_inpK3dGM=wrap("GM_inpK3dGM", jnp.asarray(f3["K3dGMP"]), sz))
    rp = GG.replay_params(g, sz)
    objs = dict(grid=grid, params=rp, cfg=cfg, gm=gm, state=StateH(hMixLayer=GG.fa("hMixLayer", f2["hMixLayer"],
                                                                                   "xy", sz)))
    if patch is not None:
        objs = patch(objs)
    grid, rp, gm, state, cfg = objs["grid"], objs["params"], objs["gm"], objs["state"], objs["cfg"]
    ours = {}
    TN = TENSOR

    # 3. GMREDI_INIT_VARIA on prior-filled tensors
    gm0 = gm.replace(**{n: wrap(n, jnp.asarray(f3[GG.PRIOR[n]]), sz) for n in TN})
    r0 = jax.jit(lambda gm_: gmredi_init_varia(gm_, cfg=cfg))(gm0)
    for n in TN:
        ours[f"initvaria_{n}"] = np.asarray(getattr(r0, n).data)
    # 5a. GMREDI_CALC_TENSOR
    sig = [xyz(n, f3[n] * np.asarray(g[m]), sz) for n, m in (("sigmaX", "maskW"), ("sigmaY", "maskS"),
                                                             ("sigmaR", "maskC"))]
    r1 = tensor_jit(cfg, 1-OLx, sNx+OLx, 1-OLy, sNy+OLy)(grid, rp, gm0, state, *sig, jnp.float64(0.0))
    for n in TN:
        ours[f"tensor_{n}"] = np.asarray(getattr(r1, n).data)

    # 5b. GMREDI_SLOPE_LIMIT, kPos 1..3
    def slopes(kPos, grid, params, gm, state, dSx, dSy, dSr, sX, sY, sSq, tF):
        res = {n: [] for n in ("SlopeX", "SlopeY", "SlopeSqr", "taperFct", "dSigmaDr")}
        z = jnp.zeros_like(grid.R_low.data)
        for k in range(1, Nr + 1):
            def two(name, a):
                return xy(name, a[:, k-1], sz)
            depthZ, rDepth = (grid.rF, grid.rF[1] - grid.rF[k]) if kPos == 3 else (grid.rC, grid.rF[1] - grid.rC[k])
            o = gmredi_slope_limit(two("SlopeX", sX), two("SlopeY", sY), two("SlopeSqr", sSq), two("taperFct", tF),
                                   xy("hTransLay", grid.R_low.data, sz), xy("baseSlope", z, sz),
                                   xy("recipLambda", z, sz), two("dSigmaDr", dSr), two("dSigmaDx", dSx),
                                   two("dSigmaDy", dSy), xy("Lrho", z, sz), state.hMixLayer, rDepth, depthZ,
                                   grid.kLowC, kPos, k, 0.0, 0, cfg=cfg, params=params, gm=gm)
            for n, a in zip(("SlopeX", "SlopeY", "SlopeSqr", "taperFct"), o[:4]):
                res[n].append(a.data)
            res["dSigmaDr"].append(o[7].data)
        return {n: jnp.stack(v, axis=1) for n, v in res.items()}
    for kPos in (1, 2, 3):
        r2 = jax.jit(lambda *a, kPos=kPos: slopes(kPos, *a))(
            grid, rp, gm, state, *(jnp.asarray(f3[n]) for n in ("dSx", "dSy", "dSr", "SlopeXP", "SlopeYP",
                                                                 "SlopeSqrP", "taperFctP")))
        for n, a in r2.items():
            ours[f"slope{kPos}_{n}"] = np.asarray(a)

    # 5c. GMREDI_X/Y/RTRANSPORT: Kwx..Kvy = the priors, Kuz/Kvz as GMREDI_CALC_TENSOR left them (the harness does not
    # reset them)
    iMin, iMax, jMin, jMax = 0, sNx+1, 0, sNy+1
    gmT = gm.replace(**{n: wrap(n, jnp.asarray(f3[GG.PRIOR[n]]), sz) for n in ("Kwx", "Kwy", "Kwz", "Kux", "Kvy")},
                     Kuz=r1.Kuz, Kvz=r1.Kvz)

    def transports(grid, gm, Tr, xA, yA, mFk, mUp, dfP):
        T = xyz("Tracer", Tr, sz)
        res = {"dfX": [], "dfY": [], "dfR": []}
        for k in range(1, Nr + 1):
            def two(name, a):
                return xy(name, a[:, k-1], sz)
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

    # 5d. GMREDI_CALC_DIFF
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

    # 5e. GMREDI_RESIDUAL_FLOW: the bolus velocity from GM_PsiX/Y as GMREDI_CALC_TENSOR left them
    gmR = gm.replace(GM_PsiX=r1.GM_PsiX, GM_PsiY=r1.GM_PsiY)
    u, v, w = jax.jit(lambda grid, gm, a, b, c: gmredi_residual_flow(a, b, c, 0, cfg=cfg, gm=gm, grid=grid))(
        grid, gmR, *(xyz(n, f3[n], sz) for n in ("uFld", "vFld", "wFld")))
    ours.update(uRes=np.asarray(u.data), vRes=np.asarray(v.data), wRes=np.asarray(w.data))

    # 5f. GAD_DST3_ADV_X / _Y
    kc = gad_kernel_cfg(cfg)

    def dst3(grid, deltaTloc, Tr, uTr, vTr, uF, vF, uC, vC, prior):
        res = {n: [] for n in ("dst3x_cflT", "dst3x_cflF", "dst3y_cflT", "dst3y_cflF")}
        for k in range(1, Nr + 1):
            def two(name, a):
                return xy(name, a[:, k-1], sz)
            mW, mS = two("maskLocW", grid.maskW.data), two("maskLocS", grid.maskS.data)
            T = two("tracer", Tr)
            res["dst3x_cflT"].append(gad_dst3_adv_x(k, True, deltaTloc, two("uTrans", uTr), two("uFld", uF), mW, T,
                                                    two("uT", prior), cfg=kc, grid=grid).data)
            res["dst3x_cflF"].append(gad_dst3_adv_x(k, False, deltaTloc, two("uTrans", uTr), two("uFld", uC), mW, T,
                                                    two("uT", prior), cfg=kc, grid=grid).data)
            res["dst3y_cflT"].append(gad_dst3_adv_y(k, True, deltaTloc, two("vTrans", vTr), two("vFld", vF), mS, T,
                                                    two("vT", prior), cfg=kc, grid=grid).data)
            res["dst3y_cflF"].append(gad_dst3_adv_y(k, False, deltaTloc, two("vTrans", vTr), two("vFld", vC), mS, T,
                                                    two("vT", prior), cfg=kc, grid=grid).data)
        return {n: jnp.stack(v, axis=1) for n, v in res.items()}
    r5 = jax.jit(dst3)(grid, jnp.float64(par["deltaTloc"]),
                       *(jnp.asarray(f3[n]) for n in ("Tracer", "uTr", "vTr", "uFld", "vFld", "uCfl", "vCfl", "dfP")))
    ours.update({n: np.asarray(a) for n, a in r5.items()})

    # 5g. MOM_U/V_BOTDRAG_COEFF with bottomDragFld
    bp = BotDragParams(bottomDragLinear=jnp.float64(g2["bottomDragLinear"]),
                       bottomDragQuadratic=jnp.float64(g2["bottomDragQuadratic"]), no_slip_bottom=True,
                       bottomVisc_pCell=False, selectBotDragQuadr=0, usingZCoords=True)
    ctrlf = CtrlFields(bottomDragFld=GG.fa("bottomDragFld", f2["bDragFld"], "xy", sz))
    if patch is not None and "ctrlf" in objs:
        ctrlf = objs["ctrlf"]

    def botdrag(grid, params, ctrlf, uF, vF, kap, KEp, prior):
        kapR = FArray(jnp.concatenate([jnp.asarray(kap), jnp.zeros_like(jnp.asarray(kap)[:, :1])], axis=1),
                      "kappaRU", **dict(bounds("xy", sz), k=(1, Nr+1)))
        res = {n: [] for n in io.OUT2[4:]}
        for k in range(1, Nr + 1):
            def two(name, a):
                return xy(name, a[:, k-1], sz)
            for c, fn in (("U", mom_u_botdrag_coeff), ("V", mom_v_botdrag_coeff)):
                _, cT = fn(k, True, two("uFld", uF), two("vFld", vF), kapR, two("KE", KEp), two("cDrag", prior), 0,
                           cfg=cfg, grid=grid, params=params, ctrlf=ctrlf)
                KEf, cF = fn(k, False, two("uFld", uF), two("vFld", vF), kapR, two("KE", KEp), two("cDrag", prior),
                             0, cfg=cfg, grid=grid, params=params, ctrlf=ctrlf)
                res[f"bot{c}_cDrag_keT"].append(cT.data)
                res[f"bot{c}_cDrag_keF"].append(cF.data)
                res[f"bot{c}_KE_keF"].append(KEf.data)
        return {n: jnp.stack(v, axis=1) for n, v in res.items()}
    r6 = jax.jit(botdrag)(grid, bp, ctrlf, *(jnp.asarray(f3[n]) for n in ("uFld", "vFld", "KappaRxP", "KEP", "dfP")))
    ours.update({n: np.asarray(a) for n, a in r6.items()})

    # 5h. COST_ATLANTIC_HEAT
    cp = CostParams(HeatCapacity_Cp=jnp.float64(g2["HeatCapacity_Cp"]), rhoConst=jnp.float64(g2["rhoConst"]))
    cm = CostMean(cMeanVVel=xyz("cMeanVVel", f3["vFld"], sz), cMeanThetaVVel=xyz("cMeanThetaVVel", f3["vTr"], sz))
    atl = jax.jit(lambda grid, cp, cm: cost_atlantic_heat(cfg=cfg, grid=grid, params=cp, cost=cm,
                                                          myXGlobalLo=int(g2["myXGlobalLo"]),
                                                          myYGlobalLo=int(g2["myYGlobalLo"])))(grid, cp, cm)
    ours["objf_atl"] = np.asarray(atl)

    ref = {n: (out[n] if n in out else out2[n]) for n in ours}
    return ours, ref


# ---------------------------------------------------------------------------------------------------------------
# grdchk positions (GRDCHK_GET_POSITION with nbeg = 0) of the four variants

_GRDCHK_DEFAULTS = (("nbeg", 82), ("nend", 83), ("nstep", 84), ("iGloPos", 87), ("jGloPos", 88), ("kGloPos", 89),
                    ("iGloTile", 90), ("jGloTile", 91), ("obcsglo", 94), ("recglo", 95))


def grdchk_settings(inp):
    """data.grdchk of the variant with the defaults of pkg/grdchk/grdchk_readparms.F:82-95 (fortran_default)."""
    from mitjax.params_io import RunParams, fortran_default
    e = grid_gate.experiment(EXP, inp)
    rp = RunParams(e.run)
    return {n: int(rp.get("data.grdchk", "GRDCHK_NML", n,
                          default=fortran_default(f"pkg/grdchk/grdchk_readparms.F:{line}", n, e)))
            for n, line in _GRDCHK_DEFAULTS}


def grdchk_case(inp, maskC=None):
    """Our grdchk positions of the variant: (points [(icomp, LocResult)], GrdchkMask, settings). The control is
    genarr 2-D (ncvarnrmax 1) or 3-D (Nr) as CTRL_INIT_FIXED registers it (ctrl_init_fixed.F:231-261: ncvarrecs 1,
    ncvarxmax sNx, ncvarymax sNy, ncvargrd 'c'); iLocTile = iGloTile - (myXGlobalLo-1)/sNx with myXGlobalLo = 1
    (grdchk_readparms.F:140-141, single process). maskC: the dumped G00 mask (iteration 0) unless given."""
    from mitjax.pkg.grdchk import grdchk as gc
    e = grid_gate.experiment(EXP, inp)
    sz = e.cfg.size
    s = grdchk_settings(inp)
    mC = np.asarray(oracle(inp).field(0, "G00_geometry", "maskC")) if maskC is None else maskC
    nr = sz.Nr if CONTROL[inp][0] == 3 else 1              # genarr3d: Nr levels; genarr2d (xx_bottomdrag): 1
    nw = gc.ctrl_init_wet_nwetctile(mC, sz)
    gm = gc.grdchk_get_mask(nw, ncvargrd="c", ncvarnrmax=nr, ncvarxmax=sz.sNx, ncvarymax=sz.sNy, ncvarrecs=1)
    kw = dict(gm=gm, maskC=mC, sz=sz, iLocTile=s["iGloTile"], jLocTile=s["jGloTile"], ncvargrd="c", ncvarrecs=1,
              ncvarnrmax=nr, ncvarxmax=sz.sNx, ncvarymax=sz.sNy)
    pos = {n: s[n] for n in ("iGloPos", "jGloPos", "kGloPos", "obcsglo", "recglo")}
    pts = gc.grdchk_points(nbeg=s["nbeg"], nstep=s["nstep"], nend=s["nend"], position=pos, **kw)
    return pts, gm, s


def fd_stdout(inp):
    """STDOUT lines of lane A's zero-adxx FD oracle run of the variant (reference_runs M2_FD, kind fdzero)."""
    from mitjax.io import stdout as so
    reg = grid_gate._registry()
    runs = [r for r in reg.RUNS if (r["exp"], r["input"], r["kind"]) == (EXP, inp, "fdzero")]
    if len(runs) != 1:
        raise FileNotFoundError(f"{EXP}/{inp}: {len(runs)} registered fdzero runs")
    return so.read_stdout(reg.run_top(runs[0]) / "rundir" / "output.txt")


# ---------------------------------------------------------------------------------------------------------------
# controls: CTRL_MAP_INI_GENARR (xx_theta, xx_kapgm, xx_kapredi via genarr3d; xx_bottomdrag via genarr2d)

CONTROL = {"input_ad": (3, "xx_theta", "theta"), "input_ad.kapgm": (3, "xx_kapgm", "GM_inpK3dGM"),
           "input_ad.kapredi": (3, "xx_kapredi", "GM_inpK3dRedi"),
           "input_ad.bottomdrag": (2, "xx_bottomdrag", "bottomDragFld")}


@functools.lru_cache(maxsize=None)
def exchanger_code_ad(inp="input_ad"):
    """The exchanger of the code_ad layout (exch1, 2x2 tiles of 45x20, OLx = OLy = 3), built from the oracle's own
    exchange probe (stage X00_exch_probe of the dumps-on run, mitjax.eesupp.exch_maps.build_maps) and checked to
    reproduce every probe output bitwise (scripts/make_exch_maps.py's check). No map file of this layout is
    registered in mitjax/eesupp/exch_maps.MAP_SHA256."""
    import importlib.util
    from mitjax import paths
    from mitjax.eesupp import exch_maps as EM
    from mitjax.eesupp.exchange import Exchanger
    spec = importlib.util.spec_from_file_location("_mjx_make_exch_maps", paths.REPO / "scripts" / "make_exch_maps.py")
    mm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mm)
    ds = oracle(inp)
    m = EM.build_maps(ds, 0)
    bad = mm.check_against_probe(m, ds, 0)
    if sum(bad.values()):
        raise ValueError(f"exchange maps of {inp} do not reproduce the probe: {bad}")
    return Exchanger(m)


def run_top(inp, kind):
    reg = grid_gate._registry()
    runs = [r for r in reg.RUNS if (r["exp"], r["input"], r["kind"]) == (EXP, inp, kind)]
    if len(runs) != 1:
        raise FileNotFoundError(f"{EXP}/{inp}: {len(runs)} registered runs of kind {kind!r} "
                                f"(reference/reference_runs.py find_run({EXP!r}, {inp!r}, {kind!r}))")
    return reg.run_top(runs[0])


def read_global(path, nz, sz):
    """A global big-endian float64 MDS record of nz levels (useSingleCpuIO files: Nx*Ny*nz values) -> [tile, nz, j, i]
    with zero halos (tile bi-1 + (bj-1)*nSx)."""
    Nx, Ny = sz.sNx * sz.nSx, sz.sNy * sz.nSy
    raw = np.fromfile(path, ">f8", count=Nx * Ny * nz).reshape(nz, Ny, Nx)
    out = np.zeros((sz.nSx * sz.nSy, nz, sz.sNy + 2 * sz.OLy, sz.sNx + 2 * sz.OLx))
    for bj in range(sz.nSy):
        for bi in range(sz.nSx):
            out[bi + bj * sz.nSx, :, sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] = \
                raw[:, bj * sz.sNy:(bj + 1) * sz.sNy, bi * sz.sNx:(bi + 1) * sz.sNx]
    return out


def control_case(inp, kind="yardstick", it_state=0, xx_override=None, ds=None):
    """CTRL_MAP_INI_GENARR of the variant on run `kind`'s control and weight files: (fields after the map, effective
    {(dim, iarr): record}, oracle effective [tile, nz, j, i] or None, maskC). The model variables before the map:
    theta, salt of I02_ini_fields (INI_FIELDS, initialise_varia.F:225, before PACKAGES_INIT_VARIABLES :263) of the
    dumps-on run `ds` (default: the variant's jdon run), the K3D fields of our GMREDI_INIT_VARIA, bottomDragFld = 0
    (ctrl_init_variables.F:83-91). `xx_override(arr) -> arr` plants a control."""
    from mitjax.pkg.ctrl.ctrl_map_ini_genarr import ctrl_map_ini_genarr
    from mitjax.pkg.ctrl.ctrl_readparms import ctrl_readparms_genarr
    e = grid_gate.experiment(EXP, inp)
    cfg, sz = e.cfg, e.cfg.size
    ds = ds or oracle(inp)
    top = run_top(inp, kind)
    dim, name, target = CONTROL[inp]
    nz = sz.Nr if dim == 3 else 1
    g2, g3 = ctrl_readparms_genarr(e.run, 2), ctrl_readparms_genarr(e.run, 3)
    g = (g3 if dim == 3 else g2)[0]
    xx = read_global(top / "rundir" / f"{name}.0000000000.data", nz, sz)
    if xx_override is not None:
        xx = xx_override(xx)
    w = read_global(top / "rundir" / g.weight.strip(), nz, sz)
    ref_eff_path = top / "rundir" / f"{name}.effective.0000000000.data"
    ref_eff = read_global(ref_eff_path, nz, sz) if ref_eff_path.exists() else None
    maskC = FArray(jnp.asarray(ds.field(0, "G00_geometry", "maskC")), "maskC", **bounds("xyz", sz))
    gm = setup_gm(inp)
    fields = {"theta": xyz("theta", ds.field(it_state, "I02_ini_fields", "theta"), sz),
              "salt": xyz("salt", ds.field(it_state, "I02_ini_fields", "salt"), sz),
              "GM_inpK3dGM": gm.GM_inpK3dGM, "GM_inpK3dRedi": gm.GM_inpK3dRedi,
              "bottomDragFld": xy("bottomDragFld", np.zeros((sz.nSx * sz.nSy, sz.sNy + 2 * sz.OLy,
                                                              sz.sNx + 2 * sz.OLx)), sz)}
    if dim == 3:
        xx_in, w_in = xyz(name, xx, sz), xyz("w", w, sz)
    else:
        xx_in, w_in = xy(name, xx[:, 0], sz), xy("w", w[:, 0], sz)
    out, eff = ctrl_map_ini_genarr(fields, cfg=cfg, genarr2d=g2, genarr3d=g3, xx_in={(dim, 1): xx_in},
                                   weight_in={(dim, 1): w_in}, maskC=maskC, ex=exchanger_code_ad())
    return out, eff, ref_eff, maskC


def interior(a, sz):
    a = np.asarray(a)
    if a.ndim == 3:
        a = a[:, None]
    return a[:, :, sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx]
