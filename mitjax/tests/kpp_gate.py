"""Helpers of the KPP gates (M3 sub-lane KPP, plan Task 29).

Two oracles:
  * the KPP replay harness reference/replay_kpp/ (vermix/code build, synthetic columns over the real one-column grid
    with land/shallow overrides; runs of `input` (MDJWF) and `input.dd` (LINEAR, KPPuseDoubleDiff) registered in
    reference/replay_kpp/CURRENT): KPP_READPARMS + KPP_INIT_FIXED + KPP_INIT_VARIA values, STATEKPP, KPP_FORCING_SURF,
    KPPMIX, KPP_DOUBLEDIFF unit replays, KPP_CALC, KPP_DO_EXCH, KPP_CALC_DIFF_T/S, KPP_CALC_VISC, KPP_TRANSPORT_T/S;
  * the registered dumps-on runs (reference/reference_runs.py, kind "jdon") of vermix/input and vermix/input.dd:
    KPP_CALC replayed per dumped iteration from the oracle's inputs (theta, salt, uVel, vVel at S00_begin, the
    surface forcing at P01_external_forcing_surf, IVDConvCount at P03_mxlayer; the KPP.h priors of the points
    KPP_CALC does not write: KPP_INIT_VARIA's values at the first iteration, P10_kpp_exch of the previous one) and
    compared with P07_kpp; KPP_DO_EXCH of P07_kpp compared with P10_kpp_exch.
Comparison by element equality on every point of every tile, halos included, both arrays finite, plus the count of
differing bit patterns (sign of zeros). Every jit argument that holds a REAL parameter is traced (never closed over).
"""

import dataclasses
import functools
import importlib.util
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from mitjax import paths
from mitjax.farray import FArray
from mitjax.tests import grid_gate as gg
from mitjax.tests import init_gate as ig

EXP = "vermix"
VARIANTS = ("input", "input.dd")


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ForcingP:
    """The forcing parameters KPP reads (ini_parms_forcing's HeatCapacity_Cp traced, selectPenetratingSW static)."""
    HeatCapacity_Cp: object
    selectPenetratingSW: int = dataclasses.field(metadata=dict(static=True))


def _replay_io():
    spec = importlib.util.spec_from_file_location("_mjx_replay_kpp_io",
                                                  paths.REPO / "reference" / "replay_kpp" / "replay_io.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RIO = _replay_io()


def replay_rundir(inp):
    """The harness run directory of vermix/<inp> named by reference/replay_kpp/CURRENT (relative to $MJX_REFERENCE)."""
    p = paths.REPO / "reference" / "replay_kpp" / "CURRENT"
    for line in p.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        exp, i, rel = line.split()
        if (exp, i) == (EXP, inp):
            return paths.REFERENCE / rel
    raise FileNotFoundError(f"{p}: no replay run registered for {EXP}/{inp}")


@functools.lru_cache(maxsize=None)
def setup(inp):
    """SimpleNamespace(e, cfg, sz, params, grid, ex, eos (None for an unported EOS), fp, kpp (READPARMS+INIT_FIXED),
    tp, io) of vermix/<inp>."""
    from mitjax.model.grid import UNSET_RL
    from mitjax.model.src.ini_eos import ini_eos
    from mitjax.model.src.ini_parms import ini_parms_dyn, ini_parms_io
    from mitjax.model.src.ini_parms_forcing import ini_parms_forcing
    from mitjax.model.src.ini_parms_tracer import ini_parms_tracer
    from mitjax.params_io import RunParams
    from mitjax.pkg.kpp.kpp_init_fixed import kpp_init_fixed
    from mitjax.pkg.kpp.kpp_readparms import kpp_readparms
    e = gg.experiment(EXP, inp)
    cfg = e.cfg
    prm = ig.params(EXP, inp)
    params = ini_parms_dyn(e, prm.grid, prm.time, prm.init)
    unported = []
    try:
        params = ini_parms_tracer(e, params, prm.time, prm.init)
    except NotImplementedError as err:                 # vermix/input's legacy diffKzT/diffKzS (ini_parms_tracer)
        unported.append(str(err))
    # vermix has no registered map file: the maps decoded in memory from the run's own exchange probe
    # (X00_exch_probe) by the decoder the registered files are made with (as go_gate does for code_ad builds)
    from mitjax.eesupp.exch_maps import build_maps
    from mitjax.eesupp.exchange import Exchanger
    ds, _, _ = gg.oracle(EXP, inp)
    ex = Exchanger(build_maps(ds))
    grid = gg.build_grid(EXP, inp, params=prm.grid, ex=ex)
    rp = RunParams(e.run)
    eos_p = SimpleNamespace(fluidIsWater=params.fluidIsWater, usingPCoords=params.usingPCoords,
                            eosType=params.eosType,
                            tAlpha=rp.get("data", "PARM01", "tAlpha") if rp.has("data", "PARM01", "tAlpha")
                            else UNSET_RL,
                            sBeta=rp.get("data", "PARM01", "sBeta") if rp.has("data", "PARM01", "sBeta")
                            else UNSET_RL)
    try:
        eos = ini_eos(cfg=cfg, params=eos_p)
        if params.eosType.strip() in ("JMD95Z", "JMD95P", "UNESCO", "MDJWF"):   # as drivers/model.Model
            from mitjax.model.src.ini_parms import set_ref_state_eos
            params = set_ref_state_eos(params, grid, e)                   # set_ref_state.F:51-100 (surf_pRef)
    except NotImplementedError as err:
        eos = None                                     # MDJWF (vermix/input): INI_EOS / FIND_RHO_2D not ported
        unported.append(str(err))
    fpp = ini_parms_forcing(e)
    fp = ForcingP(HeatCapacity_Cp=jnp.float64(fpp.HeatCapacity_Cp), selectPenetratingSW=int(fpp.selectPenetratingSW))
    io = ini_parms_io(e)
    kpp = kpp_readparms(e, prm.time, io)
    kpp = kpp_init_fixed(kpp, cfg=cfg, grid=grid)
    return SimpleNamespace(e=e, cfg=cfg, sz=cfg.size, params=params, grid=grid, ex=ex, eos=eos, fp=fp, kpp=kpp,
                           tp=prm.time, io=io, unported=tuple(unported))


def require_model(s):
    """STATEKPP and KPP_CALC need the equation of state and the tracer parameters (CALC_3D_DIFFUSIVITY's diffKrNrT/S):
    raise NotImplementedError with the setup's reasons when a variant's are not ported."""
    if s.unported:
        raise NotImplementedError("; ".join(s.unported))


def compare(ours, ref):
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
    return sum(v[1] + v[4] for v in result.values() if v[0] != "shape")


# ---------------------------------------------------------------------------------------------------------------
# containers built inside the traced programs

def _xy(name, a, sz):
    return FArray(a, name, i=(1-sz.OLx, sz.sNx+sz.OLx), j=(1-sz.OLy, sz.sNy+sz.OLy))


def _xyz(name, a, sz, nk=None):
    return FArray(a, name, i=(1-sz.OLx, sz.sNx+sz.OLx), j=(1-sz.OLy, sz.sNy+sz.OLy),
                  k=(1, sz.Nr if nk is None else nk))


def case_objects(s, d, grid, kpp):
    """(grid with the case's maskC/W/S, fCori; kpp with nzmax; state; ff; KPP.h priors) from a case's inputs `d`
    (jax arrays, inside a traced program)."""
    sz = s.sz
    grid = grid.replace(maskC=_xyz("maskC", d["maskC"], sz), maskW=_xyz("maskW", d["maskW"], sz),
                        maskS=_xyz("maskS", d["maskS"], sz), fCori=_xy("fCori", d["fCori"], sz))
    kpp = kpp.replace(nzmax=_xy("nzmax", d["nzmax"].astype(jnp.int32), sz))
    state = SimpleNamespace(theta=_xyz("theta", d["theta"], sz), salt=_xyz("salt", d["salt"], sz),
                            uVel=_xyz("uVel", d["uVel"], sz), vVel=_xyz("vVel", d["vVel"], sz),
                            IVDConvCount=_xyz("IVDConvCount", d["IVDConv"], sz),
                            # the harness runs no INITIALISE_VARIA: DYNVARS.h totPhiHyd is the never-written common (0),
                            # read by FIND_RHO_2D's PRESSURE_FOR_EOS for a pressure-dependent EOS (MDJWF, input)
                            totPhiHyd=_xyz("totPhiHyd", jnp.zeros_like(d["theta"]), sz))
    ff = SimpleNamespace(surfaceForcingU=_xy("surfaceForcingU", d["sfU"], sz),
                         surfaceForcingV=_xy("surfaceForcingV", d["sfV"], sz),
                         surfaceForcingT=_xy("surfaceForcingT", d["sfT"], sz),
                         surfaceForcingS=_xy("surfaceForcingS", d["sfS"], sz),
                         adjustColdSST_diag=_xy("adjustColdSST_diag", d["adjCold"], sz), Qsw=_xy("Qsw", d["Qsw"], sz))
    kppf = {"KPPviscAz": _xyz("KPPviscAz", d["pViscAz"], sz), "KPPdiffKzS": _xyz("KPPdiffKzS", d["pDiffKzS"], sz),
            "KPPdiffKzT": _xyz("KPPdiffKzT", d["pDiffKzT"], sz), "KPPghat": _xyz("KPPghat", d["pGhat"], sz),
            "KPPhbl": _xy("KPPhbl", d["pHbl"], sz), "KPPfrac": _xy("KPPfrac", d["pFrac"], sz)}
    return grid, kpp, state, ff, kppf


def units_fn(s, planted=None):
    """jit(f(params, grid, kpp, fp, d) -> {output name: array}): KPP_FORCING_SURF, KPPMIX, KPP_DOUBLEDIFF on the case's
    synthetic intermediate inputs, as the harness calls them (the_main_loop.F 2c). `planted(name, value)` plants
    the error of a negative control in an input (static)."""
    from mitjax.pkg.kpp.kpp_forcing_surf import kpp_forcing_surf
    from mitjax.pkg.kpp.kpp_routines import imt_view, kpp_doublediff, kppmix
    cfg, sz = s.cfg, s.sz
    Nr = sz.Nr

    def f(params, grid, kpp, fp, d):
        if planted is not None:
            d = planted(d)
        grid, kpp, state, ff, _ = case_objects(s, d, grid, kpp)
        al = _xyz("TTALPHA", d["ttalphaI"], sz, Nr + 1)
        be = _xyz("SSBETA", d["ssbetaI"], sz, Nr + 1)
        dbI = _xyz("dbloc", d["dblocI"], sz)
        rhoS = _xy("rhoSurf", d["sdensI"], sz)
        nanxy = _xy("x", jnp.full(d["sfU"].shape, jnp.nan), sz)
        ust, bo, bosol, dvs = kpp_forcing_surf(
            rhoS, ff.surfaceForcingU, ff.surfaceForcingV, ff.surfaceForcingT, ff.surfaceForcingS,
            ff.adjustColdSST_diag, ff.Qsw, dbI, al, be, nanxy, nanxy, nanxy,
            _xyz("dVsq", jnp.full(d["theta"].shape, jnp.nan), sz), 1, 2-sz.OLx, sz.sNx+sz.OLx-1, 2-sz.OLy,
            sz.sNy+sz.OLy-1, 0.0, cfg=cfg, grid=grid, params=params, fp=fp, kpp=kpp, state=state)
        T = d["theta"].shape[0]
        n = d["theta"].shape[-1]*d["theta"].shape[-2]
        vd = {md: FArray(jnp.full((T, Nr + 2, n), -1.0), "vddiff", i=(1, n), k=(0, Nr + 1)) for md in (1, 2, 3)}
        msk = _xy("msk", d["maskC"][:, 0], sz)
        vd, gh, hbl = kppmix(imt_view(kpp.nzmax), imt_view(_xyz("shsq", d["shsqI"], sz)),
                             imt_view(_xyz("dvsq", d["dVsqI"], sz)), imt_view(_xy("ustar", d["ustarI"], sz)),
                             imt_view(msk), imt_view(_xy("bo", d["boI"], sz)),
                             imt_view(_xy("bosol", d["bosolI"], sz)), imt_view(dbI),
                             imt_view(_xyz("Ritop", d["RitopI"], sz)), imt_view(grid.fCori),
                             imt_view(_xyz("kS", d["kapSI"], sz)), imt_view(_xyz("kT", d["kapTI"], sz)), 1, vd,
                             imt_view(_xyz("ghat", d["ghatI"], sz)), imt_view(nanxy), 0.0, 0,
                             cfg=cfg, kpp=kpp, params=params)
        kT, kS = kpp_doublediff(al, be, _xyz("kT", d["kapTI"], sz), _xyz("kS", d["kapSI"], sz), 1, 1-sz.OLx,
                                sz.sNx+sz.OLx, 1-sz.OLy, sz.sNy+sz.OLy, cfg=cfg, kpp=kpp, state=state)
        shp = d["theta"].shape
        return {"fs_ustar": ust.data, "fs_bo": bo.data, "fs_bosol": bosol.data, "fs_dVsq": dvs.data,
                "mx_vddiff": jnp.stack([vd[md].data.reshape((T, Nr + 2) + shp[-2:]) for md in (1, 2, 3)], axis=1),
                "mx_ghat": gh.data.reshape(shp), "mx_hbl": hbl.reshape(shp[:1] + shp[-2:]),
                "dd_kapT": kT.data, "dd_kapS": kS.data}
    return jax.jit(f)


def calc_fn(s, with_statekpp_only=False):
    """jit(f(params, grid, kpp, fp, eos, d) -> {name: array}): STATEKPP and KPP_CALC (the_main_loop.F 2c, 2d)."""
    from mitjax.pkg.kpp.kpp_calc import kpp_calc
    from mitjax.pkg.kpp.kpp_routines import statekpp
    require_model(s)
    cfg = s.cfg

    def f(params, grid, kpp, fp, eos, d):
        grid, kpp, state, ff, kppf = case_objects(s, d, grid, kpp)
        rho1, dbl, dbs, al, be = statekpp(1, cfg=cfg, grid=grid, params=params, eos=eos, state=state)
        out = {"sk_rho1": rho1.data, "sk_dbloc": dbl.data, "sk_dbsfc": dbs.data, "sk_alpha": al.data,
               "sk_beta": be.data}
        if with_statekpp_only:
            return out
        new = kpp_calc(0.0, 0, cfg=cfg, grid=grid, params=params, fp=fp, eos=eos, state=state, ff=ff, kpp=kpp,
                       kppf=kppf)
        out.update({n: new[n].data for n in ("KPPviscAz", "KPPdiffKzS", "KPPdiffKzT", "KPPghat", "KPPhbl",
                                             "KPPfrac")})
        return out
    return jax.jit(f)


def after_fn(s):
    """jit(f(params, grid, kpp, fp, d, kppo) -> {name: array}): KPP_DO_EXCH of the oracle's KPP_CALC output, then
    KPP_CALC_DIFF_T/S, KPP_CALC_VISC, KPP_TRANSPORT_T/S on the oracle's exchanged fields (the_main_loop.F 2d-2e)."""
    from mitjax.pkg.kpp.kpp_calc_diff_s import kpp_calc_diff_s
    from mitjax.pkg.kpp.kpp_calc_diff_t import kpp_calc_diff_t
    from mitjax.pkg.kpp.kpp_calc_visc import kpp_calc_visc
    from mitjax.pkg.kpp.kpp_do_exch import kpp_do_exch
    from mitjax.pkg.kpp.kpp_transport_s import kpp_transport_s
    from mitjax.pkg.kpp.kpp_transport_t import kpp_transport_t
    cfg, sz, ex = s.cfg, s.sz, s.ex
    Nr = sz.Nr

    def f(params, grid, kpp, fp, d, kppo):
        grid, kpp, state, ff, _ = case_objects(s, d, grid, kpp)
        kppf = {"KPPviscAz": _xyz("KPPviscAz", kppo["KPPviscAz"], sz),
                "KPPdiffKzS": _xyz("KPPdiffKzS", kppo["KPPdiffKzS"], sz),
                "KPPdiffKzT": _xyz("KPPdiffKzT", kppo["KPPdiffKzT"], sz),
                "KPPghat": _xyz("KPPghat", kppo["KPPghat"], sz), "KPPhbl": _xy("KPPhbl", kppo["KPPhbl"], sz),
                "KPPfrac": _xy("KPPfrac", kppo["KPPfrac"], sz)}
        ex_ = kpp_do_exch(kppf, cfg=cfg, ex=ex)
        out = {"ex_viscAz": ex_["KPPviscAz"].data}
        kppf = dict(kppf, KPPviscAz=_xyz("KPPviscAz", kppo["ex_viscAz"], sz))
        kT, kS = _xyz("kT", d["kapTI"], sz), _xyz("kS", d["kapSI"], sz)
        out["cd_T0"] = kpp_calc_diff_t(1-sz.OLx, sz.sNx+sz.OLx, 1-sz.OLy, sz.sNy+sz.OLy, 0, Nr, kT, cfg=cfg,
                                       kppf=kppf).data
        out["cd_S0"] = kpp_calc_diff_s(1-sz.OLx, sz.sNx+sz.OLx, 1-sz.OLy, sz.sNy+sz.OLy, 0, Nr, kS, cfg=cfg,
                                       kppf=kppf).data
        kRU, kRV = _xyz("kRU", d["kapRU"], sz), _xyz("kRV", d["kapRV"], sz)
        for k in range(1, Nr + 1):
            kT = kpp_calc_diff_t(0, sz.sNx+1, 0, sz.sNy+1, k, Nr, kT, cfg=cfg, kppf=kppf)
            kS = kpp_calc_diff_s(0, sz.sNx+1, 0, sz.sNy+1, k, Nr, kS, cfg=cfg, kppf=kppf)
            kRU, kRV = kpp_calc_visc(0, sz.sNx+1, 0, sz.sNy+1, k, kRU, kRV, cfg=cfg, grid=grid, params=params,
                                     kppf=kppf)
        out.update(cd_Tk=kT.data, cd_Sk=kS.data, cv_RU=kRU.data, cv_RV=kRV.data)
        trT, trS = [], []
        for k in range(1, Nr + 1):
            km1 = max(1, k-1)                                                  # MINMAX-INT: host level index
            df0 = _xy("df", d["dfPrior"][:, k-1], sz)
            trT.append(kpp_transport_t(0, sz.sNx+1, 0, sz.sNy+1, k, km1, df0, 0.0, 0, cfg=cfg, grid=grid,
                                       params=params, fp=fp, ff=ff, kppf=kppf).data)
            trS.append(kpp_transport_s(0, sz.sNx+1, 0, sz.sNy+1, k, km1, df0, 0.0, 0, cfg=cfg, grid=grid, ff=ff,
                                       kppf=kppf).data)
        out.update(tr_T=jnp.stack(trT, axis=1), tr_S=jnp.stack(trS, axis=1))
        return out
    return jax.jit(f)


def replay(inp, which=("units", "after", "calc"), planted=None, cases=None):
    """{case: {name: comparison}} of the chosen groups against the harness outputs of vermix/<inp>."""
    s = setup(inp)
    rd = replay_rundir(inp)
    ins = RIO.read_inputs(rd, _size_dict(s.sz))
    outs = RIO.read_outputs(rd, _size_dict(s.sz))
    res = {}
    fu = units_fn(s, planted) if "units" in which else None
    fa = after_fn(s) if "after" in which else None
    fc = calc_fn(s) if "calc" in which else None
    for c, (d, o) in enumerate(zip(ins, outs), start=1):
        if cases is not None and c not in cases:
            continue
        dj = {k: jnp.asarray(v) for k, v in d.items()}
        r = {}
        if fu is not None:
            ours = jax.device_get(fu(s.params, s.grid, s.kpp, s.fp, dj))
            r.update(compare(ours, {k: o[k] for k in ours}))
        if fa is not None:
            kppo = {k: jnp.asarray(o[k]) for k in ("KPPviscAz", "KPPdiffKzS", "KPPdiffKzT", "KPPghat", "KPPhbl",
                                                    "KPPfrac", "ex_viscAz")}
            ours = jax.device_get(fa(s.params, s.grid, s.kpp, s.fp, dj, kppo))
            r.update(compare(ours, {k: o[k] for k in ours}))
        if fc is not None:
            ours = jax.device_get(fc(s.params, s.grid, s.kpp, s.fp, s.eos, dj))
            r.update(compare(ours, {k: o[k] for k in ours}))
        res[c] = r
    return res


def _size_dict(sz):
    return {k: getattr(sz, k) for k in ("sNx", "sNy", "OLx", "OLy", "nSx", "nSy", "Nr")}


# ---------------------------------------------------------------------------------------------------------------
# dump gates (registered dumps-on runs)

DUMP_FIELDS = ("KPPviscAz", "KPPdiffKzS", "KPPdiffKzT", "KPPghat", "KPPhbl", "KPPfrac")


def init_varia_fields(s):
    """(kpp with nzmax, KPP.h fields) after KPP_INIT_VARIA (KPPfrac: 0, the zero of an uninitialised common block,
    KPP.h:44 COMMON /kpp_short/; SHORTWAVE_HEATING is not compiled)."""
    from mitjax.pkg.kpp.kpp_init_varia import kpp_init_varia
    from mitjax.pkg.kpp.kpp_params_h import kpp_fields
    sz = s.sz
    T, ny, nx = sz.nSx*sz.nSy, sz.sNy+2*sz.OLy, sz.sNx+2*sz.OLx
    f = kpp_fields(sz, {"KPPfrac": np.zeros((T, ny, nx))})
    return kpp_init_varia(s.kpp, f, cfg=s.cfg, grid=s.grid, params=s.params)


def dump_inputs(inp, it, prior_override=None):
    """{name: array} of the teacher-forced inputs of KPP_CALC at dumped iteration `it` and the KPP.h priors."""
    ds, _, _ = gg.oracle(EXP, inp)
    s = setup(inp)
    d = {n: ds.field(it, "S00_begin", n) for n in ("theta", "salt", "uVel", "vVel", "totPhiHyd")}
    d["IVDConvCount"] = ds.field(it, "P03_mxlayer", "IVDConvCount")
    for n in ("surfaceForcingU", "surfaceForcingV", "surfaceForcingT", "surfaceForcingS", "Qsw"):
        d[n] = ds.field(it, "P01_external_forcing_surf", n)[:, 0]
    d["adjustColdSST_diag"] = np.zeros_like(d["Qsw"])          # INI_FFIELDS' 0. _d 0; nothing sets it in vermix
    its = ds.iterations()
    if it == its[0]:
        _, f = init_varia_fields(s)
        prior = {n: np.asarray(f[n].data) for n in DUMP_FIELDS}
    else:
        prev = its[its.index(it) - 1]
        prior = {n: ds.field(prev, "P10_kpp_exch", n) for n in DUMP_FIELDS}
        prior["KPPhbl"], prior["KPPfrac"] = prior["KPPhbl"][:, 0], prior["KPPfrac"][:, 0]
    if prior_override is not None:
        prior = prior_override(prior)
    return d, prior


def dump_calc_fn(s):
    from mitjax.pkg.kpp.kpp_calc import kpp_calc
    require_model(s)
    cfg, sz = s.cfg, s.sz

    def f(params, grid, kpp, fp, eos, d, prior, myTime, myIter):
        state = SimpleNamespace(theta=_xyz("theta", d["theta"], sz), salt=_xyz("salt", d["salt"], sz),
                                uVel=_xyz("uVel", d["uVel"], sz), vVel=_xyz("vVel", d["vVel"], sz),
                                IVDConvCount=_xyz("IVDConvCount", d["IVDConvCount"], sz),
                                totPhiHyd=_xyz("totPhiHyd", d["totPhiHyd"], sz))     # PRESSURE_FOR_EOS (MDJWF)
        ff = SimpleNamespace(**{n: _xy(n, d[n], sz) for n in ("surfaceForcingU", "surfaceForcingV",
                                                              "surfaceForcingT", "surfaceForcingS",
                                                              "adjustColdSST_diag", "Qsw")})
        kppf = {n: (_xy if n in ("KPPhbl", "KPPfrac") else _xyz)(n, prior[n], sz) for n in DUMP_FIELDS}
        new = kpp_calc(myTime, myIter, cfg=cfg, grid=grid, params=params, fp=fp, eos=eos, state=state, ff=ff,
                       kpp=kpp, kppf=kppf)
        return {n: new[n].data for n in DUMP_FIELDS}
    return jax.jit(f)


def dump_case(inp, it, prior_override=None, kpp_override=None):
    """(ours, oracle) {name: array} of KPP_CALC at dumped iteration `it` (vs P07_kpp) and of KPP_DO_EXCH of the
    oracle's P07_kpp (vs P10_kpp_exch, names suffixed `_exch`)."""
    from mitjax.pkg.kpp.kpp_do_exch import kpp_do_exch
    s = setup(inp)
    ds, _, _ = gg.oracle(EXP, inp)
    kpp, _ = init_varia_fields(s)
    if kpp_override is not None:
        kpp = kpp_override(kpp)
    d, prior = dump_inputs(inp, it, prior_override)
    tp = s.tp
    myTime = jnp.float64(tp.startTime + tp.deltaTClock*(it - tp.nIter0))
    f = dump_calc_fn(s)
    ours = jax.device_get(f(s.params, s.grid, kpp, s.fp, s.eos, {k: jnp.asarray(v) for k, v in d.items()},
                            {k: jnp.asarray(v) for k, v in prior.items()}, myTime, jnp.int32(it)))
    ref = {n: ds.field(it, "P07_kpp", n) for n in DUMP_FIELDS}
    ref["KPPhbl"], ref["KPPfrac"] = ref["KPPhbl"][:, 0], ref["KPPfrac"][:, 0]
    sz = s.sz
    p07 = {n: (_xy if n in ("KPPhbl", "KPPfrac") else _xyz)(n, jnp.asarray(ref[n]), sz) for n in DUMP_FIELDS}
    ex = jax.device_get(jax.jit(lambda f_: {n: v.data for n, v in kpp_do_exch(f_, cfg=s.cfg, ex=s.ex).items()})(p07))
    for n in DUMP_FIELDS:
        ours[n + "_exch"] = ex[n]
        r = ds.field(it, "P10_kpp_exch", n)
        ref[n + "_exch"] = r[:, 0] if n in ("KPPhbl", "KPPfrac") else r
    return ours, ref
