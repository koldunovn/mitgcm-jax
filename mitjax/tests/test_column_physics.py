"""COL gates (M1 sub-lane COL): the equation of state and column-physics kernels of mitjax/model/src bitwise equal to
gfortran on the COL replay harness (reference/replay_col; runs named by reference/replay_col/CURRENT): every output,
every point of every tile and level incl. halos and land, element equality of bit patterns plus finiteness; float
parameters traced (jit arguments) under the gate XLA flags (conftest.py). Per routine a planted error measured to
bite; INI_EOS against the harness's EOS.h; gradients finite on all lanes, FD at smooth points.

Experiments (each routine where the M1 experiment runs it, with that build's CPP options):
    tutorial_baroclinic_gyre/code  (LINEAR EOS; CALC_OCE_MXLAYER method 1, FIND_ALPHA)
    global_ocean.90x40x15/code     (JMD95P, selectP_inEOS_Zc = 2; r* INTEGRATE_FOR_W; ALLOW_PTRACERS)
    tutorial_global_oce_optim/code_ad forward-only (JMD95Z, selectP_inEOS_Zc = 0; ALLOW_AUTODIFF; IMPLDIFF)
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax.farray import FArray
from mitjax.model.src import (calc_ivdc, calc_oce_mxlayer, calc_viscosity, find_alpha, find_rho, impldiff,
                              ini_eos, integrate_for_w, pressure_for_eos, solve_tridiagonal)
from mitjax.ops.scan_k import level
from mitjax.tests import col_replay as cr

RUNS = cr.current_runs()
IDS = [f"{e}/{i}" for e, i, _ in RUNS]


@functools.lru_cache(maxsize=None)
def replay(idx):
    e, i, d = RUNS[idx]
    return cr.Replay(e, i, d)


def by_exp(name):
    for n, (e, _, _) in enumerate(RUNS):
        if e == name:
            return replay(n)
    raise LookupError(name)


def _stack(levels):
    return jnp.stack([x.data for x in levels], axis=1)


# ---------------------------------------------------------------------------------------------------------------
# drivers: each calls the routine as the harness (reference/replay_col/code/the_main_loop.F) does

def drive_eos(R, m_find_rho=find_rho, m_pfe=pressure_for_eos):
    cfg, sz = R.cfg, R.size
    Nr, i0, i1, j0, j1 = sz.Nr, 1 - sz.OLx, sz.sNx + sz.OLx, 1 - sz.OLy, sz.sNy + sz.OLy

    def run(tFld, sFld, totPhi, pLoc, rhoPrior, grid, params, eos):
        state = cr.Common(totPhiHyd=totPhi)
        kw = dict(cfg=cfg, grid=grid, params=params, eos=eos, state=state)
        out = {n: [] for n in ("rhoFull", "rhoPart", "rhoP0", "locPres", "bulkMod")}
        for k in range(1, Nr + 1):
            t2, s2, prior = level(tFld, k), level(sFld, k), level(rhoPrior, k)
            out["rhoFull"].append(m_find_rho.find_rho_2d(i0, i1, j0, j1, k, t2, s2, prior, k, **kw))
            out["rhoPart"].append(m_find_rho.find_rho_2d(0, sz.sNx + 1, 0, sz.sNy + 1, max(k - 1, 1), t2, s2,
                                                         prior, k, **kw))
            out["rhoP0"].append(m_find_rho.find_rhop0(i0, i1, j0, j1, t2, s2, prior, cfg=cfg, eos=eos))
            dp0 = params.surf_pRef - eos.eosRefP0
            p2 = m_pfe.pressure_for_eos(i0, i1, j0, j1, k, dp0, prior, cfg=cfg, grid=grid, params=params,
                                        state=state)
            out["locPres"].append(p2)
            out["bulkMod"].append(m_find_rho.find_bulkmod(i0, i1, j0, j1, p2, t2, s2, prior, cfg=cfg, eos=eos))
        res = {n: _stack(v) for n, v in out.items()}
        res["rhoScal"] = m_find_rho.find_rho_scalar(tFld.data, jnp.abs(sFld.data), pLoc.data, cfg=cfg,
                                                    params=params, eos=eos)
        return res

    grid, params, eos = R.commons()
    args = [R.fields3(n) for n in ("tFld", "sFld", "totPhi", "pLoc", "rhoPrior")]
    return run, args + [grid, params, eos]


def drive_tridiag(R, mod=solve_tridiagonal):
    cfg, sz = R.cfg, R.size

    def run(a, b, c, y):
        return mod.solve_tridiagonal(0, sz.sNx + 1, 0, sz.sNy + 1, a, b, c, y, -1, cfg=cfg)

    return run, [R.fields3(n) for n in ("aTri", "bTri", "cTri", "yTri")]


def drive_impldiff(R, tracerId, mod=impldiff):
    cfg, sz = R.cfg, R.size

    def run(kap, g, grid, params):
        return mod.impldiff(0, sz.sNx + 1, 0, sz.sNy + 1, tracerId, kap, grid.recip_hFacC, g, cfg=cfg, grid=grid,
                            params=params).data

    grid, params, _ = R.commons()
    return run, [R.fields3("kappaRX"), R.fields3("gTr"), grid, params]


def drive_ivdc(R, mod=calc_ivdc):
    sz = R.size

    def run(sigmaR, prior, grid):
        state = cr.Common(IVDConvCount=prior)
        for k in range(sz.Nr, 1, -1):
            state = state.replace(IVDConvCount=mod.calc_ivdc(1 - sz.OLx, sz.sNx + sz.OLx, 1 - sz.OLy,
                                                             sz.sNy + sz.OLy, k, sigmaR, 0.0, 0, grid=grid,
                                                             state=state))
        return state.IVDConvCount.data

    grid, _, _ = R.commons()
    return run, [R.fields3("sigmaR"), R.fields3("ivdcPrior"), grid]


def drive_w(R, mod=integrate_for_w):
    cfg, sz = R.cfg, R.size
    myIter = int(R.g["myIter"])

    def run(u, v, w, rDh, grid, params):
        for k in range(sz.Nr, 0, -1):
            w = mod.integrate_for_w(k, u, v, None, rDh, w, myIter, cfg=cfg, grid=grid, params=params)
        return w.data

    grid, params, _ = R.commons()
    return run, [R.fields3("uFld"), R.fields3("vFld"), R.fields3("wPrior"), R.fields2("rStarDhDt"), grid, params]


def drive_visc(R, mod=calc_viscosity):
    cfg, sz = R.cfg, R.size
    ij = cr._ij(sz)

    def run(params):
        prior = FArray(jnp.full((sz.nSx * sz.nSy, sz.Nr + 1, sz.sNy + 2 * sz.OLy, sz.sNx + 2 * sz.OLx), jnp.nan),
                       "kappaRU", k=(1, sz.Nr + 1), **ij)
        u, v = mod.calc_viscosity(0, sz.sNx + 1, 0, sz.sNy + 1, prior, prior, cfg=cfg, params=params)
        return u.data, v.data

    _, params, _ = R.commons()
    return run, [params]


def _gmredi(R):
    """GMREDI.h values CALC_OCE_MXLAYER reads, from data.gmredi or the gmredi_readparms.F defaults
    (GM_useSubMeso = .FALSE., GM_taper_scheme = ' ', GM_useBatesK3d = .FALSE.)."""
    v = {k: x.value for (f, g, k), x in R.experiment.run.vars.items() if f == "data.gmredi" and x.value is not None}
    return cr.Common(static={"GM_useSubMeso": bool(v.get("gm_usesubmeso", False)),
                             "GM_taper_scheme": str(v.get("gm_taper_scheme", " ")),
                             "GM_useBatesK3d": bool(v.get("gm_usebatesk3d", False))})


def _diag_fields(R):
    """Diagnostics listed in data.diagnostics (DIAGNOSTICS_LIST fields): stands for DIAGNOSTICS_IS_ON at the first
    time step (every listed diagnostic is active for accumulation)."""
    names = set()
    for (f, g, k), x in R.experiment.run.vars.items():
        if f == "data.diagnostics" and k == "fields" and x.value:
            names.update(str(s).strip() for s in dict(x.value).values())
    return names


def drive_mxl(R, mod=calc_oce_mxlayer, m_alpha=find_alpha):
    cfg, sz = R.cfg, R.size
    gm = _gmredi(R) if cfg.cpp.ALLOW_GMREDI and cfg.use_flag("useGMRedi") else None
    listed = _diag_fields(R)

    def run(rhoSurf, sigmaR, tFld, sFld, hPrior, aPrior, grid, params, eos):
        state = cr.Common(theta=tFld, salt=sFld, hMixLayer=hPrior)
        alpha = m_alpha.find_alpha(1 - sz.OLx, sz.sNx + sz.OLx, 1 - sz.OLy, sz.sNy + sz.OLy, 1, 1, aPrior,
                                   params=params, eos=eos) if eos.equationOfState.rstrip() == "LINEAR" else aPrior
        h = mod.calc_oce_mxlayer(rhoSurf, sigmaR, 0.0, 0, cfg=cfg, grid=grid, params=params, eos=eos, state=state,
                                 gmredi=gm, diagnostics_is_on=lambda n: n in listed)
        return alpha.data, h.data

    grid, params, eos = R.commons()
    rhoSurf = FArray(jnp.asarray(R.out["rhoFull"][:, 0]), "rhoSurf", **cr._ij(sz))
    aPrior = FArray(jnp.asarray(R.inputs["rhoPrior"][:, 0]), "alphaLoc", **cr._ij(sz))
    return run, [rhoSurf, R.fields3("sigmaR"), R.fields3("tFld"), R.fields3("sFld"), R.fields2("hMixPrior"), aPrior,
                 grid, params, eos]


def _check(name, got, want, R):
    n, tot, fin = cr.bit_equal(got, want)
    print(f"{R.exp}: {name}: {n} of {tot} points differ; finite {fin}")
    assert fin, f"{name}: non-finite values"
    assert n == 0, f"{name}: {n} of {tot} points differ"


# ---------------------------------------------------------------------------------------------------------------
# replay gates

@pytest.mark.parametrize("idx", range(len(RUNS)), ids=IDS)
def test_eos_replay_bitwise(idx):
    R = replay(idx)
    run, args = drive_eos(R)
    got = jax.jit(run)(*args)
    for n in ("rhoFull", "rhoPart", "rhoP0", "locPres", "bulkMod", "rhoScal"):
        _check(n, got[n], R.out[n], R)


@pytest.mark.parametrize("idx", range(len(RUNS)), ids=IDS)
def test_ini_eos_matches_harness(idx):
    """INI_EOS from the experiment's namelists (eosType, tAlpha, sBeta as INI_PARMS leaves them) == EOS.h dumped by
    the harness after INITIALISE_FIXED: every coefficient bitwise."""
    R = replay(idx)
    v = {k: x.value for (f, g, k), x in R.experiment.run.vars.items() if f == "data" and x.value is not None}
    p = cr.Common(static={"fluidIsWater": True, "usingPCoords": False, "eosType": R.eosType(),
                          # ini_parms.F:421-422  tAlpha = UNSET_RL, sBeta = UNSET_RL before the namelist READ
                          "tAlpha": float(v.get("talpha", ini_eos.UNSET_RL)),
                          "sBeta": float(v.get("sbeta", ini_eos.UNSET_RL))})
    e = ini_eos.ini_eos(cfg=R.cfg, params=p)
    for n in ("eosJMDCFw", "eosJMDCSw", "eosJMDCKFw", "eosJMDCKSw", "eosJMDCKP"):
        _check(n, getattr(e, n).data, R.g[n], R)
    for n in ("eosRefP0", "tAlpha", "sBeta"):
        _check(n, np.float64(getattr(e, n)), np.float64(R.g[n]), R)


@pytest.mark.parametrize("idx", range(len(RUNS)), ids=IDS)
def test_tridiagonal_replay_bitwise(idx):
    R = replay(idx)
    run, args = drive_tridiag(R)
    y, err = jax.jit(run)(*args)
    _check("yTri", y.data, R.out["yTri"], R)
    assert np.array_equal(np.asarray(err), R.out["errTri"].astype(np.int32)), (err, R.out["errTri"])


@pytest.mark.parametrize("idx", range(len(RUNS)), ids=IDS)
@pytest.mark.parametrize("tracerId,out", [(1, "gImpT"), (-1, "gImpM")])
def test_impldiff_replay_bitwise(idx, tracerId, out):
    R = replay(idx)
    run, args = drive_impldiff(R, tracerId)
    _check(out, jax.jit(run)(*args), R.out[out], R)


@pytest.mark.parametrize("idx", range(len(RUNS)), ids=IDS)
def test_ivdc_replay_bitwise(idx):
    R = replay(idx)
    run, args = drive_ivdc(R)
    got = jax.jit(run)(*args)
    _check("ivdc", got, R.out["ivdc"], R)
    unstable = np.count_nonzero(R.out["ivdc"][:, 1:] == 1.0)
    print(f"{R.exp}: {unstable} statically unstable points (IVDConvCount = 1)")
    assert unstable > 0


@pytest.mark.parametrize("idx", range(len(RUNS)), ids=IDS)
def test_integrate_for_w_replay_bitwise(idx):
    R = replay(idx)
    run, args = drive_w(R)
    _check("wFld", jax.jit(run)(*args), R.out["wFld"], R)


@pytest.mark.parametrize("idx", range(len(RUNS)), ids=IDS)
def test_calc_viscosity_replay_bitwise(idx):
    R = replay(idx)
    run, args = drive_visc(R)
    u, v = jax.jit(run)(*args)
    _check("kappaRU", u, R.out["kappaRU"], R)
    _check("kappaRV", v, R.out["kappaRV"], R)


@pytest.mark.parametrize("idx", range(len(RUNS)), ids=IDS)
def test_mxlayer_replay_bitwise(idx):
    R = replay(idx)
    run, args = drive_mxl(R)
    alpha, h = jax.jit(run)(*args)
    _check("hMix", h, R.out["hMix"], R)
    changed = np.count_nonzero(R.out["hMix"] != R.inputs["hMixPrior"])
    print(f"{R.exp}: hMixLayer written at {changed} points")
    if R.eosType().rstrip() == "LINEAR":
        _check("alpha", alpha, R.out["alpha"], R)
        assert changed > 0                       # method 1 ran (MXLDEPTH is diagnosed in this experiment)
    else:
        assert changed == 0                      # calcMixLayerDepth is false here


# ---------------------------------------------------------------------------------------------------------------
# negative controls: a planted error, measured to bite on the same replay data

CONTROLS = [
    # (experiment, driver, module, old, new, output)
    ("global_ocean.90x40x15", "eos", "find_rho", "+ KP[10]*t\n", "+ KP[11]*t\n", "rhoFull"),
    ("tutorial_global_oce_optim", "eos", "find_rho", "+ KP[10]*t\n", "+ KP[11]*t\n", "rhoFull"),
    # the ALLOW_AUTODIFF forward branch of FIND_RHO_2D (find_rho.F:75-83, code_ad builds) removed
    ("tutorial_global_oce_optim", "eos", "find_rho", "if cfg.cpp.ALLOW_AUTODIFF:", "if False:", "rhoPart"),
    ("tutorial_baroclinic_gyre", "eos", "find_rho", "refTemp = params.tRef[kRef]", "refTemp = params.tRef[k]",
     "rhoPart"),
    # FIND_RHOP0's s <= 0 branch (:350 s = 0): s*s with the unclipped salinity bites at the s < 0 points only
    ("global_ocean.90x40x15", "eos", "find_rho", "+ Sw[9]*s*s)", "+ Sw[9]*sFld[i, j]*sFld[i, j])", "rhoP0"),
    ("global_ocean.90x40x15", "pfe", "pressure_for_eos", "+ params.phiRef[2*k]) + dpRef)",
     "+ params.phiRef[2*k-1]) + dpRef)", "locPres"),
    ("tutorial_global_oce_optim", "pfe", "pressure_for_eos", "params.pRef4EOS[k] + dpRef",
     "params.pRef4EOS[max(k-1, 1)] + dpRef", "locPres"),
    ("tutorial_baroclinic_gyre", "tridiag", "solve_tridiagonal", "- x[\"cp\"][i, j]*ykp1[i, j])",
     "- x[\"cp\"][i, j]*ykp1[i, j]*1.0000000000000002)", "yTri"),
    ("tutorial_baroclinic_gyre", "tridiag", "solve_tridiagonal",
     "jnp.where(nzk, x[\"c\"][i, j]*rv, 0.0)", "jnp.where(nzk, x[\"c\"][i, j]*rv, 1.0)", "yTri"),
    # the c mask (impldiff.F:151) removed; the a mask (:133) cannot bite on these grids: no wet cell lies under a
    # dry one, and where recip_hFac(k) = 0 the product is 0 anyway (measured: 0 points)
    ("tutorial_global_oce_optim", "impldiff", "impldiff", "recip_hFac[i, j, k+1] == 0.0, 0.0,",
     "recip_hFac[i, j, k+1] == 7.0, 0.0,", "gImpT"),
    ("tutorial_global_oce_optim", "impldiff", "impldiff", "* grid.deepFac2F[k+1]*params.rhoFacF[k+1]))",
     "* grid.deepFac2F[k]*params.rhoFacF[k+1]))", "gImpT"),
    # IVDC (decided 2026-10-01): the convective branch removed
    ("tutorial_baroclinic_gyre", "ivdc", "calc_ivdc", "> 0.0, 1.0, 0.0))", "> 0.0, 0.0, 0.0))", "ivdc"),
    ("global_ocean.90x40x15", "w", "integrate_for_w", "(wFld[i, j, k+1]*grid.deepFac2F[k+1]*params.rhoFacF[k+1]\n"
     "                 + conv2d[i, j]*grid.recip_rA[i, j]\n                 - rStarDhDt",
     "(wFld[i, j, k+1]*grid.deepFac2F[k]*params.rhoFacF[k+1]\n"
     "                 + conv2d[i, j]*grid.recip_rA[i, j]\n                 - rStarDhDt", "wFld"),
    ("tutorial_baroclinic_gyre", "w", "integrate_for_w", "conv2d = conv2d.at[i, j].set(-(uTrans[i+1, j]-uTrans[i, j]",
     "conv2d = conv2d.at[i, j].set(-(uTrans[i+1, j]-uTrans[i+1, j]", "wFld"),
    ("tutorial_baroclinic_gyre", "visc", "calc_viscosity", "ki = min(k, Nr)", "ki = max(min(k, Nr)-1, 1)",
     "kappaRU"),
    ("tutorial_baroclinic_gyre", "mxl", "calc_oce_mxlayer", "kIn = (x[\"k\"] <= grid.kLowC[i, j])",
     "kIn = (x[\"k\"] < grid.kLowC[i, j])", "hMix"),
]


def _run_driver(R, kind, mod, out):
    if kind in ("eos", "pfe"):
        run, args = drive_eos(R, **({"m_find_rho": mod} if kind == "eos" else {"m_pfe": mod}))
        return jax.jit(run)(*args)[out]
    if kind == "tridiag":
        run, args = drive_tridiag(R, mod)
        return jax.jit(run)(*args)[0].data
    if kind == "impldiff":
        run, args = drive_impldiff(R, 1, mod)
        return jax.jit(run)(*args)
    if kind == "ivdc":
        run, args = drive_ivdc(R, mod)
        return jax.jit(run)(*args)
    if kind == "w":
        run, args = drive_w(R, mod)
        return jax.jit(run)(*args)
    if kind == "visc":
        run, args = drive_visc(R, mod)
        return jax.jit(run)(*args)[0]
    if kind == "mxl":
        run, args = drive_mxl(R, mod)
        return jax.jit(run)(*args)[1]
    raise ValueError(kind)


@pytest.mark.parametrize("exp,kind,module,old,new,out", CONTROLS,
                         ids=[f"{c[0]}:{c[2]}:{n}" for n, c in enumerate(CONTROLS)])
def test_negative_control_bites(exp, kind, module, old, new, out):
    R = by_exp(exp)
    mod = cr.planted(module, old, new)
    got = _run_driver(R, kind, mod, out)
    n, tot, _ = cr.bit_equal(got, R.out[out])
    print(f"{exp}: planted {module}: {old!r} -> {new!r}: {n} of {tot} points of {out} differ")
    assert n > 0


# ---------------------------------------------------------------------------------------------------------------
# gradients: finite on every lane (halos, land, s <= 0, b = 0 pivots), FD at smooth points

def _grad_case(R, kind):
    """(elem(*args) -> tuple of weighted output arrays w*out, number of differentiated leading args, args). The
    cost is the sum of all elements; FD differences are taken element by element before summing, so that points a
    perturbation does not reach cancel exactly (no round-off from the rest of the field)."""
    rng = np.random.default_rng(11)
    W = lambda n: jnp.asarray(rng.standard_normal(R.out[n].shape))      # noqa: E731
    if kind == "eos":
        run, args = drive_eos(R)
        w = {n: W(n) for n in ("rhoFull", "rhoPart", "rhoScal")}
        return (lambda *a: tuple(w[n] * run(*a)[n] for n in w)), 3, args
    if kind == "tridiag":
        run, args = drive_tridiag(R)
        w = W("yTri")
        return (lambda *a: (w * run(*a)[0].data,)), 4, args
    if kind == "impldiff":
        run, args = drive_impldiff(R, 1)
        w = W("gImpT")
        return (lambda *a: (w * run(*a),)), 2, args
    if kind == "w":
        run, args = drive_w(R)
        w = W("wFld")
        return (lambda *a: (w * run(*a),)), 4, args
    if kind == "mxl":
        run, args = drive_mxl(R)
        w = W("hMix")
        return (lambda *a: (w * run(*a)[1],)), 3, args
    raise ValueError(kind)


GRAD_CASES = [("tutorial_baroclinic_gyre", "eos"), ("global_ocean.90x40x15", "eos"),
              ("tutorial_global_oce_optim", "eos"), ("tutorial_baroclinic_gyre", "tridiag"),
              ("tutorial_global_oce_optim", "impldiff"), ("global_ocean.90x40x15", "w"),
              ("tutorial_baroclinic_gyre", "w"), ("tutorial_baroclinic_gyre", "mxl")]


@pytest.mark.parametrize("exp,kind", GRAD_CASES, ids=[f"{e}:{k}" for e, k in GRAD_CASES])
def test_gradients_finite_and_fd(exp, kind):
    R = by_exp(exp)
    elem, nd, args = _grad_case(R, kind)
    cost = lambda *a: sum(jnp.sum(e) for e in elem(*a))                 # noqa: E731
    g = jax.jit(jax.grad(cost, argnums=tuple(range(nd))))(*args)
    for n, gi in enumerate(g):
        gd = np.asarray(gi.data)
        assert np.all(np.isfinite(gd)), f"{kind}: d/d(arg {n}) has {np.count_nonzero(~np.isfinite(gd))} non-finite"
    f = jax.jit(elem)
    rng = np.random.default_rng(5)
    worst = 0.0
    for n in range(nd):
        x = np.asarray(args[n].data)
        gd = np.asarray(g[n].data)
        # smooth points: a gradient well above zero; positive salinity for the EOS
        cand = np.argwhere(np.abs(gd) > 1e-6 * np.max(np.abs(gd)))
        if kind == "eos":
            s = np.asarray(args[1].data)
            cand = np.array([c for c in cand if s[tuple(c)] > 1.0])
        for c in cand[rng.choice(len(cand), size=min(3, len(cand)), replace=False)]:
            c = tuple(c)
            h = 1e-4 * max(abs(x[c]), 1e-3)
            xp, xm = x.copy(), x.copy()
            xp[c] += h
            xm[c] -= h
            ap, am = list(args), list(args)
            ap[n] = type(args[n])(jnp.asarray(xp), args[n].name, tiled=args[n].tiled, _dims=args[n].dims)
            am[n] = type(args[n])(jnp.asarray(xm), args[n].name, tiled=args[n].tiled, _dims=args[n].dims)
            fp, fm = f(*ap), f(*am)
            fd = sum(float(np.sum(np.asarray(p) - np.asarray(m))) for p, m in zip(fp, fm)) / (2 * h)
            rel = abs(fd - gd[c]) / max(abs(gd[c]), 1e-300)
            worst = max(worst, rel)
            assert rel < 1e-5, f"{kind} arg {n} at {c}: FD {fd:.12e} vs AD {gd[c]:.12e} (rel {rel:.2e})"
    print(f"{exp}:{kind}: gradients finite on all lanes; worst FD rel. error {worst:.2e}")


# ---------------------------------------------------------------------------------------------------------------
# gates against the oracle's substep dumps of the registered runs (reference/reference_runs.py, kind jdon): the
# kernels on the model's own state at the dumped iterations (P02_rho_sigma_ivdc after the DO_OCEANIC_PHYS k loop,
# do_oceanic_phys.F:759-887; P03_mxlayer after CALC_OCE_MXLAYER). Parameters and grid from the same experiment's
# COL harness dump (INITIALISE_FIXED values; the harness overrides touch no field these routines read).

DUMP_RUNS = [("tutorial_baroclinic_gyre", "input"), ("global_ocean.90x40x15", "input"),
             ("tutorial_global_oce_optim", "input_ad")]


@functools.lru_cache(maxsize=None)
def dumpset(exp, inp):
    import sys
    sys.path.insert(0, str(cr.REPO / "reference"))
    import reference_runs as rr
    from mitjax.io.dump import DumpSet
    return DumpSet(rr.run_top(rr.find_run(exp, inp, "jdon")) / "dumps")


@pytest.mark.parametrize("exp,inp", DUMP_RUNS, ids=[e for e, _ in DUMP_RUNS])
def test_rhoInSitu_and_ivdc_vs_dumps(exp, inp):
    """FIND_RHO_2D (do_oceanic_phys.F:759-766: full range, kRef = k) == the dumped rhoInSitu, and CALC_IVDC
    (:867-875, k = Nr..2) on the dumped sigmaR == the dumped IVDConvCount, every dumped iteration, all points."""
    R, d = by_exp(exp), dumpset(exp, inp)
    sz, cfg = R.size, R.cfg
    grid, params, eos = R.commons()
    ij = cr._ij(sz)
    fa = lambda a, n: FArray(jnp.asarray(a), n, k=(1, sz.Nr), **ij)      # noqa: E731

    def run(theta, salt, totPhi, sigmaR, ivdc_prior, grid, params, eos):
        state = cr.Common(totPhiHyd=totPhi, IVDConvCount=ivdc_prior)
        rho = []
        for k in range(1, sz.Nr + 1):
            rho.append(find_rho.find_rho_2d(1 - sz.OLx, sz.sNx + sz.OLx, 1 - sz.OLy, sz.sNy + sz.OLy, k,
                                            level(theta, k), level(salt, k), level(ivdc_prior, k), k, cfg=cfg,
                                            grid=grid, params=params, eos=eos, state=state))
        for k in range(sz.Nr, 1, -1):
            state = state.replace(IVDConvCount=calc_ivdc.calc_ivdc(1 - sz.OLx, sz.sNx + sz.OLx, 1 - sz.OLy,
                                                                   sz.sNy + sz.OLy, k, sigmaR, 0.0, 0, grid=grid,
                                                                   state=state))
        return _stack(rho), state.IVDConvCount.data

    f = jax.jit(run)
    for it in d.iterations():
        P = lambda n: d.field(it, "P02_rho_sigma_ivdc", n)                # noqa: E731
        # theta, salt as DO_OCEANIC_PHYS leaves them (S04: FREEZE_SURFACE resets surface theta before the EOS
        # loop where allowFreezing; S00_begin differs there: 22 points in global_ocean, 10 in optim at it 1)
        S = lambda n: d.field(it, "S04_oceanic_phys", n)                  # noqa: E731
        prior = P("IVDConvCount").copy()
        prior[:, 1:] = np.nan                                             # levels 2..Nr must all be written
        rho, ivdc = f(fa(S("theta"), "theta"), fa(S("salt"), "salt"), fa(P("totPhiHyd"), "totPhiHyd"),
                      fa(P("sigmaR"), "sigmaR"), fa(prior, "IVDConvCount"), grid, params, eos)
        _check(f"rhoInSitu it {it}", rho, P("rhoInSitu"), R)
        _check(f"IVDConvCount it {it}", ivdc, P("IVDConvCount"), R)
        print(f"{exp} it {it}: convective points (IVDConvCount = 1): {int(np.sum(P('IVDConvCount') == 1.0))}")


def test_mxlayer_vs_dumps():
    """CALC_OCE_MXLAYER (method 1, tutorial_baroclinic_gyre) on the dumped rhoInSitu(kSrf), sigmaR, theta, salt ==
    the dumped hMixLayer after the call (P03), every dumped iteration."""
    exp, inp = DUMP_RUNS[0]
    R, d = by_exp(exp), dumpset(exp, inp)
    run, _ = drive_mxl(R)
    grid, params, eos = R.commons()
    sz = R.size
    ij = cr._ij(sz)
    f = jax.jit(run)
    for it in d.iterations():
        P = lambda n: d.field(it, "P02_rho_sigma_ivdc", n)                # noqa: E731
        S = lambda n: d.field(it, "S04_oceanic_phys", n)                  # noqa: E731
        rhoSurf = FArray(jnp.asarray(P("rhoInSitu")[:, 0]), "rhoSurf", **ij)
        f3 = lambda a, n: FArray(jnp.asarray(a), n, k=(1, sz.Nr), **ij)  # noqa: E731
        hPrior = FArray(jnp.asarray(P("hMixLayer")[:, 0]), "hMixLayer", **ij)
        aPrior = FArray(jnp.zeros_like(hPrior.data), "alphaLoc", **ij)
        _, h = f(rhoSurf, f3(P("sigmaR"), "sigmaR"), f3(S("theta"), "theta"), f3(S("salt"), "salt"), hPrior,
                 aPrior, grid, params, eos)
        want = d.field(it, "P03_mxlayer", "hMixLayer")[:, 0]
        _check(f"hMixLayer it {it}", h, want, R)
