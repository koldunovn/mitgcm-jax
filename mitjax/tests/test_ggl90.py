"""GGL90 gates (M3 sub-lane GGL90, plan Task 29): pkg/ggl90 against the gfortran oracle, bitwise.

1. Replay (reference/replay_ggl90, runs in reference/replay_ggl90/CURRENT): GGL90_READPARMS (every GGL90.h
   parameter as the Fortran read it), GGL90_INIT_VARIA (TKE, IDEMIX_E, IDEMIX_F_B/F_S from the run's files),
   GGL90_CALC for every flag case of the harness (vermix/code: mxlMaxFlag 0-3, Langmuir, calcMeanVertShear,
   mxlSurfFlag, both bottom conditions, GGL90_MISSING_HFAC_BUG; global_ocean.90x40x15/code: IDEMIX), GGL90_EXCHANGES,
   GGL90_CALC_DIFF, GGL90_CALC_VISC, GGL90_ADD_STOKESDRIFT; every point incl. halos, element equality + bit
   patterns + finite. IDEMIX with glibc_asin (mitjax/ops/libm.py), as in the model.
2. Dumps (jdon runs of vermix input.ggl90 / input.gglLC, global_ocean.90x40x15 input.idemix): GGL90_CALC teacher-
   forced at every dumped iteration vs P04_ggl90.
3. Negative controls, each measured to bite: re-associated b3d and mixing-length products (rounding only), a level shift in
   the mixing-length downward sweep, a 1-ulp change of GGL90ck, XLA's asin in IDEMIX.
4. Gradients: finite on every lane (land, halos, TKE = 0, zero wind), tangent vs adjoint dot test, FD at a smooth
   point.
All tier1x (mitjax/tests/manifest_ggl90.py). Fails (not skips) without the harness runs or dumps.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from mitjax.tests import ggl90_gate as gg
from mitjax.pkg.ggl90 import ggl90_calc as calc_mod
from mitjax.pkg.ggl90.ggl90_add_stokesdrift import ggl90_add_stokesdrift
from mitjax.pkg.ggl90.ggl90_calc_diff import ggl90_calc_diff
from mitjax.pkg.ggl90.ggl90_calc_visc import ggl90_calc_visc
from mitjax.pkg.ggl90.ggl90_exchanges import ggl90_exchanges
from mitjax.pkg.ggl90.ggl90_init_varia import ggl90_init_varia
from mitjax.pkg.ggl90.ggl90_readparms import ggl90_readparms

EXPS = ("vermix", "global_ocean.90x40x15")


def _cases(exp):
    return range(1, len(gg.replay(exp).inputs["cases"]) + 1)


# ---------------------------------------------------------------------------------------------------- 1. replay

@pytest.mark.parametrize("exp", EXPS)
def test_readparms_as_fortran(exp):
    """Every GGL90.h parameter the harness wrote after GGL90_READPARMS equals ggl90_readparms' value bit for bit."""
    rep = gg.replay(exp)
    ggl = ggl90_readparms(rep.e)
    names = {"GGL90mixingLengt": "GGL90mixingLengthMin", "calcMeanVertShea": "calcMeanVertShear"}
    diff = {}
    for rec, v in rep.parm.items():
        if rec.startswith("init_"):
            continue
        n = names.get(rec, rec)
        ours = float(getattr(ggl, n))
        if np.float64(ours).view(np.int64) != np.float64(v).view(np.int64):
            diff[n] = (ours, v)
    assert diff == {}
    assert len([r for r in rep.parm if not r.startswith("init_")]) >= 18


@pytest.mark.parametrize("exp", EXPS)
def test_init_varia_as_fortran(exp):
    rep = gg.replay(exp)
    sz = rep.size
    from mitjax.eesupp.exch_maps import load_maps
    from mitjax.eesupp.exchange import Exchanger
    from mitjax.pkg.rw.read_rec import RW
    ex = Exchanger(load_maps(exp)) if exp != "vermix" else None
    ggl = ggl90_readparms(rep.e)
    rw = RW(rep.rundir, 32, sz)
    out = ggl90_init_varia(ggl, cfg=rep.cfg, grid=rep.grid(), nIter0=0, pickupSuff=" ", ex=ex, rw=rw)
    res = {"TKE": gg.compare(out.GGL90TKE.data, rep.parm["init_TKE"])}
    if "init_E" in rep.parm:
        res["E"] = gg.compare(out.IDEMIX_E.data, rep.parm["init_E"])
        res["F_B"] = gg.compare(out.IDEMIX_F_B.data, rep.parm["init_F_B"])
        res["F_S"] = gg.compare(out.IDEMIX_F_S.data, rep.parm["init_F_S"])
        assert np.count_nonzero(rep.parm["init_F_B"]) > 100 and np.count_nonzero(rep.parm["init_F_S"]) > 100
    assert gg.bad(res) == {}, res


@pytest.mark.parametrize("exp", EXPS)
def test_calc_replay_bitwise(exp):
    """GGL90_CALC for every harness case: TKE, viscosities, diffusivity (and IDEMIX_E) bitwise on every point incl.
    halos."""
    fails = {}
    for ic in _cases(exp):
        res, out = gg.replay_case(exp, ic)
        if gg.bad(res):
            fails[ic] = gg.bad(res)
        assert np.count_nonzero(gg.replay(exp).out[f"TKE_c{ic}"] != gg.replay(exp).inputs["TKE"]) > 0
    assert fails == {}


def test_idemix_xla_asin_bites():
    """Negative control of the IDEMIX gates: XLA's arcsin in place of glibc_asin changes IDEMIX_E and TKE."""
    res, _ = gg.replay_case("global_ocean.90x40x15", 1, module=gg.with_xla_asin())
    n = {k: v[1] for k, v in res.items()}
    print("IDEMIX case 1 with XLA asin, differing points:", n)
    assert n["IDEMIX_E"] > 0 and n["GGL90TKE"] > 0


@pytest.mark.parametrize("exp", EXPS)
def test_exchanges_calc_diff_visc_stokes(exp):
    rep = gg.replay(exp)
    sz, x = rep.size, rep.inputs
    ncase = len(x["cases"])
    _, last = gg.replay_case(exp, ncase)
    res = {}
    if exp != "vermix":
        from mitjax.eesupp.exch_maps import load_maps
        from mitjax.eesupp.exchange import Exchanger
        ex = Exchanger(load_maps(exp))
    else:
        ex = None
    exch = ggl90_exchanges(cfg=rep.cfg, ggl=last, ex=ex)
    res["TKE_exch"] = gg.compare(exch.GGL90TKE.data, rep.out["TKE_exch"])
    if "E_exch" in rep.out:
        res["E_exch"] = gg.compare(exch.IDEMIX_E.data, rep.out["E_exch"])
    ggl = rep.ggl_case(x["cases"][-1])
    params = rep.params()
    kap = gg.f3("KappaRx", x["kappaRx"], sz)
    kap = ggl90_calc_diff(1-sz.OLx, sz.sNx+sz.OLx, 1-sz.OLy, sz.sNy+sz.OLy, 0, sz.Nr, kap, cfg=rep.cfg,
                          params=params, ggl=ggl)
    res["calcDiff"] = gg.compare(kap.data, rep.out["calcDiff"])
    zero = np.zeros((x["kappaRU"].shape[0], 1) + x["kappaRU"].shape[2:])
    kU = gg.f3("KappaRU", np.concatenate([x["kappaRU"], zero], axis=1), sz, sz.Nr+1)
    kV = gg.f3("KappaRV", np.concatenate([x["kappaRV"], zero], axis=1), sz, sz.Nr+1)
    for k in range(1, sz.Nr+1):
        kU, kV = ggl90_calc_visc(1-sz.OLx, sz.sNx+sz.OLx, 1-sz.OLy, sz.sNy+sz.OLy, k, kU, kV, params=params,
                                 ggl=ggl)
    res["calcViscU"] = gg.compare(kU.data, rep.out["calcViscU"])
    res["calcViscV"] = gg.compare(kV.data, rep.out["calcViscV"])
    if "stokesU" in rep.out:
        st = rep.state()
        g = rep.grid()
        gl = ggl.replace(useLANGMUIR=True)
        uR, vR = [], []
        for k in range(1, sz.Nr+1):
            u0 = gg.f2("uRes", -np.ones(x["sfU"].shape), sz)
            v0 = gg.f2("vRes", -np.ones(x["sfU"].shape), sz)
            u, v = ggl90_add_stokesdrift(u0, v0, gg.f2("uFld", x["uVel"][:, k-1], sz),
                                         gg.f2("vFld", x["vVel"][:, k-1], sz), k, cfg=rep.cfg, grid=g, ggl=gl,
                                         state=st)
            uR.append(np.asarray(u.data))
            vR.append(np.asarray(v.data))
        res["stokesU"] = gg.compare(np.stack(uR, 1), rep.out["stokesU"])
        res["stokesV"] = gg.compare(np.stack(vR, 1), rep.out["stokesV"])
    assert gg.bad(res) == {}, res


# ---------------------------------------------------------------------------------------------- 3. controls

def _planted_sum():
    """SQRTTWO*SQRTTKE/SQRT(...) re-associated as SQRTTWO*(SQRTTKE/SQRT(...)) in the initial mixing length (:358-360):
    a rounding-level change. (Measured not to bite: a commuted sum, IEEE addition is commutative, job 27840865; and
    KappaM/Pr as KappaM*(1/Pr), absorbed in vermix.)"""
    return gg.planted("ggl90_calc", "SQRTTWO * SQRTTKE[i, j, k]/jnp.sqrt(MAX(Nsquare[i, j, k], GGL90eps, p=\"b\")) * mskLoc",
                      "SQRTTWO * (SQRTTKE[i, j, k]/jnp.sqrt(MAX(Nsquare[i, j, k], GGL90eps, p=\"b\"))) * mskLoc")


def _planted_b3d():
    """b3d = 1 - c3d - a3d re-associated as 1 - (c3d + a3d) (:748): rounding level; bites in the vermix dumps at
    iteration 0 (9 points) and in the replay (measured 2026-10-02)."""
    return gg.planted("ggl90_calc", "b3d = b3d.at[i, j, k].set(1.0 - c3d[i, j, k] - a3d[i, j, k]",
                      "b3d = b3d.at[i, j, k].set(1.0 - (c3d[i, j, k] + a3d[i, j, k])")


def _planted_sweep():
    ml = gg.planted("ggl90_mixinglength", '"drF_km1": grid.drF[k-1]', '"drF_km1": grid.drF[k]')
    return gg.planted("ggl90_calc", "from mitjax.pkg.ggl90.ggl90_mixinglength import ggl90_mixinglength",
                      "from mitjax.pkg.ggl90.ggl90_mixinglength import ggl90_mixinglength  # rebound",
                      rebind={"ggl90_mixinglength": ml.ggl90_mixinglength})


def test_negative_controls_bite():
    """Each planted error changes the replay result (measured, vermix case 1 and 2; the 1-ulp GGL90ck on the
    IDEMIX case)."""
    for ic in (1, 2):
        res, _ = gg.replay_case("vermix", ic, module=_planted_sum())
        assert res["GGL90TKE"][1] > 0, ("re-associated mixing length", ic, res)
        res, _ = gg.replay_case("vermix", ic, module=_planted_b3d())
        assert res["GGL90TKE"][1] > 0, ("b3d", ic, res)
        res, _ = gg.replay_case("vermix", ic, module=_planted_sweep())
        assert sum(v[1] for v in res.values()) > 0, ("sweep level", ic, res)
    res, _ = gg.replay_case("global_ocean.90x40x15", 1,
                            ggl_override=lambda g: g.replace(GGL90ck=np.nextafter(g.GGL90ck, 1.0)))
    assert res["GGL90TKE"][1] > 0, res


# ------------------------------------------------------------------------------------------------- 2. dumps

@pytest.mark.parametrize("exp,inp", gg.DUMP_VARIANTS)
def test_calc_dump_bitwise(exp, inp):
    """GGL90_CALC teacher-forced at every dumped iteration vs P04_ggl90 (IDEMIX with glibc_asin); the control
    bites: b3d re-associated (vermix, iteration 0; a mixing-length error is absorbed there: L = GGL90mixingLengthMin
    in the near-minimum TKE column) and XLA's arcsin (idemix)."""
    ds = gg.dumpset(exp, inp)
    fails = {}
    for it in ds.iterations():
        res, _ = gg.dump_case(exp, inp, it)
        if gg.bad(res):
            fails[it] = gg.bad(res)
    assert fails == {}
    res, _ = gg.dump_case(exp, inp, ds.iterations()[0],
                          module=_planted_b3d() if exp == "vermix" else gg.with_xla_asin())
    assert sum(v[1] for v in res.values()) > 0


# --------------------------------------------------------------------------------------------- 4. gradients

def _vermix_fn(case=2):
    rep = gg.replay("vermix")
    cs = rep.inputs["cases"][case - 1]
    ggl0 = rep.ggl_case(cs)
    grid, params, st0 = rep.grid(), rep.params(), rep.state()
    sz = rep.size
    sig0 = gg.f3("sigmaR", rep.inputs["sigmaR"], sz)
    x0 = {"TKE": ggl0.GGL90TKE.data, "uVel": st0.uVel.data, "vVel": st0.vVel.data, "sigmaR": sig0.data,
          "sfU": st0.surfaceForcingU.data, "sfV": st0.surfaceForcingV.data}

    def f(x):
        ggl = ggl0.replace(GGL90TKE=gg.wrap("GGL90TKE", x["TKE"], sz))
        st = gg.Common(uVel=gg.f3("uVel", x["uVel"], sz), vVel=gg.f3("vVel", x["vVel"], sz),
                       surfaceForcingU=gg.f2("surfaceForcingU", x["sfU"], sz),
                       surfaceForcingV=gg.f2("surfaceForcingV", x["sfV"], sz))
        out = calc_mod.ggl90_calc(gg.f3("sigmaR", x["sigmaR"], sz), 0.0, 0, cfg=rep.cfg, grid=grid, params=params,
                                  ggl=ggl, state=st)
        return {n: getattr(out, n).data for n in gg.OUT_FIELDS}
    return f, x0


def _rand_like(tree, seed):
    leaves, tdef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    return jax.tree_util.tree_unflatten(tdef, [jax.random.normal(k, l.shape, l.dtype) for k, l in zip(keys, leaves)])


def _vdot(a, b):
    return sum(jnp.vdot(x, y) for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


@pytest.mark.parametrize("case", (1, 2, 3))
def test_gradient_finite_and_dot_test(case):
    """Reverse-mode gradient finite on every lane (inputs hold TKE = 0, zero wind, unstable and neutral sigmaR); the
    tangent-adjoint dot test <J v, w> = <v, J^T w> to 1e-12 relative."""
    f, x0 = _vermix_fn(case)
    w = _rand_like(jax.eval_shape(f, x0), 1)
    v = _rand_like(x0, 2)
    y, vjp = jax.vjp(f, x0)
    (gx,) = vjp(w)
    assert all(bool(jnp.all(jnp.isfinite(a))) for a in jax.tree_util.tree_leaves(gx))
    _, jv = jax.jvp(f, (x0,), (v,))
    lhs, rhs = float(_vdot(jv, w)), float(_vdot(v, gx))
    assert abs(lhs - rhs) <= 1e-12 * max(abs(lhs), abs(rhs)), (lhs, rhs)


def test_gradient_fd_smooth_point():
    """Central finite differences along a direction on the interior vs the tangent, at a smooth point (TKE and
    sigmaR moved away from their switches): the relative error has a plateau below 1e-6 over an h sweep."""
    f, x0 = _vermix_fn(1)
    x0 = dict(x0)
    x0["TKE"] = jnp.abs(x0["TKE"]) + 1e-4
    x0["sigmaR"] = -jnp.abs(x0["sigmaR"]) - 1e-4
    v = _rand_like(x0, 3)
    v = {k: a * 1e-3 * (jnp.abs(x0[k]) + 1e-6) for k, a in v.items()}
    J = lambda x: sum(jnp.sum(a) for a in jax.tree_util.tree_leaves(f(x)))
    _, tan = jax.jvp(J, (x0,), (v,))
    errs = []
    for h in (1e-2, 1e-3, 1e-4):
        xp = {k: x0[k] + h*v[k] for k in x0}
        xm = {k: x0[k] - h*v[k] for k in x0}
        fd = (J(xp) - J(xm))/(2*h)
        errs.append(abs(float(fd - tan))/max(abs(float(tan)), 1e-300))
    print("FD rel. errors", errs, "tangent", float(tan))
    assert min(errs) < 1e-6, errs


# ------------------------------------------------------------------------------------------ 5. INI_PARMS aliases

@pytest.mark.parametrize("exp,inp", (("vermix", "input.ggl90"), ("global_ocean.90x40x15", "input.idemix")))
def test_ini_parms_values_as_fortran(exp, inp):
    """The PARAMS.h values GGL90 reads, from our INI_PARMS (vermix: the viscAz / diffKzT / diffKzS aliases,
    ini_parms.F:523, :570, :602), equal the harness's dump of the Fortran's values bit for bit; a changed diffKzS
    reaches diffKrNrS (the alias is live, not the diffKrNrT fallback of :632, which has the same value in vermix)."""
    p = gg.model_params(exp, inp)
    hg = gg.replay(exp).g
    for n in ("gravity", "recip_rhoConst", "dTtracerLev", "diffKrNrS", "viscArNr"):
        v = np.asarray(getattr(getattr(p, n), "data", getattr(p, n)), np.float64)
        assert np.array_equal(v.view(np.int64), np.asarray(hg[n], np.float64).view(np.int64)), n
    if exp == "vermix":
        from mitjax.drivers.model import with_namelist
        from mitjax.model.src.ini_parms import ini_parms, ini_parms_dyn
        from mitjax.model.src.ini_parms_tracer import ini_parms_tracer
        e = with_namelist(gg.experiment(exp, inp), {("data", "PARM01", "diffKzS"): 3.0e-5})
        prm = ini_parms(e, None)
        q = ini_parms_tracer(e, ini_parms_dyn(e, prm.grid, prm.time, prm.init), prm.time, prm.init)
        assert np.all(np.asarray(q.diffKrNrS.data) == 3.0e-5)
