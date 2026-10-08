"""PTRACERS lane gates (M2, plan Task 26): pkg/ptracers, the convective-adjustment family, CG2D_NSA, SWFRAC,
KPP_CALC_DUMMY, COST_TRACER.

1. Replay (reference/replay_ptracers, tutorial_tracer_adjsens/code_ad, synthetic inputs on the real grid incl. land
   and halos; reference/replay_ptracers/CURRENT): CONVECTIVE_ADJUSTMENT and CONVECTIVE_ADJUSTMENT_INI (theta, salt
   and the passive tracer through PTRACERS_CONVECT), PTRACERS_APPLY_FORCING (every level), the experiment's own
   PTRACERS_FORCING_SURF (two index ranges), KPP_CALC_DUMMY (with CALC_3D_DIFFUSIVITY's ALLOW_3D_DIFFKR arm),
   COST_TRACER, CG2D_NSA (zero and random first guess: solution, normalised rhs, residuals, iteration count) and
   SWFRAC: bitwise on every point the Fortran writes (element equality, bit patterns, finite), under the gate XLA
   flags, float parameters traced.
2. Substep dumps (registered jdon runs): the convective adjustment of tutorial_tracer_adjsens teacher-forced from
   S05_thermodynamics_sync (theta, salt) and T04_ptracers_integrate (pTracer) against S15_tracers_correction, steps
   0-2; PTRACERS_FORCING_SURF of the three experiments (each build's own version) against the surfaceForcingPTr of
   T04 (inputs: pTracer of S00_begin, surfaceForcingS of S04_oceanic_phys).
3. PTRACERS_READPARMS + PTRACERS_INIT_FIXED against the PTRACERS_CHECK printout of each oracle run (the PARAMS.h
   values they read teacher-forced from the same printout).
4. Negative controls (planted errors, each measured to bite), gradient finiteness on every lane and FD checks.
Tier 1x (a few minutes on CPU).
"""

import functools
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax.tests import ptracers_gate as pg

RUN = ("tutorial_tracer_adjsens", "input_ad")
EXPS = (("tutorial_tracer_adjsens", "input_ad"), ("tutorial_tracer_adjsens", "input_ad.som81"),
        ("tutorial_global_oce_latlon", "input"), ("tutorial_advection_in_gyre", "input"))
IDS = [f"{e}/{i}" for e, i in EXPS]


@functools.lru_cache(maxsize=None)
def replay():
    runs = {(e, i): rd for e, i, rd in pg.current_runs()}
    return pg.Replay(*RUN, runs[RUN])


@functools.lru_cache(maxsize=None)
def exchanger():
    from mitjax.eesupp.exch_maps import build_maps
    from mitjax.eesupp.exchange import Exchanger
    ds, _, _ = oracle(*RUN)
    return Exchanger(build_maps(ds))


@functools.lru_cache(maxsize=None)
def oracle(exp, inp):
    from mitjax.tests import rstar_gate as rsg
    return rsg.oracle(exp, inp, "jdon")


def _ok(r):
    n, tot, fin = r
    return n == 0 and fin


# ---------------------------------------------------------------------------------------------------------------
# 1. replay drivers

def conv_driver(mod_adj=None, mod_ini=None):
    """jitted (theta, salt, pTracer, commons..., myTime) -> outputs of CONVECTIVE_ADJUSTMENT(_INI) for all tiles."""
    from mitjax.model.src import convective_adjustment as ca
    from mitjax.model.src import convective_adjustment_ini as ci
    R = replay()
    fa = (mod_adj or ca).convective_adjustment
    fi = (mod_ini or ci).convective_adjustment_ini

    def run(which, t, s, p, grid, params, eos, ptr, tc):
        st = pg.state_of(R.experiment.cfg, theta=R.f3(t, "theta"), salt=R.f3(s, "salt"))
        st, ptf = (fa if which == "A" else fi)(tc, 3, cfg=R.cfg, grid=grid, params=params, eos=eos, state=st,
                                                 ptr=ptr, ptf=R.ptf(p3=p))
        return st.theta.data, st.salt.data, ptf.pTracer[0].data
    return {w: jax.jit(functools.partial(run, w)) for w in ("A", "I")}


def conv_results(drv, tshift=0.0):
    R = replay()
    grid, params, eos, ptr, _ = R.commons()
    I = R.inputs
    out = {}
    oa = drv["A"](I["tA"], I["sA"], I["pA"], grid, params, eos, ptr, jnp.float64(R.g["tCall"] + tshift))
    oi = drv["I"](I["tB"], I["sB"], I["pB"], grid, params, eos, ptr, jnp.float64(0.))
    for tag, o in (("convA", oa), ("convI", oi)):
        for v, n in zip(o, ("t", "s", "p")):
            out[f"{tag}_{n}"] = pg.bit_equal(v, R.out[f"{tag}_{n}"])
    return out


def leaf_results(mods=None):
    """{record: bit_equal} for PTRACERS_APPLY_FORCING, PTRACERS_FORCING_SURF, KPP_CALC_DUMMY, COST_TRACER, SWFRAC."""
    from mitjax.model.src import swfrac as swm
    from mitjax.pkg.cost import cost_tracer as ctm
    from mitjax.pkg.kpp import kpp_calc_dummy as kpm
    from mitjax.pkg.ptracers import ptracers_apply_forcing as afm
    from mitjax.pkg.ptracers.ptracers_forcing_surf import routine_of_build
    mods = mods or {}
    af = mods.get("apply_forcing", afm).ptracers_apply_forcing
    fs = mods.get("forcing_surf", None)
    fs = fs.ptracers_forcing_surf if fs is not None else None
    kp = mods.get("kpp", kpm).kpp_calc_dummy
    ct = mods.get("cost", ctm).cost_tracer
    sw = mods.get("swfrac", swm).swfrac
    R = replay()
    cfg, sz, I = R.cfg, R.size, R.inputs
    fs = fs or routine_of_build(cfg)
    grid, params, eos, ptr, _ = R.commons()
    res = {}

    def apply(g3, sfp, grid, params, ptr):
        out = []
        for k in range(1, sz.Nr+1):
            g2 = af(R.f2(g3[:, k-1], "gPtracer"), R.f2(sfp, "surfForcPtr"), 0, sz.sNx+1, 0, sz.sNy+1, k, 1, 0., 0,
                    cfg=cfg, grid=grid, params=params, ptr=ptr)
            out.append(g2.data)
        return jnp.stack(out, axis=1)
    res["gForc"] = pg.bit_equal(jax.jit(apply)(I["gPrior"], I["sfP"], grid, params, ptr), R.out["gForc"])

    def surf(sfS, pTr, prior, relax, grid, params, ptr, full):
        ptf = R.ptf(p3=pTr, sf=prior)
        ff = SimpleNamespace(surfaceForcingS=R.f2(sfS, "surfaceForcingS"))
        rng = (1-sz.OLx, sz.sNx+sz.OLx, 1-sz.OLy, sz.sNy+sz.OLy) if full else (1, sz.sNx, 1, sz.sNy)
        ptf = fs(R.f2(relax, "relaxForcingS"), *rng, 0., 0, cfg=cfg, grid=grid, params=params, ptr=ptr, ptf=ptf,
                 ff=ff)
        return ptf.surfaceForcingPTr[0].data
    for full, n in ((True, "sfPTr_full"), (False, "sfPTr_int")):
        o = jax.jit(functools.partial(surf, full=full))(I["sfS"], I["pTr"], I["sfPrior"], I["relaxS"], grid,
                                                          params, ptr)
        res[n] = pg.bit_equal(o, R.out[n])

    def kpp(ivdc, dkr, prior3, prior2, grid, params):
        st = pg.state_of(R.experiment.cfg, IVDConvCount=R.f3(ivdc, "IVDConvCount"), diffKr=R.f3(dkr, "diffKr"))
        k = {"KPPhbl": R.f2(prior2, "KPPhbl"), "KPPfrac": R.f2(prior2, "KPPfrac")}
        for n in ("KPPghat", "KPPviscAz", "KPPdiffKzS", "KPPdiffKzT"):
            k[n] = R.f3(prior3, n)
        k = kp(0., 0, cfg=cfg, grid=grid, params=params, state=st, kpp=k)
        return {n: v.data for n, v in k.items()}
    o = jax.jit(kpp)(I["ivdc"], I["dKr"], I["gPrior"], I["kppPrior"], grid, params)
    res.update({n: pg.bit_equal(o[n], R.out[n]) for n in o})

    ex = exchanger()       # built outside the jit: the lru_cache must not keep traced arrays (UnexpectedTracerError)

    def cost(objf, pTr, grid, params, ptr):
        return ct(objf, cfg=cfg, grid=grid, params=params, ptr=ptr, ptf=R.ptf(p3=pTr), ex=ex)
    res["objf_tracer"] = pg.bit_equal(jax.jit(cost)(jnp.asarray(I["objfPrior"]), I["pTr"], grid, params, ptr),
                                      R.out["objf_tracer"])
    o = jax.jit(lambda s, f: sw(pg.replay_io.NSW, f, s, 0., 0))(jnp.asarray(I["swdk"]), jnp.float64(I["swFact"][0]))
    res["swdk"] = pg.bit_equal(o, R.out["swdk"])
    return res


def dst3fl_r_results(mod=None):
    """GAD_DST3FL_ADV_R on every level as the harness calls it (tracer tA, wFld wF(k), rTrans rTr(k), dTarg =
    dTtracerLev(k), prior gPrior(k)) vs the record dst3flR."""
    from mitjax.pkg.generic_advdiff import gad_dst3fl_adv_r as dm
    from mitjax.pkg.generic_advdiff.gad_calc_rhs import gad_kernel_cfg
    f = (mod or dm).gad_dst3fl_adv_r
    R = replay()
    sz, I = R.size, R.inputs
    grid, params, _, _, _ = R.commons()
    kc = gad_kernel_cfg(R.experiment.cfg)

    def run(t, w, r, prior, grid, params):
        out = []
        for k in range(1, sz.Nr+1):
            wT = f(k, params.dTtracerLev[k], R.f2(r[:, k-1], "rTrans"), R.f2(w[:, k-1], "wFld"), R.f3(t, "tracer"),
                   R.f2(prior[:, k-1], "wT"), cfg=kc, grid=grid)
            out.append(wT.data)
        return jnp.stack(out, axis=1)
    o = jax.jit(run)(I["tA"], I["wF"], I["rTr"], I["gPrior"], grid, params)
    return {"dst3flR": pg.bit_equal(o, R.out["dst3flR"])}


def test_replay_gad_dst3fl_adv_r():
    """GAD_DST3FL_ADV_R (PTRACERS lane, the vertical kernel of scheme 33 in GAD_ADVECTION) bitwise on every level and
    point; the synthetic CFL numbers span 0 to beyond 1 and the limiter switches."""
    res = dst3fl_r_results()
    assert all(_ok(r) for r in res.values()), res
    R = replay()
    assert np.count_nonzero(R.out["dst3flR"] != R.inputs["gPrior"]) > 10000


def cg2d_nsa_results(mod=None):
    from mitjax.model.src import cg2d_nsa as cnm
    f = (mod or cnm).cg2d_nsa
    R = replay()
    cfg, I = R.cfg, R.inputs
    _, params, _, _, cg2dh = R.commons()
    ex = exchanger()
    run = jax.jit(lambda b, x, c, p: f(R.f2(b, "cg2d_b"), R.f2(x, "cg2d_x"), 200, -1, cfg=cfg, cg2dh=c, params=p,
                                       ex=ex))
    res = {}
    for n, x0 in ((1, np.zeros_like(I["cgX0"])), (2, I["cgX0"])):
        b, x, fr, mr, lr, ni, nm, printed = run(I["cgB"], x0, cg2dh, params)
        want = R.out["cg_res"][5*n-5:5*n]
        # the printed rhsMax (cg2d_nsa.F:140-150): the largest |normalised rhs| of the interior (no ties, no NaN)
        sz = R.size
        bi = np.asarray(R.out[f"cg_b{n}"])[:, sz.OLy:sz.OLy+sz.sNy, sz.OLx:sz.OLx+sz.sNx]
        res[f"rhsMax{n}"] = pg.bit_equal(np.array([printed["rhsMax"]]), np.array([np.max(np.abs(bi))]))
        res[f"cg_x{n}"] = pg.bit_equal(x.data, R.out[f"cg_x{n}"])
        res[f"cg_b{n}"] = pg.bit_equal(b.data, R.out[f"cg_b{n}"])
        res[f"cg_res{n}"] = pg.bit_equal(np.array([fr, mr, lr, float(ni), float(nm)]), want)
    return res


def test_replay_convective_adjustment():
    """CONVECTIVE_ADJUSTMENT and CONVECTIVE_ADJUSTMENT_INI bitwise (theta, salt, pTracer, all points); convection
    fires on a large part of the synthetic columns."""
    res = conv_results(conv_driver())
    assert all(_ok(r) for r in res.values()), res
    R = replay()
    for tag, src in (("convA", "A"), ("convI", "B")):        # every field changed on many points (incl. pTracer)
        for f in ("t", "s", "p"):
            assert np.count_nonzero(R.out[f"{tag}_{f}"] != R.inputs[f"{f}{src}"]) > 10000, (tag, f)


def test_replay_leaf_routines():
    """PTRACERS_APPLY_FORCING, PTRACERS_FORCING_SURF (tracer_adjsens's own), KPP_CALC_DUMMY, COST_TRACER, SWFRAC
    bitwise."""
    res = leaf_results()
    assert all(_ok(r) for r in res.values()), res


def test_replay_cg2d_nsa():
    """CG2D_NSA bitwise: solution (halos incl.), normalised rhs, first/min/last residual, iterations (164, 189)."""
    res = cg2d_nsa_results()
    assert all(_ok(r) for r in res.values()), res
    R = replay()
    assert R.out["cg_res"][3] > 100 and R.out["cg_res"][8] > 100


# (relpath, old, new, which gate) -- each measured to make its gate fail
PLANTS = (
    ("mitjax/model/src/convective_weights.py", "safe_div(d2, dS, unstable)", "safe_div(d1, dS, unstable)", "conv"),
    ("mitjax/model/src/convectively_mixtracer.py", "- weightB[i, j]*delTrac", "+ weightB[i, j]*delTrac", "conv"),
    ("mitjax/pkg/ptracers/ptracers_apply_forcing.py", "* grid.recip_drF[k]*grid.recip_hFacC[i, j, k])",
     "* grid.recip_hFacC[i, j, k])", "apply_forcing"),
    ("mitjax/pkg/kpp/kpp_calc_dummy.py", 'kpp["KPPhbl"].at[i, j].set(1.0)', 'kpp["KPPhbl"].at[i, j].set(1.0+2.**-52)',
     "kpp"),
    ("mitjax/pkg/cost/cost_tracer.py", "* g.rA[i, j]*g.drF[k]*params.dTtracerLev[k])",
     "* (g.rA[i, j]*g.drF[k])*params.dTtracerLev[k])", "cost"),
    ("mitjax/model/src/swfrac.py", "deep = facz < -200.", "deep = facz <= -200.", "swfrac"),
    ("mitjax/pkg/generic_advdiff/gad_dst3fl_adv_r.py", "wCFL = jnp.abs(wLoc*dTarg*recip_drC[k])",
     "wCFL = jnp.abs(wLoc*dTarg*recip_drC[k+1])", "dst3flR"),
    ("mitjax/model/src/cg2d_nsa.py", "cg2d_s = EXCH_XY_RL(cg2d_s, ex=ex)                                      # :304",
     "cg2d_s = cg2d_s                                                        # :304", "cg2d"),
)


@pytest.mark.parametrize("relpath,old,new,which", PLANTS, ids=[p[0].split("/")[-1] + ":" + p[3] for p in PLANTS])
def test_negative_controls(relpath, old, new, which):
    """Each planted error makes its replay gate fail (number of differing points > 0)."""
    mod = pg.planted(relpath, old, new)
    if which == "conv":
        from mitjax.model.src import convective_adjustment as ca
        from mitjax.model.src import convective_adjustment_ini as ci
        # the planted leaf is reached through copies of the two drivers that call the planted module
        name = "convective_weights" if "weights" in relpath else "convectively_mixtracer"
        import sys
        sys.modules[mod.__name__] = mod
        a = pg.planted("mitjax/model/src/convective_adjustment.py", f"from mitjax.model.src.{name} import {name}",
                       f"{name} = __import__('sys').modules['{mod.__name__}'].{name}")
        i = pg.planted("mitjax/model/src/convective_adjustment_ini.py",
                       f"from mitjax.model.src.{name} import {name}",
                       f"{name} = __import__('sys').modules['{mod.__name__}'].{name}")
        res = conv_results(conv_driver(a, i))
    elif which == "cg2d":
        res = cg2d_nsa_results(mod)
    elif which == "dst3flR":
        res = dst3fl_r_results(mod)
    else:
        key = {"apply_forcing": "apply_forcing", "kpp": "kpp", "cost": "cost", "swfrac": "swfrac"}[which]
        res = leaf_results({key: mod})
        want = {"apply_forcing": ("gForc",), "kpp": ("KPPhbl",), "cost": ("objf_tracer",),
                "swfrac": ("swdk",)}[which]
        res = {k: res[k] for k in want}
    bad = sum(r[0] for r in res.values())
    assert bad > 0, (which, res)


def test_negative_control_clock():
    """A planted cAdjFreq = 2*deltaTClock at an odd clock step (DIFFERENT_MULTIPLE false: with cAdjFreq = deltaTClock
    every time is within half a step of a multiple) leaves the fields unadjusted: the gate bites."""
    R = replay()
    grid, params, eos, ptr, _ = R.commons()
    I = R.inputs
    drv = conv_driver()
    p2 = params.replace(cAdjFreq=2.0*params.cAdjFreq)
    o = drv["A"](I["tA"], I["sA"], I["pA"], grid, p2, eos, ptr, jnp.float64(R.g["tCall"]))
    assert pg.bit_equal(o[0], R.out["convA_t"])[0] > 10000
    assert pg.bit_equal(o[0], I["tA"])[0] == 0


# ---------------------------------------------------------------------------------------------------------------
# 2. substep dumps

def _dump(exp, inp, it, stage, name):
    ds, _, _ = oracle(exp, inp)
    return ds.field(it, stage, name)


@pytest.mark.parametrize("inp", ("input_ad", "input_ad.som81"))
def test_dump_convective_adjustment_tracer_adjsens(inp):
    """tutorial_tracer_adjsens (input_ad, and input_ad.som81 with staggerTimeStep: S14 instead of S05; the harness
    dump of input_ad's build serves both, same code_ad build and the same PARM01/PARM03 values the routine reads):
    TRACERS_CORRECTION_STEP's CONVECTIVE_ADJUSTMENT (forward_step.F:1025) teacher-forced
    from S05 (theta, salt after THERMODYNAMICS) and T04 (pTracer after PTRACERS_INTEGRATE): theta, salt, pTracer of
    S15 bitwise at the dumped steps, every point. The clock: myTime = startTime + (it+1)*deltaTClock at :1025 (after
    the iteration shift of forward_step.F:807). hFacC from S07_update_rstar_T of the step (UPDATE_R_STAR(.TRUE.) at
    forward_step.F:839 rescales hFacC in this r* run before the correction step; S00's hFacC differs by 1 ulp in the
    weights at steps 1-2, measured); the PARAMS.h/EOS.h/GRID.h scalars
    and vectors from the same build's harness dump (pt_grid.bin, cAdjFreq as INI_PARMS set it: ini_parms.F:1104-1105);
    PTRACERS_StepFwd = .TRUE. (PTRACERS_INIT_VARIA). Negative control: a planted cAdjFreq = 2*deltaTClock (no
    adjustment at odd steps) bites."""
    from mitjax.model.src.convective_adjustment import convective_adjustment
    R = replay()
    exp = RUN[0]
    ds, its, _ = oracle(exp, inp)
    tstage = "S05_thermodynamics_sync" if any(s == "S05_thermodynamics_sync" for (_, s, _) in ds.keys(its[0])) \
        else "S14_thermodynamics_stagger"          # som81: staggerTimeStep, THERMODYNAMICS at forward_step.F:1005
    grid, params, eos, ptr, _ = R.commons()
    ptr = ptr.replace(static=dict(PTRACERS_StepFwd=(True,)))                  # ptracers_init_varia.F:45

    def run(t, s, p, hFacC, grid, params, eos, ptr, tc):
        g = grid.replace(hFacC=R.f3(hFacC, "hFacC"))
        st = pg.state_of(R.experiment.cfg, theta=R.f3(t, "theta"), salt=R.f3(s, "salt"))
        st, ptf = convective_adjustment(tc, 0, cfg=R.cfg, grid=g, params=params, eos=eos, state=st, ptr=ptr,
                                        ptf=R.ptf(p3=p))
        return st.theta.data, st.salt.data, ptf.pTracer[0].data
    f = jax.jit(run)
    dT = float(R.g["deltaTClock"])
    res, ctrl, changed = {}, 0, 0
    for it in its:
        t = _dump(exp, inp, it, tstage, "theta")
        s = _dump(exp, inp, it, tstage, "salt")
        p = _dump(exp, inp, it, "T04_ptracers_integrate", "pTracer_01")
        h = _dump(exp, inp, it, "S07_update_rstar_T", "hFacC")
        tc = float(R.g["startTime"]) + (it + 1)*dT
        out = f(t, s, p, h, grid, params, eos, ptr, jnp.float64(tc))
        for v, n in zip(out, ("theta", "salt", "pTracer_01")):
            res[(it, n)] = pg.bit_equal(v, _dump(exp, inp, it, "S15_tracers_correction", n))
        changed += int(np.count_nonzero(_dump(exp, inp, it, "S15_tracers_correction", "theta") != t))
        if (it + 1) % 2 == 1:                       # planted cAdjFreq = 2*deltaTClock: no adjustment at odd steps
            out = f(t, s, p, h, grid, params.replace(cAdjFreq=2.0*params.cAdjFreq), eos, ptr, jnp.float64(tc))
            ctrl += pg.bit_equal(out[0], _dump(exp, inp, it, "S15_tracers_correction", "theta"))[0]
    assert all(_ok(r) for r in res.values()), res
    assert changed > 0 and ctrl > 0, (changed, ctrl)


def _mnc_monitor(e, name="monitor_mnc", default=True):
    """MNC_PARAMS.h monitor_mnc (pickup_write_mnc): data.mnc's value, else mnc_readparms.F:101 .TRUE. (:97 .FALSE.),
    as monitor_gate.make_cfg."""
    from mitjax.params_io import RunParams
    rp = RunParams(e.run)
    return bool(rp.get("data.mnc", "MNC_01", name)) if rp.has("data.mnc", "MNC_01", name) else default


def _ptr_of(exp, inp):
    """PtracersParams of PTRACERS_READPARMS + PTRACERS_INIT_FIXED, PARAMS.h inputs from the oracle printout."""
    from mitjax.io import stdout as so
    from mitjax.pkg.ptracers.ptracers_init_fixed import ptracers_init_fixed
    from mitjax.pkg.ptracers.ptracers_readparms import ptracers_readparms
    from mitjax.tests.col_replay import experiment
    _, _, rd = oracle(exp, inp)
    d = so.parameter_dict(so.parameters(so.read_stdout(rd / "output.txt")))
    num = lambda n: [so.fortran_number(v) for v in d[n].values()]
    e = experiment(exp, inp)
    pp = dict(baseTime=num("baseTime")[0], saltAdvScheme=int(num("saltAdvScheme")[0]), diffKhS=num("diffKhS")[0],
              diffK4S=num("diffK4S")[0], diffKrNrS=num("diffKrNrS"), useGMRedi=e.cfg.use_flag("useGMRedi"),
              useDOWN_SLOPE=e.cfg.use_flag("useDOWN_SLOPE"), useKPP=e.cfg.use_flag("useKPP"),
              doAB_onGtGs=d["doAB_onGtGs"].values()[0] == "T", dTtracerLev=num("dTtracerLev"),
              monitorFreq=num("monitorFreq")[0], useMNC=e.cfg.use_flag("useMNC"),
              monitor_mnc=_mnc_monitor(e), pickup_write_mnc=_mnc_monitor(e, "pickup_write_mnc", False))
    ptr = ptracers_readparms(e, pp)
    ptr = ptracers_init_fixed(ptr, cfg=e.cfg, multiDimAdvection=d["multiDimAdvection"].values()[0] == "T")
    nIter0 = int(num("nIter0")[0])
    return ptr.replace(static=dict(PTRACERS_StepFwd=(True,) * ptr.PTRACERS_num,          # ptracers_init_varia.F:45
                                   PTRACERS_startAB=tuple(nIter0 - ptr.PTRACERS_Iter0    # ptracers_init_varia.F:46
                                                          for _ in range(ptr.PTRACERS_num)))), d


@pytest.mark.parametrize("exp,inp", EXPS, ids=IDS)
def test_ptracers_readparms_vs_printout(exp, inp):
    """PTRACERS_READPARMS + PTRACERS_INIT_FIXED: every PTRACERS_CHECK value the port keeps equals the printout
    (logicals and integers exactly, reals to the 16 printed digits, which hold these values exactly)."""
    from mitjax.io import stdout as so
    ptr, d = _ptr_of(exp, inp)
    n = ptr.PTRACERS_numInUse
    assert int(so.fortran_number(d["PTRACERS_numInUse"].values()[0])) == n
    L = lambda name: [v == "T" for v in d[name].values()][:n]
    for name in ("PTRACERS_ImplVertAdv", "PTRACERS_MultiDimAdv", "PTRACERS_SOM_Advection", "PTRACERS_AdamsBashGtr",
                 "PTRACERS_AdamsBash_Tr", "PTRACERS_useGMRedi", "PTRACERS_useKPP", "PTRACERS_useDWNSLP"):
        assert list(getattr(ptr, name))[:n] == L(name), name
    assert [int(so.fortran_number(v)) for v in d["PTRACERS_advScheme"].values()][:n] == list(ptr.PTRACERS_advScheme)[:n]
    assert (d["PTRACERS_doAB_onGpTr"].values()[0] == "T") == ptr.PTRACERS_doAB_onGpTr
    assert (d["PTRACERS_addSrelax2EmP"].values()[0] == "T") == ptr.PTRACERS_addSrelax2EmP
    assert (d["PTRACERS_startAllTrc"].values()[0] == "T") == ptr.PTRACERS_startAllTrc
    f = lambda name: np.array([so.fortran_number(v) for v in d[name].values()])
    assert np.array_equal(f("PTRACERS_dTLev"), np.asarray(ptr.PTRACERS_dTLev.data))
    assert np.array_equal(f("PTRACERS_diffKh")[:n], np.array(ptr.PTRACERS_diffKh[:n]))
    assert np.array_equal(f("PTRACERS_diffK4")[:n], np.array(ptr.PTRACERS_diffK4[:n]))
    assert np.array_equal(f("PTRACERS_EvPrRn")[:n], np.array(ptr.PTRACERS_EvPrRn[:n]))
    Nr = len(ptr.PTRACERS_dTLev.data)
    assert np.array_equal(f("PTRACERS_diffKrNr")[:Nr], np.asarray(ptr.PTRACERS_diffKrNr[0].data))
    assert np.array_equal(f("PTRACERS_ref")[:Nr], np.asarray(ptr.PTRACERS_ref[0].data))


@pytest.mark.parametrize("exp,inp", EXPS, ids=IDS)
def test_dump_forcing_surf(exp, inp):
    """Each build's PTRACERS_FORCING_SURF (latlon: age-tracer relaxation, tracer_adjsens: surfaceForcingS,
    advection_in_gyre: pkg/ptracers' zero) at DO_OCEANIC_PHYS's range (1-OLx..sNx+OLx, do_oceanic_phys.F:560-563):
    surfaceForcingPTr of T04 bitwise at the dumped steps (PTRACERS_INTEGRATE does not write it). Inputs: pTracer of
    S00_begin, surfaceForcingS of S04, hFacC of S00, drF(1) from the oracle printout. Negative control: pkg/ptracers'
    version in place of the experiment's (latlon, tracer_adjsens) bites."""
    from mitjax.farray import FArray
    from mitjax.pkg.ptracers import ptracers_forcing_surf as pfs
    from mitjax.pkg.ptracers.ptracers_fields_h import PtracersFields
    from mitjax.io import stdout as so
    from mitjax.tests.col_replay import Cfg, Common, experiment
    ptr, d = _ptr_of(exp, inp)
    e = experiment(exp, inp)
    cfg = Cfg(e.cfg)
    cfg.experiment, cfg.code_dir, cfg.exp_dir = e.cfg.experiment, e.cfg.code_dir, e.cfg.exp_dir
    sz = e.cfg.size
    ij = dict(i=(1-sz.OLx, sz.sNx+sz.OLx), j=(1-sz.OLy, sz.sNy+sz.OLy))
    drF = np.array([so.fortran_number(v) for v in d["drF"].values()])
    params = Common({"usingPCoords": False}, {})
    ds, its, _ = oracle(exp, inp)

    def run(fs, p, sfS, h, drF, prior):
        grid = Common(hFacC=FArray(jnp.asarray(h), "hFacC", k=(1, sz.Nr), **ij),
                      drF=FArray(jnp.asarray(drF), "drF", k=(1, sz.Nr), tiled=False))
        ptf = PtracersFields(pTracer=(FArray(jnp.asarray(p), "pTracer", k=(1, sz.Nr), **ij),), gpTrNm1=(None,),
                             surfaceForcingPTr=(FArray(jnp.asarray(prior), "surfaceForcingPTr", **ij),))
        ff = SimpleNamespace(surfaceForcingS=FArray(jnp.asarray(sfS), "surfaceForcingS", **ij))
        ptf = fs(None, 1-sz.OLx, sz.sNx+sz.OLx, 1-sz.OLy, sz.sNy+sz.OLy, 0., 0, cfg=cfg, grid=grid, params=params,
                 ptr=ptr, ptf=ptf, ff=ff)
        return ptf.surfaceForcingPTr[0].data
    own = pfs.routine_of_build(cfg)
    res, ctrl = {}, 0
    for it in its:
        p = _dump(exp, inp, it, "S00_begin", "pTracer_01")
        sfS = _dump(exp, inp, it, "S04_oceanic_phys", "surfaceForcingS")[:, 0]
        h = _dump(exp, inp, it, "S00_begin", "hFacC")
        prior = np.full(sfS.shape, np.nan)
        want = _dump(exp, inp, it, "T04_ptracers_integrate", "surfaceForcingPTr_01")
        res[it] = pg.bit_equal(jax.jit(functools.partial(run, own))(p, sfS, h, drF, prior), want[:, 0])
        if own is not pfs.ptracers_forcing_surf:
            ctrl += pg.bit_equal(jax.jit(functools.partial(run, pfs.ptracers_forcing_surf))(p, sfS, h, drF, prior),
                                 want[:, 0])[0]
    assert all(_ok(r) for r in res.values()), res
    if own is not pfs.ptracers_forcing_surf:
        assert ctrl > 0


# ---------------------------------------------------------------------------------------------------------------
# 4. gradients

def test_gradient_convective_adjustment():
    """d(sum of w*outputs)/d(theta, salt, pTracer) of CONVECTIVE_ADJUSTMENT: finite on every lane (halos, land);
    the tangent along a random direction agrees with a central FD (h = 1e-6 .. 1e-8 at a smooth point: no interface
    changes its stability under the perturbation) to 1e-7 relative, and the adjoint passes the dot test with the
    tangent. At an exactly neutral interface the FD and the derivative differ by O(1) for every h (measured: 1.8 %
    of the directional derivative with the raw synthetic set), the kink of `IF (...) .LT. 0.`."""
    from mitjax.model.src.convective_adjustment import convective_adjustment
    R = replay()
    grid, params, eos, ptr, _ = R.commons()
    I = R.inputs

    def f(t, s, p):
        st = pg.state_of(R.experiment.cfg, theta=R.f3(t, "theta"), salt=R.f3(s, "salt"))
        st, ptf = convective_adjustment(jnp.float64(R.g["tCall"]), 3, cfg=R.cfg, grid=grid, params=params, eos=eos,
                                        state=st, ptr=ptr, ptf=R.ptf(p3=p))
        return st.theta.data, st.salt.data, ptf.pTracer[0].data
    rng = np.random.default_rng(1)
    # the synthetic set has 5 % exactly neutral interfaces (a kink of the stability test): a 1e-2 offset makes every
    # interface strictly stable or unstable, far beyond what the FD steps can flip (smooth point)
    x = (jnp.asarray(I["tA"] + 1e-2*rng.standard_normal(I["tA"].shape)), jnp.asarray(I["sA"]),
         jnp.asarray(I["pA"]))
    w = tuple(jnp.asarray(rng.standard_normal(a.shape)) for a in x)
    loss = jax.jit(lambda *a: sum(jnp.vdot(o, wi) for o, wi in zip(f(*a), w)))
    g = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(*x)
    assert all(bool(jnp.all(jnp.isfinite(gi))) for gi in g)
    v = tuple(jnp.asarray(rng.standard_normal(a.shape)) for a in x)
    _, tan = jax.jit(lambda *a: jax.jvp(f, x, a))(*v)
    dot_t = sum(float(jnp.vdot(ti, wi)) for ti, wi in zip(tan, w))
    dot_a = sum(float(jnp.vdot(gi, vi)) for gi, vi in zip(g, v))
    assert abs(dot_t - dot_a) <= 1e-12 * abs(dot_a), (dot_t, dot_a)
    ff = jax.jit(f)
    errs = []
    for h in (1e-6, 1e-7, 1e-8):
        up = ff(*(a + h*b for a, b in zip(x, v)))
        dn = ff(*(a - h*b for a, b in zip(x, v)))
        fd = sum(float(jnp.vdot((u - d_)/(2*h), wi)) for u, d_, wi in zip(up, dn, w))
        errs.append(abs(fd - dot_t) / abs(dot_t))
    assert min(errs) < 1e-7, errs


def test_gradient_cg2d_nsa_solve():
    """cg2d_nsa_solve (the implicit rule): forward bit for bit cg2d_nsa's; d/d cg2d_b finite on every lane; tangent
    vs central FD of the literal solver at h = 1e-3 (the solve is linear in b: the FD error is the solver's tolerance)
    to 1e-6 relative; dot test of tangent and adjoint to 1e-10."""
    from mitjax.model.src.cg2d_nsa import cg2d_nsa, cg2d_nsa_solve
    R = replay()
    _, params, _, _, cg2dh = R.commons()
    ex = exchanger()
    I = R.inputs

    def sol(fn, b):
        return fn(R.f2(b, "cg2d_b"), R.f2(jnp.zeros_like(b), "cg2d_x"), 200, -1, cfg=R.cfg, cg2dh=cg2dh,
                  params=params, ex=ex)[1].data
    b = jnp.asarray(I["cgB"])
    x_lit = jax.jit(lambda b: sol(cg2d_nsa, b))(b)
    x_rule = jax.jit(lambda b: sol(cg2d_nsa_solve, b))(b)
    assert pg.bit_equal(x_rule, x_lit)[0] == 0
    rng = np.random.default_rng(2)
    v = jnp.asarray(rng.standard_normal(b.shape))
    w = jnp.asarray(rng.standard_normal(b.shape))
    sz = R.size
    interior = np.zeros(b.shape, bool)
    interior[:, sz.OLy:sz.OLy+sz.sNy, sz.OLx:sz.OLx+sz.sNx] = True
    w = jnp.where(interior, w, 0.)
    _, tan = jax.jit(lambda b, v: jax.jvp(lambda bb: sol(cg2d_nsa_solve, bb), (b,), (v,)))(b, v)
    g = jax.jit(jax.grad(lambda bb: jnp.vdot(sol(cg2d_nsa_solve, bb), w)))(b)
    assert bool(jnp.all(jnp.isfinite(g)))
    dt, da = float(jnp.vdot(tan, w)), float(jnp.vdot(g, v))
    assert abs(dt - da) <= 1e-10 * abs(da), (dt, da)
    h = 1e-3
    lit = jax.jit(lambda b: sol(cg2d_nsa, b))
    fd = float(jnp.vdot((lit(b + h*v) - lit(b - h*v))/(2*h), w))
    assert abs(fd - dt) <= 1e-6 * abs(dt), (fd, dt)


def test_gradient_swfrac():
    """SWFRAC: derivative finite at every point (incl. the deep lanes) and equal to the FD of the literal routine at
    smooth points (h = 1e-6, 1e-7) to 1e-8 relative."""
    from mitjax.model.src.swfrac import swfrac
    R = replay()
    sw = jnp.asarray(R.inputs["swdk"])
    f = lambda s: swfrac(sw.shape[0], 1.0, s, 0., 0)
    g = jax.jit(jax.jacfwd(f))(sw)
    d = jnp.diagonal(g)
    assert bool(jnp.all(jnp.isfinite(g)))
    smooth = np.abs(np.asarray(sw) + 200.0) > 1e-3
    for h in (1e-6, 1e-7):
        fd = (f(sw + h) - f(sw - h)) / (2*h)
        rel = np.abs(np.asarray(fd - d))[smooth] / np.maximum(np.abs(np.asarray(d))[smooth], 1e-300)
        assert np.all(rel[np.abs(np.asarray(d))[smooth] > 1e-12] < 1e-6), rel.max()


@pytest.mark.parametrize("exp,inp", EXPS, ids=IDS)
def test_dump_fields_blocking_exch(exp, inp):
    """PTRACERS_FIELDS_BLOCKING_EXCH (DO_FIELDS_BLOCKING_EXCHANGES, forward_step.F:1093): pTracer of
    S15_tracers_correction exchanged == pTracer of S16_blocking_exchanges bitwise (all points), with the exchanger
    measured on the run's own probe (X00_exch_probe); negative control: the unexchanged field differs in the halos."""
    from mitjax.eesupp.exch_maps import build_maps
    from mitjax.eesupp.exchange import Exchanger
    from mitjax.farray import FArray
    from mitjax.pkg.ptracers.ptracers_fields_blocking_exch import ptracers_fields_blocking_exch
    from mitjax.pkg.ptracers.ptracers_fields_h import PtracersFields
    from mitjax.tests.col_replay import Cfg, experiment
    ptr, _ = _ptr_of(exp, inp)
    if ptr.PTRACERS_SOM_Advection[0]:      # the SOM moments are not dumped: exchange the tracer only
        ptr = ptr.replace(static=dict(PTRACERS_SOM_Advection=(False,) * ptr.PTRACERS_num))
    e = experiment(exp, inp)
    cfg = Cfg(e.cfg)
    sz = e.cfg.size
    ds, its, _ = oracle(exp, inp)
    ex = Exchanger(build_maps(ds))
    ij = dict(i=(1-sz.OLx, sz.sNx+sz.OLx), j=(1-sz.OLy, sz.sNy+sz.OLy), k=(1, sz.Nr))

    def run(p):
        ptf = PtracersFields(pTracer=(FArray(p, "pTracer", **ij),), gpTrNm1=(None,), surfaceForcingPTr=(None,))
        return ptracers_fields_blocking_exch(cfg=cfg, ptr=ptr, ptf=ptf, ex=ex).pTracer[0].data
    f = jax.jit(run)
    res, ctrl = {}, 0
    for it in its:
        p = _dump(exp, inp, it, "S15_tracers_correction", "pTracer_01")
        want = _dump(exp, inp, it, "S16_blocking_exchanges", "pTracer_01")
        res[it] = pg.bit_equal(f(jnp.asarray(p)), want)
        ctrl += pg.bit_equal(p, want)[0]
    assert all(_ok(r) for r in res.values()), res
    assert ctrl > 0


@pytest.mark.parametrize("exp,inp", EXPS, ids=IDS)
def test_dump_init_varia(exp, inp):
    """PTRACERS_INIT_VARIA (packages_init_variables.F:334) against the state at the start of the first step
    (S00_begin, nIter0): pTracer, gpTrNm1, surfaceForcingPTr bitwise on every point. advection_in_gyre reads its
    initial dye field (PTRACERS_Iter0 = nIter0) and exchanges it; latlon starts from PTRACERS_ref = 0; in
    tutorial_tracer_adjsens (cAdjFreq /= 0) INITIALISE_VARIA then calls CONVECTIVE_ADJUSTMENT_INI
    (initialise_varia.F:283-295) on theta, salt of I02_ini_fields (after INI_FIELDS; nothing in between writes them)
    and the passive tracer: theta, salt and pTracer of S00_begin bitwise through that chain (with the exchange of the
    zero control's CTRL_MAP_GENARR3D, see the comment)."""
    from mitjax.eesupp.exch_maps import build_maps
    from mitjax.eesupp.exchange import Exchanger
    from mitjax.farray import FArray
    from mitjax.pkg.ptracers.ptracers_init_varia import ptracers_init_varia
    from mitjax.pkg.rw.read_rec import RW
    from mitjax.tests import init_gate as ig
    from mitjax.tests.col_replay import Cfg, Common, experiment
    ptr, d = _ptr_of(exp, inp)
    e = experiment(exp, inp)
    cfg = Cfg(e.cfg)
    sz = e.cfg.size
    ds, its, rundir = oracle(exp, inp)
    it0 = its[0]
    ex = Exchanger(build_maps(ds))
    prm = ig.params(exp, inp)
    rw = RW(rundir, prm.init.readBinaryPrec, sz)
    ij = dict(i=(1-sz.OLx, sz.sNx+sz.OLx), j=(1-sz.OLy, sz.sNy+sz.OLy))
    like3 = FArray(jnp.zeros((sz.nSx*sz.nSy, sz.Nr, sz.sNy+2*sz.OLy, sz.sNx+2*sz.OLx)), "f3", k=(1, sz.Nr), **ij)
    like2 = FArray(jnp.zeros((sz.nSx*sz.nSy, sz.sNy+2*sz.OLy, sz.sNx+2*sz.OLx)), "f2", **ij)
    maskC = _dump(exp, inp, it0, "G00_geometry", "maskC")
    grid = Common(maskC=FArray(jnp.asarray(maskC), "maskC", k=(1, sz.Nr), **ij))
    ptr2, ptf = ptracers_init_varia(prm.time.nIter0, prm.init.pickupSuff, cfg=cfg, grid=grid, ptr=ptr, ex=ex, rw=rw,
                                    like3=like3, like2=like2)
    assert ptr2.PTRACERS_StepFwd == (True,) * ptr.PTRACERS_num
    res = {}
    if cfg.cpp.INCLUDE_CONVECT_INI_CALL and float(d["cAdjFreq"].values()[0].replace("E", "e")) != 0.:
        from mitjax.model.src.convective_adjustment_ini import convective_adjustment_ini
        R = replay()
        rg, params, eos, _, _ = R.commons()
        rg = rg.replace(hFacC=R.f3(_dump(exp, inp, it0, "S00_begin", "hFacC"), "hFacC"))
        st = pg.state_of(R.experiment.cfg, theta=R.f3(_dump(exp, inp, it0, "I02_ini_fields", "theta"), "theta"),
                         salt=R.f3(_dump(exp, inp, it0, "I02_ini_fields", "salt"), "salt"))
        st, ptf = jax.jit(lambda st, ptf, g, p, eo: convective_adjustment_ini(
            0., 0, cfg=R.cfg, grid=g, params=p, eos=eo, state=st, ptr=ptr2, ptf=ptf))(st, ptf, rg, params, eos)
        res["theta"] = pg.bit_equal(st.theta.data, _dump(exp, inp, it0, "S00_begin", "theta"))
        res["salt"] = pg.bit_equal(st.salt.data, _dump(exp, inp, it0, "S00_begin", "salt"))
    if cfg.cpp.ALLOW_CTRL and e.cfg.use_flag("useCTRL"):
        # tutorial_tracer_adjsens controls pTracer (xx_ptr1): CTRL_INIT_VARIABLES (packages_init_variables.F:619,
        # after PTRACERS_INIT_VARIA :334) -> CTRL_MAP_GENARR3D adds xx_gen*mask3D (zero in this forward run:
        # interior unchanged) and calls EXCH_XYZ_RL( fld ) (pkg/ctrl/ctrl_map_genarr.F:380-399); not ported here
        # (ctrl genarr: GOADK lane), its effect on the value is that exchange: measured to matter (the southern halo
        # rows receive the northern interior's PTRACERS_ref = 1 where the land mask had set 0)
        from mitjax.eesupp.exch_rs import EXCH_XYZ_RL
        ptf = ptf.set("pTracer", 1, jax.jit(lambda f: EXCH_XYZ_RL(f, ex=ex))(ptf.pTracer[0]))
    res["pTracer_01"] = pg.bit_equal(ptf.pTracer[0].data, _dump(exp, inp, it0, "S00_begin", "pTracer_01"))
    res["gpTrNm1_01"] = pg.bit_equal(ptf.gpTrNm1[0].data, _dump(exp, inp, it0, "S00_begin", "gpTrNm1_01"))
    res["surfaceForcingPTr_01"] = pg.bit_equal(ptf.surfaceForcingPTr[0].data,
                                               _dump(exp, inp, it0, "S00_begin", "surfaceForcingPTr_01")[:, 0])
    assert all(_ok(r) for r in res.values()), res


@pytest.mark.parametrize("exp,inp", EXPS, ids=IDS)
def test_monitor_ptracer_blocks(exp, inp):
    """PTRACERS_MONITOR through pkg/monitor on the oracle's own pTracer: the 'MONITOR ptracer field statistics'
    blocks of the oracle STDOUT, character for character, at nIter0 (S00_begin) and after each dumped step n
    (S17_monitor of n, the state at forward_step.F:1154; PTRACERS_OUTPUT runs after it in DO_THE_MODEL_IO and nothing
    in between writes pTracer). The blocks are in file order one per iteration (monitorFreq <= deltaTClock in these
    runs; checked). Negative control: the largest wet interior pTracer value x1.5 changes the block (a 1-ulp change does
    not reach the 14 printed digits: measured)."""
    from mitjax.eesupp.print import MessageUnits
    from mitjax.farray import FArray
    from mitjax.pkg.monitor.mon_init import mon_init
    from mitjax.pkg.monitor.monitor_h import MonitorCommon
    from mitjax.pkg.ptracers.ptracers_fields_h import PtracersFields
    from mitjax.pkg.ptracers.ptracers_monitor import ptracers_monitor
    from mitjax.tests import monitor_gate as mg
    o = mg.oracle(exp, inp)
    cfg = mg.make_cfg(o)
    params = mg.make_params(o, cfg)
    ptr, _ = _ptr_of(exp, inp)
    blocks = [b for b in o.blocks if b.kind == "MONITOR ptracer field statistics"]
    assert ptr.PTRACERS_monitorFreq <= params.deltaTClock and len(blocks) >= len(o.its) + 1
    sz = cfg
    ij = dict(i=(1-sz.OLx, sz.sNx+sz.OLx), j=(1-sz.OLy, sz.sNy+sz.OLy), k=(1, sz.Nr))
    it0 = o.its[0]

    def ours(tsnumber, p, grid):
        mon = MonitorCommon()
        mon_init(cfg=cfg, params=params, mon=mon)
        mon.io = MessageUnits()
        ptf = PtracersFields(pTracer=(FArray(p, "pTracer", **ij),), gpTrNm1=(None,), surfaceForcingPTr=(None,))
        myTime = params.startTime + params.deltaTClock*float(tsnumber - params.nIter0)   # forward_step.F:808
        ptracers_monitor(myTime, tsnumber, cfg=cfg, params=params, grid=grid, ptr=ptr, ptf=ptf, mon=mon)
        return mon.io.records(mon.mon_ioUnit)
    diffs, ctrl = {}, 0
    for b, it in enumerate([None] + list(o.its)):
        tsn = it0 if it is None else it + 1
        p = (_dump(exp, inp, it0, "S00_begin", "pTracer_01") if it is None
             else _dump(exp, inp, it, "S17_monitor", "pTracer_01"))
        # hFacC as the monitor gate takes it: S00_begin of nIter0, else the last stage of step n that dumps it (r*:
        # UPDATE_R_STAR(.TRUE.) rescales it in tutorial_tracer_adjsens, measured: the block after step 1 differs in the
        # 7th digit with nIter0's hFacC)
        grid = (mg.make_grid(o, cfg, params, it0, "S00_begin", it0) if it is None
                else mg.make_grid(o, cfg, params, it0, mg._last_stage_with(o, it, "hFacC"), it))
        theirs = o.raw_lines(blocks[b].first - 1, blocks[b].last + 1)
        got = ours(tsn, jnp.asarray(p), grid)
        diffs[tsn] = mg.compare(got, theirs)
        if b == 1:                              # planted: the largest wet interior value x1.5
            q = np.array(p)
            inter = np.zeros(q.shape, bool)
            inter[:, :, sz.OLy:sz.OLy+sz.sNy, sz.OLx:sz.OLx+sz.sNx] = True
            sel = (np.asarray(grid.maskInC.data)[:, None] * np.ones_like(q) > 0) & inter
            big = np.unravel_index(np.argmax(np.where(sel, np.abs(q), -1.)), q.shape)   # the largest wet value
            q[big] *= 1.5
            ctrl = len(mg.compare(ours(tsn, jnp.asarray(q), grid), theirs))
    assert all(not d for d in diffs.values()), diffs
    assert ctrl > 0


def test_ini_mixing_tracer_adjsens():
    """INI_MIXING (ALLOW_3D_DIFFKR, tutorial_tracer_adjsens; diffKrFile unset): diffKr = diffKrNrS(k) of
    INI_PARMS (ini_parms_tracer, ALLOW_3D_DIFFKR now accepted) == the DYNVARS.h diffKr of S00_begin at nIter0 bitwise;
    diffKrNrS == the printout. Negative control: diffKrNrT in place of diffKrNrS bites."""
    from mitjax.model.src.ini_mixing import ini_mixing
    from mitjax.model.src.ini_parms import ini_parms_dyn
    from mitjax.model.src.ini_parms_tracer import ini_parms_tracer
    from mitjax.model.state import empty_state
    from mitjax.tests import init_gate as ig
    from mitjax.tests.col_replay import experiment
    from mitjax.io import stdout as so
    e = experiment(*RUN)
    prm = ig.params(*RUN)
    p = ini_parms_tracer(e, ini_parms_dyn(e, prm.grid, prm.time, prm.init), prm.time, prm.init)
    ds, its, rundir = oracle(*RUN)
    d = so.parameter_dict(so.parameters(so.read_stdout(rundir / "output.txt")))
    assert list(np.asarray(p.diffKrNrS.data)) == [so.fortran_number(v) for v in d["diffKrNrS"].values()]
    st = ini_mixing(empty_state(e.cfg), " ", cfg=e.cfg, params=p, ex=None, rw=None)
    want = _dump(*RUN, its[0], "S00_begin", "diffKr")
    assert _ok(pg.bit_equal(st.diffKr.data, want))
    bad = ini_mixing(empty_state(e.cfg), " ", cfg=e.cfg, params=p.replace(traced=dict(diffKrNrS=p.diffKrNrT)),
                     ex=None, rw=None)
    assert pg.bit_equal(bad.diffKr.data, want)[0] > 0
