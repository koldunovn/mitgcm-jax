"""Helpers of the lab_sea/input_ad gates (M4 step 7 part 2, lane M4ADLAB session 2).

Oracle: lane A's registered dumps-on run job27855987-jdon of lab_sea/input_ad (build lab_sea-code_ad-63cdc0b, the
forward of the code_ad build), iterations 0, 1, 2 (nIter0 = 0, a cold start from the Levitus files).

`stub_model` is scaffolding (as lane M4LAB session 1 did for lab_sea/input): the Model of input_ad does not set up
yet (the lane's gap list: pkg/ecco's 2-D gencost terms and pkg/ctrl's genarr2d controls refuse,
G9-G10; the stubs of G1-G4 and G7 went in session 3), so it is built
with those settings replaced (STUB_NAMELIST, STUB_USE_OFF, STUB_CPP_OFF; set-up measured: dev job 27893278). The
kernels under test run with the unplanted code_ad build (`ad_cfg`) and the unplanted SEAICE_PARM01 values
(`ad_sp`): the stubs only make the Model's grid, parameters and initial fields available. Each port of a gap
removes its stub."""

import dataclasses
import functools

import jax
import jax.numpy as jnp

EXP = ("lab_sea", "input_ad")
JDON = "job27855987-jdon"
S = "data.seaice"
STUB_NAMELIST = {
}   # the sea-ice stubs of G2-G4 (session 2) and G1's useDOWN_SLOPE are gone (session 3)
STUB_USE_OFF = ()          # G8-G10 ported (session 4): no stub left, `stub_model` is the real Model of input_ad
STUB_CPP_OFF = ()
# the SEAICE_PARM01 values of input_ad/data.seaice that session 2's stubs replaced (asserted in `ad_sp`), with those
# SEAICE_READPARMS derives from them: SEAICEdiffKhHeff = SEAICEdiffKhArea, SEAICEdiffKhSnow = SEAICEdiffKhSalt =
# SEAICEdiffKhHeff when unset (seaice_readparms.F:1056-1061; data.seaice sets none of them)
SP_REAL = dict(SEAICEdiffKhArea=200.0, SEAICEdiffKhHeff=200.0, SEAICEdiffKhSnow=200.0, SEAICEdiffKhSalt=200.0,
               SEAICE_areaLossFormula=3, SEAICE_areaGainFormula=2, useMaykutSatVapPoly=True, postSolvTempIter=0)

# SEAICE_LSR's sweeps per pass in the code_ad oracle (ICOUNT1 = u, ICOUNT2 = v), measured by the Fortran probe of
# lane M4ADLAB session 1 (job 27882556): {iteration: ((pass 1 u, v), (pass 2 u, v))}
ICOUNT = {0: ((28, 58), (28, 52)), 1: ((32, 68), (32, 82)), 2: ((36, 78), (38, 62))}


def ad_experiment():
    from mitjax.config.params import load
    return load(*EXP)


def stub_experiment():
    from mitjax.config.params import load
    from mitjax.drivers.model import with_namelist
    from mitjax.tests.m4col_gate import CppPlant
    e = with_namelist(load(*EXP), STUB_NAMELIST)
    cfg = dataclasses.replace(e.cfg, use=tuple((k, False if k in STUB_USE_OFF else v) for k, v in e.cfg.use),
                              cpp=CppPlant(e.cfg.cpp, off=STUB_CPP_OFF))
    return dataclasses.replace(e, cfg=cfg)


@functools.lru_cache(maxsize=None)
def stub_model():
    """(Model, DumpSet) of lab_sea/input_ad with the stubs, on the oracle's dumps-on run directory."""
    from mitjax import paths
    from mitjax.drivers.model import Model
    from mitjax.io.dump import DumpSet
    from mitjax.xla_flags import set_gate_xla_flags
    set_gate_xla_flags()
    top = paths.REFERENCE_RUNS / EXP[0] / EXP[1] / JDON
    return Model(stub_experiment(), top / "rundir"), DumpSet(top / "dumps")


@functools.lru_cache(maxsize=None)
def ad_cfg():
    """The unplanted configuration of the code_ad build (CPP options and package switches of input_ad)."""
    return ad_experiment().cfg


def ad_sp(m):
    """The Model's SEAICE parameters, which hold input_ad's own SEAICE_PARM01 values since session 3 (no sea-ice stub
    left; asserted against SP_REAL)."""
    sp = m.arrays.pkc["sp"]
    bad = {k: (getattr(sp, k), v) for k, v in SP_REAL.items() if getattr(sp, k) != v}
    assert not bad, bad
    return sp


def clock(m, it):
    """(myTime, myIter) at the start of iteration `it` (input_ad: nIter0 = 0, startTime = 0)."""
    tp = m.prm.time
    return jnp.float64(tp.startTime + tp.deltaTClock*(it - tp.nIter0)), jnp.int32(it)


def lsr_fn(m, cfg=None):
    """jit(f(sp, op, sf, st, myTime, myIter) -> (sf, out)) of SEAICE_LSR (the forward-only wrapper of the Model)."""
    from mitjax.ad.seaice_lsr_rule import seaice_lsr_forward_only
    cfg = ad_cfg() if cfg is None else cfg

    def f(sp, op, sf, st, myTime, myIter):
        return seaice_lsr_forward_only(myTime, myIter, sf, cfg=cfg, sp=sp, op=op, grid=m.arrays.grid, state=st,
                                       ex=m.ex)
    return jax.jit(f)


def lsr_diffs(m, ds, its=(0, 1, 2), sp=None, fn=None):
    """{it: ({field: differing points} at Y06_lsr, out)}: SEAICE_LSR teacher-forced from the oracle's SEAICE.h before
    Y06_lsr (Y04_solver_inputs) and uVel/vVel of S00_begin, every SEAICE.h field the stage dumps, incl. halos."""
    from mitjax.tests import m4off_gate as G
    sp = ad_sp(m) if sp is None else sp
    fn = lsr_fn(m) if fn is None else fn
    res = {}
    for it in its:
        sf, _, _, st = G.inputs_at(m, ds, it, "Y06_lsr")
        sf2, out = fn(sp, m.arrays.pkc["op"], sf, st, *clock(m, it))
        res[it] = (G.differing(ds, it, "Y06_lsr", {n: v for n, v in sf2.items() if hasattr(v, "data")}),
                   jax.tree_util.tree_map(jax.device_get, out))
    return res


def dynsolver_fn(m, cfg=None):
    """jit(f(sp, op, sf, ff, exf, st, myTime, myIter) -> {stage: {name: FArray}}) of SEAICE_MODEL's start as far as
    SEAICE_DYNSOLVER: the uwind / vwind exchange (seaice_model.F:124-135), the ALLOW_AUTODIFF reset of uIceNm1 /
    vIceNm1 (:137-153) and SEAICE_DYNSOLVER (:186) with its probes (Y01 .. Y09) and "I01_dynsolver"."""
    from mitjax.eesupp.exch_rs import _rewrap
    from mitjax.farray import loop_i, loop_j
    from mitjax.pkg.seaice.seaice_dynsolver import seaice_dynsolver
    cfg = ad_cfg() if cfg is None else cfg
    pkc = m.arrays.pkc

    def f(sp, op, sf, ff, exf, st, myTime, myIter):
        out = {}
        exf = dict(exf)
        u, v = m.ex.EXCH_UV_AGRID_3D_RL(exf["uwind"].data, exf["vwind"].data, True)   # seaice_model.F:127
        exf["uwind"], exf["vwind"] = _rewrap(u, exf["uwind"]), _rewrap(v, exf["vwind"])
        if cfg.cpp.flag("ALLOW_AUTODIFF"):                                     # seaice_model.F:137-153
            sz = cfg.size
            jA, iA = loop_j(1-sz.OLy, sz.sNy+sz.OLy), loop_i(1-sz.OLx, sz.sNx+sz.OLx)
            sf = {**sf, **{n: sf[n].at[iA, jA].set(0.) for n in ("uIceNm1", "vIceNm1")}}

        def probe(stage, vals, ff_):
            out[stage] = {**{n: getattr(ff_, n) for n in ff_.names()}, **sf, **vals}
        sf2, ff2 = seaice_dynsolver(myTime, myIter, sf, ff, exf, cfg=cfg, sp=sp, op=op, exfp=pkc["exfp"],
                                    grid=m.arrays.grid, state=st, ex=m.ex, probe=probe)
        out["I01_dynsolver"] = {**{n: getattr(ff2, n) for n in ff2.names()},
                                **{n: v for n, v in sf2.items() if n != "_lsr_out"}}
        return out, sf2["_lsr_out"]
    return jax.jit(f)


def dynsolver_diffs(m, ds, its=(0, 1, 2), sp=None, fn=None):
    """{it: ({stage: {field: differing points}}, lsr out)} of SEAICE_DYNSOLVER teacher-forced from the oracle's
    fields before I00_seaice_begin (SEAICE.h, FFIELDS.h, EXF_FIELDS.h of the stages before; theta, salt, uVel,
    vVel, etaN of S00_begin), compared at Y01 .. Y09 and I01_dynsolver, every dumped field, every point."""
    from mitjax.tests import m4off_gate as G
    sp = ad_sp(m) if sp is None else sp
    fn = dynsolver_fn(m) if fn is None else fn
    out = {}
    for it in its:
        sf, ff, exf, st = G.inputs_at(m, ds, it, "I00_seaice_begin")
        res, lo = fn(sp, m.arrays.pkc["op"], sf, ff, exf, st, *clock(m, it))
        out[it] = ({stage: G.differing(ds, it, stage, {n: v for n, v in vals.items() if hasattr(v, "data")})
                    for stage, vals in res.items()}, jax.tree_util.tree_map(jax.device_get, lo))
    return out


def advdiff_fn(m, cfg=None):
    """jit(f(sp, op, sf, myTime, myIter) -> sf) of SEAICE_ADVDIFF(UICE, VICE) as SEAICE_MODEL calls it
    (seaice_model.F:230)."""
    from mitjax.pkg.seaice.seaice_advdiff import seaice_advdiff
    cfg = ad_cfg() if cfg is None else cfg

    def f(sp, op, sf, myTime, myIter):
        return seaice_advdiff(sf["UICE"], sf["VICE"], myTime, myIter, sf, cfg=cfg, sp=sp, op=op,
                              grid=m.arrays.grid, ex=m.ex)
    return jax.jit(f)


def advdiff_diffs(m, ds, its=(0, 1, 2), sp=None, fn=None):
    """{it: {field: differing points}} of SEAICE_ADVDIFF teacher-forced from the oracle's SEAICE.h before I02_advdiff
    against I02_advdiff, every dumped SEAICE.h field, every point incl. halos."""
    from mitjax.tests import m4off_gate as G
    sp = ad_sp(m) if sp is None else sp
    fn = advdiff_fn(m) if fn is None else fn
    out = {}
    for it in its:
        sf, _, _, _ = G.inputs_at(m, ds, it, "I02_advdiff")
        sf2 = fn(sp, m.arrays.pkc["op"], sf, *clock(m, it))
        out[it] = G.differing(ds, it, "I02_advdiff", {n: v for n, v in sf2.items() if hasattr(v, "data")})
    return out


# ------------------------------------------------------------------ the reverse pass of one SEAICE_LSR call
def lsr_reverse(m, ds, it, cfg=None, seed=20261005, n_sweeps=None):
    """The reverse pass of one teacher-forced SEAICE_LSR call (input_ad, iteration `it`) through the executed sweeps:
    J(x) = sum(w_u * UICE) + sum(w_v * VICE) after the call, x = every float64 SEAICE.h input, w random.
    Session 4 (plan decision 14, A1 in the kernel): the code_ad build defines SEAICE_LSR_ADJOINT_ITER, so the
    kernel's DO m loop is mitjax/ad/lsr_sweeps.taped_sweeps (fixed-length scan, per-sweep cond, remat) and the
    gradient is the kernel's own; the literal while_loop forward is the same kernel with the option planted off.
    `n_sweeps`: a planted scan length (negative control of the bitwise check). Returns dict: fwd_bitwise (the
    kernel's forward == the while_loop forward on UICE, VICE), nonfinite ({input: points} of the gradient), dot
    (<grad J, dx>, the JVP J'(x) dx, relative difference) for a random dx on the smooth inputs UICE, VICE, FORCEX0,
    FORCEY0, and fd ({eps: central difference of the literal while_loop forward along dx})."""
    import numpy as np
    import mitjax.ad.lsr_sweeps as SW
    import mitjax.pkg.seaice.seaice_lsr as LSR
    from mitjax.tests import m4off_gate as G
    from mitjax.tests.m4col_gate import CppPlant
    cfg = ad_cfg() if cfg is None else cfg
    assert cfg.cpp.flag("SEAICE_LSR_ADJOINT_ITER", "SEAICE_OPTIONS.h")
    cfg_wl = dataclasses.replace(cfg, cpp=CppPlant(cfg.cpp, off=("SEAICE_LSR_ADJOINT_ITER",)))
    sp, op = ad_sp(m), m.arrays.pkc["op"]
    sf, _, _, st = G.inputs_at(m, ds, it, "Y06_lsr")
    myTime, myIter = clock(m, it)

    def run_with(c):
        def run(sf_):
            sf2, _ = LSR.seaice_lsr(myTime, myIter, sf_, cfg=c, sp=sp, op=op, grid=m.arrays.grid, state=st, ex=m.ex)
            return sf2["UICE"].data, sf2["VICE"].data
        return run
    run, run_wl = run_with(cfg), run_with(cfg_wl)
    keys = [k for k, v in sf.items() if hasattr(v, "data") and v.data.dtype == jnp.float64]
    xs = {k: sf[k] for k in keys}
    u0, v0 = jax.jit(run_wl)(sf)
    real = SW.taped_sweeps
    try:
        if n_sweeps is not None:                     # a planted tape length (the kernel reads the module's function)
            SW.taped_sweeps = lambda cond, body, init, n=None: real(cond, body, init, n_sweeps)
            jax.clear_caches()
        u1, v1 = jax.jit(run)(sf)
    finally:
        SW.taped_sweeps = real
    res = {"fwd_bitwise": bool(np.array_equal(np.asarray(u0).view(np.int64), np.asarray(u1).view(np.int64))
                               and np.array_equal(np.asarray(v0).view(np.int64), np.asarray(v1).view(np.int64)))}
    if n_sweeps is not None:
        jax.clear_caches()
        return res
    rng = np.random.default_rng(seed)
    wu, wv = jnp.asarray(rng.standard_normal(u0.shape)), jnp.asarray(rng.standard_normal(v0.shape))

    def J_of(r):
        def J(x):
            u, v = r({**sf, **x})
            return jnp.sum(u*wu) + jnp.sum(v*wv)
        return J
    J, J_wl = J_of(run), J_of(run_wl)
    gx = jax.jit(jax.grad(J))(xs)
    res["nonfinite"] = {k: int(np.count_nonzero(~np.isfinite(np.asarray(gx[k].data)))) for k in keys}
    dirs = ("UICE", "VICE", "FORCEX0", "FORCEY0")
    dx = {k: type(v)(jnp.asarray(rng.standard_normal(v.data.shape))*jnp.abs(v.data).mean()*1e-2 if k in dirs
                     else jnp.zeros_like(v.data), v.name, tiled=v.tiled, _dims=v.dims) for k, v in xs.items()}
    _, jv = jax.jit(lambda x, d: jax.jvp(J, (x,), (d,)))(xs, dx)
    gd = sum(float(jnp.sum(gx[k].data*dx[k].data)) for k in dirs)
    res["dot"] = (gd, float(jv), abs(float(jv) - gd)/abs(gd))
    jax.clear_caches()
    Jw = jax.jit(J_wl)
    fd = {}
    for eps in (1e-3, 1e-4, 1e-5):
        xp = {k: type(v)(v.data + eps*dx[k].data, v.name, tiled=v.tiled, _dims=v.dims) for k, v in xs.items()}
        xm = {k: type(v)(v.data - eps*dx[k].data, v.name, tiled=v.tiled, _dims=v.dims) for k, v in xs.items()}
        fd[eps] = (float(Jw(xp)) - float(Jw(xm)))/(2*eps)
    res["fd"] = fd
    res["fd_rel"] = min(abs(v - gd)/abs(gd) for v in fd.values())
    return res


def free_step_fn(m, cfg=None):
    """jit(step)(arrays, carry, iloop, myTime, myIter) -> (carry, probes): the driver's whole FORWARD_STEP with every
    probe kept (keys as goadk_model_gate.compare_step reads them), chained from the Model's own carry (session 3)."""
    from mitjax.farray import FArray
    from mitjax.model.src.forward_step import forward_step
    cfg = m.cfg if cfg is None else cfg
    fp, pks = m.fp, m.pks

    def step(a, carry, iloop, myTime, myIter):
        probes = {}

        def probe(stage, values):
            key = f"{stage[0]}@{stage[1]}" if isinstance(stage, tuple) else stage
            probes[key] = dict(values) if isinstance(stage, tuple) else values
        state, ff, phi0surf, flow, pk = carry
        probes["_in"] = dict(ff=ff, phi0surf=phi0surf, gm=pk.get("gm"), grid=a.grid)
        state, ff, phi0surf, myTime, myIter, out = forward_step(
            iloop, myTime, myIter, cfg=cfg, grid=a.grid, params=a.params, fp=fp, eos=a.eos, cg2dh=a.cg2dh,
            cg2d_params=a.cg2d_params, state=state, ff=ff, phi0surf=phi0surf, ex=a.ex, probe=probe, pk=pk,
            pkc=a.pkc, pks=pks)
        flow2 = tuple(FArray(f.data, c.name, tiled=c.tiled, _dims=c.dims) for f, c in zip(out["flow"], flow))
        return (state, ff, phi0surf, flow2, out["pk"]), probes
    return jax.jit(step)


def free_steps(m, ds, its=(0, 1, 2), cfg=None):
    """{it: (compared stages, {(stage, field): differing points})}: the steps 0..max(its) chained from the Model's own
    initial carry (input_ad is a cold start: nIter0 = 0), every dumped stage the probes hold compared on every point
    (goadk_model_gate.compare_step). Free, not teacher-forced: the CD scheme's carried velocities (useCDscheme) and
    other carried fields are not dumped at S00_begin, so a teacher carry at iterations >= 1 would not be the
    oracle's state."""
    from mitjax.tests import goadk_model_gate as M
    from mitjax.tests import m4lab_gate as L
    f = free_step_fn(m, cfg)
    tp = m.prm.time
    carry = m.initial_carry()
    out = {}
    for it in range(max(its) + 1):
        k = it - tp.nIter0
        carry, probes = f(m.arrays, carry, jnp.int32(k + 1), jnp.float64(tp.startTime + tp.deltaTClock*k),
                          jnp.int32(it))
        if it in its:
            res = M.compare_step(m, ds, it, probes)
            out[it] = (sorted(res), L.n_bad(res))
    return out
