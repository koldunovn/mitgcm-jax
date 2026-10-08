#!/usr/bin/env python3
"""Localisation of the multi-step growth of the sharded gradient difference (lane M4COSTSHARD session 2; measurement
script, not a gate): global_ocean.cs32x15/input_ad, run mode (cg2dFullAdjoint = .FALSE.: the CG2D operator passive),
GenarrAdjoint's objective dfc/dxx_theta over the 5-step window, as scripts/m4costshard_shardgrad.py.

The objective is GenarrAdjoint's J (CTRL_MAP_INI_GENARR, integrate with the same schedule and SAVE_NAMES, COST_FINAL
on the gathered fields) with two script-side instruments, both identities in the forward:
  * the derivative solve of the CG2D implicit rule (mitjax/model/src/cg2d.py `_derivative_solve`, mitjax/ad/
    cg2d_rule.make_pcg_solve) replaced, in this process only, by the same `_pcg` call (same matvec, preconditioner,
    vdot, tolerance 1e-13 and max_iters) that also hands (iterations, final relative residual, rhs, solution) to a
    jax.debug.callback; the forward solve `_forward_solve` likewise wrapped to record (numIters, firstResidual,
    lastResidual, minResidualSq) of every forward CG2D;
  * a custom_vjp identity `tap(st, iloop, K)` on the step state before every step iloop and before COST_FINAL
    (iloop = nTimeSteps + 1): its backward records the cotangent of every tiled FArray of the state (the adjoint
    state lambda_{iloop-1} after the reverse of steps > iloop-1) and, in the perturbation mode, adds a 1-ulp
    relative +-1 pattern (fixed seed) to the cotangents when iloop == K (K a traced argument: one compile).
Under shard_map every device reports its local tiles (device d holds tiles d*Tloc ..); the host assembles them.

Modes:
  log      P = 1 and P = 6: per forward / backward CG2D the counts and residuals, per backward solve and per step
           boundary the P = 6 - P = 1 differences; gradients compared bit for bit with REF's uninstrumented g1 / g6
           (the instruments must not change a bit).
  perturb  P = 1 only: base (no noise; == REF g1 bit for bit), then 1-ulp noise injected (a) into the adjoint state
           at one boundary K = 6 .. 1, (b) into the right-hand side of one backward CG2D solve n = 0 .. 4 (n = 0: the
           first reverse solve, i.e. step 5's), (c) into the solution of solve n = 0 and 4; each: max|g - g_base| /
           max|g_base| (the R5 measure), to compare with the P = 6 difference.

    m4costshard_cg2dlog.py OUT_DIR {log|perturb} REF_DIR
"""
import gc
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

from mitjax.xla_flags import gate_xla_flags

os.environ["XLA_FLAGS"] = gate_xla_flags("").replace("device_count=4", "device_count=6")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402

OUT, WHAT, REF = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
assert WHAT in ("log", "perturb", "leaves", "inject"), WHAT
if (OUT / "result.json").exists():
    raise SystemExit(f"{OUT}/result.json exists: choose a new OUT_DIR")
OUT.mkdir(parents=True, exist_ok=True)
assert len(jax.devices()) == 6, jax.devices()

from mitjax import paths  # noqa: E402
from mitjax.ad import cg2d_rule as CR  # noqa: E402
from mitjax.drivers import adjoint_run as AR  # noqa: E402
from mitjax.drivers.checkpoint import SAVE_NAMES, integrate  # noqa: E402
from mitjax.drivers.model import Model  # noqa: E402
from mitjax.drivers.run import load_experiment, make_rundir  # noqa: E402
from mitjax.drivers.sharded_grad import place_model, seed, tile_specs  # noqa: E402
from mitjax.eesupp import exch_maps as EM  # noqa: E402
from mitjax.eesupp.shard import TileSharding  # noqa: E402
from mitjax.farray import FArray  # noqa: E402
from mitjax.model.src import cg2d as C  # noqa: E402
from mitjax.pkg.ctrl.ctrl_readparms import ctrl_readparms_genarr  # noqa: E402

ULP = 2.0 ** -52
S = {"axis": None, "rec": {}, "noise": WHAT in ("perturb", "leaves"), "leaves": WHAT == "leaves", "nsolve": 0,
     "pert": (-1, ""), "inject": WHAT == "inject", "delta": {}}
R = {"what": WHAT, "ref": str(REF), "t": {}}


def save():
    (OUT / "result.json").write_text(json.dumps(R, indent=1, default=float))


def release():
    jax.clear_caches()
    gc.collect()


def bits(a, b):
    return int(np.count_nonzero(np.ascontiguousarray(a).view(np.int64) != np.ascontiguousarray(b).view(np.int64)))


def relmax(a, b):
    d = float(np.max(np.abs(a - b)))
    s = float(np.max(np.abs(b)))
    return d / s if s > 0 else (0.0 if d == 0 else float("inf"))


_PAT = {}


def pattern(shape):
    """A fixed +-1 pattern of `shape` (numpy constant; seed 0)."""
    if shape not in _PAT:
        _PAT[shape] = np.where(np.random.default_rng(0).random(shape) < 0.5, -1.0, 1.0)
    return _PAT[shape]


def _dev():
    return lax.axis_index(S["axis"]) if S["axis"] else jnp.zeros((), jnp.int32)


def _host_rec(kind, dev, *vals):
    S["rec"].setdefault((kind, int(dev)), []).append([np.array(v) for v in vals])


# ---------------------------------------------------------------- instrument 1: the CG2D solves
_orig_forward_solve = C._forward_solve


def _forward_solve_log(static, A, b, x_first, op_consts, solve_consts):
    x, aux = _orig_forward_solve(static, A, b, x_first, op_consts, solve_consts)
    if not S["noise"]:
        jax.debug.callback(partial(_host_rec, "fwd"), _dev(), aux["numIters"], aux["firstResidual"],
                           aux["lastResidual"], aux["minResidualSq"])
    return x, aux


def _host_flags(_):
    n = S["nsolve"]
    S["nsolve"] += 1
    k, what = S["pert"]
    return np.array([1.0 if (n == k and what == "rhs") else 0.0, 1.0 if (n == k and what == "sol") else 0.0])


def _host_delta(r):
    """inject mode: (rhs, solution) differences P6 - P1 of backward solve n (log run's bwd<n>.npz) for the solves
    selected in S["delta"] ({n: (use_rhs, use_sol)}), zeros otherwise."""
    n = S["nsolve"]
    S["nsolve"] += 1
    out = np.zeros((2,) + r.shape)
    if n in S["delta"]:
        d = np.load(REF / f"bwd{n}.npz")
        ur, us = S["delta"][n]
        if not any(k < n for k in S["delta"]):          # no earlier injection: the P = 1 rhs, bitwise
            assert np.array_equal(np.asarray(r).view(np.int64), d["r1"].view(np.int64)), n
        if ur:
            out[0] = d["r6"] - d["r1"]                  # exact (Sterbenz): r1 + (r6 - r1) == r6
        if us:
            out[1] = d["y6"] - d["y1"]
    return out


def _derivative_solve_log(A, op_consts, solve_consts, r):
    """make_pcg_solve(cg2d_operator, precond=cg2d_preconditioner, tol=DERIVATIVE_TOL, max_iters=DERIVATIVE_MAX_ITERS)
    of cg2d.py, the same `_pcg` call (cg2d_rule._pcg_solve), plus the record / the perturbation."""
    if S["inject"]:
        dl = jax.pure_callback(_host_delta, jax.ShapeDtypeStruct((2,) + r.shape, r.dtype), r)
        r = r + dl[0]
        y, it, rel = CR._pcg(lambda v: C.cg2d_operator(A, op_consts, v),
                             lambda v: C.cg2d_preconditioner(A, op_consts, v),
                             partial(CR.default_vdot, op_consts), r, float(C.DERIVATIVE_TOL),
                             int(C.DERIVATIVE_MAX_ITERS))
        return y + dl[1]
    if S["noise"]:
        f = jax.pure_callback(_host_flags, jax.ShapeDtypeStruct((2,), jnp.float64), jnp.sum(r))
        r = r + f[0] * (ULP * r * pattern(r.shape))
    y, it, rel = CR._pcg(lambda v: C.cg2d_operator(A, op_consts, v),
                         lambda v: C.cg2d_preconditioner(A, op_consts, v),
                         partial(CR.default_vdot, op_consts), r, float(C.DERIVATIVE_TOL), int(C.DERIVATIVE_MAX_ITERS))
    if S["noise"]:
        y = y + f[1] * (ULP * y * pattern(y.shape))
    else:
        jax.debug.callback(partial(_host_rec, "bwd"), _dev(), it, rel, r, y)
    return y


C._forward_solve = _forward_solve_log
C._derivative_solve = _derivative_solve_log


# ---------------------------------------------------------------- instrument 2: the adjoint state at step boundaries
def _is_fa(x):
    return isinstance(x, FArray)


def _tiled(tree):
    flat, _ = jax.tree.flatten_with_path(tree, is_leaf=_is_fa)
    return [(jax.tree_util.keystr(p), x) for p, x in flat if isinstance(x, FArray) and x.tiled
            and jnp.issubdtype(x.data.dtype, jnp.floating)]


@jax.custom_vjp
def tap(st, iloop, K):
    return st


def _tap_fwd(st, iloop, K):
    return st, (iloop, K)


def _tap_bwd(res, ct):
    iloop, K = res
    if S["leaves"]:
        # K = [boundary, leaf]: the noise on the one float leaf of the cotangent (jax.tree order) at that boundary
        f = (iloop == K[0]).astype(jnp.float64)
        flat, tdef = jax.tree.flatten(ct)
        out, n = [], 0
        names = [jax.tree_util.keystr(pth) for pth, _ in jax.tree.flatten_with_path(ct)[0]]
        S["leaf_names"] = []
        for nm, c in zip(names, flat):
            if jnp.issubdtype(c.dtype, jnp.floating):
                S["leaf_names"].append((nm, tuple(c.shape)))
                c = c + (f * (K[1] == n).astype(jnp.float64)) * (ULP * c * pattern(c.shape))
                n += 1
            out.append(c)
        ct = jax.tree.unflatten(tdef, out)
    elif S["noise"]:
        f = (iloop == K).astype(jnp.float64)

        def one(c):
            if not jnp.issubdtype(c.dtype, jnp.floating):
                return c
            return c + f * (ULP * c * pattern(c.shape))
        ct = jax.tree.map(one, ct)
    else:
        named = _tiled(ct)
        S["bnd_names"] = [n for n, _ in named]
        jax.debug.callback(partial(_host_rec, "bnd"), _dev(), iloop, *[x.data for _, x in named])
    return ct, None, None


tap.defvjp(_tap_fwd, _tap_bwd)

# ---------------------------------------------------------------- the objective
t0 = time.time()
EXP = ("global_ocean.cs32x15", "input_ad")
exp_dir = paths.UPSTREAM / "verification" / EXP[0]
m = Model(load_experiment(exp_dir, EXP[1]), make_rundir(exp_dir, EXP[1], OUT / "run"))
key = next((3, g.iarr) for g in ctrl_readparms_genarr(m.exp.run, 3, m.cfg) if g.file.strip() == "xx_theta")
a = AR.GenarrAdjoint(m, key=key)
model = a.model                            # run mode: the default model (operator passive)
theta = a.theta_farray()
if int(os.environ.get("MJX_SHARD_NSTEPS", "0")):      # the first k steps of the window only (as m4costshard_shardgrad)
    a.xs = a.xs[:int(os.environ["MJX_SHARD_NSTEPS"])]
NST = int(a.xs.shape[0])
R["nsteps"] = NST
R["t"]["setup"] = time.time() - t0


def J2(theta_data, mo, st0, xs, K):
    mo, s0 = AR.genarr_apply(m, key, theta_data, mo, st0)

    def step2(mo_, st, iloop):
        return a._stepfn(mo_, tap(st, iloop, K), iloop)
    s_n, _ = integrate(step2, mo, s0, xs, schedule=a.schedule, save_names=SAVE_NAMES)
    s_n = tap(s_n, jnp.asarray(NST + 1, jnp.int32), K)
    return AR.final_cost_gathered(m, mo, s_n)


def body(th, mo, s, x, sd, K):
    val, pull = jax.vjp(lambda t: J2(t.data, mo, s, x, K), th)
    return val, pull(sd)[0]


def assemble(kind, nproc, Tloc):
    """Per event: the records of every device, tiled arrays concatenated in device order (tile order)."""
    devs = sorted(d for (k, d) in S["rec"] if k == kind)
    if not devs:
        return []
    n = len(S["rec"][(kind, devs[0])])
    assert all(len(S["rec"][(kind, d)]) == n for d in devs), [len(S["rec"][(kind, d)]) for d in devs]
    ev = []
    for e in range(n):
        per = [S["rec"][(kind, d)][e] for d in devs]
        out = []
        for j, v0 in enumerate(per[0]):
            if np.ndim(v0) == 0:
                vals = [p[j] for p in per]
                assert all(np.array_equal(vals[0], v) for v in vals), (kind, e, j, vals)   # replicated scalars
                out.append(v0)
            else:
                out.append(np.concatenate([p[j] for p in per], axis=0))
        ev.append(out)
    return ev


K_NONE = jnp.asarray(-99, jnp.int32)
REC = {}
if WHAT == "log":
    ref1, ref6 = np.load(REF / "g1.npy"), np.load(REF / "g6.npy")
    for nproc in (1, 6):
        S["rec"] = {}
        t1 = time.time()
        if nproc == 1:
            S["axis"] = None
            f = jax.jit(body)
            J, g = jax.block_until_ready(f(theta, model, a.st0, a.xs, seed(1.0), K_NONE))
            g = np.asarray(g.data)
            Tloc = 0
            ref = ref1
        else:
            sh = TileSharding(EM.load_cube_maps(*EXP), nproc)
            S["axis"] = sh.axis
            th_s = tile_specs(sh, theta)
            f = sh.shard_map(body, in_specs=(th_s, tile_specs(sh, model), tile_specs(sh, a.st0), sh.REP, sh.REP,
                                             sh.REP), out_specs=(sh.REP, th_s))
            J, g = jax.block_until_ready(f(sh.put_tree(theta), place_model(sh, model), sh.put_tree(a.st0), a.xs,
                                           seed(1.0), K_NONE))
            g = np.asarray(g.data)[:sh.layout.nTiles]
            Tloc = sh.blocks.Tloc
            ref = ref6
        R[f"P{nproc}"] = dict(J=float(J), grad_bitwise_vs_ref=bits(g, ref) == 0, bits_vs_ref=bits(g, ref),
                              t=time.time() - t1)
        np.save(OUT / f"g{nproc}.npy", g)
        REC[nproc] = {k: assemble(k, nproc, Tloc) for k in ("fwd", "bwd", "bnd")}
        R[f"P{nproc}"]["fwd"] = [dict(numIters=int(e[0]), firstResidual=float(e[1]), lastResidual=float(e[2]),
                                      minResidualSq=float(e[3])) for e in REC[nproc]["fwd"]]
        R[f"P{nproc}"]["bwd"] = [dict(iters=int(e[0]), rel_residual=float(e[1])) for e in REC[nproc]["bwd"]]
        R[f"P{nproc}"]["bnd_iloop"] = [int(e[0]) for e in REC[nproc]["bnd"]]
        save()
        del f
        release()
    g1, g6 = np.load(OUT / "g1.npy"), np.load(OUT / "g6.npy")
    R["grad_rel_P6_vs_P1"] = relmax(g6, g1)
    # forward CG2D: counts and residuals per call
    f1, f6 = R["P1"]["fwd"], R["P6"]["fwd"]
    R["fwd_identical"] = f1 == f6
    # backward CG2D: counts, residuals, rhs / solution differences per solve
    cmp = []
    for n, (e1, e6) in enumerate(zip(REC[1]["bwd"], REC[6]["bwd"])):
        r1, y1, r6, y6 = e1[2], e1[3], e6[2], e6[3]
        dr, dy = relmax(r6, r1), relmax(y6, y1)
        loc = np.unravel_index(np.argmax(np.abs(y6 - y1)), y1.shape)
        cmp.append(dict(n=n, iters_P1=int(e1[0]), iters_P6=int(e6[0]), rel_res_P1=float(e1[1]),
                        rel_res_P6=float(e6[1]), rhs_rel=dr, sol_rel=dy, amplification=(dy / dr if dr else None),
                        rhs_bits=bits(r6, r1), sol_bits=bits(y6, y1), sol_maxdiff_at=[int(i) for i in loc],
                        sol_rel_per_tile=[relmax(y6[t], y1[t]) if np.max(np.abs(y1[t])) > 0 else 0.0
                                          for t in range(y1.shape[0])]))
        np.savez(OUT / f"bwd{n}.npz", r1=r1, y1=y1, r6=r6, y6=y6)
    R["bwd_cmp"] = cmp
    # step boundaries: the adjoint state, P = 6 - P = 1, per tiled field
    names = S.get("bnd_names", [])
    bcmp = []
    for e1, e6 in zip(REC[1]["bnd"], REC[6]["bnd"]):
        assert int(e1[0]) == int(e6[0])
        per = {}
        worst = (0.0, None)
        for nm, x1, x6 in zip(names, e1[1:], e6[1:]):
            s = float(np.max(np.abs(x1)))
            d = float(np.max(np.abs(x6 - x1)))
            if s == 0 and d == 0:
                continue
            rel = d / s if s else float("inf")
            loc = np.unravel_index(np.argmax(np.abs(x6 - x1)), x1.shape)
            per[nm] = dict(rel=rel, absmax=s, maxdiff=d, at=[int(i) for i in loc], bits=bits(x6, x1))
            if rel > worst[0]:
                worst = (rel, nm)
        bcmp.append(dict(iloop=int(e1[0]), worst=worst, fields=per))
    R["bnd_cmp"] = bcmp
    save()
elif WHAT == "inject":
    # the actual P = 6 - P = 1 differences of the backward CG2D solves (REF = the log run's OUT_DIR with bwd<n>.npz,
    # g1.npy) put into the P = 1 run: rhs of solve n alone, solution of solve n alone (r6 / y6 exactly), every rhs
    ref1 = np.load(REF / "g1.npy")
    S["axis"] = None
    f = jax.jit(body)
    runs = [("base", {})] + [(f"rhs n={n}", {n: (True, False)}) for n in range(NST)] \
        + [(f"sol n={n}", {n: (False, True)}) for n in range(NST)] \
        + [("rhs all", {n: (True, False) for n in range(NST)})]
    g6 = np.load(REF / "g6.npy")
    gb = None
    R["runs"] = []
    for name, sel_ in runs:
        S["nsolve"], S["delta"] = 0, sel_
        J, g = jax.block_until_ready(f(theta, model, a.st0, a.xs, seed(1.0), K_NONE))
        g = np.asarray(g.data)
        if gb is None:
            gb = g
            R["base"] = dict(J=float(J), grad_bitwise_vs_ref=bits(g, ref1) == 0, nsolve=S["nsolve"],
                             P6_rel=relmax(g6, ref1))
        else:
            R["runs"].append(dict(name=name, rel=relmax(g, gb), bits=bits(g, gb), nsolve=S["nsolve"],
                                  rel_to_P6=relmax(g, g6)))
        save()
elif WHAT == "leaves":
    # 1-ulp noise on one cotangent leaf of the adjoint state at boundary K (K = 2: lambda_1, entering the reverse of
    # step 1; K = 5: lambda_4): which field carries the sensitivity of the gradient to last-bit changes
    ref1 = np.load(REF / "g1.npy")
    S["axis"] = None
    f = jax.jit(body)
    J, gb = jax.block_until_ready(f(theta, model, a.st0, a.xs, seed(1.0), jnp.asarray([-99, -1], jnp.int32)))
    gb = np.asarray(gb.data)
    R["base"] = dict(J=float(J), grad_bitwise_vs_ref=bits(gb, ref1) == 0, bits_vs_ref=bits(gb, ref1))
    R["leaf_names"] = [list(x) for x in S["leaf_names"]]
    R["runs"] = []
    for K in (2, 5):
        for n, (nm, shp) in enumerate(S["leaf_names"]):
            J, g = jax.block_until_ready(f(theta, model, a.st0, a.xs, seed(1.0), jnp.asarray([K, n], jnp.int32)))
            g = np.asarray(g.data)
            if bits(g, gb):
                R["runs"].append(dict(K=K, leaf=nm, shape=list(shp), rel=relmax(g, gb), bits=bits(g, gb)))
        save()
    R["runs"].sort(key=lambda r: (r["K"], -r["rel"]))
    save()
else:
    ref1 = np.load(REF / "g1.npy")
    S["axis"] = None
    f = jax.jit(body)
    runs = [("base", -99, (-1, ""))] + [(f"bnd K={k}", k, (-1, "")) for k in range(NST + 1, 0, -1)] \
        + [(f"solve rhs n={n}", -99, (n, "rhs")) for n in range(NST)] \
        + [(f"solve sol n={n}", -99, (n, "sol")) for n in (0, NST - 1)]
    gb = None
    R["runs"] = []
    for name, K, pert in runs:
        t1 = time.time()
        S["nsolve"], S["pert"] = 0, pert
        J, g = jax.block_until_ready(f(theta, model, a.st0, a.xs, seed(1.0), jnp.asarray(K, jnp.int32)))
        g = np.asarray(g.data)
        if gb is None:
            gb = g
            R["base"] = dict(J=float(J), grad_bitwise_vs_ref=bits(g, ref1) == 0, bits_vs_ref=bits(g, ref1),
                             nsolve=S["nsolve"])
            np.save(OUT / "g_base.npy", g)
        else:
            d = np.abs(g - gb)
            loc = np.unravel_index(np.argmax(d), d.shape)
            lev = [float(np.max(d[:, k]) / np.max(np.abs(gb))) for k in range(d.shape[1])]
            R["runs"].append(dict(name=name, J=float(J), rel=relmax(g, gb), bits=bits(g, gb),
                                  at=[int(i) for i in loc], rel_per_level=lev, nsolve=S["nsolve"],
                                  t=time.time() - t1))
            np.save(OUT / f"g_{name.replace(' ', '_').replace('=', '')}.npy", g)
        save()
R["t"]["total"] = time.time() - t0
save()
print(json.dumps({k: v for k, v in R.items() if k not in ("bnd_cmp",)}, indent=1, default=float)[:20000])
