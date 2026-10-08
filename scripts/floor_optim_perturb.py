#!/usr/bin/env python3
"""The 1-ulp floor of plan decision 18 for tutorial_global_oce_optim/input_ad's R5 window, and its sharded gradient
differences (lane FLOOR; measurement script, not a gate).

Sibling of scripts/m4costshard_cg2dlog.py's `perturb` mode (written for cs32x15's GenarrAdjoint objective, 6 devices):
the same method on the objective of mitjax/tests/test_r5_sharded_grad_cpu.py -- fc(xx) = COST_FINAL(THE_MAIN_LOOP(
CTRL_MAP_INI_GENTIM2D(xx))), xx = the xx_qnet record, 10 steps, per-step checkpoint (adjoint_run.sharded_problem,
mitjax/tests/r5_shardgrad.Setup), jax.vjp with seed 1.

Modes:
  perturb  P = 1 only. (1) the uninstrumented reference g1: r5_shardgrad.Setup.p1 (the test's single-device driver,
           sharded_grad.value_and_grad_seed_fn). (2) the same J with a custom_vjp identity `tap(st, iloop, K)` on the
           step state before every step iloop and before COST_FINAL (iloop = nTimeSteps + 1), whose backward adds a
           1-ulp relative +-1 pattern (2^-52 x c x pattern, numpy seed 0 per shape, as m4costshard_cg2dlog.pattern) to
           every floating cotangent leaf when iloop == K (K traced: one compile). Runs: base (K = -99: no noise; must
           equal g1 bit for bit), then K = nTimeSteps + 1 .. 1. Each: max|g - g_base| / max|g_base| (the R5 measure).
           floor = the largest of the boundary values (decision 18).
  shard    P = 1 (Setup.p1) and r5_shardgrad.Sharded at each listed P (fake CPU devices, 4): J bitwise, max|g_P - g_1|
           / max|g_1|, the differing points, the location of the largest difference, finiteness, padding exactly 0.

    floor_optim_perturb.py OUT_DIR perturb
    floor_optim_perturb.py OUT_DIR shard 2,3,4
"""
import json
import sys
import time
from pathlib import Path

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()                       # the gate flags of the R5 tests (4 fake CPU devices)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from mitjax.drivers.checkpoint import SAVE_NAMES, integrate  # noqa: E402
from mitjax.drivers.sharded_grad import seed  # noqa: E402
from mitjax.tests import r5_shardgrad as RS  # noqa: E402

OUT, WHAT = Path(sys.argv[1]), sys.argv[2]
assert WHAT in ("perturb", "shard"), WHAT
if (OUT / "result.json").exists():
    raise SystemExit(f"{OUT}/result.json exists: choose a new OUT_DIR")
OUT.mkdir(parents=True, exist_ok=True)

ULP = 2.0 ** -52
R = {"what": WHAT, "t": {}}


def save():
    (OUT / "result.json").write_text(json.dumps(R, indent=1, default=float))


def relmax(a, b):
    d = float(np.max(np.abs(a - b)))
    s = float(np.max(np.abs(b)))
    return d / s if s > 0 else (0.0 if d == 0 else float("inf"))


_PAT = {}


def pattern(shape):
    """A fixed +-1 pattern of `shape` (numpy constant; seed 0), as m4costshard_cg2dlog.pattern."""
    if shape not in _PAT:
        _PAT[shape] = np.where(np.random.default_rng(0).random(shape) < 0.5, -1.0, 1.0)
    return _PAT[shape]


@jax.custom_vjp
def tap(st, iloop, K):
    return st


def _tap_fwd(st, iloop, K):
    return st, (iloop, K)


def _tap_bwd(res, ct):
    iloop, K = res
    f = (iloop == K).astype(jnp.float64)

    def one(c):
        if not jnp.issubdtype(c.dtype, jnp.floating):
            return c
        return c + f * (ULP * c * pattern(c.shape))
    return jax.tree.map(one, ct), None, None


tap.defvjp(_tap_fwd, _tap_bwd)

t0 = time.time()
su = RS.Setup("floor")
NST = int(su.xs.shape[0])
R["nsteps"] = NST
R["t"]["setup"] = time.time() - t0

t1 = time.time()
J1, g1 = su.p1()
RS.release()
R["J1"] = J1
R["J1_repr"] = repr(J1)
R["g1_absmax"] = float(np.max(np.abs(g1)))
R["t"]["p1"] = time.time() - t1
np.save(OUT / "g1.npy", g1)
save()

if WHAT == "perturb":
    def J2(theta, mo, st0, xs, K):
        # grad._objective's J (params_fn, init_fn = identity on st0, integrate, acc + final_cost) with the taps
        m_ = su.params_fn(theta, mo)

        def step2(mo_, st, iloop):
            return su.step(mo_, tap(st, iloop, K), iloop)
        s_n, acc = integrate(step2, m_, st0, xs, schedule=RS.SCHEDULE, segments=None, cost=None,
                             save_names=tuple(SAVE_NAMES))
        s_n = tap(s_n, jnp.asarray(NST + 1, jnp.int32), K)
        return acc + su.final_cost(m_, s_n)

    def body(th, mo, s, x, sd, K):
        val, pull = jax.vjp(lambda t: J2(t, mo, s, x, K), th)
        return val, pull(sd)[0]

    f = jax.jit(body)
    runs = [("base", -99)] + [(f"bnd K={k}", k) for k in range(NST + 1, 0, -1)]
    gb = None
    R["runs"] = []
    for name, K in runs:
        t1 = time.time()
        J, g = jax.block_until_ready(f(su.theta, su.model, su.st0, su.xs, seed(1.0), jnp.asarray(K, jnp.int32)))
        g = np.asarray(g.data)
        if gb is None:
            gb = g
            R["base"] = dict(J=float(J), J_bitwise=float(J) == J1, grad_bitwise_vs_ref=RS.bits_differ(g, g1) == 0,
                             bits_vs_ref=RS.bits_differ(g, g1), t=time.time() - t1)
            np.save(OUT / "g_base.npy", g)
        else:
            d = np.abs(g - gb)
            loc = np.unravel_index(np.argmax(d), d.shape)
            R["runs"].append(dict(name=name, K=K, J=float(J), rel=relmax(g, gb), bits=RS.bits_differ(g, gb),
                                  at=[int(i) for i in loc], t=time.time() - t1))
        save()
    bnd = [x["rel"] for x in R["runs"]]
    R["floor"] = max(bnd)
    R["floor_at"] = R["runs"][int(np.argmax(bnd))]["name"]
    R["C10_floor"] = 10 * R["floor"]
else:
    procs = [int(p) for p in sys.argv[3].split(",")]
    for p in procs:
        t1 = time.time()
        sd = RS.Sharded(su, p)
        J, gp = sd.value_and_grad()
        g = sd.unpad(gp)
        d = np.abs(g - g1)
        loc = np.unravel_index(np.argmax(d), d.shape)
        R[f"P{p}"] = dict(J=J, J_bitwise=J == J1, rel=RS.rel(g, g1), bits_differ=RS.bits_differ(g, g1),
                          at=[int(i) for i in loc], finite=bool(np.all(np.isfinite(gp))),
                          pad_zero=bool(np.all(gp[sd.sh.layout.nTiles:] == 0.0)), Tpad=int(gp.shape[0]),
                          t=time.time() - t1)
        np.save(OUT / f"g{p}.npy", g)
        save()
        del sd
        RS.release()
    gs = {p: np.load(OUT / f"g{p}.npy") for p in procs}
    R["bitwise_pairs"] = {f"P{a}-P{b}": RS.bits_differ(gs[a], gs[b]) == 0 for a in procs for b in procs if a < b}
R["t"]["total"] = time.time() - t0
save()
print(json.dumps(R, indent=1, default=float)[:20000])
