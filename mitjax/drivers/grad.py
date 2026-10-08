"""Gradient drivers over a window of model steps and the gradient trust utilities (plan Task 7c).

    theta -> J:   st0 = init_fn(theta, st0_base); model = params_fn(theta, model_base);
                  (st_n, acc) = integrate(step, model, st0, xs); J = final_cost(model, st_n) + acc

Copied from the ECCO port's `adjoint/grad.py` (R) with its ECCO specifics removed (AdjointConfig semantics, the State
class with a field dict); the State is any pytree (dict, NamedTuple, dataclass). `theta` is any pytree of controls;
`init_fn(theta, st)` puts initial-state controls into the State (default `replace_fields`), `params_fn(theta, model)`
puts parameter or field controls into the model (default: identity). Both run inside the differentiated function.
`cost(model, st, x)` is accumulated after every step, `final_cost(model, st_n)` applied to the last State.

Drivers (same J bitwise, same gradients; test_checkpoint.py):
  value_and_grad(...)          one jitted jax.value_and_grad of the whole window, schedule "none" | "step" | "sqrt";
                               with stats_fn also the per-step cotangent statistics (checkpoint.py, adjoint monitor)
  chunked_value_and_grad(...)  the window in chunks of `chunk_steps` steps: forward with every `boundary_stride`-th
                               chunk-boundary State parked on the HOST, then a reverse walk over the chunks, one jitted
                               function that calls jax.vjp per chunk ([L-AD-35]: never jit the function vjp returns),
                               carrying the State cotangent; device memory is one chunk whatever the window length
                               ([L-AD-34]). Returns ChunkedGrad with the per-chunk cotangent-norm trace.
The model is an argument of every jitted function ([L-ARCH-5]); `captured_constants` measures what a lowering captured.

Trust utilities ([L-AD-25]: a gradient is not trusted until these pass):
  dot_test(f, x, v, w)          <J v, w> (jax.jvp) vs <v, J^T w> (jax.vjp), at several tangent amplitudes
  fd_sweep(J, x, d, hs, ad)     central differences over step sizes h against the AD directional derivative, with the
                                forward noise floor from repeated evaluations
  cotangent_norm, field_norms, amplification   per-step reverse growth: median, log-spread, worst 3 consecutive
                                chunks (median <= 1.010/step, log-spread <= 0.020, worst-3 <= 1.030/step; [L-AD-23])
"""

import dataclasses
import re
import time
import warnings
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.drivers.checkpoint import SAVE_NAMES, integrate, make_sinks, n_steps, prepare_state, take_steps

# ------------------------------------------------------------------------------------------------- control binding


def replace_fields(theta, st):
    """Default init_fn: theta is a dict {State field: array}; the State with those fields replaced (dict, NamedTuple or
    dataclass State)."""
    if not theta:
        return st
    if isinstance(st, dict):
        unknown = set(theta) - set(st)
        if unknown:
            raise KeyError(f"controls not in the State: {sorted(unknown)}")
        return {**st, **theta}
    if hasattr(st, "_replace"):
        return st._replace(**theta)
    if dataclasses.is_dataclass(st):
        return dataclasses.replace(st, **theta)
    raise TypeError(f"replace_fields: no field replacement for a State of type {type(st).__name__}; pass init_fn")


def keep_model(theta, model):
    """Default params_fn: the model does not depend on theta."""
    return model


def _objective(step, *, schedule, segments, cost, final_cost, init_fn, params_fn, save_names=SAVE_NAMES,
               stats_fn=None):
    def J(theta, model, st0, xs, sinks=None):
        m = params_fn(theta, model)
        s0 = init_fn(theta, st0)
        s_n, acc = integrate(step, m, s0, xs, schedule=schedule, segments=segments, cost=cost, save_names=save_names,
                             stats_fn=stats_fn, sinks=sinks)
        out = acc
        if final_cost is not None:
            out = out + final_cost(m, s_n)
        return out
    return J


# Jitted whole-window functions per configuration, kept across calls: a repeat (repeat floor, FD sweep, linearity
# guard) would otherwise build a new jax.jit and retrace + recompile the window (R2, 10 steps, sqrt: 395 s on a CPU node
# for a repeat; lane SHARDGRAD). The key holds the function objects (identity): pass the same step / cost / init_fn /
# params_fn / stats_fn objects to hit it. Small on purpose: each entry holds executables (as _CHUNK_CACHE below).
_VG_CACHE = {}
_VG_CACHE_MAX = 4


def _cached(key, build):
    if key in _VG_CACHE:
        return _VG_CACHE[key]
    while len(_VG_CACHE) >= _VG_CACHE_MAX:
        _VG_CACHE.pop(next(iter(_VG_CACHE)))
    _VG_CACHE[key] = f = build()
    return f


def value_and_grad(step, theta, model, st0, xs, *, final_cost=None, cost=None, schedule="step", segments=None,
                   init_fn=replace_fields, params_fn=keep_model, save_names=SAVE_NAMES, stats_fn=None, jit=True):
    """(J, dJ/dtheta) over the whole window in one jax.value_and_grad; with stats_fn (f(model, State cotangent) ->
    pytree of scalars) (J, dJ/dtheta, stats) where stats has a leading axis of n + 1: the statistics of dJ/dState at
    the step boundaries 0..n (0 = the initial State, n = the final one). Memory grows with the window ("step": one
    State per step; "sqrt": ~2 sqrt(N)); for long windows use chunked_value_and_grad. With jit, the jitted function is
    kept per configuration (_VG_CACHE): a repeat with the same function objects reuses the compiled program."""
    key = ("vg", step, schedule, segments, cost, final_cost, init_fn, params_fn, tuple(save_names), stats_fn)

    def build():
        J = _objective(step, schedule=schedule, segments=segments, cost=cost, final_cost=final_cost, init_fn=init_fn,
                       params_fn=params_fn, save_names=save_names, stats_fn=stats_fn)
        if stats_fn is None:
            return jax.value_and_grad(J)

        def vg_stats(theta, model, st0, xs, sinks):
            val, (g, stats) = jax.value_and_grad(lambda th, sk: J(th, model, st0, xs, sk), argnums=(0, 1))(theta,
                                                                                                           sinks)
            return val, g, stats
        return vg_stats

    vg = _cached(key, lambda: jax.jit(build())) if jit else build()
    if stats_fn is None:
        return vg(theta, model, st0, xs)
    sinks = make_sinks(stats_fn, model, prepare_state(step, model, st0, jax.tree.map(lambda a: a[0], xs)),
                       n_steps(xs))
    return vg(theta, model, st0, xs, sinks)


def objective(step, theta, model, st0, xs, *, final_cost=None, cost=None, init_fn=replace_fields,
              params_fn=keep_model, jit=True):
    """J(theta) alone (forward only, schedule "none"): for finite differences. With jit, kept per configuration as
    value_and_grad's."""
    def build():
        return _objective(step, schedule="none", segments=None, cost=cost, final_cost=final_cost, init_fn=init_fn,
                          params_fn=params_fn)
    key = ("objective", step, cost, final_cost, init_fn, params_fn)
    J = _cached(key, lambda: jax.jit(build())) if jit else build()
    return J(theta, model, st0, xs)


# ------------------------------------------------------------------------------------------- what a lowering captured

_CAPTURED = re.compile(r"\(([0-9.]+)\s*([KMGT]?B) total\)")
_UNIT = {"B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3, "TB": 1024 ** 4}


def captured_constants(fn, *args):
    """Bytes of array constants that lowering `fn(*args)` captured (closed-over arrays become compile-time constants:
    [L-ARCH-5], [L-XLA-4]); 0 when everything arrives as an argument. Reads JAX's own report (config
    `jax_captured_constants_warn_bytes`, set to 0 for the call). fn: a jitted function or a plain one (jitted here)."""
    f = fn if hasattr(fn, "lower") else jax.jit(fn)
    old = jax.config.jax_captured_constants_warn_bytes
    jax.config.update("jax_captured_constants_warn_bytes", 0)
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            f.lower(*args)
    finally:
        jax.config.update("jax_captured_constants_warn_bytes", old)
    total = 0.0
    for w in rec:
        m = _CAPTURED.search(str(w.message))
        if m:
            total += float(m.group(1)) * _UNIT[m.group(2)]
    return int(round(total))


# ----------------------------------------------------------------------------------------- chunked reverse accumulation


class ChunkedGrad(NamedTuple):
    loss: float
    grad: Any
    trace: list            # norm of the State cotangent at every chunk boundary, end of window first
    host_gb: float         # chunk-boundary States parked on the host
    forward_seconds: float
    reverse_seconds: float
    n_chunks: int
    chunk_steps: int
    boundary_stride: int
    schedule: str
    field_trace: list = None   # per chunk boundary (same order as `trace`): {State leaf path: cotangent norm}


def _floating(x):
    return hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)


def field_norms(ct):
    """{State leaf path: Euclidean norm} of a State cotangent (floating leaves only): a screen can say WHICH field
    grows; the total norm mixes units ([L-AD-27])."""
    return {jax.tree_util.keystr(p): float(jnp.sqrt(jnp.sum(jnp.square(v))))
            for p, v in jax.tree_util.tree_flatten_with_path(ct)[0] if _floating(v)}


def cotangent_norm(tree):
    """Euclidean norm over every floating leaf of a cotangent pytree (float0 / integer leaves skipped)."""
    leaves = [x for x in jax.tree.leaves(tree) if _floating(x)]
    if not leaves:
        return 0.0
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves)))


def _float0_like(x):
    return np.zeros(np.shape(x), jax.dtypes.float0)


def _state_ct(ct_st):
    """Cotangent of a State to feed back into a VJP: integer leaves (the iteration counter) get float0 zeros."""
    return jax.tree.map(lambda x: x if jnp.issubdtype(jnp.result_type(x), jnp.floating) else _float0_like(x), ct_st)


def _add(a, b):
    return b if a is None else jax.tree.map(jnp.add, a, b)


# Jitted chunk forward/VJP per configuration, kept across calls (a repeat or an FD sweep would otherwise recompile the
# chunk; [L-PERF-3]). Small on purpose: each entry holds executables; one configuration per process is the intended use.
_CHUNK_CACHE = {}
_CHUNK_CACHE_MAX = 2


def _chunk_fns(step, schedule, segments, cost, params_fn, save_names):
    key = (step, schedule, segments, cost, params_fn, save_names)
    if key in _CHUNK_CACHE:
        return _CHUNK_CACHE[key]

    def chunk(theta, model, carry, xs):
        return integrate(step, params_fn(theta, model), carry[0], xs, schedule=schedule, segments=segments,
                         cost=cost, acc0=carry[1], save_names=save_names)

    @jax.jit
    def bwd(theta, model, carry, xs, cot):
        _, pull = jax.vjp(lambda th, c: chunk(th, model, c, xs), theta, carry)
        return pull(cot)

    while len(_CHUNK_CACHE) >= _CHUNK_CACHE_MAX:
        _CHUNK_CACHE.pop(next(iter(_CHUNK_CACHE)))
    _CHUNK_CACHE[key] = (jax.jit(chunk), bwd)
    return _CHUNK_CACHE[key]


def chunked_value_and_grad(step, theta, model, st0, *, n_chunks, chunk_steps, xs_fn, final_cost=None, cost=None,
                           schedule="step", segments=None, init_fn=replace_fields, params_fn=keep_model,
                           boundary_stride=1, save_names=SAVE_NAMES, on_chunk=None, on_cot=None, log=None):
    """(J, dJ/dtheta) with chunked reverse accumulation. xs_fn(c) -> the stacked inputs of chunk c (host numpy,
    `chunk_steps` steps, built fresh per call: it is called again in the reverse walk). Boundaries every
    `boundary_stride` chunks are kept on the host (always the window start and end); the spans between are rebuilt
    once each in the reverse walk. on_chunk(c, cot_norm) after each reverse chunk; on_cot(ct_state) with the State
    cotangent itself at every chunk boundary, the seed first, in `trace` order. Returns ChunkedGrad."""
    stride = max(1, int(boundary_stride))
    say = log or (lambda *a: None)
    fwd, bwd = _chunk_fns(step, schedule, segments, cost, params_fn, tuple(save_names))

    def seed(theta, model, carry):
        s, a = carry
        out = a
        if final_cost is not None:
            out = out + final_cost(params_fn(theta, model), s)
        return out

    # forward, boundaries to the host
    t0 = time.time()
    x0 = jax.tree.map(lambda a: a[0], xs_fn(0))
    st0 = prepare_state(step, params_fn(theta, model), st0, x0)
    s0 = jax.jit(init_fn)(theta, st0)
    carry = (s0, jnp.zeros((), jnp.float64))
    kept = {0: jax.device_get(carry)}
    for c in range(n_chunks):
        carry = jax.block_until_ready(fwd(theta, model, carry, jax.device_put(xs_fn(c))))
        if (c + 1) % stride == 0 or c == n_chunks - 1:
            kept[c + 1] = jax.device_get(carry)
    host_gb = sum(np.asarray(a).nbytes for a in jax.tree.leaves(kept)) / 1e9
    loss, pull = jax.vjp(lambda th, c: seed(th, model, c), theta, carry)
    g_th, cot = pull(jnp.ones((), jnp.float64))
    grad = g_th if final_cost is not None else None
    cot = (_state_ct(cot[0]), cot[1])
    fwd_t = time.time() - t0
    trace = [cotangent_norm(cot[0])]
    ftrace = [field_norms(cot[0])]
    if on_cot is not None:
        on_cot(cot[0])
    say(f"chunked forward: {n_chunks} chunks x {chunk_steps} steps, J = {float(loss):.16e}, host {host_gb:.2f} GB, "
        f"{fwd_t:.1f} s")
    del carry

    # reverse walk
    t1 = time.time()
    for lo in reversed(range(0, n_chunks, stride)):
        hi = min(lo + stride, n_chunks)
        span = {lo: kept[lo]}
        c_in = jax.device_put(kept[lo])
        for c in range(lo, hi - 1):   # rebuild the span between kept boundaries (stride > 1)
            c_in = jax.block_until_ready(fwd(theta, model, c_in, jax.device_put(xs_fn(c))))
            span[c + 1] = jax.device_get(c_in)
        del c_in
        for c in reversed(range(lo, hi)):
            g_c, cot = bwd(theta, model, jax.device_put(span[c]), jax.device_put(xs_fn(c)), cot)
            cot = (_state_ct(cot[0]), cot[1])
            grad = _add(grad, g_c)
            trace.append(cotangent_norm(cot[0]))
            ftrace.append(field_norms(cot[0]))
            if on_cot is not None:
                on_cot(cot[0])
            if on_chunk is not None:
                on_chunk(c, trace[-1])
        del span
    # initial state: theta -> st0
    _, pull0 = jax.vjp(lambda th: init_fn(th, st0), theta)
    grad = _add(grad, pull0(cot[0])[0])
    rev_t = time.time() - t1
    say(f"chunked reverse: {rev_t:.1f} s")
    return ChunkedGrad(float(loss), grad, trace, host_gb, fwd_t, rev_t, n_chunks, chunk_steps, stride, schedule,
                       ftrace)


def chunks_of(xs, chunk_steps):
    """xs_fn for a window whose stacked inputs are already on the host: chunk c = steps c*K .. (c+1)*K-1."""
    n = n_steps(xs)
    if n % chunk_steps:
        raise ValueError(f"{n} steps is not a multiple of chunk_steps={chunk_steps}")
    return (lambda c: take_steps(xs, c * chunk_steps, (c + 1) * chunk_steps)), n // chunk_steps


# ------------------------------------------------------------------------------------------------- trust utilities


def tree_vdot(a, b):
    """Sum over floating leaves of <a, b> (float64)."""
    tot = 0.0
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        if jnp.issubdtype(jnp.result_type(x), jnp.floating):
            tot += float(jnp.vdot(jnp.ravel(x), jnp.ravel(y)))
    return tot


def dot_test(f, x, v, w, amps=None, jit=True):
    """Adjoint dot test of y = f(x): <J v, w> (jax.jvp, the tangent-linear model) vs <v, J^T w> (jax.vjp, the
    adjoint). v: tangent like x; w: cotangent like f(x). Returns (lhs, rhs, rel); with `amps` (tangent amplitudes,
    e.g. (1.0, 1e-6): a TL that is not linear in the tangent fails at small amplitude) a list of (amp, lhs, rhs, rel)
    from one adjoint and one compiled tangent-linear function."""
    jvp = lambda x, v: jax.jvp(f, (x,), (v,))[1]  # noqa: E731
    vjp = lambda x, w: jax.vjp(f, x)[1](w)[0]     # noqa: E731
    if jit:
        jvp, vjp = jax.jit(jvp), jax.jit(vjp)
    rhs1 = tree_vdot(v, vjp(x, w))
    out = []
    for amp in ((1.0,) if amps is None else amps):
        lhs = tree_vdot(jvp(x, jax.tree.map(lambda z: z * amp, v)), w)
        rhs = amp * rhs1
        out.append((amp, lhs, rhs, abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)))
    return out[0][1:] if amps is None else out


class FDRow(NamedTuple):
    h: float
    fd: float
    ad: float
    rel_err: float
    noise: float    # forward noise floor / h (the FD error the noise alone can cause)


def fd_sweep(J, x, d, hs, ad, repeats=2):
    """Central differences (J(x + h d) - J(x - h d)) / 2h for every h in hs against `ad` = <dJ/dx, d>. The forward
    noise floor is the spread of `repeats` evaluations of J(x) (0 on a deterministic CPU run; GPU atomics make it
    nonzero); a row means something only where |fd - ad| is well above the noise. Returns [FDRow]."""
    j0 = np.array([float(J(x)) for _ in range(max(1, repeats))])
    spread = float(np.max(j0) - np.min(j0))      # NaN if any evaluation is NaN (Python's max() would skip it)
    rows = []
    for h in hs:
        xp = jax.tree.map(lambda a, b: a + h * b, x, d)
        xm = jax.tree.map(lambda a, b: a - h * b, x, d)
        fd = (float(J(xp)) - float(J(xm))) / (2.0 * h)
        rows.append(FDRow(float(h), fd, float(ad), abs(fd - ad) / max(abs(ad), 1e-300), spread / h))
    return rows


class Amplification(NamedTuple):
    median: float        # per-step growth of the cotangent norm, median over chunks
    log_spread: float    # std of log per-step rates
    worst3: float        # worst per-step growth over 3 consecutive chunks
    rates: tuple         # per-chunk per-step growth, end of window first
    passes: bool         # median <= 1.010, log-spread <= 0.020, worst-3 <= 1.030 (fesom_jax bars), and no bad rate
    bad_rates: int = 0   # rates that are 0 or not finite (a dead or broken reverse path): any one fails `passes`


def amplification(trace, steps_per_chunk, bars=(1.010, 0.020, 1.030)):
    """Per-step reverse growth from a chunk-boundary cotangent-norm trace (trace[0] = seed at the window end, each
    further entry one chunk earlier)."""
    tr = [float(t) for t in trace]
    k = max(1, int(steps_per_chunk))
    rates = [(tr[i + 1] / tr[i]) ** (1.0 / k) if tr[i] > 0 else float("nan") for i in range(len(tr) - 1)]
    fin = np.array([r for r in rates if np.isfinite(r) and r > 0])
    bad = len(rates) - int(fin.size)      # a zero or non-finite rate (dead reverse path, NaN) is reported and fails
    if fin.size == 0:
        return Amplification(float("nan"), float("nan"), float("nan"), tuple(rates), False, bad)
    w = max(1, min(3, len(tr) - 1))
    sus = [(tr[i + w] / tr[i]) ** (1.0 / (k * w)) for i in range(len(tr) - w) if tr[i] > 0]
    worst = float(np.max(sus)) if sus else float("nan")
    med, spread = float(np.median(fin)), float(np.std(np.log(fin)))
    ok = bad == 0 and med <= bars[0] and spread <= bars[1] and worst <= bars[2]
    return Amplification(med, spread, worst, tuple(rates), bool(ok), bad)
