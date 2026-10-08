"""Time integration for reverse mode: the model step in a `lax.scan` with rematerialisation schedules (plan Task 7c).

    model = ...                                   # pytree of everything the step reads besides State and inputs
    st_n, acc = integrate(step, model, st0, xs, schedule="step")   # step(model, st, x) -> st

Copied from the ECCO port's `adjoint/checkpoint.py` (R) with its LLC/ECCO specifics removed (EXF record windows,
AdjointConfig/make_step, the FORWARD_STEP import, fields added to the State on the fly); the c66g -> master audit of
the Fortran it encodes is docs/AUDIT_C66G_MASTER.md, section "Checkpoint stack and gradient drivers".

What the Fortran fixes here: the step is one call of master's FORWARD_STEP (model/src/forward_step.F; its call order is
re-derived in Task 4) and the State is everything FORWARD_STEP carries from one call to the next, including the
Adams-Bashforth histories (adams_bashforth3.F:64 @63cdc0b: `gTrNm(..., Nr, nSx, nSy, 2)` keeps the two previous
tendencies per field, indexed by the parity of myIter, `m1 = 1 + MOD(myIter+1,2)`, `m2 = 1 + MOD(myIter,2)` at :80-81;
the tendency branch :115-123 separates the history index `kl = kArg` from the tendency index `k = MIN(kArg, kSize)`). The driver itself is model-independent:
- the scan carry is (State, acc) and nothing else: a step reads only its State, the model and its own input x;
  `acc` accumulates an optional running cost cost(model, st_new, x) after every step (a time-integrated objective
  needs no stored trajectory);
- `model` and the per-step inputs `xs` are ARGUMENTS of the jitted function, never closures: closed-over model data
  became compile-time constants in the ECCO port (275 s vs 59 s compile, constant folding, a CUBIN too large) and
  closed-over floats break the bitwise gates ([L-ARCH-5], [L-XLA-4]; grad.captured_constants measures it);
- a step that changes the State's structure or dtypes is an error (prepare_state), not silently patched.

Schedules (what the reverse pass stores; the forward values are bitwise the same in every schedule, tested):
  "none"   plain scan: every intermediate of every step is kept (tiny windows and tests only)
  "step"   per-step remat: the State of every step boundary is kept; each step is recomputed once in the reverse pass
  "sqrt"   two-level remat: S ~ sqrt(N) outer segments (`sqrt_segments`), each a checkpointed scan of per-step
           checkpointed steps; S + N/S States stored, every step recomputed twice; the N mod S remainder runs as a
           per-step-checkpointed tail
Longer windows: grad.chunked_value_and_grad (chunks of this integrator, boundary States parked on the host) — the
nested schedule host chunks -> sqrt segments -> steps.

Per-step cotangent statistics (the adjoint monitor, MITgcm's `%MON ad_dynstat_*`, pkg/monitor/monitor_ad.F:113-134,
MON_WRITESTATS_RL of adEtaN, aduVel, advVel, adwVel, adTheta, adSalt): `integrate(..., stats_fn=f, sinks=z)` applies an
identity tap to the State at every step boundary n = 0..N; in the reverse pass the tap passes the State cotangent
through unchanged and returns f(model, ct_n) as the cotangent of the n-th sink. The gradient with respect to `sinks`
(zeros made by `make_sinks`) is therefore the per-step statistics, with no host callback and no extra pass; the State
and parameter gradients are bitwise those without the tap (tested). The literal MON_WRITESTATS_RL statistics
(hFac/area/dr-weighted max, min, mean, sd, del2) are a later port in `pkg/monitor`; any f(model, ct) -> pytree of
scalars works here.
"""

import math
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from mitjax.ad.cg2d_rule import SAVE_NAME as CG2D_SAVE_NAME

SCHEDULES = ("none", "step", "sqrt")
# Intermediates the rematerialised reverse pass keeps instead of recomputing (jax.ad_checkpoint.checkpoint_name in the
# solvers): the CG2D solution. Recomputing a step would otherwise re-run the literal CG2D iteration (most of a GPU step
# in the ECCO port); keeping it costs one 2-D field per stored step.
SAVE_NAMES = (CG2D_SAVE_NAME,)


# ---------------------------------------------------------------------------------------------------- stacked inputs


def stack_steps(steps):
    """List of per-step input pytrees -> one pytree with a leading step axis (numpy, on the host)."""
    return jax.tree.map(lambda *a: np.stack([np.asarray(x) for x in a]), *steps)


def take_steps(xs, lo, hi):
    """Steps lo..hi-1 of a stacked input (a fresh numpy copy: safe to device_put and free)."""
    return jax.tree.map(lambda a: np.array(a[lo:hi]), xs)


def n_steps(xs):
    return int(jax.tree.leaves(xs)[0].shape[0])


# ---------------------------------------------------------------------------------------------------------- State


def prepare_state(step, model, st, x0):
    """The State cast to exactly the dtypes the step returns (a scan carry must keep its types; weak types from
    Python scalars or `jnp.full` would otherwise compile a second program, [L-ARCH-7]). x0: one step's input.
    Raises ValueError when the step's output State has another structure or shape than its input."""
    out = jax.eval_shape(step, model, st, x0)
    t_in, t_out = jax.tree.structure(st), jax.tree.structure(out)
    if t_in != t_out:
        raise ValueError(f"the step changes the State structure:\n in  {t_in}\n out {t_out}")
    leaves = []
    for (path, a), sd in zip(jax.tree_util.tree_flatten_with_path(st)[0], jax.tree.leaves(out)):
        if jnp.shape(a) != sd.shape:
            raise ValueError(f"State leaf {jax.tree_util.keystr(path)}: shape {jnp.shape(a)} in, {sd.shape} out")
        leaves.append(jnp.asarray(a, sd.dtype))
    return jax.tree.unflatten(t_in, leaves)


# ------------------------------------------------------------------------------------------- cotangent statistics


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _tap(stats_fn, model, st, sink):
    """Identity on the State; its reverse pass reports stats_fn(model, State cotangent) as the sink's cotangent."""
    return st


def _tap_fwd(stats_fn, model, st, sink):
    return st, model


def _zero_ct(x):
    if jnp.issubdtype(jnp.result_type(x), jnp.floating):
        return jnp.zeros_like(x)
    return np.zeros(np.shape(x), jax.dtypes.float0)


def _tap_bwd(stats_fn, model, ct):
    return jax.tree.map(_zero_ct, model), ct, stats_fn(model, ct)


_tap.defvjp(_tap_fwd, _tap_bwd)


def make_sinks(stats_fn, model, st, n):
    """Zeros shaped like stats_fn(model, st) with a leading axis of n + 1 (step boundaries 0..n)."""
    shape = jax.eval_shape(stats_fn, model, st)
    return jax.tree.map(lambda s: jnp.zeros((n + 1,) + s.shape, s.dtype), shape)


# ------------------------------------------------------------------------------------------------- the integrator


def sqrt_segments(n):
    """Outer segment count S of the two-level scheme: minimises the stored States S + n//S + n mod S (S ~ sqrt(n); a
    divisor of n when one is near: 24 steps -> 4 x 6, not 5 x 4 + 4)."""
    n = max(1, int(n))
    return min(range(1, n + 1), key=lambda S: (S + n // S + n % S, abs(S - math.sqrt(n))))


def integrate(step: Callable, model, st, xs, *, schedule="step", segments=None, cost=None, acc0=None,
              save_names=SAVE_NAMES, stats_fn=None, sinks=None):
    """Run n = n_steps(xs) steps of `step` from `st` in a lax.scan; returns (st_n, acc).

    schedule: "none" | "step" | "sqrt" (module docstring); segments: outer segment count for "sqrt" (default
    sqrt_segments(n)). cost(model, st_new, x) -> scalar is added to acc (acc0, default 0.0) after every step.
    save_names: named intermediates the per-step checkpoint keeps (SAVE_NAMES; () recomputes everything).
    stats_fn, sinks: per-step cotangent statistics (module docstring; sinks from make_sinks(stats_fn, model, st, n)).
    """
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule {schedule!r}; allowed {SCHEDULES}")
    if (stats_fn is None) != (sinks is None):
        raise ValueError("stats_fn and sinks go together (sinks = make_sinks(stats_fn, model, st, n))")
    n = n_steps(xs)
    acc = jnp.zeros((), jnp.float64) if acc0 is None else acc0
    st = prepare_state(step, model, st, jax.tree.map(lambda a: a[0], xs))
    tapped = stats_fn is not None
    if tapped:
        xs = (xs, jax.tree.map(lambda a: a[:n], sinks))

    def body(carry, xin):
        s, a = carry
        if tapped:
            x, sink = xin
            s = _tap(stats_fn, model, s, sink)
        else:
            x = xin
        s1 = step(model, s, x)
        if cost is not None:
            a = a + cost(model, s1, x)
        return (s1, a), None

    def finish(carry):
        if not tapped:
            return carry
        s, a = carry
        return _tap(stats_fn, model, s, jax.tree.map(lambda z: z[n], sinks)), a

    carry = (st, acc)
    if schedule == "none":
        carry, _ = lax.scan(body, carry, xs)
        return finish(carry)
    policy = jax.checkpoint_policies.save_only_these_names(*save_names) if save_names else None
    step_body = jax.checkpoint(body, prevent_cse=False, policy=policy)
    if schedule == "step":
        carry, _ = lax.scan(step_body, carry, xs)
        return finish(carry)
    S = sqrt_segments(n) if segments is None else int(segments)
    M = n // S if S > 0 else 0
    if S <= 1 or M < 1:
        carry, _ = lax.scan(step_body, carry, xs)
        return finish(carry)
    used = S * M

    def seg_body(c, seg_xs):
        c, _ = lax.scan(step_body, c, seg_xs)
        return c, None

    main = jax.tree.map(lambda a: a[:used].reshape((S, M) + a.shape[1:]), xs)
    carry, _ = lax.scan(jax.checkpoint(seg_body, prevent_cse=False), carry, main)
    if used < n:
        carry, _ = lax.scan(step_body, carry, jax.tree.map(lambda a: a[used:], xs))
    return finish(carry)


def run_loop(step_jit, model, st, xs_steps):
    """Reference driver: a Python loop over a jitted step, xs_steps = list of per-step inputs (the scan == loop gate;
    forward runs that are not differentiated)."""
    for x in xs_steps:
        st = step_jit(model, st, x)
    return st
