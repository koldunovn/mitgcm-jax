"""Gradient drivers on the tile-sharded model (plan Task 17): grad.py's window objective, unchanged, inside
jit(shard_map(check_vma=True)) over the mesh of a TileSharding.

    sh = TileSharding(maps, nproc)                                  # eesupp/shard.py
    model4 = place_model(sh, model)                                 # tiled leaves padded + sharded, ex -> sh.ex
    st04, theta4 = sh.put_tree(st0), sh.put_tree(theta)
    vg = sharded_value_and_grad_fn(sh, step, theta, model, st0, final_cost=Jn, init_fn=..., schedule="sqrt")
    J, g4 = vg(theta4, model4, st04, xs, 1.0)                       # g4 padded, tile axis sharded
    g = sh.unpad_tree(g4)

One code path: the body is grad._objective -- integrate (checkpoint.py: lax.scan, per-step checkpoint, sqrt
segments) of the same step and the same costs as the single-device drivers -- run on each device's tile block with
the ShardedExchanger in `model` (exchanges = ppermute rounds, global sums = psum of the zero-padded per-tile partials
followed by the fixed tile-order chain), and differentiated by jax.vjp INSIDE the shard_map: the reverse pass is the
transpose of the same per-device program (ppermute -> inverse ppermute, gather -> scatter-add, psum -> broadcast; JAX
derives them, nothing is written by hand). With P=1 the same program runs on one device.

What the caller's functions must do (they run inside shard_map, on the local tile block):
  * final_cost / cost return a value INVARIANT over the mesh axis: a global sum through the exchanger in `model`
    (ex.global_sum_rl / ex.global_sum_tile, which drop the padding tiles); out_specs P() refuses anything else
    (check_vma). The same function at P=1 with the single-device Exchanger gives the single-device cost.
  * step(model, st, x) reads the exchanger from `model` (never closes over one).
The seed (the cotangent of J) is a traced argument: value_and_grad is seed 1.0; the linearity guard of the first
reverse program ([L-AD-32]: vjp(0) = 0, vjp(2 ct) = 2 vjp(ct)) reuses the compiled program with seeds 0 and 2.

Placement and specs: tiled FArrays are padded to Tpad = P*ceil(nTiles/P) tiles (copies of tile 1, sharded_exchange.
TileBlocks) and sharded over the tile axis, everything else replicated (`tile_specs`, `place_model`; the
ShardedExchanger is sharded over the device axis). The padding tiles compute what tile 1 computes, nothing reads
them, the costs drop them, so the gradient on them is exactly 0 (tested: finiteness and zeros incl. padding).

Long windows: `sharded_chunked_value_and_grad`, grad.chunked_value_and_grad's walk (boundaries parked on the host,
here in the padded layout) with the chunk forward, the chunk VJP and the seed as jit(shard_map) programs.

jit around the shard_map whose body holds jax.checkpoint ([L-PAR-13]); model, State, inputs and seed are arguments,
never closures ([L-ARCH-5]). [L-PAR-11]: fake CPU devices served the ECCO forward gates but not its sharded
gradients; here the CPU probe (P = 1, 2, 3, 4 on fake devices) is measured (tests/test_sharded_grad_cpu.py) and the
4-A100 vs 1-A100 comparison is tier 2 (tests/test_sharded_grad_gpu.py).
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from mitjax.drivers.checkpoint import SAVE_NAMES, integrate, prepare_state
from mitjax.drivers.grad import (ChunkedGrad, _add, _objective, _state_ct, cotangent_norm, field_norms, keep_model,
                                 replace_fields)
from mitjax.eesupp.exchange import Exchanger
from mitjax.farray import FArray


def _is_node(x):
    return isinstance(x, (FArray, Exchanger))


def tile_specs(sh, tree):
    """PartitionSpec prefix tree of a model pytree as `place_model` / TileSharding.put_tree place it: tiled FArrays and
    the exchanger (a ShardedExchanger, or the single-device Exchanger place_model replaces by one) sharded over the
    tile axis (sh.TILES), every other leaf replicated (sh.REP)."""
    def one(x):
        if isinstance(x, Exchanger) or (isinstance(x, FArray) and x.tiled):
            return sh.TILES
        return sh.REP
    return jax.tree.map(one, tree, is_leaf=_is_node)


def place_model(sh, tree, ex=None):
    """A host pytree placed on the mesh: every Exchanger node (single-device) replaced by the placed ShardedExchanger
    sh.ex (or `ex`), every other node placed by TileSharding.put_tree (tiled FArrays padded + sharded, the rest
    replicated; values copied bit for bit)."""
    ex = sh.ex if ex is None else ex

    def one(x):
        if isinstance(x, Exchanger):
            return ex
        return sh.put_tree(x)
    return jax.tree.map(one, tree, is_leaf=_is_node)


def window_step(step):
    """integrate's step(model, st, x) from a drivers.model step `step(arrays, carry, iloop, myTime, myIter)`: the
    model is (Arrays, cost data), the State is (carry, myTime, myIter) and x = iloop (as test_r1_gradient.ck_step)."""
    def f(model, st, iloop):
        carry, t, it = st
        carry, t, it, _ = step(model[0], carry, iloop, t, it)
        return (carry, t, it)
    return f


def _objective_with_state(step, *, schedule, segments, cost, final_cost, init_fn, params_fn, save_names):
    """grad._objective's J (the same integrate call, the same sum), returning (J, final State): the final State comes
    out of the gradient program's own forward pass (jax.vjp has_aux), so a forward comparison costs no second
    program. Tested bitwise against grad.value_and_grad (test_sharded_grad_cpu.py)."""
    def J(theta, model, st0, xs):
        m = params_fn(theta, model)
        s0 = init_fn(theta, st0)
        s_n, acc = integrate(step, m, s0, xs, schedule=schedule, segments=segments, cost=cost, save_names=save_names)
        out = acc
        if final_cost is not None:
            out = out + final_cost(m, s_n)
        return out, s_n
    return J


def _vjp_body(step, with_state, **kw):
    if with_state:
        J = _objective_with_state(step, **kw)

        def body(theta, model, st0, xs, seed):
            val, pull, s_n = jax.vjp(lambda th: J(th, model, st0, xs), theta, has_aux=True)
            return val, pull(seed)[0], s_n
        return body
    J = _objective(step, **kw)

    def body(theta, model, st0, xs, seed):
        val, pull = jax.vjp(lambda th: J(th, model, st0, xs), theta)
        return val, pull(seed)[0]
    return body


def sharded_value_and_grad_fn(sh, step, theta, model, st0, *, final_cost=None, cost=None, schedule="step",
                              segments=None, init_fn=replace_fields, params_fn=keep_model, save_names=SAVE_NAMES,
                              xs_specs=None, with_state=False):
    """f(theta4, model4, st04, xs, seed) -> (J, seed * dJ/dtheta) [+ the final State with with_state] for placed
    arguments: jit(shard_map(check_vma)) of jax.vjp of grad._objective (the single-device drivers' J). theta, model,
    st0: host (or placed) trees, used only for their structure (the specs). xs_specs: specs of the per-step inputs
    (default replicated)."""
    body = _vjp_body(step, with_state, schedule=schedule, segments=segments, cost=cost, final_cost=final_cost,
                     init_fn=init_fn, params_fn=params_fn, save_names=tuple(save_names))
    th_s, st_s = tile_specs(sh, theta), tile_specs(sh, st0)
    in_specs = (th_s, tile_specs(sh, model), st_s, sh.REP if xs_specs is None else xs_specs, sh.REP)
    out_specs = (sh.REP, th_s, st_s) if with_state else (sh.REP, th_s)
    return sh.shard_map(body, in_specs=in_specs, out_specs=out_specs)


def sharded_forward_fn(sh, step, model, st0, *, final_cost=None, cost=None, xs_specs=None):
    """f(model4, st04, xs) -> (st_n, J): the forward window (schedule "none") under jit(shard_map(check_vma)); J =
    acc + final_cost(model, st_n) as in the gradient drivers (0 if neither cost is given)."""
    def body(model, st0, xs):
        s0 = prepare_state(step, model, st0, jax.tree.map(lambda a: a[0], xs))
        s_n, acc = integrate(step, model, s0, xs, schedule="none", cost=cost)
        return s_n, (acc if final_cost is None else acc + final_cost(model, s_n))

    st_s = tile_specs(sh, st0)
    in_specs = (tile_specs(sh, model), st_s, sh.REP if xs_specs is None else xs_specs)
    return sh.shard_map(body, in_specs=in_specs, out_specs=(st_s, sh.REP))


def value_and_grad_seed_fn(step, *, final_cost=None, cost=None, schedule="step", segments=None, init_fn=replace_fields,
                           params_fn=keep_model, save_names=SAVE_NAMES, with_state=False):
    """The single-device counterpart of sharded_value_and_grad_fn (same J, same vjp, same seed argument), jitted:
    f(theta, model, st0, xs, seed) -> (J, seed * dJ/dtheta) [+ final State]. Seed 1.0 is grad.value_and_grad."""
    return jax.jit(_vjp_body(step, with_state, schedule=schedule, segments=segments, cost=cost, final_cost=final_cost,
                             init_fn=init_fn, params_fn=params_fn, save_names=tuple(save_names)))


def forward_fn(step, *, final_cost=None, cost=None):
    """Single-device counterpart of sharded_forward_fn, jitted: f(model, st0, xs) -> (st_n, J)."""
    def f(model, st0, xs):
        s0 = prepare_state(step, model, st0, jax.tree.map(lambda a: a[0], xs))
        s_n, acc = integrate(step, model, s0, xs, schedule="none", cost=cost)
        return s_n, (acc if final_cost is None else acc + final_cost(model, s_n))
    return jax.jit(f)


def seed(x=1.0):
    """The cotangent seed of J as the drivers take it (float64 scalar)."""
    return jnp.float64(x)


# ------------------------------------------------------------------------------------- chunked reverse accumulation


def _is_int(x):
    return not jnp.issubdtype(jnp.result_type(x), jnp.floating)


def _ct_in(cot, primal):
    """Inside shard_map: the carry cotangent as the pullback takes it (integer leaves -> float0 zeros). Across the
    shard_map boundary those leaves travel as integer zeros of the primal's dtype (shard_map takes no float0)."""
    return jax.tree.map(lambda c, p: np.zeros(np.shape(p), jax.dtypes.float0) if _is_int(p) else c, cot, primal)


def _ct_out(ct, primal):
    """Inside shard_map: a pullback's carry cotangent with its float0 leaves as integer zeros (see _ct_in)."""
    return jax.tree.map(lambda c, p: jnp.zeros(jnp.shape(p), jnp.result_type(p)) if _is_int(p) else c, ct, primal)


def put_padded_tree(sh, specs, tree):
    """A host tree in the padded layout (as jax.device_get returns a placed tree) placed again: the leaves under a
    sh.TILES spec with sh.put_padded, the rest replicated. specs: tile_specs of the tree (a prefix tree)."""
    rep = NamedSharding(sh.mesh, sh.REP)

    def one(spec, sub):
        if spec == sh.TILES:
            return jax.tree.map(lambda a: sh.put_padded(np.asarray(a)), sub)
        return jax.tree.map(lambda a: jax.device_put(np.asarray(a), rep), sub)
    return jax.tree.map(one, specs, tree, is_leaf=lambda x: isinstance(x, PartitionSpec))


def sharded_chunk_fns(sh, step, theta, model, st0, *, schedule="step", segments=None, cost=None, final_cost=None,
                      params_fn=keep_model, save_names=SAVE_NAMES):
    """The three programs of the sharded chunked driver, each jit(shard_map(check_vma)) on sh's mesh:
      fwd(theta, model, carry, xs) -> carry               one chunk forward (carry = (State, acc), as grad.py's chunk)
      bwd(theta, model, carry, xs, cot) -> (g, cot)       jax.vjp of the chunk at carry, pulled back from cot
      seed(theta, model, carry, one) -> (J, g, cot)       J = acc + final_cost at the window end and its pullback
    Integer carry leaves (the iteration counter) cross the shard_map boundary as integer zeros in the cotangents."""
    th_s, mo_s, st_s = tile_specs(sh, theta), tile_specs(sh, model), tile_specs(sh, st0)
    ca_s = (st_s, sh.REP)

    def chunk(theta, model, carry, xs):
        return integrate(step, params_fn(theta, model), carry[0], xs, schedule=schedule, segments=segments, cost=cost,
                         acc0=carry[1], save_names=tuple(save_names))

    def bwd(theta, model, carry, xs, cot):
        _, pull = jax.vjp(lambda th, c: chunk(th, model, c, xs), theta, carry)
        g, ct = pull(_ct_in(cot, carry))
        return g, _ct_out(ct, carry)

    def seed(theta, model, carry, one):
        def J(th, c):
            out = c[1]
            if final_cost is not None:
                out = out + final_cost(params_fn(th, model), c[0])
            return out
        loss, pull = jax.vjp(J, theta, carry)
        g, ct = pull(one)
        return loss, g, _ct_out(ct, carry)

    fwd = sh.shard_map(chunk, in_specs=(th_s, mo_s, ca_s, sh.REP), out_specs=ca_s)
    bwd = sh.shard_map(bwd, in_specs=(th_s, mo_s, ca_s, sh.REP, ca_s), out_specs=(th_s, ca_s))
    seed = sh.shard_map(seed, in_specs=(th_s, mo_s, ca_s, sh.REP), out_specs=(sh.REP, th_s, ca_s))
    return fwd, bwd, seed, ca_s


def sharded_chunked_value_and_grad(sh, step, theta, model, st0, *, n_chunks, chunk_steps, xs_fn, final_cost=None,
                                   cost=None, schedule="step", segments=None, init_fn=replace_fields,
                                   params_fn=keep_model, boundary_stride=1, save_names=SAVE_NAMES, on_chunk=None,
                                   on_cot=None, log=None, fns=None):
    """grad.chunked_value_and_grad on the sharded model: the same walk (forward with every `boundary_stride`-th
    chunk-boundary carry parked on the HOST in the padded layout, then the reverse walk over the chunks carrying the
    State cotangent), with the chunk forward, the chunk VJP and the seed as jit(shard_map) programs
    (sharded_chunk_fns; pass `fns` to reuse them). theta, model, st0: placed trees (sh.put_tree / place_model).
    init_fn must be local (no exchanges): it and its pullback run on the placed global arrays outside the shard_map.
    Returns grad.ChunkedGrad (grad padded, tile axis sharded)."""
    stride = max(1, int(boundary_stride))
    say = log or (lambda *a: None)
    fwd, bwd, seed, ca_s = fns if fns is not None else sharded_chunk_fns(
        sh, step, theta, model, st0, schedule=schedule, segments=segments, cost=cost, final_cost=final_cost,
        params_fn=params_fn, save_names=save_names)
    t0 = time.time()
    s0 = jax.jit(init_fn)(theta, st0)
    carry = (s0, jax.device_put(jnp.zeros((), jnp.float64), NamedSharding(sh.mesh, sh.REP)))
    kept = {0: jax.device_get(carry)}
    for c in range(n_chunks):
        carry = jax.block_until_ready(fwd(theta, model, carry, xs_fn(c)))
        if (c + 1) % stride == 0 or c == n_chunks - 1:
            kept[c + 1] = jax.device_get(carry)
    host_gb = sum(np.asarray(a).nbytes for a in jax.tree.leaves(kept)) / 1e9
    loss, g_th, cot = seed(theta, model, carry, jnp.float64(1.0))
    grad = g_th if final_cost is not None else None
    fwd_t = time.time() - t0
    trace = [cotangent_norm(cot[0])]
    ftrace = [field_norms(cot[0])]
    if on_cot is not None:
        on_cot(cot[0])
    say(f"sharded chunked forward: {n_chunks} chunks x {chunk_steps} steps, J = {float(loss):.16e}, host "
        f"{host_gb:.2f} GB, {fwd_t:.1f} s")
    del carry
    t1 = time.time()
    for lo in reversed(range(0, n_chunks, stride)):
        hi = min(lo + stride, n_chunks)
        span = {lo: kept[lo]}
        c_in = put_padded_tree(sh, ca_s, kept[lo])
        for c in range(lo, hi - 1):   # rebuild the span between kept boundaries (stride > 1)
            c_in = jax.block_until_ready(fwd(theta, model, c_in, xs_fn(c)))
            span[c + 1] = jax.device_get(c_in)
        del c_in
        for c in reversed(range(lo, hi)):
            g_c, cot = bwd(theta, model, put_padded_tree(sh, ca_s, span[c]), xs_fn(c), cot)
            grad = _add(grad, g_c)
            trace.append(cotangent_norm(cot[0]))
            ftrace.append(field_norms(cot[0]))
            if on_cot is not None:
                on_cot(cot[0])
            if on_chunk is not None:
                on_chunk(c, trace[-1])
        del span
    # initial state: theta -> st0 (local: no exchange; on the placed global arrays)
    _, pull0 = jax.vjp(lambda th: init_fn(th, st0), theta)
    grad = _add(grad, pull0(_state_ct(cot[0]))[0])
    rev_t = time.time() - t1
    say(f"sharded chunked reverse: {rev_t:.1f} s")
    return ChunkedGrad(float(loss), grad, trace, host_gb, fwd_t, rev_t, n_chunks, chunk_steps, stride, schedule,
                       ftrace)
