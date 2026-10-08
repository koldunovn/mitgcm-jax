"""Helpers of the sharded-gradient gates (plan Task 17) on R2 tutorial_baroclinic_gyre (4 tiles of 31 x 31; set-up
r2_gate.model, the same as the R2 P=4 forward gate): the 10-step window as `integrate` sees it, a cost of the final
state, the P=1 reference and the P-device runs through mitjax/drivers/sharded_grad.py, and the comparisons.

TEST FIXTURE (not model code): the cost J = sum over the interior wet points of w_T * theta + w_eta * etaN of the
final State, w uniform in [0.5, 1.5) (fixed seed) times maskC (surface level for etaN), summed through the exchanger in
Fortran order (`ex.global_sum_rl`: per-tile DO k / DO j / DO i chains, then tile 1 + tile 2 + ... ; padding tiles
dropped), so P=1 and every P sum the same numbers in the same order. Every interior point next to a tile edge has a
nonzero weight, so the gradient reaches the neighbour tiles through the exchange transposes on every partition edge.
Controls: theta = (State, FFIELDS.h, phi0surf) of the initial carry (test_r1_gradient's choice).
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from mitjax.drivers import sharded_grad as SG
from mitjax.drivers.model import Arrays, make_step
from mitjax.eesupp.sharded_exchange import ShardedExchanger

N_STEPS = 10
SCHEDULE = "sqrt"


def model():
    from mitjax.tests import r2_gate as r2
    return r2.model()


def weights(m, seed=17, tiles=None):
    """{"theta": FArray like theta, "etaN": FArray like etaN}: uniform [0.5, 1.5) on the interior wet points, 0 on
    halos and land (and on every tile not in `tiles`, 0-based storage order, when given)."""
    from mitjax.farray import FArray
    rng = np.random.default_rng(seed)
    mc = np.asarray(m.grid.maskC.data)
    wt = rng.uniform(0.5, 1.5, mc.shape) * mc
    we = rng.uniform(0.5, 1.5, mc[:, 0].shape) * mc[:, 0]
    if tiles is not None:
        off = [t for t in range(mc.shape[0]) if t not in tiles]
        wt[off], we[off] = 0.0, 0.0
    th, eta = m.state0.theta, m.state0.etaN
    return {"theta": FArray(jnp.asarray(wt), "wTheta", tiled=True, _dims=th.dims),
            "etaN": FArray(jnp.asarray(we), "wEtaN", tiled=True, _dims=eta.dims)}


def make_final_cost(cfg):
    """J(model, st_n) (module docstring); model = (Arrays, weights), st = ((State, ff, phi0surf, flow), t, it)."""
    sz = cfg.size
    J = slice(sz.OLy, sz.OLy + sz.sNy)
    I = slice(sz.OLx, sz.OLx + sz.sNx)

    def final_cost(model, st):
        arrays, w = model
        ex = arrays.ex
        s = st[0][0]
        th = w["theta"].data[:, :, J, I] * s.theta.data[:, :, J, I]
        jt = ex.global_sum_rl(th.reshape(th.shape[0], -1, sz.sNx))          # DO k; DO j; DO i per tile
        je = ex.global_sum_rl(w["etaN"].data[:, J, I] * s.etaN.data[:, J, I])
        return jt + je
    return final_cost


def init_fn(theta, st):
    """theta = (State, ff, phi0surf) of the initial carry (the flow slot is overwritten by every step)."""
    carry, t, it = st
    return ((theta[0], theta[1], theta[2], carry[3]), t, it)


class Setup:
    """The R2 window on one device: step, model = (Arrays, weights), st0, theta, xs, final_cost."""

    def __init__(self, m, n=N_STEPS):
        self.m = m
        self.arrays = Arrays(grid=m.grid, params=m.params, eos=m.eos, cg2dh=m.cg2dh, cg2d_params=m.cg2d_params,
                             ex=m.ex)
        self.model = (self.arrays, weights(m))
        from mitjax.farray import FArray
        s = m.state0
        # THERMODYNAMICS' flow slot (any value: every step writes it before anything reads it, drivers/model.py),
        # named as THERMODYNAMICS' tracer arm names its local copies (uFld, vFld, wFld). drivers.model.make_step keeps
        # whatever names the carry has (ADVECT lane), so Model.initial_carry's uVel, vVel, wVel scan as well
        flow = tuple(FArray(getattr(s, v).data, n, tiled=True, _dims=getattr(s, v).dims)
                     for v, n in (("uVel", "uFld"), ("vVel", "vFld"), ("wVel", "wFld")))
        carry = (s, m.ff, m.phi0surf, flow)
        self.st0 = (carry, jnp.float64(m.prm.time.startTime), jnp.int32(m.prm.time.nIter0))
        self.theta = (s, m.ff, m.phi0surf)
        self.xs = jnp.arange(1, n + 1, dtype=jnp.int32)
        self.step = SG.window_step(make_step(m.cfg, m.fp))
        self.final_cost = make_final_cost(m.cfg)

    def p1_value_and_grad(self, schedule=SCHEDULE):
        """The single-device reference: mitjax.drivers.grad.value_and_grad (the R1-gated driver)."""
        from mitjax.drivers.grad import value_and_grad
        return value_and_grad(self.step, self.theta, self.model, self.st0, self.xs, final_cost=self.final_cost,
                              init_fn=init_fn, schedule=schedule)

    def p1_seed_fn(self, schedule=SCHEDULE, with_state=True):
        """jit f(theta, model, st0, xs, seed) -> (J, seed * grad, final State) on one device."""
        return SG.value_and_grad_seed_fn(self.step, final_cost=self.final_cost, init_fn=init_fn, schedule=schedule,
                                         with_state=with_state)

    def p1_forward_fn(self):
        return SG.forward_fn(self.step, final_cost=self.final_cost)


class Sharded:
    """The same window on a P-device TileSharding: placed arguments and the compiled drivers."""

    def __init__(self, su, nproc, devices=None, ex_fn=None):
        """ex_fn(sh.ex) -> the exchanger to use (negative controls); default sh.ex."""
        from mitjax.eesupp import exch_maps as EM
        from mitjax.eesupp.shard import TileSharding
        self.su = su
        self.sh = sh = TileSharding(EM.load_maps(su.m.exp), nproc, devices=devices)
        self.model = SG.place_model(sh, su.model, ex=None if ex_fn is None else ex_fn(sh.ex))
        self.st0 = sh.put_tree(su.st0)
        self.theta = sh.put_tree(su.theta)
        self.vg = SG.sharded_value_and_grad_fn(sh, su.step, su.theta, su.model, su.st0, final_cost=su.final_cost,
                                               init_fn=init_fn, schedule=SCHEDULE, with_state=True)
        self.fwd = SG.sharded_forward_fn(sh, su.step, su.model, su.st0, final_cost=su.final_cost)

    def value_and_grad(self, seed=1.0, w=None):
        """(J, seed * dJ/dtheta, final State), all padded / placed; w: other cost weights (host FArrays, placed here;
        the same compiled program: the weights are an argument)."""
        model = self.model if w is None else (self.model[0], self.sh.put_tree(w))
        return jax.block_until_ready(self.vg(self.theta, model, self.st0, self.su.xs, SG.seed(seed)))

    def forward(self):
        return jax.block_until_ready(self.fwd(self.model, self.st0, self.su.xs))


# ------------------------------------------------------------------------------------------------------ comparisons


def leaves(tree):
    """[(path, numpy float64 array)] of the floating leaves."""
    out = []
    for p, x in jax.tree_util.tree_flatten_with_path(tree)[0]:
        a = np.asarray(x)
        if np.issubdtype(a.dtype, np.floating):
            out.append((jax.tree_util.keystr(p), np.asarray(a, np.float64)))
    return out


def compare(a, b):
    """Per-leaf comparison of two trees of the same structure: {path: (max |a-b|, max |a|, points whose bits
    differ)}; NaN on either side counts as differing."""
    out = {}
    for (p, x), (_, y) in zip(leaves(a), leaves(b)):
        if x.shape != y.shape:
            out[p] = ("shape", x.shape, y.shape)
            continue
        nb = int(np.count_nonzero(np.ascontiguousarray(x).view(np.int64) != np.ascontiguousarray(y).view(np.int64)))
        d = np.abs(x - y)
        out[p] = (float(np.max(d)) if d.size else 0.0, float(np.max(np.abs(x))) if x.size else 0.0, nb)
    return out


def summary(cmp):
    """(max over leaves of max|a-b| / max over ALL leaves of max|a| (one scale), max per-leaf relative max|a-b|/max|a|
    over leaves with a nonzero, total differing points)."""
    vals = [v for v in cmp.values() if v[0] != "shape"]
    if len(vals) != len(cmp):
        return float("inf"), float("inf"), -1
    scale = max((v[1] for v in vals), default=0.0)
    dmax = max((v[0] for v in vals), default=0.0)
    per = max((v[0] / v[1] for v in vals if v[1] > 0), default=0.0)
    return (dmax / scale if scale > 0 else dmax), per, sum(v[2] for v in vals)


def nonfinite(tree):
    """{path: number of non-finite points} of the floating leaves (empty when all finite)."""
    return {p: int(np.count_nonzero(~np.isfinite(x))) for p, x in leaves(tree) if not np.all(np.isfinite(x))}


def padding_nonzero(sh, tree):
    """{path: number of nonzero points on the padding tiles} of a padded tree's tiled leaves."""
    from mitjax.farray import FArray
    out = {}
    nT = sh.layout.nTiles
    for p, x in jax.tree_util.tree_flatten_with_path(tree, is_leaf=lambda v: isinstance(v, FArray))[0]:
        if isinstance(x, FArray) and x.tiled:
            a = np.asarray(x.data)[nT:]
            n = int(np.count_nonzero(a))
            if n:
                out[jax.tree_util.keystr(p)] = n
    return out


def timed(f, *a, **k):
    t0 = time.time()
    out = f(*a, **k)
    jax.block_until_ready(out)
    return out, time.time() - t0


# ------------------------------------------------------------------------------------------------ negative controls


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def _ppermute_wrong_transpose(x, axis_name, perm):
    return lax.ppermute(x, axis_name, perm=list(perm))


def _wt_fwd(x, axis_name, perm):
    return lax.ppermute(x, axis_name, perm=list(perm)), None


def _wt_bwd(axis_name, perm, _, ct):
    # the planted error: the cotangent travels in the FORWARD direction (perm), not the inverse permutation
    return (lax.ppermute(ct, axis_name, perm=list(perm)),)


_ppermute_wrong_transpose.defvjp(_wt_fwd, _wt_bwd)


@jax.tree_util.register_pytree_node_class
class WrongTransposeExchanger(ShardedExchanger):
    """TEST FIXTURE (negative control): ShardedExchanger whose ppermute rounds transpose in the wrong direction. The
    forward is the real exchanger's bit for bit (the same ppermutes); only the reverse pass is wrong."""

    def _sources(self, plan, tab, own):
        buf = own[0] if len(own) == 1 else jnp.concatenate(own, axis=-1)
        parts = [buf]
        if plan.perms:
            send = jnp.where(tab["send_ok"], buf[..., tab["send_idx"]], jnp.zeros((), buf.dtype))
            for perm, sl, o in zip(plan.perms, plan.slots, plan.offs):
                parts.append(_ppermute_wrong_transpose(send[..., o:o + sl], self.axis_name, tuple(perm)))
        return (jnp.concatenate(parts, axis=-1) if len(parts) > 1 else buf), buf


@jax.tree_util.register_pytree_node_class
class NoVaryExchanger(ShardedExchanger):
    """TEST FIXTURE (negative control): ShardedExchanger without the invariant -> varying cast (`vary` is the identity,
    as on one device): the values a solve closes over stay invariant ([L-PAR-10])."""

    def vary(self, x):
        return x


def as_exchanger(cls, ex):
    """The placed ShardedExchanger `ex` re-typed as the fixture class `cls` (same tables, same plans)."""
    return cls(ex.layout, ex.blocks, ex.plans, ex.tabs, ex.axis_name, ex.rx2)


def compare_saved(path_a, path_b):
    """Comparison of two saved gradients (npz of `leaves`, e.g. the 4-A100 and the 1-A100 tier-2 jobs): summary() of
    the per-leaf differences."""
    a, b = np.load(path_a), np.load(path_b)
    if sorted(a.files) != sorted(b.files):
        raise ValueError("the two files hold different leaves")
    cmp = {}
    for k in a.files:
        x, y = a[k], b[k]
        nb = int(np.count_nonzero(np.ascontiguousarray(x).view(np.int64) != np.ascontiguousarray(y).view(np.int64)))
        cmp[k] = (float(np.max(np.abs(x - y))) if x.size else 0.0, float(np.max(np.abs(x))) if x.size else 0.0, nb)
    return summary(cmp), cmp
