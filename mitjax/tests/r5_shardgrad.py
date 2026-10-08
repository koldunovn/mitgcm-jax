"""Helpers of the R5 sharded-adjoint gates (plan Task 17 for tutorial_global_oce_optim/input_ad): the optim adjoint
of drivers/adjoint_run.py (fc(xx) = COST_FINAL(THE_MAIN_LOOP(CTRL_MAP_INI_GENTIM2D(xx))), 10 steps, per-step
checkpoint) on a TileSharding through drivers/sharded_grad.py, against the single-device driver.

The objective is the one of the first TAF match: the same step, params_fn and final_cost (adjoint_run.problem),
which read the grid, the exchanger and their inputs from the model argument -- on a TileSharding the placed model
holds each device's tile block and the ShardedExchanger. COST_FINAL gathers its tiled inputs to every tile in tile
order (ex.all_tiles) and runs the single-device routine, so fc is the same sum in the same order at every P.
"""

import gc
import time

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.drivers import sharded_grad as SG

SCHEDULE = "step"


def release():
    """Free the compiled programs of this process (each XLA:CPU kernel holds ~3 memory mappings, vm.max_map_count
    = 65530; one optim gradient program holds ~20k: PORTING_LESSONS, KERNEL_GUIDE §4)."""
    jax.clear_caches()
    gc.collect()


class Setup:
    """The optim window: Model (run driver, r5_gate.driver_model), step, model (Arrays), st0, theta (FArray), xs."""

    def __init__(self, tag="shard"):
        from mitjax.drivers.adjoint_run import sharded_problem
        from mitjax.tests import r5_gate as r5
        self.m = r5.driver_model(tag)
        (self.step, self.model, self.st0, self.xs, self.theta, self.params_fn,
         self.final_cost) = sharded_problem(self.m)

    def kw(self):
        return dict(final_cost=self.final_cost, init_fn=lambda t, s: s, params_fn=self.params_fn, schedule=SCHEDULE)

    def p1(self, seed=1.0):
        """The single-device driver: jit(vjp of grad._objective) (sharded_grad.value_and_grad_seed_fn) ->
        (J, seed * dJ/dtheta) as numpy."""
        f = SG.value_and_grad_seed_fn(self.step, **self.kw())
        J, g = jax.block_until_ready(f(self.theta, self.model, self.st0, self.xs, SG.seed(seed)))
        return float(J), np.asarray(g.data)


class Sharded:
    """The same window on a P-device TileSharding (fake CPU devices or GPUs)."""

    def __init__(self, su, nproc, devices=None):
        from mitjax.eesupp import exch_maps as EM
        from mitjax.eesupp.shard import TileSharding
        self.su = su
        cfg = su.m.cfg
        self.sh = sh = TileSharding(EM.load_maps(cfg.experiment, code=cfg.code_dir), nproc, devices=devices)
        self.model = SG.place_model(sh, su.model)
        self.st0 = sh.put_tree(su.st0)
        self.theta = sh.put_tree(su.theta)
        self.vg = SG.sharded_value_and_grad_fn(sh, su.step, su.theta, su.model, su.st0, **su.kw())

    def value_and_grad(self, seed=1.0):
        """(J, seed * dJ/dtheta padded [Tpad, j, i]) as numpy."""
        J, g = jax.block_until_ready(self.vg(self.theta, self.model, self.st0, self.su.xs, SG.seed(seed)))
        return float(J), np.asarray(g.data)

    def unpad(self, g):
        return np.asarray(g)[:self.sh.layout.nTiles]


def bits_differ(a, b):
    a, b = np.ascontiguousarray(a, np.float64), np.ascontiguousarray(b, np.float64)
    return int(np.count_nonzero(a.view(np.int64) != b.view(np.int64)))


def rel(a, b):
    """max |a - b| / max |b|."""
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def timed(f, *a, **k):
    t0 = time.time()
    out = f(*a, **k)
    return out, time.time() - t0
