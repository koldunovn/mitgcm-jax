"""grad.value_and_grad / grad.objective keep their jitted function per configuration (lane SHARDGRAD, session 2): a
repeat with the same function objects neither retraces nor recompiles, and gives bitwise the same values; another
configuration (schedule, final cost) builds its own; jit=False is unchanged. Toy step of test_checkpoint.py
(seconds; one test: the tier-1 test budget is 100)."""

import jax
import numpy as np

from mitjax.drivers import grad as gr
from mitjax.tests import test_checkpoint as tc


def _counting_step():
    n = {"traces": 0}

    def step(model, st, x):
        n["traces"] += 1                     # Python side effect: runs only while tracing
        return tc._step(model, st, x)
    return step, n


def _bits(t):
    return [np.asarray(a).view(np.uint64) if np.asarray(a).dtype == np.float64 else np.asarray(a)
            for a in jax.tree.leaves(t)]


def test_value_and_grad_and_objective_repeat_reuse_the_program():
    model, st0, xs = tc._setup()
    theta = tc._theta(st0, model)
    step, n = _counting_step()
    kw = dict(final_cost=tc._final_cost, cost=tc._cost, init_fn=tc._init_fn, params_fn=tc._params_fn,
              schedule="sqrt")
    J1, g1 = gr.value_and_grad(step, theta, model, st0, xs, **kw)
    traced = n["traces"]
    assert traced > 0
    J2, g2 = gr.value_and_grad(step, theta, model, st0, xs, **kw)
    assert n["traces"] == traced                                   # no retrace (so no recompile)
    assert all(np.array_equal(a, b) for a, b in zip(_bits((J1, g1)), _bits((J2, g2))))
    J3, g3 = gr.value_and_grad(step, theta, model, st0, xs, **dict(kw, schedule="step"))
    assert n["traces"] > traced                                    # another configuration: its own program
    assert float(J3) == float(J1)
    J4, g4 = gr.value_and_grad(step, theta, model, st0, xs, jit=False, **kw)
    assert float(J4) == float(J1) or abs(float(J4) - float(J1)) <= 1e-12 * abs(float(J1))

    # with the per-step statistics, and grad.objective
    step, n = _counting_step()
    kw = dict(final_cost=tc._final_cost, cost=tc._cost, init_fn=tc._init_fn, params_fn=tc._params_fn)
    a = gr.value_and_grad(step, theta, model, st0, xs, stats_fn=tc._stats, **kw)
    t0 = n["traces"]
    b = gr.value_and_grad(step, theta, model, st0, xs, stats_fn=tc._stats, **kw)
    t1 = n["traces"]
    assert all(np.array_equal(x, y) for x, y in zip(_bits(a), _bits(b)))
    j1 = gr.objective(step, theta, model, st0, xs, **kw)
    t2 = n["traces"]
    j2 = gr.objective(step, theta, model, st0, xs, **kw)
    assert float(j1) == float(j2) and n["traces"] == t2 and t2 > t1
    # the stats repeat re-traces only make_sinks' eval_shape of stats_fn / prepare_state (host-side shapes), never the
    # window: at most the two eval_shape traces of the step
    assert t1 - t0 <= 1, (t0, t1)
