"""THE_MAIN_LOOP / MAIN_DO_LOOP of a plain forward run (plan Task 11 skeleton): the time loop as a `lax.scan` over the
iterations, with the iteration counters literal to the Fortran.

Fortran (@63cdc0b, a build without ALLOW_AUTODIFF / ALLOW_OPENAD, i.e. every forward M1 run):
  * the_model_main.F:623-624   myTime = startTime ; myIter = nIter0           (then CALL THE_MAIN_LOOP)
  * the_main_loop.F:386        CALL INITIALISE_VARIA                          (INITIALISE_VARIA's MONITOR prints the
                                                                              block of (startTime, nIter0))
  * the_main_loop.F:644, :700 DO iloop = 1, nTimeSteps ... CALL MAIN_DO_LOOP( iloop, myTime, myIter )
  * main_do_loop.F:221         CALL FORWARD_STEP( iloop, myTime, myIter )
  * forward_step.F:807-808     myIter = nIter0 + iLoop ; myTime = startTime + deltaTClock*iLoop   (after DYNAMICS:
                               every later routine of the step, the monitor at its end included, sees the new values)
With ALLOW_AUTODIFF (tutorial_global_oce_optim/code_ad) the_main_loop.F:650-653 also reset nIter0 (from startTime) and
myIter, myTime to the start of the step (nIter0 + iloop-1, startTime + deltaTClock*(iloop-1)) before every step, which
the counters below give as well (`step_start_counters`): the start-of-step values equal the previous step's update.

`the_main_loop(step, model, state, ...)` runs `step(model, state, iloop, myTime, myIter) -> (state, myTime, myIter,
out)` -- one FORWARD_STEP, which updates the counters itself where forward_step.F does (`forward_step_counters`) --
for iloop = iloop0 .. iloop0+n-1 in one `lax.scan`; `model` and the State are jit ARGUMENTS, never closures [F§1];
the config is static (closed over by `step`). Step 1 runs eagerly in front of the scan when the caller says it
differs (an Adams-Bashforth start: `first_eager=True`), so that the scan body is the same for every iteration it runs.
Restart: running k iterations, then n-k from the returned (state, myTime, myIter) with iloop0 = k+1, is the same
computation as n iterations at once (the scan carries everything the next step reads); the model-level restart
gate (pickup in between) is mitjax/tests/test_r1_driver.py. The host-side parts of the step (monitor, pickups,
printed lines) and the chunking of the run at the steps they need: mitjax/drivers/run.py.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


def start_counters(tp):
    """the_model_main.F:623-624: (myTime, myIter) = (startTime, nIter0), as a float64 / int32 pair."""
    return jnp.float64(tp.startTime), jnp.int32(tp.nIter0)


def forward_step_counters(iLoop, *, nIter0, startTime, deltaTClock):
    """forward_step.F:807-808  myIter = nIter0 + iLoop ; myTime = startTime + deltaTClock*iLoop.
    `deltaTClock*iLoop`: the INTEGER iLoop converted to REAL*8 (exact), one product, one sum, as the Fortran."""
    myIter = nIter0 + iLoop
    myTime = startTime + deltaTClock * iLoop.astype(jnp.float64)
    return myTime, myIter


def step_start_counters(iloop, *, nIter0, startTime, deltaTClock):
    """the_main_loop.F:652-653 (ALLOW_AUTODIFF builds): myIter = nIter0 + (iloop-1),
    myTime = startTime + deltaTClock*(iloop-1)."""
    return forward_step_counters(iloop - 1, nIter0=nIter0, startTime=startTime, deltaTClock=deltaTClock)


# ADVECT lane: iloop0 is a traced int32 (was static): a static iloop0 recompiled the scan for every chunk, ~60 s
# each for advect_xz (20 monitor chunks of 10 steps: the 200-step run took > 20 min); only the length n is static
@partial(jax.jit, static_argnums=(0, 6))
def _scan(step, model, state, myTime, myIter, iloop0, n):
    def body(carry, iloop):
        st, t, it = carry
        st, t, it, out = step(model, st, iloop, t, it)
        return (st, t, it), out
    iloops = iloop0 + jnp.arange(n, dtype=jnp.int32)
    (state, myTime, myIter), outs = lax.scan(body, (state, myTime, myIter), iloops)
    return state, myTime, myIter, outs


def the_main_loop(step, model, state, myTime, myIter, *, nTimeSteps, iloop0=1, first_eager=False):
    """DO iloop = iloop0, iloop0+nTimeSteps-1: one `step` per iteration (see the module docstring).
    Returns (state, myTime, myIter, outs) with `outs` stacked over the iterations (leading axis)."""
    if nTimeSteps <= 0:
        return state, myTime, myIter, None
    outs0 = None
    if first_eager:
        state, myTime, myIter, outs0 = step(model, state, jnp.int32(iloop0), myTime, myIter)
        iloop0, nTimeSteps = iloop0 + 1, nTimeSteps - 1
        outs0 = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], outs0)
        if nTimeSteps == 0:
            return state, myTime, myIter, outs0
    state, myTime, myIter, outs = _scan(step, model, state, myTime, myIter, jnp.int32(iloop0), int(nTimeSteps))
    if outs0 is not None:
        outs = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b]), outs0, outs)
    return state, myTime, myIter, outs
