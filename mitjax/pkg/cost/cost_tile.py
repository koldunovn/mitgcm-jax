"""COST_TILE   @63cdc0b pkg/cost/cost_tile.F:57-155"""

import copy

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.pkg.cost.cost_accumulate_mean import cost_accumulate_mean


def cost_tile(cost, myTime, myIter, *, cfg, endTime, lastinterval, tracer=None, **state):
    """COST_TILE( myTime, myIter, myThid )

    C     | o this routine computes is called at each time step to
    C     |   accumulate the cost function for the tiles of this processor

    The time test (:119) is on the model clock (endTime from INI_PARMS, lastinterval from data.cost: host floats;
    myTime a host float, or traced in a scan step, then both arms are computed and selected with where);
    it decides whether COST_ACCUMULATE_MEAN runs this step (ALLOW_COST_TEMP defined: :121-123). The other calls
    (shelfice, streamice, thsice) are not compiled in this build. `state`: theta, uVel, vVel,
    maskC, maskW, maskS (FArrays), deltaTClock and the traced lastinterval for COST_ACCUMULATE_MEAN
    (`lastinterval_traced`).
    PTRACERS lane: ALLOW_COST_TRACER (tutorial_tracer_adjsens/code_ad): COST_TRACER on every tile after the time test
    (:146-152, every step); `tracer`: dict(grid=, params=, ptr=, ptf=, ex=) of COST_TRACER
    (mitjax/pkg/cost/cost_tracer)."""
    cost = _cost_tile_mean(cost, myTime, cfg=cfg, endTime=endTime, lastinterval=lastinterval, **state)
    if cfg.cpp.ALLOW_COST_TRACER:                                       # :146-152 (PTRACERS lane)
        from mitjax.pkg.cost.cost_tracer import cost_tracer
        cost = copy.copy(cost)
        cost.objf_tracer = cost_tracer(cost.objf_tracer, cfg=cfg, **tracer)   # DO bj, bi: CALL COST_TRACER
    return cost


def _cost_tile_mean(cost, myTime, *, cfg, endTime, lastinterval, **state):
    """COST_TILE's time-tested part (:119-144)."""
    if not (cfg.cpp.ALLOW_COST_TEST or cfg.cpp.ALLOW_COST_ATLANTIC_HEAT or cfg.cpp.ALLOW_COST_TEMP):  # :121
        return cost
    st = dict(state)
    lt = st.pop("lastinterval_traced")
    cond = myTime > (endTime - lastinterval)                            # :119
    if isinstance(cond, (bool, np.bool_)):                              # host clock
        if cond:
            cost = cost_accumulate_mean(cost, cfg=cfg, lastinterval=lt, **st)      # :122
        return cost
    # traced clock (a scan step): a run-time test of the model clock (KERNEL_GUIDE §4): both arms, selected
    acc = cost_accumulate_mean(copy.copy(cost), cfg=cfg, lastinterval=lt, **st)    # :122
    return jax.tree_util.tree_map(lambda a, b: jnp.where(cond, a, b), acc, cost)
