"""COST_HFLUX of tutorial_global_oce_optim (experiment-specific: compiled from the experiment's code_ad)
    @63cdc0b verification/tutorial_global_oce_optim/code_ad/cost_hflux.F:9-93
"""

import jax.numpy as jnp

from mitjax.eesupp.global_sum import _zero_plus
from mitjax.farray import loop_i, loop_j
from mitjax.pkg.cost.cost_h import chain_sum, require_code_dir

CODE_DIR = "tutorial_global_oce_optim/code_ad"


def cost_hflux(cost, *, cfg, maskC, whfluxm, xx_gentim2d):
    """COST_HFLUX( myThid )

    C     | o the subroutine computes the cost function relative to
    C     |   mean surface hflux optimization as a simple example.

    Quirks kept [E§3]: the control is identified by its index, `iarr = 1` hard-wired (:49, "set in data.ctrl"), not
    by its file name; the penalty is on the control field xx_gentim2d itself (the deviation from the first guess 0),
    normalised by the global number of wet surface points (`tmpC`, one chain over all tiles :54-63, _GLOBAL_SUM_RL
    :64 = `0. + tmpC` on one process/thread); whfluxm is 1/Err_hflux**2 (COST_WEIGHTS). `xx_gentim2d`: {iarr: FArray}.
    Returns cost with objf_hflux_tut set (per-tile chains over j, i :70-85)."""
    require_code_dir(cfg, CODE_DIR)
    sz = cfg.size
    if cfg.cpp.ALLOW_OPENAD:
        raise NotImplementedError("COST_HFLUX: ALLOW_OPENAD not ported")
    if not cfg.cpp.ALLOW_COST_HFLUXM:                                    # :37
        return cost
    iarr = 1                                                             # :49
    j = loop_j(1, sz.sNy)
    i = loop_i(1, sz.sNx)
    m = maskC[i, j, 1]                                                   # [T, sNy, sNx]
    tmpC = chain_sum(m.reshape(1, -1, m.shape[-1]))[0]                   # :54-63 one chain over all tiles
    tmpC = _zero_plus(tmpC)                                              # :64 _GLOBAL_SUM_RL
    tmpC = jnp.where(tmpC > 0.0, 1.0/jnp.where(tmpC > 0.0, tmpC, 1.0), tmpC)   # :65
    term = tmpC*m*whfluxm[i, j]*(xx_gentim2d[iarr][i, j])**2             # :73-81
    cost.objf_hflux_tut = chain_sum(term)                                # :70-85 locfc per tile
    return cost
