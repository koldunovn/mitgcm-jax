"""RBCS_ADD_TENDENCY: pkg/rbcs/rbcs_add_tendency.F @63cdc0b (M3 Task 30)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.pkg.rbcs.rbcs_readparms import maskLEN


def rbcs_add_tendency(gTendency, k, tracerNum, myTime, myIter, *, cfg, params, rbcs=None, theta=None):
    """RBCS_ADD_TENDENCY( gTendency, k, bi, bj, tracerNum, myTime, myIter, myThid )   @63cdc0b
    pkg/rbcs/rbcs_add_tendency.F:7-134

    C     Add to tendency array the contribution from 3-D field relaxation

    `gTendency`: the caller's 2-D tendency of level `k` (FArray i, j); tracerNum: 1 = Temp, 2 = Salt, -1 / -2 = U / V
    (Python ints); `params`: Params with RBCS_PARAMS.h; `rbcs`: RBCS_FIELDS.h (RBC_mask, RBCtemp) and `theta`
    (DYNVARS.h) for tracerNum 1. Ported: rbcsVanishingFac = 1 (:61-63, rbcsVanishingTime <= 0), the temperature
    relaxation (:88-99) on i = 0..sNx+1, j = 0..sNy+1 of level k, the U/V calls with useRBCuVel / useRBCvVel off
    (:66, :76: no term). Not ported (raise): rbcsVanishingTime > 0 (:58-60), the U/V terms (:66-85), the salinity term
    (:101-111), ptracers (:113-128). Returns the new gTendency."""
    if params.rbcsVanishingTime > 0.0:                                  # :58-60
        raise NotImplementedError("RBCS_ADD_TENDENCY: rbcsVanishingTime > 0 is not ported")
    rbcsVanishingFac = jnp.float64(1.0)                                 # :62  1. _d 0
    if not cfg.cpp.DISABLE_RBCS_MOM:
        if (tracerNum == -1 and params.useRBCuVel) or (tracerNum == -2 and params.useRBCvVel):    # :66-85
            raise NotImplementedError("RBCS_ADD_TENDENCY: the U/V relaxation is not ported")
    if tracerNum == 1 and params.useRBCtemp:                            # :88-99
        irbc = min(maskLEN, tracerNum)              # :89; MINMAX-INT: integer (no tie or NaN case)
        rec_tauRlx = rbcsVanishingFac / params.tauRelaxT                # :90
        sz = cfg.size
        j = loop_j(0, sz.sNy + 1)                                       # :91
        i = loop_i(0, sz.sNx + 1)                                       # :92
        mask = rbcs["RBC_mask"][irbc - 1]
        gTendency = gTendency.at[i, j].set(
            gTendency[i, j]
            - mask[i, j, k] * rec_tauRlx
            * (theta[i, j, k] - rbcs["RBCtemp"][i, j, k]))             # :93-95
    if tracerNum == 2 and params.useRBCsalt:                            # :101-111
        raise NotImplementedError("RBCS_ADD_TENDENCY: the salinity relaxation is not ported")
    if cfg.cpp.ALLOW_PTRACERS and tracerNum > 2:                        # :113-128
        raise NotImplementedError("RBCS_ADD_TENDENCY: the ptracer relaxation is not ported")
    return gTendency
