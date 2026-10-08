"""COST_TEMP of tutorial_global_oce_optim (experiment-specific: compiled from the experiment's code_ad)
    @63cdc0b verification/tutorial_global_oce_optim/code_ad/cost_temp.F:9-87
"""

import jax.numpy as jnp

from mitjax.eesupp.global_sum import _zero_plus
from mitjax.farray import loops_kji
from mitjax.pkg.cost.cost_h import chain_sum, require_code_dir

CODE_DIR = "tutorial_global_oce_optim/code_ad"


def cost_temp(cost, *, cfg, maskC, wtheta, thetalev):
    """COST_TEMP( myThid )

    C     | o the subroutine computes the sum of the squared errors
    C     |   relatively to the Levitus climatology

    Quirks of the Fortran, kept [E§3]: only the top Nk = 2 levels enter (:44, `Nk = 2`, not Nr); the normalisation
    `tmp` is the number of wet points of those two levels over ALL tiles (one running sum across the tile loop,
    :50-61, then _GLOBAL_SUM_RL, :62), so every tile's term is divided by the global count; the comparison is with
    the time mean cMeanTheta accumulated by COST_TILE over the last `lastinterval`, against the ANNUAL Levitus
    temperature (`lev_t_an.bin`).
    `thetalev`: the FArray READ_FLD_XYZ_RL('lev_t_an.bin', ' ', thetalev, 0) gives (:47; readBinaryPrec=32 in
    data: float32 values, exact in REAL*8; interior). `wtheta`: [tile, Nr] (COST_WEIGHTS).
    Sums: `tmp` one chain over bj, bi, k, j, i (tile order bi + (bj-1)*nSx is the storage order); GLOBAL_SUM_R8 on one
    process and one thread is `0. + tmp` (eesupp/src/global_sum.F:122-127); `locfc` one chain per tile over k, j, i.
    Returns cost with objf_temp_tut set."""
    require_code_dir(cfg, CODE_DIR)
    sz = cfg.size
    if cfg.cpp.ALLOW_OPENAD:
        raise NotImplementedError("COST_TEMP: ALLOW_OPENAD not ported")
    if not cfg.cpp.ALLOW_COST_TEMP:                                      # :35
        return cost
    Nk = 2                                                               # :44
    k, j, i = loops_kji((1, Nk), (1, sz.sNy), (1, sz.sNx))
    m = maskC[i, j, k]                                                   # [T, Nk, sNy, sNx]
    tmp = chain_sum(m.reshape(1, -1, m.shape[-1]))[0]                    # :50-61 one chain over all tiles
    tmp = _zero_plus(tmp)                                                # :62 _GLOBAL_SUM_RL (0. + tmp)
    tmp = jnp.where(tmp > 0.0, 1.0/jnp.where(tmp > 0.0, tmp, 1.0), tmp)  # :63 IF (tmp.GT.0.) tmp = 1. _d 0/tmp
    wk = jnp.stack([wtheta[:, kk - 1] for kk in range(1, Nk + 1)], axis=1)[:, :, None, None]   # wtheta(k,bi,bj)
    term = tmp*m*wk*(cost.cMeanTheta[i, j, k] - thetalev[i, j, k])**2    # :72-74
    cost.objf_temp_tut = chain_sum(term)                                 # :68-79 locfc per tile
    return cost
