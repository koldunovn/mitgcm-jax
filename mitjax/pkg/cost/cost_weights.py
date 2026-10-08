"""COST_WEIGHTS of tutorial_global_oce_optim (experiment-specific: compiled from the experiment's code_ad)
    @63cdc0b verification/tutorial_global_oce_optim/code_ad/cost_weights.F:9-127
"""

import jax.numpy as jnp
import numpy as np

from mitjax.farray import loop_i, loop_j
from mitjax.pkg.cost.cost_h import require_code_dir

CODE_DIR = "tutorial_global_oce_optim/code_ad"


def cost_weights(*, cfg, tmpwti, Err_hflux, ex, xy):
    """COST_WEIGHTS( myThid )

    C     | o Set weights used in the cost function

    File reads are the caller's (host I/O): `tmpwti` = the Nr REAL*8 values of record 1 of Err_levitus_15layer.bin
    (direct access, RECL = MDS_RECLEN(precFloat64, Nr), :70-75; big-endian, the build's -fconvert), `Err_hflux` = the
    FArray of record 1 of Err_hflux.bin read by READ_REC_3D_RL(..., precFloat64, 1, whfluxm, 1, 0) (:104-105;
    interior set, halos 0 as initialised at :52-63). `xy(data, name)`: the 2-D FArray constructor. Returns
    (whfluxm FArray, wtheta [tile, Nr]). The PRINT_MESSAGE of :85-88 goes to STDOUT (not reproduced here).
    ALLOW_NONDIMENSIONAL_CONTROL_IO undefined (:122-124)."""
    require_code_dir(cfg, CODE_DIR)
    sz = cfg.size
    if cfg.cpp.ALLOW_NONDIMENSIONAL_CONTROL_IO:
        raise NotImplementedError("COST_WEIGHTS: ALLOW_NONDIMENSIONAL_CONTROL_IO not ported")
    nT = Err_hflux.data.shape[0]
    wti = np.zeros(sz.Nr)                                                  # :49-51
    wtheta = jnp.zeros((nT, sz.Nr))                                        # :59-61
    if cfg.cpp.ALLOW_COST_TEMP:                                            # :67
        wti = np.asarray(tmpwti, np.float64).copy()                        # :82-84
        w = 1.0/wti/wti                                                    # :95  1. _d 0/wti(k)/wti(k)
        wtheta = jnp.broadcast_to(jnp.asarray(w), (nT, sz.Nr))             # :92-98 (every tile)
    whfluxm = Err_hflux                                                    # :52-63 (zero halos) + :104-105
    if cfg.cpp.ALLOW_COST_HFLUXM:                                          # :103
        whfluxm = xy(ex.EXCH_XY_RL(whfluxm.data), "whfluxm")              # :106 _EXCH_XY_RL
        j = loop_j(1 - sz.OLy, sz.sNy + sz.OLy)                            # :109
        i = loop_i(1 - sz.OLx, sz.sNx + sz.OLx)                            # :110
        w = whfluxm[i, j]
        nz = w != 0.0                                                      # :112
        safe = jnp.where(nz, w, 1.0)                                       # guard: the ELSE lanes stay finite
        whfluxm = whfluxm.at[i, j].set(jnp.where(nz, 1.0/safe/safe, 1.0))  # :113-116
    return whfluxm, wtheta
