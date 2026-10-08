"""MON_ADVCFLW   @63cdc0b pkg/monitor/mon_advcflw.F:8-52"""

import jax.numpy as jnp
import numpy as np

from mitjax.farray import loops_kji
from mitjax.ops.fortran_minmax_host import max_chain
from mitjax.pkg.monitor.mon_out import mon_out_rl
from mitjax.pkg.monitor.monitor_h import global_max_rl, mon_foot_max


def mon_advcflw(label, W, rDz, dT, *, cfg, mon, ex=None):
    """MON_ADVCFLW( label, W, rDz, dT, myThid )

    C     Calculates maximum CFL number in the vertical direction.

    W: FArray (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy); rDz: FArray rDz(Nr) (not tiled); dT host float."""
    sNx, sNy, Nr = cfg.sNx, cfg.sNy, cfg.Nr
    theMax = np.float64(0.0)                                        # :33
    K, J, I = loops_kji((1, Nr), (1, sNy), (1, sNx))                # :35-46
    tmpVal = jnp.abs(W[I, J, K])*rDz[K]*dT                          # :40
    theMax = max_chain(theMax, tmpVal, p="b")                      # :41
    theMax = global_max_rl(theMax, ex)                              # :47
    mon_out_rl(label, theMax, mon_foot_max, cfg=cfg, mon=mon)       # :49
