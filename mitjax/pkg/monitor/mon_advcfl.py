"""MON_ADVCFL   @63cdc0b pkg/monitor/mon_advcfl.F:8-52"""

import jax.numpy as jnp
import numpy as np

from mitjax.farray import loops_kji
from mitjax.ops.fortran_minmax_host import max_chain
from mitjax.pkg.monitor.mon_out import mon_out_rl
from mitjax.pkg.monitor.monitor_h import global_max_rl, mon_foot_max


def mon_advcfl(label, U, rDx, dT, *, cfg, mon, ex=None):
    """MON_ADVCFL( label, U, rDx, dT, myThid )

    C     Calculates maximum CFL number

    U: FArray (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy); rDx: FArray (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy); dT host
    float. `theMax=max(theMax,tmpVal)` (:41) is one chain over all tiles (the new value wins ties and NaN: $MJX_REFERENCE/minmax_sites), from
    `theMax=0.` (:33); abs(U)*rDx*dT is (abs(U)*rDx)*dT."""
    sNx, sNy, Nr = cfg.sNx, cfg.sNy, cfg.Nr
    theMax = np.float64(0.0)                                        # :33
    K, J, I = loops_kji((1, Nr), (1, sNy), (1, sNx))                # :35-46
    tmpVal = jnp.abs(U[I, J, K])*rDx[I, J]*dT                       # :40
    theMax = max_chain(theMax, tmpVal, p="b")                      # :41
    theMax = global_max_rl(theMax, ex)                              # :47
    mon_out_rl(label, theMax, mon_foot_max, cfg=cfg, mon=mon)       # :49
