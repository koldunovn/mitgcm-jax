"""MON_ADVCFLW2   @63cdc0b pkg/monitor/mon_advcflw2.F:8-56"""

import jax.numpy as jnp
import numpy as np

from mitjax.farray import loops_kji
from mitjax.ops.fortran_minmax import MAX
from mitjax.ops.fortran_minmax_host import max_chain
from mitjax.pkg.monitor.mon_out import mon_out_rl
from mitjax.pkg.monitor.monitor_h import global_max_rl, mon_foot_max


def mon_advcflw2(label, W, rHFac, rDrF, dT, *, cfg, mon, ex=None):
    """MON_ADVCFLW2( label, W, rHFac, rDrF, dT, myThid )

    C     Calculates maximum CFL number in the vertical relevant for tracer
    C     Adv. Pb. with Partial Cell.

    W, rHFac: FArrays (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy); rDrF: FArray rDrF(Nr) (not tiled). The inner
    `max( rDrF(K)*rHfac(K), rDrF(K-1)*rHfac(K-1) )` (:42-44) keeps the K term on ties and NaN (the oracle's
    compilation, $MJX_REFERENCE/minmax_sites; checked by running the oracle's own mon_advcflw2.o on -0/+0 and NaN,
    mitjax/tests/fortran_minmax/insitu_mon_advcflw2.F); the outer `theMax=max(theMax,tmpVal)` (:45) is the chain
    over all tiles in which the new value wins ties and NaN."""
    sNx, sNy, Nr = cfg.sNx, cfg.sNy, cfg.Nr
    theMax = np.float64(0.0)                                        # :35
    if Nr < 2:                                                      # DO K=2,Nr is empty
        theMax = global_max_rl(theMax, ex)
        mon_out_rl(label, theMax, mon_foot_max, cfg=cfg, mon=mon)
        return
    K, J, I = loops_kji((2, Nr), (1, sNy), (1, sNx))                # :37-50
    a = rDrF[K]*rHFac[I, J, K]                                      # :43
    b = rDrF[K-1]*rHFac[I, J, K-1]                                  # :44
    tmpVal = jnp.abs(W[I, J, K])*dT*MAX(a, b, p="a")                # :42-44
    theMax = max_chain(theMax, tmpVal, p="b")                      # :45
    theMax = global_max_rl(theMax, ex)                              # :51
    mon_out_rl(label, theMax, mon_foot_max, cfg=cfg, mon=mon)       # :53
