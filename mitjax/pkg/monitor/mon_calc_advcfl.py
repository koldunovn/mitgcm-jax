"""MON_CALC_ADVCFL_TILE, MON_CALC_ADVCFL_GLOB   @63cdc0b pkg/monitor/mon_calc_advcfl.F:1-176

Tracer advective CFL, computed in THERMODYNAMICS (model/src/thermodynamics.F:278-283 per tile, :387-390 global) when
monOutputCFL, stored in mon_trAdvCFL (MONITOR.h) and printed by the next MONITOR call (monitor.F:149-154).
"""

import jax.numpy as jnp
import numpy as np

from mitjax.farray import FArray, loop_i, loop_j
from mitjax.ops import fortran_minmax_host as host
from mitjax.ops.fortran_minmax import MAX, MIN
from mitjax.ops.fortran_minmax_host import max_chain
from mitjax.pkg.monitor.monitor_h import global_max_rl

# eesupp/inc/EEPARAMS.h:71-72 zeroRL = 0.0 _d 0
zeroRL = 0.0


def _maxz(a):
    """MAX(a, zeroRL) (:87, :92, :110): the first argument wins ties and NaN ($MJX_REFERENCE/minmax_sites); keeps a = -0."""
    return MAX(a, zeroRL, p="a")                                    # :86-89, :91-94, :109-112


def _minz(a):
    """MIN(a, zeroRL) (:88, :93, :111): the first argument wins ties and NaN."""
    return MIN(a, zeroRL, p="a")                                    # :86-89, :91-94, :109-112


def mon_calc_advcfl_tile(myNr, uFld, vFld, wFld, dT_lev, maxCFL, myIter, *, cfg, grid):
    """MON_CALC_ADVCFL_TILE( myNr, bi, bj, uFld, vFld, wFld, dT_lev, maxCFL, myIter, myThid )   :13-120

    C     Calculate Maximum advective CFL in 3 direction (x,y,z)
    C     for current tile bi,bj
    C     uFld    :: zonal velocity        vFld :: merid velocity      wFld :: vert. velocity
    C     dT_lev  :: tracer time-step      maxCFL  :: maximum advective CFL in 3 directions

    uFld, vFld, wFld: FArrays (1-OLx:sNx+OLx,1-OLy:sNy+OLy,myNr) of every tile (the bi,bj tile loop is implicit);
    dT_lev: dT_lev(myNr) (host floats). maxCFL (O): [tile, 3] host array, returned.

    The k loop runs from myNr down to 1 and carries rTrans (the transport of level k+1 is read as rTransKp1 at level
    k, :103-104), so it is a Python loop over k in the Fortran order; the per-tile maxima are gfortran MAX chains over
    k (descending), j, i, here per tile, the new value winning ties and NaN ($MJX_REFERENCE/minmax_sites). Grid inputs: dyG, dxG, hFacW, hFacS, recip_rA, recip_hFacC, recip_drF,
    deepFacC, recip_deepFac2C, deepFac2F, rhoFacF, recip_rhoFacC (GRID.h) as at the THERMODYNAMICS call."""
    sNx, sNy, OLx, OLy = cfg.sNx, cfg.sNy, cfg.OLx, cfg.OLy
    g = grid
    T = uFld.data.shape[0]
    maxCFL = np.zeros((T, 3))                                       # :59-61 (0.)
    tile_max = [[np.float64(0.0)] * 3 for _ in range(T)]
    rTrans = FArray(jnp.zeros((T, sNy + 2*OLy, sNx + 2*OLx)), "rTrans",       # :63-67
                    i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy))
    uTrans = rTrans.local("uTrans")
    vTrans = rTrans.local("vTrans")
    for k in range(myNr, 0, -1):                                    # :69
        j = loop_j(1, sNy+1)                                        # :73-80
        i = loop_i(1, sNx+1)
        uTrans = uTrans.at[i, j].set(uFld[i, j, k]*g.dyG[i, j]*g.deepFacC[k]
                                     * g.hFacW[i, j, k])
        vTrans = vTrans.at[i, j].set(vFld[i, j, k]*g.dxG[i, j]*g.deepFacC[k]
                                     * g.hFacS[i, j, k])
        j = loop_j(1, sNy)                                          # :81-97
        i = loop_i(1, sNx)
        recVol_dT = (dT_lev[k-1]
                     * g.recip_rA[i, j]*g.recip_deepFac2C[k]
                     * g.recip_hFacC[i, j, k])
        cfl_u = (+_maxz(uTrans[i+1, j])
                 - _minz(uTrans[i, j]))*recVol_dT
        cfl_v = (+_maxz(vTrans[i, j+1])
                 - _minz(vTrans[i, j]))*recVol_dT
        rTransKp1 = rTrans[i, j]                                    # :101-115
        rTrans = rTrans.at[i, j].set(wFld[i, j, k]
                                     * g.deepFac2F[k]*g.rhoFacF[k])
        recVol_dT = (dT_lev[k-1]
                     * g.recip_deepFac2C[k]*g.recip_rhoFacC[k]
                     * g.recip_drF[k]*g.recip_hFacC[i, j, k])
        cfl_w = (+_maxz(rTrans[i, j])
                 - _minz(rTransKp1))*recVol_dT
        cu, cv, cw = (np.asarray(c) for c in (cfl_u, cfl_v, cfl_w))
        for t in range(T):                                          # MAX chains in j, i order within level k
            for n, c in enumerate((cu, cv, cw)):
                tile_max[t][n] = _chain_max(tile_max[t][n], c[t])
    for t in range(T):
        maxCFL[t] = tile_max[t]
    return maxCFL


def _chain_max(m, values):
    """maxCFL(n) = MAX( maxCFL(n), tmpVal ) (:90, :95, :113): the new value wins ties and NaN
    ($MJX_REFERENCE/minmax_sites)."""
    return max_chain(m, values, p="b")                              # :90, :95, :113


def mon_calc_advcfl_glob(maxCFL, myIter, *, mon, ex=None):
    """MON_CALC_ADVCFL_GLOB( maxCFL, myIter, myThid )   :127-176

    C     Calculate Maximum advective CFL in 3 direction (x,y,z)
    C     in global domain (from tile-max value)

    maxCFL: [tile, 3] host array in tile order (the DO bj, DO bi loop, :157-163)."""
    uCFL = np.float64(0.0)                                          # :154-156
    vCFL = np.float64(0.0)
    wCFL = np.float64(0.0)
    for t in range(maxCFL.shape[0]):                                # :157-163
        uCFL = host.MAX(uCFL, maxCFL[t, 0], p="b")                  # :159
        vCFL = host.MAX(vCFL, maxCFL[t, 1], p="b")                  # :160
        wCFL = host.MAX(wCFL, maxCFL[t, 2], p="b")                  # :161
    uCFL = global_max_rl(uCFL, ex)                                  # :164-166
    vCFL = global_max_rl(vCFL, ex)
    wCFL = global_max_rl(wCFL, ex)
    mon.mon_trAdvCFL = [uCFL, vCFL, wCFL]                           # :169-173
