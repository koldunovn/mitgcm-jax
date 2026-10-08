"""GAD_SOM_LIM_R: the Prather (1986) limiter on the vertical moments, applied before the vertical SOM advection.

GAD.h loop ranges come from `gad_h.py` (cited there).
"""

import jax.numpy as jnp

from mitjax.farray import loops_kji
from mitjax.ops.fortran_minmax import MAX, MIN
from mitjax.pkg.generic_advdiff import gad_h

# pkg/generic_advdiff/gad_som_lim_r.F:54-55   _RL three; PARAMETER( three = 3. _d 0 )
three = 3.0


def adv_r_ranges(cfg):
    """(iMinAdvR, iMaxAdvR, jMinAdvR, jMaxAdvR) = (1, sNx, 1, sNy) of GAD.h:108-109 (gad_h.py)."""
    return gad_h.iMinAdvR, gad_h.iMaxAdvR(cfg.sNx), gad_h.jMinAdvR, gad_h.jMaxAdvR(cfg.sNy)


def gad_som_lim_r(limiter, sm_v, sm_o, sm_x, sm_y, sm_z, sm_xx, sm_yy, sm_zz, sm_xy, sm_xz, sm_yz, *, cfg):
    """GAD_SOM_LIM_R(bi,bj, limiter, sm_v, sm_o, sm_x, sm_y, sm_z, sm_xx, sm_yy, sm_zz, sm_xy, sm_xz, sm_yz, myThid)
    @63cdc0b pkg/generic_advdiff/gad_som_lim_r.F:7-82

    C !DESCRIPTION:
    C  Apply limiter before calculating vertical advection
    C        Second-Order Moments Advection of tracer in Z-direction
    C        ref: M.J.Prather, 1986, JGR, 91, D6, pp 6671-6681.
    C !INPUT PARAMETERS:
    C  limiter      :: 0: no limiter ; 1: Prather, 1986 limiter
    C !OUTPUT PARAMETERS:
    C  sm_v         :: volume of grid cell
    C  sm_o         :: tracer content of grid cell (zero order moment)
    C  sm_x,y,z     :: 1rst order moment of tracer distribution, in x,y,z direction
    C  sm_xx,yy,zz  ::  2nd order moment of tracer distribution, in x,y,z direction
    C  sm_xy,xz,yz  ::  2nd order moment of tracer distr., in cross direction xy,xz,yz

    Returns the 11 arrays (sm_v .. sm_yz), 3-D FArrays (1-OLx:sNx+OLx, 1-OLy:sNy+OLy, 1:Nr); only sm_z, sm_zz, sm_xz,
    sm_yz change, on the points (1:sNx, 1:sNy, 1:Nr) (GAD.h:108-109). `limiter` is static.

    The loop nest :61-78 is k-vectorised: each (i,j,k) iteration reads and writes only its own point. `0.` is an exact
    REAL*4 literal; `1.5 _d 0` and `three` are doubles.
    """
    iMinAdvR, iMaxAdvR, jMinAdvR, jMaxAdvR = adv_r_ranges(cfg)
    Nr = cfg.Nr
    if limiter == 1:                                                # :60-79
        k, j, i = loops_kji((1, Nr), (jMinAdvR, jMaxAdvR), (iMinAdvR, iMaxAdvR))
#     If flux-limiting transport is to be applied, place limits on
#     appropriate moments before transport.
        slpmax = jnp.where(sm_o[i, j, k] > 0., sm_o[i, j, k], 0.)
        s1max = slpmax*1.5
        s1new = MIN(s1max, MAX(-s1max, sm_z[i, j, k], p="a"), p="b")    # :69
        s2new = MIN((slpmax+slpmax-jnp.abs(s1new)/three),             # :70-71
                     MAX(jnp.abs(s1new)-slpmax, sm_zz[i, j, k], p="a"), p="b")
        sm_xz = sm_xz.at[i, j, k].set(MIN(slpmax, MAX(-slpmax, sm_xz[i, j, k], p="a"), p="b"))  # :72
        sm_yz = sm_yz.at[i, j, k].set(MIN(slpmax, MAX(-slpmax, sm_yz[i, j, k], p="a"), p="b"))  # :73
        sm_z = sm_z.at[i, j, k].set(s1new)
        sm_zz = sm_zz.at[i, j, k].set(s2new)

    return sm_v, sm_o, sm_x, sm_y, sm_z, sm_xx, sm_yy, sm_zz, sm_xy, sm_xz, sm_yz
