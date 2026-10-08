"""INI_THETA of advect_xy: verification/advect_xy/code/ini_theta.F @63cdc0b (the experiment's own version, which the
build compiles instead of model/src/ini_theta.F)."""

import jax.numpy as jnp

from mitjax.eesupp.exch_rs import EXCH_XYZ_RL
from mitjax.farray import loop_i, loop_j, loops_kji
from mitjax.model.src.ini_theta import check_ini_tracer
from mitjax.ops.libm import glibc_exp
from mitjax.pkg.rw.read_rec import READ_FLD_XYZ_RL


def ini_theta(state, *, cfg, grid, params, ex, rw):
    """INI_THETA( myThid )   @63cdc0b verification/advect_xy/code/ini_theta.F:6-152

    C     | SUBROUTINE INI_THETA
    C     | o Set model initial temperature field.

    :51-73 theta = tRef(k) on every point, then on the interior a Gaussian blob
    theta = EXP( -0.5 _d 0*( rD/20. _d 3 )**2 ), rD = SQRT( (xC - 40. _d 3)**2 + (yC - 40. _d 3)**2 + (rC(k) +
    50. _d 3)**2 ) (all REAL*8 literals; `**2` is a product, EXP is glibc's exp: mitjax/ops/libm.glibc_exp);
    :75 _EXCH_XYZ_RL; :78-81 hydrogThetaFile; :84-126 maskIniTemp and the wet-point check (as model/src); :129-144
    with allowFreezing (no checkIniTemp condition here, unlike model/src) theta raised to Tfreezing = -1.9 _d 0.
    PLOT_FIELD_XYZRL (:146-149) is STDOUT only. Pointwise, vectorised over k."""
    sz = cfg.size
    ip = params.init
    k, j, i = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
    tRef = jnp.asarray(ip.tRef)[None, :, None, None]
    theta = state.theta.at[i, j, k].set(jnp.broadcast_to(tRef, state.theta.data.shape))      # :56
    ki, ji, ii = loops_kji((1, sz.Nr), (1, sz.sNy), (1, sz.sNx))
    rC = jnp.asarray(grid.rC.data)[None, :, None, None]
    rD = jnp.sqrt((grid.xC[ii, ji] - 40.e3)**2                                                 # :63-66
                  + (grid.yC[ii, ji] - 40.e3)**2
                  + (rC + 50.e3)**2)
    theta = theta.at[ii, ji, ki].set(glibc_exp(-0.5*(rD/20.e3)**2))                           # :67
    theta = EXCH_XYZ_RL(theta, ex=ex)                                                          # :75
    if str(ip.hydrogThetaFile).strip() != "":                                                  # :78-81
        theta = READ_FLD_XYZ_RL(ip.hydrogThetaFile, " ", theta, 0, rw=rw)
        theta = EXCH_XYZ_RL(theta, ex=ex)
    if ip.maskIniTemp:                                                                         # :88-94
        theta = theta.at[i, j, k].set(jnp.where(grid.maskC[i, j, k] == 0., 0., theta[i, j, k]))
    check_ini_tracer(theta, ip.tRef, cfg, grid, "THETA", ip.checkIniTemp)                     # :95-126
    Tfreezing = -1.9                                                                           # :129
    if ip.allowFreezing:                                                                       # :130-144
        theta = theta.at[i, j, k].set(jnp.where(theta[i, j, k] < Tfreezing, Tfreezing, theta[i, j, k]))
    return state.replace(theta=theta)
