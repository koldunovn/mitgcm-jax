"""INI_SALT of advect_xy: verification/advect_xy/code/ini_salt.F @63cdc0b (the experiment's own version)."""

import jax.numpy as jnp

from mitjax.eesupp.exch_rs import EXCH_XYZ_RL
from mitjax.farray import loops_kji
from mitjax.model.src.ini_theta import check_ini_tracer
from mitjax.pkg.rw.read_rec import READ_FLD_XYZ_RL


def ini_salt(state, *, cfg, grid, params, ex, rw):
    """INI_SALT( myThid )   @63cdc0b verification/advect_xy/code/ini_salt.F:6-133

    C     | SUBROUTINE INI_SALT
    C     | o Set model initial salinity field.

    :50-72 salt = sRef(k) on every point, then on EVERY point (halos included, :60-61) salt = sRef(k)+1. _d 0 where
    rD = SQRT( (xC - 40. _d 3)**2 + (yC - 40. _d 3)**2 + (rC(k) + 50. _d 3)**2 ) <= 60. _d 3; :74 _EXCH_XYZ_RL;
    :77-80 hydrogSaltFile; :83-125 maskIniSalt and the wet-point check. PLOT_FIELD_XYZRL (:127-130) is STDOUT only."""
    sz = cfg.size
    ip = params.init
    k, j, i = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
    sRef = jnp.asarray(ip.sRef)[None, :, None, None]
    salt = state.salt.at[i, j, k].set(jnp.broadcast_to(sRef, state.salt.data.shape))          # :55
    rC = jnp.asarray(grid.rC.data)[None, :, None, None]
    rD = jnp.sqrt((grid.xC[i, j] - 40.e3)**2                                                   # :62-65
                  + (grid.yC[i, j] - 40.e3)**2
                  + (rC + 50.e3)**2)
    salt = salt.at[i, j, k].set(jnp.where(rD <= 60.e3, sRef + 1., salt[i, j, k]))            # :66
    salt = EXCH_XYZ_RL(salt, ex=ex)                                                            # :74
    if str(ip.hydrogSaltFile).strip() != "":                                                   # :77-80
        salt = READ_FLD_XYZ_RL(ip.hydrogSaltFile, " ", salt, 0, rw=rw)
        salt = EXCH_XYZ_RL(salt, ex=ex)
    if ip.maskIniSalt:                                                                         # :87-93
        salt = salt.at[i, j, k].set(jnp.where(grid.maskC[i, j, k] == 0., 0., salt[i, j, k]))
    check_ini_tracer(salt, ip.sRef, cfg, grid, "SALT", ip.checkIniSalt)                       # :94-125
    return state.replace(salt=salt)
