"""INI_SALT: model/src/ini_salt.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.eesupp.exch_rs import EXCH_XYZ_RL
from mitjax.farray import loops_kji
from mitjax.model.src.ini_theta import check_ini_tracer
from mitjax.pkg.rw.read_rec import READ_FLD_XYZ_RL


def _blank(s):
    return str(s).strip() == ""


def ini_salt(state, *, cfg, grid, params, ex, rw):
    """INI_SALT( myThid )   @63cdc0b model/src/ini_salt.F:7-128

    C     | SUBROUTINE INI_SALT
    C     | o Set model initial salinity field.

    As INI_THETA without the freezing branch: salt = sRef(k) on every point (:28); hydrogSaltFile read
    (READ_FLD_XYZ_RL) and _EXCH_XYZ_RL (:35-51; the MNC read with useMNC .AND. mnc_read_salt raises); with
    maskIniSalt, salt = 0. on dry points; the wet-point check with checkIniSalt (check_ini_tracer)."""
    sz = cfg.size
    ip = params.init
    k, j, i = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
    sRef = jnp.asarray(ip.sRef)[None, :, None, None]
    salt = state.salt.at[i, j, k].set(jnp.broadcast_to(sRef, state.salt.data.shape))
    if not _blank(ip.hydrogSaltFile):
        if cfg.cpp.ALLOW_MNC and ip.useMNC and ip.mnc_read_salt:
            raise NotImplementedError("INI_SALT: the MNC read of hydrogSaltFile is not ported")
        salt = READ_FLD_XYZ_RL(ip.hydrogSaltFile, " ", salt, 0, rw=rw)
        salt = EXCH_XYZ_RL(salt, ex=ex)
    if ip.maskIniSalt:
        salt = salt.at[i, j, k].set(jnp.where(grid.maskC[i, j, k] == 0., 0., salt[i, j, k]))
    check_ini_tracer(salt, ip.sRef, cfg, grid, "SALT", ip.checkIniSalt)
    return state.replace(salt=salt)
