"""INI_THETA: model/src/ini_theta.F @63cdc0b."""

import jax.numpy as jnp
import numpy as np

from mitjax.eesupp.exch_rs import EXCH_XYZ_RL
from mitjax.farray import loop_i, loop_j, loops_kji
from mitjax.pkg.rw.read_rec import READ_FLD_XYZ_RL


def _blank(s):
    return str(s).strip() == ""


def check_ini_tracer(fld, ref, cfg, grid, name, check):
    """The wet-point check shared by INI_THETA (:85-127) and INI_SALT: count the interior wet points (maskC .NE. 0.)
    with the tracer identically 0. on the levels where the reference is not 0.; with the check on, a non-zero count
    stops the model (STOP 'ABNORMAL END: S/R INI_<name>'), otherwise it prints a warning (STDOUT only). Host-side
    (the initialisation runs eagerly)."""
    sz = cfg.size
    k, j, i = loops_kji((1, sz.Nr), (1, sz.sNy), (1, sz.sNx))
    f = np.asarray(fld[i, j, k])
    m = np.asarray(grid.maskC[i, j, k])
    levels = (np.asarray(ref) != 0.)[None, :, None, None]
    localWarnings = int(np.count_nonzero((m != 0.) & (f == 0.) & levels))
    if localWarnings != 0 and check:
        raise ValueError(f" INI_{name}: found{localWarnings:10d} wet grid-pts with {name.lower()}=0 identically.\n"
                         f"ABNORMAL END: S/R INI_{name}")
    return localWarnings


def ini_theta(state, *, cfg, grid, params, ex, rw):
    """INI_THETA( myThid )   @63cdc0b model/src/ini_theta.F:7-155

    C     | SUBROUTINE INI_THETA
    C     | o Set model initial temperature field.
    C     | There are several options for setting the initial
    C     | temperature file
    C     |  1. Inline code
    C     |  2. Vertical profile ( uniform T in X and Y )
    C     |  3. Three-dimensional data from a file. For example from
    C     |     Levitus or from a checkpoint file from a previous
    C     |     integration.

    :54-64 theta = tRef(k) on every point; :66-82 hydrogThetaFile read (READ_FLD_XYZ_RL) and _EXCH_XYZ_RL; the MNC
    read (:67-75, useMNC .AND. mnc_read_theta) raises. :85-108 with maskIniTemp, theta = 0. on dry points (all
    points); the wet-point check (:96-127, check_ini_tracer). :130-147 with checkIniTemp .AND. allowFreezing, theta
    raised to Tfreezing = -1.9 _d 0 on every point. PLOT_FIELD_XYZRL (:149-152) writes STDOUT only."""
    sz = cfg.size
    ip = params.init
    k, j, i = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
    tRef = jnp.asarray(ip.tRef)[None, :, None, None]
    theta = state.theta.at[i, j, k].set(jnp.broadcast_to(tRef, state.theta.data.shape))      # :59
    if not _blank(ip.hydrogThetaFile):                                         # :66
        if cfg.cpp.ALLOW_MNC and ip.useMNC and ip.mnc_read_theta:              # :67-75
            raise NotImplementedError("INI_THETA: the MNC read of hydrogThetaFile is not ported")
        theta = READ_FLD_XYZ_RL(ip.hydrogThetaFile, " ", theta, 0, rw=rw)      # :77
        theta = EXCH_XYZ_RL(theta, ex=ex)                                      # :81
    if ip.maskIniTemp:                                                         # :89-95
        theta = theta.at[i, j, k].set(jnp.where(grid.maskC[i, j, k] == 0., 0., theta[i, j, k]))
    check_ini_tracer(theta, ip.tRef, cfg, grid, "THETA", ip.checkIniTemp)     # :96-127
    if ip.checkIniTemp and ip.allowFreezing:                                   # :130-147
        Tfreezing = -1.9                                                       # :131  -1.9 _d 0
        theta = theta.at[i, j, k].set(jnp.where(theta[i, j, k] < Tfreezing, Tfreezing, theta[i, j, k]))
    return state.replace(theta=theta)
