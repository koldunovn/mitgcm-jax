"""INI_VEL of advect_xy: verification/advect_xy/code/ini_vel.F @63cdc0b (the experiment's own version)."""

from mitjax.eesupp.exch_rs import EXCH_UV_XYZ_RL
from mitjax.farray import loops_kji
from mitjax.pkg.rw.read_rec import READ_FLD_XYZ_RL


def ini_vel(state, *, cfg, grid, params, ex, rw):
    """INI_VEL( myThid )   @63cdc0b verification/advect_xy/code/ini_vel.F:6-80

    As model/src/ini_vel.F with uVel = vVel = 1. _d 0 (instead of 0.) on every point (:47-48), then the input files
    (:55-64) and the masks (:66-77)."""
    sz = cfg.size
    ip = params.init
    k, j, i = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
    uVel = state.uVel.at[i, j, k].set(1.)                                      # :47
    vVel = state.vVel.at[i, j, k].set(1.)                                      # :48
    if str(ip.uVelInitFile).strip() != "" or str(ip.vVelInitFile).strip() != "":
        if str(ip.uVelInitFile).strip() != "":
            uVel = READ_FLD_XYZ_RL(ip.uVelInitFile, " ", uVel, 0, rw=rw)
        if str(ip.vVelInitFile).strip() != "":
            vVel = READ_FLD_XYZ_RL(ip.vVelInitFile, " ", vVel, 0, rw=rw)
        uVel, vVel = EXCH_UV_XYZ_RL(uVel, vVel, True, ex=ex)                   # :63
    uVel = uVel.at[i, j, k].set(uVel[i, j, k]*grid.maskW[i, j, k])             # :71
    vVel = vVel.at[i, j, k].set(vVel[i, j, k]*grid.maskS[i, j, k])             # :72
    return state.replace(uVel=uVel, vVel=vVel)
