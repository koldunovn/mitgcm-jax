"""INI_VEL: model/src/ini_vel.F @63cdc0b."""

from mitjax.eesupp.exch_rs import EXCH_UV_XYZ_RL
from mitjax.farray import loops_kji
from mitjax.pkg.rw.read_rec import READ_FLD_XYZ_RL


def _blank(s):
    return str(s).strip() == ""


def ini_vel(state, *, cfg, grid, params, ex, rw):
    """INI_VEL( myThid )   @63cdc0b model/src/ini_vel.F:6-80

    C     | SUBROUTINE INI_VEL
    C     | o Initialize flow field (either to zero or from input files)

    :42-53 uVel = vVel = 0. _d 0 on every point; :55-64 uVelInitFile / vVelInitFile read (READ_FLD_XYZ_RL) and
    EXCH_UV_XYZ_RL( uVel, vVel, .TRUE. ); :66-77 uVel*_maskW, vVel*_maskS on every point (MASKW_MACROS.h /
    MASKS_MACROS.h: _maskW(i,j,k,bi,bj) = maskW(i,j,k,bi,bj) without the 2-D/1-D mask options). Pointwise, vectorised
    over k."""
    sz = cfg.size
    ip = params.init
    k, j, i = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
    uVel = state.uVel.at[i, j, k].set(0.)                                      # :47
    vVel = state.vVel.at[i, j, k].set(0.)                                      # :48
    if not _blank(ip.uVelInitFile) or not _blank(ip.vVelInitFile):             # :55
        if not _blank(ip.uVelInitFile):
            uVel = READ_FLD_XYZ_RL(ip.uVelInitFile, " ", uVel, 0, rw=rw)       # :57-58
        if not _blank(ip.vVelInitFile):
            vVel = READ_FLD_XYZ_RL(ip.vVelInitFile, " ", vVel, 0, rw=rw)       # :60-61
        uVel, vVel = EXCH_UV_XYZ_RL(uVel, vVel, True, ex=ex)                   # :63
    uVel = uVel.at[i, j, k].set(uVel[i, j, k]*grid.maskW[i, j, k])             # :71
    vVel = vVel.at[i, j, k].set(vVel[i, j, k]*grid.maskS[i, j, k])             # :72
    return state.replace(uVel=uVel, vVel=vVel)
