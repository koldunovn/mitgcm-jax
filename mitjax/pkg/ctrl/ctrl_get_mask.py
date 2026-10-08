"""CTRL_GET_MASK2D   @63cdc0b pkg/ctrl/ctrl_get_mask.F:70-144"""

from mitjax.farray import loops_kji
from mitjax.pkg.ctrl.ctrl_readparms import fstr_prefix
from mitjax.pkg.ctrl.ctrl_toolbox import ctrl_cprsrs


def ctrl_get_mask2d(xx_filename, mask2D, *, cfg, maskC):
    """CTRL_GET_MASK2D( xx_filename, mask2D, myThid )

    C     | o A simple routine to return the correct 2D mask for each ctrl
    C     |   variable.
    C     | o 2D mask is in XY plane

    Branches of tutorial_global_oce_optim/code_ad: ALLOW_SHELFICE undefined (:113-120 not compiled); ALLOW_EXF
    undefined and ALLOW_ROTATE_UV_CONTROLS undefined, so :124-136 compare the file name with xx_tauu / xx_tauv
    (maskW / maskS, not ported: raise); else maskC (:140). maskC: the GRID.h FArray (k=1..Nr)."""
    if cfg.cpp.ALLOW_ROTATE_UV_CONTROLS:
        raise NotImplementedError("CTRL_GET_MASK2D: ALLOW_ROTATE_UV_CONTROLS not ported")
    iAmDone = False                                                    # :110
    # GO lane (global_ocean.cs32x15/code_ad compiles pkg/exf, useEXF off): the SHELFICE names (:113-120) and, under
    # ALLOW_EXF, IF (stressIsOnCgrid) around :127-133 only matter for xx_shi* / xx_tauu / xx_tauv (raise below)
    if cfg.cpp.ALLOW_SHELFICE and any(fstr_prefix(xx_filename, 11, n) for n in ("xx_shicoeff", "xx_shicdrag",
                                                                                 "xx_shifwflx")):
        raise NotImplementedError("CTRL_GET_MASK2D: the SHELFICE controls (maskSHI) not ported")
    if fstr_prefix(xx_filename, 7, "xx_tauu") or fstr_prefix(xx_filename, 7, "xx_tauv"):   # :127-133
        raise NotImplementedError("CTRL_GET_MASK2D: xx_tauu/xx_tauv (maskW/maskS) not ported")
    if not iAmDone:                                                    # :140
        mask2D = ctrl_cprsrs(maskC, cfg.size.Nr, mask2D, 1, sz=cfg.size)
    return mask2D


def ctrl_get_mask3d(xx_filename, mask3D, *, cfg, maskC):
    """CTRL_GET_MASK3D( xx_filename, mask3D, myThid )   @63cdc0b pkg/ctrl/ctrl_get_mask.F:16-64 (GOADK lane)

    C     | o A simple routine to return the correct 3D mask for each ctrl
    C     |   variable.

    Branch of global_ocean.90x40x15/code_ad: ALLOW_UVEL0_CONTROL / ALLOW_VVEL0_CONTROL undefined, so `IF (.TRUE.)`
    copies maskC (CTRL_CPRSRS( maskC, Nr, mask3D, Nr ), every level and point); the xx_uvel / xx_vvel masks raise.
    maskC: the GRID.h FArray (k=1..Nr); mask3D: an FArray of the same declaration (returned)."""
    if (cfg.cpp.flag("ALLOW_UVEL0_CONTROL", "CTRL_OPTIONS.h")
            and cfg.cpp.flag("ALLOW_VVEL0_CONTROL", "CTRL_OPTIONS.h")):
        raise NotImplementedError("CTRL_GET_MASK3D: xx_uvel/xx_vvel (maskW/maskS) not ported")
    sz = cfg.size
    k, j, i = loops_kji((1, sz.Nr), (1 - sz.OLy, sz.sNy + sz.OLy), (1 - sz.OLx, sz.sNx + sz.OLx))   # CTRL_CPRSRS
    return mask3D.at[i, j, k].set(maskC[i, j, k])                      # ctrl_toolbox.F:180-190 (nzOut = Nr)
