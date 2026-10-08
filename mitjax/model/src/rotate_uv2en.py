"""ROTATE_UV2EN_RL (model/src/rotate_uv2en.F @63cdc0b; the file also holds ROTATE_UV2EN_RS, not used by M1)."""

from mitjax.farray import loops_kji


def rotate_uv2en_rl(uFldX, vFldY, uFldE, vFldN, xy2en, switchGrid, applyMask, kSize, *, cfg, grid, fp):
    """rotate_uv2en_rl( uFldX, vFldY, uFldE, vFldN, xy2en, switchGrid, applyMask, kSize, mythid )   @63cdc0b
    model/src/rotate_uv2en.F:8-187

    c     o uFldX/vFldY are in the model grid directions.
    c     o uFldE/vFldN are eastward/northward.
    c     o This routine goes from uFldX/vFldY to uFldE/vFldN (for xy2en=.TRUE.)
    c         or vice versa (for xy2en=.FALSE.).
    c     o If switchGrid=.TRUE. we go from C grid uFldX/vFldY
    c         to A grid uFldE/vFldN, or vice versa.
    c     o If applyMask=.TRUE. then masks are applied to the output.
    c     o In any case it is assumed that exchanges are done on the outside.

    Arguments are all `U` (updated) in the Fortran; returns (uFldX, vFldY, uFldE, vFldN). Ported: xy2en with
    switchGrid (:83-104, :119-129), with or without applyMask, kSize = Nr (the levels are independent: each k reads
    and writes only level k, so the k loop is vectorised). Raises: xy2en = .FALSE. (:131-178), switchGrid =
    .FALSE. (:105-116), kSize other than Nr (SBO_CALC, the only M1 caller, passes Nr), and the :65-71 STOP."""
    sz = cfg.size
    if kSize != 1 and kSize != sz.Nr and applyMask:                             # :65-71
        raise ValueError("ABNROMAL END: S/R ROTATE_UV2EN (no mask has kSize levels)")
    if not xy2en or not switchGrid or kSize != sz.Nr:
        raise NotImplementedError("ROTATE_UV2EN_RL: only xy2en=.TRUE., switchGrid=.TRUE., kSize=Nr is ported")
    # kk = k (:77-81; kSize = Nr)
    tmpU = uFldE.local("tmpU")
    tmpV = vFldN.local("tmpV")
    k, j, i = loops_kji((1, kSize), (sz.sNy+sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))   # :87-90
    tmpU = tmpU.at[i, j, k].set(0.)
    tmpV = tmpV.at[i, j, k].set(0.)
    k, j, i = loops_kji((1, kSize), (1-sz.OLy, sz.sNy+sz.OLy-1), (sz.sNx+sz.OLx, sz.sNx+sz.OLx))   # :91-93
    tmpU = tmpU.at[i, j, k].set(0.)
    tmpV = tmpV.at[i, j, k].set(0.)
    k, j, i = loops_kji((1, kSize), (1-sz.OLy, sz.sNy+sz.OLy-1), (1-sz.OLx, sz.sNx+sz.OLx-1))     # :91-104
    tmpU = tmpU.at[i, j, k].set(0.5*(uFldX[i+1, j, k] + uFldX[i, j, k]))
    tmpV = tmpV.at[i, j, k].set(0.5*(vFldY[i, j+1, k] + vFldY[i, j, k]))
    if applyMask:                                                               # :99-102
        tmpU = tmpU.at[i, j, k].set(tmpU[i, j, k]*grid.maskC[i, j, k])
        tmpV = tmpV.at[i, j, k].set(tmpV[i, j, k]*grid.maskC[i, j, k])
    k, j, i = loops_kji((1, kSize), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))         # :120-129
    uFldE = uFldE.at[i, j, k].set(grid.angleCosC[i, j]*tmpU[i, j, k] - grid.angleSinC[i, j]*tmpV[i, j, k])
    vFldN = vFldN.at[i, j, k].set(grid.angleSinC[i, j]*tmpU[i, j, k] + grid.angleCosC[i, j]*tmpV[i, j, k])
    return uFldX, vFldY, uFldE, vFldN
