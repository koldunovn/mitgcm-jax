"""STAND-IN for model/src/rotate_uv2en.F (ROTATE_UV2EN_RL)   @63cdc0b model/src/rotate_uv2en.F:8-180

A literal port of the one branch CTRL_MAP_FORCING executes (xy2en=.FALSE., switchGrid=.TRUE., applyMask=.TRUE.,
kSize=1, usingPCoords=.FALSE.). It belongs in mitjax/model/src/rotate_uv2en.py, which the ctrl lane may not write
(owned by the core lane): move it there when the core lane ports ROTATE_UV2EN.
"""

from mitjax.config.fortran import real4
from mitjax.farray import loop_i, loop_j


def rotate_uv2en_rl(uFldX, vFldY, uFldE, vFldN, xy2en, switchGrid, applyMask, kSize, *, sz, usingPCoords,
                    angleCosC, angleSinC, maskW, maskS):
    """rotate_uv2en_rl( uFldX, vFldY, uFldE, vFldN, xy2en, switchGrid, applyMask, kSize, mythid )

    c     o uFldX/vFldY are in the model grid directions.
    c     o uFldE/vFldN are eastward/northward.
    c     o This routine goes from uFldX/vFldY to uFldE/vFldN (for xy2en=.TRUE.)
    c         or vice versa (for xy2en=.FALSE.).

    2-D fields (kSize = 1) as FArrays; returns (uFldX, vFldY, uFldE, vFldN). Ported: xy2en=.FALSE. with
    switchGrid=.TRUE. (:131-167). The Fortran zeroes row j=1 and column i=1 (:148-154, REAL*4 `0.`) and then
    overwrites them in the loop :152-166, which starts at j = 1-oly+1 and i = 1-olx+1: row j=1-oly and column
    i=1-olx keep their input values (quirk kept as written)."""
    if xy2en or not switchGrid or kSize != 1:
        raise NotImplementedError("ROTATE_UV2EN_RL stand-in: only xy2en=F, switchGrid=T, kSize=1 are ported")
    if usingPCoords:                                                   # :77-81 (kSize.EQ.1 .AND. usingPCoords)
        raise NotImplementedError("ROTATE_UV2EN_RL stand-in: usingPCoords not ported")
    kk = 1                                                             # :80 kk=k, k=1
    tmpU = uFldE.local("tmpU")                                         # local tmpU, tmpV (NaN until written)
    tmpV = vFldN.local("tmpV")
    # C 1) rotation  :134-145 (every point is written)
    j = loop_j(1 - sz.OLy, sz.sNy + sz.OLy)
    i = loop_i(1 - sz.OLx, sz.sNx + sz.OLx)
    tmpU = tmpU.at[i, j].set(angleCosC[i, j]*uFldE[i, j]               # :137-139
                             + angleSinC[i, j]*vFldN[i, j])
    tmpV = tmpV.at[i, j].set(-angleSinC[i, j]*uFldE[i, j]              # :140-142
                             + angleCosC[i, j]*vFldN[i, j])
    # C 2a) go from A grid velocity points to C grid velocity points  :147-166
    zero4 = real4("0.")
    i = loop_i(1 - sz.OLx, sz.sNx + sz.OLx)                            # :148-151
    uFldX = uFldX.at[i, 1].set(zero4)
    vFldY = vFldY.at[i, 1].set(zero4)
    for jj in range(1 - sz.OLy + 1, sz.sNy + sz.OLy + 1):              # :152 DO j = 1-oly+1,sny+oly
        uFldX = uFldX.at[1, jj].set(zero4)                             # :153
        vFldY = vFldY.at[1, jj].set(zero4)                             # :154
        jl = loop_j(jj, jj)
        i = loop_i(1 - sz.OLx + 1, sz.sNx + sz.OLx)                    # :155
        uFldX = uFldX.at[i, jl].set(0.5*(tmpU[i-1, jl] + tmpU[i, jl]))    # :156-157  0.5 _d 0
        vFldY = vFldY.at[i, jl].set(0.5*(tmpV[i, jl-1] + tmpV[i, jl]))    # :158-159
        if applyMask:                                                  # :160-163
            uFldX = uFldX.at[i, jl].set(uFldX[i, jl]*maskW[i, jl, kk])
            vFldY = vFldY.at[i, jl].set(vFldY[i, jl]*maskS[i, jl, kk])
    return uFldX, vFldY, uFldE, vFldN
