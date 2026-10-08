"""CALC_ADV_FLOW: model/src/calc_adv_flow.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j


def calc_adv_flow(uFld, vFld, wFld, rTrans, uTrans, vTrans, rTransKp, maskUp, xA, yA, k, *, cfg, grid, params):
    """CALC_ADV_FLOW( uFld, vFld, wFld, rTrans, uTrans, vTrans, rTransKp, maskUp, xA, yA, k, bi, bj, myThid )
    @63cdc0b model/src/calc_adv_flow.F:7-134

    C     | SUBROUTINE CALC_ADV_FLOW
    C     | o Calculate common data (such as volume flux) for use
    C     |   by "Right hand side" subroutines.
    C     uFld     :: 3-D local copy of horizontal velocity, zonal  component
    C     vFld     :: 3-D local copy of horizontal velocity, merid. component
    C     wFld     :: 3-D local copy of vertical velocity
    C     rTrans   :: Vertical volume transport through interface k
    C     uTrans   :: Zonal volume transport through cell face
    C     vTrans   :: Meridional volume transport through cell face
    C     rTransKp :: Vertical volume transport through interface k+1
    C     maskUp   :: Land/water mask for Wvel points (interface k)
    C     xA       :: Tracer cell face area normal to X
    C     yA       :: Tracer cell face area normal to X

    Returns (rTrans, uTrans, vTrans, rTransKp, maskUp, xA, yA) (the U and O arguments in the Fortran order). uFld,
    vFld, wFld: (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr); the others 2-D. `k` a Python int (the caller's DO k), so the
    branches on k (:80, :113) are static. The macros `_dyG`, `_dxG`, `_hFacW`, `_hFacS` expand to the plain GRID.h
    fields (dyG(i,j,bi,bj), hFacW(i,j,k,bi,bj)) unless ALLOW_DEPTH_CONTROL (HFACW_MACROS.h:33-35), which raises.
    Both arms of :89-98 are ported: ALLOW_AUTODIFF (code_ad builds) recomputes rTransKp from wFld (:92-94), the plain
    build copies the rTrans of the previous call (:97); rhoFacF / rhoFacC are PARAMS.h arrays (params). The (i,j)
    nests are independent point by point (each reads only inputs). `0. _d 0` exact.
    """
    if cfg.cpp.ALLOW_DEPTH_CONTROL:
        raise NotImplementedError("CALC_ADV_FLOW: ALLOW_DEPTH_CONTROL (masked _hFacW/_hFacS) is not ported")
    sz = cfg.size
    g = grid
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    xA = xA.at[i, j].set(g.dyG[i, j]*g.deepFacC[k]                                  # :72-73
                         * g.drF[k]*g.hFacW[i, j, k])
    yA = yA.at[i, j].set(g.dxG[i, j]*g.deepFacC[k]                                  # :74-75
                         * g.drF[k]*g.hFacS[i, j, k])

    if k == sz.Nr:                                                                  # :80-85
        rTransKp = rTransKp.at[i, j].set(0.)
    elif cfg.cpp.ALLOW_AUTODIFF:                                                    # :89-94
        rTransKp = rTransKp.at[i, j].set(wFld[i, j, k+1]*g.rA[i, j]
                                         * g.maskC[i, j, k]*g.maskC[i, j, k+1]
                                         * g.deepFac2F[k+1]*params.rhoFacF[k+1])
    else:                                                                           # :95-98
        rTransKp = rTransKp.at[i, j].set(rTrans[i, j])

    uTrans = uTrans.at[i, j].set(uFld[i, j, k]*xA[i, j]*params.rhoFacC[k])          # :107
    vTrans = vTrans.at[i, j].set(vFld[i, j, k]*yA[i, j]*params.rhoFacC[k])          # :108

    if k == 1:                                                                      # :113-120
        maskUp = maskUp.at[i, j].set(0.)
        rTrans = rTrans.at[i, j].set(0.)
    else:                                                                           # :121-131
        maskUp = maskUp.at[i, j].set(g.maskC[i, j, k-1]*g.maskC[i, j, k])
        rTrans = rTrans.at[i, j].set(wFld[i, j, k]*g.rA[i, j]*maskUp[i, j]
                                     * g.deepFac2F[k]*params.rhoFacF[k])
    return rTrans, uTrans, vTrans, rTransKp, maskUp, xA, yA
