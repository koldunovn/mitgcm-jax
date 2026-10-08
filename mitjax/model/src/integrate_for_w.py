"""INTEGRATE_FOR_W: model/src/integrate_for_w.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j


def integrate_for_w(k, uFld, vFld, mFld, rStarDhDt, wFld, myIter, *, cfg, grid, params):
    """INTEGRATE_FOR_W( bi, bj, k, uFld, vFld, mFld, rStarDhDt, wFld, myIter, myThid )
    @63cdc0b model/src/integrate_for_w.F:3-201

    C     *==========================================================*
    C     | SUBROUTINE INTEGRATE_FOR_W
    C     | o Integrate for vertical velocity.
    C     *==========================================================*
    C     uFld, vFld :: Zonal and meridional flow
    C     mFld       :: added mass
    C     rStarDhDt  :: relative time derivative of column thickness = d.eta/dt / H
    C     wFld       :: Vertical flow

    uFld, vFld, wFld: FArrays (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr) (bi,bj implicit); rStarDhDt (i, j) or None without
    NONLIN_FRSURF; mFld unused (ALLOW_ADDFLUID with selectAddFluid >= 1 is not ported: it raises). k a Python int: the caller
    (INTEGR_CONTINUITY) runs k = Nr..1 and level k reads wFld(k+1). Returns wFld with level k set on 1:sNx, 1:sNy.
    Ported branches (static values of `params` as INI_PARMS leaves them): rigidLid (:93-120); r* with select_rStar
    /= 0 under NONLIN_FRSURF (:123-147); the free surface r coordinate (:174-197). Hybrid sigma (:150-171) raises.
    `_hFacW`/`_hFacS` are hFacW/hFacS (model/inc/HFACW_MACROS.h:37-39); ALLOW_DEPTH_CONTROL raises.
    """
    sz = cfg.size
    Nr = sz.Nr
    if cfg.cpp.ALLOW_ADDFLUID and params.selectAddFluid >= 1:                  # :80-89 (lane B: IF on a switch)
        raise NotImplementedError("INTEGRATE_FOR_W: selectAddFluid >= 1 (mFld, :80-89) is not ported")
    # :57-59  local 2-D arrays (1-OLx:sNx+OLx,1-OLy:sNy+OLy), declared like dxG
    uTrans = grid.dxG.local("uTrans")
    vTrans = grid.dxG.local("vTrans")
    conv2d = grid.dxG.local("conv2d")
    if cfg.cpp.ALLOW_DEPTH_CONTROL:
        # _hFacW expands textually to hFacW*maskW inside the product (HFACW_MACROS.h:33-35): not ported
        raise NotImplementedError("INTEGRATE_FOR_W: ALLOW_DEPTH_CONTROL (_hFacW = hFacW*maskW) is not ported")

    # :64-73  volume transports through tracer cell faces
    j, i = loop_j(1, sz.sNy+1), loop_i(1, sz.sNx+1)
    uTrans = uTrans.at[i, j].set(uFld[i, j, k]
                                 * grid.dyG[i, j]*grid.deepFacC[k]*params.rhoFacC[k]
                                 * grid.drF[k]*grid.hFacW[i, j, k])
    vTrans = vTrans.at[i, j].set(vFld[i, j, k]
                                 * grid.dxG[i, j]*grid.deepFacC[k]*params.rhoFacC[k]
                                 * grid.drF[k]*grid.hFacS[i, j, k])
    # :74-79
    j, i = loop_j(1, sz.sNy), loop_i(1, sz.sNx)
    conv2d = conv2d.at[i, j].set(-(uTrans[i+1, j]-uTrans[i, j]
                                   + vTrans[i, j+1]-vTrans[i, j]))

    # :91-198  vertical transport through face k
    if params.rigidLid:                                                 # :93-120
        if k == 1:
            wFld = wFld.at[i, j, k].set(0.0)                            # :98  0. (REAL*4 zero)
        elif k == Nr:
            wFld = wFld.at[i, j, k].set(
                conv2d[i, j]*grid.recip_rA[i, j]
                * grid.maskC[i, j, k]*grid.maskC[i, j, k-1]
                * grid.recip_deepFac2F[k]*params.recip_rhoFacF[k])
        else:
            wFld = wFld.at[i, j, k].set(
                (wFld[i, j, k+1]*grid.deepFac2F[k+1]*params.rhoFacF[k+1]
                 + conv2d[i, j]*grid.recip_rA[i, j]
                 )*grid.maskC[i, j, k]*grid.maskC[i, j, k-1]
                * grid.recip_deepFac2F[k]*params.recip_rhoFacF[k])
    elif cfg.cpp.NONLIN_FRSURF and not cfg.cpp.DISABLE_RSTAR_CODE and params.select_rStar != 0:  # :123-147
        if k == Nr:
            wFld = wFld.at[i, j, k].set(
                (conv2d[i, j]*grid.recip_rA[i, j]
                 - rStarDhDt[i, j]*grid.drF[k]*grid.h0FacC[i, j, k]
                 )*grid.maskC[i, j, k]
                * grid.recip_deepFac2F[k]*params.recip_rhoFacF[k])
        else:
            wFld = wFld.at[i, j, k].set(
                (wFld[i, j, k+1]*grid.deepFac2F[k+1]*params.rhoFacF[k+1]
                 + conv2d[i, j]*grid.recip_rA[i, j]
                 - rStarDhDt[i, j]*grid.drF[k]*grid.h0FacC[i, j, k]
                 )*grid.maskC[i, j, k]
                * grid.recip_deepFac2F[k]*params.recip_rhoFacF[k])
    elif cfg.cpp.NONLIN_FRSURF and not cfg.cpp.DISABLE_SIGMA_CODE and params.selectSigmaCoord != 0:  # :150-171
        raise NotImplementedError("INTEGRATE_FOR_W: hybrid sigma coordinate (selectSigmaCoord /= 0) is not ported")
    else:                                                               # :174-197  free surface (r coordinate)
        if k == Nr:
            wFld = wFld.at[i, j, k].set(
                conv2d[i, j]*grid.recip_rA[i, j]
                * grid.maskC[i, j, k]
                * grid.recip_deepFac2F[k]*params.recip_rhoFacF[k])
        else:
            wFld = wFld.at[i, j, k].set(
                (wFld[i, j, k+1]*grid.deepFac2F[k+1]*params.rhoFacF[k+1]
                 + conv2d[i, j]*grid.recip_rA[i, j]
                 )*grid.maskC[i, j, k]
                * grid.recip_deepFac2F[k]*params.recip_rhoFacF[k])
    return wFld
