"""MOM_FLUXFORM: pkg/mom_fluxform/mom_fluxform.F @63cdc0b (the driver of the flux-form momentum kernels)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MIN
from mitjax.pkg.mom_common.mom_calc_hfacz import mom_calc_hfacz
from mitjax.pkg.mom_common.mom_calc_ke import mom_calc_ke
from mitjax.pkg.mom_common.mom_u_botdrag_coeff import mom_u_botdrag_coeff
from mitjax.pkg.mom_common.mom_u_coriolis_nh import mom_u_coriolis_nh
from mitjax.pkg.mom_common.mom_u_metric_nh import mom_u_metric_nh
from mitjax.pkg.mom_common.mom_u_rviscflux import mom_u_rviscflux
from mitjax.pkg.mom_common.mom_u_sidedrag import mom_u_sidedrag
from mitjax.pkg.mom_common.mom_v_botdrag_coeff import mom_v_botdrag_coeff
from mitjax.pkg.mom_common.mom_v_metric_nh import mom_v_metric_nh
from mitjax.pkg.mom_common.mom_v_rviscflux import mom_v_rviscflux
from mitjax.pkg.mom_common.mom_v_sidedrag import mom_v_sidedrag
from mitjax.pkg.mom_fluxform.mom_calc_rtrans import mom_calc_rtrans
from mitjax.pkg.mom_fluxform.mom_u_adv_uu import mom_u_adv_uu
from mitjax.pkg.mom_fluxform.mom_u_adv_vu import mom_u_adv_vu
from mitjax.pkg.mom_fluxform.mom_u_adv_wu import mom_u_adv_wu
from mitjax.pkg.mom_fluxform.mom_u_coriolis import mom_u_coriolis
from mitjax.pkg.mom_fluxform.mom_u_del2u import mom_u_del2u
from mitjax.pkg.mom_fluxform.mom_u_metric_sphere import mom_u_metric_sphere
from mitjax.pkg.mom_fluxform.mom_u_xviscflux import mom_u_xviscflux
from mitjax.pkg.mom_fluxform.mom_u_yviscflux import mom_u_yviscflux
from mitjax.pkg.mom_fluxform.mom_v_adv_uv import mom_v_adv_uv
from mitjax.pkg.mom_fluxform.mom_v_adv_vv import mom_v_adv_vv
from mitjax.pkg.mom_fluxform.mom_v_adv_wv import mom_v_adv_wv
from mitjax.pkg.mom_fluxform.mom_v_coriolis import mom_v_coriolis
from mitjax.pkg.mom_fluxform.mom_v_del2v import mom_v_del2v
from mitjax.pkg.mom_fluxform.mom_v_metric_sphere import mom_v_metric_sphere
from mitjax.pkg.mom_fluxform.mom_v_xviscflux import mom_v_xviscflux
from mitjax.pkg.mom_fluxform.mom_v_yviscflux import mom_v_yviscflux

_OPT = "MOM_FLUXFORM_OPTIONS.h"     # mom_fluxform.F:25


def mom_fluxform(k, iMin, iMax, jMin, jMax, kappaRU, kappaRV, fVerUkm, fVerVkm, fVerUkp, fVerVkp, guDiss, gvDiss,
                 myTime, myIter, *, cfg, grid, params, state, visc=None):
    """MOM_FLUXFORM( bi,bj,k,iMin,iMax,jMin,jMax, kappaRU, kappaRV, fVerUkm, fVerVkm, fVerUkp, fVerVkp,
    guDiss, gvDiss, myTime, myIter, myThid )   @63cdc0b pkg/mom_fluxform/mom_fluxform.F:42-1151

    C     | S/R MOM_FLUXFORM
    C     | o Form the right hand-side of the momentum equation.
    C     | Terms are evaluated one layer at a time working from the bottom to the top. ...
    C     kappaRU   :: vertical viscosity at U points
    C     kappaRV   :: vertical viscosity at V points
    C     fVerUkm   :: vertical advective flux of U, interface above (k-1/2)
    C     fVerVkm   :: vertical advective flux of V, interface above (k-1/2)
    C     fVerUkp   :: vertical advective flux of U, interface below (k+1/2)
    C     fVerVkp   :: vertical advective flux of V, interface below (k+1/2)
    C     guDiss    :: dissipation tendency (all explicit terms), u component
    C     gvDiss    :: dissipation tendency (all explicit terms), v component

    Returns (state with gU, gV of level k, fVerUkm, fVerVkm, fVerUkp, fVerVkp, guDiss, gvDiss): the Fortran
    argument order of the U and O arguments (KERNEL_GUIDE §4). `k` is a Python int (the caller's DO k);
    kappaRU, kappaRV are (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr+1); the flux arguments are 2-D. Kernels: MOM lane
    (mitjax/pkg/mom_common, mitjax/pkg/mom_fluxform), called as the Fortran calls them; PARAMS.h from `params`
    (mitjax/model/src/ini_parms.Params). deepFacAdv (MOM_VISC.h /MOM_GRID_COPY/) is params.deepFacAdv.

    Ported: everything this file executes for the flux-form M1 variants; M3 lane MLAdjust: useVariableVisc
    (MOM_CALC_HDIV/RELVORT3/TENSION/STRAIN, the QGL stretching, the side masks, MOM_CALC_VISC: :330-366, :453-458;
    `visc`: MOM_VISC.h). Raise: MOM_BOUNDARY_CONSERVE, usingCylindricalGrid metric terms,
    MOM_V_CORIOLIS_NH (curvilinear / rotated grid), useDiagnostics (DIAGNOSTICS_FILL and botDragU/V
    accumulation), ALLOW_AUTODIFF-only lines (STOREs). Statement order and association as in the Fortran; the
    point loops are independent."""
    if cfg.cpp.flag("MOM_BOUNDARY_CONSERVE", _OPT):
        raise NotImplementedError("MOM_FLUXFORM: MOM_BOUNDARY_CONSERVE is not ported")
    if params.useDiagnostics:
        raise NotImplementedError("MOM_FLUXFORM: useDiagnostics (DIAGNOSTICS_FILL, botDragU/V) is not ported")
    if params.useVariableVisc and visc is None:
        raise ValueError("MOM_FLUXFORM: useVariableVisc needs `visc` (MOM_VISC.h of MOM_INIT_FIXED)")
    sz = cfg.size
    OLx, OLy, sNx, sNy = sz.OLx, sz.OLy, sz.sNx, sz.sNy
    kw = dict(cfg=cfg, grid=grid, params=params)
    kwg = dict(cfg=cfg, grid=grid)
    g = grid
    uVel, vVel, wVel, gU, gV = state.uVel, state.vVel, state.wVel, state.gU, state.gV
    rkSign = g.rkSign
    recip_rhoFacC, rhoFacC = params.recip_rhoFacC, params.rhoFacC
    deepFacAdv = params.deepFacAdv

    # :204-233 initialise local arrays (0. : REAL*4 literal, exact)
    jA = loop_j(1-OLy, sNy+OLy)
    iA = loop_i(1-OLx, sNx+OLx)
    z = guDiss.local("zero")
    gAdd = z.at[iA, jA].set(0.)
    del2 = z.at[iA, jA].set(0.)
    uCf = z.at[iA, jA].set(0.)
    vCf = z.at[iA, jA].set(0.)
    vMT = z.at[iA, jA].set(0.)
    fZon = z.at[iA, jA].set(0.)
    fMer = z.at[iA, jA].set(0.)
    fVrUp = z.at[iA, jA].set(0.)
    fVrDw = z.at[iA, jA].set(0.)
    rTransU = z.at[iA, jA].set(0.)
    rTransV = z.at[iA, jA].set(0.)
    guDiss = guDiss.at[iA, jA].set(0.)                                         # :226
    gvDiss = gvDiss.at[iA, jA].set(0.)                                         # :227

    # :237-255 the term on/off factors (`*1.`: REAL*4 1., exact)
    uDudxFac = params.afFacMom*1.
    AhDudxFac = params.vfFacMom*1.
    vDudyFac = params.afFacMom*1.
    AhDudyFac = params.vfFacMom*1.
    wDudrFac = params.afFacMom*1.
    ArDudrFac = params.vfFacMom*1.
    mtFacU = params.mtFacMom*1.
    mtNHFacU = 1.
    fuFac = params.cfFacMom*1.
    uDvdxFac = params.afFacMom*1.
    AhDvdxFac = params.vfFacMom*1.
    vDvdyFac = params.afFacMom*1.
    AhDvdyFac = params.vfFacMom*1.
    wDvdrFac = params.afFacMom*1.
    ArDvdrFac = params.vfFacMom*1.
    mtFacV = params.mtFacMom*1.
    mtNHFacV = 1.
    fvFac = params.cfFacMom*1.
    if params.implicitViscosity:                                                # :257-260
        ArDudrFac = 0.
        ArDvdrFac = 0.
    # :264-268 sideMaskFac: set in the useVariableVisc block, its only reader (:346-353)
    # :270-277 bottomDragTerms; `bottomDragLinear.NE.0.` decided on the host (params.bottomDragLinear_ne_0)
    bottomDragTerms = (params.selectImplicitDrag == 0
                       and (params.no_slip_bottom or params.selectBotDragQuadr >= 0
                            or params.bottomDragLinear_ne_0))

    hFacZ, r_hFacZ = mom_calc_hfacz(k, z, z, **kwg)                           # :280
    xA = z.local("xA")
    yA = z.local("yA")
    h0FacZ = z.local("h0FacZ")
    xA = xA.at[iA, jA].set(g.dyG[iA, jA]*g.deepFacC[k]                         # :289-290
                           * g.drF[k]*g.hFacW[iA, jA, k])
    yA = yA.at[iA, jA].set(g.dxG[iA, jA]*g.deepFacC[k]                         # :291-292
                           * g.drF[k]*g.hFacS[iA, jA, k])
    h0FacZ = h0FacZ.at[iA, jA].set(hFacZ[iA, jA])                               # :293
    if cfg.cpp.NONLIN_FRSURF:                                                   # :296-307
        if params.momViscosity and params.no_slip_sides and params.nonlinFreeSurf > 0:
            j2 = loop_j(2-OLy, sNy+OLy)
            i2 = loop_i(2-OLx, sNx+OLx)
            h0FacZ = h0FacZ.at[i2, j2].set(MIN(MIN(g.h0FacW[i2, j2, k], g.h0FacW[i2, j2-1, k], p="b"),     # :301-303
                                               MIN(g.h0FacS[i2, j2, k], g.h0FacS[i2-1, j2, k], p="b"), p="b"))

    uFld = z.local("uFld").at[iA, jA].set(uVel[iA, jA, k])                      # :315
    vFld = z.local("vFld").at[iA, jA].set(vVel[iA, jA, k])                      # :316
    uTrans = z.local("uTrans").at[iA, jA].set(uFld[iA, jA]*xA[iA, jA]*rhoFacC[k])   # :324
    vTrans = z.local("vTrans").at[iA, jA].set(vFld[iA, jA]*yA[iA, jA]*rhoFacC[k])   # :325

    KE = mom_calc_ke(k, 2, uFld, vFld, z.local("KE"), **kwg)                   # :329
    if params.useVariableVisc:                                                  # :330-366 (M3 lane MLAdjust)
        from mitjax.pkg.mom_common.mom_calc_hdiv import mom_calc_hdiv
        from mitjax.pkg.mom_common.mom_calc_relvort3 import mom_calc_relvort3
        from mitjax.pkg.mom_common.mom_calc_strain import mom_calc_strain
        from mitjax.pkg.mom_common.mom_calc_tension import mom_calc_tension
        sideMaskFac = params.sideDragFactor if params.no_slip_sides else 0.     # :264-268  0. _d 0
        hDiv = mom_calc_hdiv(k, 2, uFld, vFld, z.local("hDiv").at[iA, jA].set(0.), **kwg)   # :218, :331
        vort3 = mom_calc_relvort3(k, uFld, vFld, hFacZ, z.local("vort3").at[iA, jA].set(0.), **kw)   # :219, :332
        tension = mom_calc_tension(k, uFld, vFld, z.local("tension").at[iA, jA].set(0.), **kwg)   # :221, :333
        strain = mom_calc_strain(k, uFld, vFld, hFacZ, z.local("strain").at[iA, jA].set(0.), **kwg)   # :220, :334
        stretching = z.local("stretching").at[iA, jA].set(0.)                   # :222
        if cfg.cpp.flag("ALLOW_LEITH_QG", "MOM_COMMON_OPTIONS.h") and params.viscC2LeithQG_ne_0:   # :335-345
            from mitjax.pkg.mom_common.mom_visc_qgl import mom_visc_qgl_limit, mom_visc_qgl_stretch
            Nsquare = z.local("Nsquare").at[iA, jA].set(0.)                     # :224
            stretching, Nsquare = mom_visc_qgl_stretch(k, stretching, Nsquare, myTime, myIter, state=state, **kw)
            stretching = mom_visc_qgl_limit(k, stretching, Nsquare, uFld, vFld, vort3, myTime, myIter, **kwg)
        land = hFacZ[iA, jA] == 0.                                              # :346-353
        vort3 = vort3.at[iA, jA].set(jnp.where(land, sideMaskFac*vort3[iA, jA], vort3[iA, jA]))
        strain = strain.at[iA, jA].set(jnp.where(land, sideMaskFac*strain[iA, jA], strain[iA, jA]))

    if params.momAdvection and k == 1:                                          # :384-419
        rTransU, rTransV, *st_ = mom_calc_rtrans(k, rTransU, rTransV, myTime, myIter, state=state, **kw)   # :405
        state = st_[0] if st_ else state                     # GO lane: r* dWtrans (MOM_FLUXFORM.h) in the State
        fVerUkm = mom_u_adv_wu(k, deepFacAdv, uVel, wVel, rTransU, fVerUkm, **kw)                  # :410
        fVerVkm = mom_v_adv_wv(k, deepFacAdv, vVel, wVel, rTransV, fVerVkm, **kw)                  # :414
    if params.momAdvection:                                                     # :422-426
        rTransU, rTransV, *st_ = mom_calc_rtrans(k+1, rTransU, rTransV, myTime, myIter, state=state, **kw)
        state = st_[0] if st_ else state                     # GO lane: r* dWtrans (MOM_FLUXFORM.h) in the State

    if params.momViscosity:                                                     # :444-470
        viscAh_D = z.local("viscAh_D").at[iA, jA].set(params.viscAhD)
        viscAh_Z = z.local("viscAh_Z").at[iA, jA].set(params.viscAhZ)
        viscA4_D = z.local("viscA4_D").at[iA, jA].set(params.viscA4D)
        viscA4_Z = z.local("viscA4_Z").at[iA, jA].set(params.viscA4Z)
        if params.useVariableVisc:                                              # :453-458 (M3 lane MLAdjust)
            from mitjax.pkg.mom_common.mom_calc_visc import mom_calc_visc
            viscAh_Z, viscAh_D, viscA4_Z, viscA4_D, hDiv = mom_calc_visc(
                k, viscAh_Z, viscAh_D, viscA4_Z, viscA4_D,
                hDiv, vort3, tension, strain, stretching, KE, hFacZ, visc=visc, **kw)

    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    # ---- U component
    if params.momAdvection:                                                     # :476
        fZon = mom_u_adv_uu(k, uTrans, uFld, fZon, **kwg)                       # :494
        fMer = mom_u_adv_vu(k, vTrans, uFld, fMer, **kw)                        # :498
        fVerUkp = mom_u_adv_wu(k+1, deepFacAdv, uVel, wVel, rTransU, fVerUkp, **kw)   # :502
        rAdvDeepFac = wDudrFac*rkSign                                           # :508
        if params.useNHMTerms:                                                  # :510
            rAdvDeepFac = rAdvDeepFac*g.recip_deepFacC[k]
        if params.selectMetricTerms == 3:                                       # :512-525
            gU = gU.at[i, j, k].set(
                -g.recip_hFacW[i, j, k]*g.recip_drF[k]
                * g.recip_rAw[i, j]*g.recip_deepFac2C[k]*recip_rhoFacC[k]
                * ((fZon[i, j] - fZon[i-1, j])*uDudxFac
                   + (fMer[i, j+1] - fMer[i, j])*vDudyFac
                   * g.recip_dxC[i, j]
                   + (fVerUkp[i, j] - fVerUkm[i, j])*rAdvDeepFac))
        else:                                                                   # :526-543
            gU = gU.at[i, j, k].set(
                -g.recip_hFacW[i, j, k]*g.recip_drF[k]
                * g.recip_rAw[i, j]*g.recip_deepFac2C[k]*recip_rhoFacC[k]
                * ((fZon[i, j] - fZon[i-1, j])*uDudxFac
                   + (fMer[i, j+1] - fMer[i, j])*vDudyFac
                   + (fVerUkp[i, j] - fVerUkm[i, j])*rAdvDeepFac))
        if cfg.cpp.NONLIN_FRSURF and not cfg.cpp.DISABLE_RSTAR_CODE:           # :553-574 (GO lane)
            if params.select_rStar > 0:                                         # :556-564
                gU = gU.at[i, j, k].set(gU[i, j, k]
                                        - (state.rStarExpW[i, j] - 1.)/params.deltaTFreeSurf   # 1. _d 0
                                        * uVel[i, j, k])
            if params.select_rStar < 0:                                         # :565-572
                raise NotImplementedError("MOM_FLUXFORM: select_rStar < 0 (rStarDhWDt term) is not ported")
    else:                                                                       # :590-599
        gU = gU.at[iA, jA, k].set(0.)                                           # 0. _d 0

    if params.momViscosity:                                                     # :601-722
        if params.useBiharmonicVisc:                                            # :605-607
            del2 = mom_u_del2u(k, uFld, hFacZ, h0FacZ, del2, **kw)
        fZon = mom_u_xviscflux(k, uFld, del2, fZon, viscAh_D, viscA4_D, **kwg)   # :613
        fMer = mom_u_yviscflux(k, uFld, del2, hFacZ, fMer, viscAh_Z, viscA4_Z, **kwg)   # :617
        if not params.implicitViscosity:                                        # :621-624
            fVrUp = mom_u_rviscflux(k, uVel, kappaRU, fVrUp, **kw)
            fVrDw = mom_u_rviscflux(k+1, uVel, kappaRU, fVrDw, **kw)
        guDiss = guDiss.at[i, j].set(                                           # :630-642
            -g.recip_hFacW[i, j, k]*g.recip_drF[k]
            * g.recip_rAw[i, j]*g.recip_deepFac2C[k]
            * ((fZon[i, j] - fZon[i-1, j])*AhDudxFac
               + (fMer[i, j+1] - fMer[i, j])*AhDudyFac
               + (fVrDw[i, j] - fVrUp[i, j])*rkSign*ArDudrFac
               * recip_rhoFacC[k]))
        if params.no_slip_sides:                                                # :656-669
            gAdd = mom_u_sidedrag(k, uFld, del2, h0FacZ, viscAh_Z, viscA4_Z, params.useHarmonicVisc,
                                  params.useBiharmonicVisc, params.useVariableVisc, gAdd, **kw)
            guDiss = guDiss.at[i, j].set(guDiss[i, j] + gAdd[i, j])
        if bottomDragTerms:                                                     # :671-697
            KE, cDrag = mom_u_botdrag_coeff(k, True, uFld, vFld, kappaRU, KE, z.local("cDrag"), myIter, **kw)
            guDiss = guDiss.at[i, j].set(guDiss[i, j]
                                         - cDrag[i, j]*uFld[i, j]
                                         * g.recip_hFacW[i, j, k]*g.recip_drF[k])

    if params.useNHMTerms and not params.deepAtmosphere:                        # :735-751
        vMT = mom_u_metric_nh(k, uFld, wVel, vMT, **kw)
        gU = gU.at[i, j, k].set(gU[i, j, k]+mtNHFacU*vMT[i, j])
    if params.selectMetricTerms >= 1 and params.selectMetricTerms != 3:        # :752-783
        if params.usingSphericalPolarGrid:
            gAdd = mom_u_metric_sphere(k, uFld, vFld, gAdd, **kw)
            gU = gU.at[i, j, k].set(gU[i, j, k]+mtFacU*gAdd[i, j])
        if params.usingCylindricalGrid:
            raise NotImplementedError("MOM_FLUXFORM: MOM_U_METRIC_CYLINDER is not ported")

    # ---- V component
    if params.momAdvection:                                                     # :789
        fZon = mom_v_adv_uv(k, uTrans, vFld, fZon, **kwg)                       # :806
        fMer = mom_v_adv_vv(k, vTrans, vFld, fMer, **kwg)                       # :810
        fVerVkp = mom_v_adv_wv(k+1, deepFacAdv, vVel, wVel, rTransV, fVerVkp, **kw)   # :814
        rAdvDeepFac = wDvdrFac*rkSign                                           # :820
        if params.useNHMTerms:                                                  # :822
            rAdvDeepFac = rAdvDeepFac*g.recip_deepFacC[k]
        gV = gV.at[i, j, k].set(                                                # :826-837
            -g.recip_hFacS[i, j, k]*g.recip_drF[k]
            * g.recip_rAs[i, j]*g.recip_deepFac2C[k]*recip_rhoFacC[k]
            * ((fZon[i+1, j] - fZon[i, j])*uDvdxFac
               + (fMer[i, j] - fMer[i, j-1])*vDvdyFac
               + (fVerVkp[i, j] - fVerVkm[i, j])*rAdvDeepFac))
        if cfg.cpp.NONLIN_FRSURF and not cfg.cpp.DISABLE_RSTAR_CODE:           # :849-870 (GO lane)
            if params.select_rStar > 0:                                         # :852-860
                gV = gV.at[i, j, k].set(gV[i, j, k]
                                        - (state.rStarExpS[i, j] - 1.)/params.deltaTFreeSurf   # 1. _d 0
                                        * vVel[i, j, k])
            if params.select_rStar < 0:                                         # :861-868
                raise NotImplementedError("MOM_FLUXFORM: select_rStar < 0 (rStarDhSDt term) is not ported")
    else:                                                                       # :886-895
        gV = gV.at[iA, jA, k].set(0.)

    if params.momViscosity:                                                     # :897-1017
        if params.useBiharmonicVisc:                                            # :900-902
            del2 = mom_v_del2v(k, vFld, hFacZ, h0FacZ, del2, **kw)
        fZon = mom_v_xviscflux(k, vFld, del2, hFacZ, fZon, viscAh_Z, viscA4_Z, **kwg)   # :908
        fMer = mom_v_yviscflux(k, vFld, del2, fMer, viscAh_D, viscA4_D, **kwg)   # :912
        if not params.implicitViscosity:                                        # :916-919
            fVrUp = mom_v_rviscflux(k, vVel, kappaRV, fVrUp, **kw)
            fVrDw = mom_v_rviscflux(k+1, vVel, kappaRV, fVrDw, **kw)
        gvDiss = gvDiss.at[i, j].set(                                           # :925-937
            -g.recip_hFacS[i, j, k]*g.recip_drF[k]
            * g.recip_rAs[i, j]*g.recip_deepFac2C[k]
            * ((fZon[i+1, j] - fZon[i, j])*AhDvdxFac
               + (fMer[i, j] - fMer[i, j-1])*AhDvdyFac
               + (fVrDw[i, j] - fVrUp[i, j])*rkSign*ArDvdrFac
               * recip_rhoFacC[k]))
        if params.no_slip_sides:                                                # :951-964
            gAdd = mom_v_sidedrag(k, vFld, del2, h0FacZ, viscAh_Z, viscA4_Z, params.useHarmonicVisc,
                                  params.useBiharmonicVisc, params.useVariableVisc, gAdd, **kw)
            gvDiss = gvDiss.at[i, j].set(gvDiss[i, j] + gAdd[i, j])
        if bottomDragTerms:                                                     # :966-992
            KE, cDrag = mom_v_botdrag_coeff(k, True, uFld, vFld, kappaRV, KE, z.local("cDrag"), myIter, **kw)
            gvDiss = gvDiss.at[i, j].set(gvDiss[i, j]
                                         - cDrag[i, j]*vFld[i, j]
                                         * g.recip_hFacS[i, j, k]*g.recip_drF[k])

    if params.useNHMTerms and not params.deepAtmosphere:                        # :1030-1045
        vMT = mom_v_metric_nh(k, vFld, wVel, vMT, **kw)
        gV = gV.at[i, j, k].set(gV[i, j, k]+mtNHFacV*vMT[i, j])
    if params.selectMetricTerms >= 1:                                           # :1046-1077
        if params.usingSphericalPolarGrid:
            gAdd = mom_v_metric_sphere(k, uFld, gAdd, **kw)
            gV = gV.at[i, j, k].set(gV[i, j, k]+mtFacV*gAdd[i, j])
        if params.usingCylindricalGrid:
            raise NotImplementedError("MOM_FLUXFORM: MOM_V_METRIC_CYLINDER is not ported")

    # ---- Coriolis
    if not params.useCDscheme:                                                  # :1082-1109
        uCf = mom_u_coriolis(k, vFld, uCf, **kw)                               # :1092
        vCf = mom_v_coriolis(k, uFld, vCf, **kw)                               # :1093
        gU = gU.at[i, j, k].set(gU[i, j, k] + fuFac*uCf[i, j])                  # :1099
        gV = gV.at[i, j, k].set(gV[i, j, k] + fvFac*vCf[i, j])                  # :1100
    if params.select3dCoriScheme >= 1:                                          # :1112-1128
        uCf = mom_u_coriolis_nh(k, wVel, uCf, **kw)
        gU = gU.at[i, j, k].set(gU[i, j, k] + fuFac*uCf[i, j])
        if params.usingCurvilinearGrid or params.rotateGrid:
            raise NotImplementedError("MOM_FLUXFORM: MOM_V_CORIOLIS_NH is not ported")

    gU = gU.at[i, j, k].set(gU[i, j, k]*g.maskW[i, j, k])                       # :1133
    guDiss = guDiss.at[i, j].set(guDiss[i, j]*g.maskW[i, j, k])                 # :1134
    gV = gV.at[i, j, k].set(gV[i, j, k]*g.maskS[i, j, k])                       # :1135
    gvDiss = gvDiss.at[i, j].set(gvDiss[i, j]*g.maskS[i, j, k])                 # :1136
    return state.replace(gU=gU, gV=gV), fVerUkm, fVerVkm, fVerUkp, fVerVkp, guDiss, gvDiss

