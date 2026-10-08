"""pkg/mom_vecinv/mom_vecinv.F: right-hand side of the momentum equation in vector-invariant form, one level
(MOM_VECINV)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MIN
from mitjax.pkg.mom_common.mom_calc_absvort3 import mom_calc_absvort3
from mitjax.pkg.mom_common.mom_calc_hdiv import mom_calc_hdiv
from mitjax.pkg.mom_common.mom_calc_hfacz import mom_calc_hfacz
from mitjax.pkg.mom_common.mom_calc_ke import mom_calc_ke
from mitjax.pkg.mom_common.mom_calc_relvort3 import mom_calc_relvort3
from mitjax.pkg.mom_common.mom_calc_strain import mom_calc_strain
from mitjax.pkg.mom_common.mom_calc_tension import mom_calc_tension
from mitjax.pkg.mom_common.mom_calc_visc import mom_calc_visc
from mitjax.pkg.mom_common.mom_u_botdrag_coeff import mom_u_botdrag_coeff
from mitjax.pkg.mom_common.mom_u_coriolis_nh import mom_u_coriolis_nh
from mitjax.pkg.mom_common.mom_u_metric_nh import mom_u_metric_nh
from mitjax.pkg.mom_common.mom_u_rviscflux import mom_u_rviscflux
from mitjax.pkg.mom_common.mom_u_sidedrag import mom_u_sidedrag
from mitjax.pkg.mom_common.mom_v_botdrag_coeff import mom_v_botdrag_coeff
from mitjax.pkg.mom_common.mom_v_coriolis_nh import mom_v_coriolis_nh
from mitjax.pkg.mom_common.mom_v_metric_nh import mom_v_metric_nh
from mitjax.pkg.mom_common.mom_v_rviscflux import mom_v_rviscflux
from mitjax.pkg.mom_common.mom_v_sidedrag import mom_v_sidedrag
from mitjax.pkg.mom_vecinv.mom_vi_coriolis import mom_vi_coriolis
from mitjax.pkg.mom_vecinv.mom_vi_del2uv import mom_vi_del2uv
from mitjax.pkg.mom_vecinv.mom_vi_hdissip import mom_vi_hdissip
from mitjax.pkg.mom_vecinv.mom_vi_u_coriolis import mom_vi_u_coriolis
from mitjax.pkg.mom_vecinv.mom_vi_u_grad_ke import mom_vi_u_grad_ke
from mitjax.pkg.mom_vecinv.mom_vi_u_vertshear import mom_vi_u_vertshear
from mitjax.pkg.mom_vecinv.mom_vi_v_coriolis import mom_vi_v_coriolis
from mitjax.pkg.mom_vecinv.mom_vi_v_grad_ke import mom_vi_v_grad_ke
from mitjax.pkg.mom_vecinv.mom_vi_v_vertshear import mom_vi_v_vertshear

_OPT = "MOM_VECINV_OPTIONS.h"       # mom_vecinv.F:1-2 (MOM_VECINV_OPTIONS.h, MOM_COMMON_OPTIONS.h)
_COPT = "MOM_COMMON_OPTIONS.h"
debLevC = 3                         # EEPARAMS.h:93  PARAMETER ( debLevC=3 )


def mom_vecinv(k, iMin, iMax, jMin, jMax, kappaRU, kappaRV, fVerUkm, fVerVkm, fVerUkp, fVerVkp, guDiss, gvDiss,
               myTime, myIter, *, cfg, grid, params, state, visc, w2=None, ctrlf=None):
    """MOM_VECINV(bi,bj,k,iMin,iMax,jMin,jMax, kappaRU, kappaRV, fVerUkm, fVerVkm, fVerUkp, fVerVkp,
                  guDiss, gvDiss, myTime, myIter, myThid)   @63cdc0b pkg/mom_vecinv/mom_vecinv.F:10-1006

    C     | S/R MOM_VECINV
    C     | o Form the right hand-side of the momentum equation.
    C     | Terms are evaluated one layer at a time working from
    C     | the bottom to the top. The vertically integrated
    C     | barotropic flow tendency term is evluated by summing the
    C     | tendencies.
    C     fVerUkm :: vertical viscous flux of U, interface above (k-1/2)
    C     fVerVkm :: vertical viscous flux of V, interface above (k-1/2)
    C     fVerUkp :: vertical viscous flux of U, interface below (k+1/2)
    C     fVerVkp :: vertical viscous flux of V, interface below (k+1/2)
    C     guDiss  :: dissipation tendency (all explicit terms), u component
    C     gvDiss  :: dissipation tendency (all explicit terms), v component

    `ctrlf` (CTRL_FIELDS.h: bottomDragFld) is passed to MOM_U/V_BOTDRAG_COEFF, which read it in builds that define
    ALLOW_BOTTOMDRAG_CONTROL (CTRL_OPTIONS.h view; the 90x40x15 code_ad build).

    Returns (fVerUkp, fVerVkp, guDiss, gvDiss, state): `state` (DYNVARS.h: uVel, vVel, wVel read, gU and gV written at
    level k on jMin..jMax x iMin..iMax) comes back with the new gU, gV. `visc` holds MOM_VISC.h (L2_D ... L4rdt_Z,
    deepFacAdv); `w2` is the per-tile W2_EXCH2_TOPOLOGY.h view of mom_calc_relvort3 (cube only). `k`, iMin..jMax
    (DYNAMICS' PARAMETERs) are Python ints; the LOGICAL/INTEGER selectors of PARAMS.h and MOM_VISC.h are static
    (`params.NAME`); bottomDragLinear.NE.0 (:243) selects code: the host flag `params.bottomDragLinear_ne_0`
    (ini_parms). myTime and myIter are passed on (myIter to the bottom-drag routines).

    M3 lane MLAdjust: useStrainTensionVisc (MOM_HDISSIP, :419-427). Not ported, raise: highOrderVorticity /
    upwindVorticity (MOM_VI_U/V_CORIOLIS_C4, :744-754, :769-779), the Langmuir arm (ALLOW_GGL90_LANGMUIR, :690-697), useShelfIce
    (:516-547, :626-657), DIAGNOSTICS_FILL and the botDragU/V accumulation with useDiagnostics, the
    WRITE_LOCAL_RL/MNC snapshots (writeDiag: decided on the host as DIFFERENT_MULTIPLE(diagFreq, ...) = .FALSE. for
    diagFreq = 0, different_multiple.F:41-43; diagFreq /= 0 raises), DEBUG_CS_CORNER_UV (debugLevel >= debLevC),
    MOM_USE_OLD_DEEP_VERT_ADV (#undef in every M2 build); ALLOW_MOM_TEND_EXTRA_DIAGS guards only diagnostics
    (raise with useDiagnostics; MLAdjust compiles it). Without ALLOW_MOM_VECINV the
    body is empty (:83-1003). The `fVerUkm(1,1) = fVerUkm(1,1)` lines (:155-156, ALLOW_AUTODIFF) and the TAF
    directives change nothing. The commented-out calls (:282, :663) are not ported.

    Locals: every local the Fortran initialises (:188-224) is initialised here the same way; hFacZ (:221,
    ALLOW_AUTODIFF only) and r_hFacZ are written on every point by MOM_CALC_HFACZ; the NaN-filled locals that the
    Fortran leaves uninitialised (cDrag, viscAh_Z/D, viscA4_Z/D before :365-372) show up if read before written.
    Loop order: the (i,j) loops run on the whole range at once; each point reads only its own earlier values or
    arrays completed before the loop.
    """
    if not cfg.cpp.flag("ALLOW_MOM_VECINV", _OPT):                 # :83, :1003
        return fVerUkp, fVerVkp, guDiss, gvDiss, state
    if cfg.cpp.flag("MOM_USE_OLD_DEEP_VERT_ADV", _COPT):
        raise NotImplementedError("MOM_VECINV: MOM_USE_OLD_DEEP_VERT_ADV is not ported")
    if (cfg.cpp.flag("ALLOW_MOM_TEND_EXTRA_DIAGS", _COPT) and cfg.cpp.flag("ALLOW_DIAGNOSTICS", _OPT)
            and params.useDiagnostics):                             # :249-260, :510-514, :535-545, ... (M3 MLAdjust)
        raise NotImplementedError("MOM_VECINV: the ALLOW_MOM_TEND_EXTRA_DIAGS diagnostics are not ported")
    p = params
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    recip_hFacW, recip_hFacS, recip_drF = grid.recip_hFacW, grid.recip_hFacS, grid.recip_drF
    recip_rAw, recip_rAs, rkSign, recip_deepFac2C = grid.recip_rAw, grid.recip_rAs, grid.rkSign, grid.recip_deepFac2C
    maskW, maskS = grid.maskW, grid.maskS
    recip_rhoFacC = p.recip_rhoFacC
    uVel, vVel, wVel, gU, gV = state.uVel, state.vVel, state.wVel, state.gU, state.gV
    diagnostics = cfg.cpp.flag("ALLOW_DIAGNOSTICS", _OPT) and p.useDiagnostics

#     writeDiag = DIFFERENT_MULTIPLE(diagFreq, myTime, deltaTClock)  (:164)
    if p.diagFreq != 0.:
        raise NotImplementedError("MOM_VECINV: diagFreq /= 0 (WRITE_LOCAL_RL snapshots) is not ported")
    writeDiag = False                                               # different_multiple.F:41-43 (freq = 0)

#--   Initialise intermediate terms (:188-224)
    jA = loop_j(1-OLy, sNy+OLy)
    iA = loop_i(1-OLx, sNx+OLx)
    vF = guDiss.local("vF").at[iA, jA].set(0.)
    vrF = guDiss.local("vrF").at[iA, jA].set(0.)
    uCf = guDiss.local("uCf").at[iA, jA].set(0.)
    vCf = guDiss.local("vCf").at[iA, jA].set(0.)
    del2u = guDiss.local("del2u").at[iA, jA].set(0.)
    del2v = guDiss.local("del2v").at[iA, jA].set(0.)
    dStar = guDiss.local("dStar").at[iA, jA].set(0.)
    zStar = guDiss.local("zStar").at[iA, jA].set(0.)
    guDiss = guDiss.at[iA, jA].set(0.)
    gvDiss = gvDiss.at[iA, jA].set(0.)
    vort3 = guDiss.local("vort3").at[iA, jA].set(0.)
    omega3 = guDiss.local("omega3").at[iA, jA].set(0.)
    KE = guDiss.local("KE").at[iA, jA].set(0.)
#-    need to initialise hDiv for MOM_VI_DEL2UV(call FILL_CS_CORNER_TR_RL)
    hDiv = guDiss.local("hDiv").at[iA, jA].set(0.)
    strain = guDiss.local("strain").at[iA, jA].set(0.)
    strainBC = guDiss.local("strainBC").at[iA, jA].set(0.)
    stretching = guDiss.local("stretching").at[iA, jA].set(0.)
    Nsquare = guDiss.local("Nsquare").at[iA, jA].set(0.)               # :217 (M3 lane MLAdjust)
    tension = guDiss.local("tension").at[iA, jA].set(0.)
    hFacZ = guDiss.local("hFacZ")
    if cfg.cpp.flag("ALLOW_AUTODIFF", _OPT):                       # :220-222
        hFacZ = hFacZ.at[iA, jA].set(0.)
    r_hFacZ = guDiss.local("r_hFacZ")
    h0FacZ = guDiss.local("h0FacZ")
    uFld = guDiss.local("uFld")
    vFld = guDiss.local("vFld")
    cDrag = guDiss.local("cDrag")

#--   Term by term tracer parmeters
    ArDudrFac = p.vfFacMom*1.                                       # :228
    ArDvdrFac = p.vfFacMom*1.                                       # :230

    if p.no_slip_sides:                                             # :234-238
        sideMaskFac = p.sideDragFactor
    else:
        sideMaskFac = 0.

    bottomDragTerms = (p.selectImplicitDrag == 0 and                # :240-247
                       (p.no_slip_bottom
                        or p.selectBotDragQuadr >= 0
                        or p.bottomDragLinear_ne_0))

#--   Calculate open water fraction at vorticity points
    hFacZ, r_hFacZ = mom_calc_hfacz(k, hFacZ, r_hFacZ, cfg=cfg, grid=grid)   # :263

#     Make local copies of horizontal flow field
    uFld = uFld.at[iA, jA].set(uVel[iA, jA, k])                     # :266-271
    vFld = vFld.at[iA, jA].set(vVel[iA, jA, k])

    KE = mom_calc_ke(k, p.selectKEscheme, uFld, vFld, KE, cfg=cfg, grid=grid)   # :284

    vort3 = mom_calc_relvort3(k, uFld, vFld, hFacZ, vort3, cfg=cfg, grid=grid, params=p, w2=w2)   # :286

#-    mask vort3 and account for no-slip / free-slip BC in vort3BC:
    vort3BC = guDiss.local("vort3BC").at[iA, jA].set(vort3[iA, jA])   # :289-297
    land = hFacZ[iA, jA] == 0.
    vort3BC = vort3BC.at[iA, jA].set(jnp.where(land, sideMaskFac*vort3BC[iA, jA], vort3BC[iA, jA]))
    vort3 = vort3.at[iA, jA].set(jnp.where(land, 0., vort3[iA, jA]))

    if p.momViscosity:                                              # :305
#--    For viscous term, compute horizontal divergence, tension & strain
#      and mask relative vorticity (free-slip case):
        h0FacZ = h0FacZ.at[iA, jA].set(hFacZ[iA, jA])               # :309-313
        if cfg.cpp.flag("NONLIN_FRSURF", _OPT):                     # :314-327
            if p.no_slip_sides and p.nonlinFreeSurf > 0:
                h0FacW, h0FacS = grid.h0FacW, grid.h0FacS
                j = loop_j(2-OLy, sNy+OLy)
                i = loop_i(2-OLx, sNx+OLx)
                h0FacZ = h0FacZ.at[i, j].set(MIN(                   # :318-320
                    MIN(h0FacW[i, j, k], h0FacW[i, j-1, k], p="b"),     # :318-320
                    MIN(h0FacS[i, j, k], h0FacS[i-1, j, k], p="b"), p="b"))   # :318-320

        hDiv = mom_calc_hdiv(k, 2, uFld, vFld, hDiv, cfg=cfg, grid=grid)   # :329

        if p.useVariableVisc or p.useStrainTensionVisc:             # :331-355
            tension = mom_calc_tension(k, uFld, vFld, tension, cfg=cfg, grid=grid)
            strain = mom_calc_strain(k, uFld, vFld, hFacZ, strain, cfg=cfg, grid=grid)
#-    mask strain and account for no-slip / free-slip BC in strainBC:
            strainBC = strainBC.at[iA, jA].set(strain[iA, jA])      # :335-343
            strainBC = strainBC.at[iA, jA].set(jnp.where(land, sideMaskFac*strainBC[iA, jA], strainBC[iA, jA]))
            strain = strain.at[iA, jA].set(jnp.where(land, 0., strain[iA, jA]))
            if cfg.cpp.flag("ALLOW_LEITH_QG", _COPT) and p.viscC2LeithQG_ne_0:   # :344-354 (M3 lane MLAdjust)
                from mitjax.pkg.mom_common.mom_visc_qgl import mom_visc_qgl_limit, mom_visc_qgl_stretch
                stretching, Nsquare = mom_visc_qgl_stretch(k, stretching, Nsquare, myTime, myIter, cfg=cfg,
                                                           grid=grid, params=p, state=state)   # :346-348
                stretching = mom_visc_qgl_limit(k, stretching, Nsquare, uFld, vFld, vort3, myTime, myIter,
                                                cfg=cfg, grid=grid)                         # :349-352

#--    Calculate Lateral Viscosities
        viscAh_D = guDiss.local("viscAh_D").at[iA, jA].set(p.viscAhD)   # :365-372
        viscAh_Z = guDiss.local("viscAh_Z").at[iA, jA].set(p.viscAhZ)
        viscA4_D = guDiss.local("viscA4_D").at[iA, jA].set(p.viscA4D)
        viscA4_Z = guDiss.local("viscA4_Z").at[iA, jA].set(p.viscA4Z)
        if p.useVariableVisc:                                       # :373-393
#-     uses vort3BC & strainBC which account for no-slip / free-slip BC
            viscAh_Z, viscAh_D, viscA4_Z, viscA4_D, hDiv = mom_calc_visc(
                k, viscAh_Z, viscAh_D, viscA4_Z, viscA4_D,
                hDiv, vort3BC, tension, strainBC, stretching,
                KE, hFacZ, cfg=cfg, grid=grid, params=p, visc=visc, w2=w2)

#      Calculate del^2 u and del^2 v for bi-harmonic term
        if p.useBiharmonicVisc:                                     # :400-407
            del2u, del2v, hDiv = mom_vi_del2uv(k, hDiv, vort3, hFacZ, del2u, del2v, cfg=cfg, grid=grid, params=p,
                                               w2=w2)
            dStar = mom_calc_hdiv(k, 2, del2u, del2v, dStar, cfg=cfg, grid=grid)
            zStar = mom_calc_relvort3(k, del2u, del2v, hFacZ, zStar, cfg=cfg, grid=grid, params=p, w2=w2)

#---   Calculate dissipation terms for U and V equations
        if p.useStrainTensionVisc:                                  # :419-427 (M3 lane MLAdjust)
#      use masked strain as if free-slip since side-drag is computed separately
            from mitjax.pkg.mom_common.mom_hdissip import mom_hdissip
            guDiss, gvDiss = mom_hdissip(
                k, tension, strain, hFacZ,
                viscAh_Z, viscAh_D, viscA4_Z, viscA4_D,
                p.useHarmonicVisc, p.useBiharmonicVisc, p.useVariableVisc,
                guDiss, gvDiss, cfg=cfg, grid=grid)
        else:                                                       # :428-435
            guDiss, gvDiss = mom_vi_hdissip(
                k, hDiv, vort3, dStar, zStar, hFacZ,
                viscAh_Z, viscAh_D, viscA4_Z, viscA4_D,
                p.useHarmonicVisc, p.useBiharmonicVisc, p.useVariableVisc,
                guDiss, gvDiss, cfg=cfg, grid=grid, params=p)

        j = loop_j(jMin, jMax)
        i = loop_i(iMin, iMax)
#---  Other dissipation terms in Zonal momentum equation
#--   Vertical flux (fVer is at upper face of "u" cell)
        if not p.implicitViscosity:                                 # :441-465
            vrF = mom_u_rviscflux(k+1, uVel, kappaRU, vrF, cfg=cfg, grid=grid, params=p)
#     Combine fluxes
            fVerUkp = fVerUkp.at[i, j].set(ArDudrFac*vrF[i, j])
#--   Tendency is minus divergence of the fluxes
            guDiss = guDiss.at[i, j].set(
                guDiss[i, j]
                -recip_hFacW[i, j, k]*recip_drF[k]
                *recip_rAw[i, j]
                *(fVerUkp[i, j] - fVerUkm[i, j])*rkSign
                *recip_deepFac2C[k]*recip_rhoFacC[k])

#-- No-slip and drag BCs appear as body forces in cell abutting topography
        if p.no_slip_sides:                                         # :468-481
#-     No-slip BCs impose a drag at walls...
            vF = mom_u_sidedrag(k, uFld, del2u, h0FacZ, viscAh_Z, viscA4_Z,
                                p.useHarmonicVisc, p.useBiharmonicVisc, p.useVariableVisc,
                                vF, cfg=cfg, grid=grid, params=p)
            guDiss = guDiss.at[i, j].set(guDiss[i, j]+vF[i, j])

#-    No-slip BCs impose a drag at bottom
        if bottomDragTerms:                                         # :484-515
            KE, cDrag = mom_u_botdrag_coeff(k, True, uFld, vFld, kappaRU, KE, cDrag, myIter,
                                            cfg=cfg, grid=grid, params=p, ctrlf=ctrlf)
            vF = vF.at[i, j].set(-cDrag[i, j]*uFld[i, j]
                                 *recip_hFacW[i, j, k]*recip_drF[k])
            guDiss = guDiss.at[i, j].set(guDiss[i, j] + vF[i, j])
            if p.useDiagnostics:                                    # :502-509
                raise NotImplementedError("MOM_VECINV: botDragU accumulation (useDiagnostics) is not ported")
        if cfg.cpp.flag("ALLOW_SHELFICE", _OPT) and p.useShelfIce and p.selectImplicitDrag == 0:   # :516-547
            raise NotImplementedError("MOM_VECINV: useShelfIce (SHELFICE_U/V_DRAG_COEFF) is not ported")

#---  Other dissipation terms in Meridional momentum equation
#--   Vertical flux (fVer is at upper face of "v" cell)
        if not p.implicitViscosity:                                 # :553-575
            vrF = mom_v_rviscflux(k+1, vVel, kappaRV, vrF, cfg=cfg, grid=grid, params=p)
#     Combine fluxes -> fVerV
            fVerVkp = fVerVkp.at[i, j].set(ArDvdrFac*vrF[i, j])
#--   Tendency is minus divergence of the fluxes
            gvDiss = gvDiss.at[i, j].set(
                gvDiss[i, j]
                -recip_hFacS[i, j, k]*recip_drF[k]
                *recip_rAs[i, j]
                *(fVerVkp[i, j] - fVerVkm[i, j])*rkSign
                *recip_deepFac2C[k]*recip_rhoFacC[k])

#-- No-slip and drag BCs appear as body forces in cell abutting topography
        if p.no_slip_sides:                                         # :578-591
#-     No-slip BCs impose a drag at walls...
            vF = mom_v_sidedrag(k, vFld, del2v, h0FacZ, viscAh_Z, viscA4_Z,
                                p.useHarmonicVisc, p.useBiharmonicVisc, p.useVariableVisc,
                                vF, cfg=cfg, grid=grid, params=p)
            gvDiss = gvDiss.at[i, j].set(gvDiss[i, j]+vF[i, j])

#-    No-slip BCs impose a drag at bottom
        if bottomDragTerms:                                         # :594-625
            KE, cDrag = mom_v_botdrag_coeff(k, True, uFld, vFld, kappaRV, KE, cDrag, myIter,
                                            cfg=cfg, grid=grid, params=p, ctrlf=ctrlf)
            vF = vF.at[i, j].set(-cDrag[i, j]*vFld[i, j]
                                 *recip_hFacS[i, j, k]*recip_drF[k])
            gvDiss = gvDiss.at[i, j].set(gvDiss[i, j] + vF[i, j])
            if p.useDiagnostics:                                    # :612-619
                raise NotImplementedError("MOM_VECINV: botDragV accumulation (useDiagnostics) is not ported")

#--   if (momViscosity) end of block.

#---  Prepare for Advection & Coriolis terms:
#-    calculate absolute vorticity
    if p.useAbsVorticity:                                           # :669-670
        omega3 = mom_calc_absvort3(k, vort3, omega3, cfg=cfg, grid=grid, params=p)

    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
#--   Horizontal Coriolis terms
    if (p.useCoriolis and                                           # :680-735
            not (p.useCDscheme or p.useAbsVorticity and p.momAdvection)):
        if p.useAbsVorticity:
            uCf = mom_vi_u_coriolis(k, p.selectVortScheme, p.useJamartMomAdv,
                                    vFld, omega3, hFacZ, r_hFacZ, uCf, cfg=cfg, grid=grid)
            vCf = mom_vi_v_coriolis(k, p.selectVortScheme, p.useJamartMomAdv,
                                    uFld, omega3, hFacZ, r_hFacZ, vCf, cfg=cfg, grid=grid)
        elif cfg.cpp.flag("ALLOW_GGL90", _COPT) and cfg.cpp.flag("ALLOW_GGL90_LANGMUIR", "GGL90_OPTIONS.h") \
                and p.useLANGMUIR:                                  # :690-697
            raise NotImplementedError("MOM_VECINV: useLANGMUIR (GGL90_ADD_STOKESDRIFT) is not ported")
        else:
            uCf, vCf = mom_vi_coriolis(k, uFld, vFld, hFacZ, r_hFacZ, uCf, vCf, cfg=cfg, grid=grid, params=p)
        gU = gU.at[i, j, k].set(uCf[i, j])                          # :702-707
        gV = gV.at[i, j, k].set(vCf[i, j])
        if writeDiag:                                               # :708-721
            raise NotImplementedError("MOM_VECINV: WRITE_LOCAL_RL / MNC snapshots are not ported")
        if diagnostics:                                             # :722-727
            raise NotImplementedError("MOM_VECINV: DIAGNOSTICS_FILL (pkg/diagnostics is not ported)")
    else:
        gU = gU.at[i, j, k].set(0.)                                 # :729-734
        gV = gV.at[i, j, k].set(0.)

    if p.momAdvection:                                              # :742-875
#--   Horizontal advection of relative (or absolute) vorticity
        if p.highOrderVorticity or p.upwindVorticity:               # :744-754, :769-779
            raise NotImplementedError("MOM_VECINV: highOrderVorticity/upwindVorticity (MOM_VI_U/V_CORIOLIS_C4) "
                                      "is not ported")
        elif p.useAbsVorticity:                                     # :755-758
            uCf = mom_vi_u_coriolis(k, p.selectVortScheme, p.useJamartMomAdv,
                                    vFld, omega3, hFacZ, r_hFacZ, uCf, cfg=cfg, grid=grid)
        else:                                                       # :759-763
            uCf = mom_vi_u_coriolis(k, p.selectVortScheme, p.useJamartMomAdv,
                                    vFld, vort3, hFacZ, r_hFacZ, uCf, cfg=cfg, grid=grid)
        gU = gU.at[i, j, k].set(gU[i, j, k]+uCf[i, j])              # :764-768
        if p.useAbsVorticity:                                       # :780-783
            vCf = mom_vi_v_coriolis(k, p.selectVortScheme, p.useJamartMomAdv,
                                    uFld, omega3, hFacZ, r_hFacZ, vCf, cfg=cfg, grid=grid)
        else:                                                       # :784-788
            vCf = mom_vi_v_coriolis(k, p.selectVortScheme, p.useJamartMomAdv,
                                    uFld, vort3, hFacZ, r_hFacZ, vCf, cfg=cfg, grid=grid)
        gV = gV.at[i, j, k].set(gV[i, j, k]+vCf[i, j])              # :789-793

        if writeDiag:                                               # :800-813
            raise NotImplementedError("MOM_VECINV: WRITE_LOCAL_RL / MNC snapshots are not ported")
        if diagnostics:                                             # :815-820
            raise NotImplementedError("MOM_VECINV: DIAGNOSTICS_FILL (pkg/diagnostics is not ported)")

#--   Vertical shear terms (-w*du/dr & -w*dv/dr)
        if not p.momImplVertAdv:                                    # :823-844
            uCf = mom_vi_u_vertshear(k, visc.deepFacAdv, uVel, wVel, uCf, cfg=cfg, grid=grid, params=p)
            gU = gU.at[i, j, k].set(gU[i, j, k]+uCf[i, j])
            vCf = mom_vi_v_vertshear(k, visc.deepFacAdv, vVel, wVel, vCf, cfg=cfg, grid=grid, params=p)
            gV = gV.at[i, j, k].set(gV[i, j, k]+vCf[i, j])
            if diagnostics:                                         # :838-843
                raise NotImplementedError("MOM_VECINV: DIAGNOSTICS_FILL (pkg/diagnostics is not ported)")

#--   Bernoulli term
        uCf = mom_vi_u_grad_ke(k, KE, uCf, cfg=cfg, grid=grid)      # :847-852
        gU = gU.at[i, j, k].set(gU[i, j, k]+uCf[i, j])
        vCf = mom_vi_v_grad_ke(k, KE, vCf, cfg=cfg, grid=grid)      # :853-858
        gV = gV.at[i, j, k].set(gV[i, j, k]+vCf[i, j])
        if writeDiag:                                               # :859-872
            raise NotImplementedError("MOM_VECINV: WRITE_LOCAL_RL / MNC snapshots are not ported")

#--   3.D Coriolis term (horizontal momentum, Eastward component: -fprime*w)
    if p.select3dCoriScheme >= 1:                                   # :878-894
        uCf = mom_u_coriolis_nh(k, wVel, uCf, cfg=cfg, grid=grid, params=p)
        gU = gU.at[i, j, k].set(gU[i, j, k]+uCf[i, j])
        if p.usingCurvilinearGrid or p.rotateGrid:
#-     presently, non zero angleSinC array only supported with Curvilinear-Grid
            vCf = mom_v_coriolis_nh(k, wVel, vCf, cfg=cfg, grid=grid, params=p)
            gV = gV.at[i, j, k].set(gV[i, j, k]+vCf[i, j])

#--   Non-Hydrostatic (spherical) metric terms
    if p.useNHMTerms and not p.deepAtmosphere:                      # :897-916 (#else of MOM_USE_OLD_DEEP_VERT_ADV)
        uCf = mom_u_metric_nh(k, uFld, wVel, uCf, cfg=cfg, grid=grid, params=p)
        vCf = mom_v_metric_nh(k, vFld, wVel, vCf, cfg=cfg, grid=grid, params=p)
        gU = gU.at[i, j, k].set(gU[i, j, k] + uCf[i, j])
        gV = gV.at[i, j, k].set(gV[i, j, k] + vCf[i, j])
        if diagnostics:                                             # :910-915
            raise NotImplementedError("MOM_VECINV: DIAGNOSTICS_FILL (pkg/diagnostics is not ported)")

#--   Set du/dt & dv/dt on boundaries to zero
    gU = gU.at[i, j, k].set(gU[i, j, k]*maskW[i, j, k])             # :919-924
    gV = gV.at[i, j, k].set(gV[i, j, k]*maskS[i, j, k])

    if cfg.cpp.flag("ALLOW_DEBUG", _OPT) and p.debugLevel >= debLevC and k == 4 \
            and p.useCubedSphereExchange:                           # :926-934
        raise NotImplementedError("MOM_VECINV: DEBUG_CS_CORNER_UV (debugLevel >= debLevC) is not ported")
    if writeDiag:                                                   # :936-978
        raise NotImplementedError("MOM_VECINV: WRITE_LOCAL_RL / MNC snapshots are not ported")
    if diagnostics:                                                 # :980-1001
        raise NotImplementedError("MOM_VECINV: DIAGNOSTICS_FILL (pkg/diagnostics is not ported)")

    return fVerUkp, fVerVkp, guDiss, gvDiss, state.replace(gU=gU, gV=gV)

