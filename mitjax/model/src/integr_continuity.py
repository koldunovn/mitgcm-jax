"""INTEGR_CONTINUITY: model/src/integr_continuity.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.eesupp.exch_rs import EXCH_XY_RL, EXCH_XYZ_RL
from mitjax.farray import FArray, loop_i, loop_j
from mitjax.model.src.integrate_for_w import integrate_for_w
from mitjax.model.src.update_etah import update_etah


def integr_continuity(uFld, vFld, myTime, myIter, *, cfg, grid, params, state, ex, ff=None):
    """INTEGR_CONTINUITY( uFld, vFld, myTime, myIter, myThid )   @63cdc0b model/src/integr_continuity.F:13-344

    C     | SUBROUTINE INTEGR_CONTINUITY
    C     | o Integrate the continuity Eq : compute vertical velocity and free surface "r-anomaly" (etaN,etaH)
    C     uFld :: Zonal velocity ( m/s )
    C     vFld :: Meridional velocity ( m/s )

    Returns the State (wVel; etaN, etaH, dEtaHdt, PmEpR, etaHnm1 under exactConserv). The level loop DO k=Nr,1,-1
    (:267-303) is a recursion (level k reads wVel(k+1)): an unrolled Python loop in that order. :321-322: the exchange
    of wVel `IF ( implicitIntGravWave .OR. myIter.EQ.nIter0 )` is a select between the exchanged and the
    unexchanged field (myIter is traced). GO lane: the r* column-thickness tendency rStarDhDt (:238-262) feeds
    INTEGRATE_FOR_W. Raise: the sigma arm (UPDATE_ETAWS), usingPCoords with useRealFreshWaterFlux (:276-286),
    selectAddFluid >= 1, ALLOW_OBCS. Lane B: an ALLOW_ADDFLUID build with selectAddFluid = 0 (cs32x15) runs facMass =
    0. (:94-95), so :129-131 subtracts 0.*addMass = +0 (addMass = 0. from INI_FFIELDS; no writer with
    selectAddFluid = 0): x - 0. = x exactly, not written out.

    exactConserv (tracer lane, plan Task 13: `_exact_conserv` below, :90-235 and :316-318) needs `ff` (FFIELDS.h
    EmPmR) when myIter is traced; INITIALISE_VARIA's call at nIter0 (concrete myIter) may omit it."""
    # ADVECT lane (plan Task 14): a NONLIN_FRSURF build with select_rStar = 0 and the linear free surface
    # (advect_xz/input, input.pqm) runs none of the NONLIN_FRSURF lines (:245 select_rStar.NE.0, :336 nonlinFreeSurf
    # .GT.0 with selectSigmaCoord): only that case is let through
    # GO lane (plan Task 15a): the r* lines (:238-262, select_rStar /= 0) are ported; the sigma arm (:334-341,
    # UPDATE_ETAWS) still raises
    if cfg.cpp.NONLIN_FRSURF and not cfg.cpp.DISABLE_SIGMA_CODE and params.nonlinFreeSurf > 0 \
            and params.selectSigmaCoord != 0:
        raise NotImplementedError("INTEGR_CONTINUITY: UPDATE_ETAWS (selectSigmaCoord, :334-341) is not ported")
    if cfg.cpp.ALLOW_OBCS:
        raise NotImplementedError("INTEGR_CONTINUITY: the ALLOW_OBCS lines are not ported")
    # lane B (Task 25, cs32x15): the routine has no ALLOW_NONHYDROSTATIC lines; ALLOW_ADDFLUID's are compiled in but
    # act only with selectAddFluid >= 1 (facMass, :95; INTEGRATE_FOR_W :80-89), refused here
    if cfg.cpp.ALLOW_ADDFLUID and params.selectAddFluid >= 1:
        raise NotImplementedError("INTEGR_CONTINUITY: selectAddFluid >= 1 (addMass, :95, :129-131) is not ported")
    sz = cfg.size
    wVel = state.wVel
    addMass = None                                                              # :71  addMass(1): ALLOW_ADDFLUID off
    if params.exactConserv:                                                     # :90-235 (R2 arm, tracer lane)
        state = _exact_conserv(uFld, vFld, myIter, cfg=cfg, grid=grid, params=params, state=state, ff=ff)
    rStarDhDt = None                                                            # :80  rStarDhDt(1): NONLIN_FRSURF off
    if cfg.cpp.NONLIN_FRSURF and not cfg.cpp.DISABLE_RSTAR_CODE and params.select_rStar != 0:   # :238-262 (GO lane)
        g = grid
        j = loop_j(1, sz.sNy)                                                   # :248
        i = loop_i(1, sz.sNx)                                                   # :249
        ks = g.kSurfC[i, j]                                                     # :250
        deepFac2F_ks = g.deepFac2F.data[ks - 1]                                 # deepFac2F(ks): gather
        rhoFacF_ks = params.rhoFacF.data[ks - 1]                                # rhoFacF(ks): gather
        rStarDhDt = state.etaN.local("rStarDhDt").at[i, j].set(state.dEtaHdt[i, j]   # :251-253
                                                                * deepFac2F_ks*rhoFacF_ks
                                                                * g.recip_Rcol[i, j])
    # :267  DO k=Nr,1,-1 as a level scan (KERNEL_GUIDE §4): k = Nr and k = 1 static (INTEGRATE_FOR_W's branches)
    def level_k(k, wVel):
        wVel = integrate_for_w(k, uFld, vFld, addMass, rStarDhDt, wVel, myIter,   # :270-274
                               cfg=cfg, grid=grid, params=params)
        if k == sz.Nr and params.usingPCoords and params.fluidIsWater and params.useRealFreshWaterFlux:
            raise NotImplementedError("INTEGR_CONTINUITY: :276-286 (p coordinates, real fresh water) is not ported")
        return wVel
    from mitjax.ops.scan_k import scan_levels
    wVel = scan_levels(level_k, wVel, 1, sz.Nr, down=True, peel=(1, 1))
    if params.implicitIntGravWave:                                              # :321-322
        wVel = EXCH_XYZ_RL(wVel, ex=ex)
    else:
        wEx = EXCH_XYZ_RL(wVel, ex=ex)
        wVel = FArray(jnp.where(myIter == params.nIter0, wEx.data, wVel.data), wVel.name, _dims=wVel.dims)
    state = state.replace(wVel=wVel)
    if params.exactConserv and params.implicDiv2DFlow_ne_0:                     # :316-318 (R2 arm, tracer lane)
        eEx = EXCH_XY_RL(state.etaN, ex=ex)
        state = state.replace(etaN=FArray(jnp.where(myIter != params.nIter0, eEx.data, state.etaN.data),
                                          state.etaN.name, _dims=state.etaN.dims))
    if params.exactConserv:                                                     # :325-332
        state = update_etah(myTime, myIter, cfg=cfg, params=params, state=state, ex=ex)
    return state


# ---- R2 arm (tracer lane, plan Task 13) ----------------------------------------------------------------------------

def _exact_conserv(uFld, vFld, myIter, *, cfg, grid, params, state, ff):
    """integr_continuity.F:90-235 @63cdc0b (exactConserv): the vertically integrated divergence hDivFlow, PmEpR and
    dEtaHdt (:90-203), and etaN at the end of the step (:207-235). Returns the State (PmEpR, dEtaHdt, etaN).

    The k loop :108-136 accumulates hDivFlow level by level (a sum in k): a level scan in the Fortran order
    (KERNEL_GUIDE §4). The run-time tests on myIter (traced int32) select between values computed on both sides
    (`jnp.where`, finite on both): :141-162 (myIter.EQ.nIter0 .AND. myIter.NE.0 with real fresh water, GO lane),
    :163-188 (myIter.EQ.nIter0) and :207 (myIter.NE.nIter0). `implicDiv2Dflow.EQ.0`
    (:211) selects the update range: decided on the host (params.implicDiv2DFlow_ne_0, ini_parms_tracer).
    recip_deepFac2F(ks) with ks = kSurfC(i,j): a gather (KERNEL_GUIDE, Fortran index -> storage). Raise:
    ALLOW_ADDFLUID (facMass), ALLOW_OBCS, usePickupBeforeC54. `0. _d 0` / `0.` exact."""
    realFW = params.fluidIsWater and params.useRealFreshWaterFlux               # :93, :141-142 (GO lane)
    if realFW and params.usePickupBeforeC54:
        raise NotImplementedError("INTEGR_CONTINUITY: usePickupBeforeC54 (:146-153) is not ported")
    sz = cfg.size
    g = grid
    facEmP = 0.                                                                 # :92  0. (REAL*4, exact)
    if realFW:
        facEmP = params.mass2rUnit                                              # :93
    # :94-95 facMass = 0. (selectAddFluid < 1; >= 1 is raised by the caller): :129-131 adds nothing (docstring)
    jA = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    iA = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    hDivFlow = state.etaN.local("hDivFlow").at[iA, jA].set(0.)                  # :100-106
    uTrans = state.etaN.local("uTrans").at[iA, jA].set(0.)
    vTrans = state.etaN.local("vTrans").at[iA, jA].set(0.)
    j1 = loop_j(1, sz.sNy+1)
    i1 = loop_i(1, sz.sNx+1)
    j = loop_j(1, sz.sNy)
    i = loop_i(1, sz.sNx)
    # :108  DO k=1,Nr as a level scan (KERNEL_GUIDE §4; no level branches)
    def level_k(k, c):
        uTrans, vTrans, hDivFlow = c
        uTrans = uTrans.at[i1, j1].set(uFld[i1, j1, k]*g.dyG[i1, j1]            # :114-116
                                       * g.deepFacC[k]*params.rhoFacC[k]
                                       * g.drF[k]*g.hFacW[i1, j1, k])
        vTrans = vTrans.at[i1, j1].set(vFld[i1, j1, k]*g.dxG[i1, j1]            # :117-119
                                       * g.deepFacC[k]*params.rhoFacC[k]
                                       * g.drF[k]*g.hFacS[i1, j1, k])
        hDivFlow = hDivFlow.at[i, j].set(hDivFlow[i, j]                          # :126-128
                                         + g.maskC[i, j, k]*(uTrans[i+1, j]-uTrans[i, j]
                                                             + vTrans[i, j+1]-vTrans[i, j]))
        return uTrans, vTrans, hDivFlow
    from mitjax.ops.scan_k import scan_levels
    uTrans, vTrans, hDivFlow = scan_levels(level_k, (uTrans, vTrans, hDivFlow), 1, sz.Nr)
    ks = g.kSurfC[i, j]                                                         # :166, :182
    recip_deepFac2F_ks = g.recip_deepFac2F.data[ks - 1]                         # recip_deepFac2F(ks): gather
    # :163-171  myIter.EQ.nIter0
    PmEpR_0 = state.PmEpR.at[i, j].set(0.)                                      # :167  0. _d 0
    dEtaHdt_0 = state.dEtaHdt.at[i, j].set(-hDivFlow[i, j]*g.recip_rA[i, j]     # :168-169
                                           * recip_deepFac2F_ks)
    at_start = myIter == params.nIter0
    if realFW:                                                                  # :141-162 (GO lane)
        # myIter.EQ.nIter0 .AND. myIter.NE.0 (.AND. fluidIsWater .AND. useRealFreshWaterFlux): PmEpR from the
        # pickup's dEtaHdt; dEtaHdt itself is kept (usePickupBeforeC54 raises above)
        PmEpR_r = state.PmEpR.at[i, j].set(state.dEtaHdt[i, j]                  # :156-158
                                           + hDivFlow[i, j]*g.recip_rA[i, j]
                                           * g.recip_deepFac2F[1])
        PmEpR_r = PmEpR_r.at[i, j].set(PmEpR_r[i, j]*params.rUnit2mass)         # :159
        first_rfw = jnp.logical_and(at_start, myIter != 0)
        PmEpR_0 = FArray(jnp.where(first_rfw, PmEpR_r.data, PmEpR_0.data), state.PmEpR.name,
                         _dims=state.PmEpR.dims)
        dEtaHdt_0 = FArray(jnp.where(first_rfw, state.dEtaHdt.data, dEtaHdt_0.data), state.dEtaHdt.name,
                           _dims=state.dEtaHdt.dims)
    if ff is None:
        if isinstance(at_start, jnp.ndarray) and not bool(at_start):
            raise ValueError("INTEGR_CONTINUITY: exactConserv after nIter0 needs ff (FFIELDS.h EmPmR)")
        PmEpR, dEtaHdt = PmEpR_0, dEtaHdt_0
    else:
        # :172-187  ELSE
        PmEpR_1 = state.PmEpR.at[iA, jA].set(-ff.EmPmR[iA, jA])                 # :175-179
        dEtaHdt_1 = state.dEtaHdt.at[i, j].set(-hDivFlow[i, j]*g.recip_rA[i, j]  # :183-185
                                               * recip_deepFac2F_ks
                                               - facEmP*ff.EmPmR[i, j])
        PmEpR = FArray(jnp.where(at_start, PmEpR_0.data, PmEpR_1.data), state.PmEpR.name,
                       _dims=state.PmEpR.dims)
        dEtaHdt = FArray(jnp.where(at_start, dEtaHdt_0.data, dEtaHdt_1.data), state.dEtaHdt.name,
                         _dims=state.dEtaHdt.dims)
    # :207-235  exactConserv .AND. myIter.NE.nIter0: etaN at the end of the step
    if not params.implicDiv2DFlow_ne_0:                                         # :211-216
        etaN_1 = state.etaN.at[iA, jA].set(state.etaH[iA, jA])
    else:                                                                       # :217-224
        etaN_1 = state.etaN.at[i, j].set(state.etaH[i, j]
                                         + params.implicDiv2DFlow*dEtaHdt[i, j]*params.deltaTFreeSurf)
    etaN = FArray(jnp.where(myIter != params.nIter0, etaN_1.data, state.etaN.data), state.etaN.name,
                  _dims=state.etaN.dims)
    return state.replace(PmEpR=PmEpR, dEtaHdt=dEtaHdt, etaN=etaN)
