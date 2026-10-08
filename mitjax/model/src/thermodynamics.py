"""THERMODYNAMICS: model/src/thermodynamics.F @63cdc0b (R1 part: core lane; tracer stepping, R2 arm: tracer lane)."""

import jax.numpy as jnp
from mitjax.ops.safe import safe_div
from mitjax.farray import loops_kji


def thermodynamics(myTime, myIter, *, cfg, params, state, grid=None, fp=None, ff=None, probe=None, gm=None,
                   until=None, mix=None, rbcs=None):
    """THERMODYNAMICS( myTime, myIter, myThid )   @63cdc0b model/src/thermodynamics.F:28-419

    C     | SUBROUTINE THERMODYNAMICS
    C     | o Controlling routine for the prognostic part of the thermo-dynamics.

    Returns (state, flow): `flow` = (uFld, vFld, wFld), the local copies of uVel, vVel, wVel (:260-268) that
    MON_CALC_ADVCFL_TILE / _GLOB (:278-286, :387-390, when monOutputCFL, :131-137) read: the monitor is host-side
    (mitjax/pkg/monitor/monitor.py), so the driver applies those two calls to the returned flow on the host at the
    steps where DIFFERENT_MULTIPLE(monitorFreq, wrTime, deltaTClock) holds. Raise: linFSConserveTr (CALC_WSURF_TR);
    debugLevel >= debLevD. R5 arm (plan Task 16): with useGMRedi, GMREDI_RESIDUAL_FLOW on the flow copies
    (:269-275) and the GMREDI.h state `gm` handed to TEMP_INTEGRATE and SALT_INTEGRATE. `until` (gates only,
    forward_step): a stage of TEMP_INTEGRATE at which the caller stops; SALT_INTEGRATE is then skipped.

    R2 arm (tracer lane, plan Task 13; below the marked line): with tempStepping, recip_hFacNew (:194-256: the
    non-NONLIN_FRSURF branch :247-253, `_recip_hFacC` = recip_hFacC) and kappaRk = 0 (:194-202) are formed and
    TEMP_INTEGRATE runs (:317-326); `grid`, `fp`, `ff` (GRID.h, ForcingParams, FFIELDS.h of APPLY_FORCING_T) are
    then required. `probe`: optional, called with the T-stage values (T11-T13 inside TEMP_INTEGRATE,
    T02_temp_integrate after it). ADVECT lane (plan Task 14): with saltStepping SALT_INTEGRATE runs after it
    (:328-337, same recip_hFacNew, flow and kappaRk; probe stages T20-T23 inside it, T03_salt_integrate after it);
    the tracer part runs when tempStepping or saltStepping. The k/j/i nests are copies, independent point by point (k-vectorised). Raise:
    NONLIN_FRSURF with nonlinFreeSurf > 0 in the sigma arm (GO lane: :207-218 recip_hFacC/rStarExpC is ported;
    GOADK lane: the hFac_surfC arm :233-244, 1/hFac_surfC at k = kSurfC), PTRACERS_calcSurfCor, OBCS_APPLY_TS, ALLOW_FRICTION_HEATING. PTRACERS lane: with
    usePTRACERS, PTRACERS_INTEGRATE (:339-354; probe stage T04_ptracers_integrate after it). Lane B (Task 25): a
    build without pkg/generic_advdiff (adjustment.cs-32x32x1/code) compiles the whole body (:94-416) out: the State
    is returned as it came, with the flow of uVel, vVel, wVel only to keep the carry's shape (no MON_CALC_ADVCFL: the
    host's CFL event is off, drivers/run.chunk_ends). M3 Task 30: `rbcs` (RBC_mask, RBCtemp of RBCS_FIELDS.h) goes
    to TEMP_INTEGRATE's APPLY_FORCING_T (RBCS_ADD_TENDENCY, useRBCS)."""
    if not cfg.cpp.ALLOW_GENERIC_ADVDIFF:                                       # :94-416 (lane B)
        return state, (state.uVel, state.vVel, state.wVel)
    if params.linFSConserveTr:
        raise NotImplementedError("THERMODYNAMICS: linFSConserveTr (CALC_WSURF_TR) is not ported")
    if cfg.cpp.ALLOW_GMREDI and params.useGMRedi and gm is None:
        raise NotImplementedError("THERMODYNAMICS: useGMRedi needs the GMREDI.h state `gm`")
    if params.debugLevel >= 4:
        raise NotImplementedError("THERMODYNAMICS: debugLevel >= debLevD output is not ported")
    flow = (state.uVel, state.vVel, state.wVel)                                 # :263-265
    # PTRACERS lane: :339-354 below. DO_PTRACERS_HERE is defined by thermodynamics.F itself (:6-8: ALLOW_PTRACERS
    # without ALLOW_LONGSTEP), not by a header, so cfg.cpp reads it as undefined: the definition is spelt out here
    ptr_on = bool(cfg.cpp.ALLOW_PTRACERS) and not cfg.cpp.ALLOW_LONGSTEP and params.usePTRACERS
    if not (params.tempStepping or params.saltStepping or ptr_on):              # ADVECT lane: salt arm below
        return state, flow

    # ---- R2 arm (tracer lane): TEMP_INTEGRATE (thermodynamics.F:181-384) --------------------------------------
    from mitjax.model.src.temp_integrate import temp_integrate
    pr = probe if probe is not None else (lambda stage, values: None)
    nlfs = cfg.cpp.NONLIN_FRSURF and params.nonlinFreeSurf > 0                 # :205-206 (GO lane)
    if nlfs and not (params.select_rStar > 0) and params.selectSigmaCoord_ne_0 and not cfg.cpp.DISABLE_SIGMA_CODE:
        raise NotImplementedError("THERMODYNAMICS: recip_hFacNew of the sigma arm (:219-232) is not ported")
    if ptr_on and params.PTRACERS_calcSurfCor:                                  # :172-178 (PTRACERS lane)
        raise NotImplementedError("THERMODYNAMICS: PTRACERS_CALC_WSURF_TR (PTRACERS_linFSConserve) is not ported")
    if cfg.cpp.ALLOW_OBCS and params.useOBCS:                                   # :356-361
        raise NotImplementedError("THERMODYNAMICS: OBCS_APPLY_TS is not ported")
    if cfg.cpp.ALLOW_FRICTION_HEATING:                                          # :363-380
        raise NotImplementedError("THERMODYNAMICS: ALLOW_FRICTION_HEATING is not ported")
    sz = cfg.size
    k, j, i = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))   # :194-196
    recip_hFacNew = state.theta.local("recip_hFacNew").at[i, j, k].set(0.)     # :197  0. _d 0
    kappaRk = state.theta.local("kappaRk").at[i, j, k].set(0.)                  # :199  0. _d 0
    if nlfs and params.select_rStar > 0:                                       # :207-218 (GO lane, select_rStar > 0)
        if not cfg.cpp.DISABLE_RSTAR_CODE:
            recip_hFacNew = recip_hFacNew.at[i, j, k].set(grid.recip_hFacC[i, j, k]   # :212-213
                                                          / state.rStarExpC[i, j])
    elif nlfs and params.selectSigmaCoord_ne_0:                                 # :219-232 (DISABLE_SIGMA_CODE: empty)
        pass
    elif nlfs:                                                                  # :233-244 (GOADK lane: hFac_surfC)
        kv = jnp.arange(1, sz.Nr + 1).reshape(1, sz.Nr, 1, 1)                   # the level index k of the nest
        at = grid.kSurfC[i, j] == kv                                            # IF ( k.EQ.kSurfC(i,j,bi,bj) )
        recip_hFacNew = recip_hFacNew.at[i, j, k].set(
            jnp.where(at, safe_div(1., state.hFac_surfC[i, j], at), grid.recip_hFacC[i, j, k]))   # 1. _d 0 / ...
    else:
        recip_hFacNew = recip_hFacNew.at[i, j, k].set(grid.recip_hFacC[i, j, k])    # :247-253
    uFld = state.uVel.local("uFld").at[i, j, k].set(state.uVel[i, j, k])        # :260-268
    vFld = state.vVel.local("vFld").at[i, j, k].set(state.vVel[i, j, k])
    wFld = state.wVel.local("wFld").at[i, j, k].set(state.wVel[i, j, k])
    if cfg.cpp.ALLOW_GMREDI and params.useGMRedi:                              # :269-275 (R5 arm)
        from mitjax.pkg.gmredi.gmredi_residual_flow import gmredi_residual_flow
        uFld, vFld, wFld = gmredi_residual_flow(uFld, vFld, wFld, myIter, cfg=cfg, gm=gm, grid=grid)
    flow = (uFld, vFld, wFld)
    pr("T01_residual_flow", {"uFld": uFld, "vFld": vFld, "wFld": wFld})
    if params.tempStepping:                                                     # :317-326
        state, kappaRk = temp_integrate(recip_hFacNew, uFld, vFld, wFld, kappaRk, myTime, myIter,   # :321-325
                                        cfg=cfg, grid=grid, params=params, fp=fp, ff=ff, state=state,
                                        probe=probe, gm=gm, mix=mix, rbcs=rbcs)   # vermix lane: KPP.h
    pr("T02_temp_integrate", state)
    if until in _TEMP_STAGES:                                                   # gates only (forward_step until)
        return state, flow
    # ---- ADVECT lane (plan Task 14): SALT_INTEGRATE (thermodynamics.F:328-337)
    if params.saltStepping:                                                     # :328
        from mitjax.model.src.salt_integrate import salt_integrate
        state, kappaRk = salt_integrate(recip_hFacNew, uFld, vFld, wFld, kappaRk, myTime, myIter,   # :332-336
                                        cfg=cfg, grid=grid, params=params, fp=fp, ff=ff, state=state,
                                        probe=probe, gm=gm, mix=mix)      # vermix lane: KPP.h
    pr("T03_salt_integrate", state)
    # ---- PTRACERS lane (plan Task 26): PTRACERS_INTEGRATE (thermodynamics.F:339-354, DO_PTRACERS_HERE = ALLOW_PTRACERS
    # without ALLOW_LONGSTEP, :6-8) with the same recip_hFacNew, flow and kappaRk; the PTRACERS_PARAMS.h values ride in
    # `params` (drivers/model.ptracers_params). :168-171 (ALLOW_AUTODIFF meanSurfCorPTr = 0) is read only with
    # PTRACERS_linFSConserve (raised above and in PTRACERS_APPLY_FORCING)
    if ptr_on:
        from mitjax.pkg.ptracers.ptracers_fields_h import ptf_of_state, state_with_ptf
        from mitjax.pkg.ptracers.ptracers_integrate import ptracers_integrate
        ptf, kappaRk = ptracers_integrate(recip_hFacNew, uFld, vFld, wFld, kappaRk, myTime, myIter,   # :346-350
                                          cfg=cfg, grid=grid, params=params, ptr=params,
                                          ptf=ptf_of_state(state, params), state=state, gm=gm)
        state = state_with_ptf(state, ptf)
        pr("T04_ptracers_integrate", state)
    return state, flow


# the stages up to TEMP_INTEGRATE's end: a gate stopping there (forward_step `until`) does not need SALT_INTEGRATE
_TEMP_STAGES = ("T01_residual_flow", "T11_temp_gT", "T12_temp_step", "T13_temp_impl", "T02_temp_integrate")
