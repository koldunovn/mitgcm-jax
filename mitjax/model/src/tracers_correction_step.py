"""TRACERS_CORRECTION_STEP: model/src/tracers_correction_step.F @63cdc0b."""


def tracers_correction_step(myTime, myIter, *, cfg, params, state, grid=None, eos=None, opps=None, probe=None):
    """TRACERS_CORRECTION_STEP( myTime, myIter, myThid )   @63cdc0b model/src/tracers_correction_step.F:7-133

    C     | SUBROUTINE TRACERS_CORRECTION_STEP
    C     | o Called at the end of the time step: tracer convective adjustment and filters

    Returns the State. Filters / OPPS / pTracers blocks are compiled out in the ported builds (raise when compiled).
    PTRACERS lane: :114-119 CONVECTIVE_ADJUSTMENT when .NOT.useOPPS .AND. cAdjFreq.NE.0. (cAdjFreq as INI_PARMS left
    it, a static host value; INCLUDE_CONVECT_CALL), on theta, salt and, with usePTRACERS, the passive tracers (the
    State's PTRACERS_FIELDS.h fields; `grid`, `eos` then required).

    vermix lane (M3 Task 30): :102-113 OPPS_INTERFACE (useOPPS; `opps`: OPPS_PARAMS of OPPS_READPARMS) on iMin = 1,
    iMax = sNx, jMin = 1, jMax = sNy, probe T05_opps; the CONVECTIVE_ADJUSTMENT test is .NOT.useOPPS .AND. cAdjFreq
    .NE. 0. (:115), probe T06_convective_adjustment. A compiled-but-off filter package is fine; a used one raises.
    Returns (state, opps_counters): OPPS_INTERFACE's `counters` ({"ntimeOver"}) for the host check
    opps_calc.opps_calc_host (the static time-loop bound, COLMIX s2), None without useOPPS."""
    pr = probe if probe is not None else (lambda stage, values: None)
    for opt, use in (("ALLOW_SHAP_FILT", "useSHAP_FILT"), ("ALLOW_ZONAL_FILT", "useZONAL_FILT")):
        if getattr(cfg.cpp, opt, False) and cfg.use_flag(use):
            raise NotImplementedError(f"TRACERS_CORRECTION_STEP: {use} is not ported")
    useOPPS = bool(cfg.cpp.ALLOW_OPPS and cfg.use_flag("useOPPS"))
    opps_counters = None
    if useOPPS:                                                                 # :102-113
        from mitjax.pkg.opps.opps_interface import opps_interface
        sz = cfg.size
        theta, salt, _, opps_counters = opps_interface(1, sz.sNx, 1, sz.sNy, myTime, myIter, cfg=cfg, grid=grid,
                                                       params=params, eos=eos, state=state, op=opps)
        state = state.replace(theta=theta, salt=salt)
        pr("T05_opps", state)
    if cfg.cpp.INCLUDE_CONVECT_CALL and not useOPPS and params.cAdjFreq != 0.:  # :114-119 (PTRACERS lane)
        from mitjax.model.src.convective_adjustment import convective_adjustment
        from mitjax.pkg.ptracers.ptracers_fields_h import ptf_of_state, state_with_ptf
        ptr_on = bool(cfg.cpp.ALLOW_PTRACERS) and params.usePTRACERS
        ptf = ptf_of_state(state, params) if ptr_on else None
        state, ptf = convective_adjustment(myTime, myIter, cfg=cfg, grid=grid, params=params, eos=eos, state=state,
                                           ptr=params if ptr_on else None, ptf=ptf)        # :116-117
        if ptr_on:
            state = state_with_ptf(state, ptf)
        pr("T06_convective_adjustment", state)
    return state, opps_counters
