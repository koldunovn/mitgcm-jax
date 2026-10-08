"""FORWARD_STEP: model/src/forward_step.F @63cdc0b, the traced part of one time step (no I/O)."""

from mitjax.model.src.do_atmospheric_phys import do_atmospheric_phys
from mitjax.model.src.do_fields_blocking_exchanges import do_fields_blocking_exchanges
from mitjax.model.src.do_oceanic_phys import do_oceanic_phys
from mitjax.model.src.dynamics import dynamics
from mitjax.model.src.integr_continuity import integr_continuity
from mitjax.model.src.load_fields_driver import load_fields_driver
from mitjax.model.src.momentum_correction_step import momentum_correction_step
from mitjax.model.src.solve_for_pressure import solve_for_pressure
from mitjax.model.src.thermodynamics import thermodynamics
from mitjax.model.src.tracers_correction_step import tracers_correction_step


def forward_step(iloop, myTime, myIter, *, cfg, grid, params, fp, eos, cg2dh, cg2d_params, state, ff, phi0surf,
                 ex, probe=None, pk=None, pkc=None, pks=None, until=None):
    """FORWARD_STEP( iloop, myTime, myIter, myThid )   @63cdc0b model/src/forward_step.F:70-1222

    C     | SUBROUTINE FORWARD_STEP
    C     | o Run the ocean model and, optionally, evaluate a cost function.

    Returns (state, ff, phi0surf, myTime, myIter, out): `out` holds what the host-side parts of the step need
    (`flow`: THERMODYNAMICS' flow for MON_CALC_ADVCFL, `cg2d`: SOLVE_FOR_PRESSURE's solver scalars) and, with
    packages, `pk` (below). The host-side calls of forward_step.F are made by the driver
    (mitjax/drivers/the_main_loop.py) on concrete values: :1151-1156 MONITOR, :1182 DO_THE_MODEL_IO, :1196
    DO_WRITE_PICKUP, the printed lines. `probe(stage, values)`: optional, called with the values of the substep dump
    stages (reference/jaxdump/SUBSTEPS.md) for the gates.

    Order (forward_step.F, live lines of a plain forward build): :540 LOAD_FIELDS_DRIVER (FORCING lane; periodic
    forcing only through the host preload `pkc["forcing"]`), :627 DO_ATMOSPHERIC_PHYS, :657 DO_OCEANIC_PHYS,
    :725-736 THERMODYNAMICS (.NOT.staggerTimeStep), :766-770 DO_STAGGER_FIELDS_EXCHANGES (implicitIntGravWave:
    raises), :786-795 DYNAMICS (momStepping), :807-808 the counters, :897-906 SOLVE_FOR_PRESSURE, :911-917
    MOMENTUM_CORRECTION_STEP, :924-930 INTEGR_CONTINUITY (calc_wVelocity), :976-1009 the staggered tracer step
    (DO_STAGGER_FIELDS_EXCHANGES, THERMODYNAMICS; GO lane), :1025 TRACERS_CORRECTION_STEP, :1093
    DO_FIELDS_BLOCKING_EXCHANGES. NONLIN_FRSURF (GO
    lane): `nlfs_reset` :461-494, `nlfs_update_hfac` :832-860, `nlfs_update_cg2d` :862-875, `nlfs_calc_r_star`
    :939-961 (below). The packages of :445-523, :544-620, :1030-1150 raise when compiled and switched on (gated per
    variant).
    R5 arm (plan Task 16, tutorial_global_oce_optim/code_ad; ALLOW_AUTODIFF, ALLOW_CTRL, ALLOW_COST, GM/Redi):
    :427-431 myIter = nIter0 + (iloop-1), myTime = startTime + deltaTClock*(iLoop-1) at the start of the step;
    :433-435 / :1207-1209 AUTODIFF_INADMODE_UNSET / _SET (inAdMode stays .FALSE. in a forward run: nothing to
    carry); CTRL_MAP_GENTIM2D inside LOAD_FIELDS_DRIVER and :569-575 CTRL_MAP_FORCING (useCTRL); the GMREDI.h state
    through DO_OCEANIC_PHYS and THERMODYNAMICS; :1159-1164 COST_TILE after the (host-side) MONITOR. Package state
    `pk` (carried from step to step, returned as out["pk"] when not empty): "gm" (GMREDI.h), "genarr"
    (CTRL_GENARR.h), "cost" (cost.h); package inputs `pkc` (a jit argument, traced): "forcing" (ForcingPreload),
    "effective" (the control records of CTRL_MAP_INI_GENTIM2D: depends on the control vector), "lastinterval";
    static package settings `pks` (closed over): "gentim2d", "clock", "endTime", "lastinterval".
    `until` (gates only, static): the name of a dump stage after which the step returns at once (state, ff,
    phi0surf, myTime, myIter, out) -- a stage gate can run the front of a step whose later routines are not ported
    yet (R5: the CD scheme and INTEGR_CONTINUITY's real-fresh-water branch, GO lane)."""
    pk = dict(pk or {})
    pkc = dict(pkc or {})
    pks = dict(pks or {})
    if fp.periodicExternalForcing and "forcing" not in pkc:
        raise NotImplementedError("FORWARD_STEP: periodic external forcing needs the host preload pkc['forcing'] "
                                  "(external_fields_load.preload_periodic_forcing)")
    if params.implicitIntGravWave:
        raise NotImplementedError("FORWARD_STEP: implicitIntGravWave is not wired yet")
    for opt in ("ALLOW_SHAP_FILT", "ALLOW_ZONAL_FILT"):
        if getattr(cfg.cpp, opt):
            raise NotImplementedError(f"FORWARD_STEP: the {opt} blocks are not wired yet")
    stop = []
    pr0 = probe if probe is not None else (lambda stage, values: None)

    def pr(stage, values):
        pr0(stage, values)
        if stage == until:
            stop.append(stage)

    exf_mon = None                              # lane M4COL: EXF_FIELDS.h at EXF_MONITOR (set below with useEXF)

    def early():
        o = dict(flow=None, cg2d=None, **({"pk": pk} if pk else {}))
        if exf_mon is not None:
            o["exf_mon"] = {n: v.data for n, v in exf_mon.items() if not n[-1:] in ("0", "1")}
        return state, ff, phi0surf, myTime, myIter, o

    if cfg.cpp.ALLOW_AUTODIFF:                                                                     # :427-431
        myIter = params.nIter0 + (iloop - 1)
        myTime = params.startTime + params.deltaTClock*(iloop - 1).astype(myTime.dtype)
    # GO lane (NONLIN_FRSURF): the step's GRID.h thickness factors and CG2D.h operator are the State's (state.py)
    grid, cg2dh = nlfs_load(cfg=cfg, grid=grid, cg2dh=cg2dh, state=state)
    pr("S00_begin", state)
    state, grid = nlfs_reset(myTime, myIter, cfg=cfg, grid=grid, params=params, state=state)       # :461-494
    pr("S01_update_rstar_F", state)
    ctrl = None
    useCTRL = bool(cfg.cpp.ALLOW_CTRL) and _use(cfg, "useCTRL")
    if useCTRL:
        ctrl = dict(genarr=pk["genarr"], effective=pkc["effective"], gentim2d=pks["gentim2d"],
                    maskC=grid.maskC, clock=pks["clock"])
    rbcs = None
    if pk and "rbcs" in pk:                     # M3 Task 30: RBCS_FIELDS_LOAD in LOAD_FIELDS_DRIVER (useRBCS)
        rbcs = dict(state=pk["rbcs"], pre=pkc["rbcs"]["pre"], params=params)
    exfd = None
    if "exf" in pk:                             # lane M4COL: EXF_GETFORCING in LOAD_FIELDS_DRIVER (useEXF)
        exf_mon = {}
        exs = pks["exfs"]

        def exf_probe(stage, f_, ff_):
            pr(stage, {**f_, **{n: getattr(ff_, n) for n in ff_.names()}})
        exfd = dict(f=pk["exf"], pre=pkc["exf_pre"], exfp=pkc["exfp"], exf_fp=pkc["exf_fp"], cal=exs["cal"],
                    tp=exs["tp"], grid=grid, params=params, probe=exf_probe, mon=exf_mon)
    ff = load_fields_driver(myTime, myIter, ff, cfg=cfg, fp=fp, state=state, rw=None, ex=ex,         # :540
                            iloop=iloop, pre=pkc.get("forcing"), ctrl=ctrl, rbcs=rbcs, exf=exfd)
    if exfd is not None:
        ff, pk["exf"] = ff
    if rbcs is not None:
        ff, pk["rbcs"] = ff
    if ctrl is not None:
        ff, pk["genarr"] = ff
    pr("S02_load_fields", ff)
    if stop:
        return early()
    if useCTRL:                                                                                    # :569-575
        from mitjax.pkg.ctrl.ctrl_map_forcing import FIELDS, ctrl_map_forcing
        g = dict(angleCosC=grid.angleCosC, angleSinC=grid.angleSinC, maskW=grid.maskW, maskS=grid.maskS)
        fl = ctrl_map_forcing(myTime, myIter, cfg=cfg, gentim2d=pks["gentim2d"],
                              xx_gentim2d=pk["genarr"]["xx_gentim2d"],
                              ffields={n: getattr(ff, n) for n in FIELDS if n in ff}, grid=g, ex=ex,
                              usingPCoords=params.usingPCoords)
        ff = ff.replace(**{n: fl[n] for n in fl if n in ff})
        pr("S03_ctrl_map_forcing", ff)
        if stop:
            return early()
    state = do_atmospheric_phys(myTime, myIter, cfg=cfg, params=params, state=state, grid=grid)    # :627 (grid: lane B)
    mix0 = {n: pk[n] for n in ("ggl", "pp81", "my82", "dwnslp") if n in pk}   # vermix lane: GGL90.h, PP81.h, MY82.h
    # (lane M4ADLAB: "dwnslp", the down-slope state DWNSLP_CALC_FLOW updates in DO_OCEANIC_PHYS)
    sid = None
    lsr_out = None
    if "seaice" in pk:                          # lane M4COL: SEAICE_MODEL in DO_OCEANIC_PHYS (useSEAICE)
        sid = dict(sf=pk["seaice"], exf=pk["exf"], sp=pkc["sp"], op=pkc["op"], exfp=pkc["exfp"],
                   kgeo=pks["exfs"]["kgeo"],
                   spp=pkc.get("spp"))          # lane M4LAB session 3: SALT_PLUME.h parameters (SEAICE_GROWTH)
        if cfg.cpp.ALLOW_COST and "cost" in pk:  # lane M4ADCOL: SEAICE_COST_SENSI (do_oceanic_phys.F:463-465)
            sid["cost"] = dict(cost=pk["cost"], endTime=pks["endTime"], lastinterval=pkc["lastinterval"])
    # lane M4OFF: useSEAICE without a mixing package (offline_exf_seaice) takes the same call
    if "kpp" in pk or mix0 or sid is not None:   # PTRACERS lane: KPP.h (KPP_CALC_DUMMY), with GMREDI.h when present
        out_op = do_oceanic_phys(myTime, myIter, cfg=cfg, grid=grid, params=params, fp=fp, eos=eos, state=state,
                                 ff=ff, phi0surf=phi0surf, probe=pr, gm=pk.get("gm"), gm_params=params,
                                 kpp=pk.get("kpp"), kpp_p=pkc.get("kpp_p"), ex=ex,   # :657 (vermix lane: KPP_CALC)
                                 mix=mix0 or None,                                   # vermix lane: GGL90/PP81/MY82
                                 visc=pkc.get("visc"),                               # GM_useLeithQG (MLAdjust)
                                 seaice=sid,                                         # lane M4COL: SEAICE_MODEL
                                 salt_plume=pk.get("salt_plume"),                    # lane M4LAB: KPP_CALC
                                 until=until)
        state, ff, phi0surf = out_op[:3]
        rest = list(out_op[3:])
        if sid is not None:
            si_ret = rest.pop()
            sf_, pk["exf"] = si_ret[:2]
            if len(si_ret) == 3:
                pk["salt_plume"] = si_ret[2]             # lane M4LAB session 3: saltPlumeFlux of SEAICE_GROWTH
            sf_ = dict(sf_)
            lsr_out = sf_.pop("_lsr_out", None)      # lane M4OFF session 3: SEAICE_LSR's per-pass STDOUT values
            if "_cost" in sf_:                       # lane M4ADCOL: cost.h with SEAICE_COST_SENSI's objf_ice
                pk["cost"] = sf_.pop("_cost")
            pk["seaice"] = sf_
        if "gm" in pk:
            pk["gm"] = rest.pop(0)
        if "kpp" in pk:
            pk["kpp"] = rest.pop(0)
        if mix0:
            pk.update(rest.pop(0))
    elif "gm" in pk:
        state, ff, phi0surf, pk["gm"] = do_oceanic_phys(myTime, myIter, cfg=cfg, grid=grid, params=params,   # :657
                                                        fp=fp, eos=eos, state=state, ff=ff, phi0surf=phi0surf,
                                                        probe=pr, gm=pk["gm"], gm_params=params,
                                                        visc=pkc.get("visc"))   # GM_useLeithQG (MLAdjust)
    else:
        state, ff, phi0surf = do_oceanic_phys(myTime, myIter, cfg=cfg, grid=grid, params=params, fp=fp,   # :657
                                              eos=eos, state=state, ff=ff, phi0surf=phi0surf,
                                              probe=pr)                                    # R2 arm (tracer lane)
    pr("S04_oceanic_phys", (state, ff, phi0surf))
    if stop:
        return early()
    flow = None
    # vermix lane (M3 Task 30): the KPP.h fields KPP_CALC wrote, read by CALC_3D_DIFFUSIVITY / GAD_CALC_RHS
    # (KPP_CALC_DIFF_T/S, KPP_TRANSPORT_T/S) and CALC_VISCOSITY (KPP_CALC_VISC)
    # vermix lane: the state of the vertical-mixing packages read by CALC_3D_DIFFUSIVITY / CALC_VISCOSITY / MOM_FLUXFORM
    mixd = {n: pk[n] for n in ("ggl", "pp81", "my82", "dwnslp") if n in pk}   # dwnslp: DWNSLP_APPLY (lane M4ADLAB)
    if params.useKPP:
        mixd["kppf"] = pk["kpp"]
        mixd["kpp_p"] = pkc.get("kpp_p")             # lane M4LAB: KPP_ghatUseTotalDiffus (KPP_TRANSPORT_T/S)
        if "salt_plume" in pk:
            mixd["salt_plume"] = pk["salt_plume"]    # lane M4LAB: SALT_PLUME.h saltPlumeFlux (KPP_TRANSPORT_S)
    mixd = mixd or None
    if not params.staggerTimeStep:                                                                 # :725-736
        state, flow = thermodynamics(myTime, myIter, cfg=cfg, params=params, state=state,
                                     grid=grid, fp=fp, ff=ff, probe=pr, gm=pk.get("gm"),
                                     until=until, mix=mixd, rbcs=_rbcs_fields(pk, pkc))  # R2 arm (tracer lane)
        if stop:
            return early()
        if cfg.cpp.NONLIN_FRSURF and params.nonlinFreeSurf > 0:
            # GO lane: MON_CALC_ADVCFL (thermodynamics.F:278-283, on the host) reads hFacW, hFacS, recip_hFacC as
            # THERMODYNAMICS sees them here, before UPDATE_R_STAR(.TRUE.) (:839) changes them
            flow = tuple(flow) + (grid.hFacW, grid.hFacS, grid.recip_hFacC)
    pr("S05_thermodynamics_sync", state)
    if stop:
        return early()
    if params.momStepping:                                                                         # :786-795
        state = dynamics(myTime, myIter, cfg=cfg, grid=grid, params=params, state=state, ff=ff, fp=fp,
                         phi0surf=phi0surf, probe=pr,                     # D00a/D00b/D00c (M3 lane MLAdjust: D00b)
                         visc=pkc.get("visc"), ctrlf=pkc.get("ctrlf"),         # GOADK lane: MOM_VECINV inputs
                         mix=mixd)                                             # vermix lane: KPP_CALC_VISC
    pr("S06_dynamics", state)
    if stop:
        return early()
    myIter = params.nIter0 + iloop                                                                 # :807
    myTime = params.startTime + params.deltaTClock*iloop.astype(myTime.dtype)                      # :808
    state, grid = nlfs_update_hfac(myTime, myIter, cfg=cfg, grid=grid, params=params, state=state)  # :832-860
    pr("S07_update_rstar_T", state)
    state, cg2dh = nlfs_update_cg2d(myTime, myIter, cfg=cfg, grid=grid, params=params, cg2d_params=cg2d_params,
                                    cg2dh=cg2dh, state=state, ex=ex)                              # :862-875
    pr("S08_update_cg2d", state)
    cg2d = None
    if params.momStepping:                                                                         # :897-906
        state, cg2d = solve_for_pressure(myTime, myIter, cfg=cfg, grid=grid, params=params, state=state, ff=ff,
                                         cg2dh=cg2dh, cg2d_params=cg2d_params, ex=ex)
    pr("S09_solve_for_pressure", state)
    if params.momStepping:                                                                         # :911-917
        state = momentum_correction_step(myTime, myIter, cfg=cfg, grid=grid, params=params, state=state, ex=ex)
    pr("S10_momentum_correction", state)
    if params.calc_wVelocity:                                                                      # :924-930
        state = integr_continuity(state.uVel, state.vVel, myTime, myIter, cfg=cfg, grid=grid, params=params,
                                  state=state, ex=ex, ff=ff)                                       # ff: R2 arm
    pr("S11_integr_continuity", state)
    state, rstar = nlfs_calc_r_star(myTime, myIter, cfg=cfg, grid=grid, params=params, state=state, ex=ex)  # :939-961
    pr("S12_calc_rstar", state)
    if params.staggerTimeStep:                                                  # :976-1009 (GO lane)
        from mitjax.model.src.do_stagger_fields_exchanges import do_stagger_fields_exchanges
        state = do_stagger_fields_exchanges(myTime, myIter, cfg=cfg, params=params, state=state, ex=ex)   # :983
        pr("S13_stagger_exchanges", state)
        # :986-990 DO_STATEVARS_DIAGS (useDiagnostics): diagnostics only, the kernels see useDiagnostics = .FALSE.
        state, flow = thermodynamics(myTime, myIter, cfg=cfg, params=params, state=state,
                                     grid=grid, fp=fp, ff=ff, probe=pr, gm=pk.get("gm"),
                                     mix=mixd, rbcs=_rbcs_fields(pk, pkc))                         # :1005
        pr("S14_thermodynamics_stagger", state)
    state, opps_cnt = tracers_correction_step(myTime, myIter, cfg=cfg, params=params, state=state,  # :1025
                                    grid=grid, eos=eos,                         # grid, eos: PTRACERS lane
                                    opps=pkc.get("opps"), probe=pr)             # vermix lane: OPPS
    pr("S15_tracers_correction", state)
    state = do_fields_blocking_exchanges(cfg=cfg, params=params, state=state, ex=ex)               # :1093
    pr("S16_blocking_exchanges", state)
    if cfg.cpp.ALLOW_COST:                                                                         # :1159-1164 (R5)
        from mitjax.pkg.cost.cost_tile import cost_tile
        tracer = None
        if cfg.cpp.ALLOW_COST_TRACER:         # PTRACERS lane: COST_TILE's COST_TRACER (cost_tile.F:146-152)
            from mitjax.pkg.ptracers.ptracers_fields_h import ptf_of_state
            tracer = dict(grid=grid, params=params, ptr=params, ptf=ptf_of_state(state, params), ex=ex)
        pk["cost"] = cost_tile(pk["cost"], myTime, myIter, cfg=cfg, endTime=pks["endTime"],
                               lastinterval=pks["lastinterval"], tracer=tracer, theta=state.theta, uVel=state.uVel,
                               vVel=state.vVel, maskC=grid.maskC, maskW=grid.maskW, maskS=grid.maskS,
                               deltaTClock=params.deltaTClock, lastinterval_traced=pkc["lastinterval"])
        pr("S18_cost_tile", pk["cost"])
    if pk.get("ecco") is not None:            # lane M4ADCOL session 3: ECCO_PHYS (:1169, useECCO), ECCO.h fields
        from mitjax.pkg.ecco.ecco_phys import ecco_phys
        pk["ecco"] = dict(pk["ecco"], phys=ecco_phys(myIter, cfg=cfg, grid=grid, params=params, eos=eos,
                                                     state=state, ff=ff, fp=fp))
    out = dict(flow=flow, cg2d=cg2d)
    if lsr_out is not None:                   # lane M4OFF session 3: SEAICE_LSR's STDOUT values (with dynamics only)
        out["lsr"] = lsr_out
    if exf_mon is not None:                   # lane M4COL: EXF_FIELDS.h at EXF_MONITOR (exf_getforcing.F:380)
        out["exf_mon"] = {n: v.data for n, v in exf_mon.items() if not n[-1:] in ("0", "1")}
    if pk:                                    # R5 arm: package state (only when there is any: the key is absent
        out["pk"] = pk                        # otherwise, so other variants' out pytrees keep their shape)
    if rstar is not None:                     # GO lane: CALC_R_STAR's host counters (only when it runs: the key is
        out["rstar"] = rstar                  # absent otherwise, so other variants' out pytrees keep their shape)
    if opps_cnt is not None:                  # vermix lane: OPPS_CALC's time-loop bound counters (useOPPS only)
        out["opps"] = opps_cnt
    return state, ff, phi0surf, myTime, myIter, out


def _use(cfg, name):
    """A package switch as PACKAGES_BOOT leaves it; .FALSE. if the build has none."""
    return cfg.use_flag(name) if any(k.lower() == name.lower() for k, _ in cfg.use) else False


# ---- GO lane (plan Task 15a): the NONLIN_FRSURF blocks of FORWARD_STEP, one helper per block, called by
# forward_step above and by the substep gates (mitjax/tests/test_r4a_global_ocean.py) ----------------------------

def nlfs_load(*, cfg, grid, cg2dh, state):
    """Under NONLIN_FRSURF the GRID.h hFacC/W/S, recip_hFacC/W/S and the CG2D.h aW2d ... pC are time-varying common
    blocks carried in the State (mitjax/model/state.py NLFS_GRID, NLFS_CG2D): the step's Grid and CG2DH take the
    State's values. Without NONLIN_FRSURF: (grid, cg2dh) unchanged."""
    if not cfg.cpp.NONLIN_FRSURF:
        return grid, cg2dh
    from mitjax.model.state import NLFS_CG2D, NLFS_GRID
    grid = grid.replace(**{n: getattr(state, n) for n in NLFS_GRID})
    if cg2dh is not None:
        cg2dh = cg2dh.replace(**{n: getattr(state, n) for n in NLFS_CG2D})
    return grid, cg2dh


def _store_grid(state, grid):
    from mitjax.model.state import NLFS_GRID
    return state.replace(**{n: getattr(grid, n) for n in NLFS_GRID})


def nlfs_reset(myTime, myIter, *, cfg, grid, params, state):
    """forward_step.F:461-494 @63cdc0b (NONLIN_FRSURF): `IF (doResetHFactors)` (`#ifndef ALLOW_AUTODIFF`; under
    ALLOW_AUTODIFF unconditional) RESET_NLFS_VARS (:467), then UPDATE_R_STAR(.FALSE.) (:475, select_rStar > 0) or
    UPDATE_SURF_DR(.FALSE.) (:484). Returns (state, grid): the new GRID.h factors are stored in the State."""
    if not cfg.cpp.NONLIN_FRSURF:
        return state, grid
    if not (cfg.cpp.ALLOW_AUTODIFF or params.doResetHFactors):                 # :463-465, :491-493
        return state, grid
    from mitjax.model.src.reset_nlfs_vars import reset_nlfs_vars
    state = reset_nlfs_vars(myTime, myIter, cfg=cfg, params=params, state=state)                   # :467
    if params.select_rStar > 0:                                                 # :468
        if cfg.cpp.DISABLE_RSTAR_CODE:
            return state, grid                                                  # :469-477 compiled out
        from mitjax.model.src.update_r_star import update_r_star
        grid = update_r_star(False, myTime, myIter, cfg=cfg, grid=grid, state=state)                # :475
    else:
        from mitjax.model.src.update_surf_dr import update_surf_dr
        grid = update_surf_dr(False, myTime, myIter, cfg=cfg, grid=grid, params=params, state=state)  # :484
    return _store_grid(state, grid), grid


def nlfs_update_hfac(myTime, myIter, *, cfg, grid, params, state):
    """forward_step.F:832-860 @63cdc0b (NONLIN_FRSURF): UPDATE_R_STAR(.TRUE.) (:839, select_rStar > 0), UPDATE_SIGMA
    (:844, selectSigmaCoord /= 0: raises, not ported) or UPDATE_SURF_DR(.TRUE.) (:852). Returns (state, grid)."""
    if not cfg.cpp.NONLIN_FRSURF:
        return state, grid
    if params.select_rStar > 0:                                                 # :834
        if cfg.cpp.DISABLE_RSTAR_CODE:
            return state, grid
        from mitjax.model.src.update_r_star import update_r_star
        grid = update_r_star(True, myTime, myIter, cfg=cfg, grid=grid, state=state)                 # :839
    elif params.selectSigmaCoord_ne_0:                                          # :841
        raise NotImplementedError("FORWARD_STEP: UPDATE_SIGMA (forward_step.F:844) is not ported")
    else:
        from mitjax.model.src.update_surf_dr import update_surf_dr
        grid = update_surf_dr(True, myTime, myIter, cfg=cfg, grid=grid, params=params, state=state)   # :852
    return _store_grid(state, grid), grid


def nlfs_update_cg2d(myTime, myIter, *, cfg, grid, params, cg2d_params, cg2dh, state, ex):
    """forward_step.F:862-875 @63cdc0b: `#if NONLIN_FRSURF || ALLOW_SOLVE4_PS_AND_DRAG || ALLOW_CG2D_NSA ||
    ALLOW_DEPTH_CONTROL`, `IF (momStepping .AND. (nonlinFreeSurf.GT.2 .OR. selectImplicitDrag.EQ.2))` UPDATE_CG2D
    (:869). Returns (state, cg2dh); under NONLIN_FRSURF the new CG2D.h arrays are stored in the State."""
    if not (cfg.cpp.NONLIN_FRSURF or cfg.cpp.ALLOW_SOLVE4_PS_AND_DRAG or cfg.cpp.ALLOW_CG2D_NSA
            or cfg.cpp.ALLOW_DEPTH_CONTROL):
        return state, cg2dh
    if not (params.momStepping and (params.nonlinFreeSurf > 2 or params.selectImplicitDrag == 2)):   # :866-867
        return state, cg2dh
    if not cfg.cpp.NONLIN_FRSURF:
        raise NotImplementedError("FORWARD_STEP: UPDATE_CG2D without NONLIN_FRSURF (CG2D.h not carried)")
    from mitjax.model.src.update_cg2d import update_cg2d
    from mitjax.model.state import NLFS_CG2D
    cg2dh = update_cg2d(cg2dh, myTime, myIter, cfg=cfg, grid=grid, surface=grid, params=cg2d_params,
                        ex=ex)                                                                      # :869
    return state.replace(**{n: getattr(cg2dh, n) for n in NLFS_CG2D}), cg2dh


def nlfs_calc_r_star(myTime, myIter, *, cfg, grid, params, state, ex):
    """forward_step.F:939-961 @63cdc0b (NONLIN_FRSURF): CALC_R_STAR(etaH) (:949, select_rStar /= 0) or CALC_SURF_DR
    (:957, nonlinFreeSurf > 0 and selectSigmaCoord = 0; lane B, Task 25). Returns (state, counters): `counters`
    are CALC_R_STAR's host counters (calc_r_star_host, calc_r_star_zero_report) or CALC_SURF_DR's adjustment
    {"surf_nb", "surf_vol"} (drivers/run._surf_adjust_lines: the SURF_ADJUSTMENT line), None when neither runs."""
    if not cfg.cpp.NONLIN_FRSURF:
        return state, None
    if params.select_rStar != 0:                                                # :940
        if cfg.cpp.DISABLE_RSTAR_CODE:
            return state, None
        from mitjax.model.src.calc_r_star import calc_r_star
        return calc_r_star(state.etaH, myTime, myIter, cfg=cfg, grid=grid, params=params, state=state,
                           ex=ex)                                                                   # :949
    if params.nonlinFreeSurf > 0 and not params.selectSigmaCoord_ne_0:          # :952 (lane B, Task 25; GOADK lane: code_ad)
        from mitjax.model.src.calc_surf_dr import calc_surf_dr
        state, adjust = calc_surf_dr(state.etaH, myTime, myIter, cfg=cfg, grid=grid, params=params, state=state,
                                     ex=ex)                                                         # :957
        return state, {"surf_nb": adjust[0], "surf_vol": adjust[1]}
    return state, None


def _rbcs_fields(pk, pkc):
    """M3 Task 30: the RBCS_FIELDS.h arrays RBCS_ADD_TENDENCY reads (RBC_mask of RBCS_INIT_FIXED, RBCtemp of
    RBCS_FIELDS_LOAD), or None without pkg/rbcs."""
    if not pk or "rbcs" not in pk:
        return None
    return dict(RBC_mask=pkc["rbcs"]["RBC_mask"], RBCtemp=pk["rbcs"]["RBCtemp"])
