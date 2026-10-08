"""INITIALISE_VARIA: model/src/initialise_varia.F @63cdc0b -- the one cold-start / pickup path of every driver [F§9]."""

from mitjax.model.src.ini_dynvars import ini_dynvars
from mitjax.model.src.ini_fields import ini_fields
from mitjax.model.src.ini_nlfs_vars import ini_nlfs_vars
from mitjax.model.state import empty_state

# Routines INITIALISE_VARIA calls that later tasks port, and the State fields each writes (for the gates' named
# exemptions; a field outside State is not listed). A routine the Fortran executes for the experiment and that is
# not ported raises, unless the caller names it in `pending` (then it is skipped and the gate exempts its fields).
PENDING_WRITES = {
    "INI_FFIELDS": (),                       # FFIELDS.h only (Task 12)
    "INI_FORCING": (),                       # FFIELDS.h only (Task 12)
    "AUTODIFF_INIT_VARIA": (),               # pkg/autodiff variables (Task 16)
    "PACKAGES_INIT_VARIABLES": ("uVelD", "vVelD", "uNM1", "vNM1", "etaNm1"),   # CD_CODE_INI_VARS etc. (Task 15a)
    "COST_INIT_VARIA": (),                   # pkg/cost (Task 16)
    "CALC_R_STAR": ("rStarFacC", "rStarFacW", "rStarFacS", "rStarExpC", "rStarExpW", "rStarExpS",
                    "rStarDhCDt", "rStarDhWDt", "rStarDhSDt", "pStarFacK"),     # Task 15a
    "UPDATE_R_STAR": ("rStarFacNm1C", "rStarFacNm1W", "rStarFacNm1S", "hFac_surfC", "hFac_surfW", "hFac_surfS",
                      "hFacC", "hFacW", "hFacS", "recip_hFacC", "recip_hFacW", "recip_hFacS"),
    "CALC_SURF_DR": ("hFac_surfC", "hFac_surfW", "hFac_surfS", "hFac_surfNm1C", "hFac_surfNm1W",
                     "hFac_surfNm1S"),
    "UPDATE_SURF_DR": ("hFac_surfC", "hFac_surfW", "hFac_surfS", "hFac_surfNm1C", "hFac_surfNm1W",
                       "hFac_surfNm1S", "hFacC", "hFacW", "hFacS", "recip_hFacC", "recip_hFacW", "recip_hFacS"),
    "UPDATE_CG2D": ("aW2d", "aS2d", "aC2d", "pW", "pS", "pC"),   # CG2D.h, carried in the State under NONLIN_FRSURF
    "INTEGR_CONTINUITY": ("wVel", "dEtaHdt", "PmEpR", "etaH", "etaN", "etaHnm1"),  # Task 12 (+ UPDATE_ETAH)
}


def executed_pending(cfg, params):
    """The PENDING_WRITES routines INITIALISE_VARIA executes for this experiment (the conditions of
    initialise_varia.F, cited at the calls in `initialise_varia`)."""
    ip = params.init
    out = ["INI_FFIELDS", "INI_FORCING"]
    if cfg.cpp.ALLOW_AUTODIFF and _use(cfg, "useAUTODIFF"):
        out.append("AUTODIFF_INIT_VARIA")
    out.append("PACKAGES_INIT_VARIABLES")
    if cfg.cpp.ALLOW_COST:
        out.append("COST_INIT_VARIA")
    if cfg.cpp.NONLIN_FRSURF:
        if not cfg.cpp.DISABLE_RSTAR_CODE and ip.select_rStar != 0:
            out.append("CALC_R_STAR")
        if ip.nonlinFreeSurf > 0:
            if ip.select_rStar > 0:
                if not cfg.cpp.DISABLE_RSTAR_CODE:
                    out.append("UPDATE_R_STAR")
            elif ip.selectSigmaCoord != 0:
                raise NotImplementedError("INITIALISE_VARIA: UPDATE_SIGMA is not ported")
            else:
                out += ["CALC_SURF_DR", "UPDATE_SURF_DR"]
        if ip.nonlinFreeSurf > 2:
            out.append("UPDATE_CG2D")
    out.append("INTEGR_CONTINUITY")
    if cfg.cpp.NONLIN_FRSURF and ip.select_rStar == 0 and ip.nonlinFreeSurf > 0 and ip.selectSigmaCoord == 0:
        pass                                  # :343-346 CALC_SURF_DR again (listed above)
    return tuple(dict.fromkeys(out))


def _use(cfg, name):
    return cfg.use_flag(name) if any(k.lower() == name.lower() for k, _ in cfg.use) else False


def _call(name, pending, run):
    if name in pending:
        return
    raise NotImplementedError(f"INITIALISE_VARIA: {name} is executed for this experiment and not ported yet "
                              f"(PENDING_WRITES; name it in `pending` only in a gate that exempts its fields)")


def initialise_varia(grid, *, cfg, params, ex, rw, pending=(), dyn=None, cg2dh=None, cg2d_params=None, host=None,
                     convect_ini_by_caller=False, eos=None):
    """INITIALISE_VARIA( myThid )   @63cdc0b model/src/initialise_varia.F:36-400

    C     | SUBROUTINE INITIALISE_VARIA
    C     | o Set the initial conditions for dynamics variables
    C     |   and time dependent arrays

    Returns the initial State (mitjax/model/state.py). Order of initialise_varia.F: :184 nIter0 from startTime
    (ALLOW_AUTODIFF; INI_PARMS already set it consistently, ini_parms_time); :187-196 ALLOW_DEPTH_CONTROL raises;
    :199 INI_NLFS_VARS; :205 INI_DYNVARS; :209 INI_NH_VARS (ALLOW_NONHYDROSTATIC; lane B); :213
    INI_FFIELDS*; :225 INI_FIELDS; :231 INI_MIXING (writes only under ALLOW_3D_DIFFKR / ALLOW_BL79_LAT_VARY, which no M1
    build sets: raises if one does); :235 TAUEDDY_INIT_VARIA (ALLOW_EDDYPSI) raises; :242 INI_FORCING*; :246
    AUTODIFF_INIT_VARIA* (useAUTODIFF); :263 PACKAGES_INIT_VARIABLES*; :268 COST_INIT_VARIA*; :280-295
    CONVECTIVE_ADJUSTMENT_INI (INCLUDE_CONVECT_INI_CALL) raises; :297-328 r* / surf_dr / cg2d updates*; :334
    INTEGR_CONTINUITY*; :337-347 CALC_R_STAR / CALC_SURF_DR*; :352-370 filters and GRIDALT raise if compiled; :376
    STATE_SUMMARY, :383 MONITOR, :390 DO_THE_MODEL_IO: output only (the monitor lane and the output tasks).
    INTEGR_CONTINUITY (:334, myTime = startTime, myIter = nIter0) runs when `dyn` (the dynamics Params,
    ini_parms.ini_parms_dyn) is given; without it the routine counts as pending. `cg2dh`: INI_CG2D's CG2D.h, copied
    into the State's NLFS_CG2D fields under NONLIN_FRSURF (GO lane; NaN when not given). GO lane: with `dyn`, the
    NONLIN_FRSURF sequence :297-347 runs when not pending (CALC_R_STAR(etaH, startTime, -1), UPDATE_R_STAR(.TRUE.),
    UPDATE_CG2D (needs `cg2dh`, `cg2d_params`), INTEGR_CONTINUITY, CALC_R_STAR(etaH, startTime, nIter0)); the
    CALC_R_STAR counters go to `host` (a list: ("CALC_R_STAR", myIter, counters)) for calc_r_star_host and the
    zero-denominator report. CD_CODE_INI_VARS (packages_init_variables.F:200-210, useCDscheme) runs at :263.
    * = not ported yet (PENDING_WRITES): raises unless named in `pending`.
    PTRACERS lane: `convect_ini_by_caller` = the caller (drivers/model.Model) runs CONVECTIVE_ADJUSTMENT_INI itself
    after its PACKAGES_INIT_VARIABLES part (PTRACERS_INIT_VARIA, CTRL_MAP_INI_GENARR); without it a run that executes
    :280-295 raises.

    Eager, host side (as the grid): every operation is its own XLA computation."""
    if cfg.cpp.ALLOW_DEPTH_CONTROL:
        raise NotImplementedError("INITIALISE_VARIA: ALLOW_DEPTH_CONTROL is not ported")
    unknown = set(pending) - set(PENDING_WRITES)
    if unknown:
        raise KeyError(f"pending names no INITIALISE_VARIA routine: {sorted(unknown)}")
    run = executed_pending(cfg, params)
    state = empty_state(cfg)
    if cfg.cpp.NONLIN_FRSURF:
        # GO lane: the GRID.h thickness factors are State under NONLIN_FRSURF (mitjax/model/state.py NLFS_GRID): the
        # common block holds INI_MASKS_ETC's values until INITIALISE_VARIA's UPDATE_R_STAR / UPDATE_SURF_DR (:307,
        # :318) rewrite them; CG2D.h (NLFS_CG2D) holds INI_CG2D's (initialise_fixed.F:246) until UPDATE_CG2D (:326)
        from mitjax.model.state import NLFS_CG2D, NLFS_GRID
        state = state.replace(**{n: getattr(grid, n) for n in NLFS_GRID})
        if cg2dh is not None:
            state = state.replace(**{n: getattr(cg2dh, n) for n in NLFS_CG2D})
    state = ini_nlfs_vars(state, cfg=cfg, grid=grid, params=params, ex=ex)            # :199
    state = ini_dynvars(state, cfg=cfg, grid=grid, dyn=dyn)                            # :205
    if cfg.cpp.ALLOW_NONHYDROSTATIC:                                                   # :209 (lane B, Task 25)
        from mitjax.model.src.ini_nh_vars import ini_nh_vars
        state = ini_nh_vars(state, cfg=cfg)
    _call("INI_FFIELDS", pending, run)                                                 # :213
    state = ini_fields(state, cfg=cfg, grid=grid, params=params, ex=ex, rw=rw,         # :225
                       dyn=dyn, eos=eos, host=host)                         # vermix lane: INI_PRESSURE
    if cfg.cpp.ALLOW_3D_DIFFKR and dyn is not None:                                    # :231 (PTRACERS lane)
        from mitjax.model.src.ini_mixing import ini_mixing
        # diffKrFile (PARM05): the file's value, else set_defaults.F:370 diffKrFile = ' '
        diffKrFile = dict(cfg.static).get(("data", "parm05", "diffkrfile"), " ")
        state = ini_mixing(state, diffKrFile, cfg=cfg, params=dyn, ex=ex, rw=rw)
    # PTRACERS lane: CONVECTIVE_ADJUSTMENT_INI runs only IF ( startTime .EQ. baseTime .AND. cAdjFreq .NE. 0. )
    # (:280-282; cAdjFreq as INI_PARMS left it, known with the dynamics Params): refused then (it would have to run
    # after PTRACERS_INIT_VARIA), compiled-but-not-run otherwise (tutorial_global_oce_latlon: cAdjFreq = 0)
    convect_ini = cfg.cpp.INCLUDE_CONVECT_INI_CALL and (
        dyn is None or (params.time.startTime == params.time.baseTime and dyn.cAdjFreq != 0.))
    if convect_ini and not convect_ini_by_caller:
        raise NotImplementedError("INITIALISE_VARIA: CONVECTIVE_ADJUSTMENT_INI (:280-295) is not wired")
    for opt in (("ALLOW_3D_DIFFKR",) if dyn is None else ()) + (
            "ALLOW_BL79_LAT_VARY", "ALLOW_EDDYPSI",
            "ALLOW_SHAP_FILT", "ALLOW_ZONAL_FILT", "ALLOW_GRIDALT"):
        if getattr(cfg.cpp, opt):                                                      # :231, :235, :280, :352-370
            raise NotImplementedError(f"INITIALISE_VARIA: the {opt} branch is not ported")
    _call("INI_FORCING", pending, run)                                                 # :242
    import jax.numpy as jnp
    startTime, nIter0 = jnp.float64(params.time.startTime), jnp.int32(params.time.nIter0)
    ported = dyn is not None                     # GO lane: the NONLIN_FRSURF sequence runs with the dynamics Params
    g = grid
    if cfg.cpp.NONLIN_FRSURF:
        from mitjax.model.src.forward_step import nlfs_load
        g, cg2dh = nlfs_load(cfg=cfg, grid=grid, cg2dh=cg2dh, state=state)
    for name in run:
        if name in ("INI_FFIELDS", "INI_FORCING"):
            continue
        if name == "PACKAGES_INIT_VARIABLES" and cfg.cpp.ALLOW_CD_CODE and dyn is not None and dyn.useCDscheme:
            # GO lane: CD_CODE_INI_VARS (packages_init_variables.F:200-210) runs even while the other package
            # initialisations of :263 are pending
            from mitjax.pkg.cd_code.cd_code_ini_vars import cd_code_ini_vars
            state = cd_code_ini_vars(state, cfg=cfg, params=params, ex=ex, rw=rw)
        if name == "CALC_R_STAR" and name not in pending and ported:                   # :299-302 (GO lane)
            from mitjax.model.src.calc_r_star import calc_r_star
            state, cnt = calc_r_star(state.etaH, startTime, -1, cfg=cfg, grid=g, params=dyn, state=state, ex=ex)
            if host is not None:
                host.append(("CALC_R_STAR", -1, cnt))
            continue
        if name == "UPDATE_R_STAR" and name not in pending and ported:                 # :304-307 (GO lane)
            from mitjax.model.src.update_r_star import update_r_star
            from mitjax.model.state import NLFS_GRID
            g = update_r_star(True, startTime, nIter0, cfg=cfg, grid=g, state=state)
            state = state.replace(**{n: getattr(g, n) for n in NLFS_GRID})
            continue
        if name == "CALC_SURF_DR" and name not in pending and ported:                  # :316 (lane B, Task 25)
            from mitjax.model.src.calc_surf_dr import calc_surf_dr
            state, adj = calc_surf_dr(state.etaH, startTime, -1, cfg=cfg, grid=g, params=dyn, state=state, ex=ex)
            if host is not None:
                host.append(("CALC_SURF_DR", -1, adj))
            continue
        if name == "UPDATE_SURF_DR" and name not in pending and ported:                # :322 (lane B, Task 25)
            from mitjax.model.src.update_surf_dr import update_surf_dr
            from mitjax.model.state import NLFS_GRID
            g = update_surf_dr(True, startTime, nIter0, cfg=cfg, grid=g, params=dyn, state=state)
            state = state.replace(**{n: getattr(g, n) for n in NLFS_GRID})
            continue
        if name == "UPDATE_CG2D" and name not in pending and ported:                   # :325-327 (GO lane)
            if cg2dh is None:
                raise ValueError("INITIALISE_VARIA: UPDATE_CG2D needs cg2dh= (INI_CG2D's CG2D.h)")
            from mitjax.model.src.update_cg2d import update_cg2d
            from mitjax.model.state import NLFS_CG2D
            cg2dh = update_cg2d(cg2dh, startTime, params.time.nIter0, cfg=cfg, grid=g, surface=g,
                                params=cg2d_params, ex=ex)
            state = state.replace(**{n: getattr(cg2dh, n) for n in NLFS_CG2D})
            continue
        if name == "INTEGR_CONTINUITY" and name not in pending and dyn is not None:     # :334
            from mitjax.model.src.integr_continuity import integr_continuity
            state = integr_continuity(state.uVel, state.vVel, startTime, nIter0, cfg=cfg, grid=g, params=dyn,
                                      state=state, ex=ex)
            if cfg.cpp.NONLIN_FRSURF and "CALC_SURF_DR" in run and "CALC_SURF_DR" not in pending and ported:
                from mitjax.model.src.calc_surf_dr import calc_surf_dr                 # :343-346 (lane B)
                state, adj = calc_surf_dr(state.etaH, startTime, nIter0, cfg=cfg, grid=g, params=dyn, state=state,
                                          ex=ex)
                if host is not None:
                    host.append(("CALC_SURF_DR", int(params.time.nIter0), adj))
            if cfg.cpp.NONLIN_FRSURF and "CALC_R_STAR" in run and "CALC_R_STAR" not in pending and ported:
                from mitjax.model.src.calc_r_star import calc_r_star                   # :337-341 (GO lane)
                state, cnt = calc_r_star(state.etaH, startTime, nIter0, cfg=cfg, grid=g, params=dyn,
                                         state=state, ex=ex)
                if host is not None:
                    host.append(("CALC_R_STAR", int(params.time.nIter0), cnt))
            continue
        _call(name, pending, run)                                                      # :246-347
    return state
