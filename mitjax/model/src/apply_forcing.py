"""APPLY_FORCING_U, APPLY_FORCING_V, APPLY_FORCING_T, APPLY_FORCING_S: model/src/apply_forcing.F @63cdc0b (one .F,
four routines).

Each routine adds the forcing terms of level `k` to a 2-D tendency array `g?_arr(1-OLx:sNx+OLx,1-OLy:sNy+OLy)` of
one tile (the caller's local array, e.g. guExt in TIMESTEP, gtForc in TEMP_INTEGRATE): here an FArray over all tiles
(the bi,bj loop is implicit) declared with i, j only. `k` is a Python int (the caller's level loop). The loop bounds
are the Fortran's (0:sNx+1 / 0:sNy+1 or 1:sNx+1 etc.), not iMin..iMax, which the live branches do not use.
USE_OLD_EXTERNAL_FORCING (adjustment.cs-32x32x1; lane B, Task 25): APPLY_FORCING_U/V call EXTERNAL_FORCING_U/V on the
State's gU/gV (the caller passes it, `gU=` / `gV=`) and add the difference to g?_arr (`_old_external_forcing`); the
T and S arms raise. The package hooks (AIM, ATM_PHYS, FIZHI, EDDYPSI, RBCS,
OBCS, MYPACKAGE, ADDFLUID, FRICTION_HEATING, GEOTHERMAL_FLUX, SHORTWAVE_HEATING, FRAZIL, SHELFICE, ICEFRONT,
SALT_PLUME, BBL) raise when a build compiles them and switches them on (no M1 variant does). GO lane: ADDFLUID's
switch is selectAddFluid /= 0 (apply_forcing.F:505, :875; global_ocean.cs32x15 compiles it with selectAddFluid = 0).
M3 Task 30: RBCS (useRBCS) calls RBCS_ADD_TENDENCY at its place in each routine (`_rbcs`; `params` with
RBCS_PARAMS.h, `rbcs` the RBCS_FIELDS.h arrays and theta for APPLY_FORCING_T).
"""

from mitjax.farray import loop_i, loop_j


def _use(cfg, name):
    return cfg.use_flag(name) if any(k.lower() == name.lower() for k, _ in cfg.use) else False


def _hooks(cfg, routine, pairs, fp=None):
    for opt, use in pairs:
        if use == "selectAddFluid":     # GO lane: the ADDFLUID terms run IF ( selectAddFluid.NE.0 ...) (:505, :875)
            on = fp is None or fp.selectAddFluid != 0
        else:
            on = use is None or _use(cfg, use)
        if getattr(cfg.cpp, opt) and on:
            raise NotImplementedError(f"{routine}: {opt} ({use or 'compiled'}) is not ported")


_UV_HOOKS = (("ALLOW_AIM", "useAIM"), ("ALLOW_ATM_PHYS", "useAtm_Phys"), ("ALLOW_FIZHI", "useFIZHI"),
             ("ALLOW_EDDYPSI", None), ("ALLOW_RBCS", "useRBCS"), ("ALLOW_OBCS", "useOBCS"),
             ("ALLOW_MYPACKAGE", "useMYPACKAGE"))
_TS_HOOKS = (("ALLOW_AIM", "useAIM"), ("ALLOW_ATM_PHYS", "useAtm_Phys"), ("ALLOW_FIZHI", "useFIZHI"),
             ("ALLOW_ADDFLUID", "selectAddFluid"), ("ALLOW_SHELFICE", "useShelfIce"), ("ALLOW_ICEFRONT", "useICEFRONT"),
             ("ALLOW_SALT_PLUME", "useSALT_PLUME"), ("ALLOW_RBCS", "useRBCS"), ("ALLOW_OBCS", "useOBCS"),
             ("ALLOW_MYPACKAGE", "useMYPACKAGE"))
_T_HOOKS = (("ALLOW_FRICTION_HEATING", None), ("ALLOW_GEOTHERMAL_FLUX", None), ("SHORTWAVE_HEATING", None),
            ("ALLOW_FRAZIL", "useFRAZIL"), ("ALLOW_BBL", "useBBL"))


def _rbcs(g_arr, k, tracerNum, myTime, myIter, *, cfg, params, rbcs):
    """IF (useRBCS) CALL RBCS_ADD_TENDENCY( g_arr, k, bi, bj, tracerNum, myTime, myIter, myThid ) under #ifdef
    ALLOW_RBCS (apply_forcing.F:168-176 U, :358-365 V, :726-733 T, :958-965 S)."""
    if not (cfg.cpp.ALLOW_RBCS and _use(cfg, "useRBCS")):
        return g_arr
    if params is None:
        raise ValueError("APPLY_FORCING: useRBCS needs `params` (RBCS_PARAMS.h)")
    from mitjax.pkg.rbcs.rbcs_add_tendency import rbcs_add_tendency
    return rbcs_add_tendency(g_arr, k, tracerNum, myTime, myIter, cfg=cfg, params=params, rbcs=rbcs,
                             theta=None if rbcs is None else rbcs.get("theta"))


def _ksurface(fp, cfg, with_shelfice):
    """kSurface (apply_forcing.F:96-102 for U/V, :466-474 for T, :837-845 for S)."""
    if fp.fluidIsAir:
        return 0
    if with_shelfice and fp.usingZCoords and cfg.cpp.ALLOW_SHELFICE and _use(cfg, "useShelfIce"):
        return -1
    if fp.usingPCoords:
        return cfg.size.Nr
    return 1


def _old_external_forcing(g_arr, g_state, k, ext, iMin, iMax, jMin, jMax, myTime, *, cfg, grid, fp, ff):
    """The USE_OLD_EXTERNAL_FORCING arm of APPLY_FORCING_U/V (apply_forcing.F:67-92 / :257-282; lane B): locVar =
    gU(:,:,k); EXTERNAL_FORCING_U updates the State's gU(:,:,k); tmpVar = gU - locVar and gU = locVar (so the State
    is left as it was, bit for bit); g_arr = g_arr + tmpVar, every point of the tile."""
    if g_state is None:
        raise ValueError("APPLY_FORCING: USE_OLD_EXTERNAL_FORCING needs the State's gU / gV")
    sz = cfg.size
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    locVar = g_state[i, j, k]                                                   # :69-73 / :259-263
    g_new = ext(g_state, iMin, iMax, jMin, jMax, k, myTime, cfg=cfg, grid=grid, fp=fp, ff=ff)   # :74-76 / :264-266
    tmpVar = g_new[i, j, k] - locVar                                            # :82 / :272
    return g_arr.at[i, j].set(g_arr[i, j] + tmpVar)                             # :89 / :279


def apply_forcing_u(gU_arr, iMin, iMax, jMin, jMax, k, myTime, myIter, *, cfg, grid, fp, ff, gU=None, params=None,
                    rbcs=None):
    """APPLY_FORCING_U( gU_arr, iMin,iMax,jMin,jMax, k, bi, bj, myTime, myIter, myThid )   @63cdc0b
    model/src/apply_forcing.F:15-199

    C     | S/R APPLY_FORCING_U
    C     | o Contains problem specific forcing for zonal velocity.
    C     | Adds terms to gU for forcing by external sources
    C     | e.g. wind stress, bottom friction etc ...

    Ported: tidal forcing from phiTide2d (:127-136, momTidalForcing), the surface stress in level kSurface
    (:139-148); the kSurface = -1 branch (:149-159) is unreachable for U (kSurface is 0, Nr or 1, :96-102) and is
    kept literally. Returns the new gU_arr."""
    if cfg.cpp.USE_OLD_EXTERNAL_FORCING:                                        # :67-92 (lane B)
        from mitjax.model.src.external_forcing import external_forcing_u
        return _old_external_forcing(gU_arr, gU, k, external_forcing_u, iMin, iMax, jMin, jMax, myTime, cfg=cfg,
                                     grid=grid, fp=fp, ff=ff)
    sz = cfg.size
    kSurface = _ksurface(fp, cfg, with_shelfice=False)                         # :96-102
    _hooks(cfg, "APPLY_FORCING_U", _UV_HOOKS[:3])                               # :105-124
    if fp.momTidalForcing:                                                      # :127-136
        j = loop_j(0, sz.sNy+1)
        i = loop_i(1, sz.sNx+1)
        gU_arr = gU_arr.at[i, j].set(
            gU_arr[i, j]
            - grid.recip_dxC[i, j]*grid.recip_deepFacC[k]
            * (ff.phiTide2d[i, j] - ff.phiTide2d[i-1, j])
            * grid.maskW[i, j, k])
    if k == kSurface:                                                           # :139-148
        j = loop_j(0, sz.sNy+1)
        i = loop_i(1, sz.sNx+1)
        gU_arr = gU_arr.at[i, j].set(
            gU_arr[i, j]
            + fp.foFacMom*ff.surfaceForcingU[i, j]
            * grid.recip_drF[k]*grid.recip_hFacW[i, j, k])
    elif kSurface == -1:                                                        # :149-159
        raise NotImplementedError("APPLY_FORCING_U: kSurface = -1 (kSurfW) branch is not ported")
    _hooks(cfg, "APPLY_FORCING_U", _UV_HOOKS[3:4])                              # :161-166
    gU_arr = _rbcs(gU_arr, k, -1, myTime, myIter, cfg=cfg, params=params, rbcs=rbcs)    # :168-176 (M3)
    _hooks(cfg, "APPLY_FORCING_U", _UV_HOOKS[5:])                               # :178-194
    return gU_arr


def apply_forcing_v(gV_arr, iMin, iMax, jMin, jMax, k, myTime, myIter, *, cfg, grid, fp, ff, gV=None, params=None,
                    rbcs=None):
    """APPLY_FORCING_V( gV_arr, iMin,iMax,jMin,jMax, k, bi, bj, myTime, myIter, myThid )   @63cdc0b
    model/src/apply_forcing.F:205-388

    C     | S/R APPLY_FORCING_V
    C     | o Contains problem specific forcing for merid velocity.
    C     | Adds terms to gV for forcing by external sources
    C     | e.g. wind stress, bottom friction etc ...

    Ported: tidal forcing (:317-326), the surface stress in level kSurface (:329-338). Returns the new gV_arr."""
    if cfg.cpp.USE_OLD_EXTERNAL_FORCING:                                        # :257-282 (lane B)
        from mitjax.model.src.external_forcing import external_forcing_v
        return _old_external_forcing(gV_arr, gV, k, external_forcing_v, iMin, iMax, jMin, jMax, myTime, cfg=cfg,
                                     grid=grid, fp=fp, ff=ff)
    sz = cfg.size
    kSurface = _ksurface(fp, cfg, with_shelfice=False)                         # :286-292
    _hooks(cfg, "APPLY_FORCING_V", _UV_HOOKS[:3])                               # :295-314
    if fp.momTidalForcing:                                                      # :317-326
        j = loop_j(1, sz.sNy+1)
        i = loop_i(0, sz.sNx+1)
        gV_arr = gV_arr.at[i, j].set(
            gV_arr[i, j]
            - grid.recip_dyC[i, j]*grid.recip_deepFacC[k]
            * (ff.phiTide2d[i, j] - ff.phiTide2d[i, j-1])
            * grid.maskS[i, j, k])
    if k == kSurface:                                                           # :329-338
        j = loop_j(1, sz.sNy+1)
        i = loop_i(0, sz.sNx+1)
        gV_arr = gV_arr.at[i, j].set(
            gV_arr[i, j]
            + fp.foFacMom*ff.surfaceForcingV[i, j]
            * grid.recip_drF[k]*grid.recip_hFacS[i, j, k])
    elif kSurface == -1:                                                        # :339-349
        raise NotImplementedError("APPLY_FORCING_V: kSurface = -1 (kSurfS) branch is not ported")
    _hooks(cfg, "APPLY_FORCING_V", _UV_HOOKS[3:4])                              # :351-356
    gV_arr = _rbcs(gV_arr, k, -2, myTime, myIter, cfg=cfg, params=params, rbcs=rbcs)    # :358-365 (M3)
    _hooks(cfg, "APPLY_FORCING_V", _UV_HOOKS[5:])                               # :367-383
    return gV_arr


def apply_forcing_t(gT_arr, iMin, iMax, jMin, jMax, k, myTime, myIter, *, cfg, grid, fp, ff, params=None,
                    rbcs=None):
    """APPLY_FORCING_T( gT_arr, iMin,iMax,jMin,jMax, k, bi, bj, myTime, myIter, myThid )   @63cdc0b
    model/src/apply_forcing.F:394-763

    C     | S/R APPLY_FORCING_T
    C     | o Contains problem specific forcing for temperature.
    C     | Adds terms to gT for forcing by external sources
    C     | e.g. heat flux, climatalogical relaxation, etc ...

    Ported: the surface heat flux in level kSurface (:618-625). Raises: the compressible-air term (:554-615,
    fluidIsAir), kSurface = -1 (:626-636, shelf ice), linFSConserveTr (:638-647; TsurfCor is set by
    CALC_WSURF_TR, not ported here) and the package hooks. PTRACERS lane: the penetrating short-wave term
    (:680-693, SHORTWAVE_HEATING with selectPenetratingSW > 0; _recip_hFacC = recip_hFacC). Returns the new
    gT_arr."""
    if cfg.cpp.USE_OLD_EXTERNAL_FORCING:                                        # :448-462
        raise NotImplementedError("APPLY_FORCING_T: USE_OLD_EXTERNAL_FORCING is not ported")
    sz = cfg.size
    kSurface = _ksurface(fp, cfg, with_shelfice=True)                          # :466-474
    _hooks(cfg, "APPLY_FORCING_T", _TS_HOOKS[:4] + _T_HOOKS[:1], fp)            # :483-552
    if fp.fluidIsAir:                                                           # :554-615
        raise NotImplementedError("APPLY_FORCING_T: the fluidIsAir moist-air term is not ported")
    if k == kSurface:                                                           # :618-625
        j = loop_j(0, sz.sNy+1)
        i = loop_i(0, sz.sNx+1)
        gT_arr = gT_arr.at[i, j].set(
            gT_arr[i, j]
            + ff.surfaceForcingT[i, j]
            * grid.recip_drF[k]*grid.recip_hFacC[i, j, k])
    elif kSurface == -1:                                                        # :626-636
        raise NotImplementedError("APPLY_FORCING_T: kSurface = -1 (shelf ice) is not ported")
    if fp.linFSConserveTr:                                                      # :638-647
        raise NotImplementedError("APPLY_FORCING_T: linFSConserveTr (TsurfCor) is not ported")
    _hooks(cfg, "APPLY_FORCING_T", _T_HOOKS[1:2])                               # :649-678
    if cfg.cpp.SHORTWAVE_HEATING and fp.selectPenetratingSW > 0:                # :680-693 (PTRACERS lane)
        recip_Cp = 1.0 / fp.HeatCapacity_Cp                                     # :476  1. _d 0 / HeatCapacity_Cp
        j = loop_j(0, sz.sNy+1)
        i = loop_i(0, sz.sNx+1)
        gT_arr = gT_arr.at[i, j].set(
            gT_arr[i, j]
            + ff.Qsw[i, j]*grid.gravitySign
            * (ff.SWFrac3D[i, j, k] - ff.SWFrac3D[i, j, k+1])
            * recip_Cp*fp.mass2rUnit
            * grid.recip_drF[k]*grid.recip_hFacC[i, j, k])
    _hooks(cfg, "APPLY_FORCING_T", _T_HOOKS[3:4] + _TS_HOOKS[4:7])              # :695-724
    gT_arr = _rbcs(gT_arr, k, 1, myTime, myIter, cfg=cfg, params=params, rbcs=rbcs)     # :726-733 (M3)
    _hooks(cfg, "APPLY_FORCING_T", _TS_HOOKS[8:9] + _T_HOOKS[4:] + _TS_HOOKS[9:])   # :735-758
    return gT_arr


def apply_forcing_s(gS_arr, iMin, iMax, jMin, jMax, k, myTime, myIter, *, cfg, grid, fp, ff, params=None,
                    rbcs=None):
    """APPLY_FORCING_S( gS_arr, iMin,iMax,jMin,jMax, k, bi, bj, myTime, myIter, myThid )   @63cdc0b
    model/src/apply_forcing.F:769-993

    C     | S/R APPLY_FORCING_S
    C     | o Contains problem specific forcing for merid velocity.
    C     | Adds terms to gS for forcing by external sources
    C     | e.g. fresh-water flux, climatalogical relaxation, etc ...

    Ported: the surface salt flux in level kSurface (:904-911). Raises: kSurface = -1 (:912-922), linFSConserveTr
    (:924-933), the package hooks. Returns the new gS_arr."""
    if cfg.cpp.USE_OLD_EXTERNAL_FORCING:                                        # :819-833
        raise NotImplementedError("APPLY_FORCING_S: USE_OLD_EXTERNAL_FORCING is not ported")
    sz = cfg.size
    kSurface = _ksurface(fp, cfg, with_shelfice=True)                          # :837-845
    _hooks(cfg, "APPLY_FORCING_S", _TS_HOOKS[:4], fp)                           # :853-901
    if k == kSurface:                                                           # :904-911
        j = loop_j(0, sz.sNy+1)
        i = loop_i(0, sz.sNx+1)
        gS_arr = gS_arr.at[i, j].set(
            gS_arr[i, j]
            + ff.surfaceForcingS[i, j]
            * grid.recip_drF[k]*grid.recip_hFacC[i, j, k])
    elif kSurface == -1:                                                        # :912-922
        raise NotImplementedError("APPLY_FORCING_S: kSurface = -1 (shelf ice) is not ported")
    if fp.linFSConserveTr:                                                      # :924-933
        raise NotImplementedError("APPLY_FORCING_S: linFSConserveTr (SsurfCor) is not ported")
    _hooks(cfg, "APPLY_FORCING_S", _TS_HOOKS[4:7])                              # :935-956
    gS_arr = _rbcs(gS_arr, k, 2, myTime, myIter, cfg=cfg, params=params, rbcs=rbcs)     # :958-965 (M3)
    _hooks(cfg, "APPLY_FORCING_S", _TS_HOOKS[8:])                               # :967-988
    return gS_arr
