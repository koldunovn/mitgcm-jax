"""SEAICE_MODEL: pkg/seaice/seaice_model.F @63cdc0b."""

from mitjax.eesupp.exch_rs import EXCH_XY_RL, EXCH_XY_RS, _rewrap
from mitjax.farray import loop_i, loop_j
from mitjax.pkg.seaice.dynsolver import dynsolver
from mitjax.pkg.seaice.seaice_growth import seaice_growth
from mitjax.pkg.seaice.seaice_h import level, set_level
from mitjax.pkg.seaice.seaice_reg_ridge import seaice_reg_ridge
from mitjax.pkg.seaice.seaice_tracer_phys import seaice_tracer_phys


def seaice_model(myTime, myIter, sf, ff, exf, *, cfg, sp, op, grid, state, exfp, ex, kgeo, probe=None,
                 salt_plume=None, spp=None):
    """SEAICE_MODEL( myTime, myIter, myThid )   @63cdc0b pkg/seaice/seaice_model.F:13-415

    C     | o Time stepping of a dynamic/thermodynamic sea ice model.

    `sf` SEAICE.h / SEAICE_GRID.h (dict), `ff` FFields, `exf` EXF_FIELDS.h (dict), `exfp` ExfParams, `state` (theta,
    salt, uVel, vVel, etaN), `kgeo` the KGEO level (dynsolver.kgeo_level, static). Returns (sf, ff, exf).
    `probe(stage, sf, ff, exf)` (optional, static) is called where the dump build writes I00b_seaice_begin (before
    DYNSOLVER, :182), I01b_dynsolver (after it), I03_reg_ridge (:249), I04_growth (:277).
    Ported for this build: ALLOW_EXF with useEXF (the A-grid exchange of uwind/vwind :127), SEAICE_BGRID_DYNAMICS
    (DYNSOLVER :182) or (lane M4OFF) the C-grid build: SEAICE_DYNSOLVER (:186, probes I00_seaice_begin,
    Y01_get_dynforcing, Y09_ocean_stress, I01_dynsolver) and SEAICE_ADVDIFF(uIce, vIce) (:230, probe I02_advdiff);
    for the B-grid build no advection (SEAICEadv* .FALSE.; SEAICE_ADVDIFF :221-233 raises in SEAICE_READPARMS),
    SEAICE_REG_RIDGE (:249), usePW79thermodynamics (SEAICE_GROWTH :277) or (lane M4OFF session 3) its else arm
    (sIceLoad from HEFF and HSNOW, :281-299), the exchanges :317-342 (SEAICE_VARIABLE_SALINITY: HSALT; SHORTWAVE_HEATING: Qsw;
    ATMOSPHERIC_LOADING + useRealFreshWaterFlux: sIceLoad). usingPCoords (:85-93), ALLOW_THSICE, ALLOW_AUTODIFF,
    ALLOW_OBCS, SEAICE_ITD, DISABLE_SEAICE_GROWTH, SEAICE_USE_GROWTH_ADX raise when compiled / used.
    ALLOW_SITRACER (lane M4LAB session 4): SEAICE_TRACER_PHYS after the thermodynamics (:302-306) and the exchange of
    the tracers in use (:328-332); SEAICE_ADVDIFF, SEAICE_REG_RIDGE and SEAICE_GROWTH carry their SITRACER parts.
    Diagnostics (:101-122, :128-134, :251-259, :365-393; ALLOW_DIAGNOSTICS / useDiagnostics) are output only.
    Lane M4LAB session 3: with ALLOW_SALT_PLUME, `salt_plume` (SALT_PLUME.h) and `spp` (SaltPlumeParams) go to
    SEAICE_GROWTH, which writes saltPlumeFlux; then (sf, ff, exf, salt_plume) is returned, and the I04_growth probe
    sees saltPlumeFlux / SaltPlumeDepth with the SEAICE.h fields."""
    for o in ("SEAICE_ITD", "DISABLE_SEAICE_GROWTH", "SEAICE_USE_GROWTH_ADX"):
        if cfg.cpp.flag(o, "SEAICE_OPTIONS.h"):
            raise NotImplementedError(f"SEAICE_MODEL: {o} is not ported")
    if (cfg.cpp.flag("ALLOW_THSICE") and cfg.use_flag("useThSIce")) \
            or (cfg.cpp.flag("ALLOW_OBCS") and cfg.use_flag("useOBCS")):
        raise NotImplementedError("SEAICE_MODEL: useThSIce / useOBCS are not ported")
    # lane M4ADLAB session 2: ALLOW_AUTODIFF with the C-grid dynamics (SEAICE_DYNSOLVER / SEAICE_LSR arms ported)
    if cfg.cpp.flag("ALLOW_AUTODIFF") and (cfg.cpp.flag("SEAICE_BGRID_DYNAMICS", "SEAICE_OPTIONS.h")
                                         or cfg.cpp.flag("ALLOW_SITRACER", "SEAICE_OPTIONS.h")):
        raise NotImplementedError("SEAICE_MODEL: ALLOW_AUTODIFF with the B-grid dynamics / ALLOW_SITRACER is not "
                                  "ported")
    if op.usingPCoords:                                                        # :85-93
        raise NotImplementedError("SEAICE_MODEL: usingPCoords (phiHydLow exchange) is not ported")
    bgrid = cfg.cpp.flag("SEAICE_BGRID_DYNAMICS", "SEAICE_OPTIONS.h")
    exf = dict(exf)
    if cfg.cpp.flag("ALLOW_EXF") and cfg.use_flag("useEXF"):                   # :124-135
        u, v = ex.EXCH_UV_AGRID_3D_RL(exf["uwind"].data, exf["vwind"].data, True)   # :127
        exf["uwind"], exf["vwind"] = _rewrap(u, exf["uwind"]), _rewrap(v, exf["vwind"])
    if cfg.cpp.flag("ALLOW_AUTODIFF"):                    # :137-153 (lane M4ADCOL: 1D_ocean_ice_column/code_ad)
        # uIceNm1 = vIceNm1 = 0 on every point (:143-144; SItrBucket :146-148 only with ALLOW_SITRACER: raises above)
        sf = dict(sf)
        jA = loop_j(1-cfg.size.OLy, cfg.size.sNy+cfg.size.OLy)
        iA = loop_i(1-cfg.size.OLx, cfg.size.sNx+cfg.size.OLx)
        for n in ("uIceNm1", "vIceNm1"):
            sf[n] = sf[n].at[iA, jA].set(0.)
    if bgrid:
        if probe is not None:
            probe("I00b_seaice_begin", sf, ff, exf)
        sf, ff = dynsolver(myTime, myIter, sf, ff, exf, cfg=cfg, sp=sp, op=op, grid=grid, state=state, ex=ex,
                           kgeo=kgeo)                                          # :182
        if probe is not None:
            probe("I01b_dynsolver", sf, ff, exf)
        if sp.SEAICEadvHeff or sp.SEAICEadvArea or sp.SEAICEadvSnow or sp.SEAICEadvSalt:   # :221-233
            raise NotImplementedError("SEAICE_MODEL: SEAICE_ADVDIFF of the B-grid build (uLoc/vLoc) is not ported")
    else:                                                                      # lane M4OFF: the C-grid build
        from mitjax.pkg.seaice.seaice_advdiff import seaice_advdiff
        from mitjax.pkg.seaice.seaice_dynsolver import seaice_dynsolver
        if probe is not None:
            probe("I00_seaice_begin", sf, ff, exf)

        def dyn_probe(stage, vals, ff_):
            probe(stage, {**sf, **vals}, ff_, exf)
        sf, ff = seaice_dynsolver(myTime, myIter, sf, ff, exf, cfg=cfg, sp=sp, op=op, exfp=exfp, grid=grid,
                                  state=state, ex=ex,
                                  probe=None if probe is None else dyn_probe)  # :186
        if probe is not None:
            probe("I01_dynsolver", sf, ff, exf)
        if sp.SEAICEadvHeff or sp.SEAICEadvArea or sp.SEAICEadvSnow or sp.SEAICEadvSalt:   # :221-233
            sf = seaice_advdiff(sf["UICE"], sf["VICE"], myTime, myIter, sf, cfg=cfg, sp=sp, op=op,
                                grid=grid, ex=ex)                              # :230
        if probe is not None:
            probe("I02_advdiff", sf, ff, exf)
    sf = seaice_reg_ridge(myTime, myIter, sf, cfg=cfg, sp=sp, op=op)           # :249
    if probe is not None:
        probe("I03_reg_ridge", sf, ff, exf)
    if sp.usePW79thermodynamics:                                               # :267-300
        if salt_plume is not None:                                             # lane M4LAB: ALLOW_SALT_PLUME
            sf, ff, salt_plume = seaice_growth(myTime, myIter, sf, ff, exf, cfg=cfg, sp=sp, op=op, grid=grid,
                                               state=state, exfp=exfp, salt_plume=salt_plume, spp=spp)   # :277
        else:
            sf, ff = seaice_growth(myTime, myIter, sf, ff, exf, cfg=cfg, sp=sp, op=op, grid=grid, state=state,
                                   exfp=exfp)                                  # :277
    elif salt_plume is not None:
        raise NotImplementedError("SEAICE_MODEL: ALLOW_SALT_PLUME without usePW79thermodynamics is not ported")
    else:                                                                      # :281-299 (lane M4OFF session 3)
        sz = cfg.size
        j = loop_j(1, sz.sNy)
        i = loop_i(1, sz.sNx)
        ff = ff.replace(sIceLoad=ff.sIceLoad.at[i, j].set(sf["HEFF"][i, j]*sp.SEAICE_rhoIce      # :285-286
                                                          + sf["HSNOW"][i, j]*sp.SEAICE_rhoSnow))
    if probe is not None:
        probe("I04_growth", sf if salt_plume is None else {**sf, **salt_plume}, ff, exf)
    sitracer = cfg.cpp.flag("ALLOW_SITRACER", "SEAICE_OPTIONS.h")              # lane M4LAB session 4
    if sitracer:                                                               # :302-306
        sf = seaice_tracer_phys(myTime, myIter, sf, cfg=cfg, sp=sp)            # :305
    sf = dict(sf)
    sf["HEFF"] = EXCH_XY_RL(sf["HEFF"], ex=ex)                                 # :317
    sf["AREA"] = EXCH_XY_RL(sf["AREA"], ex=ex)                                 # :318
    sf["HSNOW"] = EXCH_XY_RL(sf["HSNOW"], ex=ex)                               # :319
    if cfg.cpp.flag("SEAICE_VARIABLE_SALINITY", "SEAICE_OPTIONS.h"):           # :325-327
        sf["HSALT"] = EXCH_XY_RL(sf["HSALT"], ex=ex)
    if sitracer:                                                               # :328-332
        for iTr in range(1, sp.SItrNumInUse + 1):                              # :329
            sf["SItracer"] = set_level(sf["SItracer"], iTr,                    # :330
                                       EXCH_XY_RL(level(sf["SItracer"], iTr), ex=ex))
    ff = ff.replace(EmPmR=EXCH_XY_RS(ff.EmPmR, ex=ex),                        # :333
                    saltFlux=EXCH_XY_RS(ff.saltFlux, ex=ex),                   # :334
                    Qnet=EXCH_XY_RS(ff.Qnet, ex=ex))                           # :335
    if cfg.cpp.flag("SHORTWAVE_HEATING"):                                      # :336-338
        ff = ff.replace(Qsw=EXCH_XY_RS(ff.Qsw, ex=ex))
    if cfg.cpp.flag("ATMOSPHERIC_LOADING") and op.useRealFreshWaterFlux:       # :339-342
        ff = ff.replace(sIceLoad=EXCH_XY_RS(ff.sIceLoad, ex=ex))
    if salt_plume is not None:
        return sf, ff, exf, salt_plume
    return sf, ff, exf
