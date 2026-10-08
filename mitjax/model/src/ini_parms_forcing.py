"""INI_PARMS / SET_PARMS (model/src/ini_parms.F, set_parms.F @63cdc0b), forcing group: the PARAMS.h values the forcing
routines read (LOAD_FIELDS_DRIVER, EXTERNAL_FIELDS_LOAD, INI_FORCING, EXTERNAL_FORCING_SURF, FORCING_SURF_RELAX,
FREEZE_SURFACE, APPLY_FORCING_U/V/T/S).

Same construction as mitjax/model/src/ini_parms.py (M1 core lane): namelist values from the run's parameter files
(mitjax/config), every value the files do not set from its cited SET_DEFAULTS line (`fortran_default`, checked to be
live in this build), and the derivation statements of ini_parms.F / set_parms.F ported as code with their lines.
A fourth group beside GridParams / TimeParams / InitParams, kept in this lane's own file because ini_parms.py belongs
to the core lane; to be folded into ini_parms.py when the lanes merge.

Host-side, frozen, never traced: the float values reach the kernels as Python floats (constants of the traced
program, as the Fortran PARAMETER-like run constants); none decides a branch inside a kernel except through the
static logical/integer switches here.
"""

from dataclasses import dataclass, field

import numpy as np

from mitjax.model.grid import UNSET_RL
from mitjax.model.src.ini_parms import (UNSET_I, _blank, _get, _get_or_unset, _use, ini_parms_time,
                                        load_ref_files_tref_sref)
from mitjax.params_io import RunParams

SD = "model/src/set_defaults.F"

# eesupp/inc/EEPARAMS.h:90-95 @63cdc0b
debLevZero = 0              # EEPARAMS.h:90  PARAMETER ( debLevZero=0 )
debLevA = 1                 # EEPARAMS.h:91  PARAMETER ( debLevA=1 )
debLevB = 2                 # EEPARAMS.h:92  PARAMETER ( debLevB=2 )
debLevC = 3                 # EEPARAMS.h:93  PARAMETER ( debLevC=3 )
debLevD = 4                 # EEPARAMS.h:94  PARAMETER ( debLevD=4 )


@dataclass(frozen=True)
class ForcingParams:
    """PARAMS.h forcing values as INI_PARMS and SET_PARMS leave them (ini_parms_forcing)."""
    # PARM01
    rhoConst: float
    rhoConstFresh: float
    gravity: float
    HeatCapacity_Cp: float
    sIceLoadFac: float
    useRealFreshWaterFlux: bool
    temp_EvPrRn: float
    salt_EvPrRn: float
    convertFW2Salt: float
    selectAddFluid: int
    allowFreezing: bool
    momForcing: bool
    momTidalForcing: bool
    tempForcing: bool
    saltForcing: bool
    linFSConserveTr: bool
    selectPenetratingSW: int
    nonlinFreeSurf: int
    select_rStar: int
    selectSigmaCoord: int
    staggerTimeStep: bool
    debugLevel: int
    eosType: str
    # PARM03
    tauThetaClimRelax: float
    tauSaltClimRelax: float
    periodicExternalForcing: bool
    externForcingPeriod: float
    externForcingCycle: float
    deltaTClock: float
    nIter0: int
    monitorFreq: float
    # PARM05 (CHARACTER*(MAX_LEN_FNAM), blank = ' ')
    zonalWindFile: str
    meridWindFile: str
    thetaClimFile: str
    saltClimFile: str
    EmPmRfile: str
    saltFluxFile: str
    surfQFile: str
    surfQnetFile: str
    surfQswFile: str
    pLoadFile: str
    geothermalFile: str
    lambdaThetaFile: str
    lambdaSaltFile: str
    # derived (ini_parms.F, set_parms.F)
    recip_rhoConst: float
    recip_gravity: float
    mass2rUnit: float
    rUnit2mass: float
    foFacMom: float
    doThetaClimRelax: bool
    doSaltClimRelax: bool
    usingPCoords: bool
    usingZCoords: bool
    fluidIsAir: bool
    fluidIsWater: bool
    tRef: np.ndarray = field(repr=False, default=None)     # PARAMS.h tRef(Nr) (LOAD_REF_FILES)
    selectBalanceEmPmR: int = 0     # lane B (INI_FORCING's ALLOW_BALANCE_FLUXES arm): ini_parms.F:727-734
    addMassFile: str = " "          # lane B: PARM05, set_defaults.F:389
    wghtBalanceFile: str = " "      # lane B: PARM05, set_defaults.F:395
    dTtracerLev: np.ndarray = field(repr=False, default=None)   # PARAMS.h dTtracerLev(Nr) (ini_parms.F:1064-1066)
    balanceQnet: bool = False       # GO lane (EXTERNAL_FORCING_SURF ALLOW_BALANCE_FLUXES): PARM01, set_defaults.F:270
    balanceThetaClimRelax: bool = False     # GO lane (FORCING_SURF_RELAX ALLOW_BALANCE_RELAX): set_defaults.F:272
    balanceSaltClimRelax: bool = False      # GO lane: set_defaults.F:273


def ini_parms_forcing(exp):
    """The forcing parameters of an experiment (cited per value; see the module docstring)."""
    rp = RunParams(exp.run)
    cfg = exp.cfg
    g = lambda grp, key, line: _get(exp, rp, grp, key, f"{SD}:{line}")         # noqa: E731
    # ini_parms.F:447-459: buoyancyRelation; every M1 variant is OCEANIC (ini_parms_grid raises otherwise)
    buoyancyRelation = g("PARM01", "buoyancyRelation", 176)
    if str(buoyancyRelation).strip() == "ATMOSPHERIC":                          # ini_parms.F:451-453 (lane B)
        usingPCoords, usingZCoords, fluidIsAir, fluidIsWater = True, False, True, False
    elif str(buoyancyRelation).strip() != "OCEANIC":
        raise NotImplementedError(f"INI_PARMS forcing group: buoyancyRelation={buoyancyRelation!r} is not ported")
    else:
        usingPCoords, usingZCoords, fluidIsAir, fluidIsWater = False, True, False, True   # ini_parms.F:447-459
    rhoNil = g("PARM01", "rhoNil", 104)
    rhoConst = _get_or_unset(rp, "PARM01", "rhoConst", UNSET_RL)                # set_defaults.F:105 UNSET_RL
    if rhoConst == UNSET_RL:                                                    # ini_parms.F:483
        rhoConst = rhoNil
    rhoConstFresh = _get_or_unset(rp, "PARM01", "rhoConstFresh", UNSET_RL)      # set_defaults.F:109 UNSET_RL
    if rhoConstFresh == UNSET_RL:                                               # ini_parms.F:484
        rhoConstFresh = rhoConst
    gravity = g("PARM01", "gravity", 101)
    useRealFreshWaterFlux = g("PARM01", "useRealFreshWaterFlux", 263)
    selectAddFluid = g("PARM01", "selectAddFluid", 262)
    convertFW2Salt = _get_or_unset(rp, "PARM01", "convertFW2Salt", UNSET_RL)    # set_defaults.F:110 UNSET_RL
    if convertFW2Salt == UNSET_RL:                                              # ini_parms.F:647-651
        convertFW2Salt = 35.                                                    # REAL*4 literal 35. (exact)
        if useRealFreshWaterFlux:
            convertFW2Salt = -1.
        if selectAddFluid >= 1:
            convertFW2Salt = -1.
    selectPenetratingSW = _get_or_unset(rp, "PARM01", "selectPenetratingSW", UNSET_I)   # set_defaults.F:268
    if selectPenetratingSW == UNSET_I:                                          # ini_parms.F:652-657
        selectPenetratingSW = 0
        if cfg.cpp.SHORTWAVE_HEATING and fluidIsWater:
            selectPenetratingSW = 1
    recip_rhoConst = 1.0 / rhoConst                                             # ini_parms.F:763 (rhoConst > 0)
    recip_gravity = 1.0 / gravity                                               # ini_parms.F:784 (gravity > 0)
    if usingPCoords:                                                            # ini_parms.F:1569-1575
        mass2rUnit, rUnit2mass = gravity, recip_gravity
    else:
        mass2rUnit, rUnit2mass = recip_rhoConst, rhoConst
    # set_parms.F:92-93, 211-215, 244-253
    momStepping = g("PARM01", "momStepping", 192)
    tempStepping = g("PARM01", "tempStepping", 194)
    saltStepping = g("PARM01", "saltStepping", 198)
    momForcing = bool(momStepping and g("PARM01", "momForcing", 188))           # set_parms.F:92
    momTidalForcing = bool(momForcing and g("PARM01", "momTidalForcing", 189))  # set_parms.F:93
    foFacMom = 1.0 if momForcing else 0.0                                       # set_parms.F:211-215
    tempForcing = bool(tempStepping and g("PARM01", "tempForcing", 196))        # set_parms.F:244
    saltForcing = bool(saltStepping and g("PARM01", "saltForcing", 200))        # set_parms.F:247
    tauThetaClimRelax = g("PARM03", "tauThetaClimRelax", 329)
    tauSaltClimRelax = g("PARM03", "tauSaltClimRelax", 330)
    if _use(cfg, "useEXF"):
        # lane M4OFF: PACKAGES_READPARMS (initialise_fixed.F:133) -> EXF_READPARMS overwrites both before SET_PARMS
        # (:140) reads them (exf_readparms.F:1076 tauThetaClimRelax = climsstTauRelax under ALLOW_CLIMSST_RELAXATION,
        # :1090 tauSaltClimRelax = climsssTauRelax under ALLOW_CLIMSSS_RELAXATION; its STOP when `data` sets them
        # stays in exf_readparms)
        from mitjax.pkg.exf.exf_readparms import exf_clim_tau
        if cfg.cpp.flag("ALLOW_CLIMSST_RELAXATION", "EXF_OPTIONS.h"):
            tauThetaClimRelax = exf_clim_tau(exp, "climsstTauRelax")
        if cfg.cpp.flag("ALLOW_CLIMSSS_RELAXATION", "EXF_OPTIONS.h"):
            tauSaltClimRelax = exf_clim_tau(exp, "climsssTauRelax")
    useOffLine, useKPP = _use(cfg, "useOffLine"), _use(cfg, "useKPP")
    doThetaClimRelax = bool((tempForcing or (useOffLine and useKPP)) and tauThetaClimRelax > 0.)   # :250-251
    doSaltClimRelax = bool((saltForcing or (useOffLine and useKPP)) and tauSaltClimRelax > 0.)     # :252-253
    # debugLevel: set_defaults.F:240-246 (debugMode from eedata, eesupp/src/eeset_parms.F:125, or PARM01)
    debugMode = rp.get("data", "PARM01", "debugMode") if rp.has("data", "PARM01", "debugMode") else (
        rp.get("eedata", "EEPARM", "debugMode") if rp.has("eedata", "EEPARM", "debugMode") else False)
    if rp.has("data", "PARM01", "debugLevel"):
        debugLevel = int(rp.get("data", "PARM01", "debugLevel"))
    elif debugMode:
        debugLevel = debLevD                                                    # set_defaults.F:241
    else:
        debugLevel = debLevA if cfg.cpp.ALLOW_AUTODIFF else debLevB             # set_defaults.F:243, 245
    periodicExternalForcing = g("PARM03", "periodicExternalForcing", 331)
    externForcingPeriod = g("PARM03", "externForcingPeriod", 332)
    externForcingCycle = g("PARM03", "externForcingCycle", 333)
    if periodicExternalForcing:                                                 # ini_parms.F:1068-1080
        if externForcingCycle * externForcingPeriod == 0.:
            raise ValueError("S/R INI_PARMS: externForcingCycle,externForcingPeriod =0")
        if int(externForcingCycle / externForcingPeriod) != externForcingCycle / externForcingPeriod:
            raise ValueError("S/R INI_PARMS: externForcingCycle <> N*externForcingPeriod")
    tp = ini_parms_time(exp)
    # monitorFreq as read; ini_parms.F:1187-1196 (monitorFreq < 0 -> from dumpFreq/diagFreq/chkPtFreq) is not
    # ported: the one reader here (SBO_READPARMS) raises on a negative value
    monitorFreq = g("PARM03", "monitorFreq", 351)
    f5 = {k: g("PARM05", k, line) for k, line in (
        ("zonalWindFile", 375), ("meridWindFile", 376), ("thetaClimFile", 377), ("saltClimFile", 378),
        ("EmPmRfile", 379), ("saltFluxFile", 380), ("surfQnetFile", 382), ("surfQswFile", 383),
        ("pLoadFile", 387), ("geothermalFile", 392), ("lambdaThetaFile", 393), ("lambdaSaltFile", 394))}
    f5["surfQFile"] = g("PARM05", "surfQfile", 381)
    return ForcingParams(
        rhoConst=float(rhoConst), rhoConstFresh=float(rhoConstFresh), gravity=float(gravity),
        HeatCapacity_Cp=float(g("PARM01", "HeatCapacity_Cp", 174)), sIceLoadFac=float(g("PARM01", "sIceLoadFac", 179)),
        useRealFreshWaterFlux=bool(useRealFreshWaterFlux),
        temp_EvPrRn=float(_get_or_unset(rp, "PARM01", "temp_EvPrRn", UNSET_RL)),   # set_defaults.F:264 UNSET_RL
        salt_EvPrRn=float(g("PARM01", "salt_EvPrRn", 265)), convertFW2Salt=float(convertFW2Salt),
        selectAddFluid=int(selectAddFluid), allowFreezing=bool(g("PARM01", "allowFreezing", 220)),
        momForcing=momForcing, momTidalForcing=momTidalForcing, tempForcing=tempForcing, saltForcing=saltForcing,
        linFSConserveTr=bool(g("PARM01", "linFSConserveTr", 255)), selectPenetratingSW=int(selectPenetratingSW),
        nonlinFreeSurf=int(g("PARM01", "nonlinFreeSurf", 257)), select_rStar=int(g("PARM01", "select_rStar", 260)),
        selectSigmaCoord=int(g("PARM04", "selectSigmaCoord", 48)),
        staggerTimeStep=bool(g("PARM01", "staggerTimeStep", 183)), debugLevel=int(debugLevel),
        eosType=str(g("PARM01", "eosType", 175)).strip(),
        tauThetaClimRelax=float(tauThetaClimRelax), tauSaltClimRelax=float(tauSaltClimRelax),
        periodicExternalForcing=bool(periodicExternalForcing), externForcingPeriod=float(externForcingPeriod),
        externForcingCycle=float(externForcingCycle), deltaTClock=float(tp.deltaTClock), nIter0=int(tp.nIter0),
        monitorFreq=float(monitorFreq),
        **{k: (" " if _blank(v) else str(v)) for k, v in f5.items()},
        recip_rhoConst=recip_rhoConst, recip_gravity=recip_gravity, mass2rUnit=mass2rUnit, rUnit2mass=rUnit2mass,
        foFacMom=foFacMom, doThetaClimRelax=doThetaClimRelax, doSaltClimRelax=doSaltClimRelax,
        usingPCoords=usingPCoords, usingZCoords=usingZCoords, fluidIsAir=fluidIsAir, fluidIsWater=fluidIsWater,
        tRef=load_ref_files_tref_sref(exp)[0], dTtracerLev=np.asarray(tp.dTtracerLev, dtype=np.float64),
        selectBalanceEmPmR=_select_balance_empmr(rp), balanceQnet=bool(g("PARM01", "balanceQnet", 270)),
        balanceThetaClimRelax=bool(g("PARM01", "balanceThetaClimRelax", 272)),
        balanceSaltClimRelax=bool(g("PARM01", "balanceSaltClimRelax", 273)),
        **{k: (" " if _blank(v) else str(v)) for k, v in (
            ("addMassFile", g("PARM05", "addMassFile", 389)),
            ("wghtBalanceFile", g("PARM05", "wghtBalanceFile", 395)))})


def _select_balance_empmr(rp):
    """selectBalanceEmPmR as INI_PARMS leaves it (lane B): set_defaults.F:269 UNSET_I, balanceEmPmR :347 .FALSE.;
    ini_parms.F:727-734 (unset -> 0, or 1 with balanceEmPmR; a conflict stops)."""
    sel = _get_or_unset(rp, "PARM01", "selectBalanceEmPmR", UNSET_I)
    bal = bool(rp.get("data", "PARM01", "balanceEmPmR")) if rp.has("data", "PARM01", "balanceEmPmR") else False
    if sel == UNSET_I:
        return 1 if bal else 0
    if sel != 1 and bal:
        raise ValueError(f"S/R INI_PARMS: selectBalanceEmPmR={sel} conflicts with \"balanceEmPmR\"")
    return int(sel)
