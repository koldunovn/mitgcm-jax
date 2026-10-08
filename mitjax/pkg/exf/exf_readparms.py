"""EXF_READPARMS: pkg/exf/exf_readparms.F @63cdc0b."""

import numpy as np

from mitjax.model.grid import UNSET_RL
from mitjax.params_io import RunParams, fortran_default
from mitjax.pkg.exf.exf_param_h import ExfParams

_RP = "pkg/exf/exf_readparms.F"
_C2K = 276
_F = "data.exf"

# The fields of EXF_NML_02 this port reads (EXF_GETFFIELDS / EXF_GETCLIM of the builds ported so far), as
# (field, prefix of its namelist variables): `<prefix>startdate1`, `<prefix>period`, `<prefix>file`, ...
FIELDS = (("hflux", "hflux"), ("atemp", "atemp"), ("aqh", "aqh"), ("evap", "evap"), ("precip", "precip"),
          ("snowprecip", "snowprecip"), ("sflux", "sflux"), ("runoff", "runoff"), ("saltflx", "saltflx"),
          ("ustress", "ustress"), ("vstress", "vstress"), ("uwind", "uwind"), ("vwind", "vwind"),
          ("wspeed", "wspeed"), ("swflux", "swflux"), ("lwflux", "lwflux"), ("swdown", "swdown"),
          ("lwdown", "lwdown"), ("apressure", "apressure"), ("climsst", "climsst"), ("climsss", "climsss"),
          ("tidePot", "tidePot"))

# exf_readparms.F line of the literal default of each namelist variable (:300-724, before the READs at :939-985)
_LINES = {
    "hfluxstartdate1": 375, "hfluxstartdate2": 376, "hfluxperiod": 377, "hfluxconst": 378, "hfluxfile": 609,
    "hflux_exfremo_intercept": 379, "hflux_exfremo_slope": 380, "exf_inscal_hflux": 683,
    "atempstartdate1": 382, "atempstartdate2": 383, "atempperiod": 384, "atempfile": 610,
    "atemp_exfremo_intercept": 386, "atemp_exfremo_slope": 387, "exf_inscal_atemp": 696,
    "aqhstartdate1": 389, "aqhstartdate2": 390, "aqhperiod": 391, "aqhconst": 392, "aqhfile": 611,
    "aqh_exfremo_intercept": 393, "aqh_exfremo_slope": 394, "exf_inscal_aqh": 698,
    "evapstartdate1": 417, "evapstartdate2": 418, "evapperiod": 419, "evapconst": 420, "evapfile": 614,
    "evap_exfremo_intercept": 421, "evap_exfremo_slope": 422, "exf_inscal_evap": 701,
    "precipstartdate1": 424, "precipstartdate2": 425, "precipperiod": 426, "precipconst": 427, "precipfile": 615,
    "precip_exfremo_intercept": 428, "precip_exfremo_slope": 429, "exf_inscal_precip": 692,
    "snowprecipstartdate1": 431, "snowprecipstartdate2": 432, "snowprecipperiod": 433, "snowprecipconst": 434,
    "snowprecipfile": 616, "snowprecip_exfremo_intercept": 435, "snowprecip_exfremo_slope": 436,
    "exf_inscal_snowprecip": 693,
    "sfluxstartdate1": 410, "sfluxstartdate2": 411, "sfluxperiod": 412, "sfluxconst": 413, "sfluxfile": 617,
    "sflux_exfremo_intercept": 414, "sflux_exfremo_slope": 415, "exf_inscal_sflux": 684,
    "runoffstartdate1": 438, "runoffstartdate2": 439, "runoffperiod": 440, "runoffconst": 441, "runofffile": 618,
    "runoff_exfremo_intercept": 442, "runoff_exfremo_slope": 443, "exf_inscal_runoff": 703,
    "saltflxstartdate1": 449, "saltflxstartdate2": 450, "saltflxperiod": 451, "saltflxconst": 452,
    "saltflxfile": 620, "saltflx_exfremo_intercept": 453, "saltflx_exfremo_slope": 454, "exf_inscal_saltflx": 705,
    "ustressstartdate1": 456, "ustressstartdate2": 457, "ustressperiod": 458, "ustressconst": 459,
    "ustressfile": 621, "ustress_exfremo_intercept": 460, "ustress_exfremo_slope": 461, "exf_inscal_ustress": 685,
    "vstressstartdate1": 463, "vstressstartdate2": 464, "vstressperiod": 465, "vstressconst": 466,
    "vstressfile": 622, "vstress_exfremo_intercept": 467, "vstress_exfremo_slope": 468, "exf_inscal_vstress": 686,
    "uwindstartdate1": 470, "uwindstartdate2": 471, "uwindperiod": 472, "uwindconst": 473, "uwindfile": 623,
    "uwind_exfremo_intercept": 474, "uwind_exfremo_slope": 475, "exf_inscal_uwind": 687,
    "vwindstartdate1": 477, "vwindstartdate2": 478, "vwindperiod": 479, "vwindconst": 480, "vwindfile": 624,
    "vwind_exfremo_intercept": 481, "vwind_exfremo_slope": 482, "exf_inscal_vwind": 688,
    "wspeedstartdate1": 484, "wspeedstartdate2": 485, "wspeedperiod": 486, "wspeedconst": 487, "wspeedfile": 625,
    "wspeed_exfremo_intercept": 488, "wspeed_exfremo_slope": 489, "exf_inscal_wspeed": 689,
    "swfluxstartdate1": 491, "swfluxstartdate2": 492, "swfluxperiod": 493, "swfluxconst": 494, "swfluxfile": 626,
    "swflux_exfremo_intercept": 495, "swflux_exfremo_slope": 496, "exf_inscal_swflux": 690,
    "lwfluxstartdate1": 498, "lwfluxstartdate2": 499, "lwfluxperiod": 500, "lwfluxconst": 501, "lwfluxfile": 627,
    "lwflux_exfremo_intercept": 502, "lwflux_exfremo_slope": 503, "exf_inscal_lwflux": 691,
    "swdownstartdate1": 505, "swdownstartdate2": 506, "swdownperiod": 507, "swdownconst": 508, "swdownfile": 628,
    "swdown_exfremo_intercept": 509, "swdown_exfremo_slope": 510, "exf_inscal_swdown": 706,
    "lwdownstartdate1": 512, "lwdownstartdate2": 513, "lwdownperiod": 514, "lwdownconst": 515, "lwdownfile": 629,
    "lwdown_exfremo_intercept": 516, "lwdown_exfremo_slope": 517, "exf_inscal_lwdown": 707,
    "apressurestartdate1": 519, "apressurestartdate2": 520, "apressureperiod": 521, "apressurefile": 630,
    "apressure_exfremo_intercept": 523, "apressure_exfremo_slope": 524, "exf_inscal_apressure": 702,
    "climsststartdate1": 541, "climsststartdate2": 542, "climsstperiod": 543, "climsstTauRelax": 544,
    "climsstconst": 545, "climsstfile": 633, "climsst_exfremo_intercept": 546, "climsst_exfremo_slope": 547,
    "exf_inscal_climsst": 708,
    "climsssstartdate1": 549, "climsssstartdate2": 550, "climsssperiod": 551, "climsssTauRelax": 552,
    "climsssconst": 553, "climsssfile": 634, "climsss_exfremo_intercept": 554, "climsss_exfremo_slope": 555,
    "exf_inscal_climsss": 709,
    "exf_outscal_hflux": 716, "exf_outscal_sflux": 717, "exf_outscal_ustress": 718, "exf_outscal_vstress": 719,
    "exf_outscal_swflux": 720, "exf_outscal_sst": 721, "exf_outscal_sss": 722, "exf_outscal_apressure": 723,
    "exf_offset_atemp": 697,
    "useExfCheckRange": 307, "select_ZenAlbedo": 308, "readStressOnAgrid": 310, "rotateStressOnAgrid": 311,
    "readStressOnCgrid": 312,
    "useRelativeWind": 318, "noNegativeEvap": 319,
    "cen2kel": 329, "gravity_mks": 330, "atmrho": 331, "atmcp": 332, "flamb": 333, "flami": 334,
    "cvapor_fac": 335, "cvapor_exp": 336, "humid_fac": 339, "gamma_blk": 340, "saltsat": 341, "sstExtrapol": 342,
    "cdrag_1": 343, "cdrag_2": 344, "cdrag_3": 345, "cstanton_1": 355, "cstanton_2": 356, "cdalton": 357,
    "zolmin": 358, "psim_fac": 359, "zref": 360, "hu": 361, "ht": 362, "umin": 363, "exf_albedo": 367,
    "ice_emissivity": 371, "snow_emissivity": 372,
    "repeatPeriod": 600, "windstressmax": 601, "exf_scal_BulkCdn": 603, "climtempfreeze": 606,
    "exf_iprec": 676, "exf_yftype": 678, "useExfYearlyFields": 679, "twoDigitYear": 680,
    # ALLOW_RUNOFTEMP (lane M4CS32ICE): runoftemp shares runoff's period, start and mask (exf_getffields.F:443-452)
    "runoftempconst": 445, "runoftemp_exfremo_intercept": 446, "runoftemp_exfremo_slope": 447,
    "runoftempfile": 619, "exf_inscal_runoftemp": 704,
    "useExfZenIncoming": 309, "tidePotfile": 631,
    # EXF_ALLOW_TIDES (lane M4CS32ICE: compiled in global_ocean.cs32x15, no file)
    "tidePotstartdate1": 526, "tidePotstartdate2": 527, "tidePotperiod": 528, "tidePotconst": 529,
    "tidePot_exfremo_intercept": 530, "tidePot_exfremo_slope": 531, "exf_inscal_tidePot": 712,
    "exf_outscal_tidePot": 724,
}
# EXF_SEAICE_FRACTION (lane M4ADLAB session 3: lab_sea/code_ad compiles it, input_ad sets no areamask variable): read
# only when the build compiles the option (the field exists in EXF_FIELDS.h only then, :301-308)
_LINES_AREAMASK = {
    "areamaskstartdate1": 533, "areamaskstartdate2": 534, "areamaskperiod": 535, "areamaskTauRelax": 536,
    "areamaskconst": 537, "areamask_exfremo_intercept": 538, "areamask_exfremo_slope": 539, "areamaskfile": 632,
    "exf_inscal_areamask": 713, "exf_outscal_areamask": 725,
}
# the namelist group that holds each variable (exf_readparms.F:70-281 NAMELIST statements)
_NML_01 = ("windstressmax", "repeatPeriod", "exf_albedo", "ocean_emissivity", "ice_emissivity", "snow_emissivity",
           "exf_iprec", "exf_yftype", "useExfYearlyFields", "twoDigitYear", "useExfCheckRange", "useRelativeWind",
           "noNegativeEvap", "useAtmWind", "readStressOnAgrid", "readStressOnCgrid", "rotateStressOnAgrid",
           "useStabilityFct_overIce", "select_ZenAlbedo", "useExfZenIncoming", "cen2kel", "gravity_mks", "atmrho",
           "atmcp", "flamb", "flami", "cvapor_fac", "cvapor_exp", "cvapor_fac_ice", "cvapor_exp_ice", "humid_fac",
           "gamma_blk", "saltsat", "sstExtrapol", "cdrag_1", "cdrag_2", "cdrag_3", "cdrag_8", "cdragMax", "umax",
           "cstanton_1", "cstanton_2", "cdalton", "zolmin", "psim_fac", "zref", "hu", "ht", "umin", "exf_iceCd",
           "exf_iceCe", "exf_iceCh", "exf_scal_BulkCdn", "climtempfreeze", "exf_verbose", "exf_debugLev",
           "exf_monFreq", "exf_adjMonFreq", "exf_adjMonSelect", "diags_opOceWeighted")
_NML_03_PREFIX = ("exf_inscal_", "exf_outscal_", "exf_offset_")


def _group(name):
    if name in _NML_01:
        return "EXF_NML_01"
    if name.startswith(_NML_03_PREFIX) or "_exfremo_" in name:
        return "EXF_NML_03"
    return "EXF_NML_02"


def exf_readparms(exp, *, params):
    """EXF_READPARMS( myThid )   @63cdc0b pkg/exf/exf_readparms.F:3-1101

    C     | S/R EXF_READPARMS
    C     | o Read parameters for the external forcing package

    `params` PARAMS.h (surf_pRef; celsius2K is read from `data` here). exf_debugLev / exf_verbose / exf_monFreq only
    select print-out and are not carried. Returns ExfParams, or None when
    useEXF is .FALSE. (:283-288).

    Defaults: the literal of :300-724 read from its line by `fortran_default` (table `_LINES`); the non-literal ones
    as code with their lines: `atempconst = celsius2K` (:385), `apressureconst = surf_pRef` (:522),
    `ocean_emissivity = 5.50 _d -8 / 5.670 _d -8` (:370), every `<fld>StartTime = UNSET_RL` (:639-663; EXF_INIT_FIXED
    sets them from the start dates), `<fld>RepCycle = repeatPeriod` after the EXF_NML_01 READ (:940-966). After the
    READs (:1036-1101): `hq = ht` (:1044), stressIsOnCgrid (:1045-1053), useExfZenAlbedo (:1055-1056). Raises for
    options this port does not carry: wspeedfile, ustress/vstressfile with useAtmWind, a climsst/climsss file
    (relaxation through EXF), useExfYearlyFields, select_ZenAlbedo /= 0 (ALLOW_ZENITHANGLE)."""
    cfg = exp.cfg
    if not cfg.use_flag("useEXF"):                                             # :283-288
        return None
    rp = RunParams(exp.run)
    v = {}
    seaice_fraction = cfg.cpp.flag("EXF_SEAICE_FRACTION", "EXF_OPTIONS.h")
    lines = {**_LINES, **_LINES_AREAMASK} if seaice_fraction else _LINES
    for name, line in lines.items():
        g = _group(name)
        v[name] = rp.get(_F, g, name, default=fortran_default(f"{_RP}:{line}", name, exp))
    # useAtmWind: the default depends on the build (:313-317): .TRUE. with ALLOW_ATM_WIND (:314), .FALSE. without
    # (:316, lane M4CS32ICE: global_ocean.cs32x15's EXF_OPTIONS.h undefines it)
    awl = 314 if cfg.cpp.flag("ALLOW_ATM_WIND", "EXF_OPTIONS.h") else 316
    v["useAtmWind"] = rp.get(_F, "EXF_NML_01", "useAtmWind", default=fortran_default(f"{_RP}:{awl}", "useAtmWind",
                                                                                    exp))
    # lane M4ADCOL: cdrag_8, cdragMax, umax: the Large & Yeager (2009) literals under ALLOW_DRAG_LARGEYEAGER09
    # (:347-349), UNSET_RL otherwise (:351-353)
    ly09 = cfg.cpp.flag("ALLOW_DRAG_LARGEYEAGER09", "EXF_OPTIONS.h")
    for name, line in (("cdrag_8", 347), ("cdragMax", 348), ("umax", 349)):
        if ly09:
            v[name] = rp.get(_F, "EXF_NML_01", name, default=fortran_default(f"{_RP}:{line}", name, exp))
        else:
            v[name] = rp.get(_F, "EXF_NML_01", name) if rp.has(_F, "EXF_NML_01", name) else UNSET_RL
    # non-literal defaults (code)
    v["atempconst"] = rp.get(_F, "EXF_NML_02", "atempconst") if rp.has(_F, "EXF_NML_02", "atempconst") \
        else float(params_celsius2K(exp))                                      # :385 atempconst = celsius2K
    v["apressureconst"] = rp.get(_F, "EXF_NML_02", "apressureconst") \
        if rp.has(_F, "EXF_NML_02", "apressureconst") else float(params_surf_pRef(exp))   # :522 = surf_pRef
    v["ocean_emissivity"] = rp.get(_F, "EXF_NML_01", "ocean_emissivity") \
        if rp.has(_F, "EXF_NML_01", "ocean_emissivity") else 5.50e-8 / 5.670e-8      # :370
    for fld, p in FIELDS + ((("areamask", "areamask"),) if seaice_fraction else ()):
        st = p + "StartTime"                                                   # :639-663 UNSET_RL (:661 areamask)
        v[st] = rp.get(_F, "EXF_NML_02", st) if rp.has(_F, "EXF_NML_02", st) else UNSET_RL
        rc = p + "RepCycle"                                                    # :940-966 = repeatPeriod
        v[rc] = rp.get(_F, "EXF_NML_02", rc) if rp.has(_F, "EXF_NML_02", rc) else v["repeatPeriod"]
    if v["exf_yftype"].strip() != "RL":                                        # :1033-1035
        raise RuntimeError("S/R EXF_READPARAMS: value of exf_yftype not allowed")
    v["hq"] = v["ht"]                                                          # :1044
    v["stressIsOnCgrid"] = v["readStressOnCgrid"]                              # :1045
    if cfg.cpp.flag("ALLOW_BULKFORMULAE", "EXF_OPTIONS.h") and v["useAtmWind"]:   # :1046-1048
        v["stressIsOnCgrid"] = False
    if cfg.cpp.flag("USE_EXF_INTERPOLATION", "EXF_OPTIONS.h"):                # :1049-1053
        raise NotImplementedError("EXF_READPARMS: USE_EXF_INTERPOLATION is not ported")
    v["useExfZenAlbedo"] = 1 <= v["select_ZenAlbedo"] <= 3                    # :1055-1056
    # host flag of the REAL-namelist test that selects code in EXF_GETFORCING (exf_getforcing.F:168): a gradient or a
    # perturbation of sstExtrapol must not cross 0
    v["sstExtrapol_gt_0"] = bool(v["sstExtrapol"] > 0.)
    if cfg.cpp.flag("ALLOW_CLIMSST_RELAXATION", "EXF_OPTIONS.h"):              # :1067-1078
        if params_tau(exp, "tauThetaClimRelax") != 0.:
            raise RuntimeError("EXF_READPARMS: with EXF, cannot use \"tauThetaClimRelax\" in \"data\"\n"
                               "ABNORMAL END: S/R EXF_READPARMS")
    if cfg.cpp.flag("ALLOW_CLIMSSS_RELAXATION", "EXF_OPTIONS.h"):              # :1081-1092
        if params_tau(exp, "tauSaltClimRelax") != 0.:
            raise RuntimeError("EXF_READPARMS: with EXF, cannot use \"tauSaltClimRelax\" in \"data\"\n"
                               "ABNORMAL END: S/R EXF_READPARMS")
    # EXF_CHECK's wind checks (exf_check.F:110-126; the routine itself is not carried): STOP after the errors
    if v["useAtmWind"] and (v["ustressfile"].strip() or v["vstressfile"].strip()):      # exf_check.F:110-117
        raise RuntimeError("EXF_CHECK: use u,v_wind components but not wind-stress\nABNORMAL END: S/R EXF_CHECK")
    if not v["useAtmWind"] and (v["uwindfile"].strip() or v["vwindfile"].strip()):      # exf_check.F:119-126
        raise RuntimeError("EXF_CHECK: read-in wind-stress but not u,v_wind components\n"
                           "ABNORMAL END: S/R EXF_CHECK")
    # options not ported (raise at setup)
    if v["tidePotfile"].strip():
        raise NotImplementedError("EXF: tidePotFile (EXF_ALLOW_TIDES) is not ported")
    if seaice_fraction and v["areamaskfile"].strip():
        raise NotImplementedError("EXF: areamaskfile (EXF_SEAICE_FRACTION read from a file: EXF_GETFFIELD_START, "
                                  "EXF_SET_FLD records, EXF_MONITOR's statistics) is not ported")
    if v["useExfZenIncoming"]:
        raise NotImplementedError("EXF: useExfZenIncoming (ALLOW_ZENITHANGLE) is not ported")
    # lane M4OFF: a climsst / climsss file is carried (EXF_GETCLIM, EXF_MAPFIELDS -> SST/SSS, the relaxation of
    # FORCING_SURF_RELAX with tauThetaClimRelax = climsstTauRelax, :1076, applied by ini_parms_forcing)
    if v["rotateStressOnAgrid"]:
        raise NotImplementedError("EXF: rotateStressOnAgrid (EXF_SET_UV vector rotation) is not ported")
    if v["useExfYearlyFields"]:
        raise NotImplementedError("EXF: useExfYearlyFields is not ported")
    if v["useExfZenAlbedo"]:
        raise NotImplementedError("EXF: select_ZenAlbedo in 1..3 (ALLOW_ZENITHANGLE) is not ported")
    r = {k: np.float64(x) for k, x in v.items() if isinstance(x, float) and not isinstance(x, bool)}
    s = tuple(sorted((k, x) for k, x in v.items() if k not in r))
    return ExfParams(r=r, s=s)


def exf_clim_tau(exp, name):
    """climsstTauRelax / climsssTauRelax of data.exf (EXF_NML_02, else their literal default :544 / :552): the value
    EXF_READPARMS writes into PARAMS.h tauThetaClimRelax / tauSaltClimRelax (:1076 / :1090, ALLOW_CLIMSST/SSS_RELAXATION).
    Read by ini_parms_forcing (SET_PARMS runs after PACKAGES_READPARMS, initialise_fixed.F:133, :140)."""
    rp = RunParams(exp.run)
    return rp.get(_F, "EXF_NML_02", name, default=fortran_default(f"{_RP}:{_LINES[name]}", name, exp))


def params_surf_pRef(exp):
    """PARAMS.h surf_pRef of `data` (PARM01, else set_defaults.F:103 `101325. _d 0`): read here, not from the Model's
    params, which carry it only for the non-linear equations of state (ini_parms.set_ref_state_eos; lane M4OFF: the
    LINEAR EOS of offline_exf_seaice)."""
    rp = RunParams(exp.run)
    return rp.get("data", "PARM01", "surf_pRef", default=fortran_default("model/src/set_defaults.F:103", "surf_pRef",
                                                                         exp))


def params_tau(exp, name):
    """tauThetaClimRelax / tauSaltClimRelax of `data` (PARM03, else set_defaults.F:329-330 `0. _d 0`)."""
    rp = RunParams(exp.run)
    line = {"tauThetaClimRelax": 329, "tauSaltClimRelax": 330}[name]
    return rp.get("data", "PARM03", name, default=fortran_default(f"model/src/set_defaults.F:{line}", name, exp))


def params_celsius2K(exp):
    """PARAMS.h celsius2K of `data` (PARM01, else its set_defaults.F default)."""
    rp = RunParams(exp.run)
    return rp.get("data", "PARM01", "celsius2K", default=fortran_default(f"model/src/set_defaults.F:{_C2K}",
                                                                         "celsius2K", exp))
