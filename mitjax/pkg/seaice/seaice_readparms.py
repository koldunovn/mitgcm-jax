"""SEAICE_READPARMS: pkg/seaice/seaice_readparms.F @63cdc0b."""

import numpy as np

from mitjax.model.grid import UNSET_RL
from mitjax.params_io import RunParams, fortran_default
from mitjax.pkg.seaice.seaice_params_h import ONE, SITR_REAL_ARRAYS, SItrMaxNum, SeaiceParams, ZERO, nITD, siEps

_RP = "pkg/seaice/seaice_readparms.F"
_F = "data.seaice"
_G = "SEAICE_PARM01"
UNSET_I = 123456789            # EEPARAMS.h:85  PARAMETER ( UNSET_I = 123456789 )

# seaice_readparms.F line of the literal default of each SEAICE_PARM01 variable this port carries (:244-560, before
# the READ at :680); the variables whose default is not a literal are set as code in seaice_readparms() below
_LINES = {
    "SEAICEuseDYNAMICS": 245, "SEAICEuseFREEDRIFT": 250, "SEAICEuseTilt": 252, "SEAICEheatConsFix": 253,
    "SEAICEuseMetricTerms": 259, "SEAICErestoreUnderIce": 264, "SEAICE_growMeltByConv": 265,
    "SEAICE_salinityTracer": 266, "useHB87stressCoupling": 268, "SEAICEupdateOceanStress": 269,
    "SEAICEscaleSurfStress": 270, "SEAICEaddSnowMass": 271, "usePW79thermodynamics": 272,
    "useMaykutSatVapPoly": 304, "SEAICEadvHeff": 306, "SEAICEadvArea": 307, "SEAICEadvSnow": 308,
    "SEAICEmomAdvection": 314, "SEAICEuseFlooding": 321, "SEAICEuseKrylov": 343,
    "SEAICEuseJFNK": 346, "SEAICE_zetaMin": 370, "SEAICE_zetaMaxFac": 371,
    "SEAICE_rhoIce": 400, "SEAICE_rhoSnow": 401, "SEAICE_drag": 403, "OCEAN_drag": 404, "SEAICE_waterDrag": 405,
    "SEAICE_dryIceAlb": 408, "SEAICE_wetIceAlb": 409, "SEAICE_drySnowAlb": 410, "SEAICE_wetSnowAlb": 411, "HO": 412,
    "SEAICE_wetAlbTemp": 427, "SEAICE_strength": 433, "SEAICE_cStar": 434, "SEAICE_eccen": 436,
    "SEAICE_dalton": 443, "SEAICE_iceConduct": 470, "SEAICE_snowConduct": 471, "SEAICE_snowThick": 472,
    "SEAICE_shortwave": 473, "SEAICE_salt0": 474, "SEAICE_saltFrac": 475, "SEAICE_multDim": 489,
    "SEAICE_useMultDimSnow": 494, "SEAICE_mcPheeStepFunc": 496, "SEAICE_doOpenWaterGrowth": 505,
    "SEAICE_doOpenWaterMelt": 506, "SEAICE_areaLossFormula": 507, "SEAICE_areaGainFormula": 508,
    "SEAICE_tempFrz0": 509, "SEAICE_dTempFrz_dS": 510, "AreaFile": 516, "HsnowFile": 517, "HsaltFile": 518,
    "HeffFile": 519, "uIceFile": 520, "vIceFile": 521, "IMAX_TICE": 524, "postSolvTempIter": 525,
    "LSR_ERROR": 541, "SEAICE_hice_reg": 546, "SEAICE_area_max": 547, "SEAICE_airTurnAngle": 549,
    "SEAICE_waterTurnAngle": 550, "MIN_ATEMP": 551, "MIN_LWDOWN": 552, "MIN_TICE": 553, "SEAICE_EPS": 555,
    "SEAICE_EPS_SQ": 556, "SEAICEwriteState": 558,
    # lane M4OFF (offline_exf_seaice, C-grid): advection and the (unused) solver settings of data.seaice
    "SEAICEuseFluxForm": 305, "SEAICE_clipVelocities": 324, "SEAICEadvScheme": 327, "SEAICE_no_slip": 322,
    "SEAICE_tensilFac": 438, "SEAICE_tensilDepth": 439, "SEAICEstressFactor": 514,
    # lane M4OFF session 3 (input.dyn_lsr): the C-grid dynamics with the LSR solver
    "SEAICEuseStrImpCpl": 251, "SEAICEuseTEM": 254, "SEAICEuseMCS": 255, "SEAICEuseMCE": 256, "SEAICEuseTD": 257,
    "SEAICEusePL": 258, "SEAICE_2ndOrderBC": 323, "SEAICE_maskRHS": 325, "SEAICEetaZmethod": 326,
    "SEAICEuseLSRflex": 344, "SEAICEpresH0": 372, "SEAICEpresPow0": 373, "SEAICEpresPow1": 374,
    "SEAICEsideDrag": 406, "SEAICEdWatMin": 407, "SEAICEpressReplFac": 435, "SEAICEmcMu": 441,
    "SEAICEusePicardAsPrecon": 528, "SEAICE_LSRrelaxU": 529, "SEAICE_LSRrelaxV": 530, "SOLV_NCHECK": 531,
    "SEAICEpreconNL_Iter": 534, "SEAICEpreconLinIter": 535, "SEAICEuseMultiTileSolver": 542,
    # lane M4LAB session 3 (lab_sea, SEAICE_ALLOW_BOTTOMDRAG): the switch of SEAICE_BOTTOMDRAG_COEFFS (:80)
    "SEAICEbasalDragK2": 424,
}
# variables read only to refuse what is not ported (not carried in SeaiceParams; lane M4ADLAB session 3)
_CHECK_ONLY = {"SEAICE_tauAreaObsRelax": 515}
# variables whose default is UNSET_I (lane M4OFF)
_UNSET_I = {"SEAICEadvSchArea": 328, "SEAICEadvSchHeff": 329, "SEAICEadvSchSnow": 330, "SEAICEadvSchSalt": 331,
            "SEAICElinearIterMax": 533, "SEAICEnonLinIterMax": 532}
# variables whose default is UNSET_RL (a code default: UNSET_RL is a PARAMETER, not a literal)
_UNSET_RL = {"ICE2WATR": 402, "SEAICE_drag_south": 413, "SEAICE_waterDrag_south": 414,
             "SEAICE_dryIceAlb_south": 415, "SEAICE_wetIceAlb_south": 416, "SEAICE_drySnowAlb_south": 417,
             "SEAICE_wetSnowAlb_south": 418, "HO_south": 419, "SEAICE_cBasalStar": 421, "SEAICE_eccfr": 437,
             "SEAICE_mcPheeTaper": 497, "SEAICE_availHeatTaper": 498, "SEAICE_mcPheePiston": 499,
             "SEAICE_frazilFrac": 500, "SEAICE_gamma_t": 501, "SEAICE_gamma_t_frz": 502,
             "SEAICE_availHeatFrac": 503, "SEAICE_availHeatFracFrz": 504, "SEAICE_deltaMin": 554,
             # lane M4OFF
             "SEAICEdiffKhArea": 332, "SEAICEdiffKhHeff": 333, "SEAICEdiffKhSnow": 334, "SEAICEdiffKhSalt": 335,
             "DIFF1": 336}
# the other carried variables of SEAICE_PARM01 (code defaults, below)
_CODE = ("SEAICE_deltaTtherm", "SEAICE_deltaTdyn", "SEAICE_initialHEFF", "SEAICE_rhoAir", "SEAICE_cpAir",
         "SEAICE_lhEvap", "SEAICE_lhFusion", "SEAICE_emissivity", "SEAICE_ice_emiss", "SEAICE_snow_emiss",
         "SEAICE_area_floor", "SEAICE_area_reg", "SEAICE_PDF", "SEAICEselectMetricTerms",
         # lane M4OFF: SEAICEadvSalt (:309-313, SEAICE_VARIABLE_SALINITY), LSR_mixIniGuess (:536-540,
         # SEAICE_ALLOW_FREEDRIFT); SEAICE_monFreq (:559 = monitorFreq) is read by the driver (drivers/run.py)
         "SEAICEadvSalt", "LSR_mixIniGuess", "SEAICE_monFreq",
         # session 3: SEAICE_OLx/OLy (:356-357 = OLx-2, OLy-2); SEAICEnonLinTol has no default in
         # seaice_readparms.F (read only from data.seaice; see seaice_readparms())
         "SEAICE_OLx", "SEAICE_OLy", "SEAICEnonLinTol",
         # lane M4ADCOL: SEAICE_mon_mnc (:563 monitor_mnc under ALLOW_MNC, :566 .FALSE.; read by SEAICE_MONITOR only
         # with useMNC); carried as set in data.seaice, None = the default (resolved in seaice_monitor)
         "SEAICE_mon_mnc")


def seaice_readparms(exp, *, exf, tp, params):
    """SEAICE_READPARMS( myThid )   @63cdc0b pkg/seaice/seaice_readparms.F:3-1560

    C     | o Routine to read in file data.seaice

    `exf` ExfParams (EXF_READPARMS ran first: the useEXF defaults :447-456 read atmrho, atmcp, flamb, flami,
    stefanBoltzmann, ocean/ice/snow_emissivity), `tp` the time parameters (dTtracerLev(1) for the default time steps
    :339-340), `params` PARAMS.h (usingCartesianGrid, :945). Returns SeaiceParams, or None when useSEAICE is .FALSE.
    (:231-239).

    Defaults: the literal of :244-560 read from its line by `fortran_default` (table `_LINES`); UNSET_RL ones
    (`_UNSET_RL`) and the non-literal ones as code with their lines. After the READ (:680) the derived values of
    :711-1063 and :1431-1434 that this build's routines read. Only SEAICE_PARM01 is read here: ALLOW_COST's
    SEAICE_PARM02 (:682-684) is raised on if compiled; ALLOW_SITRACER's SEAICE_PARM03 (lane M4LAB session 4) is read
    by `_sitracer_parm03` (the defaults :647-661, the READ :686-688, the retired IceAgeTrFile :1311-1324) with
    SEAICE_CHECK's SItracer checks (seaice_check.F:263-307). A data.seaice variable
    this port does not carry raises (it would otherwise be silently ignored), as do the options and settings the
    ported routines do not cover (the checks at the end: dynamics, advection, ITD, EVP/JFNK/Krylov, ...).
    SEAICE_waterAlbedo (SEAICE_EXTERNAL_FLUXES: UNSET_RL, :429) and the print-out / STOP checks that change no value
    (:1404-1425, :1455-1498) are not carried; retired parameters raise through the unknown-variable check."""
    cfg = exp.cfg
    if not cfg.use_flag("useSEAICE"):                                          # :231-239
        return None
    cost = cfg.cpp.flag("ALLOW_COST", "SEAICE_OPTIONS.h")                    # lane M4ADCOL: SEAICE_PARM02
    sitracer = cfg.cpp.flag("ALLOW_SITRACER", "SEAICE_OPTIONS.h")              # lane M4LAB session 4
    if cfg.cpp.flag("SEAICE_ITD", "SEAICE_OPTIONS.h"):
        raise NotImplementedError("SEAICE_READPARMS: SEAICE_ITD is not ported")
    rp = RunParams(exp.run)
    known = set(n.lower() for n in list(_LINES) + list(_UNSET_RL) + list(_UNSET_I) + list(_CODE) + list(_CHECK_ONLY))
    for (fname, group, key), var in exp.run.vars.items():
        if fname == _F and group == _G.lower() and exp.run.is_set(fname, group, key) and key not in known:
            raise NotImplementedError(f"SEAICE_READPARMS: data.seaice variable {var.name} is not carried by "
                                      f"this port")
    v = {}
    lines = dict(_LINES)
    if not (cfg.cpp.flag("SEAICE_CGRID", "SEAICE_OPTIONS.h")
            or cfg.cpp.flag("SEAICE_BGRID_DYNAMICS", "SEAICE_OPTIONS.h")):          # :244-248 (lane M4ADCOL:
        lines["SEAICEuseDYNAMICS"] = 247                                       # code_ad compiles no dynamics)
    for name, line in lines.items():
        v[name] = rp.get(_F, _G, name, default=fortran_default(f"{_RP}:{line}", name, exp))
    for name in _UNSET_RL:                                                     # :332-554 `= UNSET_RL`
        v[name] = rp.get(_F, _G, name) if rp.has(_F, _G, name) else UNSET_RL
    for name in _UNSET_I:                                                      # :328-533 `= UNSET_I`
        v[name] = rp.get(_F, _G, name) if rp.has(_F, _G, name) else UNSET_I
    if cfg.cpp.flag("SEAICE_VARIABLE_SALINITY", "SEAICE_OPTIONS.h"):           # :309-313
        v["SEAICEadvSalt"] = rp.get(_F, _G, "SEAICEadvSalt", default=fortran_default(f"{_RP}:310", "SEAICEadvSalt",
                                                                                     exp))
    else:
        v["SEAICEadvSalt"] = rp.get(_F, _G, "SEAICEadvSalt", default=fortran_default(f"{_RP}:312", "SEAICEadvSalt",
                                                                                     exp))
    if cfg.cpp.flag("SEAICE_ALLOW_FREEDRIFT", "SEAICE_OPTIONS.h"):             # :536-540
        v["LSR_mixIniGuess"] = rp.get(_F, _G, "LSR_mixIniGuess", default=fortran_default(f"{_RP}:537",
                                                                                         "LSR_mixIniGuess", exp))
    else:
        v["LSR_mixIniGuess"] = rp.get(_F, _G, "LSR_mixIniGuess", default=fortran_default(f"{_RP}:539",
                                                                                         "LSR_mixIniGuess", exp))

    def code(name, value):
        return rp.get(_F, _G, name) if rp.has(_F, _G, name) else value
    v["SEAICEselectMetricTerms"] = code("SEAICEselectMetricTerms", UNSET_I)   # :260 UNSET_I
    v["SEAICE_deltaTtherm"] = code("SEAICE_deltaTtherm", float(tp.dTtracerLev[0]))   # :339 dTtracerLev(1)
    v["SEAICE_deltaTdyn"] = code("SEAICE_deltaTdyn", float(tp.dTtracerLev[0]))       # :340 dTtracerLev(1)
    v["SEAICE_initialHEFF"] = code("SEAICE_initialHEFF", ZERO)                       # :384 ZERO
    if cfg.use_flag("useEXF"):                                                 # :446-456 (ALLOW_EXF, useEXF)
        v["SEAICE_rhoAir"] = code("SEAICE_rhoAir", float(exf.atmrho))          # :448
        v["SEAICE_cpAir"] = code("SEAICE_cpAir", float(exf.atmcp))             # :449
        v["SEAICE_lhEvap"] = code("SEAICE_lhEvap", float(exf.flamb))           # :450
        v["SEAICE_lhFusion"] = code("SEAICE_lhFusion", float(exf.flami))       # :451
        v["SEAICE_boltzmann"] = 5.670e-8                                       # :452 = stefanBoltzmann (EXF_CONSTANTS.h:46)
        v["SEAICE_emissivity"] = code("SEAICE_emissivity", float(exf.ocean_emissivity))   # :453
        v["SEAICE_ice_emiss"] = code("SEAICE_ice_emiss", float(exf.ice_emissivity))       # :454
        v["SEAICE_snow_emiss"] = code("SEAICE_snow_emiss", float(exf.snow_emissivity))    # :455
    else:
        raise NotImplementedError("SEAICE_READPARMS: useEXF = .FALSE. defaults (:457-469) are not ported")
    v["SEAICE_monFreq"] = code("SEAICE_monFreq", float(params.monitorFreq))   # :559 = monitorFreq (PARAMS.h)
    v["SEAICE_mon_mnc"] = code("SEAICE_mon_mnc", None)                         # :563 / :566 (seaice_monitor)
    v["SEAICE_OLx"] = code("SEAICE_OLx", cfg.size.OLx-2)                       # :356 OLx-2
    v["SEAICE_OLy"] = code("SEAICE_OLy", cfg.size.OLy-2)                       # :357 OLy-2
    # SEAICEnonLinTol (SEAICE_PARAMS.h:557) gets no default in SEAICE_READPARMS (:244-560 set the retired
    # JFNKgamma_nonlin instead, :358): an unset value is the common block's never-written storage, which the
    # oracle build holds as 0. (zero-initialised static storage; SEAICE_SUMMARY prints 0.000000000000000E+00 in
    # lab_sea/results/output.txt:1384-1385) — carried as that value; docs/ISSUES_UPSTREAM.md
    v["SEAICEnonLinTol"] = code("SEAICEnonLinTol", 0.0)
    v["SEAICE_area_floor"] = code("SEAICE_area_floor", siEps)                 # :544 siEPS
    v["SEAICE_area_reg"] = code("SEAICE_area_reg", siEps)                     # :545 siEPS
    n_itd = nITD(cfg)
    pdf = [UNSET_RL]*n_itd                                                     # :490-492 (ndef SEAICE_ITD)
    elems, _ = rp.get_array(_F, _G, "SEAICE_PDF")
    for idx, val in elems.items():
        pdf[idx[0]-1] = val
    # ---- after the READ (:680)
    if v["SEAICE_deltaMin"] == UNSET_RL:                                       # :711
        v["SEAICE_deltaMin"] = v["SEAICE_EPS"]
    if v["SEAICE_eccfr"] == UNSET_RL:                                          # :713
        v["SEAICE_eccfr"] = v["SEAICE_eccen"]
    tmp = float(v["SEAICE_multDim"])                                           # :716
    for l in range(1, v["SEAICE_multDim"] + 1):                                # :717-719
        if pdf[l-1] == UNSET_RL:
            pdf[l-1] = ONE/tmp
    for l in range(v["SEAICE_multDim"] + 1, n_itd + 1):                       # :720-722
        if pdf[l-1] == UNSET_RL:
            pdf[l-1] = 0.0
    v["SEAICE_PDF"] = tuple(float(x) for x in pdf)
    if v["ICE2WATR"] == UNSET_RL:                                              # :724
        v["ICE2WATR"] = v["SEAICE_rhoIce"]*params.recip_rhoConst
    for n, base in (("SEAICE_drag_south", "SEAICE_drag"), ("SEAICE_waterDrag_south", "SEAICE_waterDrag"),
                    ("SEAICE_dryIceAlb_south", "SEAICE_dryIceAlb"), ("SEAICE_wetIceAlb_south", "SEAICE_wetIceAlb"),
                    ("SEAICE_drySnowAlb_south", "SEAICE_drySnowAlb"),
                    ("SEAICE_wetSnowAlb_south", "SEAICE_wetSnowAlb"), ("HO_south", "HO")):   # :725-738
        if v[n] == UNSET_RL:
            v[n] = v[base]
    if v["SEAICE_cBasalStar"] == UNSET_RL:                                     # :740-741
        v["SEAICE_cBasalStar"] = v["SEAICE_cStar"]
    dt1 = float(tp.dTtracerLev[0])                                             # :747-757
    if (v["SEAICE_deltaTtherm"] != dt1 or v["SEAICE_deltaTdyn"] < v["SEAICE_deltaTtherm"]
            or (v["SEAICE_deltaTdyn"]/v["SEAICE_deltaTtherm"]) != int(v["SEAICE_deltaTdyn"]/v["SEAICE_deltaTtherm"])):
        raise RuntimeError("Unsupported combination of SEAICE_deltaTtherm, SEAICE_deltaTdyn, and dTtracerLev(1)\n"
                           "ABNORMAL END: S/R SEAICE_READPARMS")
    v["SEAICEuseEVP"] = False                                                  # :759
    # :760-833 (SEAICE_ALLOW_EVP): SEAICEuseEVP = .TRUE. only if SEAICE_deltaTevp, SEAICE_evpAlpha, SEAICE_evpBeta
    # or SEAICEaEVPcoeff is set in data.seaice (:763-769); this port does not carry them (the unknown-variable check
    # above raises if data.seaice sets one), so SEAICEuseEVP stays .FALSE. (lane M4OFF: compiled, not used)
    if v["SEAICEuseFREEDRIFT"]:                                                # :834 (SEAICE_ALLOW_FREEDRIFT)
        raise NotImplementedError("SEAICE_READPARMS: SEAICEuseFREEDRIFT is not ported")
    # :844-858 (ndef SEAICE_ITD): useHibler79IceStrength = .TRUE. (only printed / read by the C-grid strength)
    v["SEAICEuseLSR"] = (not v["SEAICEuseFREEDRIFT"] and not v["SEAICEuseEVP"]         # :861-862
                         and not v["SEAICEuseJFNK"] and not v["SEAICEuseKrylov"])
    if v["SEAICEuseJFNK"] and v["SEAICEusePicardAsPrecon"] and not v["SEAICEuseKrylov"]:   # :865-866
        v["SEAICEuseLSR"] = True
    if v["SEAICEuseJFNK"] and not v["SEAICEusePicardAsPrecon"]:                # :867-868
        v["SEAICEuseKrylov"] = False
    if v["SEAICEnonLinIterMax"] == UNSET_I:                                    # :871-876
        if v["SEAICEuseLSR"]:
            v["SEAICEnonLinIterMax"] = 2
        if v["SEAICEuseJFNK"] or v["SEAICEuseKrylov"]:
            v["SEAICEnonLinIterMax"] = 10
    if v["SEAICEuseLSR"] and not v["SEAICEusePicardAsPrecon"]:                 # :878-879
        v["SEAICEnonLinIterMax"] = max(v["SEAICEnonLinIterMax"], 2)   # MINMAX-INT: INTEGER MAX (:879)
    if not v["SEAICEuseLSR"]:                                                  # :882
        v["SEAICEuseLSRflex"] = False
    if v["SEAICEselectMetricTerms"] == UNSET_I:                                # :916-933
        v["SEAICEselectMetricTerms"] = 2 if v["SEAICEuseMetricTerms"] else 0
    elif not v["SEAICEuseMetricTerms"] and v["SEAICEselectMetricTerms"] > 0:
        raise RuntimeError("Cannot set both: SEAICEuseMetricTerms= False AND SEAICEselectMetricTerms > 0")
    if v["SEAICEselectMetricTerms"] < 0 or v["SEAICEselectMetricTerms"] > 2:  # :934-943
        raise RuntimeError("SEAICEselectMetricTerms needs to be >=0 and <= 2")
    if params.usingCartesianGrid:                                              # :945
        v["SEAICEselectMetricTerms"] = 0
    if v["SEAICE_mcPheeTaper"] == UNSET_RL:                                    # :950-961
        if v["SEAICE_availHeatTaper"] == UNSET_RL:
            v["SEAICE_mcPheeTaper"] = 0.0
        else:
            v["SEAICE_mcPheeTaper"] = v["SEAICE_availHeatTaper"]
    elif v["SEAICE_availHeatTaper"] != UNSET_RL:
        raise RuntimeError("both SEAICE_mcPheeTaper & SEAICE_availHeatTaper are set")
    if v["SEAICE_gamma_t_frz"] != UNSET_RL:                                    # :964-973
        if v["SEAICE_frazilFrac"] == UNSET_RL:
            v["SEAICE_frazilFrac"] = v["SEAICE_deltaTtherm"]/v["SEAICE_gamma_t_frz"]
        else:
            raise RuntimeError("both SEAICE_frazilFrac & SEAICE_gamma_t_frz are set")
    if v["SEAICE_availHeatFracFrz"] != UNSET_RL:                               # :974-988
        if v["SEAICE_frazilFrac"] == UNSET_RL:
            v["SEAICE_frazilFrac"] = v["SEAICE_availHeatFracFrz"]
        else:
            raise RuntimeError("both SEAICE_frazilFrac / SEAICE_gamma_t_frz & SEAICE_availHeatFracFrz are set")
    if v["SEAICE_gamma_t"] != UNSET_RL and v["SEAICE_frazilFrac"] == UNSET_RL:   # :990-993
        v["SEAICE_frazilFrac"] = v["SEAICE_deltaTtherm"]/v["SEAICE_gamma_t"]
    if v["SEAICE_availHeatFrac"] != UNSET_RL and v["SEAICE_frazilFrac"] == UNSET_RL:   # :995-998
        v["SEAICE_frazilFrac"] = v["SEAICE_availHeatFrac"]
    if v["SEAICE_frazilFrac"] == UNSET_RL:                                     # :999-1001
        v["SEAICE_frazilFrac"] = 1.0
    if v["SEAICE_gamma_t"] != UNSET_RL:                                        # :1005-1014
        if v["SEAICE_availHeatFrac"] == UNSET_RL:
            v["SEAICE_availHeatFrac"] = v["SEAICE_deltaTtherm"]/v["SEAICE_gamma_t"]
        else:
            raise RuntimeError("both SEAICE_gamma_t & SEAICE_availHeatFrac are set")
    if v["SEAICE_mcPheePiston"] != UNSET_RL and v["SEAICE_availHeatFrac"] != UNSET_RL:   # :1015-1026
        raise RuntimeError("both SEAICE_mcPheePiston & SEAICE_availHeatFrac / SEAICE_gamma_t are set")
    if v["SEAICE_clipVelocities"] and not cfg.cpp.flag("SEAICE_ALLOW_CLIPVELS", "SEAICE_OPTIONS.h"):
        raise RuntimeError("SEAICE_CHECK: SEAICE_clipVelocities = .TRUE.\nSEAICE_CHECK: but "      # seaice_check.F:944-954
                           "SEAICE_ALLOW_CLIPVELS is not defined in SEAICE_OPTIONS.h\nABNORMAL END: S/R SEAICE_CHECK")
    if cfg.use_flag("useThSIce"):                                              # :1028-1037
        raise NotImplementedError("SEAICE_READPARMS: useThSIce is not ported")
    if v["SEAICElinearIterMax"] == UNSET_I:                                    # :885-896 (dynamics only)
        if cfg.cpp.flag("ALLOW_AUTODIFF"):                                    # lane M4ADCOL
            v["SEAICElinearIterMax"] = 500         # :889 SOLV_MAX_FIXED (SEAICE_SIZE.h:37 PARAMETER = 500)
        else:
            v["SEAICElinearIterMax"] = 1500                                    # :891 (no ALLOW_AUTODIFF)
        if v["SEAICEuseJFNK"] or v["SEAICEuseKrylov"]:                         # :895
            v["SEAICElinearIterMax"] = 10
    if not v["SEAICE_no_slip"]:                                                # :906
        v["SEAICE_2ndOrderBC"] = False
    if v["SEAICEuseLSR"]:                                                      # :907
        v["SEAICE_2ndOrderBC"] = False
    if v["SEAICE_2ndOrderBC"]:                                                 # :911-914
        v["SEAICE_OLx"] = cfg.size.OLx-3
        v["SEAICE_OLy"] = cfg.size.OLy-3
    if v["SEAICEadvSchArea"] == UNSET_I:                                       # :1039-1050
        v["SEAICEadvSchArea"] = v["SEAICEadvSchHeff"]
    if v["SEAICEadvSchArea"] == UNSET_I:
        v["SEAICEadvSchArea"] = v["SEAICEadvScheme"]
    if v["SEAICEadvScheme"] != v["SEAICEadvSchArea"]:
        v["SEAICEadvScheme"] = v["SEAICEadvSchArea"]
    if v["SEAICEadvSchHeff"] == UNSET_I:
        v["SEAICEadvSchHeff"] = v["SEAICEadvSchArea"]
    if v["SEAICEadvSchSnow"] == UNSET_I:
        v["SEAICEadvSchSnow"] = v["SEAICEadvSchHeff"]
    if v["SEAICEadvSchSalt"] == UNSET_I:
        v["SEAICEadvSchSalt"] = v["SEAICEadvSchHeff"]
    if v["SEAICEdiffKhArea"] == UNSET_RL:                                      # :1052-1061
        v["SEAICEdiffKhArea"] = v["SEAICEdiffKhHeff"]
    if v["SEAICEdiffKhArea"] == UNSET_RL:
        v["SEAICEdiffKhArea"] = 0.0
    if v["SEAICEdiffKhHeff"] == UNSET_RL:
        v["SEAICEdiffKhHeff"] = v["SEAICEdiffKhArea"]
    if v["SEAICEdiffKhSnow"] == UNSET_RL:
        v["SEAICEdiffKhSnow"] = v["SEAICEdiffKhHeff"]
    if v["SEAICEdiffKhSalt"] == UNSET_RL:
        v["SEAICEdiffKhSalt"] = v["SEAICEdiffKhHeff"]
    if v["SEAICE_EPS_SQ"] == -99999.:                                          # :1062-1063
        v["SEAICE_EPS_SQ"] = v["SEAICE_EPS"] * v["SEAICE_EPS"]
    if cfg.cpp.flag("ALLOW_GENERIC_ADVDIFF"):                                  # :1065-1074
        # ENUM_CENTERED_2ND = 2, ENUM_UPWIND_3RD = 3, ENUM_CENTERED_4TH = 4 (generic_advdiff/GAD.h:25, :29, :33)
        v["SEAICEmultiDimAdvection"] = v["SEAICEadvScheme"] not in (2, 3, 4)
    else:
        v["SEAICEmultiDimAdvection"] = False
    if v["DIFF1"] == UNSET_RL:                                                 # :1511-1512
        v["DIFF1"] = 0.0
    v["facOpenGrow"] = 0.0                                                     # :1431
    v["facOpenMelt"] = 0.0                                                     # :1432
    if v["SEAICE_doOpenWaterGrowth"]:                                          # :1433
        v["facOpenGrow"] = 1.0
    if v["SEAICE_doOpenWaterMelt"]:                                            # :1434
        v["facOpenMelt"] = 1.0
    # ---- what the ported routines of this build do not cover
    if v["SEAICEuseDYNAMICS"] and not cfg.cpp.flag("SEAICE_CGRID", "SEAICE_OPTIONS.h"):
        raise NotImplementedError("SEAICE: SEAICEuseDYNAMICS of the B-grid build (DYNSOLVER's LSR) is not ported")
    if v["SEAICEuseDYNAMICS"]:
        # lane M4OFF session 3: the C-grid LSR of offline_exf_seaice/input.dyn_lsr (SEAICE_LSR with
        # SEAICEuseLSRflex); the solvers and arms that run does not execute raise
        for n in ("SEAICEuseKrylov", "SEAICEuseJFNK", "SEAICEuseEVP", "SEAICEuseFREEDRIFT",
                  "SEAICEusePicardAsPrecon", "SEAICEuseMultiTileSolver", "SEAICE_maskRHS", "SEAICEmomAdvection",
                  "SEAICEuseTD", "SEAICEusePL", "SEAICEuseMCS", "SEAICEuseMCE", "SEAICEuseTEM"):
            if v[n]:
                raise NotImplementedError(f"SEAICE: SEAICEuseDYNAMICS with {n} is not ported (lane M4OFF: LSR only)")
        if not v["SEAICEuseLSR"]:
            raise NotImplementedError("SEAICE: SEAICEuseDYNAMICS is ported for SEAICE_LSR only (with "
                                      "SEAICEuseLSRflex, or lane M4LAB session 3 the SOLV_NCHECK max-norm arm, "
                                      "seaice_lsr.F:934-983)")
        if v["LSR_mixIniGuess"] >= 2:                                          # -1: the default without FREEDRIFT
            raise NotImplementedError("SEAICE_LSR: LSR_mixIniGuess >= 2 (the residual mix seaice_lsr.F:646-689) "
                                      "is not ported (0: SEAICE_FREEDRIFT, lane M4LAB session 3)")
        if v["SEAICEetaZmethod"] not in (0, 3) or v["SEAICEsideDrag"] != 0.0:
            raise NotImplementedError("SEAICE: SEAICEetaZmethod not in (0, 3) / SEAICEsideDrag /= 0 not ported")
    if (v["SEAICEadvHeff"] or v["SEAICEadvArea"] or v["SEAICEadvSnow"] or v["SEAICEadvSalt"]) and not (
            v["SEAICEuseFluxForm"] and v["DIFF1"] == 0.0 and not v["SEAICEadvSalt"]
            and (not v["SEAICEmultiDimAdvection"]       # lane M4ADCOL: ADVECT (seaice_advdiff.F:579-656, scheme 2)
                 or all(v[n] in (77, 41, 7, 30, 33)        # 30 / 33: lane M4ADCS32ICE (DST3, DST3FL)
                        for n in ("SEAICEadvSchArea", "SEAICEadvSchHeff", "SEAICEadvSchSnow")))
            and (not v["SEAICEmultiDimAdvection"]       # lane M4ADLAB: SEAICE_DIFFUSION after ADVECT (:584-634)
                 or all(v[n] == 0.0 for n in ("SEAICEdiffKhArea", "SEAICEdiffKhHeff", "SEAICEdiffKhSnow")))):
        raise NotImplementedError("SEAICE: SEAICE_ADVDIFF is ported multi-dimensional with scheme 77, 41, 7, 30 or 33 "
                                  "(lane M4OFF) without diffusion, or not multi-dimensional (ADVECT, lane M4ADCOL; "
                                  "SEAICE_DIFFUSION, lane M4ADLAB), flux form, no HSALT advection")
    if cfg.cpp.flag("SEAICE_ALLOW_BOTTOMDRAG", "SEAICE_OPTIONS.h") and v["SEAICEbasalDragK2"] > 0.0:
        raise NotImplementedError("SEAICE_BOTTOMDRAG_COEFFS: SEAICEbasalDragK2 > 0 (seaice_bottomdrag_coeffs.F:81-163) "
                                  "is not ported")
    # SEAICE_tauAreaObsRelax (:515 -999. _d 0): the ice-area relaxation towards exf_iceFraction (seaice_reg_ridge.F:
    # 120-148, EXF_SEAICE_FRACTION) runs only with > 0: not ported; SEAICE_CHECK stops on > 0 without
    # EXF_SEAICE_FRACTION (seaice_check.F:775-781)
    tau = rp.get(_F, _G, "SEAICE_tauAreaObsRelax", default=fortran_default(f"{_RP}:515", "SEAICE_tauAreaObsRelax",
                                                                           exp))
    if tau > 0.0:
        raise NotImplementedError("SEAICE_REG_RIDGE: SEAICE_tauAreaObsRelax > 0 (the ice-area relaxation "
                                  "seaice_reg_ridge.F:122-147; SEAICE_CHECK :775-781) is not ported")
    # SEAICE_areaGainFormula 1..2 / SEAICE_areaLossFormula 1..3 (seaice_check.F:95-107): ported (lane M4ADLAB session
    # 3; formula 3's build-dependent winners are checked by SEAICE_GROWTH, plan decision 15)
    if v["SEAICE_areaLossFormula"] not in (1, 2, 3) or v["SEAICE_areaGainFormula"] not in (1, 2):
        raise NotImplementedError("SEAICE_CHECK: SEAICE_areaGainFormula / SEAICE_areaLossFormula out of range "
                                  "(seaice_check.F:95-107 stops)")
    # useMaykutSatVapPoly and postSolvTempIter = 0: ported (lane M4ADLAB session 2, seaice_solve4temp.py)
    if v["postSolvTempIter"] not in (0, 2):
        raise NotImplementedError("SEAICE_SOLVE4TEMP: postSolvTempIter = 1 is not ported")
    # SEAICE_multDim > 1 (lane M4LAB session 3): the non-ITD category loops of SEAICE_GROWTH are ported (lab_sea's 7)
    for f in ("uIceFile", "vIceFile"):
        if v[f].strip():
            raise NotImplementedError(f"SEAICE_INIT_VARIA: {f} is not ported")
    if cfg.cpp.flag("ALLOW_AUTODIFF") and cfg.cpp.flag("ALLOW_AUTODIFF_TAMC", "AUTODIFF_OPTIONS.h"):
        # lane M4ADCS32ICE: not SEAICE_READPARMS's -- the build's tamc.h PARAMETERs maxpass / maxcube, which
        # SEAICE_ADVECTION's STOPs read (seaice_advection.F:164-173, :245-251); carried here as static values
        v.update(_tamc_maxpass_maxcube(exp))
    if sitracer:                                                               # :647-661, :686-688
        v.update(_sitracer_parm03(exp, rp))
    if cost:                                                                   # :615-645, :682-684
        v.update(_cost_parm02(exp, rp))
    r = {k: np.float64(x) for k, x in v.items() if isinstance(x, float) and not isinstance(x, bool)}
    for k in SITR_REAL_ARRAYS:                 # SEAICE_TRACER.h:35-37 REAL arrays: traced, as SEAICE_PDF (session 4)
        if k in v:
            r[k] = np.asarray(v[k], np.float64)
    # SEAICE_PDF (REAL array, SEAICE_PARAMS.h) is traced too (lane M4LAB session 3): with SEAICE_multDim 7 its seven
    # equal static values let XLA rewrite SEAICE_GROWTH's category sums (:879-884), measured: Qsw off on 8 points
    r["SEAICE_PDF"] = np.asarray(v["SEAICE_PDF"], np.float64)
    s = tuple(sorted((k, x) for k, x in v.items() if k not in r))
    return SeaiceParams(r=r, s=s)


# SEAICE_PARM02 (seaice_readparms.F:207-219, #ifdef ALLOW_COST): every variable of the namelist (lane M4ADCOL)
_PARM02 = ("mult_ice_export", "mult_ice", "cost_ice_flag", "costIceStart1", "costIceStart2", "costIceEnd1",
           "costIceEnd2", "SEAICE_cutoff_area", "SEAICE_cutoff_heff", "SEAICE_clamp_salt", "SEAICE_clamp_theta",
           "mult_smrsst", "smrsstbarfile", "mult_smrsss", "smrsssbarfile", "mult_smrarea", "smrareabarfile",
           "smrareadatfile", "smrarea_errfile", "wsmrarea0", "wmean_smrarea", "smrareastartdate1",
           "smrareastartdate2")


def _cost_parm02(exp, rp):
    """SEAICE_READPARMS' ALLOW_COST part (lane M4ADCOL): the defaults of :615-645, the READ of SEAICE_PARM02
    (:682-684) and the retired-parameter checks (:1355-1384, each an error, then the STOP of :1422-1425).
    Returns {name: value} of the values the ported cost routines read: mult_ice_export, mult_ice (REAL,
    SEAICE_COST.h:16-17, traced), cost_ice_flag (INTEGER, :41-42), SEAICE_cutoff_area / _heff (read only by
    pkg/ecco's COST_GENCOST_SEAICEV4, carried as set). costIceStart1/2, costIceEnd1/2 are read by
    SEAICE_COST_INIT_FIXED only, which this build does not call (coverage of input_ad: not executed); their default
    (:616-620, :623-626: CAL_GETDATE of startTime) is not carried: an unset value raises only if a reader is ported."""
    g2 = "SEAICE_PARM02"
    known = set(n.lower() for n in _PARM02)
    for (fname, group, key), var in exp.run.vars.items():
        if fname == _F and group == g2.lower() and exp.run.is_set(fname, group, key) and key not in known:
            raise NotImplementedError(f"SEAICE_READPARMS: data.seaice variable {var.name} (SEAICE_PARM02) is not "
                                      f"carried by this port")

    def get(name, default):
        return rp.get(_F, g2, name) if rp.has(_F, g2, name) else default
    v = {"mult_ice_export": float(get("mult_ice_export", 0.0)),                 # :621 0. _d 0
         "mult_ice": float(get("mult_ice", 0.0)),                               # :622 0. _d 0
         "cost_ice_flag": int(get("cost_ice_flag", 1)),                         # :627 1
         "SEAICE_cutoff_area": float(get("SEAICE_cutoff_area", 0.0001)),        # :628 0.0001 _d 0
         "SEAICE_cutoff_heff": float(get("SEAICE_cutoff_heff", 0.0))}           # :629 0. _d 0
    retired = [n for n in ("SEAICE_clamp_salt", "SEAICE_clamp_theta", "mult_smrsst", "smrsstbarfile", "mult_smrsss",
                           "smrsssbarfile", "mult_smrarea", "smrareabarfile", "smrareadatfile", "smrarea_errfile",
                           "wsmrarea0", "wmean_smrarea", "smrareastartdate1", "smrareastartdate2")
               if rp.has(_F, g2, n) and str(rp.get(_F, g2, n)).strip()]       # :631-644 UNSET / blank defaults
    if retired:                                                                # :1355-1384, :1422-1425
        raise RuntimeError(f"S/R SEAICE_READPARMS: retired SEAICE_PARM02 parameters set: {retired}\n"
                           "ABNORMAL END: S/R SEAICE_READPARMS")
    return v


# SEAICE_PARM03 (seaice_readparms.F:221-227, #ifdef ALLOW_SITRACER): every variable of the namelist
_PARM03 = ("SItrFile", "SItrName", "SItrNameLong", "SItrUnit", "SItrMate", "SItrFromOcean0", "SItrFromOceanFrac",
           "SItrFromFlood0", "SItrFromFloodFrac", "SItrExpand0", "IceAgeTrFile", "SItrNumInUse")


def _tamc_maxpass_maxcube(exp):
    """{mjx_tamc_maxpass, mjx_tamc_maxcube}: PARAMETER( maxpass = .. ) and PARAMETER( maxcube = .. ) of the build's
    tamc.h (pkg/autodiff/tamc.h:85-93 or the experiment's override, e.g. global_ocean.cs32x15/code_ad/tamc.h:76-82),
    preprocessed after the option headers a .F file includes before it (its declarations sit inside #ifdef
    ALLOW_AUTODIFF_TAMC). maxpass is declared there only #ifndef ALLOW_PTRACERS (PTRACERS_SIZE.h otherwise: not read,
    {} returned, and SEAICE_ADVECTION raises)."""
    from mitjax.config import cpp_options, fortran
    if exp.cfg.cpp.ALLOW_PTRACERS:            # not read here: SEAICE_ADVECTION refuses such a build (no value)
        return {}
    text = cpp_options.preprocess_text(exp.farm, exp.toolchain, '#include "PACKAGES_CONFIG.h"\n#include "CPP_OPTIONS.h"'
                                       '\n#include "AUTODIFF_OPTIONS.h"\n#include "tamc.h"\n')
    env = fortran.parameters(fortran.statements(text, "tamc.h"), exp.cfg.size.env(), source="tamc.h")
    return {"mjx_tamc_maxpass": int(env["maxpass"]), "mjx_tamc_maxcube": int(env["maxcube"])}


def _sitracer_parm03(exp, rp):
    """SEAICE_READPARMS' ALLOW_SITRACER part (lane M4LAB session 4): the defaults of :647-661, the READ of
    SEAICE_PARM03 (:686-688), the retired IceAgeTrFile (:602-605 ' ', :1311-1324: an error, then the STOP of
    :1422-1425), and SEAICE_CHECK's SItracer specification checks (seaice_check.F:263-307, run before any time
    step: PRINT_ERROR + errCount, then its STOP). Returns {name: value}: SItrNumInUse (INTEGER), SItrFile, SItrName,
    SItrNameLong, SItrUnit (CHARACTER*(MAX_LEN_FNAM), SEAICE_TRACER.h:47-50) and SItrMate (CHARACTER*(4), :51) as
    tuples of SItrMaxNum strings without trailing blanks (Fortran .EQ. pads the shorter operand with blanks, so
    `SItrName(iTr).EQ.'age'` is `== "age"` of the stripped value; a longer SItrMate is cut to 4 characters), the
    five REAL arrays (:35-37) as tuples of float. SEAICE_INIT_FIXED (:95-144) overwrites some of them per tracer
    name (seaice_init_fixed.py)."""
    g3 = "SEAICE_PARM03"
    known = set(n.lower() for n in _PARM03)
    for (fname, group, key), var in exp.run.vars.items():
        if fname == _F and group == g3.lower() and exp.run.is_set(fname, group, key) and key not in known:
            raise NotImplementedError(f"SEAICE_READPARMS: data.seaice variable {var.name} (SEAICE_PARM03) is not "
                                      f"carried by this port")

    def arr(name, default, conv):
        vals = [default]*SItrMaxNum                                            # DO iTracer = 1, SItrMaxNum
        elems, _ = rp.get_array(_F, g3, name)
        for idx, val in elems.items():
            if not 1 <= idx[0] <= SItrMaxNum:
                raise ValueError(f"SEAICE_READPARMS: {name}({idx[0]}) outside 1..SItrMaxNum = {SItrMaxNum}")
            vals[idx[0]-1] = conv(val)
        return tuple(vals)
    s_ = lambda x: str(x).rstrip()                                             # noqa: E731
    v = {"SItrNumInUse": int(rp.get(_F, g3, "SItrNumInUse")) if rp.has(_F, g3, "SItrNumInUse")
         else SItrMaxNum,                                                      # :648 SItrNumInUse=SItrMaxNum
         "SItrFile": arr("SItrFile", "", s_),                                  # :650 ' '
         "SItrName": arr("SItrName", "", s_),                                  # :651 ' '
         "SItrNameLong": arr("SItrNameLong", "", s_),                          # :652 ' '
         "SItrUnit": arr("SItrUnit", "", s_),                                  # :653 ' '
         "SItrMate": arr("SItrMate", "HEFF", lambda x: str(x)[:4].rstrip()),   # :654 'HEFF' (CHARACTER*(4))
         "SItrFromOcean0": arr("SItrFromOcean0", ZERO, float),                 # :655 ZERO
         "SItrFromOceanFrac": arr("SItrFromOceanFrac", ZERO, float),           # :656 ZERO
         "SItrFromFlood0": arr("SItrFromFlood0", ZERO, float),                 # :657 ZERO
         "SItrFromFloodFrac": arr("SItrFromFloodFrac", ZERO, float),           # :658 ZERO
         "SItrExpand0": arr("SItrExpand0", ZERO, float)}                       # :659 ZERO
    if not 0 <= v["SItrNumInUse"] <= SItrMaxNum:      # the arrays are SItrMaxNum long (SEAICE_TRACER.h:23-51)
        raise ValueError(f"SEAICE_READPARMS: SItrNumInUse = {v['SItrNumInUse']} outside 0..SItrMaxNum")
    elems, _ = rp.get_array(_F, g3, "IceAgeTrFile")                            # :602-605 ' '
    if any(str(x).strip() for x in elems.values()):                            # :1311-1324, :1422-1425
        raise RuntimeError('S/R SEAICE_READPARMS: "IceAgeTrFile" is no longer allowed in file "data.seaice"\n'
                           "S/R SEAICE_READPARMS: since ALLOW_SITRACER replaced and extended SEAICE_AGE\n"
                           "ABNORMAL END: S/R SEAICE_READPARMS")
    errors = []
    for iTracer in range(1, v["SItrNumInUse"] + 1):                            # seaice_check.F:269-305
        n = iTracer - 1
        if v["SItrFromOceanFrac"][n] < 0.0 or v["SItrFromOceanFrac"][n] > 1.0:   # :271-278
            errors.append("SItrFromOceanFrac cannot be specified  outside of the [0. 1.] range")
        if v["SItrFromFloodFrac"][n] < 0.0 or v["SItrFromFloodFrac"][n] > 1.0:   # :280-287
            errors.append("SItrFromFloodFrac cannot be specified  outside of the [0. 1.] range")
        if v["SItrName"][n] != "salinity" and (v["SItrFromOceanFrac"][n] != ZERO
                                               or v["SItrFromFloodFrac"][n] != ZERO):   # :298-304
            errors.append("SItrFromOceanFrac / SItrFromFloodFrac is only  available for SItrName = \"salinity\" "
                          "(for now)")
    if errors:
        raise RuntimeError("\n".join(errors) + "\nABNORMAL END: S/R SEAICE_CHECK")
    return v
