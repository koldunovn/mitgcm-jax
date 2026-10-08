"""AUTODIFF_READPARMS: pkg/autodiff/autodiff_readparms.F @63cdc0b (lane M4ADCOL: the adjoint-mode switches).

TAF's adjoint sweep runs ADAUTODIFF_INADMODE_SET (pkg/autodiff/autodiff_inadmode_set_ad.F:53-80) before the
backward pass of every step: inAdMode = .TRUE., useKPP = useKPPinAdMode, useGMRedi = useGMRediInAdMode, useSEAICE =
useSEAICEinAdMode, useGGL90 = useGGL90inAdMode, useSALT_PLUME = useSALT_PLUMEinAdMode, the SEAICE FREEDRIFT /
DYNAMICS switches, SEAICEadjMODE = SEAICEapproxLevInAd, SINegFac = SIregFacInAd (if set), viscFacAdj = viscFacInAd.
The forward routine AUTODIFF_INADMODE_SET (autodiff_inadmode_set.F) does nothing. So TAF's derivative is the
derivative of the forward model only when every one of these equals its forward value. JAX differentiates the
forward code: `adjoint_mode_check` reads the run's data.autodiff and raises when the adjoint mode would differ (a
backward-only switch for that setting is not ported; plan decision 11), so a gradient compared with TAF is always
one TAF computes with the forward's settings.
"""

from mitjax.params_io import RunParams

_F, _G = "data.autodiff", "AUTODIFF_PARM01"
_RP = "pkg/autodiff/autodiff_readparms.F"
UNSET_RL = 1.234567e5           # EEPARAMS.h:84  PARAMETER ( UNSET_RL = 1.234567D5 )


def autodiff_adjoint_mode(exp):
    """The AUTODIFF_PARM01 values of the adjoint mode with their defaults (:79-98) and the derived values (:144-160):
    dict name -> value (host)."""
    cfg = exp.cfg
    rp = RunParams(exp.run)

    def get(name, default):
        return rp.get(_F, _G, name) if rp.has(_F, _G, name) else default
    use = dict(cfg.use)
    v = {"useKPPinAdMode": bool(get("useKPPinAdMode", True)),                  # :82 .TRUE.
         "useGMRediInAdMode": bool(get("useGMRediInAdMode", True)),            # :83 .TRUE.
         "useSEAICEinAdMode": bool(get("useSEAICEinAdMode", True)),            # :84 .TRUE.
         "useGGL90inAdMode": bool(get("useGGL90inAdMode", True)),              # :85 .TRUE.
         "useSALT_PLUMEinAdMode": bool(get("useSALT_PLUMEinAdMode", True)),    # :86 .TRUE.
         "inAdExact": bool(get("inAdExact", True)),                            # :88 .TRUE.
         "useApproxAdvectionInAdMode": bool(get("useApproxAdvectionInAdMode", False)),   # :89 .FALSE.
         "SEAICEapproxLevInAd": int(get("SEAICEapproxLevInAd", 0)),            # :90 0
         "viscFacInFw": float(get("viscFacInFw", 1.0)),                        # :91 1. _d 0
         "viscFacInAd": float(get("viscFacInAd", 1.0)),                        # :92 1. _d 0
         "SIregFacInAd": float(get("SIregFacInAd", UNSET_RL)),                 # :93 UNSET_RL
         "SIregFacInFw": float(get("SIregFacInFw", UNSET_RL)),                 # :94 UNSET_RL
         "SEAICEuseFREEDRIFTswitchInAd": bool(get("SEAICEuseFREEDRIFTswitchInAd", False)),   # :97 .FALSE.
         "SEAICEuseDYNAMICSswitchInAd": bool(get("SEAICEuseDYNAMICSswitchInAd", False))}     # :98 .FALSE.
    for n, u in (("useKPPinAdMode", "useKPP"), ("useGMRediInAdMode", "useGMRedi"),       # :145-149
                 ("useSEAICEinAdMode", "useSEAICE"), ("useGGL90inAdMode", "useGGL90"),
                 ("useSALT_PLUMEinAdMode", "useSALT_PLUME")):
        v[n] = v[n] and bool(use.get(u, False))
    if use.get("useSEAICE", False) and not v["useSEAICEinAdMode"]:             # :156-157
        v["SEAICEapproxLevInAd"] = min(v["SEAICEapproxLevInAd"], 0)   # MINMAX-INT: INTEGER (no tie or NaN case)
    if v["useSEAICEinAdMode"]:                                                 # :158-159
        v["SEAICEapproxLevInAd"] = max(v["SEAICEapproxLevInAd"], 0)   # MINMAX-INT: INTEGER (no tie or NaN case)
    return v


def adjoint_mode_check(exp):
    """Raise unless TAF's adjoint mode of this run equals its forward (module docstring). Returns the values."""
    cfg = exp.cfg
    if not (cfg.cpp.flag("ALLOW_AUTODIFF") and dict(cfg.use).get("useAUTODIFF", False)):
        return None
    v = autodiff_adjoint_mode(exp)
    use = dict(cfg.use)
    diff = []
    for n, u in (("useKPPinAdMode", "useKPP"), ("useGMRediInAdMode", "useGMRedi"),
                 ("useSEAICEinAdMode", "useSEAICE"), ("useGGL90inAdMode", "useGGL90"),
                 ("useSALT_PLUMEinAdMode", "useSALT_PLUME")):
        if v[n] != bool(use.get(u, False)):
            diff.append(f"{n} = {v[n]} with {u} = {use.get(u, False)}")
    if not v["inAdExact"]:
        diff.append("inAdExact = .FALSE. (KPP RI_IWMIX kpp_routines.F:1198, ...)")
    if v["useApproxAdvectionInAdMode"]:
        diff.append("useApproxAdvectionInAdMode = .TRUE.")
    if v["SEAICEapproxLevInAd"] != 0:          # forward SEAICEadjMODE = 0 (seaice_readparms.F:249)
        diff.append(f"SEAICEapproxLevInAd = {v['SEAICEapproxLevInAd']}")
    if v["viscFacInAd"] != v["viscFacInFw"]:
        diff.append("viscFacInAd /= viscFacInFw")
    if v["viscFacInFw"] != 1.0:     # :152 viscFacAdj = viscFacInFw (forward MOM_CALC_VISC :511-672; set_defaults 1.)
        raise NotImplementedError("AUTODIFF: viscFacInFw /= 1 (viscFacAdj of the forward) is not ported")
    if v["SIregFacInAd"] != UNSET_RL or v["SIregFacInFw"] != UNSET_RL:
        diff.append("SIregFacInAd / SIregFacInFw set")
    if v["SEAICEuseFREEDRIFTswitchInAd"] or v["SEAICEuseDYNAMICSswitchInAd"]:
        diff.append("SEAICEuseFREEDRIFTswitchInAd / SEAICEuseDYNAMICSswitchInAd")
    if diff:
        raise NotImplementedError(f"AUTODIFF: TAF's adjoint mode differs from the forward ({'; '.join(diff)}); "
                                  "the backward-only switch is not ported")
    return v
