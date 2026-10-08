"""The run's own adjoint-mode switches (data.autodiff) as backward-only options of the Model's arrays (lane
M4ADCS32ICE session 2; plan decisions 11, 17 revised).

TAF's reverse sweep runs with the AD-mode switches of AUTODIFF_PARM01 (ADAUTODIFF_INADMODE_SET,
autodiff_inadmode_set_ad.F:53-80; mitjax/pkg/autodiff/autodiff_readparms.py). `with_run_switches(m, model)` returns
the Model's arrays with every switch the run sets turned into its backward-only hook, and refuses a switch that has
no hook (so that a gradient called "the run's" is never silently the forward's derivative):
- useApproxAdvectionInAdMode -> params.mjx_approx_advection_in_ad = "routine" (GAD_ADVECTION) and
  sp.mjx_approx_advection_in_ad = `seaice_level` ("flux" by default; mitjax/ad/approx_advection.py);
- SEAICEuseFREEDRIFTswitchInAd -> sp.mjx_freedrift_in_ad = True (mitjax/ad/freedrift_switch.py).
The forward of the returned arrays is the forward of `model` (the hooks change derivatives only). The LSR derivative
(mitjax/ad/modes.lsr_derivative) and the CG2D derivative (cg2dFullAdjoint, mitjax/ad/modes.py) are set separately.
"""

from mitjax.pkg.autodiff.autodiff_readparms import UNSET_RL, autodiff_adjoint_mode


def with_run_switches(m, model, *, seaice_level="flux"):
    """`model` (Arrays of the Model `m`) with the run's data.autodiff switches as backward-only hooks."""
    from mitjax.ad import approx_advection as AA
    from mitjax.ad import freedrift_switch as FD
    exp, cfg = m.exp, m.cfg
    if not (cfg.cpp.flag("ALLOW_AUTODIFF") and dict(cfg.use).get("useAUTODIFF", False)):
        raise ValueError("with_run_switches: the build has no pkg/autodiff (no adjoint mode to follow)")
    v = autodiff_adjoint_mode(exp)
    use = dict(cfg.use)
    bad = []
    for n, u in (("useKPPinAdMode", "useKPP"), ("useGMRediInAdMode", "useGMRedi"),
                 ("useSEAICEinAdMode", "useSEAICE"), ("useGGL90inAdMode", "useGGL90"),
                 ("useSALT_PLUMEinAdMode", "useSALT_PLUME")):
        if v[n] != bool(use.get(u, False)):
            bad.append(f"{n} = {v[n]} with {u} = {use.get(u, False)}")
    if not v["inAdExact"]:
        bad.append("inAdExact = .FALSE.")
    if v["SEAICEapproxLevInAd"] != 0:
        bad.append(f"SEAICEapproxLevInAd = {v['SEAICEapproxLevInAd']}")
    if v["viscFacInAd"] != v["viscFacInFw"]:
        bad.append("viscFacInAd /= viscFacInFw")
    if v["SIregFacInAd"] != UNSET_RL or v["SIregFacInFw"] != UNSET_RL:
        bad.append("SIregFacInAd / SIregFacInFw set")
    if v["SEAICEuseDYNAMICSswitchInAd"]:
        bad.append("SEAICEuseDYNAMICSswitchInAd")
    params, pkc = model.params, dict(model.pkc)
    sp = pkc.get("sp")
    if v["useApproxAdvectionInAdMode"]:
        if not cfg.cpp.flag("ALLOW_AUTODIFF_TAMC", "AUTODIFF_OPTIONS.h"):      # gad_advection.F:193 (#ifdef)
            bad.append("useApproxAdvectionInAdMode without ALLOW_AUTODIFF_TAMC")
        for t in ("temp", "salt"):
            sch = (params.static_items().get(f"{t}AdvScheme"), params.static_items().get(f"{t}VertAdvScheme"))
            multi = params.static_items().get(f"{t}MultiDimAdvec")
            if AA.ENUM_DST3_FLUX_LIMIT in sch and not multi:                   # gad_calc_rhs.F:176-184: no hook
                bad.append(f"useApproxAdvectionInAdMode with {t} scheme 33 outside GAD_ADVECTION (GAD_CALC_RHS)")
            if AA.ENUM_DST3_FLUX_LIMIT in sch and params.static_items().get(f"{t}ImplVertAdv"):
                bad.append(f"useApproxAdvectionInAdMode with {t} implicit vertical advection (not covered)")
        if cfg.cpp.ALLOW_PTRACERS and use.get("usePTRACERS", False):
            bad.append("useApproxAdvectionInAdMode with pTracers (PTRACERS_INTEGRATE: no hook)")
        if seaice_level not in AA.SEAICE_LEVELS[1:]:
            raise ValueError(f"seaice_level {seaice_level!r}: one of {AA.SEAICE_LEVELS[1:]}")
        params = params.replace(static={AA.OPTION: "routine"})
        if sp is not None:
            sp = sp.replace(**{AA.OPTION: seaice_level})
    if v["SEAICEuseFREEDRIFTswitchInAd"]:
        if sp is None:
            bad.append("SEAICEuseFREEDRIFTswitchInAd without pkg/seaice")
        elif not cfg.cpp.flag("SEAICE_ALLOW_FREEDRIFT", "SEAICE_OPTIONS.h"):
            bad.append("SEAICEuseFREEDRIFTswitchInAd without SEAICE_ALLOW_FREEDRIFT")
        elif sp.SEAICEuseFREEDRIFT:                                            # AD mode would run the LSR
            bad.append("SEAICEuseFREEDRIFTswitchInAd with SEAICEuseFREEDRIFT = .TRUE. (LSR in AD mode)")
        elif not (sp.SEAICEuseEVP or sp.LSR_mixIniGuess == 0):                 # seaice_dynsolver.F:304-305
            bad.append("SEAICEuseFREEDRIFTswitchInAd where the forward does not call SEAICE_FREEDRIFT")
        elif sp.SEAICEuseEVP or not sp.SEAICEuseLSR:
            bad.append("SEAICEuseFREEDRIFTswitchInAd with EVP / without the LSR (not covered)")
        else:
            sp = sp.replace(**{FD.OPTION: True})
    if bad:
        raise NotImplementedError("with_run_switches: TAF's adjoint mode of this run has no backward-only hook here: "
                                  + "; ".join(bad))
    if sp is not None:
        pkc["sp"] = sp
    return model.replace(params=params, pkc=pkc)
