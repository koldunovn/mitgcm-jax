"""PACKAGES_WRITE_PICKUP: model/src/packages_write_pickup.F @63cdc0b (host side)."""


def packages_write_pickup(permPickup, suffix, myTime, myIter, *, exp, cfg, params, io, state, mds, printed=None,
                          pk=None, sp=None):
    """PACKAGES_WRITE_PICKUP( permPickup, suffix, myTime, myIter, myThid )
    @63cdc0b model/src/packages_write_pickup.F:11-247

    C     | o Write pickup files for each package which needs it to restart

    Ported (ADVECT lane, plan Task 14): GAD_WRITE_PICKUP under ALLOW_GENERIC_ADVDIFF with useGAD (:75-80);
    CD_CODE_WRITE_PICKUP under ALLOW_CD_CODE with useCDscheme (:82-87, GO lane);
    DIAGNOSTICS_WRITE_PICKUP under useDiagnostics (:205-210) writes only with diag_pickup_write
    (pkg/diagnostics/diagnostics_write_pickup.F:69; diagnostics_readparms.F:157 default .FALSE.): nothing unless the
    run's data.diagnostics sets it (then raises). Every other package call raises when its switch is on (:82-224).
    `exp`: the experiment (data.diagnostics). Returns the file names written. PTRACERS lane: PTRACERS_WRITE_PICKUP
    under ALLOW_PTRACERS with usePTRACERS; the records it prints go to `printed` (a list) when given."""
    from mitjax.params_io import RunParams
    use = {k: v for k, v in cfg.use}
    written = []
    if cfg.cpp.ALLOW_GENERIC_ADVDIFF and use.get("useGAD", False):            # :75-80
        from mitjax.pkg.generic_advdiff.gad_write_pickup import gad_write_pickup
        written += gad_write_pickup(suffix, myTime, myIter, cfg=cfg, params=params, io=io, state=state, mds=mds)
    if cfg.cpp.ALLOW_CD_CODE and params.useCDscheme:                            # :82-87 (GO lane)
        from mitjax.pkg.cd_code.cd_code_write_pickup import cd_code_write_pickup
        fn = cd_code_write_pickup(permPickup, suffix, myTime, myIter, cfg=cfg, io=io, state=state, mds=mds)
        if fn:
            written.append(fn)
    if cfg.cpp.ALLOW_DIAGNOSTICS and use.get("useDiagnostics", False):          # :205-210
        rp = RunParams(exp.run)
        if rp.has("data.diagnostics", "DIAGNOSTICS_LIST", "diag_pickup_write") and \
                rp.get("data.diagnostics", "DIAGNOSTICS_LIST", "diag_pickup_write"):
            raise NotImplementedError("PACKAGES_WRITE_PICKUP: DIAGNOSTICS_WRITE_PICKUP (diag_pickup_write) is not "
                                      "ported")
    if cfg.cpp.ALLOW_PTRACERS and use.get("usePTRACERS", False):               # PTRACERS lane (:153-158)
        from mitjax.pkg.ptracers.ptracers_write_pickup import ptracers_write_pickup
        lines, fns = ptracers_write_pickup(permPickup, suffix, myTime, myIter, cfg=cfg, ptr=params, io=io,
                                           state=state, mds=mds)
        written += fns
        if printed is not None:                                                 # its PRINT_MESSAGE records
            printed.extend(lines)
    if cfg.cpp.ALLOW_GMREDI and use.get("useGMRedi", False):                  # :123-128 (GO lane)
        from mitjax.pkg.gmredi.gmredi_write_pickup import gmredi_write_pickup
        written += gmredi_write_pickup(permPickup, suffix, myTime, myIter, exp=exp, cfg=cfg)
    if cfg.cpp.ALLOW_GGL90 and use.get("useGGL90", False):                    # :116-121 (vermix lane)
        from mitjax.pkg.ggl90.ggl90_write_pickup import ggl90_write_pickup
        from mitjax.pkg.rw.write_rec import WRITE_REC_3D_RL
        for fn, prec, irec, fld in ggl90_write_pickup(permPickup, suffix, myTime, myIter, cfg=cfg, ggl=pk["ggl"]):
            WRITE_REC_3D_RL(fn, prec, cfg.size.Nr, fld, irec, myIter, mds=mds, globalFile=io.globalFiles)
            if fn not in written:
                written.append(fn)
    if cfg.cpp.flag("ALLOW_SEAICE") and use.get("useSEAICE", False):           # :183-188 (lane M4COL)
        from mitjax.pkg.seaice.seaice_write_pickup import seaice_write_pickup
        written.append(seaice_write_pickup(permPickup, suffix, myTime, myIter, cfg=cfg, sp=sp, sf=pk["seaice"],
                                           io=io, mds=mds, useThSIce=use.get("useThSIce", False)))
    if cfg.cpp.ALLOW_ECCO and use.get("useECCO", False):                     # :236-240 (lane M4ADCOL session 3)
        from mitjax.pkg.ecco.ecco_write_pickup import ecco_write_pickup
        written += ecco_write_pickup(permPickup, suffix, myTime, myIter, cfg=cfg)
    calls = ("useOBCS", "useBBL", "useCheapAML", "useFLT", "useGCHEM",
             "useStreamIce", "useShelfIce", "useThSIce", "useLand", "useAtm_Phys", "useFizhi",
             "useMYPACKAGE")                                                    # the switches of :89-224
    others = sorted(k for k in calls if use.get(k, False))
    if others:                                                                  # :89-224
        raise NotImplementedError(f"PACKAGES_WRITE_PICKUP: the pickups of {others} are not ported")
    return written
