"""DO_OCEANIC_PHYS: model/src/do_oceanic_phys.F @63cdc0b (as far as the M1 variants without GM/Redi run it)."""

from mitjax.farray import FArray, loop_i, loop_j, loops_kji
from mitjax.model.src.external_forcing_surf import external_forcing_surf
from mitjax.model.src.find_rho import find_rho_2d


def _level(fld, k):
    """`fld(1-OLx,1-OLy,k,bi,bj)` passed to a 2-D dummy (sequence association, KERNEL_GUIDE §4): level k."""
    (_, ilo, ihi), (_, jlo, jhi), (_, klo, _) = fld.dims
    kk = k - klo                                                    # k: a Python int or a traced level (KIdx)
    return FArray(fld.data[:, getattr(kk, "value", kk)], fld.name, i=(ilo, ihi), j=(jlo, jhi))


def _set_level(fld, k, lev):
    """Write a 2-D dummy back into level k of the actual argument."""
    (_, _, _), (_, _, _), (_, klo, _) = fld.dims
    kk = k - klo
    return FArray(fld.data.at[:, getattr(kk, "value", kk)].set(lev.data), fld.name, tiled=fld.tiled, _dims=fld.dims)


def do_oceanic_phys(myTime, myIter, *, cfg, grid, params, fp, eos, state, ff, phi0surf, probe=None, gm=None,
                    gm_params=None, kpp=None, kpp_p=None, ex=None, mix=None, visc=None, seaice=None, until=None,
                    salt_plume=None):
    """DO_OCEANIC_PHYS( myTime, myIter, myThid )   @63cdc0b model/src/do_oceanic_phys.F:43-1136

    C     | SUBROUTINE DO_OCEANIC_PHYS
    C     | o Controlling routine for oceanic physics and parameterization

    Returns (state, ff, phi0surf), and with `gm` (GMREDI.h, useGMRedi) (state, ff, phi0surf, gm). Ported: :247-248
    kSrf, :553-558 FREEZE_SURFACE (allowFreezing; R5 arm), :560-563 iMin..jMax = full range, :574-581
    EXTERNAL_FORCING_SURF (FORCING lane), :620-636 the zeroed locals, :758-767 FIND_RHO_2D of every level into
    rhoInSitu (COL lane), the upward k loop :803-887 (R2 arm below), the package calls after it only as far as their
    switches are off (KPP, PP81, MY82, GGL90, ...: compiled-but-off is fine). The levels of :759-766 are independent:
    unrolled loop.

    R2 arm (tracer lane, plan Task 13; `_oceanic_phys_k_loop` below): doDiagsRho (:254-266), calcConvect (:270),
    the upward k loop :803-887 with GRAD_SIGMA and CALC_IVDC, and CALC_OCE_MXLAYER (:895-900). `probe`: optional,
    called with the dump stages P01_external_forcing_surf, P02_rho_sigma_ivdc, P03_mxlayer, P05_gmredi_tensor and
    P06_gmredi_exch.

    R5 arm (plan Task 16, tutorial_global_oce_optim/code_ad): FREEZE_SURFACE (:553-558); the ALLOW_AUTODIFF resets
    of :637-713 inside the tile loop (rhoInSitu = 0., IVDConvCount = 0. (REAL*4 literals, exact) and, under
    ALLOW_GMREDI, Kwx Kwy Kwz Kux Kvy (Kuz Kvz with GM_EXTRA_DIAGONAL) = 0 on every point; GGL90 / SALT_PLUME /
    KPP / GM_BOLUS_ADVEC / GM_VISBECK_VARIABLE_K fields raise if compiled) before FIND_RHO_2D; the ALLOW_AUTODIFF
    `IF (fluidIsWater)` around FIND_RHO_2D (:729-788; its atmosphere ELSE raises); GMREDI_CALC_TENSOR
    (calcGMRedi, :1031-1043; the ELSE GMREDI_CALC_TENSOR_DUMMY raises) and GMREDI_DO_EXCH (:1094-1098). `gm`: the
    GMREDI.h state (mitjax/pkg/gmredi/gmredi_h.Gmredi), `gm_params`: what GMREDI_CALC_TENSOR reads of PARAMS.h
    (rVel2wUnit, wUnit2rVel, rUnit2z, z2rUnit, usingZCoords).

    PTRACERS lane (tutorial_tracer_adjsens/code_ad: pkg/kpp compiled, useKPP = .FALSE.): `kpp`, the KPP.h fields
    {name: FArray} (mitjax/pkg/kpp/kpp_calc_dummy.KPP_FIELDS), then appended to the returned tuple: the
    ALLOW_AUTODIFF reset KPPdiffKzS = KPPdiffKzT = 0. _d 0 on every point (:700-709; without ALLOW_OFFLINE no IF) and
    KPP_CALC_DUMMY (:949-965, calcKPP = useKPP = .FALSE. with ALLOW_AUTODIFF and not ALLOW_OFFLINE). The forward
    model reads none of these fields.

    Lane B (Task 25): a fluid that is not water in a build without ALLOW_AUTODIFF returns after the surface forcing
    test (the `IF ( fluidIsWater )` of :574-1069 bypasses FIND_RHO_2D, the k loop and the mixing schemes).

    vermix lane (M3 Task 30): calcKPP = useKPP (:269; ALLOW_OFFLINE's offlineLoadKPP raises): KPP_CALC (:949-958,
    probe P07_kpp) after the k loop, KPP_DO_EXCH (:1100-1104, after GMREDI_DO_EXCH; probe P10_kpp_exch). `kpp`: the
    KPP.h fields (returned last, as in the PTRACERS arm), `kpp_p`: KPP_PARAMS.h (mitjax/pkg/kpp/kpp_params_h.Kpp),
    `ex`: the exchanger of KPP_DO_EXCH. `mix`: {"ggl": GGL90.h, "pp81": PP81.h, "my82": MY82.h} of the switched-on
    packages: PP81_CALC (:967-971, probe P08_pp81), MY82_CALC (:991-995, P09_my82), GGL90_CALC (:1003-1011, Nr > 1,
    P04_ggl90) after KPP, with sigmaR of the k loop; GGL90_EXCHANGES (:1106-1109, P11_ggl90_exch: the GGL90.h fields
    after it; in a build that compiles pkg/ggl90 with useGGL90 = .FALSE. the dump stage still writes the never-set
    static common, zeros). Returned last (a dict) when given. KL10 raises.

    Lane M4COL (M4 step 3): `seaice` (dict: sf = SEAICE.h, exf = EXF_FIELDS.h, sp, op, exfp, kgeo): SEAICE_MODEL
    (:407-474, useSEAICE; the CADJ STOREs are TAF directives) at :453, before FREEZE_SURFACE and EXTERNAL_FORCING_SURF;
    probes I00b_seaice_begin .. I04_growth (inside it) and P13_seaice_model (after it). Then (sf, exf) is appended to
    the returned tuple (last). Lane M4LAB: `salt_plume` (SALT_PLUME.h saltPlumeFlux, SaltPlumeDepth; ALLOW_SALT_PLUME
    compiled) for KPP_CALC; GMREDI_CALC_TENSOR gets the KPP.h fields (KPPhbl, useKPP). Session 3: with `seaice["spp"]`
    (SaltPlumeParams) SEAICE_MODEL's SEAICE_GROWTH writes saltPlumeFlux first (KPP_CALC reads the new value), and the
    sea-ice entry of the returned tuple is (sf, exf, salt_plume). Lane M4ADLAB session 3: useDOWN_SLOPE with
    `mix["dwnslp"]` (the down-slope state, returned in `mix`): rhoInSitu by DWNSLP_CALC_RHO (:733-741) instead of
    FIND_RHO_2D, DWNSLP_CALC_FLOW after GMREDI_CALC_TENSOR (:1045-1061, kBottom = kLowC); without GMREDI.h raises. `until` (gates only, static): "P01_external_forcing_surf" returns right after that
    probe, the package state unchanged (the front of a step whose later routines are not ported for the build)."""
    calcKPP = bool(cfg.cpp.ALLOW_KPP and params.useKPP)                         # :269
    if calcKPP and (kpp is None or kpp_p is None or ex is None):
        raise ValueError("DO_OCEANIC_PHYS: useKPP needs the KPP.h fields `kpp`, KPP_PARAMS.h `kpp_p` and `ex`")
    if calcKPP and cfg.cpp.ALLOW_OFFLINE:                                       # :271-277
        raise NotImplementedError("DO_OCEANIC_PHYS: ALLOW_OFFLINE (offlineLoadKPP) is not ported")
    if params.useGMRedi and gm is None:
        raise NotImplementedError("DO_OCEANIC_PHYS: useGMRedi needs the GMREDI.h state `gm`")
    mix = dict(mix or {})
    for pkg, key in (("usePP81", "pp81"), ("useMY82", "my82"), ("useGGL90", "ggl")):     # vermix lane
        if getattr(params, pkg) and key not in mix:
            raise ValueError(f"DO_OCEANIC_PHYS: {pkg} needs its package state `mix[{key!r}]`")
    for pkg in ("useKL10", "useSALT_PLUME", "useBBL"):
        if getattr(params, pkg):                                                # GO lane: live switch (ini_parms_dyn)
            raise NotImplementedError(f"DO_OCEANIC_PHYS: {pkg} is not wired")
    useDWNSLP = bool(cfg.cpp.ALLOW_DOWN_SLOPE and params.useDOWN_SLOPE)        # lane M4ADLAB session 3
    if useDWNSLP and "dwnslp" not in mix:
        raise ValueError("DO_OCEANIC_PHYS: useDOWN_SLOPE needs the down-slope state `mix['dwnslp']`")
    pr = probe if probe is not None else (lambda stage, values: None)
    sz = cfg.size
    si_out = ()
    si_area = None
    if cfg.cpp.flag("ALLOW_SEAICE") and params_use(cfg, "useSEAICE"):                   # :407-474 (lane M4COL)
        if seaice is None:
            raise ValueError("DO_OCEANIC_PHYS: useSEAICE needs the sea-ice state `seaice`")
        from mitjax.pkg.seaice.seaice_model import seaice_model

        def si_probe(stage, sf_, ff_, exf_):
            pr(stage, _seaice_values(sf_, ff_, exf_))
        out_si = seaice_model(myTime, myIter, seaice["sf"], ff, seaice["exf"], cfg=cfg, sp=seaice["sp"],
                              op=seaice["op"], grid=grid, state=state, exfp=seaice["exfp"], ex=ex,
                              kgeo=seaice["kgeo"], probe=si_probe,       # :453
                              salt_plume=salt_plume if seaice.get("spp") is not None else None,
                              spp=seaice.get("spp"))     # lane M4LAB session 3: SEAICE_GROWTH writes saltPlumeFlux
        sf, ff, exf_f = out_si[:3]
        if len(out_si) == 4:
            salt_plume = out_si[3]                       # SALT_PLUME.h for KPP_CALC (:949-958) and the caller
        pr("P13_seaice_model", _seaice_values(sf, ff, exf_f))
        if cfg.cpp.ALLOW_COST and seaice.get("cost") is not None:              # :463-465 (lane M4ADCOL)
            from mitjax.pkg.seaice.seaice_cost_sensi import seaice_cost_sensi
            c = seaice["cost"]
            cost = seaice_cost_sensi(c["cost"], myTime, myIter, cfg=cfg, sp=seaice["sp"], AREA=sf["AREA"],
                                     HEFF=sf["HEFF"], rA=grid.rA, endTime=c["endTime"],
                                     startTime=params.startTime, lastinterval=c["lastinterval"],
                                     deltaTClock=params.deltaTClock)
            sf = dict(sf, _cost=cost)           # returned with SEAICE.h (forward_step pops it into pk["cost"])
        si_out = ((sf, exf_f) + ((salt_plume,) if len(out_si) == 4 else ()),)
        si_area = sf["AREA"]                    # SEAICE.h AREA for FORCING_SURF_RELAX :74-91 (lane M4LAB)
    elif seaice is not None:
        raise ValueError("DO_OCEANIC_PHYS: `seaice` given without useSEAICE")
    if params.allowFreezing:                                                    # :553-558 (R5 arm)
        from mitjax.model.src.freeze_surface import freeze_surface
        state, ff = freeze_surface(myTime, myIter, ff, cfg=cfg, grid=grid, fp=fp, state=state)
    iMin, iMax = 1-sz.OLx, sz.sNx+sz.OLx                                        # :560-561
    jMin, jMax = 1-sz.OLy, sz.sNy+sz.OLy                                        # :562-563
    if cfg.cpp.ALLOW_AUTODIFF or params.fluidIsWater:                           # :565-575 (AUTODIFF: no IF)
        ff, state, phi0surf = external_forcing_surf(iMin, iMax, jMin, jMax, myTime, myIter, ff, cfg=cfg,
                                                    grid=grid, fp=fp, state=state, phi0surf=phi0surf,
                                                    ptr=params,                 # ptr: PTRACERS lane
                                                    sp=None if seaice is None else seaice["sp"],   # lane M4OFF
                                                    area=si_area)               # lane M4LAB
    pr("P01_external_forcing_surf", ff)
    if until == "P01_external_forcing_surf":                                    # gates only (docstring)
        return ((state, ff, phi0surf) + ((gm,) if gm is not None else ()) + ((kpp,) if kpp is not None else ())
                + ((mix,) if mix else ()) + si_out)
    if cfg.cpp.ALLOW_AUTODIFF:                                                  # :637-713 (R5 arm)
        state, gm = _autodiff_resets(cfg=cfg, params=params, state=state, gm=gm, kpp_on=kpp is not None)
        if kpp is not None:                                                     # :700-709 (PTRACERS lane)
            k3, j3, i3 = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
            kpp = dict(kpp, KPPdiffKzS=kpp["KPPdiffKzS"].at[i3, j3, k3].set(0.0),     # 0. _d 0
                       KPPdiffKzT=kpp["KPPdiffKzT"].at[i3, j3, k3].set(0.0))
    if cfg.cpp.ALLOW_AUTODIFF and not params.fluidIsWater:                      # :729-788
        raise NotImplementedError("DO_OCEANIC_PHYS: the ALLOW_AUTODIFF atmosphere arm (:767-788) is not ported")
    if not cfg.cpp.ALLOW_AUTODIFF and not params.fluidIsWater:                  # :574-1069 (lane B, Task 25)
        # the IF ( fluidIsWater ) of a build without ALLOW_AUTODIFF ends at :1069: FIND_RHO_2D, the k loop, the
        # mixed layer and the mixing / GMREDI_CALC_TENSOR calls are bypassed; after it GMREDI_DO_EXCH (:1094-1098)
        if params.useGMRedi:
            raise NotImplementedError("DO_OCEANIC_PHYS: useGMRedi with fluidIsAir is not ported")
        return (state, ff, phi0surf) + ((kpp,) if kpp is not None else ()) + si_out   # kpp: PTRACERS lane
    def rho_k(k, rhoInSitu):                                                    # :759-766  DO k=1,Nr
        if useDWNSLP:                                                           # :733-741 (lane M4ADLAB)
            from mitjax.pkg.down_slope.dwnslp_calc_rho import dwnslp_calc_rho
            rhoLoc = dwnslp_calc_rho(state.theta, state.salt, _level(rhoInSitu, k), k, cfg=cfg, grid=grid,
                                     params=params, eos=eos, state=state)       # :736-739
            return _set_level(rhoInSitu, k, rhoLoc)
        rhoLoc = find_rho_2d(iMin, iMax, jMin, jMax, k, _level(state.theta, k), _level(state.salt, k),
                             _level(rhoInSitu, k), k, cfg=cfg, grid=grid, params=params, eos=eos, state=state)
        return _set_level(rhoInSitu, k, rhoLoc)
    from mitjax.ops.scan_k import scan_levels                              # KERNEL_GUIDE §4: a level scan
    state = state.replace(rhoInSitu=scan_levels(rho_k, state.rhoInSitu, 1, sz.Nr))
    state, sigmas = _oceanic_phys_k_loop(iMin, iMax, jMin, jMax, myTime, myIter, cfg=cfg, grid=grid, params=params,
                                         eos=eos, state=state, probe=probe,
                                         gm=gm)                   # R2 arm (tracer lane): :803-900
    if calcKPP:                                                                 # :949-958 (vermix lane)
        from mitjax.pkg.kpp.kpp_calc import kpp_calc
        kpp = kpp_calc(myTime, myIter, cfg=cfg, grid=grid, params=params, fp=fp, eos=eos, state=state, ff=ff,
                       kpp=kpp_p, kppf=kpp, salt_plume=salt_plume)   # salt_plume: SALT_PLUME.h (lane M4LAB)
        pr("P07_kpp", kpp)
    elif cfg.cpp.ALLOW_KPP and cfg.cpp.ALLOW_AUTODIFF and not cfg.cpp.ALLOW_OFFLINE:   # :959-965 (PTRACERS lane)
        if kpp is None:                       # calcKPP = useKPP = .FALSE. (refused above): the ELSE of :959-963
            raise NotImplementedError("DO_OCEANIC_PHYS: KPP_CALC_DUMMY needs the KPP.h state `kpp`")
        from mitjax.pkg.kpp.kpp_calc_dummy import kpp_calc_dummy
        kpp = kpp_calc_dummy(myTime, myIter, cfg=cfg, grid=grid, params=params, state=state, kpp=kpp)   # :961-962
    sigmaR = sigmas[2] if sigmas is not None else None
    if params.usePP81:                                                          # :967-971 (vermix lane)
        from mitjax.pkg.pp81.pp81_calc import pp81_calc
        mix["pp81"] = pp81_calc(sigmaR, myTime, myIter, cfg=cfg, grid=grid, params=params, eos=eos, state=state,
                                pp=mix["pp81"])
        pr("P08_pp81", {n: getattr(mix["pp81"], n) for n in ("PPviscAr", "PPdiffKr")})   # the dumped PP81.h fields
    if params.useMY82:                                                          # :991-995 (vermix lane)
        from mitjax.pkg.my82.my82_calc import my82_calc
        mix["my82"] = my82_calc(sigmaR, myTime, myIter, cfg=cfg, grid=grid, params=params, eos=eos, state=state,
                                my=mix["my82"])
        pr("P09_my82", {n: getattr(mix["my82"], n) for n in ("MYviscAr", "MYdiffKr", "MYhbl")})   # dumped MY82.h
    if params.useGGL90 and cfg.size.Nr > 1:                                     # :1003-1011 (vermix lane)
        from types import SimpleNamespace
        from mitjax.pkg.ggl90.ggl90_calc import ggl90_calc
        st = SimpleNamespace(uVel=state.uVel, vVel=state.vVel, surfaceForcingU=ff.surfaceForcingU,
                             surfaceForcingV=ff.surfaceForcingV)              # DYNVARS.h + FFIELDS.h it reads
        mix["ggl"] = ggl90_calc(sigmaR, myTime, myIter, cfg=cfg, grid=grid, params=params, ggl=mix["ggl"], state=st,
                                gm=gm)
        pr("P04_ggl90", mix["ggl"])
    if gm is None:
        if useDWNSLP:
            raise NotImplementedError("DO_OCEANIC_PHYS: useDOWN_SLOPE without GMREDI.h is not ported")
        if calcKPP:                                                             # :1100-1104 (vermix lane)
            kpp = _kpp_do_exch(kpp, cfg=cfg, ex=ex)
            pr("P10_kpp_exch", kpp)
        mix = _ggl90_exchanges(mix, cfg=cfg, params=params, ex=ex, pr=pr)      # :1106-1109 (vermix lane)
        return (state, ff, phi0surf) + ((kpp,) if kpp is not None else ()) + ((mix,) if mix else ()) + si_out
    # ---- R5 arm: GM/Redi (plan Task 16) ----
    if cfg.cpp.ALLOW_GMREDI:
        if params.useGMRedi:                                                    # :1031-1037 calcGMRedi
            from mitjax.pkg.gmredi.gmredi_calc_tensor import gmredi_calc_tensor
            gm = gmredi_calc_tensor(iMin, iMax, jMin, jMax, *sigmas, myTime, myIter, cfg=cfg, grid=grid,
                                    params=gm_params, gm=gm, state=state,
                                    visc=visc,                  # visc: GM_useLeithQG (M3 lane MLAdjust)
                                    kppf=kpp if calcKPP else None)   # KPPhbl (useKPP, lane M4LAB)
        elif cfg.cpp.ALLOW_AUTODIFF and not cfg.cpp.ALLOW_OFFLINE:              # :1038-1043
            raise NotImplementedError("DO_OCEANIC_PHYS: GMREDI_CALC_TENSOR_DUMMY is not ported")
    pr("P05_gmredi_tensor", gm)
    if useDWNSLP:                                                               # :1045-1061 (lane M4ADLAB)
        from mitjax.pkg.down_slope.dwnslp_calc_flow import dwnslp_calc_flow
        mix["dwnslp"] = dwnslp_calc_flow(state.rhoInSitu, mix["dwnslp"], cfg=cfg, params=params,
                                         usingPCoords=params.usingPCoords)     # :1055-1059 kBottom = kLowC
    if cfg.cpp.ALLOW_GMREDI and params.useGMRedi:                               # :1094-1098
        from mitjax.pkg.gmredi.gmredi_do_exch import gmredi_do_exch
        gm = gmredi_do_exch(myTime, myIter, cfg=cfg, gm=gm, params=params)   # GOADK lane: params
    pr("P06_gmredi_exch", gm)
    if calcKPP:                                                                 # :1100-1104 (vermix lane)
        kpp = _kpp_do_exch(kpp, cfg=cfg, ex=ex)
        pr("P10_kpp_exch", kpp)
    mix = _ggl90_exchanges(mix, cfg=cfg, params=params, ex=ex, pr=pr)          # :1106-1109 (vermix lane)
    return (state, ff, phi0surf, gm) + ((kpp,) if kpp is not None else ()) + ((mix,) if mix else ()) + si_out


def params_use(cfg, name):
    """A package switch as PACKAGES_BOOT leaves it; .FALSE. if the build has none."""
    return cfg.use_flag(name) if any(k.lower() == name.lower() for k, _ in cfg.use) else False


def _seaice_values(sf, ff, exf):
    """{name: FArray} of a sea-ice probe: EXF_FIELDS.h, the FFIELDS.h fields, SEAICE.h (the dump stages' fields)."""
    vals = dict(exf)
    vals.update({n: getattr(ff, n) for n in ff.names()})
    vals.update(sf)
    return vals


def _ggl90_exchanges(mix, *, cfg, params, ex, pr):
    """do_oceanic_phys.F:1106-1109 `IF ( useGGL90 ) CALL GGL90_EXCHANGES( myThid )` and the dump stage after it
    (P11_ggl90_exch, written in every build that compiles pkg/ggl90: with useGGL90 = .FALSE. the GGL90.h fields are
    the static common's zeros, never written)."""
    if not cfg.cpp.ALLOW_GGL90:
        return mix
    if params.useGGL90:
        from mitjax.pkg.ggl90.ggl90_exchanges import ggl90_exchanges
        mix["ggl"] = ggl90_exchanges(cfg=cfg, ggl=mix["ggl"], ex=ex)
        pr("P11_ggl90_exch", mix["ggl"])
    else:
        from mitjax.pkg.ggl90.ggl90_h import declare
        pr("P11_ggl90_exch", {n: declare(n, cfg.size, fill=0.0) for n in ("GGL90TKE", "GGL90viscArU",
                                                                          "GGL90viscArV", "GGL90diffKr")})
    return mix


def _kpp_do_exch(kpp, *, cfg, ex):
    """KPP_DO_EXCH (do_oceanic_phys.F:1102-1104, vermix lane)."""
    from mitjax.pkg.kpp.kpp_do_exch import kpp_do_exch
    return kpp_do_exch(kpp, cfg=cfg, ex=ex)


def _autodiff_resets(*, cfg, params, state, gm, kpp_on=False):
    """do_oceanic_phys.F:637-713 (#ifdef ALLOW_AUTODIFF, inside the tile loop, every point incl. halos):
    rhoInSitu = 0. (:643), IVDConvCount = 0. (:665-671; ALLOW_OFFLINE's IF (calcConvect) not compiled in R5:
    raises with useOffLine), and under ALLOW_GMREDI Kwx, Kwy, Kwz, Kux, Kvy (:678-682) and with GM_EXTRA_DIAGONAL
    Kuz, Kvz (:682-683), with GM_BOLUS_ADVEC GM_PsiX, GM_PsiY (:685-688; GOADK lane, global_ocean code_ad) = 0. _d 0.
    The literals `0.` are REAL*4 (exact zero). GGL90 / SALT_PLUME_VOLUME / GM_VISBECK_VARIABLE_K fields: raise when
    compiled (no ported build); KPP: the caller's arm (`kpp_on`, PTRACERS lane)."""
    for opt in ("ALLOW_GGL90",) + (() if kpp_on else ("ALLOW_KPP",)):      # KPP: the caller (PTRACERS lane)
        if getattr(cfg.cpp, opt):
            raise NotImplementedError(f"DO_OCEANIC_PHYS: the ALLOW_AUTODIFF resets of {opt} fields are not ported")
    if cfg.cpp.ALLOW_SALT_PLUME and cfg.cpp.flag("SALT_PLUME_VOLUME"):
        raise NotImplementedError("DO_OCEANIC_PHYS: the ALLOW_AUTODIFF resets of SPforcingS/T are not ported")
    if cfg.cpp.ALLOW_OFFLINE:
        raise NotImplementedError("DO_OCEANIC_PHYS: the ALLOW_OFFLINE conditions of :660-707 are not ported")
    sz = cfg.size
    k, j, i = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
    state = state.replace(rhoInSitu=state.rhoInSitu.at[i, j, k].set(0.),          # :643
                          IVDConvCount=state.IVDConvCount.at[i, j, k].set(0.))    # :668
    if cfg.cpp.ALLOW_GMREDI and gm is not None:                                  # :675-698
        if cfg.cpp.flag("GM_VISBECK_VARIABLE_K", "GMREDI_OPTIONS.h"):
            raise NotImplementedError("DO_OCEANIC_PHYS: the reset of VisbeckK is not ported")
        names = ["Kwx", "Kwy", "Kwz", "Kux", "Kvy"]                              # :676-680
        if cfg.cpp.flag("GM_EXTRA_DIAGONAL", "GMREDI_OPTIONS.h"):
            names += ["Kuz", "Kvz"]                                              # :682-683
        if cfg.cpp.flag("GM_BOLUS_ADVEC", "GMREDI_OPTIONS.h"):                   # :685-688 (GOADK lane: code_ad)
            names += ["GM_PsiX", "GM_PsiY"]
        gm = gm.replace(**{n: getattr(gm, n).at[i, j, k].set(0.) for n in names})
    return state, gm


# ---- R2 arm (tracer lane, plan Task 13) ----------------------------------------------------------------------------

def _oceanic_phys_k_loop(iMin, iMax, jMin, jMax, myTime, myIter, *, cfg, grid, params, eos, state, probe, gm=None):
    """do_oceanic_phys.F:254-270 (doDiagsRho, calcConvect) and :620-636, :803-900 (the zeroed sigma/rho locals, the
    upward k loop with GRAD_SIGMA and CALC_IVDC, CALC_OCE_MXLAYER) @63cdc0b, inside the tile loop of :604-1065.
    Returns the State (IVDConvCount, hMixLayer).

    doDiagsRho: DIAGNOSTICS_IS_ON is evaluated on the host (`params.diagnostics_is_on`: the names it returns
    .TRUE. for in this run, empty unless the run's useDiagnostics; mitjax/model/src/ini_parms_tracer.py), so
    `useDiagnostics .AND. fluidIsWater` (:256) is `fluidIsWater` with the lookups in that tuple. calcConvect
    (`ivdc_kappa.NE.0.`, :270) is decided on the host (params.ivdc_kappa_ne_0). The k loop calls per-level routines
    in the Fortran order (DO k=Nr,1,-1): an unrolled Python loop (KERNEL_GUIDE §4). Raise: p coordinates (:824-835),
    ALLOW_LEITH_QG, GMREDI_WITH_STABLE_ADJOINT, doDiagsRho >= 4 (DIAGS_RHO_L), ALLOW_OFFLINE with useOffLine,
    SALT_PLUME (the ALLOW_AUTODIFF resets :637-713 sit before FIND_RHO_2D: in `do_oceanic_phys`, R5 arm). Returns
    (state, (sigmaX, sigmaY, sigmaR)), or (state, None) when no branch of the loop runs. DIAGNOSTICS_FILL of DRHODR
    (:931-936) and the fills of :1112-1129 are output only (pkg/diagnostics is not ported)."""
    from mitjax.model.src.calc_ivdc import calc_ivdc
    from mitjax.model.src.calc_oce_mxlayer import calc_oce_mxlayer
    from mitjax.model.src.grad_sigma import grad_sigma
    pr = probe if probe is not None else (lambda stage, values: None)
    st = params.static_items()
    if "diagnostics_is_on" not in st:
        if params.useDiagnostics:
            raise NotImplementedError("DO_OCEANIC_PHYS: doDiagsRho needs params.diagnostics_is_on (ini_parms_tracer)")
        diag_on = ()
    else:
        diag_on = params.diagnostics_is_on
    calcConvect = params.ivdc_kappa_ne_0                                        # :270
    doDiagsRho = 0                                                              # :254
    if params.fluidIsWater:                                                     # :256
        if "MXLDEPTH" in diag_on:                                               # :257-258
            doDiagsRho = doDiagsRho + 1
        if "DRHODR  " in diag_on:                                               # :259-260
            doDiagsRho = doDiagsRho + 2
        if "WdRHO_P " in diag_on:                                               # :261-262
            doDiagsRho = doDiagsRho + 4
        if "WdRHOdP " in diag_on:                                               # :263-264
            doDiagsRho = doDiagsRho + 8
    calcGMRedi = params.useGMRedi                                               # :268
    if not (calcGMRedi or calcConvect or doDiagsRho >= 1 or params.usePP81 or params.useKL10   # :807-810 (the
            or params.useMY82 or params.useGGL90 or params.useSALT_PLUME):      # vermix lane: the mixing packages)
        if probe is not None:           # lane M4COL: the k loop runs no branch; sigmaX/Y/R keep the zeros of :620-629
            sz = cfg.size               # (the dump after the loop, P02, of 1D_ocean_ice_column: KPP alone)
            k3, j3, i3 = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
            z = {n: state.theta.local(n).at[i3, j3, k3].set(0.) for n in ("sigmaX", "sigmaY", "sigmaR")}
            pr("P02_rho_sigma_ivdc", {**z, "rhoInSitu": state.rhoInSitu, "IVDConvCount": state.IVDConvCount,
                                      "hMixLayer": state.hMixLayer, "totPhiHyd": state.totPhiHyd,
                                      "phiHydLow": state.phiHydLow})
        return state, None
    if cfg.cpp.ALLOW_OFFLINE and params.useOffLine:                             # :271-277
        raise NotImplementedError("DO_OCEANIC_PHYS: ALLOW_OFFLINE (useOffLine) is not ported")
    if cfg.cpp.flag("GMREDI_WITH_STABLE_ADJOINT"):          # ALLOW_LEITH_QG (:847-853): ported (M3 lane MLAdjust)
        raise NotImplementedError("DO_OCEANIC_PHYS: GMREDI_WITH_STABLE_ADJOINT is not ported")
    if doDiagsRho >= 4:                                                         # :877-884
        raise NotImplementedError("DO_OCEANIC_PHYS: DIAGS_RHO_L (doDiagsRho >= 4) is not ported")
    sz = cfg.size
    kSrf = 1                                                                    # :247
    if params.usingPCoords:                                                     # :248
        kSrf = sz.Nr
    t = state.theta
    k3, j3, i3 = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
    sigmaX = t.local("sigmaX").at[i3, j3, k3].set(0.)                           # :620-629  0. _d 0
    sigmaY = t.local("sigmaY").at[i3, j3, k3].set(0.)
    sigmaR = t.local("sigmaR").at[i3, j3, k3].set(0.)
    jA = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    iA = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    rhoKm1 = _level(t, 1).local("rhoKm1").at[iA, jA].set(0.)                    # :631-636
    rhoKp1 = _level(t, 1).local("rhoKp1").at[iA, jA].set(0.)
    rhoInSitu = state.rhoInSitu
    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    # :803 DO k=Nr,1,-1 (KERNEL_GUIDE §4): k = Nr .. 2 in a level scan, k = 1 (the `k > 1` tests) static
    def sigma_k(k, c):
        rhoKm1, rhoKp1, sigmaX, sigmaY, sigmaR, state = c
        if (calcGMRedi or (k > 1 and calcConvect)                               # :807-810
                or params.usePP81 or params.useKL10
                or params.useMY82 or params.useGGL90
                or params.useSALT_PLUME or doDiagsRho >= 1):
            if k > 1:                                                           # :811
                if params.usingZCoords:                                         # :812-823
                    rhoKp1 = rhoKp1.at[i, j].set(rhoInSitu[i, j, k])
                    rhoKm1 = find_rho_2d(iMin, iMax, jMin, jMax, k, _level(state.theta, k-1),
                                         _level(state.salt, k-1), rhoKm1, k-1,
                                         cfg=cfg, grid=grid, params=params, eos=eos, state=state)
                else:                                                           # :824-835
                    raise NotImplementedError("DO_OCEANIC_PHYS: p-coordinate densities (:824-835) are not ported")
            sigmaX, sigmaY, sigmaR = grad_sigma(iMin, iMax, jMin, jMax, k,      # :841-845
                                                _level(rhoInSitu, k), rhoKm1, rhoKp1, sigmaX, sigmaY, sigmaR,
                                                cfg=cfg, grid=grid, params=params)
            if "sigmaRfield" in state:                                          # :847-853 ALLOW_LEITH_QG (M3)
                state = state.replace(sigmaRfield=state.sigmaRfield.at[i, j, k].set(sigmaR[i, j, k]))
        if k > 1 and calcConvect:                                               # :867-875
            IVDConvCount = calc_ivdc(iMin, iMax, jMin, jMax, k, sigmaR, myTime, myIter, grid=grid, state=state)
            state = state.replace(IVDConvCount=IVDConvCount)
        return (rhoKm1, rhoKp1, sigmaX, sigmaY, sigmaR, state)

    from mitjax.ops.scan_k import scan_levels
    c = (rhoKm1, rhoKp1, sigmaX, sigmaY, sigmaR, state)
    if sz.Nr >= 2:
        c = scan_levels(sigma_k, c, 2, sz.Nr, down=True)
    c = sigma_k(1, c)
    rhoKm1, rhoKp1, sigmaX, sigmaY, sigmaR, state = c

    pr("P02_rho_sigma_ivdc", {"sigmaX": sigmaX, "sigmaY": sigmaY, "sigmaR": sigmaR, "rhoInSitu": rhoInSitu,
                              "IVDConvCount": state.IVDConvCount, "hMixLayer": state.hMixLayer,
                              "totPhiHyd": state.totPhiHyd, "phiHydLow": state.phiHydLow})
    if calcGMRedi or doDiagsRho % 2 == 1:                                       # :896-900
        # COL's CALC_OCE_MXLAYER tests `useDiagnostics` (calc_oce_mxlayer.F:79) with the run's value: here
        # .TRUE. exactly when DIAGNOSTICS_IS_ON can return .TRUE. (diag_on non-empty)
        hMixLayer = calc_oce_mxlayer(_level(rhoInSitu, kSrf), sigmaR, myTime, myIter, cfg=cfg, grid=grid,
                                     params=params.replace(static={"useDiagnostics": bool(diag_on)}), eos=eos,
                                     state=state, diagnostics_is_on=lambda name: name in diag_on, gmredi=gm)
        state = state.replace(hMixLayer=hMixLayer)
        pr("P03_mxlayer", {"rhoInSitu": rhoInSitu, "IVDConvCount": state.IVDConvCount, "hMixLayer": hMixLayer,
                           "totPhiHyd": state.totPhiHyd, "phiHydLow": state.phiHydLow})
    if cfg.cpp.ALLOW_SALT_PLUME and params.useSALT_PLUME:                       # :902-929
        raise NotImplementedError("DO_OCEANIC_PHYS: SALT_PLUME is not ported")
    return state, (sigmaX, sigmaY, sigmaR)
