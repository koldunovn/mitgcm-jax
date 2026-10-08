"""PTRACERS_FORCING_SURF: pkg/ptracers/ptracers_forcing_surf.F @63cdc0b, and the selection of the build's own version
(an experiment's code directory may hold its own ptracers_forcing_surf.F: genmake2 takes the first directory holding
the file, the experiment's code directory before pkg/ptracers, tools/genmake2:2945-2961)."""

import importlib

from mitjax.farray import loop_i, loop_j

# the experiments whose code directory holds its own ptracers_forcing_surf.F, and the port of that version
_OWN = {("tutorial_global_oce_latlon", "code"): "mitjax.verification.tutorial_global_oce_latlon.code",
        ("tutorial_tracer_adjsens", "code_ad"): "mitjax.verification.tutorial_tracer_adjsens.code_ad"}


def routine_of_build(cfg, name="ptracers_forcing_surf"):
    """The function the build compiles for `name` (ptracers_forcing_surf or ptracers_apply_forcing): the
    experiment's own version when verification/<exp>/<code dir>/<name>.F exists, else pkg/ptracers'."""
    from mitjax.config.params import code_path
    if (code_path(cfg) / f"{name}.F").exists():
        mod = _OWN.get((cfg.experiment, cfg.code_dir))
        if mod is None:
            raise NotImplementedError(f"{cfg.experiment}/{cfg.code_dir}'s own {name}.F is not ported")
        return getattr(importlib.import_module(f"{mod}.{name}"), name)
    return getattr(importlib.import_module(f"mitjax.pkg.ptracers.{name}"), name)


def check_unported(routine, *, params, ptr):
    """The branches of PTRACERS_FORCING_SURF after the zero initialisation that no ported run takes (shared by the
    experiments' versions, which change only the initialisation): PTRACERS_addSrelax2EmP (:74-104; its MAX/MIN
    sites have no oracle table yet) and every tracer with PTRACERS_EvPrRn set (the fresh-water tracer flux of
    :114-191, which acts only where PTRACERS_StepFwd .AND. PTRACERS_EvPrRn .NE. UNSET_RL)."""
    if ptr.PTRACERS_addSrelax2EmP:
        raise NotImplementedError(f"{routine}: PTRACERS_addSrelax2EmP (:74-104) is not ported")
    for iTrc in range(1, ptr.PTRACERS_numInUse+1):
        if ptr.PTRACERS_StepFwd[iTrc-1] and ptr.PTRACERS_EvPrRn_ne_UNSET[iTrc-1]:
            raise NotImplementedError(f"{routine}: PTRACERS_EvPrRn set for tracer {iTrc} (:114-191) is not ported")


def ptracers_forcing_surf(relaxForcingS, iMin, iMax, jMin, jMax, myTime, myIter, *, cfg, grid, params, ptr, ptf,
                          ff=None):
    """PTRACERS_FORCING_SURF( relaxForcingS, bi, bj, iMin, iMax, jMin, jMax, myTime, myIter, myThid )
    @63cdc0b pkg/ptracers/ptracers_forcing_surf.F:7-198

    C     Precomputes surface forcing term for pkg/ptracers.
    C     Precomputation is needed because of non-local KPP transport term,
    C     routine KPP_TRANSPORT_PTR.
    C  relaxForcingS        :: Salt forcing due to surface relaxation

    Returns `ptf` (PTRACERS_FIELDS.h) with surfaceForcingPTr set on iMin:iMax, jMin:jMax for every tracer
    1..PTRACERS_numInUse (:62-71: 0. _d 0; the points outside the range keep their values). add2EmP = 0 (:105-111,
    PTRACERS_addSrelax2EmP = .FALSE.). The rest acts only for tracers with PTRACERS_EvPrRn set (`check_unported`
    raises for those and for PTRACERS_addSrelax2EmP)."""
    check_unported("PTRACERS_FORCING_SURF", params=params, ptr=ptr)
    j = loop_j(jMin, jMax)                                                      # :64
    i = loop_i(iMin, iMax)                                                      # :65
    for iTrc in range(1, ptr.PTRACERS_numInUse+1):                              # :62
        s = ptf.surfaceForcingPTr[iTrc-1]
        ptf = ptf.set("surfaceForcingPTr", iTrc, s.at[i, j].set(0.))            # :66
    return ptf
