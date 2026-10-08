"""Manifest fragment of the M2 sub-lane PTRACERS (pkg/ptracers, convective adjustment, CG2D_NSA, SWFRAC,
KPP_CALC_DUMMY, COST_TRACER; owner: PTRACERS lane). Tier 1x only (tier 1 is near its 100-test budget)."""

MANIFEST = {
    # replay gates on the PTRACERS harness run (reference/replay_ptracers/CURRENT), substep-dump gates of the three
    # M2 ptracers experiments, PTRACERS_CHECK printout, negative controls, gradients (a few minutes)
    "mitjax/tests/test_ptracers.py": "tier1x",
    # tutorial_advection_in_gyre end to end: initial state, steps 1-3 at every dumped stage, the whole run, the INI_PARMS
    # values of the lane against the printouts (about 1.5 min)
    "mitjax/tests/test_ptracers_gyre.py": "tier1x",
    # tutorial_global_oce_latlon end to end (GM/Redi, DST3FL, the age-tracer overrides): initial state, steps 1-3 at
    # every dumped stage, the whole run and its end-of-run pickups (about 6 min)
    "mitjax/tests/test_ptracers_latlon.py": "tier1x",
    # tutorial_tracer_adjsens/input_ad forward Model: INI_CG2D tolerance line, initial state, steps 1-3 at every
    # dumped stage (cost scalars included), the whole run with the solver and COST_FINAL lines (about 5 min)
    "mitjax/tests/test_ptracers_adjsens.py": "tier1x",
    # tutorial_tracer_adjsens/input_ad adjoint: admCst/admGrd vs TAF, FD vs lane A's FD oracle, trust protocol, TLM
    # vs results, negative control (gradient, forward and tangent programs, ~35 min on a CPU node)
    "mitjax/tests/test_ptracers_adjsens_ad.py": "tier1x",
    # its adjoint monitor (ad_dynstat, ad_forcing, ad_trcstat) from a step-by-step reverse sweep vs TAF (~16 min)
    "mitjax/tests/test_ptracers_adjsens_admon.py": "tier1x",
}
