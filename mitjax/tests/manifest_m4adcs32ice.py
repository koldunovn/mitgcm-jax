"""Manifest fragment of lane M4ADCS32ICE (M4 step 7, last item: global_ocean.cs32x15 input_ad.seaice and
input_ad.seaice_dynmix of the code_ad build; owner: lane M4ADCS32ICE)."""

MANIFEST = {
    # session 1: the code_ad forward step of both variants teacher-forced bitwise against lane A's dumps (every
    # stage, every dumped iteration), negative controls on this lane's ported arms (EXF_BULKFORMULAE solve4Stress =
    # .FALSE., SEAICE_ADVECTION DST3 / DST3FL). About 15-20 min (two variants, plants recompile).
    "mitjax/tests/test_m4adcs32ice_forward.py": "tier1x",
    # session 1: the whole forward run of both variants through the run driver (%MON, COST_FINAL lines) and GRDCHK's
    # perturbed costs / finite differences at the 4 xx_theta points in every printed digit of the FD oracle
    # job27856074; planted cost weight as the control. About 25-30 min (two variants, nine cost runs each).
    "mitjax/tests/test_m4adcs32ice_run.py": "tier1x",
    # session 2: the reverse pass of EXF_WIND + EXF_BULKFORMULAE's solve4Stress = .FALSE. roots finite at planted
    # zero-stress cells (atemp = 0 lanes), forward bitwise vs X04; controls: each root unguarded. About 2 min.
    "mitjax/tests/test_m4adcs32ice_nan_guard.py": "tier1x",
    # session 2: the static adjoint options refuse silently wrong combinations (LSR A1 loop bound, data.autodiff
    # switches without a hook); Models only, about 2-3 min.
    "mitjax/tests/test_m4adcs32ice_switches.py": "tier1x",
    # session 2: input_ad.seaice adjoint: exact mode (A1) vs TAF's TLM, FD, dot test; run mode (approximate
    # advection switch) vs TAF's ADM and the switch's effect. About 50-60 min (two gradient programs, the tangent).
    "mitjax/tests/test_m4adcs32ice_adjoint.py": "tier1x",
    # session 2: input_ad.seaice_dynmix adjoint: run mode (free-drift switch) vs TAF's ADM, exact mode vs TAF's TLM,
    # the switch's effect on SEAICE_DYNSOLVER. About 45-55 min (two gradient programs, kernel programs).
    "mitjax/tests/test_m4adcs32ice_dynmix_adjoint.py": "tier1x",
}
