"""Manifest fragment of the R4 global_ocean.90x40x15 lane (M1 lane GO, plan Tasks 15a/15c). No tier-1 test yet (the
lane's one tier-1 slot is kept for the whole-run smoke once GM/Redi and the periodic forcing preload, lane R5, are
merged)."""

MANIFEST = {
    # teacher-forced substep gates of steps 36000-36002 (S01, S06 incl. the CD chain, S07, S08, S12, G00 R), zero
    # denominator count + planted zero, negative controls, gradient finiteness + FD (~3-5 min on a compute node)
    "mitjax/tests/test_r4a_global_ocean.py": "tier1x",
    # session 3: implicit vertical advection (input_ad teacher-forced T13/T23) and advect_xz/input.nlfs in-model
    "mitjax/tests/test_implicit_vert_adv.py": "tier1x",
    "mitjax/tests/test_r3_nlfs.py": "tier1x",
    # session 4: GAD_DST2U1_IMPL_R and SOLVE_PENTADIAGONAL vs the gfortran replay harness reference/replay_go
    "mitjax/tests/test_replay_go.py": "tier1x",
    # session 5 (Task 15c): global_ocean in-model steps vs every dumped stage, the whole run (%MON, %SBO, digits,
    # pickups), P=4 == P=1 for the whole run (~10 min on a compute node)
    "mitjax/tests/test_r4c_global_ocean_run.py": "tier1x",
    # session 6 (Task 18): docs/M1_ACCEPTANCE.md's table re-derived from the acceptance jobs' files (text, seconds)
    "mitjax/tests/test_m1_acceptance_doc.py": "tier1x",
    # session 7: P=N == P=1 whole runs of the job-only variants, carry finiteness (named NaN storage), the one-step
    # VJP finite on every lane (~5 min); the three advect_xz whole runs at P=2 (~9 min)
    "mitjax/tests/test_m1_pn.py": "tier1x",
    "mitjax/tests/test_m1_pn_advect_xz.py": "tier1x",
    # session 8 (M2 Task 25): global_ocean.cs32x15 front to S04, CG2D without RHS normalisation, SOLVE_FOR_PRESSURE,
    # steps 72000-72002 bitwise at every dumped stage (~4 min)
    "mitjax/tests/test_cs32_go.py": "tier1x",
    # the cs32x15 whole run through the driver (%MON, digits, pickups) and P=6 == P=1 (subprocess; ~10 min)
    "mitjax/tests/test_cs32_go_run.py": "tier1x",
    # session 9: global_ocean.cs32x15/input_ad forward (3 steps bitwise at every stage, whole run %MON + COST_FINAL
    # lines; ~9 min)
    "mitjax/tests/test_cs32_ad_go.py": "tier1x",
    # session 10/11 (CS32AD): the input_ad adjoint vs TAF with the run's own CG2D derivative setting (>= 10 digits,
    # planted cost weight), the exact option (fc bitwise, gradient differs; vs FD h-sweep), FD lines, dot test,
    # repeat (~30 min)
    "mitjax/tests/test_cs32_ad_adjoint.py": "tier1x",
    # session 11: the CG2D derivative switch (mitjax/ad/modes.py): setting per run, forward byte-identical, effect on
    # the live cs32x15/input_ad CG2D, empty switch with cg2dFullAdjoint (jaxpr sha256) + its negative control, dot test
    "mitjax/tests/test_ad_modes_cg2d.py": "tier1x",
}
