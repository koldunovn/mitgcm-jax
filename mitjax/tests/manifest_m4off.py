"""Manifest fragment of lane M4OFF (M4 step 4: offline_exf_seaice/input.thermo, input.dyn_lsr; owner: lane M4OFF)."""

MANIFEST = {
    # session 1: the front of FORWARD_STEP of the -cal build (EXF without pkg/cal, climsst, surf_pRef) X01..S02 at
    # iterations 0-2; the C-grid SEAICE_INIT_FIXED/INIT_VARIA (I00), SEAICE_DYNSOLVER without dynamics (Y01, Y09),
    # SEAICE_ADVDIFF (I02, blind spot asserted), SEAICE_REG_RIDGE without SEAICE_VARIABLE_SALINITY (I03), bitwise,
    # with measured negative controls. Session 2: SEAICE_GROWTH of this build (I04) with controls, blind spots and
    # gradients (finite, dot, FD + control), SEAICE_MODEL I00..P13, the whole step 0-2 at every dumped stage (free
    # and teacher-forced) with a control. No tier-1 test (tier 1 is at its budget).
    "mitjax/tests/test_m4off_seaice.py": "tier1x",
    # session 2: the 120-step run through the run driver (MONITOR records incl. the SEAICE schedule, digits,
    # pickups), the restart with controls, P=4 / P=2 == P=1, the whole-step gradient window (finite, FD + control,
    # dot test, sharded gradient P=4 == P=1)
    "mitjax/tests/test_m4off_run.py": "tier1x",
    # session 3 (input.dyn_lsr): SEAICE_LSR teacher-forced Y04 -> Y06 with its STDOUT lines, SEAICE_MODEL every
    # dynamics stage (Y02 ice strength, Y06 LSR, Y09, I02 PPM advection, P13 no thermodynamics), controls, glibc
    # log10, the forward-only LSR derivative guard
    "mitjax/tests/test_m4off_lsr.py": "tier1x",
    # session 3: the whole step 0-2 at every dumped stage, the 12-step run through the run driver (MONITOR records,
    # LSR lines, digits, pickups), the restart, P=4 / P=2 == P=1
    "mitjax/tests/test_m4off_lsr_run.py": "tier1x",
}
