"""Manifest fragment of lane M4COL (M4 step 3: 1D_ocean_ice_column/input; owner: lane M4COL)."""

MANIFEST = {
    # pkg/cal + pkg/exf of 1D_ocean_ice_column/input: calendar and EXF start times vs the oracle's STDOUT,
    # EXF_GETFORCING bitwise vs the dump stages X01-X06 at iterations 0-2 (teacher-forced and own chain), measured
    # negative controls, gradient finiteness / dot test / FD. No tier-1 test (tier 1 is at its budget).
    "mitjax/tests/test_m4col_exf.py": "tier1x",
    # pkg/seaice of 1D_ocean_ice_column/input (session 2): SEAICE_READPARMS vs the STDOUT summary, SEAICE_INIT_FIXED /
    # INIT_VARIA vs I00b at iteration 0, DYNSOLVER+OSTRES, SEAICE_REG_RIDGE, SEAICE_GROWTH (+SOLVE4TEMP,
    # BUDGET_OCEAN), SEAICE_MODEL bitwise vs I00b..P13 at iterations 0-2 (teacher-forced and own chain), measured
    # negative controls, gradient finiteness / dot test / FD.
    "mitjax/tests/test_m4col_seaice.py": "tier1x",
    # session 3: the wiring of pkg/cal, pkg/exf, pkg/seaice in the driver Model and FORWARD_STEP: the front of the step
    # (S00..P01) bitwise vs the dumps at iterations 0-2, negative controls, the MONITOR / EXF / SEAICE monitor
    # records vs the oracle STDOUT, gradient smoke (finite, dot test, FD) w.r.t. theta. Session 4: steps 0-2 free and
    # teacher-forced at every dumped stage (KPP P07/P10, CG2D C01/C02), negative controls of the new arms (pkg/kpp,
    # JMD95Z FIND_ALPHA/BETA, DST3), the whole run (%MON, digits, pickups), the restart and its controls, gradient
    # smoke over whole steps.
    "mitjax/tests/test_m4col_column.py": "tier1x",
}
