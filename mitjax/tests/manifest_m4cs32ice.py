"""Manifest fragment of lane M4CS32ICE (M4 step 6: global_ocean.cs32x15/input.seaice; owner: lane M4CS32ICE)."""

MANIFEST = {
    # session 1: the EXF front of the first step (S00 .. S02 at iteration 36000 from the pickups: useAtmWind .FALSE.,
    # the C-grid stress on the cube, Large&Yeager04, records without pkg/cal, runoff temperature), measured
    # negative controls (LY04 off, RUNOFTEMP off), the tides blind spot. No tier-1 test (tier 1 is at its budget).
    "mitjax/tests/test_m4cs32ice_exf.py": "tier1x",
    # session 2: the sea ice on the cube (metric terms, every SEAICE_MODEL routine teacher-forced at 36000-36002 with
    # measured controls, the LSR underflow-band inconsistency, SEAICE_MODEL, the whole first step from the pickups)
    "mitjax/tests/test_m4cs32ice_seaice.py": "tier1x",
    # session 2: the 10-step run through the run driver (%MON, digits, solver lines, pickups), restart, P=6/P=2 == P=1
    "mitjax/tests/test_m4cs32ice_run.py": "tier1x",
}
