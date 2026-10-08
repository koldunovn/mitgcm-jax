"""Manifest fragment of lane M4LAB (M4 step 5: lab_sea/input; owner: lane M4LAB)."""

MANIFEST = {
    # session 1: the exchange map, the start from the pickups and the front (G00, S00..S02 at iteration 1 free,
    # 2-3 teacher-forced), SEAICE_INIT_FIXED/INIT_VARIA from pickup_seaice incl. the spherical metric terms (I00),
    # FORCING_SURF_RELAX without restoring under ice (P01, teacher-forced), the SEAICE_LSR build-option guard; each
    # with a measured negative control. No tier-1 test (tier 1 is at its budget).
    "mitjax/tests/test_m4lab_init.py": "tier1x",
    # session 2: the ocean physics teacher-forced at P13 (P01 .. S05: KPP SWFRAC and salt-plume argument arms,
    # KPP_TRANSPORT_T/S GMREDI + salt-plume terms, GMREDI_CALC_TENSOR with KPPhbl, 'ldd97', no diagonal options,
    # LONGSTEP checks), each with a measured negative control or an asserted blind spot
    "mitjax/tests/test_m4lab_ocean.py": "tier1x",
    # session 3: the sea-ice gaps L4-L12 teacher-forced (SEAICE_GROWTH 7 categories + saltPlumeFlux, OS7MP advection,
    # SEAICE_FREEDRIFT, the plain ZEBRA LSR with bottom drag / metric terms, the whole SEAICE_MODEL from I00), each
    # with a measured negative control or an asserted blind spot
    "mitjax/tests/test_m4lab_seaice.py": "tier1x",
    # session 4: L3 ALLOW_SITRACER and lab_sea/input end to end (the 9-step run through the run driver: MONITOR
    # records incl. seaice_sitracer, solver lines, digits, the three pickups; the ADVCAP control; restart; P=4/2 == P=1)
    "mitjax/tests/test_m4lab_run.py": "tier1x",
    # session 4: derivatives of the SITRACER kernels (SEAICE_TRACER_PHYS, SEAICE_ADVDIFF's tracer part): finite on
    # every lane, dot test, FD at a smooth state, the FD control
    "mitjax/tests/test_m4lab_sitracer.py": "tier1x",
}
