"""Manifest fragment of lane M4ADCOL (M4 step 7: 1D_ocean_ice_column/input_ad against TAF; owner: lane M4ADCOL)."""

MANIFEST = {
    # session 1: SEAICE_PARM02 and the adjoint-mode guard (TAF's ADAUTODIFF_INADMODE_SET equals the forward), with
    # planted negative controls; session 3: ECCO_READPARMS and the Model's acceptance of useECCO. Seconds; no tier-1
    # test.
    "mitjax/tests/test_m4adcol_setup.py": "tier1x",
    # sessions 2-3: the forward of input_ad (useECCO as the run sets it) -- steps 0-2 every dumped stage bitwise,
    # negative controls of the new arms, CTRL_GET_GEN_REC's pkg/cal arm, the 10-step run (%MON, f_ice, fc, pickups).
    # About 25 min on a CPU node (dev job 27881112).
    "mitjax/tests/test_m4adcol_forward.py": "tier1x",
    # session 2: our global fc at the eight grdchk perturbations vs the FD oracle job27856073 in every digit. ~15 min.
    "mitjax/tests/test_m4adcol_fd.py": "tier1x",
    # session 3: pkg/ecco's gencost path -- bar and misfit files byte-identical, f_gencost lines, costfunction_ecco,
    # three planted negative controls (mult_gencost, data record shift, sum1mon). About 10 min (four runs).
    "mitjax/tests/test_m4adcol_ecco.py": "tier1x",
    # session 3: the adjoint vs TAF -- admGrd >= 10 digits at the four grdchk points (measured 12-14), finite, zero
    # off the wet interior, bitwise repeat, a planted weight error fails. About 10 min.
    "mitjax/tests/test_m4adcol_adjoint.py": "tier1x",
}
