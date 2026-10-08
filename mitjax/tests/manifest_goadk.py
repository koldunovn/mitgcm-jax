"""Manifest fragment of the GOADK sub-lane (M2 sub-lane GOADK, plan Task 24; owner: lane GOADK)."""

MANIFEST = {
    # GM advective (bolus) form of global_ocean.90x40x15/code_ad: GMREDI_CALC_TENSOR (+ CALC_PSI_BOLUS, SLOPE_PSI,
    # K3D, ExtraDiag) and GMREDI_RESIDUAL_FLOW bitwise vs lane A's dumps-on runs of the four input_ad variants,
    # negative controls, gradient finiteness / FD, unported options
    "mitjax/tests/test_goadk.py": "tier1x",
    # session 3: global_ocean.90x40x15/input_ad* through the driver Model: free-running steps vs every dumped stage
    # (jdon, ctrlxx), whole 10-step runs vs the oracle STDOUT (%MON, cg2d, COST_FINAL)
    "mitjax/tests/test_goadk_model.py": "tier1x",
    # session 4: the adjoint of input_ad vs TAF (admCst, admGrd, FD vs the oracle, trust protocol, negative control;
    # ~45 min, ~31 GB: one gradient compile ~27 min) and the control map of all four variants vs the planted Models
    "mitjax/tests/test_goadk_adjoint.py": "tier1x",
    # session 5: the adjoint of the other three variants vs TAF, one file each (own process; ~45 min, ~31 GB each):
    # admGrd >= 10 digits, FD lines, dot test, planted weight error (helpers goadk_adjoint_gate.py)
    "mitjax/tests/test_goadk_adjoint_kapgm.py": "tier1x",
    "mitjax/tests/test_goadk_adjoint_kapredi.py": "tier1x",
    "mitjax/tests/test_goadk_adjoint_bottomdrag.py": "tier1x",
}
