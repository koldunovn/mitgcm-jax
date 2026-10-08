"""Manifest fragment of lane M4ADLAB (M4 step 7: lab_sea/input_ad against TAF; owner: lane M4ADLAB)."""

MANIFEST = {
    # session 1: the targets (TAF's output_adm.txt = the FD oracle job27856073 in every printed grdchk digit; TAF's
    # adjoint gradients recorded) and TAF's adjoint mode (= the forward), each with a planted negative control.
    # Seconds, file reads only; no tier-1 test.
    "mitjax/tests/test_m4adlab_targets.py": "tier1x",
    # session 2: the sea-ice arms of the code_ad build (G6 ALLOW_AUTODIFF arms of SEAICE_LSR / DYNSOLVER, G2
    # SEAICE_DIFFUSION), teacher-forced bitwise on input_ad's oracle, and the finite reverse pass of one LSR call
    # (the NaN cause: SEAICE_OCEANDRAG_COEFFS's unguarded SQRT). Minutes (LSR compiles, a 500-step scan gradient).
    "mitjax/tests/test_m4adlab_seaice.py": "tier1x",
    # session 3: model/src and eesupp arms of the code_ad build (CG2D_SINGLECPU_SUM, ...), teacher-forced bitwise.
    "mitjax/tests/test_m4adlab_model.py": "tier1x",
    # session 4: the whole 4-step run on the real Model (pkg/ecco 2-D gencost, pkg/ctrl genarr2d): %MON, every
    # COST_FINAL line, the cost / ecco files and pickups byte-identical; negative controls by COST_FINAL replays.
    # About 10 min (one run).
    "mitjax/tests/test_m4adlab_run.py": "tier1x",
    # session 4: the ten grdchk perturbations (xx_atemp record 1, eps 1e-3) against the FD oracle job27856073 in
    # every printed digit of global fc. About 40 min (ten runs).
    "mitjax/tests/test_m4adlab_fd.py": "tier1x",
    # session 5: the adjoint (xx_atemp record 1): admGrd = the FD of the forward (1e-5; controls: planted weight,
    # TAF's own values), admGrd vs TAF measured (3-4 digits) and the 10-digit match a strict xfail. About 25 min.
    "mitjax/tests/test_m4adlab_adjoint.py": "tier1x",
}
