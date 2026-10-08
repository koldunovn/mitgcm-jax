"""Manifest fragment of the M1 core lane's Task 11 and Task 12 tests."""

MANIFEST = {
    # initial State bitwise vs S00_begin / G00 group R for every M1 variant (global_ocean xfail strict: open signed
    # zeros in halos), negative controls (tRef one ulp, no exchange, swapped pickup fields), INI_PARMS time values
    # vs the printout, State pytree (~25 tests, a few minutes)
    "mitjax/tests/test_init.py": "tier1x",
    # time counters vs %MON time_tsnumber/time_secondsf for every variant, negative control, split-run identity
    # (~11 tests, ~1 min)
    "mitjax/tests/test_restart.py": "tier1x",
    # R1 tutorial_barotropic_gyre: substeps of steps 1-3, initial state, negative controls, gradient (~1-2 min)
    "mitjax/tests/test_r1_barotropic_gyre.py": "tier1x",
    # R1 through the run driver: scan == step loop (+ late-start control), pickup round trip, restart gate 10 == 5 +
    # pickup + 5 (+ dropped / zeroed GuNm1 controls) (~5 tests, several minutes: three scan programs, four runs)
    "mitjax/tests/test_r1_driver.py": "tier1x",
    # R1 gradients through the scan + checkpoint drivers: full-carry finiteness (step == sqrt), FD h-sweeps of one
    # etaN / uVel point (cg2dTargetResidual 1e-13 copy), TL vs adjoint dot test (~4 tests, several minutes)
    "mitjax/tests/test_r1_gradient.py": "tier1x",
    # R1: the variant's ONE tier-1 smoke test: `python -m mitjax run` path, 10 steps, STDOUT records vs the oracle,
    # testreport digits, pickup.ckptA bytes vs the oracle
    "mitjax/tests/test_r1_barotropic_gyre_tier1.py": "tier1",
}
