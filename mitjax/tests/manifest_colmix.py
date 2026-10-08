"""Manifest fragment of the M3 sub-lane COLMIX (pkg/pp81, pkg/my82, pkg/opps; owner: COLMIX lane). All tier1x (no
tier-1 test: the tier-1 budget is reserved for the per-variant smoke tests)."""

MANIFEST = {
    # replay gates of PP81 / MY82 / OPPS on the three COLMIX harness runs (reference/replay_colmix/CURRENT): READPARMS,
    # INIT_VARIA, CALC, CALC_DIFF, CALC_VISC, OPPS_INTERFACE/OPPS_CALC bitwise (JMD95Z passes; MDJWF passes
    # xfail(strict) until FIND_RHO's MDJWF is ported), negative controls, gradients (finite, dot test, FD); minutes
    "mitjax/tests/test_colmix.py": "tier1x",
}
