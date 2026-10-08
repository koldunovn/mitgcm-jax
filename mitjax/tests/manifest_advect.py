"""Manifest fragment of the M1 ADVECT lane (plan Task 14, R3 advect_xy / advect_xz)."""

MANIFEST = {
    # R3: initial state, every dumped stage of steps 1-3 and the later MONITOR step of the jaxdump3 runs bitwise for
    # advect_xy/input, input.ab3_c4, advect_xz/input, input.pqm; negative controls; gradients per scheme (finite,
    # FD); whole runs vs the oracle STDOUT and results/ (~25 tests, compile-heavy: PPM/PQM/SOM at Nr = 20)
    "mitjax/tests/test_r3_advection.py": "tier1x",
    # R3: whole runs of advect_xy/input and advect_xz/input: %MON blocks and digits. Moved from tier1 to tier1x by the
    # main session: master tier 1 timed out at 20 min with them (jobs 27832396, 27832475; budget 10 min)
    "mitjax/tests/test_r3_advection_tier1.py": "tier1x",
    # R3: the one cheap tier-1 check (< 30 s incl. compile): advect_xy/input steps 1-3 at T20, T11, S16 bitwise
    "mitjax/tests/test_r3_advection_quick.py": "tier1",
}
