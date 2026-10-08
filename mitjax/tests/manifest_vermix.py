"""Manifest fragment of the vermix integration (M3 Task 30; owner: lane KPP/vermix)."""

MANIFEST = {
    # vermix variants end to end through the run driver: initial state, steps 0-2 at every dumped stage, whole run
    # (%MON, digits, pickups); the MDJWF variants are strict xfails until the EOS is ported
    "mitjax/tests/test_vermix.py": "tier1x",
}
