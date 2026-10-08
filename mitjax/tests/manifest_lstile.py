"""Manifest fragment of lane LSTILE (docs plan 20261006 S4: lab_sea one-tile oracle check; owner: lane LSTILE)."""

MANIFEST = {
    # mitjax's one-tile lab_sea/input against the retile oracle (four-tile gate criterion: MONITOR/solver records,
    # digits, pickups), its four-tile-SIZE.h negative control, the Fortran's own tiling dependence; no tier-1 test
    "mitjax/tests/test_lstile_run.py": "tier1x",
}
