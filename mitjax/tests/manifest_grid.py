"""Manifest fragment of the grid and geometry (plan Task 10; owner: M1 core lane)."""

MANIFEST = {
    # barotropic gyre grid bitwise vs the oracle's G00 dump on all points (one test, ~10 s; needs the registered
    # dumps-on run and the exchange map)
    "mitjax/tests/test_grid_tier1.py": "tier1",
    # every M1 variant bitwise on all points, the two negative controls per variant (delR one ulp, no exchange),
    # unported options raise, Grid pytree (~29 tests, a few minutes)
    "mitjax/tests/test_grid.py": "tier1x",
}
