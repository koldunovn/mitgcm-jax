"""Grid gate in tier 1 (plan Task 10): the barotropic gyre's grid, every GRID.h field the oracle dumps, bitwise on all
points (halos and land included). The per-variant gates and the negative controls are in test_grid.py (tier1x)."""

from mitjax.tests import grid_gate as gg


def test_grid_bitwise_barotropic_gyre():
    exp, inp = "tutorial_barotropic_gyre", "input"
    result, _ = gg.compare(gg.build_grid(exp, inp), exp, inp)
    bad = gg.failures(result)
    assert not bad, f"fields differing from the oracle (n, ndiff, nonfinite ours, nonfinite oracle, ndiff bits): {bad}"
    assert len(result) >= 62 and sum(v[0] for v in result.values()) == 243945
