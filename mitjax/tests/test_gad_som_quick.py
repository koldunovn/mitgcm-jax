"""Tier-1 gate of M1 sub-lane GAD-B (one test): every ported SOM routine (GAD_SOM_ADVECT schemes 80/81,
GAD_SOM_ADV_X/_Y limiter 0/1, GAD_SOM_LIM_R, GAD_SOM_EXCHANGES, GAD_EXCH_SOM) element-equal to gfortran on all points
of the advect_xy replay harness run (reference/replay_gad_b/CURRENT). The full gate set (advect_xz, negative controls,
gradients) is test_gad_som.py (tier 1x)."""

from mitjax.tests import gad_som_replay as h


def test_som_replay_bitwise_advect_xy():
    R = h.current_runs()["advect_xy"]
    res = h.run_all(R)
    assert len(res) == 106
    bad = {k: v for k, v in res.items() if v}
    assert bad == {}, f"advect_xy: points differing from gfortran (of {R.out['gTracer_80'].size} each): {bad}"
