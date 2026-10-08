"""Compact replay gate of the GAD-A kernels (one test of ~10 s, tier 1x like the full gates in test_gad_simple.py;
written for tier 1, moved to tier 1x when tier 1 reached its budget, coordinator 2026-10-01): on the
advect_xz/input.nlfs harness run (reference/replay_gad_a/CURRENT; 2 tiles, Nr = 20, OLx = OLy = 4 > sNy = 1), every
GAD-A routine as the harness calls it is bitwise equal to gfortran on all points incl. halos and unwritten points
(levels 1, 2, 3 and Nr for the per-level routines; all levels for the implicit-matrix routines, which accumulate over
k). Fails, not skips, when the run is missing.
"""

import numpy as np

from mitjax import paths
from mitjax.tests import gad_a_replay as G


def test_gad_simple_replay_advect_xz_bitwise():
    line = [ln.split() for ln in (G.REPO / "reference" / "replay_gad_a" / "CURRENT").read_text().splitlines()
            if ln.startswith("advect_xz ")]
    assert len(line) == 1, line
    exp, inp, rel = line[0]
    R = G.Replay(exp, inp, paths.REFERENCE / rel)
    levels = (1, 2, 3, R.cfg.Nr)
    bad, nonfinite = {}, []
    for case in G.CASES:
        lv = None if case in G.IMPL else levels
        ks = slice(None) if lv is None else [k - 1 for k in lv]
        for name, o in zip(G.case_outputs(case), G.run_case(R, case, levels=lv)):
            n = G.bit_diff(o, R.out[name][:, ks])
            if n:
                bad[name] = n
            if not np.all(np.isfinite(o)):
                nonfinite.append(name)
    assert bad == {} and nonfinite == [], (bad, nonfinite)
