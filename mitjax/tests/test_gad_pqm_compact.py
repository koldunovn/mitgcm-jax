"""Tier-1 gate of lane GAD-C: the PPM/PQM schemes advect_xz runs, bitwise against the gfortran replay harness
reference/replay_gad_c (the full set of gates is mitjax/tests/test_gad_pqm.py, tier1x)."""

import pytest

from mitjax.tests import gad_c_replay as gr
from mitjax.tests.test_gad_pqm import OUT, mismatches


@pytest.fixture(scope="module")
def R():
    return gr.current_replay()


def test_gad_c_used_schemes_bitwise(R):
    """Tier 1: the schemes advect_xz runs (PPM 42 for theta of `input`; PQM 51/52 of `input.pqm`; calc_CFL = T),
    X/Y on three levels (surface, interior, bottom) and R, bitwise on all points against gfortran."""
    bad = {}
    for case in gr.USED:
        n = mismatches(R, case, None if case[2] is None else (1, 9, R.cfg.Nr))
        if n:
            bad[OUT[case]] = n
    assert bad == {}, bad
