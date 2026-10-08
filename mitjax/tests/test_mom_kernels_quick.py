"""Quick MOM-lane gate (tier1x): the leaf kernels of pkg/mom_common and pkg/mom_fluxform bitwise against the
gfortran replay harness (reference/replay_mom) on tutorial_barotropic_gyre's Cartesian grid (1 tile, Nr = 1): all 79
outputs, every point incl. halos and the points a routine does not write, element-wise equal, equal bit patterns and
finite, under the gate XLA flags with the REAL parameters traced. The other experiments, negative controls and
gradient gates are in test_mom_kernels.py. Fails, does not skip, without the replay run.
"""

from mitjax.tests.test_mom_kernels import bitwise


def test_mom_replay_bitwise_barotropic_gyre():
    bad, bits = bitwise("tutorial_barotropic_gyre")
    assert bad == {} and bits == {}, (bad, bits)
