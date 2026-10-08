"""M1 acceptance (plan Task 18), GO lane: P=2 == P=1 for the whole runs of advect_xz/input, input.pqm and input.nlfs
(200 steps each; before: job evidence only, docs/M1_ACCEPTANCE.md), and their final carries finite except the named
NaN storage (go_gate.NAN_STORAGE: dWtransC/U/V, never written nor read as momStepping = .FALSE.). The run driver's
Model and step at P = 1 (jit) and inside jit(shard_map(check_vma=True)) on 2 fake CPU devices: every carry leaf and
every per-step output bit for bit (go_gate.whole_run_pn). About 9 min on a CPU compute node (see test_m1_pn.py for
the other variants).
"""

import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

from mitjax.tests import go_gate as G  # noqa: E402


@pytest.mark.parametrize("inp", ["input", "input.pqm", "input.nlfs"])
def test_p2_equals_p1_whole_run_and_nan_storage(inp):
    m, c1, o1, cN, oN, ndiff, same = G.whole_run_pn("advect_xz", inp, 2)
    assert len(o1) == m.prm.time.nTimeSteps == 200
    assert ndiff == 0 and same, (ndiff, same)
    assert G.nonfinite_leaves(c1) == G.expected_nonfinite("advect_xz", m.cfg.size)
