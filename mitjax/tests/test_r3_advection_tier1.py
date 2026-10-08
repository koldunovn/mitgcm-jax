"""R3 advect_xy / advect_xz: the lane's two tier-1 smoke runs (plan Task 14; helpers in advect_gate.py): the whole
advect_xy/input run (SOM 80 for theta, DST3FL 33 multi-dimensional for salt, 2 tiles) and the whole advect_xz/input
run (PPM-WENO 42 multi-dimensional for theta, SOM 81 for salt, sNy = 1 < OLy) through the run driver's Model and
forward: every %MON record and MONITOR banner identical to the oracle STDOUT, tools/testreport_jax.py digits vs
results/ >= the yardstick (the oracle's own digits) and vs the oracle STDOUT >= 16; the end-of-run pickups
(pickup.ckptA.*, pickup_somT/S.ckptA.*: WRITE_PICKUP incl. its AB3 branch, GAD_WRITE_PICKUP) byte for byte equal to
the oracle's."""

import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()


@pytest.mark.parametrize("v", ["advect_xy/input", "advect_xz/input"])
def test_whole_run_monitor_and_digits(v):
    from mitjax.tests import advect_gate as ag
    exp, inp = v.split("/")
    m, res, o = ag.whole_run(exp, inp, "tier1")
    diffs, rows, nblocks = ag.run_verdict(exp, inp, res, o)
    assert not any(diffs.values()), diffs
    assert nblocks == sum("%MON time_tsnumber" in r for r in o.raw) > 1, nblocks
    assert rows
    for name, ours, yard, vs_oracle in rows:
        assert ours >= yard, (name, ours, yard)
        assert vs_oracle >= 16, (name, vs_oracle)
    same, extra = ag.pickup_diffs(m, o)             # end-of-run pickups byte for byte (WRITE_PICKUP, GAD_WRITE_PICKUP)
    assert same and all(same.values()) and not extra, (same, extra)
