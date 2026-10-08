"""R3 advect_xy/input: the lane's one cheap tier-1 check (plan Task 14; < 30 s incl. compile): steps 1-3 of
advect_xy/input (SOM 80 for theta, GAD_ADVECTION with DST3FL 33 and GAD_MULTIDIM_COMPRESSIBLE for salt, 2 tiles,
the SOM moments carried and exchanged) at the stages T20_salt_adv (the multi-dimensional tendency), T11_temp_gT (the
SOM tendency) and S16_blocking_exchanges (State after the step's exchanges): every field bitwise on every point incl.
halos. The full set of gates is tier 1x (test_r3_advection.py, test_r3_advection_tier1.py)."""

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()


def test_advect_xy_steps_1_to_3_bitwise():
    from mitjax.tests import advect_gate as ag
    m = ag.model("advect_xy", "input")
    stages = ("T20_salt_adv", "T11_temp_gT", "S16_blocking_exchanges")
    res, _ = ag.run_steps(m, m.params, 3, stages=stages)
    assert len(res) == 3
    for k, per_stage in enumerate(res):
        assert set(per_stage) == set(stages), (k, sorted(per_stage))
        for st, r in per_stage.items():
            assert r, (k, st)
            assert not ag.bad(r), (k, st, ag.bad(r))
