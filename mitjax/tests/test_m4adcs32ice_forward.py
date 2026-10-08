"""global_ocean.cs32x15 input_ad.seaice / input_ad.seaice_dynmix forward (M4 step 7, last item; lane M4ADCS32ICE):
the code_ad build's FORWARD_STEP through the driver Model on the cube vs lane A's dumps (helpers:
mitjax/tests/m4adcs32ice_gate.py; oracle job27855988-jdon of each variant, the standard oracle: no subnormal met).

* every dumped stage of every dumped iteration (36000-36001, input_ad.seaice; 36000-36002, input_ad.seaice_dynmix),
  stepped from the Model's own initial carry (no teacher forcing), every point of every tile incl. halos and cube
  corners, bit patterns (S18_cost_tile: cost.h);
* negative controls that bite where the gaps of this lane live (measured, never assumed): ALLOW_BULK_LARGEYEAGER04
  planted on (EXF_BULKFORMULAE's solve4Stress = .TRUE. arm, exf_bulkformulae.F:233-234: X04 must differ) and the
  sea-ice scheme planted from 33 to 30 (input_ad.seaice) / 30 to 33 (input_ad.seaice_dynmix) (SEAICE_ADVECTION's
  GAD_DST3FL / GAD_DST3 arms, seaice_advection.F:385-392: I02_advdiff must differ).
Costs: about 5-10 min per variant on a CPU compute node (two compiles of the step per test).
"""

import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

from mitjax.tests import m4adcs32ice_gate as A  # noqa: E402

KEY_STAGES = {"S00_begin", "X04_exf_bulkformulae", "S03_ctrl_map_forcing", "I00_seaice_begin", "Y03_freedrift",
              "Y06_lsr", "I01_dynsolver", "I02_advdiff", "I04_growth", "P13_seaice_model", "D00c_mom_vecinv",
              "S06_dynamics", "T13_temp_impl", "T23_salt_impl", "S16_blocking_exchanges", "S18_cost_tile"}


@pytest.fixture(autouse=True)
def _drop_compiled():
    """Release the compiled programs between tests (PORTING_LESSONS "vm.max_map_count")."""
    import gc

    import jax
    yield
    jax.clear_caches()
    gc.collect()


@pytest.mark.parametrize("v", ["seaice", "dynmix"])
def test_steps_bitwise(v):
    r = A.run(v)
    bad, ncmp = A.run_compare(r)
    print(v, "stages", len(ncmp), "fields", sum(ncmp.values()), "bad", {k: sorted(b)[:8] for k, b in bad.items()})
    for it in r.its:
        got = {s for (i, s) in ncmp if i == it and ncmp[(i, s)] > 0}
        assert KEY_STAGES <= got, (it, sorted(KEY_STAGES - got))
    assert bad == {}, {k: sorted(b)[:8] for k, b in bad.items()}
    assert sum(ncmp.values()) > 1000


@pytest.mark.parametrize("v", ["seaice", "dynmix"])
def test_control_bulkformulae_ly04(v):
    from mitjax.tests import m4cs32ice_gate as G
    r = A.run(v)
    bad, _ = A.step(G.planted_cpp(r, on=("ALLOW_BULK_LARGEYEAGER04",)), r.its[0], until="S02_load_fields")
    assert "X04_exf_bulkformulae" in bad, sorted(bad)


@pytest.mark.parametrize("v,planted", [("seaice", 30), ("dynmix", 33)])
def test_control_seaice_advection_scheme(v, planted):
    r = A.run(v)
    rp = A.planted_model(r, v, {("data.seaice", "SEAICE_PARM01", "SEAICEadvScheme"): planted})
    assert rp.m.arrays.pkc["sp"].SEAICEadvSchHeff == planted
    bad, _ = A.step(rp, r.its[0], until="I04_growth")
    assert "I02_advdiff" in bad, sorted(bad)
