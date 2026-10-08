"""The static adjoint options of lane M4ADCS32ICE session 2 refuse the combinations that would be silently wrong
(global_ocean.cs32x15 input_ad.seaice / input_ad.seaice_dynmix Models; no gradient program, ~2 min on a CPU node).

* mitjax/ad/modes.lsr_derivative: the default for a build without SEAICE_LSR_ADJOINT_ITER is "forward_only"; the
  explicit "sweeps" (A1) is refused when SEAICElinearIterMax exceeds SOLV_MAX_FIXED = 500 or is not a host integer
  (the scan would truncate the Fortran loop); an unknown option is refused.
* drivers/ad_switches.with_run_switches: sets exactly the hooks of the run's data.autodiff (input_ad.seaice:
  useApproxAdvectionInAdMode -> GAD "routine", sea ice "flux"; input_ad.seaice_dynmix: SEAICEuseFREEDRIFTswitchInAd ->
  mjx_freedrift_in_ad), and refuses a switch without a hook (planted in data.autodiff: SEAICEapproxLevInAd = 1,
  viscFacInAd = 2., SEAICEuseDYNAMICSswitchInAd = .TRUE.), and the free-drift switch where the forward runs the free
  drift (planted SEAICEuseFREEDRIFT = .TRUE. on the parameters).
"""

from types import SimpleNamespace

import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

from mitjax.tests import m4adcs32ice_gate as A  # noqa: E402


@pytest.fixture(scope="module", params=["seaice", "dynmix"])
def mv(request):
    return request.param, A.run(request.param).m


def test_lsr_derivative_options(mv):
    from mitjax.ad.modes import lsr_derivative, with_lsr_derivative
    _, m = mv
    sp = m.arrays.pkc["sp"]
    assert lsr_derivative(m.cfg, sp) == "forward_only"
    assert lsr_derivative(m.cfg, with_lsr_derivative(sp, "sweeps")) == "sweeps"
    with pytest.raises(RuntimeError, match="SOLV_MAX_FIXED"):
        lsr_derivative(m.cfg, with_lsr_derivative(sp.replace(SEAICElinearIterMax=501), "sweeps"))
    with pytest.raises(RuntimeError, match="host integer"):
        lsr_derivative(m.cfg, with_lsr_derivative(sp, "sweeps").replace(SEAICElinearIterMax=200.0))
    with pytest.raises(ValueError):
        with_lsr_derivative(sp, "implicit")


def test_run_switches_set(mv):
    from mitjax.drivers.ad_switches import with_run_switches
    v, m = mv
    mo = with_run_switches(m, m.arrays)
    sp = mo.pkc["sp"]
    got = (mo.params.static_items().get("mjx_approx_advection_in_ad"), getattr(sp, "mjx_approx_advection_in_ad", None),
           getattr(sp, "mjx_freedrift_in_ad", None))
    assert got == {"seaice": ("routine", "flux", None), "dynmix": (None, None, True)}[v], got
    assert m.arrays.params.static_items().get("mjx_approx_advection_in_ad") is None        # the input unchanged


@pytest.mark.parametrize("plant", [("SEAICEapproxLevInAd", (1, "int")), ("viscFacInAd", (2.0, "float")),
                                   ("SEAICEuseDYNAMICSswitchInAd", (True, "bool"))])
def test_run_switches_refuse_unhooked(mv, plant):
    from mitjax.drivers.ad_switches import with_run_switches
    from mitjax.drivers.model import with_namelist
    _, m = mv
    exp = with_namelist(m.exp, {("data.autodiff", "AUTODIFF_PARM01", plant[0]): plant[1]})
    with pytest.raises(NotImplementedError, match="no backward-only hook"):
        with_run_switches(SimpleNamespace(exp=exp, cfg=m.cfg), m.arrays)


def test_freedrift_switch_refused_when_forward_is_free_drift(mv):
    from mitjax.drivers.ad_switches import with_run_switches
    v, m = mv
    if v != "dynmix":
        pytest.skip("input_ad.seaice does not set SEAICEuseFREEDRIFTswitchInAd")
    sp = m.arrays.pkc["sp"].replace(SEAICEuseFREEDRIFT=True)
    with pytest.raises(NotImplementedError, match="SEAICEuseFREEDRIFT = .TRUE."):
        with_run_switches(m, m.arrays.replace(pkc={**m.arrays.pkc, "sp": sp}))
