"""Lane M4ADCOL (M4 step 7, session 1): set-up pieces of 1D_ocean_ice_column/input_ad (code_ad build).

Seconds-scale checks of the run-time parameters the input_ad port reads and of the guard that keeps a JAX gradient
comparable with TAF's: SEAICE_PARM02 (seaice_readparms.F:615-645, :682-684, :1355-1384) and the adjoint-mode check
(pkg/autodiff/autodiff_readparms.py: TAF's ADAUTODIFF_INADMODE_SET must leave the model as the forward runs it).
Each check has a planted negative control. Session 3: pkg/ecco's ECCO_READPARMS and the Model's acceptance of useECCO
(the forward: test_m4adcol_forward.py, test_m4adcol_ecco.py)."""

import pytest

EXP = ("1D_ocean_ice_column", "input_ad")


def _exp(overrides=None):
    from mitjax.config.params import load
    from mitjax.drivers.model import with_namelist
    e = load(*EXP)
    return with_namelist(e, overrides) if overrides else e


def test_seaice_parm02_values_and_retired_control():
    from mitjax.params_io import RunParams
    from mitjax.pkg.seaice.seaice_readparms import _cost_parm02
    e = _exp()
    v = _cost_parm02(e, RunParams(e.run))
    # data.seaice:39-49 (input_ad): mult_ice = 1., cost_ice_flag = 2; the others at their defaults :621-629
    assert v == {"mult_ice_export": 0.0, "mult_ice": 1.0, "cost_ice_flag": 2, "SEAICE_cutoff_area": 0.0001,
                 "SEAICE_cutoff_heff": 0.0}
    # negative control: a retired parameter (:631, :1370-1376) stops the run
    bad = _exp({("data.seaice", "SEAICE_PARM02", "SEAICE_clamp_salt"): (35.0, float)})
    with pytest.raises(RuntimeError, match="retired SEAICE_PARM02"):
        _cost_parm02(bad, RunParams(bad.run))


def test_adjoint_mode_equals_forward_and_controls():
    from mitjax.pkg.autodiff.autodiff_readparms import adjoint_mode_check
    v = adjoint_mode_check(_exp())
    # data.autodiff of input_ad sets only dumpAdByRec: every adjoint-mode switch at its default (:82-98, :145-159)
    assert v["inAdExact"] is True and v["useKPPinAdMode"] is True and v["useSEAICEinAdMode"] is True
    assert v["SEAICEapproxLevInAd"] == 0 and v["useApproxAdvectionInAdMode"] is False
    # negative controls: settings under which TAF's backward pass differs from the forward
    for ov, msg in (({("data.autodiff", "AUTODIFF_PARM01", "inAdExact"): (False, bool)}, "inAdExact"),
                    ({("data.autodiff", "AUTODIFF_PARM01", "SEAICEapproxLevInAd"): (1, int)}, "SEAICEapproxLevInAd"),
                    ({("data.autodiff", "AUTODIFF_PARM01", "useKPPinAdMode"): (False, bool)}, "useKPPinAdMode")):
        with pytest.raises(NotImplementedError, match=msg):
            adjoint_mode_check(_exp(ov))


def test_input_ad_accepted_with_ecco_and_readparms():
    """Session 3: pkg/ecco is ported (mitjax/pkg/ecco): the integrated Model accepts useECCO; ECCO_READPARMS' gencost
    table of data.ecco (theta, salt: flag 1, 3-D, pointer3d 1 / 2, 'month', mult 0). Negative controls: an
    ecco_cost_nml key (no old-style cost term is ported) and an unset gencost_spzero (ecco_readparms.F:890-899)
    raise."""
    from mitjax.drivers.model import check_packages
    from mitjax.pkg.ecco.ecco_readparms import ecco_readparms
    e = _exp()
    check_packages(e.cfg)
    ep = ecco_readparms(e.run, e.cfg)
    got = [(g.k, g.gencost_flag, g.gencost_is3d, g.gencost_pointer3d, g.gencost_barfile, g.gencost_avgperiod,
            g.mult_gencost, g.gencost_spmin, g.gencost_spmax, g.gencost_spzero) for g in ep.gencost if g.using_gencost]
    assert got == [(1, 1, True, 1, "m_theta_month", "month", 0.0, -1.8, 40.0, 0.0),
                   (2, 1, True, 2, "m_salt_month", "month", 0.0, 25.0, 40.0, 0.0)], got
    assert ep.cost_iprec == 32 and not (ep.using_cost_altim or ep.using_cost_sst or ep.using_cost_seaice)
    bad = _exp({("data.ecco", "ECCO_COST_NML", "mult_temp"): (1.0, float)})
    with pytest.raises(NotImplementedError, match="ecco_cost_nml"):
        ecco_readparms(bad.run, bad.cfg)
    import dataclasses
    run = bad.run
    vars_ = dict(e.run.vars)
    key = ("data.ecco", "ecco_gencost_nml", "gencost_spzero")
    vars_[key] = dataclasses.replace(vars_[key], value={(1,): 9876.0, (2,): 0.0})
    with pytest.raises(RuntimeError, match="gencost_spzero not set"):
        ecco_readparms(dataclasses.replace(run, vars=vars_), e.cfg)
