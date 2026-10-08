"""Initial-state gates (plan Task 11, tier1x): the State INITIALISE_VARIA builds, bitwise (equal bit patterns) against the
oracle's state at the start of the first step (S00_begin; group R of G00_geometry), every point of every tile incl.
halos, for every M1 variant; negative controls measured to bite; the INI_PARMS values against the oracle's printout.

Exemptions are the fields of the INITIALISE_VARIA routines later tasks port (initialise_varia.PENDING_WRITES), for the
routines the variant executes. Measured 2026-10-01 (login node, gate XLA flags): every non-exempt field bitwise in
eight variants; wVel differs only in the sign of zeros in the cold-start variants (written by INTEGR_CONTINUITY,
exempt); global_ocean.90x40x15's halo +0 of uVel, vVel, guNm1, gvNm1 after READ_PICKUP comes from the exch2 vector
exchange arithmetic (sa1*u + sa2*v, mitjax/eesupp/exchange.py apply_rx2) and is bitwise since lane B session 5.
"""

import dataclasses

import numpy as np
import pytest

from mitjax.tests import grid_gate as gg
from mitjax.tests import init_gate as ig

IDS = [f"{e}/{i}" for e, i in gg.VARIANTS]
# global_ocean's pickup-halo +0 (formerly open here) is the exch2 buffer arithmetic sa1*u + sa2*v of READ_PICKUP's
# EXCH_UV_3D_RL (lane A); the exchangers compute it since lane B session 5 (job 27830224: XPASS strict), marker removed.
OPEN = {}


def _cases():
    out = []
    for e, i in gg.VARIANTS:
        marks = [pytest.mark.xfail(strict=True, reason=OPEN[(e, i)])] if (e, i) in OPEN else []
        out.append(pytest.param(e, i, marks=marks, id=f"{e}/{i}"))
    return out


@pytest.mark.parametrize("exp,inp", _cases())
def test_initial_state_bitwise(exp, inp):
    st, pend = ig.build_state(exp, inp)
    res = ig.compare(st, exp, inp)
    exempt = ig.exempt_fields(pend)
    bad = ig.failures(res, exempt)
    assert not bad, f"{exp}/{inp}: (n, ndiff, nonfinite ours, nonfinite oracle, ndiff bits, stage): {bad}"
    for must in ("uVel", "vVel", "theta", "salt", "etaN", "gU", "gV", "totPhiHyd", "rhoInSitu"):
        assert must in res, f"{must} not gated"
    gated = [k for k in res if k not in exempt]
    assert len(gated) >= 15, gated


@pytest.mark.parametrize("exp,inp", [("tutorial_barotropic_gyre", "input"), ("tutorial_baroclinic_gyre", "input")],
                         ids=["tutorial_barotropic_gyre/input", "tutorial_baroclinic_gyre/input"])
def test_negative_control_tref_one_ulp(exp, inp):
    """Planted error: tRef(1) moved by one ulp. Bites: theta differs at the wet points of level 1 (INI_THETA's
    theta = tRef(k), ini_theta.F:59)."""
    prm = ig.params(exp, inp)
    t = np.array(prm.init.tRef, copy=True)
    t[0] = np.nextafter(t[0], np.inf)
    bad_prm = dataclasses.replace(prm, init=dataclasses.replace(prm.init, tRef=t))
    st, pend = ig.build_state(exp, inp, prm=bad_prm)
    bad = ig.failures(ig.compare(st, exp, inp), ig.exempt_fields(pend))
    assert "theta" in bad and bad["theta"][1] > 0, bad


@pytest.mark.parametrize("exp,inp", [("advect_xy", "input"), ("global_ocean.90x40x15", "input")],
                         ids=["advect_xy/input", "global_ocean.90x40x15/input"])
def test_negative_control_no_exchange(exp, inp):
    """Planted error: every exchange returns its input. Bites: advect_xy's theta blob (EXCH_XYZ_RL of
    ini_theta.F:75) and the pickup fields of global_ocean (READ_PICKUP's exchanges) differ at halo points."""
    class NoEx(gg.NoExchange):
        def EXCH_3D_RL(self, phi):
            return phi

        def EXCH_UV_3D_RL(self, u, v, withSigns):
            return u, v
    grid = gg.build_grid(exp, inp)
    st, pend = ig.build_state(exp, inp, grid=grid, ex=NoEx())
    bad = ig.failures(ig.compare(st, exp, inp), ig.exempt_fields(pend))
    assert "theta" in bad and bad["theta"][1] > 0, bad


def test_negative_control_pickup_record():
    """Planted error: the pickup's Uvel read as Vvel (field names swapped in the meta list). Bites: uVel, vVel differ."""
    from mitjax.pkg.rw import read_mflds
    exp, inp = "global_ocean.90x40x15", "input"
    orig = read_mflds.READ_MFLDS_SET

    def swapped(*a, **k):
        st, n, p = orig(*a, **k)
        fl = list(st.fldList)
        iu, iv = fl.index("Uvel    "), fl.index("Vvel    ")
        fl[iu], fl[iv] = fl[iv], fl[iu]
        return dataclasses.replace(st, fldList=tuple(fl)), n, p
    import mitjax.model.src.read_pickup as rp
    rp.READ_MFLDS_SET = swapped
    try:
        st, pend = ig.build_state(exp, inp)
    finally:
        rp.READ_MFLDS_SET = orig
    bad = ig.failures(ig.compare(st, exp, inp), ig.exempt_fields(pend))
    assert {"uVel", "vVel"} <= set(bad) and bad["uVel"][1] > 1000, bad


@pytest.mark.parametrize("exp,inp", gg.VARIANTS, ids=IDS)
def test_time_parameters_vs_printout(exp, inp):
    """INI_PARMS time-stepping values (ini_parms_time) equal the oracle's CONFIG_SUMMARY printout."""
    from mitjax.config.params import fortran_text
    from mitjax.io.stdout import parameter_dict, parameters, read_stdout
    _, _, rundir = gg.oracle(exp, inp)
    printed = parameter_dict(parameters(read_stdout(rundir / "output.txt")))
    tp = ig.params(exp, inp).time
    checked = 0
    for name, kind in (("nIter0", "int"), ("nTimeSteps", "int"), ("nEndIter", "int"), ("deltaTMom", "float"),
                       ("deltaTFreeSurf", "float"), ("deltaTClock", "float"), ("startTime", "float"),
                       ("endTime", "float"), ("baseTime", "float")):
        p = printed.get(name)
        if p is None:
            continue
        assert fortran_text(getattr(tp, name), kind) == p.values()[0], (name, getattr(tp, name), p.values())
        checked += 1
    p = printed.get("dTtracerLev")
    if p is not None:
        vals = p.values()
        assert [fortran_text(x, "float") for x in tp.dTtracerLev] == vals[:len(tp.dTtracerLev)]
        checked += 1
    assert checked >= 6, f"only {checked} time parameters found in the printout"


def test_state_pytree_and_fields():
    import jax

    from mitjax.model.state import State, fields_of
    exp, inp = "advect_xy", "input"
    st, _ = ig.build_state(exp, inp)
    cfg = gg.experiment(exp, inp).cfg
    assert st.names() == fields_of(cfg)
    assert isinstance(st.guNm, tuple) and len(st.guNm) == 2          # AB3 build: guNm(...,1:2)
    leaves, tree = jax.tree_util.tree_flatten(st)
    back = jax.tree_util.tree_unflatten(tree, leaves)
    assert back.names() == st.names()
    with pytest.raises(AttributeError):
        st.guNm1                                                       # not declared in an AB3 build
    with pytest.raises(KeyError):
        st.replace(notAField=1)
    with pytest.raises(AttributeError):
        st.uVel = None
    assert isinstance(State(), State)
