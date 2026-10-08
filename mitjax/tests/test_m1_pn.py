"""M1 acceptance (plan Task 18), GO lane: P=N == P=1 for the whole runs whose equality docs/M1_ACCEPTANCE.md had
from a job only, and every M1 whole-run carry finite except the named NaN storage.

* P=N == P=1: the run driver's Model and step (python -m mitjax run's) for the variant's whole run at P = 1 (jit)
  and inside jit(shard_map(check_vma=True)) on N fake CPU devices (go_gate.whole_run_pn / run_steps_p): every carry
  leaf and every per-step output bit for bit; advect_xy/input.ab3_c4 and tutorial_global_oce_optim/input_ad forward
  here, the three advect_xz variants in test_m1_pn_advect_xz.py (the other M1 variants: test_r2_baroclinic_gyre,
  test_r3_advection, test_r4c_global_ocean_run; tutorial_barotropic_gyre has one tile).
* finite: the non-finite values of the final carry are exactly go_gate.NAN_STORAGE's (field and count), storage the
  Fortran neither initialises nor reads during the run (reasons there, with file:line); the barotropic gyre,
  baroclinic gyre and advect_xy/input have none (P = 1 runs here); global_ocean in test_r4c.
* no backward pass reads that storage (the AD rule "masked / halo / padding lanes compute finite values"): the VJP of
  MOM_CALC_RTRANS, its only reader, is finite on every lane with the NaN storage in its input; control: a NaN in a
  coefficient it multiplies by bites.
Costs: about 6 min on a CPU compute node.
"""

import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from mitjax.tests import go_gate as G  # noqa: E402

PN = [("advect_xy", "input.ab3_c4", 2), ("tutorial_global_oce_optim", "input_ad", 4)]
FINITE_P1 = [("tutorial_barotropic_gyre", "input"), ("tutorial_baroclinic_gyre", "input"), ("advect_xy", "input")]


@pytest.mark.parametrize("exp,inp,nproc", PN, ids=[f"{e}-{i}" for e, i, _ in PN])
def test_pn_equals_p1_whole_run_and_nan_storage(exp, inp, nproc):
    m, c1, o1, cN, oN, ndiff, same = G.whole_run_pn(exp, inp, nproc)
    assert len(o1) == m.prm.time.nTimeSteps
    assert ndiff == 0 and same, (ndiff, same)
    assert G.nonfinite_leaves(c1) == G.expected_nonfinite(exp, m.cfg.size)


@pytest.mark.parametrize("exp,inp", FINITE_P1, ids=[f"{e}-{i}" for e, i in FINITE_P1])
def test_whole_run_carry_finite(exp, inp):
    m, c1, *_ = G.whole_run_pn(exp, inp, None, tag="finite")
    assert G.expected_nonfinite(exp, m.cfg.size) == {}
    assert G.nonfinite_leaves(c1) == {}


def test_backward_does_not_read_nan_storage():
    """The only reader of the NaN storage during the run is MOM_CALC_RTRANS (dWtransU/V, global_ocean; the ptracers
    fields have no reader with usePTRACERS off, advect_xz never calls it). Its VJP over the k = 1 .. Nr+1 chain of
    MOM_FLUXFORM (mom_fluxform.F:405, :422-426) from the initial State (dWtransC/U/V all NaN) w.r.t. every field it
    reads (wVel, rStarDhC/W/SDt, dWtransC/U/V) is finite on every lane for a random cotangent on every output
    (rTransU/V of each k, the final dWtrans), and the outputs keep exactly the rim NaN. Control: one NaN in a
    coefficient the routine multiplies by (h0FacW, interior point) makes the gradient non-finite. The full-step VJP
    (every floating carry leaf) of advect_xz/input and global_ocean.90x40x15/input: docs/M1_ACCEPTANCE.md section 5
    (jobs; compile 6 min and more)."""
    from mitjax.farray import FArray
    from mitjax.pkg.mom_fluxform.mom_calc_rtrans import mom_calc_rtrans
    m = G.driver_model("rtrans-vjp")
    sz, state0 = m.cfg.size, m.initial_carry()[0]
    names = ("wVel", "rStarDhCDt", "rStarDhWDt", "rStarDhSDt", "dWtransC", "dWtransU", "dWtransV")
    x0 = {n: getattr(state0, n) for n in names}
    assert all(not np.all(np.isfinite(np.asarray(x0[n].data))) for n in ("dWtransC", "dWtransU", "dWtransV"))

    def chain(grid):
        def f(x):
            st = state0.replace(**x)
            rU = grid.rA.local("rTransU", fill=0.)
            rV = grid.rA.local("rTransV", fill=0.)
            out = []
            for k in range(1, sz.Nr + 2):
                rU, rV, *st_ = mom_calc_rtrans(k, rU, rV, 0., 0, cfg=m.cfg, grid=grid, params=m.params, state=st)
                st = st_[0] if st_ else st
                out += [rU.data, rV.data]
            return out + [st.dWtransC.data, st.dWtransU.data, st.dWtransV.data]
        return f

    def vjp(grid):
        f = chain(grid)
        y, back = jax.vjp(f, x0)
        rng = np.random.default_rng(20261002)
        ct = [jnp.asarray(rng.standard_normal(np.shape(v))) for v in y]
        return y, jax.tree.leaves(back(ct)[0])

    y, gr = vjp(m.grid)
    assert all(np.all(np.isfinite(np.asarray(g))) for g in gr)
    assert [int(np.count_nonzero(~np.isfinite(np.asarray(v)))) for v in y[-3:]] == [0, G._rim(sz), G._rim(sz)]
    h = m.grid.h0FacW
    bad = m.grid.replace(h0FacW=FArray(h.data.at[0, 0, sz.OLy + 2, sz.OLx + 2].set(jnp.nan), h.name, tiled=h.tiled,
                                       _dims=h.dims))
    _, gr = vjp(bad)
    assert not all(np.all(np.isfinite(np.asarray(g))) for g in gr)


def test_negative_controls_bite(monkeypatch):
    """The comparisons above bite: the P=N carry with theta moved by 1 ulp gives one differing leaf, a per-step
    output with an extra entry unequal outputs; one NaN planted in a P=1 carry field changes nonfinite_leaves."""
    run = G.run_steps_p

    def planted(m, n, nproc=None):
        c, o = run(m, n, nproc)
        if nproc:
            st = c[0]
            th = st.theta
            c = (st.replace(theta=jax.tree.map(lambda a: np.nextafter(a, np.inf), th)),) + tuple(c[1:])
            o = o[:-1] + [{"planted": np.zeros(1)}]         # advect_xy's step outputs are None (no leaves)
        return c, o
    monkeypatch.setattr(G, "run_steps_p", planted)
    exp, inp = "advect_xy", "input.ab3_c4"
    m, c1, o1, cN, oN, ndiff, same = G.whole_run_pn(exp, inp, 2, tag="control")
    assert ndiff == 1 and not same, (ndiff, same)
    st = c1[0]
    bad = (st.replace(theta=jax.tree.map(lambda a: np.where(np.arange(a.size).reshape(a.shape) == 7, np.nan, a),
                                          st.theta)),) + tuple(c1[1:])
    assert G.nonfinite_leaves(bad) == {"theta": 1} != G.expected_nonfinite(exp, m.cfg.size)
