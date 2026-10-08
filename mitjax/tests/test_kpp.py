"""KPP gates (M3 sub-lane KPP, plan Task 29; pkg/kpp as vermix/input and vermix/input.dd run it).

1. Fixed values: KPP_READPARMS's parameters, KPP_INIT_FIXED's Vtc, cg, deltaz, deltau, zgrid, hwide and the wmt/wst
   tables, KPP_INIT_VARIA's nzmax and KPP.h fields, against the replay harness's kpp_fixed.bin (bitwise).
2. Replay gate (reference/replay_kpp, 12 cases x 25 columns of 9 regimes incl. land and shallow columns, halos):
   KPP_FORCING_SURF, KPPMIX (RI_IWMIX, BLDEPTH, WSCALE, BLMIX, ENHANCE), KPP_DOUBLEDIFF on synthetic intermediates;
   KPP_DO_EXCH, KPP_CALC_DIFF_T/S, KPP_CALC_VISC, KPP_TRANSPORT_T/S; STATEKPP and the whole KPP_CALC (input.dd:
   LINEAR EOS; input: MDJWF, FIND_ALPHA/FIND_BETA of lane EOSAB since the vermix session 3 merge).
   Every point, element equality, finite on both sides, equal bit patterns.
3. Dump gate: KPP_CALC at the dumped iterations 0, 1, 2 of the registered dumps-on runs (teacher-forced inputs) vs
   P07_kpp, and KPP_DO_EXCH of P07_kpp vs P10_kpp_exch (input.dd and input).
4. Negative controls, each measured to bite (counts asserted > 0).
5. Gradients of KPP_CALC (input.dd, a replay case with every regime): finite on every lane (land, halos, shallow
   columns), tangent vs adjoint dot test, FD h-sweep at a smooth point. Switches (kbl, IF, MAX/MIN) differentiated
   as written (JAX's own derivative).
All under the gate XLA flags (conftest.py), REAL parameters as traced jit arguments.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax.tests import kpp_gate as G

VARIANTS = G.VARIANTS


def _assert_bitwise(res, what):
    bad = G.failures(res)
    assert not bad, f"{what}: {bad}"


# ------------------------------------------------------------------------------------------------- 1. fixed values

@pytest.mark.parametrize("inp", VARIANTS)
def test_fixed_values_bitwise(inp):
    s = G.setup(inp)
    fx = G.RIO.read_fixed(G.replay_rundir(inp), G._size_dict(s.sz))
    for k, v in fx["scalars"].items():
        ours = getattr(s.kpp, k)
        ours = float(ours) if isinstance(ours, (bool, int)) else ours
        assert np.float64(ours).view(np.int64) == np.float64(v).view(np.int64), (k, ours, v)
    for n in ("zgrid", "hwide", "wmt", "wst"):
        a, b = np.asarray(getattr(s.kpp, n).data), fx[n]
        assert a.shape == b.shape and np.array_equal(a.view(np.int64), b.view(np.int64)), n
    kpp, f = G.init_varia_fields(s)
    assert np.array_equal(np.asarray(kpp.nzmax.data, np.float64), fx["nzmax"])
    for n in ("KPPviscAz", "KPPdiffKzS", "KPPdiffKzT", "KPPghat", "KPPhbl"):
        _assert_bitwise(G.compare({n: f[n].data}, {n: fx[n]}), n)


# ------------------------------------------------------------------------------------------------- 2. replay gate

@pytest.mark.parametrize("inp", VARIANTS)
def test_replay_units_and_after_bitwise(inp):
    res = G.replay(inp, which=("units", "after"))
    assert len(res) == 12
    for c, r in res.items():
        assert len(r) == 18
        _assert_bitwise(r, f"{inp} case {c}")


@pytest.mark.parametrize("inp", ["input", "input.dd"])
def test_replay_statekpp_and_kpp_calc_bitwise(inp):
    res = G.replay(inp, which=("calc",))
    for c, r in res.items():
        assert len(r) == 11
        _assert_bitwise(r, f"{inp} case {c}")


# ------------------------------------------------------------------------------------------------- 3. dump gate

@pytest.mark.parametrize("inp", ["input", "input.dd"])
def test_kpp_calc_and_exch_bitwise_vs_dumps(inp):
    ds, _, _ = G.gg.oracle(G.EXP, inp)
    its = ds.iterations()
    assert its == [0, 1, 2]
    for it in its:
        o, r = G.dump_case(inp, it)
        res = G.compare(o, r)
        assert len(res) == 12
        _assert_bitwise(res, f"{inp} it {it}")


# ------------------------------------------------------------------------------------------------- 4. negative controls

def test_negative_control_units_vtc():
    """Vtc larger by a relative 1e-12 (BLDEPTH's unresolved shear; one ulp was measured NOT to bite in case 1: the
    change is absorbed by the roundings of Rib and the hbl clamps): KPPMIX outputs must differ."""
    s = G.setup("input.dd")
    k2 = s.kpp.replace(Vtc=s.kpp.Vtc * (1.0 + 1e-12))
    fu = G.units_fn(s)
    rd = G.replay_rundir("input.dd")
    d = G.RIO.read_inputs(rd, G._size_dict(s.sz))[0]
    o = G.RIO.read_outputs(rd, G._size_dict(s.sz))[0]
    ours = jax.device_get(fu(s.params, s.grid, k2, s.fp, {k: jnp.asarray(v) for k, v in d.items()}))
    res = G.compare(ours, {k: o[k] for k in ours})
    n = sum(res[k][1] for k in ("mx_vddiff", "mx_ghat", "mx_hbl"))
    assert n > 0, res


def test_negative_control_calc_ricr_one_ulp():
    """Ricr one ulp larger: KPP_CALC's boundary-layer depth and coefficients must differ (input.dd, case 1)."""
    s = G.setup("input.dd")
    k2 = s.kpp.replace(Ricr=np.nextafter(s.kpp.Ricr, np.inf))
    fc = G.calc_fn(s)
    rd = G.replay_rundir("input.dd")
    d = G.RIO.read_inputs(rd, G._size_dict(s.sz))[0]
    o = G.RIO.read_outputs(rd, G._size_dict(s.sz))[0]
    ours = jax.device_get(fc(s.params, s.grid, k2, s.fp, s.eos, {k: jnp.asarray(v) for k, v in d.items()}))
    res = G.compare(ours, {k: o[k] for k in ours})
    assert res["KPPhbl"][1] > 0 and G.n_differing(res) > 0, res


def test_negative_control_after_no_exchange():
    """KPP_CALC_VISC fed the un-exchanged KPPviscAz: the viscosities must differ at the halo-reading points."""
    s = G.setup("input.dd")
    rd = G.replay_rundir("input.dd")
    d = G.RIO.read_inputs(rd, G._size_dict(s.sz))[0]
    o = G.RIO.read_outputs(rd, G._size_dict(s.sz))[0]
    kppo = {k: jnp.asarray(o[k]) for k in ("KPPviscAz", "KPPdiffKzS", "KPPdiffKzT", "KPPghat", "KPPhbl", "KPPfrac")}
    kppo["ex_viscAz"] = kppo["KPPviscAz"]                                      # planted: exchange skipped
    fa = G.after_fn(s)
    ours = jax.device_get(fa(s.params, s.grid, s.kpp, s.fp, {k: jnp.asarray(v) for k, v in d.items()}, kppo))
    res = G.compare(ours, {k: o[k] for k in ours})
    assert res["cv_RU"][1] + res["cv_RV"][1] > 0, res


def test_negative_control_dump_prior():
    """The KPP.h prior of the points KPP_CALC does not write, changed by +1 (step 1): P07_kpp must differ there."""
    def plant(prior):
        return dict(prior, KPPviscAz=prior["KPPviscAz"] + 1.0, KPPghat=prior["KPPghat"] + 1.0)
    o, r = G.dump_case("input.dd", 1, prior_override=plant)
    res = G.compare(o, r)
    assert res["KPPviscAz"][1] > 0 and res["KPPghat"][1] > 0, res


# ------------------------------------------------------------------------------------------------- 5. gradients

GRAD_IN = ("theta", "salt", "uVel", "vVel", "sfU", "sfV", "sfT", "sfS")
GRAD_OUT = ("KPPviscAz", "KPPdiffKzS", "KPPdiffKzT", "KPPghat", "KPPhbl")


def _grad_problem(case=1):
    s = G.setup("input.dd")
    rd = G.replay_rundir("input.dd")
    d = {k: jnp.asarray(v) for k, v in G.RIO.read_inputs(rd, G._size_dict(s.sz))[case - 1].items()}
    # columns that sit exactly on a switch: neutral (sfT = 0: bo = 0, the sign of a perturbed bo flips `stable`;
    # uniform T, S: dbloc = 0) and calm (sfU = sfV = 0: ustarX = 0 flips the zRef < drF(1) arm's SQRT switch)
    # and columns with equal theta at two consecutive levels (alphaDT = 0: KPP_DOUBLEDIFF's diffusive-convection arm
    # switches on with nuddt = numol*0.909*exp(4.6*exp(-5.4*(1/Rrho-1))) -> numol*0.909 /= 0 as Rrho -> 0+, a jump
    # of the Fortran's own formula; the synthetic columns clamp deep theta at -1.9)
    th = np.asarray(d["theta"])
    smooth = ((np.asarray(d["sfT"]) != 0) & ((np.asarray(d["sfU"]) != 0) | (np.asarray(d["sfV"]) != 0))
              & ~np.any(th[:, 1:] == th[:, :-1], axis=1))
    fc = G.calc_fn(s)
    rng = np.random.default_rng(7)
    out0 = jax.device_get(fc(s.params, s.grid, s.kpp, s.fp, s.eos, d))
    w = {n: jnp.asarray(rng.normal(size=np.shape(out0[n])) / (np.abs(out0[n]).max() + 1e-30)) for n in GRAD_OUT}

    def J(x):
        dd = dict(d, **x)
        out = fc(s.params, s.grid, s.kpp, s.fp, s.eos, dd)
        return sum(jnp.sum(w[n] * out[n]) for n in GRAD_OUT)
    x0 = {n: d[n] for n in GRAD_IN}
    return J, x0, rng, smooth


def test_gradient_finite_every_lane():
    J, x0, _, _ = _grad_problem()
    g = jax.jit(jax.grad(J))(x0)
    for n in GRAD_IN:
        a = np.asarray(g[n])
        assert np.all(np.isfinite(a)), (n, int(np.count_nonzero(~np.isfinite(a))))
    assert any(np.any(np.asarray(g[n]) != 0) for n in GRAD_IN)


def test_gradient_dot_test_and_fd():
    """Dot test on a direction over every lane; FD h-sweep on a direction supported on the columns that are not on a
    switch (a perturbation of an exactly neutral or calm column, or of a column with alphaDT = 0, crosses a jump at any
    h: measured, job 27840961 and a dev run, FD errors growing as 1/h with those columns in the direction)."""
    J, x0, rng, smooth = _grad_problem()
    scale = {n: float(np.abs(np.asarray(x0[n])).max()) or 1.0 for n in GRAD_IN}
    v = {n: jnp.asarray(rng.normal(size=x0[n].shape) * 1e-3 * scale[n]) for n in GRAD_IN}
    _, tl = jax.jit(lambda x, t: jax.jvp(J, (x,), (t,)))(x0, v)
    g = jax.jit(jax.grad(J))(x0)
    ad = sum(float(jnp.sum(g[n] * v[n])) for n in GRAD_IN)
    tl = float(tl)
    assert abs(tl - ad) <= 1e-12 * max(abs(tl), abs(ad)), (tl, ad)
    m = jnp.asarray(smooth)
    v = {n: v[n] * (m[:, None] if v[n].ndim == 4 else m) for n in GRAD_IN}
    _, tl = jax.jit(lambda x, t: jax.jvp(J, (x,), (t,)))(x0, v)
    tl = float(tl)
    Jj = jax.jit(J)
    errs = []
    for h in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        xp = {n: x0[n] + h * v[n] for n in GRAD_IN}
        xm = {n: x0[n] - h * v[n] for n in GRAD_IN}
        fd = (float(Jj(xp)) - float(Jj(xm))) / (2 * h)
        errs.append(abs(fd - tl) / max(abs(tl), 1e-300))
    assert tl != 0.0 and min(errs) < 1e-6, (tl, errs)


# ------------------------------------------------------------------------------------------------- 6. unported options

def test_unported_kpp_freq_raises():
    s = G.setup("input.dd")
    k2 = s.kpp.replace(kpp_freq_eq_deltaTClock=False)
    fc = G.calc_fn(s)
    rd = G.replay_rundir("input.dd")
    d = {k: jnp.asarray(v) for k, v in G.RIO.read_inputs(rd, G._size_dict(s.sz))[0].items()}
    with pytest.raises(NotImplementedError):
        fc(s.params, s.grid, k2, s.fp, s.eos, d)
