"""GM/Redi gates (M1 sub-lane GMREDI; the kernels of plan Task 15b, gated by replay; the in-model gates of Tasks
15b/16 come with the time loop).

1. Dump gate: GMREDI_CALC_TENSOR replayed from the oracle's own inputs at every dumped iteration of the registered
   dumps-on runs of global_ocean.90x40x15/input (gkw91, GM_BOLUS_ADVEC) and tutorial_global_oce_optim/input_ad (dm95,
   forward-only code_ad: ALLOW_AUTODIFF, GM_EXCLUDE_*), compared with P05_gmredi_tensor on every point of every tile
   (halos and land included): element equality, finite on both sides, equal bit patterns.
2. Replay gate: reference/replay_gmredi harness (synthetic stratification incl. unstable and neutral columns, slopes
   beyond GM_maxSlope and GM_slopeSqCutoff, land, overridden unit and K factors): every ported routine, every output
   point, same criteria; GMREDI_READPARMS's derived values and GMREDI_INIT_FIXED's factors against the Fortran's.
3. Negative controls, each measured to bite (counts asserted > 0).
4. Gradients: finite on every lane (land, halos, neutral and unstable lanes, cut-off lanes); FD h-sweep at smooth
   points; tangent-linear vs adjoint dot test. The taper and clipping are differentiated as written (JAX's own
   derivative of max / where; Nikolay 2026-10-01).
5. Unported options raise.
All run under the gate XLA flags (conftest.py) with the REAL parameters as traced jit arguments (inside `Gmredi`).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax.pkg.gmredi import gmredi_slope_limit as slope_mod
from mitjax.tests import gmredi_gate as G

EXPS = G.EXPERIMENTS
IDS = [e for e, _ in EXPS]


# ------------------------------------------------------------------------------------------------- 1. dump gate

@pytest.mark.parametrize("exp,inp", EXPS, ids=IDS)
def test_calc_tensor_bitwise_vs_dumps(exp, inp):
    ds, _, _ = G.grid_gate.oracle(exp, inp)
    its = ds.iterations()
    assert len(its) == 3
    for it in its:
        ours, ref = G.dump_case(exp, inp, it)
        res = G.compare_fields(ours, ref)
        assert set(res) == set(G.tensor_names(G.setup(exp, inp)[0].cfg))
        assert not G.failures(res), (it, G.failures(res))


def _bites(exp, inp, **kw):
    ds, _, _ = G.grid_gate.oracle(exp, inp)
    return sum(G.n_differing(G.compare_fields(*G.dump_case(exp, inp, it, **kw))) for it in ds.iterations())


def _ulp(x):
    return np.float64(np.nextafter(np.float64(x), np.inf))


@pytest.mark.parametrize("exp,inp", EXPS, ids=IDS)
def test_dump_negative_controls_bite(exp, inp):
    """Planted errors, each measured to change the gated output in this experiment."""
    counts = {}
    counts["GM_Small_Number x2"] = _bites(exp, inp, gm_override=lambda g: g.replace(
        GM_Small_Number=np.float64(2.0) * g.GM_Small_Number))
    counts["GM_Kmin_horiz +1ulp"] = _bites(exp, inp, gm_override=lambda g: g.replace(
        GM_Kmin_horiz=_ulp(g.GM_Kmin_horiz)))
    counts["GM_maxSlope +1ulp"] = _bites(exp, inp, gm_override=lambda g: g.replace(GM_maxSlope=_ulp(g.GM_maxSlope)))
    counts["GM_Sd +1ulp"] = _bites(exp, inp, gm_override=lambda g: g.replace(GM_Sd=_ulp(g.GM_Sd)))

    def rim(name, a):
        b = np.array(a, copy=True)
        b[:, :, :, 0] += 1.0          # i = 1-OLx: a point GMREDI_CALC_TENSOR does not write (it keeps the prior)
        return b
    counts["prior at i=1-OLx +1"] = _bites(exp, inp, prior_override=rim)
    if G.setup(exp, inp)[0].cfg.cpp.ALLOW_AUTODIFF:
        # the code_ad build zeroes every tensor point first (gmredi_calc_tensor.F:263-279), so the prior cannot
        # show; the planted error removes that zeroing
        counts["no ALLOW_AUTODIFF zeroing + prior at i=1-OLx +1"] = _bites(
            exp, inp, prior_override=rim, cfg_override=lambda c: G.CfgOverride(c, ALLOW_AUTODIFF=False))
    print(exp, "dump negative controls (differing points over the 3 dumped iterations):", counts)
    scheme = G.setup(exp, inp)[1].GM_taper_scheme.strip()
    # gkw91 (cut-off 1e48): the GM_Small_Number clamp shows in the slopes; dm95 (cut-off 1e8): every clamped lane
    # has SlopeSqr >= 1e8 whatever the clamp value (cut, taperFct = 0), so the clamp control cannot bite there and
    # the dm95 width GM_Sd is used instead
    must = (["GM_Small_Number x2", "GM_maxSlope +1ulp"] if scheme == "gkw91" else ["GM_Sd +1ulp"])
    must += ["GM_Kmin_horiz +1ulp"]
    must += (["no ALLOW_AUTODIFF zeroing + prior at i=1-OLx +1"] if G.setup(exp, inp)[0].cfg.cpp.ALLOW_AUTODIFF
             else ["prior at i=1-OLx +1"])
    for name in must:
        assert counts[name] > 0, (name, counts)


def test_dump_negative_control_xla_tanh_bites(monkeypatch):
    """dm95 with XLA's tanh instead of glibc's (tutorial_global_oce_optim): must differ from the oracle."""
    exp, inp = EXPS[1]
    monkeypatch.setattr(slope_mod, "glibc_tanh", jnp.tanh)
    n = _bites(exp, inp)
    print("xla tanh differing points", n)
    assert n > 0


# ------------------------------------------------------------------------------------------------- 2. replay gate

@pytest.mark.parametrize("exp", IDS)
def test_readparms_and_init_fixed_vs_fortran(exp):
    for name, (ours, ref) in G.fixed_checks(exp).items():
        ours, ref = np.asarray(ours, np.float64), np.asarray(ref, np.float64)
        assert ours.shape == ref.shape, name
        assert np.all(np.isfinite(ours)) and np.all(np.isfinite(ref)), name
        assert np.array_equal(ours.view(np.int64), ref.view(np.int64)), (name, ours, ref)


def _replay_outputs_used(exp, ours):
    cfg = G.replay_case(exp)[0].cfg
    if not cfg.cpp.GM_BOLUS_ADVEC:
        assert not any("GM_Psi" in n for n in ours)
    return ours


@pytest.mark.parametrize("exp", IDS)
def test_replay_bitwise(exp):
    ours, ref = G.replay_run(exp)
    _replay_outputs_used(exp, ours)
    res = G.compare_fields(ours, ref)
    print(exp, {k: v[:2] for k, v in res.items()})
    assert not G.failures(res), G.failures(res)
    # the synthetic inputs reach the branches they are meant to reach (in the Fortran's own outputs)
    e = G.replay_case(exp)[0]
    sz = e.cfg.size
    cut = G.setup(exp, dict(EXPS)[exp])[1].GM_slopeSqCutoff
    i = (slice(None), slice(None), slice(1, sz.sNy + 2*sz.OLy - 1), slice(1, sz.sNx + 2*sz.OLx - 1))
    for kp in (1, 2, 3):
        t, s = ref[f"slope{kp}_taperFct"][i], ref[f"slope{kp}_SlopeSqr"][i]
        assert np.any(t == 1.0) and np.any((t > 0) & (t < 1)) and np.any(t == 0.0), kp
        assert np.any(s == 0.0) and np.any(s == cut), kp


def _replay_bites(exp, patch):
    ours, ref = G.replay_run(exp, patch=patch)
    return G.n_differing(G.compare_fields(ours, ref))


@pytest.mark.parametrize("exp", IDS)
def test_replay_negative_controls_bite(exp):
    def unit(objs):
        p = objs["params"]
        return {**objs, "params": dataclasses.replace(p, z2rUnit=G.fa("z2rUnit", jnp.ones_like(p.z2rUnit.data), "r",
                                                                         objs["cfg"].size))}

    def swapK(objs):
        g = objs["gm"]
        return {**objs, "gm": g.replace(GM_isopycK=g.GM_background_K, GM_background_K=g.GM_isopycK)}

    def skew(objs):
        return {**objs, "gm": objs["gm"].replace(GM_skewflx=np.float64(1.0))}

    def deep(objs):
        gr = objs["grid"]
        return {**objs, "grid": gr.replace(recip_deepFacC=G.fa("recip_deepFacC", jnp.ones_like(gr.recip_deepFacC.data),
                                                               "r", objs["cfg"].size))}
    counts = {"z2rUnit = 1": _replay_bites(exp, unit), "isopycK <-> background_K": _replay_bites(exp, swapK),
              "GM_skewflx = 1": _replay_bites(exp, skew), "recip_deepFacC = 1": _replay_bites(exp, deep)}
    print(exp, counts)
    for k, v in counts.items():
        assert v > 0, (k, counts)


def test_replay_negative_control_xla_tanh_bites(monkeypatch):
    monkeypatch.setattr(slope_mod, "glibc_tanh", jnp.tanh)
    n = _replay_bites(IDS[1], None)
    print("xla tanh differing points (replay)", n)
    assert n > 0


# ------------------------------------------------------------------------------------------------- 4. gradients

def _weights(f, x0, seed):
    out = jax.eval_shape(f, x0)
    rng = np.random.default_rng(seed)
    return {n: jnp.asarray(rng.standard_normal(v.shape)) for n, v in out.items()}


@pytest.mark.parametrize("exp,inp", EXPS, ids=IDS)
@pytest.mark.parametrize("source", ["dump", "synthetic"])
def test_gradient_finite_everywhere(exp, inp, source):
    ds, _, _ = G.grid_gate.oracle(exp, inp)
    x0, gm, state = (G.inputs_dump(exp, inp, ds.iterations()[0]) if source == "dump"
                     else G.inputs_synthetic(exp, inp))
    f = G.tensor_fn(exp, inp, gm, state)
    J = G.cost_fn(f, _weights(f, x0, 1))
    g = jax.jit(jax.grad(J))(x0)
    bad = {n: int(np.count_nonzero(~np.isfinite(np.asarray(v)))) for n, v in g.items()}
    assert not any(bad.values()), bad
    assert any(np.any(np.asarray(g[n]) != 0) for n in ("sigmaX", "sigmaY", "sigmaR"))


@pytest.mark.parametrize("exp,inp", EXPS, ids=IDS)
def test_fd_and_dot_test(exp, inp):
    """FD h-sweep at smooth points, each with a local cost (random weights on the outputs within +-1 level and +-2
    points of the perturbed point, zero elsewhere: the forward-noise floor of a cost summed over every output point
    was ~1e-6; with the local cost the best of the sweep was 2.5e-8 at worst, 2026-10-01), and the tangent-linear vs
    adjoint dot test on the whole field."""
    ds, _, _ = G.grid_gate.oracle(exp, inp)
    x0, gm, state = G.inputs_dump(exp, inp, ds.iterations()[0])
    f = G.tensor_fn(exp, inp, gm, state)

    def cost(x, w):
        out = f(x)
        return sum(jnp.sum(w[n] * out[n]) for n in sorted(w))
    J = jax.jit(cost)
    dJ = jax.jit(jax.grad(cost))
    shapes = {n: v.shape for n, v in jax.eval_shape(f, x0).items()}
    e = G.setup(exp, inp)[0]
    sz = e.cfg.size
    # smooth points: stably stratified wet interior points (dSigmaDr far above GM_Small_Number)
    sR = np.asarray(x0["sigmaR"])
    mC = np.asarray(G.setup(exp, inp)[2].maskC.data)
    t, k, j, i = np.nonzero((sR < -1e-5) & (mC == 1))
    inner = ((j >= sz.OLy + 2) & (j < sz.sNy + sz.OLy - 2) & (i >= sz.OLx + 2) & (i < sz.sNx + sz.OLx - 2)
             & (k >= 2) & (k < sz.Nr - 1))
    rng = np.random.default_rng(3)
    picks = rng.choice(np.nonzero(inner)[0], size=8, replace=False)
    checked, worst = 0, 0.0
    for name in ("sigmaX", "sigmaY", "sigmaR"):
        for p in picks[:4]:
            idx = (t[p], k[p], j[p], i[p])
            w = {}
            for n, shp in shapes.items():
                a = np.zeros(shp)
                win = (idx[0], slice(idx[1] - 1, idx[1] + 2), slice(idx[2] - 2, idx[2] + 3),
                       slice(idx[3] - 2, idx[3] + 3))
                a[win] = rng.standard_normal(a[win].shape)
                w[n] = jnp.asarray(a)
            gp = float(np.asarray(dJ(x0, w)[name])[idx])
            x = float(np.asarray(x0[name])[idx])
            if gp == 0.0 or x == 0.0:
                continue
            errs = []
            for h in (abs(x) * 10.0 ** -e_ for e_ in range(2, 9)):
                xp = {**x0, name: x0[name].at[idx].add(h)}
                xm = {**x0, name: x0[name].at[idx].add(-h)}
                fd = (float(J(xp, w)) - float(J(xm, w))) / (2 * h)
                errs.append(abs(fd - gp) / abs(gp))
            print(exp, "FD", name, tuple(int(q) for q in idx), gp, min(errs))
            assert min(errs) < 1e-6, (name, idx, gp, errs)
            checked += 1
            worst = max(worst, min(errs))
    print(exp, "FD points", checked, "worst best-of-sweep rel. error", worst)
    assert checked >= 6
    # tangent linear vs adjoint, in relative directions (v = |x| * N(0,1)): an O(1) direction on a sigma of 1e-19
    # makes the gkw91 slopes of lanes just above GM_Small_Number linearize to terms of 1e20 and more that cancel
    v = jax.tree_util.tree_map(lambda a, r: jnp.abs(a) * r, x0, G.random_like(x0, 4))
    _, jv = jax.jit(lambda x, d: jax.jvp(f, (x,), (d,)))(x0, v)
    wv = G.random_like(jv, 5)
    _, vjp = jax.vjp(f, x0)
    (wt,) = jax.jit(vjp)(wv)
    lhs = sum(float(jnp.vdot(jv[n], wv[n])) for n in jv)
    rhs = sum(float(jnp.vdot(v[n], wt[n])) for n in v)
    print(exp, "dot test", lhs, rhs, abs(lhs - rhs) / max(abs(lhs), abs(rhs)))
    assert abs(lhs - rhs) <= 1e-12 * max(abs(lhs), abs(rhs))


# ------------------------------------------------------------------------------------------------- 5. unported

@pytest.mark.parametrize("exp,inp", EXPS, ids=IDS)
def test_unported_options_raise(exp, inp):
    ds, _, _ = G.grid_gate.oracle(exp, inp)
    x0, gm, state = G.inputs_dump(exp, inp, ds.iterations()[0])
    cases = [(dict(GM_taper_scheme="fm07"), (NotImplementedError, RuntimeError)),
             (dict(GM_taper_scheme="nonsense"), (RuntimeError,))]
    # 'ldd97' (the Lrho fields gmredi_calc_tensor.F:164-211, GMREDI_SLOPE_LIMIT :570-592) is ported (lane M4LAB,
    # lab_sea/input; gated bitwise by test_m4lab_ocean.py: P05_gmredi_tensor, with the 'gkw91' control): its place
    # here is taken by 'fm07', which is still not ported
    # 'linear' (GMREDI_SLOPE_LIMIT :522-537, GMREDI_SLOPE_PSI :270-288) is ported (M3 lane MLAdjust, input.QGLthGM;
    # gated bitwise by test_mladjust.py: P05_gmredi_tensor / P06_gmredi_exch): no longer an unported option
    # GM_AdvForm with 'gkw91' (GMREDI_SLOPE_PSI :290-312) is ported (GO lane, global_ocean.cs32x15; gated bitwise by
    # test_cs32_go.py::test_front_to_s04_bitwise, P05_gmredi_tensor): no longer an unported option
    for change, err in cases:
        f = G.tensor_fn(exp, inp, gm.replace(**change), state)
        with pytest.raises(err):
            jax.eval_shape(f, x0)
    # PTRACERS lane: GM_ExtraDiag without the K3D fields (Kuz, Kvz from GM_isoFac/bolFac) is ported, gated bitwise by
    # tutorial_tracer_adjsens (test_ptracers_adjsens.py: P05_gmredi_tensor, S04 Kuz Kvz): it traces
    jax.eval_shape(G.tensor_fn(exp, inp, gm.replace(GM_ExtraDiag=True), state), x0)
