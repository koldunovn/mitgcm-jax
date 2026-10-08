"""The CG2D derivative switch (mitjax/ad/modes.py, plan "Decisions 2026-10-02" item 3 revised): the derivative of CG2D
follows the run's cg2dFullAdjoint as TAF's CG2D_MAD does (pkg/autodiff/cg2d_mad.F:181-236 @63cdc0b: .FALSE. = the
operator passive, .TRUE. = the operator's adjoint added), with the exact derivative selectable explicitly
(Cg2dParams.mjx_cg2d_derivative = "exact"). A backward-only switch: the three tests of LESSONS_CARRIED L-AD-16.

* the setting per run (config only): cs32x15/input_ad passive (cg2dFullAdjoint = .FALSE., NONLIN_FRSURF), its "exact"
  option active; 90x40x15/input_ad.bottomdrag active (cg2dFullAdjoint = .TRUE.); 90x40x15/input_ad and optim passive
  (no control reaches their operator); a forward build (cs32x15/input, no pkg/autodiff) exact; a bad option raises;
* live fixture global_ocean.cs32x15/input_ad, CG2D on the oracle's C01_cg2d_inputs of the first dumped iteration
  (cg2d_b, cg2d_x, the six operator arrays; the Model's cg2dNorm, cg2dTolerance_sq), J = sum(w * EXCH_XY_RL(cg2d_x))
  as SOLVE_FOR_PRESSURE reads it (solve_for_pressure.F:315), w with weight on halo lanes and tile edges:
  - forward byte-identical: every output of cg2d_solve bit for bit and the forward jaxpr text (addresses normalised)
    equal for the run setting and "exact";
  - effect: the run setting's cotangents of aW2d, aS2d, aC2d, pW, pS, pC and cg2dNorm are 0 on every lane, the exact
    option's of aW2d, aS2d, aC2d are nonzero (the negative control: a switch without effect fails here); cg2d_b's
    cotangent bit for bit the same in both; both finite on every lane;
  - empty switch: with cg2dFullAdjoint = .TRUE. (input_ad.bottomdrag's setting) the gradient program of the run setting
    is the exact option's (sha256 of the jaxpr text, addresses normalised), so bottomdrag's gradient is unchanged;
    negative control: with cg2dFullAdjoint = .FALSE. the run setting's program differs from the exact one's;
  - the run setting's tangent and adjoint are transposes (dot test over cg2d_b and the operator).
The whole-model effect (cs32x15/input_ad admGrd: run setting vs TAF >= 10 digits, exact vs FD) is
test_cs32_ad_adjoint.py. Costs: the Model of cs32x15/input_ad (~1-2 min) and a few CG2D programs; tier 1x.
"""

import hashlib
import re

import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

EXP = ("global_ocean.cs32x15", "input_ad")
OPS = ("aW2d", "aS2d", "aC2d", "pW", "pS", "pC")
DOT_TOL = 1e-12


def _sha(jaxpr):
    """sha256 of a jaxpr's text with object addresses normalised (L-AD-16)."""
    return hashlib.sha256(re.sub(r" at 0x[0-9a-f]+", " at 0x", str(jaxpr)).encode()).hexdigest()


@pytest.mark.parametrize("exp,inp,run,exact", [
    ("global_ocean.cs32x15", "input_ad", False, True),
    ("global_ocean.90x40x15", "input_ad.bottomdrag", True, True),
    ("global_ocean.90x40x15", "input_ad", False, True),
    ("tutorial_global_oce_optim", "input_ad", False, True),
    ("global_ocean.cs32x15", "input", True, True),
])
def test_setting_per_run(exp, inp, run, exact):
    from mitjax.ad import modes
    from mitjax.model.src.cg2d_h import ini_parms_cg2d
    from mitjax.tests.grid_gate import experiment
    e = experiment(exp, inp)
    p = ini_parms_cg2d(e)
    assert p.mjx_cg2d_derivative == "run"
    assert modes.cg2d_operator_adjoint(e.cfg, p) is run
    assert modes.cg2d_operator_adjoint(e.cfg, modes.with_cg2d_derivative(p, "exact")) is exact
    with pytest.raises(ValueError):
        modes.with_cg2d_derivative(p, "taf")


@pytest.fixture(scope="module")
def fx():
    """The cs32x15/input_ad CG2D replay: J(b, ops, cg2d_params) and its inputs."""
    import jax
    import jax.numpy as jnp
    from mitjax.model.src import cg2d as cg2d_mod
    from mitjax.tests import cg2d_gate as cg
    from mitjax.tests import cube_run_gate as C
    r = C.CubeRun(*EXP)
    m = r.m
    sz = m.cfg.size
    it = r.its[0]
    b0 = jnp.asarray(cg.dump2d(r.ds, it, "C01_cg2d_inputs", "cg2d_b"))
    x0 = jnp.asarray(cg.dump2d(r.ds, it, "C01_cg2d_inputs", "cg2d_x"))
    ops0 = {n: jnp.asarray(cg.dump2d(r.ds, it, "C01_cg2d_inputs", n)) for n in OPS}
    ops0["cg2dNorm"] = jnp.asarray(m.cg2dh.cg2dNorm)
    w = np.random.default_rng(20261002).standard_normal(b0.shape)
    w[:, :sz.OLy + 1, :] *= 10.
    w[:, -sz.OLy - 1:, :] *= 10.
    w[:, :, :sz.OLx + 1] *= 10.
    w[:, :, -sz.OLx - 1:] *= 10.
    w = jnp.asarray(w)

    def solve(b, ops, p):
        c = m.cg2dh.replace(**{n: cg.xy(ops[n], n, sz) for n in OPS}, cg2dNorm=ops["cg2dNorm"])
        return cg2d_mod.cg2d_solve(cg.xy(b, "cg2d_b", sz), cg.xy(x0, "cg2d_x", sz), p.cg2dMaxIters,
                                   p.cg2dUseMinResSol - 1, cfg=m.cfg, cg2dh=c, params=p, ex=m.ex)

    def J(b, ops, p):
        x = solve(b, ops, p)[1]
        return jnp.sum(w * m.ex.EXCH_XY_RL(x.data))     # solve_for_pressure.F:315 _EXCH_XY_RL( cg2d_x )
    yield dict(m=m, b0=b0, ops0=ops0, solve=solve, J=J, jax=jax)
    jax.clear_caches()


def _params(fx, option, full=None):
    import dataclasses
    from mitjax.ad import modes
    p = modes.with_cg2d_derivative(fx["m"].cg2d_params, option)
    return p if full is None else dataclasses.replace(p, cg2dFullAdjoint=full)


def test_forward_byte_identical(fx):
    """cg2d_solve's outputs bit for bit and its forward program text equal for the run setting and "exact".
    Negative control: the same comparison sees a planted change of cg2d_b by a factor (1 + 1e-12)."""
    jax = fx["jax"]

    def same(x, y):
        la, lb = jax.tree.leaves(x), jax.tree.leaves(y)
        return len(la) == len(lb) and all(np.asarray(a).dtype == np.asarray(b).dtype
                                           and np.asarray(a).tobytes() == np.asarray(b).tobytes()
                                           for a, b in zip(la, lb))
    outs, shas = [], []
    for opt in ("run", "exact"):
        p = _params(fx, opt)
        f = lambda b, ops, p=p: fx["solve"](b, ops, p)                       # noqa: E731
        outs.append(jax.tree.map(np.asarray, jax.jit(f)(fx["b0"], fx["ops0"])))
        shas.append(_sha(jax.make_jaxpr(f)(fx["b0"], fx["ops0"])))
    assert same(outs[0], outs[1])
    assert shas[0] == shas[1]
    p = _params(fx, "run")
    out_planted = jax.tree.map(np.asarray, jax.jit(lambda b, ops: fx["solve"](b, ops, p))(fx["b0"] * (1.0 + 1e-12),
                                                                                       fx["ops0"]))
    assert not same(outs[0], out_planted)


def test_effect_on_live_fixture(fx):
    """Run setting (cg2dFullAdjoint = .FALSE.): the operator's and cg2dNorm's cotangents 0 on every lane; "exact":
    aW2d, aS2d, aC2d's nonzero (a switch without effect fails here); cg2d_b's cotangent bit for bit equal."""
    jax = fx["jax"]
    g = {}
    for opt in ("run", "exact"):
        p = _params(fx, opt)
        g[opt] = jax.tree.map(np.asarray, jax.jit(jax.grad(lambda b, ops: fx["J"](b, ops, p), argnums=(0, 1)))(
            fx["b0"], fx["ops0"]))
    for opt in g:
        assert all(np.all(np.isfinite(x)) for x in jax.tree.leaves(g[opt])), opt
    gb_run, gops_run = g["run"]
    gb_ex, gops_ex = g["exact"]
    nz_run = {n: int(np.count_nonzero(gops_run[n])) for n in gops_run}
    nz_ex = {n: int(np.count_nonzero(gops_ex[n])) for n in gops_ex}
    print("nonzero operator cotangents: run", nz_run, "exact", nz_ex)
    assert all(v == 0 for v in nz_run.values()), nz_run
    assert all(nz_ex[n] > 0 for n in ("aW2d", "aS2d", "aC2d")), nz_ex
    assert np.array_equal(gb_run.view(np.uint64), gb_ex.view(np.uint64))
    assert np.count_nonzero(gb_run) > 0


def test_empty_switch_with_full_adjoint(fx):
    """cg2dFullAdjoint = .TRUE. (input_ad.bottomdrag's setting): the run setting's gradient program is the exact
    option's (jaxpr sha256); negative control: with .FALSE. it is not."""
    jax = fx["jax"]

    def sha(p):
        return _sha(jax.make_jaxpr(jax.grad(lambda b, ops: fx["J"](b, ops, p), argnums=(0, 1)))(fx["b0"],
                                                                                                fx["ops0"]))
    exact = sha(_params(fx, "exact"))
    assert sha(_params(fx, "run", full=True)) == exact
    assert sha(_params(fx, "run", full=False)) != exact


def test_dot_test_run_setting(fx):
    """The run setting's tangent and adjoint are transposes: <w, TL(db, dops)> == <AD(w), (db, dops)> to DOT_TOL
    relative, the direction on cg2d_b and every operator array (the operator part has zero effect both ways)."""
    jax = fx["jax"]
    p = _params(fx, "run")
    f = lambda b, ops: fx["J"](b, ops, p)                                    # noqa: E731
    rng = np.random.default_rng(7)
    db = jax.numpy.asarray(rng.standard_normal(fx["b0"].shape))
    dops = {n: jax.numpy.asarray(rng.standard_normal(np.shape(v)) * (np.asarray(v) != 0))
            for n, v in fx["ops0"].items()}
    _, tl = jax.jit(lambda b, o, db, do: jax.jvp(f, (b, o), (db, do)))(fx["b0"], fx["ops0"], db, dops)
    gb, gops = jax.jit(jax.grad(f, argnums=(0, 1)))(fx["b0"], fx["ops0"])
    ad = float(np.sum(np.asarray(gb) * np.asarray(db))) + sum(float(np.sum(np.asarray(gops[n]) * np.asarray(dops[n])))
                                                              for n in dops)
    tl = float(tl)
    print("dot test run setting: tl", tl, "ad", ad, "rel", abs(tl - ad) / abs(ad))
    assert abs(tl - ad) <= DOT_TOL * abs(ad), (tl, ad)
