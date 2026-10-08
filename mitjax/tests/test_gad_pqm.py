"""Gates of lane GAD-C: the PPM/PQM advection kernels of pkg/generic_advdiff (mitjax/pkg/generic_advdiff/gad_ppm_*,
gad_pqm_*, gad_osc_*, gad_plm_fun) against the gfortran replay harness reference/replay_gad_c (advect_xz/code build,
real maskC, synthetic inputs with monotone stretches, extrema, plateaus, zero velocities and zero-transport rows):

  * bitwise: GAD_PPM_ADV_X/Y and GAD_PQM_ADV_X/Y (every limiter, calc_CFL T and F, every level) and
    GAD_PPM_ADV_R/GAD_PQM_ADV_R (every limiter), all tiles, ALL points incl. halos (every point of flux is written),
    element equality with a finite oracle, under the gate XLA flags (conftest.py);
  * negative controls: planted errors, each measured to bite;
  * the limiter branches the inputs exercise (mono = 0, 1, 2 lanes of the MONO/WENO profiles);
  * gradients: finite on all lanes (incl. halos, land, ties and zero velocities), tangent vs adjoint, FD at smooth
    points (inputs without plateaus or zero velocities, so no lane sits on a limiter switch).
The replay output must exist: these tests fail, not skip, without it.
"""

import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax.tests import gad_c_replay as gr

CASES = gr.replay_io.CASES
OUT = dict(zip(CASES, gr.replay_io.OUT3))


@pytest.fixture(scope="module")
def R():
    return gr.current_replay()


def run(R, case, levels=None, drivers=None):
    f = jax.jit(gr.case_fn(R.cfg, case, levels, drivers))
    return np.asarray(f(*gr.jax_inputs(R)))


def mismatches(R, case, levels=None, drivers=None):
    """Points (of the compared levels) whose float64 bits differ from gfortran's; the oracle must be finite."""
    ref = R.out[OUT[case]]
    got = run(R, case, levels, drivers)
    assert np.isfinite(ref).all(), OUT[case]
    if levels is not None:
        sel = [k - 1 for k in levels]
        ref, got = ref[:, sel], got[:, sel]
    assert np.isfinite(got).all(), OUT[case]
    return gr.bit_diff(got, ref)


def test_replay_setup(R):
    """The replay grid is advect_xz's (land in maskC), the harness overrides reached the routines, the inputs are
    intact and exercise the branches they are meant to."""
    s = R.size
    assert (s["sNx"], s["sNy"], s["OLx"], s["OLy"], s["nSx"], s["Nr"]) == (10, 1, 4, 4, 2, 20)   # advect_xz SIZE.h
    m = R.grid["maskC"]
    assert (m == 0).sum() > 100 and (m == 1).sum() > 1000
    assert np.all(R.grid["recip_deepFacC"] != 1.0) and np.all(R.grid["drF"] != 100.0)
    assert np.array_equal(R.grid["recip_dxF"], R.g2["recip_dxF"]) and np.array_equal(R.grid["drF"], R.g1["drF"])
    meta = json.loads((R.rundir / "replay_in.json").read_text())
    for name, sha in meta["sha256"].items():
        assert hashlib.sha256((R.rundir / name).read_bytes()).hexdigest() == sha
    f = R.fields
    assert (np.diff(f["tracer"], axis=3) == 0).sum() > 100 and (np.diff(f["tracer"], axis=1) == 0).sum() > 100
    assert (f["uVel"] == 0).sum() > 100 and (np.abs(f["uTrans"]).sum(axis=3) == 0).sum() > 10
    assert (np.abs(f["wVel"][:, 1:]).sum(axis=1) == 0).sum() > 10
    for name in R.out:
        assert np.isfinite(R.out[name]).all(), name


def test_namelists_select_the_ported_schemes():
    """advect_xz's variants run the schemes ported here (from the experiment's data files through mitjax/config)."""
    from mitjax.config import params
    want = {"input": (42, 81), "input.pqm": (51, 52)}
    for inp, (t, s) in want.items():
        cfg = params.load("advect_xz", inp).cfg
        st = dict(cfg.static)
        assert st[("data", "parm01", "tempadvscheme")] == t and st[("data", "parm01", "saltadvscheme")] == s
        assert (cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy, cfg.size.Nr) == (10, 1, 4, 4, 20)


def test_replay_bitwise_all(R):
    """Every case of the harness (30 = 4 X/Y drivers x 3 limiters x calc_CFL T/F + 2 R drivers x 3 limiters),
    every level, all points."""
    bad = {}
    for case in CASES:
        n = mismatches(R, case)
        if n:
            bad[OUT[case]] = n
    total = sum(R.out[OUT[c]].size for c in CASES)
    print(f"bitwise: {len(CASES)} cases, {total} points compared")
    assert bad == {}, bad


# planted errors: (module, old text, new text, cases[, "all" levels]), each a bug class the gate must see. Measured
# equivalent mutants in this experiment (no output can change, so they are not used): the vsum sum range of
# GAD_PPM_ADV_R (2..Nr vs 1..Nr: with velR = 0 on 2..Nr the zero-velocity IF of GAD_PPM_FLX_R gives the same 0); the
# mask factor of GAD_OSC_LOC_X's end point (mask(ix+2) = 0 also zeroes mval in every GAD_OSC_MUL_X stencil that reads
# ohat(:,1-OLx)); the WENO threshold 1.d-6 vs 1.d-5 (no lane of these inputs has 1e-6 < fdel/fmag <= 1e-5).
PLANTS = {
    "re-associated edge value (P3E_X)": (
        "gad_ppm_p3e_x", "+ (7. / 12.)*(floc[-1]+floc[+0]))", "+ ((7. / 12.)*floc[-1]+(7. / 12.)*floc[+0]))",
        [("gad_ppm_adv_x", 42, True)]),
    "flatten test .le. -> .lt. (ties, PPM_FUN_MONO)": (
        "gad_ppm_fun", "(ff00 - ffll) <= 0.)", "(ff00 - ffll) < 0.)",
        [("gad_ppm_adv_x", 41, True), ("gad_ppm_adv_r", 42, None)]),
    "bind sign swapped (PQM_FUN_MONO)": (
        "gad_pqm_fun", "bind = jnp.where(turn, jnp.where(jnp.abs(dell)                      # :248-257\n"
        "                                     < jnp.abs(derr), -1, +1), bind)",
        "bind = jnp.where(turn, jnp.where(jnp.abs(dell)\n                                     < jnp.abs(derr), +1, -1), bind)",
        [("gad_pqm_adv_x", 51, True), ("gad_pqm_adv_r", 51, None)]),
    "upwind cell shifted (PQM_FLX_X)": (
        "gad_pqm_flx_x", "intF_p = (ivec[1] * fhat[1][ix-1, iy]", "intF_p = (ivec[1] * fhat[1][ix, iy]",
        [("gad_pqm_adv_x", 52, True)]),
    "ghost-cell zeroing dropped (PPM_ADV_Y)": (
        "gad_ppm_adv_y", "    flux = flux.at[ix, +1-OLy+1].set(0.)\n", "", [("gad_ppm_adv_y", 42, True)]),
    "vsum test .gt. -> .ge. (PQM_ADV_X; signed zeros of zero-transport rows)": (
        "gad_pqm_adv_x", "jnp.where(vsum > 0., flux_flx[ix, iy], 0.)", "jnp.where(vsum >= 0., flux_flx[ix, iy], 0.)",
        [("gad_pqm_adv_x", 52, True)]),
    "deepFac factor dropped (PPM_FLX_Y)": (
        "gad_ppm_flx_y", "* recip_dyF[ix, iy-1]\n                  * recip_deepFacC[kk])",
        "* recip_dyF[ix, iy-1])", [("gad_ppm_adv_y", 42, True)]),
    "WENO blend without the mono > 0 test (PQM_HAT_R)": (
        "gad_pqm_hat_r", "weno = weno & (fdel > 1.e-6 * fmag)", "weno = (fdel > 1.e-6 * fmag)",
        [("gad_pqm_adv_r", 52, None)]),
    "interior mask dropped (OSC_LOC_X)": (
        "gad_osc_hat_x", "# :34-37\n                    mask[ix-1, iy]*(fbar[ix-1, iy]-floc[+0]))",
        "# :34-37\n                    (fbar[ix-1, iy]-floc[+0]))", [("gad_ppm_adv_x", 42, True)], "all"),
    "zero test on wvel instead of wfac (PQM_FLX_R)": (
        "gad_pqm_flx_r", "jnp.where(wfac[ix, iy, ir] == 0.,", "jnp.where(wvel[ix, iy, ir] == 0.,",
        [("gad_pqm_adv_r", 51, None)]),
    "quadroot root sign (QUADROOT)": (
        "gad_pqm_fun", "x1 = - bb + sq ", "x1 = - bb - sq ", [("gad_pqm_adv_x", 51, True)]),
}


def test_negative_controls(R):
    bitten = {}
    for label, (module, old, new, cases, *lev) in PLANTS.items():
        levels = None if lev == ["all"] else (1, 9, R.cfg.Nr)       # the land of the slope is below level 11
        with gr.planted(module, old, new) as drivers:
            bitten[label] = sum(mismatches(R, c, None if c[2] is None else levels, drivers) for c in cases)
    print("negative controls, points differing from gfortran:", bitten)
    assert all(n > 0 for n in bitten.values()), bitten


def _mono_histogram(R, case, fun_module, fun_name):
    """Lanes per value of `mono` returned by the MONO profile routine inside the case (recorded at trace time)."""
    import importlib
    hat = importlib.import_module(f"{gr.PKG}.{case[0].replace('adv', 'hat')}")
    orig = getattr(hat, fun_name)
    rec = []

    def wrapped(*a):
        r = orig(*a)
        rec.append(r[-1])
        return r
    setattr(hat, fun_name, wrapped)
    try:
        base = gr.case_fn(R.cfg, case, None if case[2] is None else (1, 9, R.cfg.Nr))

        def f(*a):
            rec.clear()
            out = base(*a)
            return out, list(rec)
        _, monos = jax.jit(f)(*gr.jax_inputs(R))
    finally:
        setattr(hat, fun_name, orig)
    m = np.concatenate([np.asarray(x).ravel() for x in monos])
    return {int(v): int((m == v).sum()) for v in (0, 1, 2)}


def test_limiter_branches_exercised(R):
    """The synthetic inputs reach every outcome of the MONO profiles (mono = 0: monotone, 1: extremum flattened or
    edge/slope limited, 2: turning point / inflexion pushed to an edge) for PPM and PQM, in X and R."""
    hist = {}
    for case, fun in ((("gad_ppm_adv_x", 41, True), "gad_ppm_fun_mono"), (("gad_ppm_adv_r", 42, None), "gad_ppm_fun_mono"),
                      (("gad_pqm_adv_x", 51, True), "gad_pqm_fun_mono"), (("gad_pqm_adv_r", 52, None), "gad_pqm_fun_mono")):
        hist[OUT[case]] = _mono_histogram(R, case, None, fun)
    print("mono histogram (lanes):", hist)
    assert all(h[v] > 0 for h in hist.values() for v in (0, 1, 2)), hist


def _loss(R, case, levels, weights_seed=3):
    f = gr.case_fn(R.cfg, case, levels)
    ref = R.out[OUT[case]]
    rng = np.random.default_rng(weights_seed)
    w = jnp.asarray(rng.standard_normal(ref.shape) / np.abs(ref).max())     # many outputs are exactly 0

    def loss(fields, grid, deltaT, dtR):
        return jnp.sum(w * f(fields, grid, deltaT, dtR))
    return loss


GRAD_CASES = (("gad_ppm_adv_x", 42, True), ("gad_ppm_adv_y", 41, False), ("gad_ppm_adv_r", 42, None),
              ("gad_pqm_adv_x", 51, True), ("gad_pqm_adv_y", 52, True), ("gad_pqm_adv_r", 52, None),
              ("gad_pqm_adv_r", 50, None))


def test_gradients_finite(R):
    """d(weighted sum of all outputs)/d(every input field, grid field and time step) is finite on every lane:
    halos, land, plateaus (limiter ties), zero velocities and zero-transport rows included."""
    bad = {}
    for case in GRAD_CASES:
        loss = _loss(R, case, None if case[2] is None else (1, 9))
        g = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3)))(*gr.jax_inputs(R))
        for path, leaf in jax.tree_util.tree_leaves_with_path(g):
            n = int((~np.isfinite(np.asarray(leaf))).sum())
            if n:
                bad[f"{OUT[case]} {jax.tree_util.keystr(path)}"] = n
    assert bad == {}, bad


def smooth_inputs(R):
    """Inputs with no lane on a limiter switch: tracer without plateaus (smooth field + noise everywhere),
    no zero velocities or transports."""
    fields, grid, deltaT, dtR = gr.jax_inputs(R)
    rng = np.random.default_rng(7)
    shp = R.fields["tracer"].shape
    k, j, i = np.meshgrid(np.arange(shp[1]), np.arange(shp[2]), np.arange(shp[3]), indexing="ij")
    fields = dict(fields)
    fields["tracer"] = jnp.asarray(10.0 + 2.0 * np.sin(0.55 * i + 0.9 * j + 0.45 * k)[None]
                                   + 0.3 * rng.standard_normal(shp))
    for n, lo, hi in (("uVel", 0.5, 6.0), ("vVel", 0.5, 6.0), ("uCFL", 0.1, 0.9), ("vCFL", 0.1, 0.9),
                      ("wVel", 0.005, 0.07)):
        fields[n] = jnp.asarray(rng.uniform(lo, hi, shp) * np.where(rng.uniform(size=shp) < 0.5, -1.0, 1.0))
    for n, v in (("uTrans", "uVel"), ("vTrans", "vVel"), ("wTrans", "wVel")):
        fields[n] = fields[v] * (1.0e6 if n != "wTrans" else 1.0e8)
    return fields, grid, deltaT, dtR


@pytest.mark.parametrize("case", GRAD_CASES, ids=[OUT[c] for c in GRAD_CASES])
def test_fd_and_dot(R, case):
    """At smooth points: tangent (jvp) = adjoint (vjp) to round-off, and central FD agrees with them on an h sweep
    (the best h reported; FD agreement is not proof of correct physics, the bitwise forward gate is)."""
    loss = _loss(R, case, None if case[2] is None else (1, 9))
    x = smooth_inputs(R)
    xs = (x[0], x[2], x[3])                     # fields, deltaT, dtR (grid fixed)
    grid = x[1]
    L = lambda fl, dt, dtr: loss(fl, grid, dt, dtr)
    rng = np.random.default_rng(11)
    d = jax.tree_util.tree_map(lambda a: jnp.asarray(rng.standard_normal(np.shape(a))) * (jnp.abs(a) + 1e-30), xs)
    tan = float(jax.jit(lambda p, t: jax.jvp(lambda *y: L(*y), p, t)[1])(xs, d))
    g = jax.jit(jax.grad(lambda *y: L(*y), argnums=(0, 1, 2)))(*xs)
    adj = float(sum(jnp.sum(a * b) for a, b in zip(jax.tree_util.tree_leaves(g), jax.tree_util.tree_leaves(d))))
    lj = jax.jit(L)
    errs = {}
    for h in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        xp = jax.tree_util.tree_map(lambda a, b: a + h * b, xs, d)
        xm = jax.tree_util.tree_map(lambda a, b: a - h * b, xs, d)
        errs[h] = abs((float(lj(*xp)) - float(lj(*xm))) / (2 * h) - adj) / abs(adj)
    print(f"{OUT[case]}: |tan-adj|/|adj| = {abs(tan - adj) / abs(adj):.1e}; FD rel. err. by h: "
          + ", ".join(f"{h:.0e}: {e:.1e}" for h, e in errs.items()))
    assert abs(tan - adj) <= 1e-12 * abs(adj)
    assert min(errs.values()) < 1e-6, errs
