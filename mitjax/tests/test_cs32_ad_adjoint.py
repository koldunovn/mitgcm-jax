"""global_ocean.cs32x15/input_ad adjoint vs TAF (plan Task 25, M2 adjoint family; GO lane session 10): fc(xx) =
COST_FINAL(THE_MAIN_LOOP(CTRL_MAP_INI_GENARR(xx))) for the grdchk control xx_theta (genarr3d no. 1 of the build's
8 live controls; the others stay at their zero first guess, drivers/adjoint_run.GenarrAdjoint(key=)), through the
driver Model on the cube (12 tiles, VECINV, GM AdvForm, DST3 with implicit vertical advection, r* with nonlinFreeSurf
= 4, COST_TEST's TSQUARED cost), jax.value_and_grad through the per-step-checkpointed scan.

The CG2D derivative follows the run's own cg2dFullAdjoint as TAF's CG2D_MAD does (mitjax/ad/modes.py; plan
"Decisions 2026-10-02" item 3 revised): here .FALSE. (data.autodiff empty; autodiff_readparms.F:87), so the cg2d
operator is passive (cg2d_mad.F:220-236: aW2d_ad = aS2d_ad = aC2d_ad = 0; pkg/autodiff/cg2d.flow ACTIVE = 1,2),
while nonlinFreeSurf = 4 rebuilds the operator from the r* hFac every step (UPDATE_CG2D), so theta reaches it through
eta. The exact option (Cg2dParams.mjx_cg2d_derivative = "exact") differentiates the operator too: the exact gradient of
the converged solves, 6-7 digits from TAF's, and the one the FD h-sweep agrees with best at all 4 points.
Gated:
* the run setting's gradient vs TAF's `ADM adjoint_gradient` >= 10 digits at the 4 points (measured 1.7e-11 ..
  6.4e-11 with session 10's test-side reproduction: TAF's adjoint CG stops at cg2dTargetResidual = 1e-9) -- every
  part of the adjoint is TAF's; negative control: a planted cost weight (data.cost mult_test x (1 + 1e-7), a jit
  argument) changes fc and misses this gate;
* the switch on the whole model (backward-only): fc of the exact option's program equals the run setting's bit for
  bit; the gradients differ by more than the gate at every point (the effect test; unit tests of the switch:
  test_ad_modes_cg2d.py);
* `ADM ref_cost_function` and our GRDCHK finite differences (`ADM finite-diff_grad`, grdchk_eps = 1e-2) identical to
  TAF's lines; perturbed costs and FD to every printed digit of lane A's FD oracle (fdzero run);
* the exact option vs FD (second gate): the FD h-sweep's best step within HSWEEP_TOL of the exact gradient and closer
  to it than to the run setting's at every point;
* both gradients finite on every lane, zero off the wet interior; the trust protocol on the run setting: a bitwise
  repeat and the tangent-linear vs adjoint dot test.

Programs, each released before the next (PORTING_LESSONS "XLA:CPU compile aborts = vm.max_map_count"): the run
setting's gradient (run three times: plain, repeat, planted weight), the exact option's gradient, the forward (FD,
h-sweep), the tangent (dot test). Measurement scripts: scripts/cs32_adjoint.py (flag exact),
scripts/cs32_compile_study.py.
"""

import contextlib

import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

EXP, INP = "global_ocean.cs32x15", "input_ad"
RESULTS = f"verification/{EXP}/results/output_adm.txt"
NCVARCOMP = 55522                       # output_adm.txt `ph-test icomp, ncvarcomp, ichknum  1  55522  1`
ADMGRD_DIGITS = 10                      # plan Task 24/25: admGrd >= 10 digits at the 4 grdchk points
HSWEEP = (1e-2, 1e-3, 1e-4)             # FD steps of the h-sweep (grdchk_eps x 1, 0.1, 0.01)
HSWEEP_TOL = 1e-7                       # best FD vs the exact gradient (measured 8.0e-10 .. 1.7e-8, job 27843315)
DOT_TOL = 1e-12                         # jvp vs adjoint dot test (as GOADK; measured 9.4e-14 passive, job 27843316)


def _digits(a, b):
    """testreport's digit count of two numbers (-log10 of the relative difference)."""
    if a == b:
        return 16
    return int(np.floor(-np.log10(abs(a - b) / max(abs(a), abs(b)))))


@contextlib.contextmanager
def _exp():
    """The goadk helpers read their experiment from a module constant: cs32x15 while inside."""
    from mitjax.tests import goadk_gate as G
    from mitjax.tests import goadk_model_gate as MG
    old = (G.EXP, MG.EXP)
    G.EXP = MG.EXP = EXP
    MG.model.cache_clear()
    try:
        yield G, MG
    finally:
        G.EXP, MG.EXP = old
        MG.model.cache_clear()


def _release():
    import gc

    import jax
    jax.clear_caches()
    gc.collect()


def _taf():
    from mitjax import paths
    from mitjax.io import stdout as so
    raw = (paths.UPSTREAM / RESULTS).read_text(errors="replace").splitlines()
    return so.grdchk(so.read_stdout(paths.UPSTREAM / RESULTS)), [ln.rstrip() for ln in raw if " ADM  " in ln]


@pytest.fixture(scope="module")
def adj():
    """The Model (zero control), the 4 grdchk points and GenarrAdjoint(key = xx_theta): the gradient with the run's own
    CG2D derivative setting (run three times: plain, the repeat, the planted cost weight) and with the exact option
    (mitjax/ad/modes.with_cg2d_derivative(..., "exact") on the Model's cg2d_params). Each gradient program is released
    after its runs."""
    from mitjax.ad.modes import cg2d_operator_adjoint, with_cg2d_derivative
    from mitjax.drivers.adjoint_run import GenarrAdjoint
    from mitjax.io import stdout as so
    from mitjax.pkg.ctrl.ctrl_readparms import ctrl_readparms_genarr
    with _exp() as (G, MG):
        m, _ = MG.model(INP, "jdon")
        key = next((3, g.iarr) for g in ctrl_readparms_genarr(m.exp.run, 3, m.cfg) if g.file.strip() == "xx_theta")
        pts, _, _ = G.grdchk_case(INP, maskC=np.asarray(m.grid.maskC.data))
        fd_oracle = so.grdchk(G.fd_stdout(INP))
    sz = m.cfg.size
    P = [(r.itile - 1 + (r.jtile - 1) * sz.nSx, r.layer - 1, r.jtilepos - 1 + sz.OLy, r.itilepos - 1 + sz.OLx)
         for _, r in pts]
    out = dict(m=m, key=key, pts=P, fd_oracle=fd_oracle)
    a = GenarrAdjoint(m, key=key)
    out["operator_adjoint"] = cg2d_operator_adjoint(m.cfg, a.model.cg2d_params)
    try:
        fc, g = a.value_and_grad()
        out.update(a=a, fc=float(fc), g=np.asarray(g))
        fc2, g2 = a.value_and_grad()
        out.update(fc2=float(fc2), g2=np.asarray(g2))
        cf = a.model.pkc["cost_fixed"]
        prm = {**cf["params"], "mult_test": cf["params"]["mult_test"] * (1.0 + 1e-7)}
        fcb, gb = a.value_and_grad(model=a.model.replace(pkc={**a.model.pkc, "cost_fixed": {**cf, "params": prm}}))
        out.update(fcb=float(fcb), gb=np.asarray(gb))
    finally:
        _release()
    exact = a.model.replace(cg2d_params=with_cg2d_derivative(a.model.cg2d_params, "exact"))
    out["operator_adjoint_exact"] = cg2d_operator_adjoint(m.cfg, exact.cg2d_params)
    try:
        fce, ge = a.value_and_grad(model=exact)
        out.update(fce=float(fce), ge=np.asarray(ge))
    finally:
        _release()
    yield out
    _release()


def _want():
    taf, _ = _taf()
    return [p.adm["adjoint_gradient"] for p in taf.points]


def test_grdchk_points_and_control(adj):
    """The 4 grdchk points as TAF prints them (`grdchk pos`: i, j, k, bi, bj); the differentiated control is
    genarr3d no. 1 (xx_theta) of the 8 live controls."""
    taf, _ = _taf()
    sz = adj["m"].cfg.size
    pos = [(i + 1 - sz.OLx, j + 1 - sz.OLy, k + 1, t % sz.nSx + 1, t // sz.nSx + 1) for t, k, j, i in adj["pts"]]
    assert [(p.i, p.j, p.k, p.bi, p.bj) for p in taf.points] == pos
    # genarr_live lists the genarr controls only (CTRL_SIZE.h: 4 genarr3d; the 4 gentim2d are not genarr)
    assert adj["key"] == (3, 1) and adj["m"].genarr_live == ((3, 1), (3, 2), (3, 3), (3, 4)), adj["m"].genarr_live


def test_run_setting_gradient_matches_taf(adj):
    """The run's own CG2D derivative setting (cg2dFullAdjoint = .FALSE.: the operator passive, as TAF's CG2D_MAD) gives
    TAF's admGrd to >= ADMGRD_DIGITS at the 4 points (computed and as adm_lines prints them)."""
    from mitjax.drivers.adjoint_run import adm_lines
    assert adj["operator_adjoint"] is False and adj["operator_adjoint_exact"] is True
    g, pts, want = adj["g"], adj["pts"], _want()
    got = [float(g[p]) for p in pts]
    print("admGrd run setting vs TAF digits:", [_digits(x, y) for x, y in zip(got, want)],
          "rel:", [abs(x - y) / abs(y) for x, y in zip(got, want)])
    assert all(_digits(x, y) >= ADMGRD_DIGITS for x, y in zip(got, want)), (got, want)
    printed = [float(adm_lines(adj["fc"], g[p], 0.0)[1].split("=")[1]) for p in pts]
    assert all(_digits(x, y) >= ADMGRD_DIGITS for x, y in zip(printed, want)), (printed, want)


def test_negative_control_cost_weight(adj):
    """A planted error in the cost weight (data.cost mult_test x (1 + 1e-7), a jit argument: the compiled
    run-setting gradient is reused): fc changes and admGrd misses the gate at every check point."""
    gb = adj["gb"]
    assert adj["fcb"] != adj["fc"]
    assert max(_digits(float(gb[p]), y) for p, y in zip(adj["pts"], _want())) < ADMGRD_DIGITS


def test_exact_option_forward_identical_gradient_differs(adj):
    """The switch is backward-only and has an effect here: the exact option's program gives the run setting's fc bit
    for bit, and its gradient differs from the run setting's by more than the gate at every point (the exact option vs
    TAF: measured 7, 6, 6, 7 digits, session 10)."""
    assert adj["fce"] == adj["fc"]
    ge, g, pts, want = adj["ge"], adj["g"], adj["pts"], _want()
    d_run = [_digits(float(ge[p]), float(g[p])) for p in pts]
    d_taf = [_digits(float(ge[p]), y) for p, y in zip(pts, want)]
    print("admGrd exact option vs run setting digits:", d_run, "vs TAF:", d_taf,
          "rel vs TAF:", [abs(float(ge[p]) - y) / abs(y) for p, y in zip(pts, want)])
    assert max(d_run) < ADMGRD_DIGITS, d_run


def test_gradient_finite_and_on_wet_points(adj):
    """Both gradients (run setting, exact option) finite on every lane, zero off the wet interior (CTRL_MAP_GENARR3D's
    maskC: ncvarcomp = 55522 wet points) and zero on the wet interior exactly where FREEZE_SURFACE overwrites the initial theta before
    anything reads it (allowFreezing: k = 1, theta < Tfreezing = -1.9, freeze_surface.F:48, :55-59, called in
    DO_OCEANIC_PHYS at the start of the first step: 43 surface points), nonzero everywhere else."""
    m, a, g, ge = adj["m"], adj["a"], adj["g"], adj["ge"]
    sz = m.cfg.size
    interior = np.zeros(g.shape, bool)
    interior[..., sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] = True
    wet = interior & (np.asarray(m.grid.maskC.data) != 0)
    assert np.count_nonzero(wet) == NCVARCOMP and m.params.allowFreezing
    th0 = np.asarray(a.st0[0][0].theta.data)
    frozen = wet & (th0 < -1.9)
    frozen[:, 1:] = False
    assert np.count_nonzero(frozen) == 43
    for x in (g, ge):
        assert np.all(np.isfinite(x))
        assert np.count_nonzero(x[~wet]) == 0
        assert np.array_equal(wet & (x == 0), frozen)


def test_gradient_repeat_bitwise(adj):
    """Trust protocol, repeat: a second run of the run setting's gradient program gives the same fc and gradient, bit
    for bit."""
    assert adj["fc2"] == adj["fc"] and np.array_equal(adj["g2"].view(np.uint64), adj["g"].view(np.uint64))


def test_grdchk_fd_and_ref_cost_then_hsweep(adj):
    """Our GRDCHK finite differences (+-grdchk_eps at the 4 points) and fc: the `ADM ref_cost_function` and
    `ADM finite-diff_grad` lines identical to TAF's results/output_adm.txt; the perturbed costs and FD to every
    printed digit of lane A's FD oracle run. The exact option vs FD (h-sweep: central differences, h in HSWEEP, at the
    4 points): the best h within HSWEEP_TOL of the exact gradient, and closer to it than to the run setting's at every
    point (measured 8.0e-10 .. 1.7e-8 vs 1.7e-8 .. 1.9e-7, job 27843315). Releases the forward program."""
    from mitjax.drivers.adjoint_run import adm_lines
    a, fc, g, ge, pts = adj["a"], adj["fc"], adj["g"], adj["ge"], adj["pts"]
    try:
        fd = a.grdchk_fd(pts)
        sweep = [[a.grdchk_fd([p], eps=h)[0][3] for h in HSWEEP] for p in pts]
    finally:
        _release()
    best = [min(abs(x - ge[p]) / abs(ge[p]) for x in sw) for p, sw in zip(pts, sweep)]
    best_run = [min(abs(x - g[p]) / abs(g[p]) for x in sw) for p, sw in zip(pts, sweep)]
    print("h-sweep best FD vs exact option:", best, "vs run setting:", best_run)
    for (p, fp, fm, gfd), o in zip(fd, adj["fd_oracle"].points):
        assert f"{fp:.14E}" == f"{o.fcpertplus:.14E}" and f"{fm:.14E}" == f"{o.fcpertminus:.14E}", (p, fp, fm)
        assert f"{gfd:.14E}" == f"{o.adm['finite-diff_grad']:.14E}", (p, gfd)
    _, ref = _taf()
    ours = []
    for (p, fp, fm, gfd) in fd:
        ours += adm_lines(fc, g[p], gfd)
    keep = ("ref_cost_function", "finite-diff_grad")
    assert [ln for ln in ours if any(k in ln for k in keep)] == [ln for ln in ref if any(k in ln for k in keep)]
    assert max(best) < HSWEEP_TOL, best
    assert all(x < y for x, y in zip(best, best_run)), (best, best_run)


def test_dot_test(adj):
    """Trust protocol, the TL (jvp) vs adjoint dot test of the run setting's gradient (the tangent follows the same
    CG2D setting: one rule for both directions) in a random direction over its nonzero points: <dJ, v> == <grad, v>
    to DOT_TOL relative. Releases the tangent program."""
    a, g = adj["a"], adj["g"]
    v = np.random.default_rng(20261002).standard_normal(g.shape) * (g != 0)
    try:
        _, tl = a.jvp(np.asarray(v))
        tl = float(tl)
    finally:
        _release()
    ad = float(np.sum(g * v))
    print("dot test run setting: tl", tl, "ad", ad, "rel", abs(tl - ad) / abs(ad))
    assert abs(tl - ad) <= DOT_TOL * abs(ad), (tl, ad)
