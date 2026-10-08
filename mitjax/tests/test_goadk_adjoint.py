"""GOADK adjoint (M2 Task 24): global_ocean.90x40x15/input_ad vs TAF, through the driver Model and
drivers/adjoint_run.GenarrAdjoint: fc(xx) = COST_FINAL(THE_MAIN_LOOP(CTRL_MAP_INI_GENARR(xx))), the generic init.
control xx_theta (genarr3d, wunit.data weights), jax.value_and_grad through the per-step-checkpointed scan, the cg2d
implicit rule (cg2d's operator is constant: nonlinFreeSurf = 2 without r*). One module-scoped GenarrAdjoint: the
gradient program compiles once (~27 min, ~31 GB on a CPU node), the forward once (~4 min: FD, h-sweep), the tangent
once (~11 min: dot test); one test file process (tier1x per-file runner).

Gated: `ADM ref_cost_function` (admCst) to every printed digit and our GRDCHK finite differences
(`ADM finite-diff_grad`, grdchk_eps = 1e-4) to every printed digit vs TAF's results/output_adm.txt and lane A's FD
oracle (job27832347-fdzero);
`ADM adjoint_gradient` (admGrd) at the 4 grdchk points >= 10 digits vs TAF (measured 12-14); the gradient trust
protocol (FD h-sweep, tangent-linear vs adjoint dot test, a bitwise repeat); the gradient finite everywhere, nonzero
exactly on the ncvarcomp = 29309 wet interior points; negative control: a planted weight error (wunit x (1 + 1e-7))
fails the 10-digit gate. The control map itself: GenarrAdjoint's CTRL_MAP_INI_GENARR of a planted control on the
zero-control Model's initial state == the initial state of the Model built with that control (the planted -ctrlxx
runs, all four variants), so the differentiated function starts from the gated forward.

The other three variants (.kapgm, .kapredi, .bottomdrag; ~45 min each) are measured by the same protocol in session 4
(jobs 27838400-27838403: admGrd vs TAF 11-14 digits), not in this file.
"""

import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

INP = "input_ad"
RESULTS = "verification/global_ocean.90x40x15/results/output_adm.txt"
VARIANTS = ("input_ad", "input_ad.kapgm", "input_ad.kapredi", "input_ad.bottomdrag")


def _digits(a, b):
    """testreport's digit count of two numbers (-log10 of the relative difference)."""
    if a == b:
        return 16
    return int(np.floor(-np.log10(abs(a - b) / max(abs(a), abs(b)))))


def _same(x, y):
    """Element equality with NaN == NaN at the same places (the State's NaN-initialised fields) and equal bits."""
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    return bool(np.array_equal(x.reshape(-1).view(np.uint8), y.reshape(-1).view(np.uint8)))


def _points(m):
    """The 4 grdchk points of input_ad as storage indices of the xx_theta record [tile, k, j, i]."""
    from mitjax.tests import goadk_gate as G
    sz = m.cfg.size
    pts, _, _ = G.grdchk_case(INP)
    return [(r.itile - 1 + (r.jtile - 1) * sz.nSx, r.layer - 1, r.jtilepos - 1 + sz.OLy, r.itilepos - 1 + sz.OLx)
            for _, r in pts]


def _taf():
    from mitjax import paths
    from mitjax.io import stdout as so
    raw = (paths.UPSTREAM / RESULTS).read_text(errors="replace").splitlines()      # with the PID.TID prefix
    return so.grdchk(so.read_stdout(paths.UPSTREAM / RESULTS)), [ln.rstrip() for ln in raw if " ADM  " in ln]


@pytest.mark.parametrize("inp", VARIANTS)
def test_genarr_map_of_planted_control_is_the_planted_model(inp):
    """genarr_apply(zero-control Model, planted xx) == initial carry (and CTRL_FIELDS.h bottomDragFld) of the Model
    built with the planted control of the -ctrlxx run (whose steps 0-3 are bitwise vs the oracle, test_goadk_model),
    every leaf bitwise; the zero control re-applied leaves the first guess unchanged."""
    import jax
    from mitjax.drivers.adjoint_run import GenarrAdjoint, genarr_apply
    from mitjax.tests import goadk_model_gate as MG
    m0, _ = MG.model(inp, "jdon")
    mp, _ = MG.model(inp, "ctrlxx")
    a = GenarrAdjoint(m0)
    xx = mp.genarr_xx0[a.key].data
    assert np.any(np.asarray(xx) != 0)
    model, st = genarr_apply(m0, a.key, xx, m0.arrays, a.st0)
    la, ta = jax.tree_util.tree_flatten(st[0])
    lb, tb = jax.tree_util.tree_flatten(mp.initial_carry())
    assert ta == tb
    assert [n for n, (x, y) in enumerate(zip(la, lb)) if not _same(x, y)] == []
    if "ctrlf" in m0.arrays.pkc:
        assert _same(model.pkc["ctrlf"].bottomDragFld.data, mp.arrays.pkc["ctrlf"].bottomDragFld.data)
    _, s0 = genarr_apply(m0, a.key, a.theta0, m0.arrays, a.st0)
    assert all(_same(x, y) for x, y in zip(jax.tree_util.tree_leaves(s0[0]), jax.tree_util.tree_leaves(a.st0[0])))
    MG.model.cache_clear()
    _release()


def _release():
    """Drop the compiled programs: every XLA:CPU kernel holds memory mappings and a process may hold only
    vm.max_map_count of them (PORTING_LESSONS, 2026-10-02); the gradient, forward and tangent programs of this Model
    do not fit together (measured: the tangent compile aborts with the other two alive)."""
    import gc

    import jax
    jax.clear_caches()
    gc.collect()


@pytest.fixture(scope="module")
def adj():
    """The gradient program runs three times and is then released: the gradient, its repeat, and the gradient with
    a planted weight error (a jit argument, the same executable)."""
    import jax
    from mitjax.drivers.adjoint_run import GenarrAdjoint
    from mitjax.tests import goadk_model_gate as MG
    m, _ = MG.model(INP, "jdon")
    a = GenarrAdjoint(m)
    try:
        fc, g = a.value_and_grad()
        out = dict(a=a, fc=float(fc), g=np.asarray(g), pts=_points(m))
        fc2, g2 = a.value_and_grad()
        out.update(fc2=float(fc2), g2=np.asarray(g2))
        w = jax.tree_util.tree_map(lambda x: x * (1.0 + 1e-7), a.model.pkc["genarr_w"])
        fcb, gb = a.value_and_grad(model=a.model.replace(pkc={**a.model.pkc, "genarr_w": w}))
        out.update(fcb=float(fcb), gb=np.asarray(gb))
    finally:
        _release()
    yield out
    MG.model.cache_clear()
    _release()


def test_adjoint_gradient_matches_taf(adj):
    """admGrd >= 10 digits at the 4 points vs TAF (computed and as printed by adm_lines); the gradient finite on every
    lane and nonzero exactly on the wet interior points (ncvarcomp = 29309, CTRL_MAP_GENARR3D's maskC)."""
    from mitjax.drivers.adjoint_run import adm_lines
    a, fc, g, pts = adj["a"], adj["fc"], adj["g"], adj["pts"]
    taf, ref = _taf()
    want = [p.adm["adjoint_gradient"] for p in taf.points]
    assert len(want) == 4 and [(p.i, p.j, p.k, p.bi, p.bj) for p in taf.points] == \
        [(i + 1 - 3, j + 1 - 3, k + 1, t % 2 + 1, t // 2 + 1) for t, k, j, i in pts]
    assert all(_digits(float(g[p]), t) >= 10 for p, t in zip(pts, want)), [float(g[p]) for p in pts]
    ours = [float(adm_lines(fc, g[p], 0.0)[1].split("=")[1]) for p in pts]
    assert all(_digits(o, t) >= 10 for o, t in zip(ours, want)), (ours, want)
    assert np.all(np.isfinite(g))
    sz = a.m.cfg.size
    interior = np.zeros(g.shape, bool)
    interior[..., sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] = True
    wet = interior & (np.asarray(a.m.grid.maskC.data) != 0)
    assert np.count_nonzero(g[~wet]) == 0 and np.count_nonzero(g[wet]) == 29309 == np.count_nonzero(wet)


def test_gradient_repeat_bitwise(adj):
    """Trust protocol, repeat: a second run of the gradient program gives the same fc and gradient, bit for bit."""
    assert adj["fc2"] == adj["fc"] and np.array_equal(adj["g2"].view(np.uint64), adj["g"].view(np.uint64))


def test_negative_control_weight(adj):
    """A planted error in the control weight (wunit.data x (1 + 1e-7), a jit argument: the compiled gradient is
    reused): fc is unchanged (the first guess is zero) and admGrd misses the 10-digit gate at every check point."""
    taf, _ = _taf()
    gb = adj["gb"]
    assert adj["fcb"] == adj["fc"]
    assert max(_digits(float(gb[p]), t.adm["adjoint_gradient"]) for p, t in zip(adj["pts"], taf.points)) < 10


def test_grdchk_fd_and_ref_cost(adj):
    """Our GRDCHK finite differences (+-grdchk_eps at the 4 points) and fc: the `ADM ref_cost_function` and
    `ADM finite-diff_grad` lines identical to TAF's results/output_adm.txt; the perturbed costs and FD to every
    printed digit of lane A's FD oracle run."""
    from mitjax.drivers.adjoint_run import adm_lines
    from mitjax.io import stdout as so
    from mitjax.tests import goadk_gate as G
    a, fc, g, pts = adj["a"], adj["fc"], adj["g"], adj["pts"]
    fd = a.grdchk_fd(pts)
    orc = so.grdchk(G.fd_stdout(INP)).points
    for (p, fp, fm, gfd), o in zip(fd, orc):
        assert f"{fp:.14E}" == f"{o.fcpertplus:.14E}" and f"{fm:.14E}" == f"{o.fcpertminus:.14E}", (p, fp, fm)
        assert f"{gfd:.14E}" == f"{o.adm['finite-diff_grad']:.14E}", (p, gfd)
    _, ref = _taf()
    ours = []
    for (p, fp, fm, gfd) in fd:
        ours += adm_lines(fc, g[p], gfd)
    keep = ("ref_cost_function", "finite-diff_grad")
    assert [ln for ln in ours if any(k in ln for k in keep)] == [ln for ln in ref if any(k in ln for k in keep)]


def test_fd_hsweep(adj):
    """Trust protocol, FD h-sweep at the first point (central differences, the forward program of the FD test): the
    best h within 1e-6 relative of the adjoint (measured 1.6e-7 at h = 1e-3; truncation above, forward noise below).
    Releases the forward program."""
    a, g, p = adj["a"], adj["g"], adj["pts"][0]
    try:
        errs = {}
        for h in (1e-2, 1e-3, 1e-4, 1e-5):
            _, fp, fm, gfd = a.grdchk_fd([p], eps=h)[0]
            errs[h] = abs(gfd - g[p]) / abs(g[p])
    finally:
        _release()
    assert min(errs.values()) < 1e-6, errs


def test_dot_test(adj):
    """Trust protocol, the TL (jvp) vs adjoint dot test in a random direction over the wet points: <dJ, v> ==
    <grad, v> to 1e-12 relative (measured 2.1e-14). Releases the tangent program."""
    a, g = adj["a"], adj["g"]
    rng = np.random.default_rng(20261002)
    v = rng.standard_normal(g.shape) * (g != 0)
    try:
        _, tl = a.jvp(np.asarray(v))
        tl = float(tl)
    finally:
        _release()
    ad = float(np.sum(g * v))
    assert abs(tl - ad) <= 1e-12 * abs(ad), (tl, ad)
