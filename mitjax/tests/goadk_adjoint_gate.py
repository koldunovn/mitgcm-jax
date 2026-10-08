"""Helpers of the per-variant GOADK adjoint gates (test_goadk_adjoint_kapgm.py, _kapredi.py, _bottomdrag.py):
global_ocean.90x40x15/input_ad.<variant> through the driver Model and drivers/adjoint_run.GenarrAdjoint vs TAF's
results/output_adm.<variant>.txt and lane A's FD oracle (fdzero run of the variant).

One test file per variant, so the tier1x per-file runner gives each its own process. Each file holds three programs
in turn, released between them because one process cannot hold all three (PORTING_LESSONS "GOADK session 4"):
the gradient (fixture: the gradient and the gradient with a planted weight error), the forward (GRDCHK's FD), the
tangent (dot test). Measured per variant: ~45 min, ~31 GB on one CPU node.
"""

import gc

import numpy as np

RESULTS = "verification/global_ocean.90x40x15/results"


def digits(a, b):
    """testreport's digit count of two numbers (-log10 of the relative difference)."""
    if a == b:
        return 16
    return int(np.floor(-np.log10(abs(a - b) / max(abs(a), abs(b)))))


def release():
    """Drop every compiled program (jax.clear_caches): XLA:CPU kernels hold memory mappings, a process may hold only
    vm.max_map_count of them."""
    import jax
    jax.clear_caches()
    gc.collect()


def results_file(inp):
    return "output_adm.txt" if inp == "input_ad" else f"output_adm.{inp.split('.', 1)[1]}.txt"


def taf(inp):
    """(Grdchk of TAF's results file, its ` ADM  ` lines with the PID.TID prefix)."""
    from mitjax import paths
    from mitjax.io import stdout as so
    path = paths.UPSTREAM / RESULTS / results_file(inp)
    raw = path.read_text(errors="replace").splitlines()
    return so.grdchk(so.read_stdout(path)), [ln.rstrip() for ln in raw if " ADM  " in ln]


def points(inp, m):
    """The 4 grdchk points of the variant as storage indices of its control record: [tile, k, j, i] (genarr3d) or
    [tile, j, i] (genarr2d, xx_bottomdrag); tile = itile-1 + (jtile-1)*nSx."""
    from mitjax.tests import goadk_gate as G
    sz = m.cfg.size
    pts, _, _ = G.grdchk_case(inp)
    out = []
    for _, r in pts:
        t, j, i = r.itile - 1 + (r.jtile - 1) * sz.nSx, r.jtilepos - 1 + sz.OLy, r.itilepos - 1 + sz.OLx
        out.append((t, r.layer - 1, j, i) if G.CONTROL[inp][0] == 3 else (t, j, i))
    return out


def adjoint_fixture(inp):
    """Generator for a module fixture: the zero-control Model of the variant's dumps-on run directory, its
    GenarrAdjoint, the gradient, and the gradient with the control weight x (1 + 1e-7) (a jit argument: the same
    executable); the gradient program is released before the dict is yielded."""
    import jax
    from mitjax.drivers.adjoint_run import GenarrAdjoint
    from mitjax.tests import goadk_model_gate as MG
    m, _ = MG.model(inp, "jdon")
    a = GenarrAdjoint(m)
    try:
        fc, g = a.value_and_grad()
        out = dict(inp=inp, a=a, fc=float(fc), g=np.asarray(g), pts=points(inp, m))
        w = jax.tree_util.tree_map(lambda x: x * (1.0 + 1e-7), a.model.pkc["genarr_w"])
        fcb, gb = a.value_and_grad(model=a.model.replace(pkc={**a.model.pkc, "genarr_w": w}))
        out.update(fcb=float(fcb), gb=np.asarray(gb))
    finally:
        release()
    yield out
    MG.model.cache_clear()
    release()


def check_admgrd(adj, nonzero):
    """admGrd >= 10 digits at the 4 points vs TAF (computed and as adm_lines prints them); positions as TAF's
    `grdchk pos`; the gradient finite on every lane, zero off the wet interior and nonzero at exactly `nonzero` wet
    points; the planted weight error leaves fc unchanged (zero first guess) and misses the 10-digit gate."""
    from mitjax.drivers.adjoint_run import adm_lines
    a, fc, g, pts, inp = adj["a"], adj["fc"], adj["g"], adj["pts"], adj["inp"]
    t, _ = taf(inp)
    sz = a.m.cfg.size
    want = [p.adm["adjoint_gradient"] for p in t.points]
    pos = []
    for p in pts:                         # storage index -> (i, j, k, bi, bj) as `grdchk pos` prints them
        j, i = p[-2:]
        pos.append((i + 1 - sz.OLx, j + 1 - sz.OLy, p[1] + 1 if len(p) == 4 else 1, p[0] % sz.nSx + 1,
                    p[0] // sz.nSx + 1))
    assert len(want) == 4 and [(p.i, p.j, p.k, p.bi, p.bj) for p in t.points] == pos, pos
    got = [float(g[p]) for p in pts]
    assert all(digits(x, y) >= 10 for x, y in zip(got, want)), (got, want)
    printed = [float(adm_lines(fc, g[p], 0.0)[1].split("=")[1]) for p in pts]
    assert all(digits(x, y) >= 10 for x, y in zip(printed, want)), (printed, want)
    assert np.all(np.isfinite(g))
    interior = np.zeros(g.shape, bool)
    interior[..., sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] = True
    mC = np.asarray(a.m.grid.maskC.data)
    wet = interior & ((mC != 0) if g.ndim == 4 else (mC[:, 0] != 0))
    assert np.count_nonzero(g[~wet]) == 0 and np.count_nonzero(g[wet]) == nonzero
    gb = adj["gb"]
    assert adj["fcb"] == fc
    assert max(digits(float(gb[p]), y) for p, y in zip(pts, want)) < 10


def check_fd_lines(adj):
    """GRDCHK's finite differences at the 4 points (+-grdchk_eps of data.grdchk): perturbed costs and FD to every
    printed digit of the oracle's; `ADM ref_cost_function` and `ADM finite-diff_grad` lines identical to TAF's.
    Releases the forward program."""
    from mitjax.drivers.adjoint_run import adm_lines
    from mitjax.io import stdout as so
    from mitjax.tests import goadk_gate as G
    a, fc, g, pts, inp = adj["a"], adj["fc"], adj["g"], adj["pts"], adj["inp"]
    try:
        fd = a.grdchk_fd(pts)
    finally:
        release()
    orc = so.grdchk(G.fd_stdout(inp)).points
    for (p, fp, fm, gfd), o in zip(fd, orc):
        assert f"{fp:.14E}" == f"{o.fcpertplus:.14E}" and f"{fm:.14E}" == f"{o.fcpertminus:.14E}", (p, fp, fm)
        assert f"{gfd:.14E}" == f"{o.adm['finite-diff_grad']:.14E}", (p, gfd)
    _, ref = taf(inp)
    ours = []
    for (p, fp, fm, gfd) in fd:
        ours += adm_lines(fc, g[p], gfd)
    keep = ("ref_cost_function", "finite-diff_grad")
    assert [ln for ln in ours if any(k in ln for k in keep)] == [ln for ln in ref if any(k in ln for k in keep)]


def check_dot(adj, tol=1e-12):
    """TL (jvp) vs adjoint dot test in a random direction over the points of nonzero gradient: |<dJ, v> - <grad, v>|
    <= tol * |<grad, v>|. Releases the tangent program."""
    a, g = adj["a"], adj["g"]
    v = np.random.default_rng(20261002).standard_normal(g.shape) * (g != 0)
    try:
        _, tl = a.jvp(np.asarray(v))
        tl = float(tl)
    finally:
        release()
    ad = float(np.sum(g * v))
    assert abs(tl - ad) <= tol * abs(ad), (tl, ad)
