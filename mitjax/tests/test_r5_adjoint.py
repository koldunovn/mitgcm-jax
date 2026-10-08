"""R5 tutorial_global_oce_optim/input_ad: the adjoint acceptance of plan Task 16 (the first TAF match), through the
run driver's Model and mitjax/drivers/adjoint_run.py (fc(xx) = COST_FINAL(THE_MAIN_LOOP(CTRL_MAP_INI_GENTIM2D(xx))),
jax.value_and_grad through the per-step-checkpointed scan, the cg2d implicit rule). One module-scoped Adjoint: the
gradient program compiles once (~11 min on a CPU node), the forward once (FD), the tangent once (dot test).

Gated: `ADM adjoint_gradient` at the 3 grdchk points vs the TAF reference results/output_adm.txt (>= 10 digits;
measured 1.2e-15 .. 1.7e-15 relative, every printed digit); `ADM ref_cost_function` and our GRDCHK finite
differences (`ADM finite-diff_grad`, grdchk_eps = 0.1) vs the oracle's FD (job27827478-fdzero, docs/YARDSTICK.md) and
results/ (every printed digit); the gradient trust protocol: FD h-sweep, tangent-linear (jvp) vs adjoint dot test,
a repeat (bitwise); the gradient finite on every lane and zero off the wet interior points; negative control: a
planted error in one cost weight (wtheta of level 1 times 1 + 1e-7) changes fc and fails the 10-digit gate.
"""

import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

RESULTS = "verification/tutorial_global_oce_optim/results/output_adm.txt"


@pytest.fixture(scope="module")
def adj():
    from mitjax.drivers.adjoint_run import Adjoint
    from mitjax.tests import r5_gate as r5
    m = r5.driver_model("adjoint")
    a = Adjoint(m, monitor=True)          # one gradient program, with the adjoint monitor's stats hook
    fc, g, stats = a.value_and_grad()
    a.stats = stats
    return a, float(fc), np.asarray(g)


def _results_adm():
    from mitjax import paths
    lines = (paths.UPSTREAM / RESULTS).read_text(errors="replace").splitlines()
    return [ln.rstrip() for ln in lines if " ADM  " in ln]


def _digits(a, b):
    """testreport's digit count of two numbers (tools/testreport_jax: -log10 of the relative difference)."""
    if a == b:
        return 16
    return int(np.floor(-np.log10(abs(a - b) / max(abs(a), abs(b)))))


def test_adjoint_gradient_matches_taf(adj):
    """admGrd >= 10 digits at all 3 points vs TAF (computed and as printed by adm_lines); the gradient finite on
    every lane, zero on halos and land (the control enters only on the wet interior through CTRL_MAP_INI_GENTIM2D's
    mask)."""
    from mitjax.drivers.adjoint_run import adm_lines
    from mitjax.tests import r5_gate as r5
    a, fc, g = adj
    assert np.all(np.isfinite(g))
    for p, taf in zip(r5.GRDCHK_POINTS, r5.TAF_ADJOINT_GRADIENT):
        assert _digits(float(g[p]), taf) >= 10, (p, float(g[p]), taf)
    sz = a.m.cfg.size
    interior = np.zeros(g.shape, bool)
    interior[:, sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] = True
    wet = interior & (np.asarray(a.m.grid.maskC.data)[:, 0] != 0)
    assert np.all(g[~wet] == 0.0)
    ref = [float(ln.split("=")[1]) for ln in _results_adm() if "adjoint_gradient" in ln]
    assert len(ref) == 3 and tuple(ref) == r5.TAF_ADJOINT_GRADIENT
    ours = [float(adm_lines(fc, g[p], 0.0)[1].split("=")[1]) for p in r5.GRDCHK_POINTS]   # the printed values
    assert all(_digits(o, t) >= 10 for o, t in zip(ours, ref)), (ours, ref)


def test_grdchk_fd_and_ref_cost(adj):
    """Our GRDCHK finite differences (+-grdchk_eps at the 3 points) give the oracle's FD and the TAF build's
    printed FD to every printed digit, and ref_cost_function = 6.20023228182337E+00 (admCst, admFwd)."""
    from mitjax.drivers.adjoint_run import adm_lines
    from mitjax.tests import r5_gate as r5
    a, fc, g = adj
    fd = a.grdchk_fd(r5.GRDCHK_POINTS)
    for (p, fp, fm, gfd), ofd in zip(fd, r5.ORACLE_FD):
        assert f"{gfd:.14E}" == f"{ofd:.14E}", (p, gfd, ofd)     # every printed digit (1PE22.14)
    ref = _results_adm()
    ours = []
    for (p, fp, fm, gfd) in fd:
        ours += adm_lines(fc, g[p], gfd)
    want = [ln for ln in ref if "ref_cost_function" in ln or "finite-diff_grad" in ln]
    got = [ln for ln in ours if "ref_cost_function" in ln or "finite-diff_grad" in ln]
    assert got == want[:6], (got, want[:6])


def test_gradient_trust_protocol(adj):
    """FD h-sweep at the first point (central differences; the best h within 1e-6 relative of the adjoint, above
    the forward-noise floor: h -> 0 loses digits), the TL (jvp) vs adjoint dot test in a random direction over the
    wet interior (<dJ, v> from jvp == <grad, v> to 1e-12 relative), and a repeat of the gradient (bitwise)."""
    from mitjax.tests import r5_gate as r5
    a, fc, g = adj
    p = r5.GRDCHK_POINTS[0]
    errs = {}
    for h in (1.0, 0.3, 0.1, 0.03, 0.01):
        _, fp, fm, gfd = a.grdchk_fd([p], eps=h)[0]
        errs[h] = abs(gfd - g[p]) / abs(g[p])
    assert min(errs.values()) < 1e-6, errs
    rng = np.random.default_rng(20261002)
    v = rng.standard_normal(g.shape) * (g != 0)
    _, tl = a.jvp(np.asarray(v))
    ad = float(np.sum(g * v))
    assert abs(float(tl) - ad) <= 1e-12 * abs(ad), (float(tl), ad)
    fc2, g2, _ = a.value_and_grad()
    assert float(fc2) == fc and np.array_equal(np.asarray(g2).view(np.uint64), g.view(np.uint64))


def test_negative_control_cost_weight(adj):
    """A planted error in one cost weight (COST_WEIGHTS' wtheta of level 1 times 1 + 1e-7): the printed fc differs
    from the oracle's and admGrd misses the 10-digit gate at some check point."""
    import jax.numpy as jnp
    from mitjax.tests import r5_gate as r5
    a, fc, g = adj
    cf = dict(a.model.pkc["cost_fixed"])
    cf["wtheta"] = cf["wtheta"].at[:, 0].multiply(1.0 + 1e-7)
    model = a.model.replace(pkc={**a.model.pkc, "cost_fixed": cf})
    fcb, gb, _ = a.value_and_grad(model=model)
    gb = np.asarray(gb)
    assert f"{float(fcb):.14E}" != f"{r5.REF_COST:.14E}"
    assert min(_digits(float(gb[p]), t) for p, t in zip(r5.GRDCHK_POINTS, r5.TAF_ADJOINT_GRADIENT)) < 10
    del jnp


def _blocks(recs):
    """[[record, ...] per AD_MONITOR block] of the `%MON ad_` records."""
    out, cur = [], None
    for r in recs:
        if "%MON ad_time_tsnumber" in r:
            cur = []
            out.append(cur)
        if cur is not None and "%MON ad_" in r:
            cur.append(r)
    return out


def _value(rec):
    return float(rec.split("=")[1])


def test_adjoint_monitor_self_consistent(adj, monkeypatch):
    """The AD_MONITOR blocks of the reverse sweep (k = 10 .. 0, ADMONITOR at the MONITOR point of each step: the
    step-boundary cotangents of the stats hook plus COST_TILE's transpose, then COPY_ADVAR_OUTP's ADEXCH):
      * format: banners and records as the TAF build prints them (the record names, order and number layout of
        global_ocean.90x40x15/results/output_adm.txt's blocks: the same 32 records per block);
      * with optim's own adjMonitorFreq (0, set_defaults.F:352) no block, as its TAF build prints none;
      * self-consistency: the cotangents of the State at boundaries n and n-1 computed independently (jax.vjp of
        COST_FINAL, and of one step + COST_FINAL, from the forward run's boundary States) give the same printed
        statistics (>= 12 digits, zeros exactly);
      * negative control: without COST_TILE's transpose (the TAF reverse sweep runs ADCOST_TILE before ADMONITOR)
        block 10's adtheta statistics differ (they become 0)."""
    import jax
    from mitjax import paths
    from mitjax.drivers import adjoint_run as ar
    import gc
    a, fc, g = adj
    m = a.m
    # the gradient, forward and tangent programs of this module hold ~3 mappings per kernel (PORTING_LESSONS
    # "XLA:CPU compile aborts = vm.max_map_count"); free them before the step and vjp programs compiled here
    jax.clear_caches()
    gc.collect()
    carries = a.boundary_states()
    recs = ar.ad_monitor_records(m, a.stats, carries, adjMonitorFreq=1.0)
    blocks = _blocks(recs)
    n = len(carries) - 1
    assert [int(b[0].split("=")[1]) for b in blocks] == list(range(n, -1, -1))
    taf = _blocks((paths.UPSTREAM / "verification/global_ocean.90x40x15/results/output_adm.txt")
                  .read_text(errors="replace").splitlines())
    names = [r.split("=")[0] for r in taf[0]]
    for b in blocks:
        assert [r.split("=")[0] for r in b] == names
        for r, t in zip(b[1:], taf[0][1:]):          # the number field: same width and layout
            assert len(r) == len(t) and r[-4] == t[-4] == "E", (r, t)
    assert recs.count("(PID.TID 0000.0001) // Begin AD_MONITOR dynamic field statistics") == n + 1
    assert ar.ad_monitor_records(m, a.stats, carries) == []
    # self-consistency at boundaries n and n-1
    mdl = a._params_fn(a.theta0, a.model)
    vjp_n = jax.jit(jax.grad(lambda st: a._final_cost(mdl, st), allow_int=True))(carries[n])
    vjp_n1 = jax.jit(jax.grad(lambda st: a._final_cost(mdl, a._stepfn(mdl, st, a.xs[-1])), allow_int=True))(
        carries[n - 1])
    for k, gk, blk in ((n, vjp_n, blocks[0]), (n - 1, vjp_n1, blocks[1])):
        ct = {nm: np.asarray(getattr(gk[0][0], nm).data) for nm in ar.AD_STATE}
        ct.update({nm: np.asarray(getattr(gk[0][4]["cost"], nm).data) for nm in ar.AD_COST})
        st = {key: v[None].repeat(n + 1, axis=0) for key, v in ct.items()}
        mine = _blocks(ar.ad_monitor_records(m, st, carries, adjMonitorFreq=1.0))[n - k]
        for r1, r2 in zip(mine, blk):
            x, y = _value(r1), _value(r2)
            assert (x == y == 0.0) or (x != 0.0 and _digits(x, y) >= 12), (k, r1, r2)
    # negative control: COST_TILE's transpose dropped
    import mitjax.pkg.cost.cost_tile as ct_mod
    monkeypatch.setattr(ct_mod, "cost_tile", lambda cost, *args, **kw: cost)
    bad = _blocks(ar.ad_monitor_records(m, a.stats, carries, adjMonitorFreq=1.0))[0]
    th = [(r, b) for r, b in zip(bad, blocks[0]) if "adtheta" in r]
    assert any(_value(r) != _value(b) for r, b in th), th
