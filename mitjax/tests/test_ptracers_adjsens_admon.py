"""tutorial_tracer_adjsens/input_ad: the adjoint monitor (PTRACERS lane, plan Task 26) against the TAF build's
`%MON ad_*` blocks of results/output_adm.txt (k = 4 .. 0): ADMONITOR's ad_dynstat and ad_forcing (monitorSelect = 4:
adQnet, adQsw (SHORTWAVE_HEATING), adEmPmR, adfu, adfv) and ADPTRACERS_MONITOR's ad_trcstat_adptracer01.

The adjoint variables come from a step-by-step reverse sweep of the driver Model: the forward boundary States
(Adjoint.boundary_states), the cotangent of COST_FINAL at the last one, then one jitted step vjp per step back;
adjoint_run.ad_monitor_records_ptr adds COST_TILE's COST_TRACER transpose (the TAF sweep runs ADCOST_TILE before
ADMONITOR) and makes the mon_AdVarExch = 2 copies. The sweep's control gradient (through CTRL_MAP_INI_GENARR and
CONVECTIVE_ADJUSTMENT_INI) is checked against TAF's admGrd too.

Measured (job 27840080) and gated:
* blocks 4 and 3 (the reverse sweep before any CG2D_NSA adjoint): every record, every printed digit;
* blocks 2, 1, 0: ad_trcstat_adptracer01 every printed digit (the passive-tracer adjoint does not go through the
  pressure solver); ad_dynstat and ad_forcing >= 11 digits (measured: >= 13 on max/min/sd/del2, 11-12 on the means,
  which are sums with cancellation). The first block that differs is the first one after a CG2D_NSA adjoint in the
  reverse sweep: TAF differentiates the solver's iterations, we use the implicit-derivative rule (no TAF-compatible
  switch, Nikolay decides).
* negative control: without COST_TILE's COST_TRACER transpose block 4's adptracer statistics differ;
* the trust protocol's dot test: the tangent-linear model (jvp) vs this sweep's gradient in a random direction.
Tier 1x (~16 min on a CPU node: the forward step, the step vjp).
"""

import functools
import gc

import numpy as np

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

INP = "input_ad"


@functools.lru_cache(maxsize=None)
def sweep():
    """(adjoint, boundary carries, stats {name: [n+1, ...]}, control gradient) of the reverse sweep."""
    import jax
    from mitjax.drivers import adjoint_run as ar
    from mitjax.tests.test_ptracers_latlon import model
    m = model(("tutorial_tracer_adjsens", INP), "adjsens-admon")
    a = ar.GenarrAdjoint(m)
    carries = a.boundary_states()
    mdl = a.model
    n = len(carries) - 1
    cts = {n: jax.jit(jax.grad(lambda c, t, it: a._final_cost(mdl, (c, t, it))))(*carries[n])}
    vjp_step = jax.jit(lambda model, c, t, it, x, ct: jax.vjp(lambda cc: a._stepfn(model, (cc, t, it), x)[0], c)[1](
        ct)[0])
    for k in range(n - 1, -1, -1):
        c, t, it = carries[k]
        cts[k] = vjp_step(mdl, c, t, it, a.xs[k], cts[k + 1])
    g = np.asarray(jax.jit(lambda th, ct: jax.vjp(lambda t_: a._init(t_, mdl, a.st0)[0], th)[1](ct)[0])(
        a.theta0, cts[0]))
    stats = {}
    for k in range(n + 1):
        for key, v in ar.monitor_stats_ptr(mdl, (cts[k],)).items():
            stats.setdefault(key, []).append(np.asarray(v))
    jax.clear_caches()
    gc.collect()
    return a, carries, {k: np.stack(v) for k, v in stats.items()}, g


def blocks(lines):
    out, cur = [], None
    for r in lines:
        if "%MON ad_time_tsnumber" in r:
            cur = {}
            out.append(cur)
        if cur is not None and "%MON ad_" in r:
            k, v = r.split("%MON ")[1].split("=")
            cur[k.strip()] = v.strip()
    return out


def taf_blocks():
    from mitjax.tests.test_ptracers_adjsens_ad import results_lines
    return blocks(results_lines(INP))


def test_reverse_sweep_gradient_matches_taf():
    from mitjax.drivers.adjoint_run import adm_lines
    from mitjax.tests.test_ptracers_adjsens_ad import points, results_lines
    a, carries, stats, g = sweep()
    want = [ln for ln in results_lines(INP) if " ADM  adjoint_gradient" in ln]
    assert [adm_lines(0.0, g[p], 0.0)[1] for p in points(a.m)] == want


def test_ad_monitor_blocks_vs_taf():
    from mitjax.drivers import adjoint_run as ar
    from mitjax.tests.test_ptracers_adjsens_ad import digits
    a, carries, stats, g = sweep()
    ours, theirs = blocks(ar.ad_monitor_records_ptr(a.m, stats, carries)), taf_blocks()
    assert [b["ad_time_tsnumber"] for b in ours] == ["4", "3", "2", "1", "0"]
    assert len(theirs) == 5
    for bo, bt in zip(ours, theirs):
        assert set(bo) == set(bt), sorted(set(bo) ^ set(bt))
        k = int(bo["ad_time_tsnumber"])
        for name, vt in bt.items():
            if k >= 3 or name.startswith(("ad_time", "ad_trcstat")):
                assert bo[name] == vt, (k, name, bo[name], vt)
            else:
                d = digits(float(bo[name]), float(vt))
                assert d >= 11, (k, name, bo[name], vt, d)


def test_negative_control_cost_tracer_transpose(monkeypatch):
    """Without COST_TILE's COST_TRACER transpose (the TAF sweep runs ADCOST_TILE before ADMONITOR) block 4's
    adptracer statistics differ from TAF's."""
    from mitjax.drivers import adjoint_run as ar
    import mitjax.pkg.cost.cost_tracer as ct_mod
    a, carries, stats, g = sweep()
    monkeypatch.setattr(ct_mod, "cost_tracer", lambda objf, **kw: objf)
    bad = blocks(ar.ad_monitor_records_ptr(a.m, stats, carries))[0]
    good = taf_blocks()[0]
    assert any(bad[n] != good[n] for n in good if n.startswith("ad_trcstat")), bad


def test_dot_test_tangent_vs_adjoint():
    """The tangent-linear model (jax.jvp of fc through the scan) vs the reverse sweep's gradient in a random direction
    over the wet interior: <dJ, v> == <grad, v> to 1e-12 relative."""
    a, carries, stats, g = sweep()
    rng = np.random.default_rng(20261002)
    v = rng.standard_normal(g.shape) * (g != 0)
    _, tl = a.jvp(np.asarray(v))
    ad = float(np.sum(g * v))
    assert abs(float(tl) - ad) <= 1e-12 * abs(ad), (float(tl), ad)
