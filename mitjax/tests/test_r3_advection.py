"""R3 advect_xy / advect_xz (plan Task 14): the multi-dimensional advection driver GAD_ADVECTION (schemes 33, 42,
51, 52 here), GAD_SOM_ADVECT in the model (80, 81) with the moments carried in the State and exchanged in
DO_FIELDS_BLOCKING_EXCHANGES, SALT_INTEGRATE, the C4 scheme with ADAMS_BASHFORTH3 (ab3_c4), a NONLIN_FRSURF build
run with the linear free surface (advect_xz). Helpers: mitjax/tests/advect_gate.py.

* every dumped field of every stage of steps 1-3 bitwise (element equality, bit patterns, finite; all points incl.
  halos and the multi-wrap halo of advect_xz, sNy = 1 < OLy) for advect_xy/input, input.ab3_c4, advect_xz/input,
  input.pqm; the later MONITOR step of the jaxdump3 runs (steps 9 / 15);
* negative controls measured to bite; gradients per scheme finite on every lane; FD at a smooth point;
* whole runs (python -m mitjax run's Model and forward): every %MON block identical to the oracle STDOUT and
  tools/testreport_jax.py digits >= the yardstick vs results/ and 16 vs the oracle (test_r3_advection_tier1.py holds
  the two tier-1 smoke runs).

advect_xz/input.nlfs is not gated here (pending: the r*/NLFS step order with staggerTimeStep, FREESURF_RESCALE_G and
the implicit vertical advection path)."""

import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from mitjax.farray import FArray  # noqa: E402
from mitjax.tests import advect_gate as ag  # noqa: E402

VARIANTS = [f"{e}/{i}" for e, i in ag.VARIANTS]
# the stages each variant's oracle must dump (the gate checks it compared them): T10/T20 where GAD_ADVECTION runs
EXPECT = {
    "advect_xy/input": ("T20_salt_adv", "T11_temp_gT", "T21_salt_gS", "T02_temp_integrate", "T03_salt_integrate",
                        "S16_blocking_exchanges"),
    "advect_xy/input.ab3_c4": ("T11_temp_gT", "T21_salt_gS", "T02_temp_integrate", "T03_salt_integrate",
                               "S16_blocking_exchanges"),
    "advect_xz/input": ("T10_temp_adv", "T11_temp_gT", "T21_salt_gS", "T03_salt_integrate", "S16_blocking_exchanges"),
    "advect_xz/input.pqm": ("T10_temp_adv", "T20_salt_adv", "T11_temp_gT", "T21_salt_gS", "T03_salt_integrate",
                            "S16_blocking_exchanges"),
}


def _m(v):
    e, i = v.split("/")
    return ag.model(e, i)


@pytest.mark.parametrize("v", VARIANTS)
def test_initial_state_bitwise(v):
    """INITIALISE_VARIA (+ GAD_INIT_VARIA) vs S00_begin: every State field the oracle dumps, bit patterns."""
    from mitjax.tests import r1_gate as rg
    m = _m(v)
    r = rg.compare_stage(m.ds, m.it0, "S00_begin", m.state0)
    assert len(r) >= 10 and not ag.bad(r), ag.bad(r)


@pytest.mark.parametrize("v", VARIANTS)
def test_substeps_steps_1_to_3_bitwise(v):
    m = _m(v)
    for st in EXPECT[v]:
        assert st in m.stages, (st, m.stages)
    res, _ = ag.run_steps(m, m.params, 3)
    for k, per_stage in enumerate(res):
        for st, r in per_stage.items():
            assert r, (k, st, "no fields compared")
            assert not ag.bad(r), (k, st, ag.bad(r))
            assert all(x[2] == 0 for x in r.values()), (k, st, "non-finite")


def _nd(m, n, stages, mutate_state=None, at=None):
    """Bit-pattern differences per step and stage vs the oracle for n steps of a freshly traced step function (so
    that monkeypatched module attributes take effect); mutate_state(k, state) -> state before step k+1."""
    from mitjax.tests import r1_gate as rg
    from mitjax.tests import r2_gate as r2
    fn = m.step_fn(stages)
    state, ff, phi0, t, it = m.state0, m.ff, m.phi0surf, m.prm.time.startTime, m.prm.time.nIter0
    out = []
    for k in range(n):
        if mutate_state is not None:
            state = mutate_state(k, state)
        state, ff, phi0, t, it, _, pr = fn(m.grid, m.params, m.eos, m.cg2dh, m.cg2d_params, m.ex, state, ff, phi0,
                                           jnp.int32(k + 1), jnp.float64(t), jnp.int32(it))
        out.append({s: sum(v[3] for v in rg.compare_stage(m.ds, m.it0 + k, s, r2.stage_values(s, pr[s])).values())
                    for s in stages})
    return out


def test_negative_control_multidim_splitting(monkeypatch):
    """advect_xy/input, salt (DST3FL, GAD_ADVECTION): the Y pass fed theta^(n) instead of the X pass's theta^(n+1/3)
    (the planted error removes the direction splitting) -> T20 differs from step 1."""
    import mitjax.pkg.generic_advdiff.gad_advection as ga
    m = _m("advect_xy/input")
    seen = {}
    sx, sy = ga._scheme_x, ga._scheme_y

    def x(scheme, k, *a):
        seen[k] = a[4]                                  # localTij entering the X pass = tracer level k
        return sx(scheme, k, *a)

    def y(scheme, k, dt, vT, vF, mS, localTij, af, kc, g):
        return sy(scheme, k, dt, vT, vF, mS, seen[k], af, kc, g)
    monkeypatch.setattr(ga, "_scheme_x", x)
    monkeypatch.setattr(ga, "_scheme_y", y)
    r = _nd(m, 1, ("T20_salt_adv", "T11_temp_gT"))
    assert r[0]["T20_salt_adv"] > 0 and r[0]["T11_temp_gT"] == 0, r


@pytest.mark.parametrize("v", ["advect_xz/input", "advect_xz/input.pqm"])
def test_negative_control_vertical_ppm_pqm(monkeypatch, v):
    """advect_xz (PPM WENO 42 / PQM mono 51 for theta): the vertical flux of GAD_PPM_ADV_R / GAD_PQM_ADV_R dropped at
    the interface k = 2 -> T10 differs from step 1."""
    import mitjax.pkg.generic_advdiff.gad_advection as ga
    m = _m(v)
    name = "gad_ppm_adv_r" if v == "advect_xz/input" else "gad_pqm_adv_r"
    orig = getattr(ga, name)

    def planted(*a, **kw):
        afr = orig(*a, **kw)
        return FArray(afr.data.at[:, 1].set(0.), afr.name, tiled=afr.tiled, _dims=afr.dims)     # afr(:,:,2) = 0
    monkeypatch.setattr(ga, name, planted)
    r = _nd(m, 1, ("T10_temp_adv",))
    assert r[0]["T10_temp_adv"] > 0, r


@pytest.mark.parametrize("v,field,stage", [("advect_xy/input", "som_T", "T11_temp_gT"),
                                           ("advect_xz/input", "som_S", "T21_salt_gS")])
def test_negative_control_som_moments_not_carried(v, field, stage):
    """SOM (80 for theta in advect_xy, 81 for salt in advect_xz): the moments reset to zero before step 2 (not
    carried from step 1) -> the SOM tendency (gated through T11 / T21) differs from step 2; step 1 unchanged."""
    m = _m(v)

    def reset(k, state):
        if k == 1:
            return state.replace(**{field: tuple(FArray(jnp.zeros_like(a.data), a.name, tiled=a.tiled, _dims=a.dims)
                                                 for a in getattr(state, field))})
        return state
    r = _nd(m, 2, (stage,), mutate_state=reset)
    assert r[0][stage] == 0 and r[1][stage] > 0, r


def test_negative_control_som_exchanges_dropped(monkeypatch):
    """advect_xy/input: GAD_SOM_EXCHANGES (DO_FIELDS_BLOCKING_EXCHANGES :79) skipped -> the moments' halos are stale
    and the SOM tendency (T11) differs from step 2."""
    import mitjax.pkg.generic_advdiff.gad_som_exchanges as gse
    m = _m("advect_xy/input")
    monkeypatch.setattr(gse, "gad_som_exchanges", lambda *, cfg, ex, som_T, som_S: (som_T, som_S))
    r = _nd(m, 2, ("T11_temp_gT",))
    assert r[0]["T11_temp_gT"] == 0 and r[1]["T11_temp_gT"] > 0, r


def test_negative_control_ab3_beta():
    """advect_xy/input.ab3_c4: beta_AB one ulp up (the third-order weight; ADAMS_BASHFORTH3 uses it only from the
    third step, adams_bashforth3.F:93-96) -> T11 unchanged at steps 1 and 2, differs at step 3. (A first control,
    the m1/m2 slot parity flipped on both read and write, did not bite (dev job 27832279): it is a consistent
    relabelling of the two slots, which start equal.)"""
    m = _m("advect_xy/input.ab3_c4")
    p = m.params.replace(traced={"beta_AB": np.float64(np.nextafter(float(m.params.beta_AB), 1.))})
    from mitjax.tests import r1_gate as rg
    from mitjax.tests import r2_gate as r2
    st = ("T11_temp_gT",)
    fn = m.step_fn(st)
    state, ff, phi0, t, it = m.state0, m.ff, m.phi0surf, m.prm.time.startTime, m.prm.time.nIter0
    r = []
    for k in range(3):
        state, ff, phi0, t, it, _, pr = fn(m.grid, p, m.eos, m.cg2dh, m.cg2d_params, m.ex, state, ff, phi0,
                                           jnp.int32(k + 1), jnp.float64(t), jnp.int32(it))
        r.append(sum(v[3] for v in rg.compare_stage(m.ds, m.it0 + k, st[0], r2.stage_values(st[0], pr[st[0]]))
                     .values()))
    assert r[0] == 0 and r[1] == 0 and r[2] > 0, r


@pytest.mark.parametrize("v", VARIANTS)
def test_later_monitor_step_bitwise(v):
    """The jaxdump3 run's later step (9 for the advect_xz variants and ab3_c4, 15 for advect_xy/input: the step that
    writes the next %MON block) and its steps 1-2, every dumped stage bitwise."""
    e, i = v.split("/")
    m = _m(v)
    ds3, its = ag.oracle3(e, i)
    last = max(its) - m.prm.time.nIter0
    at = tuple(it - m.prm.time.nIter0 for it in its if it - m.prm.time.nIter0 in (0, 1, last))
    stages = tuple(st for st in m.stages if st in ds3.stages(its[0]))
    res, _ = ag.run_steps(m, m.params, last + 1, stages=stages, ds=ds3, at=at)
    assert len(res) == len(at)
    for k, per_stage in zip(at, res):
        for st, r in per_stage.items():
            assert r, (k, st)
            assert not ag.bad(r), (k, st, ag.bad(r))


@pytest.mark.parametrize("v", ["advect_xy/input.ab3_c4", "advect_xz/input.pqm"])
def test_whole_run_monitor_and_digits(v):
    """The whole run through the run driver (Model, forward, end-of-run pickups): every %MON record identical to
    the oracle STDOUT, testreport digits >= the yardstick vs results/ and 16 vs the oracle (the two other variants:
    test_r3_advection_tier1.py); the pickup files byte for byte."""
    e, i = v.split("/")
    m, res, o = ag.whole_run(e, i, "tier1x")
    diffs, rows, nblocks = ag.run_verdict(e, i, res, o)
    assert not any(diffs.values()), diffs
    assert nblocks == sum("%MON time_tsnumber" in r for r in o.raw) > 1 and rows
    for name, ours, yard, vs_oracle in rows:
        assert ours >= yard, (name, ours, yard)
        assert vs_oracle >= 16, (name, vs_oracle)
    same, extra = ag.pickup_diffs(m, o)             # end-of-run pickups byte for byte (WRITE_PICKUP, GAD_WRITE_PICKUP)
    assert same and all(same.values()) and not extra, (same, extra)
    if v == "advect_xy/input.ab3_c4":
        # negative control: WRITE_PICKUP (AB3 branch) of the final State with the gtNm slots exchanged (m1 <-> m2,
        # write_pickup.F:135-136) -> the pickup data differ from the oracle's; the unchanged State gives the same bytes
        import filecmp
        from mitjax.model.src.write_pickup import write_pickup
        from mitjax.pkg.mdsio.mdsio_write_field import MdsContext
        st = res.carry[0]
        for tag, state, want in (("same", st, True), ("swapped", st.replace(gtNm=(st.gtNm[1], st.gtNm[0])), False)):
            d = ag.out_dir(f"nc-pickup-{tag}")
            d.mkdir(parents=True)
            mds = MdsContext(d, m.cfg.size, exch2=False, useSingleCpuIO=m.io.useSingleCpuIO,
                             mdsioLocalDir=m.io.mdsioLocalDir, the_run_name=m.io.the_run_name)
            write_pickup(False, "ckptA", float(res.myTime), int(res.myIter), cfg=m.cfg, params=m.params,
                         ip=m.prm.init, io=m.io, state=state, mds=mds)
            ok = all(filecmp.cmp(d / f, o.stdout_path.parent / f, shallow=False)
                     for f in ("pickup.ckptA.001.001.data", "pickup.ckptA.001.002.data"))
            assert ok == want, tag


def _modulate(m, x0):
    """The gradient experiment's initial field (a changed experiment, never a changed tolerance): the anomaly of x0
    above its minimum times the smooth, asymmetric factor 1 + 0.1 xC/max(xC) + 0.07 yC/max(yC) + 0.05 rC/min(rC).
    The verification experiments centre their blobs on cell faces (advect_xy at 40 km, advect_xz at xf(5), zf(6)), so
    neighbouring cells hold exactly equal values and the limiters' ratio tests sit on their branch boundaries (AD
    takes the measure-zero branch, FD the other one); the factor breaks those ties without making the field rough.
    The halos are then exchanged (EXCH_XYZ_RL), as in the run's own initial state."""
    g = m.grid
    xC, yC = np.asarray(g.xC.data)[:, None], np.asarray(g.yC.data)[:, None]
    rC = np.asarray(g.rC.data)[None, :, None, None]
    fac = 1. + 0.1*xC/np.max(np.abs(xC)) + 0.07*yC/max(np.max(np.abs(yC)), 1.) + 0.05*rC/np.max(np.abs(rC))
    from mitjax.eesupp.exch_rs import EXCH_XYZ_RL
    x = np.asarray(x0)
    lo = np.min(x)
    f = m.state0.theta
    return EXCH_XYZ_RL(FArray(jnp.asarray(lo + (x - lo)*fac), f.name, _dims=f.dims), ex=m.ex).data   # halos as
    # the run's own: the exchanged interior


def _grad_case(v, nsteps=2, smooth_salt=False):
    """J(theta0, salt0) = sum over the tile interiors of (w_T theta + w_S salt) after `nsteps` steps (fixed random
    weights) on the modulated initial fields (_modulate), its gradient (jax.grad of the jitted step loop) and J itself
    (jitted)."""
    m = _m(v)
    fn = m.step_fn(())
    st0 = m.state0
    sz = m.cfg.size
    rng = np.random.default_rng(0)
    wT = jnp.asarray(rng.random(st0.theta.data.shape))
    wS = jnp.asarray(rng.random(st0.salt.data.shape))
    th0, sa0 = st0.theta.data, st0.salt.data
    if smooth_salt:
        # the planted top-hat salt of advect_xy (ini_salt.F: sRef + 1 inside 60 km) has no smooth point: every point
        # is flat (DST3FL's ratio guard) or on a jump (limiter clipping); the FD check runs on the smooth theta blob
        # moved into salt (theta + sRef), a changed experiment, not a changed tolerance
        sa0 = th0 + 35.
    th0, sa0 = _modulate(m, th0), _modulate(m, sa0)
    ij = (..., slice(sz.OLy, sz.OLy + sz.sNy), slice(sz.OLx, sz.OLx + sz.sNx))

    def J(th, sa):
        state = st0.replace(theta=FArray(th, st0.theta.name, _dims=st0.theta.dims),
                            salt=FArray(sa, st0.salt.name, _dims=st0.salt.dims))
        ff, phi0, t, it = m.ff, m.phi0surf, m.prm.time.startTime, m.prm.time.nIter0
        for k in range(nsteps):
            state, ff, phi0, t, it, _, _ = fn(m.grid, m.params, m.eos, m.cg2dh, m.cg2d_params, m.ex, state, ff, phi0,
                                              jnp.int32(k + 1), jnp.float64(t), jnp.int32(it))
        # anomalies about the initial minima (constants: the gradient is unchanged; the FD roundoff floor drops
        # from ~1e-6 to ~1e-8 relative at h = 1e-6, the salt of 35 no longer sets it)
        return (jnp.sum(((state.theta.data - th_lo)*wT)[ij])
                + jnp.sum(((state.salt.data - sa_lo)*wS)[ij]))
    th_lo, sa_lo = float(jnp.min(th0)), float(jnp.min(sa0))
    gT, gS = jax.jit(jax.grad(J, argnums=(0, 1)))(th0, sa0)
    return m, jax.jit(J), th0, sa0, np.asarray(gT), np.asarray(gS), ij


# advect_xz: one step (jax.grad of two unrolled PPM-WENO / PQM + SOM-81 steps at Nr = 20 takes 838 s to compile,
# one step 226 s; dev jobs 27832224, 27832415); advect_xy: two steps (5-25 s)
GRAD_STEPS = {"advect_xy/input": 2, "advect_xy/input.ab3_c4": 2, "advect_xz/input": 1, "advect_xz/input.pqm": 1}
FD_STEPS = (1e-4, 1e-5, 1e-6)


@pytest.mark.parametrize("v", VARIANTS)
def test_gradient_finite_and_fd(v):
    """Gradient of J w.r.t. the initial theta and salt: finite on every lane (halos, land, the multi-wrap halo of
    advect_xz), nonzero in the interiors; central FD at the blob-core point (anomaly >= 10 % of its range) of largest
    |dJ/dx| and at two random blob-core points of each field: best relative error over h = 1e-4, 1e-5, 1e-6 <= 1e-6.
    The threshold 1e-6 is the first session's, stated before any measurement and kept; the h range moved down from
    1e-2..1e-4 (second session, stated before the second measurement): in blobs two cells wide the limiter thresholds
    lie within 1e-4 of the core values (a DST3FL kernel probe on a smooth row: one-sided FDs differ from each other at
    h = 1e-4 near the blob's flanks), and J is taken on anomalies so that roundoff stays below 1e-7 at h = 1e-6. The
    forward has no iterative solver (momStepping = .FALSE.). The limiters (DST3FL, SOM 81, PPM-WENO, PQM-mono) are differentiated as written, so FD is meaningful
    only where they do not switch within +-h: (i) the experiment is changed (_modulate; advect_xy's top-hat salt
    replaced by the smooth blob), (ii) the points lie in the blob core: in the far field the anomaly is 1e-14..1e-29,
    every h moves the limiter ratios across their thresholds (first session's failure: dev jobs 27832281, 27832413,
    27832414; there FD converges to AD only at h = 1e-6, if at all)."""
    m, Jj, th0, sa0, gT, gS, ij = _grad_case(v, nsteps=GRAD_STEPS[v], smooth_salt=(v == "advect_xy/input"))
    assert np.isfinite(gT).all() and np.isfinite(gS).all()
    assert np.count_nonzero(gT[ij]) > 0 and np.count_nonzero(gS[ij]) > 0
    rng = np.random.default_rng(1)
    worst = {}
    for name, g, x0, idx in (("theta", gT, th0, 0), ("salt", gS, sa0, 1)):
        x = np.asarray(x0)
        inner = np.zeros(g.shape, bool)
        inner[ij] = True
        core = inner & (x - np.min(x[ij]) >= 0.1*(np.max(x[ij]) - np.min(x[ij]))) & (np.abs(g) > 0)
        cand = np.argwhere(core)
        pts = [tuple(cand[np.argmax(np.abs(g[tuple(cand.T)]))])]
        pts += [tuple(cand[i]) for i in rng.choice(len(cand), size=2, replace=False)]
        for p in pts:
            best = np.inf
            for h in FD_STEPS:
                e = np.zeros(g.shape)
                e[p] = h
                a, b = ((th0 + e, sa0), (th0 - e, sa0)) if idx == 0 else ((th0, sa0 + e), (th0, sa0 - e))
                fd = (float(Jj(*a)) - float(Jj(*b))) / (2*h)
                best = min(best, abs(fd - g[p]) / abs(g[p]))
            worst[(name, p)] = best
    assert max(worst.values()) <= 1e-6, worst


def test_p2_equals_p1_advect_xy():
    """advect_xy/input at P=2 (shard_map(check_vma=True) on 2 of the 4 fake CPU devices, one 20x10 tile per device,
    ShardedExchanger incl. the SOM moments' exchanges of GAD_SOM_EXCHANGES) == the single-device run, bit for bit, on
    every leaf of the State (som_T, som_S included), FFIELDS and phi0surf after every step of the whole run (80)."""
    from mitjax.eesupp import exch_maps as EM
    from mitjax.eesupp.shard import TileSharding
    from mitjax.tests import r2_gate as r2
    m = _m("advect_xy/input")
    sh = TileSharding(EM.load_maps(m.exp), 2)
    assert sh.layout.nTiles == 2 and sh.blocks.Tloc == 1
    step2, (s2, f2, p2) = ag.sharded_step_fn(m, sh)
    fn = m.step_fn(())
    s1, f1, p1, t1, it1 = m.state0, m.ff, m.phi0surf, m.prm.time.startTime, m.prm.time.nIter0
    t2, it2 = t1, it1
    for k in range(m.prm.time.nTimeSteps):
        s1, f1, p1, t1, it1, out1, _ = fn(m.grid, m.params, m.eos, m.cg2dh, m.cg2d_params, m.ex, s1, f1, p1,
                                          jnp.int32(k + 1), jnp.float64(t1), jnp.int32(it1))
        s2, f2, p2, t2, it2, out2 = step2(k, s2, f2, p2, t2, it2)
        d = r2.tree_bits_differ((s1, f1, p1), sh.unpad_tree((s2, f2, p2)))
        assert not d and float(t1) == float(t2) and int(it1) == int(it2), (k, d)
    assert m.prm.time.nTimeSteps == 80
