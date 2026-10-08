"""offline_exf_seaice/input.thermo end to end (M4 step 4, lane M4OFF session 2): the 120-step run through the run
driver, the restart, P=N == P=1 on the 4 tiles of 40x21, and the gradient of a whole-step window.

Oracle: lane A's registered plain run job27855986-plain (STDOUT, pickup.ckptA, pickup_seaice.ckptA) and dumps-on run
job27855986-jdon (the teacher-forced carry of iteration 1). Helpers: mitjax/tests/m4off_gate.py.

1. The whole run (drivers.run.forward, as `python -m mitjax run`): every MONITOR record (%MON lines and banners of the
   dynamics block at 0 and 120, the SEAICE block every SEAICE_monFreq = 36000 s, its own schedule, and the EXF block
   at 0) identical to the oracle's, the testreport digits at the yardstick, pickups byte-identical, no STDERR.
2. Restart: 20 steps with permanent pickups every 10 (A) vs 10 + pickup + 10 (B): B's pickups at 20 byte-identical to
   A's, B's STDOUT from the SEAICE block at 20 on identical; the carries differ only where the Fortran's restart code
   makes them differ (measured, docstring of the test). Negative controls: a zeroed siHEFF record, a missing one.
3. P=4 (one tile per device) and P=2 (two tiles per device) == P=1, every leaf of the carry bitwise after 3 steps.
4. Gradients of J = weighted sums of theta and HEFF after 2 whole steps from the teacher-forced carry of iteration 1,
   w.r.t. theta: finite on every lane, central FD at an interior point and along a field-wide consistent direction,
   dot test with random directions on every lane, the sharded gradient P=4 == P=1, the FD control.
"""

import filecmp
from pathlib import Path

import numpy as np
import pytest

from mitjax.tests import m4off_gate as G


@pytest.fixture(autouse=True)
def _release_executables():
    """Free the previous test's compiled programs (PORTING_LESSONS "vm.max_map_count"); changes no value."""
    import gc
    import jax
    jax.clear_caches()
    gc.collect()
    yield


def test_whole_run_monitor_digits_pickups():
    """The 120-step run through the run driver: every MONITOR record identical to the oracle's STDOUT (567 records:
    the dynamics block at iterations 0 and 120, the SEAICE block at 0, 10, ..., 120 (13 blocks, SEAICE_monFreq 36000
    s, C-grid UICE/VICE statistics), the EXF block at 0 (exf_monFreq 86400000 s)), the 17 testreport digits at the
    yardstick (vs results/) and vs the oracle, pickup.ckptA and pickup_seaice.ckptA byte-identical, nothing on
    STDERR (measured: dev job 27868972, 113 s)."""
    m, res, o, (diffs, rows, nblocks), (files, missing) = G.whole_run()
    print("offline_exf_seaice/input.thermo digits (name, ours vs results/, yardstick, ours vs oracle):", rows)
    assert nblocks == 2 and diffs == {"mon": 0, "banner": 0}, (nblocks, diffs)
    k0 = next(n for n, r in enumerate(o.raw) if "Begin MONITOR dynamic field statistics" in r) - 1
    a, b = G.monitor_records(res.records), G.monitor_records(o.raw[k0:])
    assert a == b and len(a) == 567, (len(a), len(b), [(x, y) for x, y in zip(a, b) if x != y][:2])
    for pre, n_blocks in (("%MON seaice_tsnumber", 13), ("%MON exf_tsnumber", 1), ("%MON time_tsnumber", 2)):
        assert sum(pre in r for r in a) == n_blocks, pre
    assert len(rows) == 17 and all(x >= y for _, x, y, _ in rows), rows
    assert all(s >= y for _, _, y, s in rows), rows
    pk = {f: ok for f, ok in files.items() if f.startswith("pickup")}
    assert sorted(pk) == ["pickup.ckptA.data", "pickup.ckptA.meta", "pickup_seaice.ckptA.data",
                          "pickup_seaice.ckptA.meta"], pk
    assert all(pk.values()), pk
    assert res.stderr == [], res.stderr[:5]


def test_whole_run_negative_control_seaice_monitor_schedule(monkeypatch):
    """The SEAICE monitor schedule: without the driver's chunk ends at SEAICE_monFreq (seaice_monitor_steps -> no
    steps) the SEAICE blocks at 10, ..., 110 are missing, so the MONITOR records differ (counted)."""
    from mitjax.drivers import run as R
    monkeypatch.setattr(R, "seaice_monitor_steps", lambda m, n: set())
    m, res, o, (diffs, rows, nblocks), _ = G.whole_run("m4off-whole-neg")
    a = G.monitor_records(res.records)
    assert sum("%MON seaice_tsnumber" in r for r in a) == 2, sum("%MON seaice_tsnumber" in r for r in a)


def test_restart_10_pickup_10_equals_20():
    """Run A: 20 steps with pChkptFreq = 10 steps (pickups at 10 and 20, ocean and sea ice); run B: nIter0 = 10
    (startTime = 10*deltaT, nTimeSteps = 10) from A's pickup.0000000010 and pickup_seaice.0000000010
    (READ_PICKUP, SEAICE_READ_PICKUP of the C-grid build). B's pickups at 20 are byte-identical to A's, B's STDOUT
    records from the SEAICE block at 20 on equal A's, and the final carries are bitwise equal except where the
    Fortran's own restart code makes them differ (measured, dev job 27869049): ETA, ZETA, PRESS0, SEAICE_zMax
    (SEAICE_INIT_VARIA :437-443 set them from the HEFF it starts from: A's cold-start file, B's restart; nothing else
    writes them without SEAICEuseDYNAMICS) and TICES levels 2..nITD of the interior (SEAICE_READ_PICKUP :302-315 copies
    level 1 there; A keeps INIT_VARIA's 273; SEAICE_multDim = 1 computes level 1 only) with TICES halos (B's exchange
    :321). None of them reaches a pickup record or a printed value."""
    a, ra, b, rb = G.restart_b("restartB")
    assert ra.pickups == ["pickup.0000000010", "pickup.0000000020"] and rb.pickups == ["pickup.0000000020"]
    for pre in ("pickup", "pickup_seaice"):
        for suf in ("data", "meta"):
            f = f"{pre}.0000000020.{suf}"
            assert filecmp.cmp(Path(a.rundir) / f, Path(b.rundir) / f, shallow=False), f

    def tail(recs):
        k = next(n for n, r in enumerate(recs) if "%MON seaice_tsnumber" in r and r.split("=")[1].strip() == "20")
        return [r for r in recs[k:] if "%CHECKPOINT" not in r]
    ta, tb = tail(ra.records), tail(rb.records)
    assert ta == tb and len(ta) == 30, (len(ta), len(tb), [(x, y) for x, y in zip(ta, tb) if x != y][:2])
    from mitjax.tests import m4col_gate as C
    d = C.carry_diffs(ra.carry, rb.carry, OL=3)
    print("restart: carry differences (points, interior points):", d)
    assert set(d) == {"seaice.ETA", "seaice.ZETA", "seaice.PRESS0", "seaice.SEAICE_zMax", "seaice.TICES"}, d
    ta_, tb_ = np.asarray(ra.carry[4]["seaice"]["TICES"].data), np.asarray(rb.carry[4]["seaice"]["TICES"].data)
    assert np.array_equal(ta_[:, 0, 3:-3, 3:-3].view(np.int64), tb_[:, 0, 3:-3, 3:-3].view(np.int64))
    assert np.all(ta_[:, 1:, 3:-3, 3:-3] == 273.0)
    assert np.all(tb_[:, 1:, 3:-3, 3:-3] == tb_[:, 1:2, 3:-3, 3:-3])


def test_restart_negative_controls():
    """B's pickup_seaice with the siHEFF record zeroed: the run differs (HEFF, and through the sea ice theta); with
    siHEFF missing from the field list: SEAICE_CHECK_PICKUP stops (cannot restart without field)."""
    def zero_heff(d):
        """siHEFF is the 3rd of the 6 records (fldList 'siTICE' 'siAREA' 'siHEFF' 'siHSNOW' 'siUICE' 'siVICE', one
        global 2-D record each: SEAICE_multDim = 1, no SEAICE_VARIABLE_SALINITY): its bytes set to 0."""
        meta = (d / "pickup_seaice.0000000010.meta").read_text()
        names = meta[meta.index("fldList"):].split("{")[1].split("}")[0].split("'")[1::2]
        p = d / "pickup_seaice.0000000010.data"
        raw = bytearray(p.read_bytes())
        size = len(raw) // len(names)
        k = names.index("siHEFF  ")
        assert k == 2 and len(names) == 6, names
        raw[k*size:(k + 1)*size] = bytes(size)
        p.write_bytes(bytes(raw))
    from mitjax.tests import m4col_gate as C
    _, ra, _, rb = G.restart_b("negZeroHeff", mutate=zero_heff)
    d = C.carry_diffs(ra.carry, rb.carry, OL=3)
    assert d.get("seaice.HEFF", (0, 0))[1] > 0 and d.get("state.theta", (0, 0))[1] > 0, d

    def drop_heff(d):
        p = d / "pickup_seaice.0000000010.meta"
        txt = p.read_text()
        assert "'siHEFF  '" in txt
        p.write_text(txt.replace("'siHEFF  '", "'siXXXX  '"))
    with pytest.raises(ValueError, match="cannot restart without field \"siHEFF  \""):
        G.restart_b("negDropHeff", mutate=drop_heff)


@pytest.mark.parametrize("nproc", [4, 2])
def test_sharded_equals_single(nproc):
    """3 steps of the driver's step (the whole carry: State, FFIELDS.h, phi0surf, flow, the sea-ice and EXF state) at
    P=nproc (jit(shard_map(check_vma=True)), ShardedExchanger on the registered 4-tile map) == P=1, every leaf
    bitwise (measured: dev job 27869049, both)."""
    assert G.sharded_vs_single(G.run(), nproc, 3) == {}


# ----------------------------------------------------------------------------------------------- 4. gradients
def _window():
    import jax
    m, ds = G.model()
    J = G.window_cost(m, G.window(m))
    carry0 = G.teacher_carry(m, ds, 1)
    return m, J, carry0, G.cost_weights(m), jax.jit(J)


def _fd_errors(Jj, x0, d, t, args, hs=(1e-2, 1e-3, 1e-4, 1e-5, 1e-6)):
    return [abs((float(Jj(x0 + h*d, *args)) - float(Jj(x0 - h*d, *args))) / (2*h) - t) / abs(t) for h in hs]


def _directions(m, th0):
    """The interior point G.PT of level 1 (10 points from every tile edge: no halo copy) and the field-wide direction
    maskC on every lane (interior and halo copies alike: a consistent model-state perturbation)."""
    d1 = np.zeros(th0.shape)
    d1[G.PT[0], 0, G.PT[1], G.PT[2]] = 1.0
    return {"point": d1, "field": np.asarray(m.grid.maskC.data) * np.ones(th0.shape)}


def test_gradient_whole_steps_window_finite_and_fd():
    """dJ/d(theta at the start of iteration 1), J = G.window_cost after 2 whole steps (EXF, the C-grid sea ice, the
    surface forcing with restoring, the implicit vertical diffusion of theta): finite on every lane, nonzero; the best
    central FD along the point and the field directions agrees with the gradient to < 1e-7 (measured: 1.6e-10 at
    h = 1e-3 and 9.6e-11 at h = 1e-5; dev job 27869055)."""
    import jax
    m, J, carry0, w, Jj = _window()
    th0 = carry0[0].theta.data
    gr = np.asarray(jax.jit(jax.grad(J))(th0, m.arrays, carry0, w))
    assert np.all(np.isfinite(gr)) and np.count_nonzero(gr) > 1000, np.count_nonzero(gr)
    for name, d in _directions(m, th0).items():
        t = float(np.sum(gr * d))
        errs = _fd_errors(Jj, th0, d, t, (m.arrays, carry0, w))
        print(name, "tangent", t, "FD rel. errors", errs)
        assert min(errs) < 1e-7, (name, t, errs)


def test_gradient_whole_steps_fd_negative_control(monkeypatch):
    """Negative control of the FD gate: every Fortran MAX/MIN derivative times (1 + 1e-4), values unchanged
    (test_m4col_column._planted_minmax_step): the best FD error along the point direction rises above 1e-7."""
    import jax
    from mitjax.ops import fortran_minmax as FM
    from mitjax.tests.test_m4col_column import _planted_minmax_step
    for key in list(FM._STEPS):
        monkeypatch.setitem(FM._STEPS, key, _planted_minmax_step(*key, 1.0 + 1e-4))
    m, J, carry0, w, Jj = _window()
    th0 = carry0[0].theta.data
    gr = np.asarray(jax.jit(jax.grad(J))(th0, m.arrays, carry0, w))
    d = _directions(m, th0)["point"]
    t = float(np.sum(gr * d))
    errs = _fd_errors(Jj, th0, d, t, (m.arrays, carry0, w))
    print("planted MAX/MIN derivatives x (1 + 1e-4): tangent", t, "FD rel. errors", errs)
    assert min(errs) > 1e-7, (t, errs)


def test_gradient_whole_steps_window_dot_test():
    """Tangent (jvp) vs adjoint (vjp) dot test of the 2-step window theta -> (theta, HEFF), random directions on
    every lane of theta and of both outputs, bar 1e-12 relative (measured 2.1e-15: no CG2D (momStepping .FALSE.) and
    no tracer advection in this run, so none of the cancelling sums of the column's window, lane DOTDIAG)."""
    import jax
    import jax.numpy as jnp
    m, ds = G.model()
    carry0 = G.teacher_carry(m, ds, 1)
    f = G.window(m)
    g = jax.jit(lambda x: f(x, m.arrays, carry0))
    x0 = carry0[0].theta.data
    y0, vjp = jax.vjp(g, x0)
    w = tuple(jax.random.normal(jax.random.PRNGKey(i), y.shape, y.dtype) for i, y in enumerate(y0))
    (xbar,) = vjp(w)
    v = jax.random.normal(jax.random.PRNGKey(7), x0.shape, x0.dtype)
    _, ydot = jax.jvp(g, (x0,), (v,))
    assert all(np.all(np.isfinite(np.asarray(t))) for t in ydot) and np.all(np.isfinite(np.asarray(xbar)))
    lhs = sum(float(jnp.vdot(a, b)) for a, b in zip(ydot, w))
    rhs = float(jnp.vdot(v, xbar))
    print("dot test", lhs, rhs, abs(lhs - rhs) / max(abs(lhs), abs(rhs)))
    assert abs(lhs - rhs) <= 1e-12 * max(abs(lhs), abs(rhs), 1.0), (lhs, rhs)


def test_sharded_gradient_p4_equals_p1():
    """dJ/dtheta of the 2-step window at P=4 (jax.value_and_grad inside jit(shard_map(check_vma=True)), the
    ShardedExchanger; J's global sums in the fixed tile order) == P=1: J and every gradient point bitwise, nothing on
    the padding (measured: 0 points differ, dev job 27869055)."""
    m, ds = G.model()
    J1, g1, J4, g4, npad = G.sharded_grad_vs_single(m, ds, 4)
    assert np.all(np.isfinite(g1)) and np.count_nonzero(g1) > 1000
    assert J4 == J1, (J4, J1)
    assert np.array_equal(g1.view(np.int64), g4.view(np.int64)), int(np.count_nonzero(g1 != g4))
    assert npad == 0, npad
