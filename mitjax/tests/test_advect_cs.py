"""advect_cs on the cubed sphere (plan Task 25 part, ADVECT lane): GAD_ADVECTION's cube passes (DST3FL 33 for theta:
three passes per level with FILL_CS_CORNER_TR_RL / _UV_RS and the overlapOnly / interiorOnly update ranges),
GAD_SOM_ADVECT's cube passes (SOM 80 for salt: GAD_SOM_PREP_CS_CORNER, GAD_SOM_FILL_CS_CORNER, the strips of
GAD_SOM_ADV_X/_Y), the experiment's own ini_vel.F, on 6 tiles of 32x32 (W2 cube topology, lane B's exchanges).
Helpers: mitjax/tests/advect_cs_gate.py.

* initial state and every dumped field of every stage of steps 1-3 bitwise (element equality, bit patterns, finite;
  all points incl. halos and corner halos) vs lane A's dumps-on run (job 27832226);
* negative controls measured to bite: a corner fill skipped, a pass order swapped;
* P=6 (one facet per device, subprocess with 6 fake devices) == P=1 bitwise;
* the whole run: every %MON record identical, testreport digits >= the yardstick vs results/ and 16 vs the oracle.
The SOM tendency is not dumped where SOM replaces GAD_ADVECTION (T20): gated through T21-T23, T03, S15, S16."""

import os
import subprocess
import sys

import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

import jax.numpy as jnp  # noqa: E402

from mitjax import paths  # noqa: E402
from mitjax.tests import advect_cs_gate as cg  # noqa: E402
from mitjax.tests import r1_gate as rg  # noqa: E402

EXPECT = ("T10_temp_adv", "T11_temp_gT", "T13_temp_impl", "T02_temp_integrate", "T21_salt_gS", "T23_salt_impl",
          "T03_salt_integrate", "S15_tracers_correction", "S16_blocking_exchanges")


def test_initial_state_bitwise():
    """INITIALISE_VARIA (with advect_cs's own INI_VEL: the solid-body rotation from fCoriG, and GAD_INIT_VARIA) vs
    S00_begin: every State field the oracle dumps, bit patterns, all points."""
    cm = cg.model()
    r = rg.compare_stage(cm.ds, cm.it0, "S00_begin", cm.m.state0)
    assert len(r) >= 10 and not cg.bad(r), cg.bad(r)
    assert "uVel" in r and "vVel" in r and "theta" in r and "salt" in r


def test_substeps_steps_1_to_3_bitwise():
    cm = cg.model()
    for st in EXPECT:
        assert st in cm.stages, (st, cm.stages)
    res, _ = cg.run_steps(cm, 3)
    for k, per_stage in enumerate(res):
        for st, r in per_stage.items():
            assert r, (k, st, "no fields compared")
            assert not cg.bad(r), (k, st, cg.bad(r))
            assert all(x[2] == 0 for x in r.values()), (k, st, "non-finite")


def _nd(cm, n, stages):
    res, _ = cg.run_steps(cm, n, stages=stages)
    return [{s: sum(v[3] for v in r[s].values()) for s in r} for r in res]


def test_negative_control_corner_fill_skipped(monkeypatch):
    """GAD_ADVECTION's corner fill before the X flux of the overlap-only pass (gad_advection.F:392-395) skipped ->
    T10 differs at step 1 (on the facets that make the call, 3 and 6)."""
    import mitjax.pkg.generic_advdiff.gad_advection as ga
    orig = ga._cs_fill

    def planted(fill4dir, A, flag, cs, kc, _n=[0]):
        _n[0] += 1
        if fill4dir == 1 and _n[0] % 4 == 1:      # the first fill call of each pass sequence: the (1) before X
            return A
        return orig(fill4dir, A, flag, cs, kc)
    monkeypatch.setattr(ga, "_cs_fill", planted)
    r = _nd(cg.model(), 1, ("T10_temp_adv",))
    assert r[0]["T10_temp_adv"] > 0, r


def test_negative_control_pass_order_swapped(monkeypatch):
    """The first two cube passes exchanged (pass 2's flags in pass 1 and vice versa, gad_advection.F:351-360) ->
    T10 differs at step 1."""
    import mitjax.pkg.generic_advdiff.gad_advection as ga
    orig = ga._cs_pass_flags
    monkeypatch.setattr(ga, "_cs_pass_flags", lambda nCFace, ipass: orig(nCFace, {1: 2, 2: 1}.get(ipass, ipass)))
    r = _nd(cg.model(), 1, ("T10_temp_adv",))
    assert r[0]["T10_temp_adv"] > 0, r


def test_negative_control_som_corner_store_skipped(monkeypatch):
    """GAD_SOM_PREP_CS_CORNER without its stored corner values (pass 2 of facets 3 and 6 reads zeros instead,
    gad_som_prep_cs_corner.F:175-222) -> the SOM tendency (T21) differs at step 1."""
    import mitjax.pkg.generic_advdiff.gad_som_prep_cs_corner as gp
    orig = gp.gad_som_prep_cs_corner

    def planted(smVol, smTr0, smTr, smCorners, *a, **kw):
        out = orig(smVol, smTr0, smTr, smCorners, *a, **kw)
        return out[0], out[1], out[2], {c: jnp.zeros_like(v) for c, v in out[3].items()}
    monkeypatch.setattr(gp, "gad_som_prep_cs_corner", planted)
    r = _nd(cg.model(), 1, ("T21_salt_gS",))
    assert r[0]["T21_salt_gS"] > 0, r


def test_p6_equals_p1():
    """P=6 (one facet per device; ShardedExchanger of the cube maps under shard_map(check_vma=True)) == P=1 bitwise on
    every leaf of the State (SOM moments included), FFIELDS and phi0surf after each of 4 steps; a subprocess with 6
    fake CPU devices (the test session has 4, mitjax/xla_flags.py)."""
    env = dict(os.environ)
    env.pop("XLA_FLAGS", None)
    env["PYTHONPATH"] = str(paths.REPO)
    r = subprocess.run([sys.executable, "-c", cg.P6_SCRIPT, "4"], env=env, capture_output=True, text=True,
                       timeout=1500)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "P6 OK 4" in r.stdout, r.stdout


def test_whole_run_cli_monitor_digits_pickups():
    """`python -m mitjax run` on advect_cs end to end (mitjax/drivers/run.py `run`: its make_rundir runs the
    experiment's prepare_run through lane A's make_rundir; 192 steps): every %MON record and banner of output.txt
    identical to the oracle STDOUT; tools/testreport_jax.py digits >= the yardstick vs results/ and 16 vs the oracle;
    the end-of-run pickups (pickup.ckptA.*, pickup_somS.ckptA.*: 6 tiles each, data and meta; WRITE_PICKUP and
    GAD_WRITE_PICKUP through MDS_WRITE_FIELD's exch2 I/O layout) byte for byte equal to the oracle's."""
    from pathlib import Path
    from types import SimpleNamespace
    from mitjax.drivers.run import run
    from mitjax.tests import advect_gate as ag
    from mitjax.tests import monitor_gate as mg
    out = Path(run(paths.UPSTREAM / "verification" / "advect_cs", "input", cg.out_dir("cli")))
    o = mg.oracle("advect_cs", "input")
    res = SimpleNamespace(records=out.read_text().rstrip("\n").split("\n"))
    diffs, rows, nblocks = ag.run_verdict("advect_cs", "input", res, o)
    assert not any(diffs.values()), diffs
    assert nblocks == sum("%MON time_tsnumber" in r for r in o.raw) > 1 and rows
    for name, ours, yard, vs_oracle in rows:
        assert ours >= yard, (name, ours, yard)
        assert vs_oracle >= 16, (name, vs_oracle)
    same, extra = ag.pickup_diffs(SimpleNamespace(rundir=out.parent), o)
    assert len(same) == 24 and all(same.values()) and not extra, (same, extra)


def test_cs_pass_flags_and_fills_equal_lane_b_schedule():
    """GAD_ADVECTION's traced per-tile pass flags and fill / flux / update decisions (gad_advection._cs_pass_flags and
    the conditions of _cs_passes) == lane B's host schedule gad_cs_passes.gad_cs_pass_schedule (gated against a
    gfortran build of the Fortran statements, test_cube.py::test_gad_cs_pass_schedule) for every facet, every
    combination of the four edge flags and every pass."""
    import itertools
    from mitjax.pkg.generic_advdiff.gad_advection import _cs_pass_flags
    from mitjax.pkg.generic_advdiff.gad_cs_passes import gad_cs_pass_schedule
    n = 0
    for face in range(1, 7):
        for N, S, E, W in itertools.product((False, True), repeat=4):
            sched = gad_cs_pass_schedule(face, N, S, E, W)
            for ipass in (1, 2, 3):
                ov, intr, cX, cY = (bool(x[0, 0, 0]) for x in _cs_pass_flags(jnp.full((1, 1, 1), face), ipass))
                want = sched[ipass-1]
                assert (ov, intr, cX, cY) == (want["overlapOnly"], want["interiorOnly"], want["calc_fluxes_X"],
                                              want["calc_fluxes_Y"])
                ev = []
                if cX:
                    do = (not ov) or N or S
                    ev += [("fill", 1)] if do and ov else []
                    ev += [("flux", "X")] if do else []
                    ev += [("fill", 2)] if do and ov and ipass == 1 else []
                    ev += [("update", "X")]
                if cY:
                    do = (not ov) or E or W
                    ev += [("fill", 2)] if do and ov else []
                    ev += [("flux", "Y")] if do else []
                    ev += [("fill", 1)] if do and ov and ipass == 1 else []
                    ev += [("update", "Y")]
                assert tuple(ev) == want["events"], (face, N, S, E, W, ipass)
                n += 1
    assert n == 6 * 16 * 3


def _cs_grad_case(nsteps):
    """J(theta0, salt0) = sum over the tile interiors of w_T (theta - min theta0) + w_S (salt - min salt0) after
    `nsteps` steps of advect_cs (fixed random weights), its gradient and J (both jitted)."""
    import jax
    from mitjax.farray import FArray
    cm = cg.model()
    m = cm.m
    fn = cm.step_fn(())
    c0 = m.initial_carry()
    st0 = c0[0]
    sz = m.cfg.size
    rng = np.random.default_rng(0)
    wT = jnp.asarray(rng.random(st0.theta.data.shape))
    wS = jnp.asarray(rng.random(st0.salt.data.shape))
    from mitjax.tests.test_r3_advection import _modulate
    # the R3 gradient experiment: the anomalies times a smooth asymmetric factor (no exact ties between neighbours,
    # which put the DST3FL limiter on a branch boundary: AD takes the measure-zero branch, FD the other one)
    th0, sa0 = _modulate(m, st0.theta.data), _modulate(m, st0.salt.data)
    th_lo, sa_lo = float(jnp.min(th0)), float(jnp.min(sa0))
    ij = (..., slice(sz.OLy, sz.OLy + sz.sNy), slice(sz.OLx, sz.OLx + sz.sNx))

    def J(th, sa):
        carry = (st0.replace(theta=FArray(th, st0.theta.name, _dims=st0.theta.dims),
                             salt=FArray(sa, st0.salt.name, _dims=st0.salt.dims)),) + tuple(c0[1:])
        t, it = m.start_counters()
        for k in range(nsteps):
            carry, t, it, _, _ = fn(m.arrays, carry, jnp.int32(k + 1), t, it)
        s = carry[0]
        return jnp.sum(((s.theta.data - th_lo)*wT)[ij]) + jnp.sum(((s.salt.data - sa_lo)*wS)[ij])
    g = jax.jit(jax.grad(J, argnums=(0, 1)))(th0, sa0)
    return J, jax.jit(J), th0, sa0, [np.asarray(x) for x in g], ij


@pytest.mark.parametrize("nsteps", [1])
def test_gradient_cube_finite_fd_dot(nsteps):
    """Gradient of J after one step of advect_cs (GAD_ADVECTION's and GAD_SOM_ADVECT's cube passes, corner fills,
    cube exchanges) w.r.t. the initial theta and salt; two steps cost 600 s (mostly the compile of jax.grad) and gave
    the same verdicts on the unmodulated fields (dev job 27834348), so tier 1x runs one. Thresholds stated before the measurement: (a) finite on every
    lane incl. halos and corner halos, nonzero in the interiors; (b) dot test: jvp(J)(u) == <grad J, u> for a random
    direction u, relative difference <= 1e-12; (c) central FD at the blob-core point (anomaly >= 10 % of its range) of
    largest |dJ/dx| and at two random blob-core points of each field: best relative error over h = 1e-4, 1e-5, 1e-6
    <= 1e-6 (the R3 protocol, test_r3_advection.py::test_gradient_finite_and_fd), on the R3 gradient experiment
    (test_r3_advection._modulate). Measured first on the unmodulated T.init / S.init (27834348): (a), (b) passed at 1
    and 2 steps; (c) SOM salt 1e-13..1e-11, DST3FL theta 0.25 / 0.095 at two of three core points (ties of the
    symmetric blob)."""
    import jax
    J, Jj, th0, sa0, (gT, gS), ij = _cs_grad_case(nsteps)
    assert np.isfinite(gT).all() and np.isfinite(gS).all()
    assert np.count_nonzero(gT[ij]) > 0 and np.count_nonzero(gS[ij]) > 0
    rng = np.random.default_rng(3)
    uT, uS = jnp.asarray(rng.standard_normal(gT.shape)), jnp.asarray(rng.standard_normal(gS.shape))
    _, jv = jax.jit(lambda a, b, c, d: jax.jvp(J, (a, b), (c, d)))(th0, sa0, uT, uS)
    dot = float(np.sum(gT*np.asarray(uT)) + np.sum(gS*np.asarray(uS)))
    assert abs(float(jv) - dot) <= 1e-12*abs(dot), (float(jv), dot)
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
            for h in (1e-4, 1e-5, 1e-6):
                e = np.zeros(g.shape)
                e[p] = h
                a, b = ((th0 + e, sa0), (th0 - e, sa0)) if idx == 0 else ((th0, sa0 + e), (th0, sa0 - e))
                fd = (float(Jj(*a)) - float(Jj(*b))) / (2*h)
                best = min(best, abs(fd - g[p]) / abs(g[p]))
            worst[(name, p)] = best
    assert max(worst.values()) <= 1e-6, worst
