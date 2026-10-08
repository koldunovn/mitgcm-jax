"""1D_ocean_ice_column/input wiring gates (M4 step 3, lane M4COL session 3): pkg/cal, pkg/exf, pkg/seaice inside the
driver Model and FORWARD_STEP (helpers: mitjax/tests/m4col_gate.py).

1. The front of FORWARD_STEP (S00_begin .. P01_external_forcing_surf: LOAD_FIELDS_DRIVER -> EXF_GETFORCING with the
   preloaded EXF records, DO_OCEANIC_PHYS -> SEAICE_MODEL, EXTERNAL_FORCING_SURF with sIceLoad) from the driver
   Model, bitwise vs lane A's dumps-on run (job27856057-jdon) at iterations 0, 1, 2 on every point (halos
   included): iteration 0 from the Model's own initial carry (INITIALISE_VARIA + CAL/EXF/SEAICE set-up), 1 and 2
   teacher-forced (m4col_gate.teacher_carry).
2. Negative controls: the EXF record preload shifted by one step (X01 differs from iteration 0 on); KGEO level 2 in
   the Model's static package settings (I01b's GWATX/GWATY differ).
3. Monitor wiring: INITIALISE_VARIA's MONITOR + SEAICE_MONITOR (DO_THE_MODEL_IO) records of nIter0, EXF_MONITOR of
   steps 1-3 (EXF_GETFORCING's snapshot) and SEAICE_MONITOR after steps 1-3 (SEAICE.h after SEAICE_MODEL, which is
   the only writer of the fields it prints) identical to the oracle STDOUT records.
4. Gradient smoke: d/d(theta at the start of iteration 1) of a scalar of the step front (surfaceForcingT at P01 +
   HEFF after SEAICE_MODEL): finite on every lane, tangent-adjoint dot test, central FD at the thin-ice state.
Session 4 (the pkg/kpp arms of this build: KPP_SMOOTH_DBLOC, KPP_SMOOTH_SHSQ, no KPP_ESTIMATE_UREF,
SHORTWAVE_HEATING; FIND_ALPHA/FIND_BETA 'JMD95Z'; GAD DST3; SEAICE_READ_PICKUP):
5. Steps 0-2 free from the Model's initial carry and teacher-forced from the oracle's state, every dumped stage
   (P02, P07_kpp, P10_kpp_exch, D-, C- (CG2D on the dumped inputs), T-, S-stages) bitwise on every point.
6. Negative controls of the new arms, each measured to bite or reported where this column cannot see it.
7. The whole 10-step run through the run driver: %MON (dynamics, SEAICE, EXF) and banners identical, testreport
   digits, pickups (ocean, sea ice) byte-identical; the restart 5 + pickup + 5 == 10 and its controls.
8. Gradient smoke over two whole steps: finite on every lane, FD along theta(k=1) of the column (the point and its
   halo copies; lane DOTDIAG: the interior lane alone reaches the w = 0 kink of the DST3 upwind flux), a planted
   derivative error as its negative control; the all-lane dot test is a strict xfail (rounding of the CG2D derivative
   PCG and of cancelling flux-form terms, lane DOTDIAG).
Costs: about 12 min on a CPU node.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

from mitjax.tests import m4col_gate as C  # noqa: E402

# S00_begin records of fields the Model keeps elsewhere (CG2D.h, the State-independent GRID.h hFac of a build
# without NONLIN_FRSURF; compared in G00), as in test_m3_channel.NOT_CARRIED
NOT_CARRIED = {("S00_begin", n) for n in ("aC2d", "aS2d", "aW2d", "pC", "pS", "pW", "hFacC", "hFacS", "hFacW",
                                          "recip_hFacC")}
PHI0 = ("X06_exf_mapfields", "I01b_dynsolver", "I04_growth", "P13_seaice_model")   # stages that dump phi0surf


@pytest.fixture(autouse=True)
def _release_executables():
    import gc
    yield
    jax.clear_caches()
    gc.collect()


def _bad_front(res):
    from mitjax.tests import goadk_model_gate as M
    return {s: v for s, v in M.bad(res).items() if s in C.STAGES}


def test_front_of_step_bitwise_iterations_0_2():
    from mitjax.tests import goadk_model_gate as M
    from mitjax.tests import r1_gate as rg
    m, ds = C.model()
    out = C.run_front(m, ds)
    for it, res, o in out:
        assert not _bad_front(res), (it, _bad_front(res))
        assert set(C.STAGES) <= set(res), (it, sorted(set(C.STAGES) - set(res)))
        left = [(s, n) for s, n in M.not_compared(ds, it, res) if s in C.STAGES and (s, n) not in NOT_CARRIED
                and not (s in PHI0 and n == "phi0surf")]
        assert not left, (it, left)
        n = sum(len(r) for s, r in res.items() if s in C.STAGES)
        print("iteration", it, "compared (stage, field) records:", n)
        assert n == 405, (it, n)
        # phi0surf at the stages before EXTERNAL_FORCING_SURF: the carry's (written only at P01)
        carry = C.teacher_carry(m, ds, it)
        for s in PHI0:
            r = rg.compare_field(np.asarray(carry[2].data), ds.field(it, s, "phi0surf"))
            assert not any(r[1:]), (it, s, r)


def test_negative_control_exf_preload_shift():
    """The EXF records of the next step at every step (ExfPreload rows shifted by one): X01 differs at iteration 0."""
    from mitjax.pkg.exf.exf_preload import ExfPreload
    m, ds = C.model()
    pre = m.arrays.pkc["exf_pre"]
    def shift(a):                       # fac [n+1]; the record FArrays [tile, n+1, j, i] (step axis 1)
        if hasattr(a, "data"):
            return type(a)(jnp.concatenate([a.data[:, 1:], a.data[:, -1:]], axis=1), a.name, tiled=a.tiled,
                           _dims=a.dims)
        return jnp.concatenate([a[1:], a[-1:]])
    sh = ExfPreload({n: tuple(shift(a) for a in r) for n, r in pre.recs.items()}, pre.dims)
    a = m.arrays.replace(pkc=dict(m.arrays.pkc, exf_pre=sh))
    out = C.run_front(m, ds, its=(0,), arrays=a)
    b = _bad_front(out[0][1])
    assert "X01_exf_getffields" in b and "atemp" in b["X01_exf_getffields"], sorted(b)


def test_negative_control_kgeo_level():
    """KGEO level 2 (pks, static): GWATX/GWATY at I01b differ at iteration 1 (moving ocean)."""
    m, ds = C.model()
    old = m.pks
    try:
        m.pks = dict(old, exfs=dict(old["exfs"], kgeo=2))
        out = C.run_front(m, ds, its=(1,))
    finally:
        m.pks = old
    b = _bad_front(out[0][1])
    assert {"GWATX", "GWATY"} <= set(b.get("I01b_dynsolver", {})), sorted(b)


def _mon(lines, prefix):
    return [ln for ln in lines if f"%MON {prefix}" in ln]


def test_monitor_blocks_initial_exf_seaice():
    from mitjax.drivers.run import MonitorHost, exf_monitor_records, seaice_monitor_records
    m, ds = C.model()
    tp = m.prm.time
    mh = MonitorHost(m)
    carry = m.initial_carry()
    dyn = mh.monitor(float(tp.startTime), int(tp.nIter0), carry)
    si0 = seaice_monitor_records(mh, float(tp.startTime), int(tp.nIter0), carry)
    ref_dyn = C.oracle_blocks("time_")
    ref_dyn_all = {}
    from mitjax import paths
    lines = (paths.REFERENCE_RUNS / C.EXP[0] / C.EXP[1] / "job27856057-plain" / "rundir" / "output.txt"
             ).read_text().splitlines()
    k0 = next(i for i, ln in enumerate(lines) if "%MON time_tsnumber" in ln)
    k1 = next(i for i, ln in enumerate(lines) if i > k0 and "End MONITOR dynamic field statistics" in ln)
    ref_dyn_all = [ln for ln in lines[k0:k1] if "%MON" in ln]
    assert ref_dyn[0][0] == ref_dyn_all[0]
    assert _mon(dyn, "") == ref_dyn_all, [(a, b) for a, b in zip(_mon(dyn, ""), ref_dyn_all) if a != b][:3]
    ref_si, ref_exf = C.oracle_blocks("seaice_"), C.oracle_blocks("exf_")
    assert _mon(si0, "seaice_") == ref_si[0] and len(ref_si[0]) == 32, si0[:4]
    out = C.run_front(m, ds)
    nexf = nsi = 0
    for it, res, o in out:
        exf = exf_monitor_records(mh, it + 1, {k: np.asarray(v) for k, v in o["exf_mon"].items()})
        assert _mon(exf, "exf_") == ref_exf[it], [(a, b) for a, b in zip(_mon(exf, "exf_"), ref_exf[it])
                                                   if a != b][:3]
        nexf += len(ref_exf[it])
    for it in (0, 1, 2):            # SEAICE.h after the step = after SEAICE_MODEL (its only writer)
        f = C.front_step(m)
        carry = C.teacher_carry(m, ds, it)
        _, _, probes = f(m.arrays, carry, jnp.int32(it + 1), jnp.float64(tp.startTime + tp.deltaTClock*it),
                         jnp.int32(tp.nIter0 + it))
        sf = {n: v for n, v in probes["P13_seaice_model"].items()}
        c2 = carry[:4] + (dict(carry[4], seaice=sf),)
        si = seaice_monitor_records(mh, float(tp.startTime + tp.deltaTClock*(it + 1)), int(tp.nIter0 + it + 1), c2)
        assert _mon(si, "seaice_") == ref_si[it + 1], [(a, b) for a, b in zip(_mon(si, "seaice_"),
                                                                              ref_si[it + 1]) if a != b][:3]
        nsi += len(ref_si[it + 1])
    assert nexf == 3 * 67 and nsi == 3 * 32, (nexf, nsi)


def _grad_fun(m, ds):
    """theta (start of iteration 1, teacher-forced) -> (surfaceForcingT at P01, HEFF after SEAICE_MODEL)."""
    from mitjax.farray import FArray
    from mitjax.model.src.forward_step import forward_step
    tp = m.prm.time
    carry = C.teacher_carry(m, ds, 1)
    a = m.arrays
    th = carry[0].theta

    def g(theta):
        state = carry[0].replace(theta=FArray(theta, th.name, tiled=th.tiled, _dims=th.dims))
        probes = {}
        _, ff, _, _, _, out = forward_step(jnp.int32(2), jnp.float64(tp.startTime + tp.deltaTClock), jnp.int32(1),
                                           cfg=m.cfg, grid=a.grid, params=a.params, fp=m.fp, eos=a.eos,
                                           cg2dh=a.cg2dh, cg2d_params=a.cg2d_params, state=state, ff=carry[1],
                                           phi0surf=carry[2], ex=a.ex, pk=carry[4], pkc=a.pkc, pks=m.pks,
                                           probe=lambda s, v: probes.__setitem__(s, v), until=C.UNTIL)
        return ff.surfaceForcingT.data, out["pk"]["seaice"]["HEFF"].data
    return jax.jit(g), th.data


def test_gradient_front_wrt_theta_smoke():
    m, ds = C.model()
    g, x0 = _grad_fun(m, ds)

    def J(x):
        sfT, heff = g(x)
        return jnp.sum(sfT[:, 2:3, 2:3]) + jnp.sum(heff[:, 2:3, 2:3])
    gr = jax.grad(J)(x0)
    assert np.all(np.isfinite(np.asarray(gr))), "non-finite gradient lane"
    assert np.any(np.asarray(gr) != 0.0)
    y0, vjp = jax.vjp(g, x0)
    w = tuple(jax.random.normal(jax.random.PRNGKey(i), y.shape, y.dtype) for i, y in enumerate(y0))
    (xbar,) = vjp(w)
    v = jax.random.normal(jax.random.PRNGKey(7), x0.shape, x0.dtype)
    _, ydot = jax.jvp(g, (x0,), (v,))
    assert all(np.all(np.isfinite(np.asarray(t))) for t in ydot) and np.all(np.isfinite(np.asarray(xbar)))
    lhs = sum(float(jnp.vdot(a, b)) for a, b in zip(ydot, w))
    rhs = float(jnp.vdot(v, xbar))
    assert abs(lhs - rhs) <= 1e-12 * max(abs(lhs), abs(rhs), 1.0), (lhs, rhs)
    d = np.zeros(x0.shape)
    d[:, 0, 2, 2] = 1.0                           # surface theta of the column (k = 1, interior point)
    d = jnp.asarray(d)
    t = float(jnp.vdot(gr, d))
    errs = []
    for h in (1e-3, 1e-4, 1e-5, 1e-6):
        fd = (float(J(x0 + h*d)) - float(J(x0 - h*d))) / (2*h)
        errs.append(abs(fd - t) / abs(t))
    print("theta(k=1) -> J: tangent", t, "FD rel. errors", errs)
    assert min(errs) < 1e-7, (t, errs)


# ---------------------------------------------------------------------------------------------------------------
# session 4: the whole step (pkg/kpp arms of this build, FIND_ALPHA/FIND_BETA JMD95Z, GAD DST3), the whole run,
# the restart, a whole-step gradient window

# not probed: CG2D's internals (C01/C02: compared by test_cg2d_c01_c02_bitwise and through S09)
NOT_PROBED = ("C01_cg2d_inputs", "C02_cg2d_solution", "X00_exch_probe", "I02_ini_fields", "G00_geometry",
              "S17_monitor")
KPP_FIELDS = ("KPPviscAz", "KPPdiffKzS", "KPPdiffKzT", "KPPghat", "KPPhbl", "KPPfrac")


def test_steps_0_2_free_every_dumped_stage_bitwise():
    """Steps 0-2 from the Model's initial carry (free, no teacher forcing), every dumped stage S00..S16 (P02, P07_kpp,
    P10_kpp_exch, D-, T-, S-stages) on every point of the tile incl. halos, bit patterns."""
    from mitjax.tests import cs32_gate as CS
    r = C.column_run()
    bad, ncmp, _ = CS.run_compare(r, 3)
    assert not any(bad.values()), {k: v for k, v in bad.items() if v}
    miss = {it: [s for s in r.stages(it) if s not in NOT_PROBED and (it, s) not in ncmp] for it in r.its[:3]}
    assert not any(miss.values()), miss
    for it in r.its[:3]:
        assert ncmp[(it, "P07_kpp")] == 6 and ncmp[(it, "P10_kpp_exch")] == 6, (it, ncmp)
    print("free steps 0-2: (iteration, stage) pairs", len(ncmp), "field records", sum(ncmp.values()))
    assert sum(ncmp.values()) == 2091, sum(ncmp.values())


def test_steps_0_2_teacher_forced_kpp_and_every_stage_bitwise():
    """The whole step of iterations 0, 1, 2 from the oracle's state at its start (m4col_gate.teacher_carry: State at
    S00_begin, FFIELDS.h / EXF / SEAICE.h / KPP.h at their last dump): KPP_CALC vs P07_kpp, KPP_DO_EXCH vs
    P10_kpp_exch (6 KPP.h fields each) and every other dumped stage, bitwise."""
    r = C.column_run()
    f = C.step_fn(r)
    for it in (0, 1, 2):
        bad, ncmp = C.teacher_step(r, it, f=f)
        assert not bad, (it, bad)
        assert ncmp["P07_kpp"] == 6 and ncmp["P10_kpp_exch"] == 6, (it, ncmp)


def test_cg2d_c01_c02_bitwise():
    """CG2D on the oracle's C01_cg2d_inputs of iterations 0-2: C02's solution (every point) and its five scalars
    bitwise (the literal solver and the solver with the implicit derivative rule)."""
    from mitjax.tests import cs32_gate as CS
    r = C.column_run()
    for it in (0, 1, 2):
        for solve in ("literal", "rule"):
            res = CS.replay_cg2d(r, it, solve)
            assert all(v[1] == 0 and v[2] == 0 for v in res.values()), (it, solve, res)


def _plant(name, r):
    """(cfg, undo) of a planted error in one of the session-4 arms (negative controls)."""
    from types import SimpleNamespace
    import mitjax.model.src.find_alpha as FA
    import mitjax.pkg.generic_advdiff.gad_calc_rhs as GR
    import mitjax.pkg.kpp.kpp_calc as KC
    import mitjax.pkg.kpp.kpp_routines as KR
    saved = [(KC, "smooth_horiz", KC.smooth_horiz), (KR, "bldepth", KR.bldepth), (KC, "kpp_calc", KC.kpp_calc),
             (FA, "find_rhop0", FA.find_rhop0), (FA, "find_beta", FA.find_beta),
             (GR, "gad_dst3_adv_r", GR.gad_dst3_adv_r)]

    def undo():
        for mod, n, f in saved:
            setattr(mod, n, f)

    def scaled(f, fac):
        def g(*a, **kw):
            o = f(*a, **kw)
            return o.__class__(o.data * fac, o.name, tiled=o.tiled, _dims=o.dims)
        return g
    cfg = None
    if name in ("no_KPP_SMOOTH_DBLOC", "no_KPP_SMOOTH_SHSQ"):          # the arm compiled out
        cfg = C.cfg_with(r.m, off=(name[3:],))
    elif name == "KPP_ESTIMATE_UREF":                                  # the other KPP_FORCING_SURF arm (vermix's)
        cfg = C.cfg_with(r.m, on=(name,))
    elif name == "bldepth_without_sw":                                 # BLDEPTH: bfsfc = bo (selectPenetratingSW 0)
        f = KR.bldepth
        KR.bldepth = lambda *a, **kw: f(*a, **dict(kw, fp=SimpleNamespace(selectPenetratingSW=0)))
    elif name == "kppfrac_not_set":                                    # KPP_CALC :635-670 skipped
        f = KC.kpp_calc
        KC.kpp_calc = lambda t, i, **kw: dict(f(t, i, **kw), KPPfrac=kw["kppf"]["KPPfrac"])
    elif name == "jmd95_rhop0":                                        # FIND_ALPHA/BETA JMD95: rhoP0 (1+1e-12)
        FA.find_rhop0 = scaled(FA.find_rhop0, 1 + 1e-12)
    elif name == "jmd95_beta":                                         # FIND_BETA JMD95: betaLoc (1+1e-12)
        FA.find_beta = scaled(FA.find_beta, 1 + 1e-12)
    elif name == "dst3_r_flux":                                        # GAD_DST3_ADV_R's flux x 1.5
        GR.gad_dst3_adv_r = scaled(GR.gad_dst3_adv_r, 1.5)
    return cfg, undo


# measured (job 27865426, iterations 0-2 teacher-forced): these bite from iteration 1 on (KPP_ESTIMATE_UREF,
# jmd95_rhop0: u = 0 and the reference state at iteration 0) or 0 on, P07_kpp and every later stage differ
BITES = ("KPP_ESTIMATE_UREF", "bldepth_without_sw", "kppfrac_not_set", "jmd95_rhop0", "jmd95_beta")
# measured NOT to bite: the column is horizontally uniform (one wet point, periodic halos: every neighbour holds the
# same value) and has w = 0 at every level (wVel of S06 is 0 at iterations 0-2), so SMOOTH_HORIZ is an identity
# here (KPP_SMOOTH_DBLOC, and KPP_SMOOTH_SHSQ's 8 neighbours equal the 4 of the plain shear) and every vertical
# DST-3 flux is 0.5*(0 + |0|)*... = 0; these arms are executed but not seen by this column's dumps (handoff:
# a multi-column KPP build, e.g. lab_sea, gates them)
BLIND = ("no_KPP_SMOOTH_DBLOC", "no_KPP_SMOOTH_SHSQ", "dst3_r_flux")


@pytest.mark.parametrize("name", BITES + BLIND)
def test_negative_controls_session4_arms(name):
    r = C.column_run()
    cfg, undo = _plant(name, r)
    try:
        f = C.step_fn(r, cfg)
        bad = {it: C.teacher_step(r, it, f=f)[0] for it in (1, 2)}
    finally:
        undo()
    print(name, {it: {s: sorted(v) for s, v in b.items() if s in ("P07_kpp", "T11_temp_gT")} for it, b in bad.items()})
    if name in BITES:
        assert all("P07_kpp" in b and "S16_blocking_exchanges" in b for b in bad.values()), bad
    else:
        assert not any(bad.values()), bad


def test_whole_run_monitor_digits_pickups():
    """The 10-step run through the run driver (drivers.run.forward, as `python -m mitjax run`): every %MON record
    (dynamics, SEAICE and EXF blocks) and the MONITOR banners identical to the oracle STDOUT, the testreport digits vs
    results/ at the oracle's own (yardstick) digits and vs the oracle, pickup.ckptA and pickup_seaice.ckptA
    byte-identical, nothing on STDERR."""
    m, res, o, (diffs, rows, nblocks), (files, missing) = C.whole_run()
    print("1D_ocean_ice_column digits (name, ours vs results/, yardstick, ours vs oracle):", rows)
    assert nblocks == 11 and diffs == {"mon": 0, "banner": 0}, (nblocks, diffs)
    k0 = next(n for n, r in enumerate(o.raw) if "Begin MONITOR dynamic field statistics" in r) - 1
    for pre, n_rec in (("%MON seaice_", 11*32), ("%MON exf_", 10*67)):
        a = [r for r in res.records if pre in r]
        b = [r for r in o.raw[k0:] if pre in r]
        assert a == b and len(a) == n_rec, (pre, len(a), len(b))
    assert len(rows) == 25 and all(a >= y for _, a, y, _ in rows), rows
    assert all(s >= y for _, _, y, s in rows), rows
    pk = {f: ok for f, ok in files.items() if f.startswith("pickup")}
    assert sorted(pk) == ["pickup.ckptA.001.001.data", "pickup.ckptA.001.001.meta",
                          "pickup_seaice.ckptA.001.001.data", "pickup_seaice.ckptA.001.001.meta"], pk
    assert all(pk.values()), pk
    assert res.stderr == [], res.stderr[:5]


def test_restart_5_pickup_5_equals_10():
    """Run A: 10 steps with permanent pickups every 5 (pChkptFreq); run B: nIter0 = 5 (startTime = 5*deltaT,
    nTimeSteps = 5) from A's pickup.0000000005 and pickup_seaice.0000000005 (READ_PICKUP, SEAICE_READ_PICKUP).
    B's pickups at 10 (ocean and sea ice) are byte-identical to A's, B's STDOUT records from step 6 on equal A's, and
    the final carries are bitwise equal except where the Fortran's own restart code makes them differ (measured):
    halos of gU, gV, guNm1, gvNm1 (READ_PICKUP exchanges GuNm1/GvNm1, read_pickup.F:556; the running model never
    does), the sea-ice ETA, ZETA (SEAICE_INIT_VARIA :437-438 set them from the restart HEFF; nothing else writes
    them without SEAICEuseDYNAMICS: A keeps the cold start's 0), TICES levels 2..nITD of the interior
    (SEAICE_READ_PICKUP :302-315 copies level 1 there; with SEAICE_multDim = 1 only level 1 is computed) and TICES
    halos (B's exchange :321). None of these reaches a pickup record or a printed value."""
    import filecmp
    from pathlib import Path
    a, ra, b, rb = C.restart_b("restartB")
    assert ra.pickups == ["pickup.0000000005", "pickup.0000000010"] and rb.pickups == ["pickup.0000000010"]
    for pre in ("pickup", "pickup_seaice"):
        for suf in ("data", "meta"):
            f = f"{pre}.0000000010.001.001.{suf}"
            assert filecmp.cmp(Path(a.rundir) / f, Path(b.rundir) / f, shallow=False), f

    def tail(recs):
        k = next(n for n, r in enumerate(recs) if "%MON time_tsnumber" in r and r.split("=")[1].strip() == "6")
        return [r for r in recs[k:] if "%CHECKPOINT" not in r]
    ta, tb = tail(ra.records), tail(rb.records)
    assert ta == tb and len(ta) > 900, (len(ta), len(tb), [(x, y) for x, y in zip(ta, tb) if x != y][:2])
    d = C.carry_diffs(ra.carry, rb.carry)
    print("restart: carry differences (points, interior points):", d)
    assert set(d) == {"state.gU", "state.gV", "state.guNm1", "state.gvNm1", "seaice.ETA", "seaice.ZETA",
                      "seaice.TICES"}, d
    assert all(d[f"state.{n}"][1] == 0 for n in ("gU", "gV", "guNm1", "gvNm1")), d
    ta_, tb_ = np.asarray(ra.carry[4]["seaice"]["TICES"].data), np.asarray(rb.carry[4]["seaice"]["TICES"].data)
    assert np.array_equal(ta_[:, 0, 2:-2, 2:-2].view(np.int64), tb_[:, 0, 2:-2, 2:-2].view(np.int64))
    # levels 2..nITD: A keeps SEAICE_INIT_VARIA's 273 (:210); B holds the level-1 value of the pickup (the copy of
    # :302-315), unchanged since (SEAICE_multDim = 1: only level 1 is computed)
    assert np.all(ta_[:, 1:, 2:-2, 2:-2] == 273.0)
    assert np.all(tb_[:, 1:, 2:-2, 2:-2] == tb_[:, 1:2, 2:-2, 2:-2]) and np.all(tb_[:, 1:, 2:-2, 2:-2] != 273.0)
    assert not np.any(np.asarray(ra.carry[4]["seaice"]["ZETA"].data))


def test_restart_negative_controls():
    """B's pickup_seaice with the siHEFF record zeroed: the run differs (HEFF, and through the sea ice the ocean);
    with siHEFF missing from the field list: SEAICE_CHECK_PICKUP stops (cannot restart without field)."""
    def zero_heff(d):
        """siHEFF is the 3rd of the 7 interior records (fldList 'siTICE' 'siAREA' 'siHEFF' ..., one 2-D record per
        field: SEAICE_multDim = 1): its bytes set to 0."""
        meta = (d / "pickup_seaice.0000000005.001.001.meta").read_text()
        names = meta[meta.index("fldList"):].split("{")[1].split("}")[0].split("'")[1::2]
        p = d / "pickup_seaice.0000000005.001.001.data"
        raw = bytearray(p.read_bytes())
        size = len(raw) // len(names)
        k = names.index("siHEFF  ")
        assert k == 2 and len(names) == 7, names
        raw[k*size:(k + 1)*size] = bytes(size)
        p.write_bytes(bytes(raw))
    _, ra, _, rb = C.restart_b("negZeroHeff", mutate=zero_heff)
    d = C.carry_diffs(ra.carry, rb.carry)
    assert d.get("seaice.HEFF", (0, 0))[1] > 0 and d.get("state.theta", (0, 0))[1] > 0, d

    def drop_heff(d):
        p = d / "pickup_seaice.0000000005.001.001.meta"
        txt = p.read_text()
        assert "'siHEFF  '" in txt
        p.write_text(txt.replace("'siHEFF  '", "'siXXXX  '"))
    with pytest.raises(ValueError, match="cannot restart without field \"siHEFF  \""):
        C.restart_b("negDropHeff", mutate=drop_heff)


NSTEP_GRAD = 2


def _whole_step_grad_fun():
    """theta at the start of iteration 1 (teacher-forced carry) -> (theta, HEFF) after NSTEP_GRAD whole steps of
    the driver Model's step (CG2D through its implicit derivative rule, cg2d_solve)."""
    from mitjax.farray import FArray
    m, ds = C.model()
    tp = m.prm.time
    carry0 = C.teacher_carry(m, ds, 1)
    a = m.arrays
    th = carry0[0].theta

    def g(theta):
        carry = (carry0[0].replace(theta=FArray(theta, th.name, tiled=th.tiled, _dims=th.dims)),) + carry0[1:]
        t, it = jnp.float64(tp.startTime + tp.deltaTClock), jnp.int32(tp.nIter0 + 1)
        for k in range(NSTEP_GRAD):
            carry, t, it, _ = m.step(a, carry, jnp.int32(2 + k), t, it)
        return carry[0].theta.data, carry[4]["seaice"]["HEFF"].data
    return jax.jit(g), th.data


def _column_direction(x0):
    """The direction of theta(k=1) of the column: the column point and every halo copy of it (all 25 lanes of level
    1). On this periodic 1x1 domain every halo lane holds a copy of the one wet point (the teacher-forced state is
    horizontally uniform), so this is the perturbation of the model's one theta(k=1) value; the gradient along it is
    the gradient with respect to the interior point after the adjoint exchange (ADEXCH adds the halo copies' cotangents
    onto their owner)."""
    d = np.zeros(x0.shape)
    d[:, 0, :, :] = 1.0
    return jnp.asarray(d)


def _central_fd_errors(J, x0, d, t, hs=(1e-3, 1e-4, 1e-5, 1e-6)):
    Jj = jax.jit(J)
    return [abs((float(Jj(x0 + h*d)) - float(Jj(x0 - h*d))) / (2*h) - t) / abs(t) for h in hs]


def _whole_step_J():
    g, x0 = _whole_step_grad_fun()

    def J(x):
        th, heff = g(x)
        return jnp.sum(th[:, :, 2:3, 2:3]) + jnp.sum(heff[:, 2:3, 2:3])
    return J, x0


def test_gradient_whole_steps_window_finite_and_fd():
    """d/d(theta at the start of iteration 1) of J = sum of theta and HEFF at the column point after NSTEP_GRAD whole
    steps (KPP, sea ice, dynamics, CG2D, tracers): finite on every lane, nonzero, central FD along theta(k=1) of the
    column (`_column_direction`: the point and its halo copies) at the step-1 state (smoke: a switch of KPP or the sea
    ice within reach of h shows as an FD outlier; the best h is asserted).

    Not along the interior lane alone (lane DOTDIAG, dev jobs 27867308, 27867566): perturbing the interior without
    its halo copies makes the column horizontally non-uniform, so the step produces horizontal flow and a vertical
    velocity whose base value is exactly 0 (wVel = -0.0 at every level). There the DST3 fluxes of GAD_DST3_ADV_R
    (gad_dst3_adv_r.F:111-115, `0.5*(rTrans+ABS(rTrans))*(...) + 0.5*(rTrans-ABS(rTrans))*(...)`) have a kink: the
    upwind side switches with the sign of w. JAX differentiates ABS at 0 as +1 (jax.lax.abs's JVP is
    select(x >= 0, t, -t), and -0.0 >= 0), i.e. the w > 0 side, while a central FD averages both sides. Measured:
    the tangent of the gT tendency with respect to w equals the forward one-sided difference to 6 digits and differs
    from the central one by up to 100 % at some levels (dev jobs 27867267, 27867308); along the interior lane the
    forward one-sided FD of J converges to the tangent as O(h) (8.9e-3, 1.0e-3, 1.1e-4, 1.1e-5 for h = 1e-3 .. 1e-6)
    while the central FD stays 4e-5 off at every h (dev job 27867566; gate 27865852). Along the column direction the
    flow stays 0 and no kink is reached (1 step: FD error 1.8e-13, dev job 27867335; 2 steps: 1.4e-8)."""
    J, x0 = _whole_step_J()
    gr = jax.jit(jax.grad(J))(x0)
    assert np.all(np.isfinite(np.asarray(gr))), "non-finite gradient lane"
    assert np.any(np.asarray(gr) != 0.0)
    d = _column_direction(x0)
    t = float(jnp.vdot(gr, d))
    errs = _central_fd_errors(J, x0, d, t)
    print("theta(k=1) of the column -> J after", NSTEP_GRAD, "steps: tangent", t, "FD rel. errors", errs)
    assert min(errs) < 1e-7, (t, errs)


def _planted_minmax_step(op, winner, factor):
    """A Fortran MAX/MIN step (mitjax/ops/fortran_minmax._step) with the same value and its derivative times
    `factor`: a planted backward-only error for the negative control."""
    better = (lambda u, v: u > v) if op == "MAX" else (lambda u, v: u < v)
    ref = jnp.maximum if op == "MAX" else jnp.minimum

    @jax.custom_jvp
    def f(m, x):
        if winner == "a":
            return jnp.where(better(x, m), x, m)
        return jnp.where(better(m, x), m, x)

    @f.defjvp
    def _jvp(primals, tangents):
        return f(*primals), factor * jax.jvp(ref, primals, tangents)[1]
    return f


def test_gradient_whole_steps_fd_negative_control(monkeypatch):
    """Negative control of test_gradient_whole_steps_window_finite_and_fd: every Fortran MAX/MIN derivative of the
    step times (1 + 1e-4), the values unchanged (a planted backward-only error; KPP, the sea ice and EXF use them):
    the best central-FD error along the column direction rises above the 1e-7 bar (measured 6.2e-5 at every h, lane
    DOTDIAG dev job 27867566; unplanted 1.4e-8)."""
    from mitjax.ops import fortran_minmax as FM
    for key in list(FM._STEPS):
        monkeypatch.setitem(FM._STEPS, key, _planted_minmax_step(*key, 1.0 + 1e-4))
    J, x0 = _whole_step_J()
    gr = jax.jit(jax.grad(J))(x0)
    d = _column_direction(x0)
    t = float(jnp.vdot(gr, d))
    errs = _central_fd_errors(J, x0, d, t)
    print("planted MAX/MIN derivative x (1 + 1e-4): tangent", t, "FD rel. errors", errs)
    assert min(errs) > 1e-7, (t, errs)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="whole-step dot test with random directions on every lane: |TL-AD| relative 1.7e-11 (1 "
                          "step) up to 1.6e-9 (2 steps). Diagnosed as rounding of two evaluation orders of the same "
                          "linear map, no inconsistent rule (lane DOTDIAG, dev jobs 27866959, 27867014, 27867553): the "
                          "CG2D derivative PCG is linear only to ~1e-12 here (its matvec cancels 4 digits: aC2d = "
                          "-4.0003829 against four neighbours 1.0); flux-form advection / implicit diffusion of salt "
                          "and theta cancel large means (34.8 vs differences ~0); shared code: 90x40x15/input fails "
                          "the same way (dev job 27867522)")
def test_gradient_whole_steps_window_dot_test():
    """Tangent (jvp) vs adjoint (vjp) dot test of the NSTEP_GRAD-step window, random directions on every lane of
    theta and of both outputs, bar 1e-12 relative (as the front-of-step test)."""
    g, x0 = _whole_step_grad_fun()
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
