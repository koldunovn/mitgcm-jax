"""R2 tutorial_baroclinic_gyre (plan Task 13): tracer stepping (TEMP_INTEGRATE, GAD_CALC_RHS centred 2nd order,
implicit vertical diffusion GAD_IMPLICIT_R), LINEAR EOS in the step, the stable IVDC path, exactConserv, and the
first P=4 forward gate. Helpers: mitjax/tests/r2_gate.py.

* tier 1x (this file): initial state (with INTEGR_CONTINUITY's exactConserv branch at nIter0) and every field of
  every dumped stage of steps 1-3 bitwise (bit patterns, all points incl. halos; P02/P03/T11-T13/T02 included);
  negative controls measured to bite; P=4 (4 fake CPU devices, one tile each, ShardedExchanger under
  shard_map(check_vma=True)) == P=1 bitwise for the whole 10-step run; gradient finiteness on every lane and a
  tile-edge cotangent gate (FD across the tile edges).
* tier 1: test_r2_baroclinic_gyre_tier1.py (the whole run vs the oracle STDOUT and results/).

IVDC (decided 2026-10-01): CALC_IVDC's convective branch (calc_ivdc.F:48) never fires in these 10 steps, so the
stable path is gated here (IVDConvCount = 0 bitwise at P02/S04/S06); the branch itself is gated by the COL lane's
replay of statically unstable columns, and in the model in Task 16 (R5)."""

import numpy as np

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def _m():
    from mitjax.tests import r2_gate as r2
    return r2, r2.model()


def test_initial_state_bitwise_incl_exact_conserv():
    """INITIALISE_VARIA incl. INTEGR_CONTINUITY at nIter0 with exactConserv (dEtaHdt = -hDivFlow/rA, PmEpR = 0,
    UPDATE_ETAH) vs S00_begin: every State field the oracle dumps, bit patterns, all points."""
    from mitjax.tests import r1_gate as rg
    r2, m = _m()
    r = rg.compare_stage(m.ds, 0, "S00_begin", m.state0)
    assert len(r) >= 19 and not r2.bad(r), r2.bad(r)
    assert "dEtaHdt" in r and "etaH" in r and "theta" in r


def test_substeps_steps_1_to_3_bitwise():
    """Every dumped field of every stage of steps 1-3 bitwise: S02, P02 (rhoInSitu, sigmaX/Y/R incl. k=1 from the
    MXLDEPTH request, IVDConvCount), P03 (hMixLayer), S04, T11 (gT after forcing and AB), T12 (T + dt gT), T13 (after
    GAD_IMPLICIT_R, with kappaRk), T02, S05, S06, S09-S11 (exactConserv etaN, dEtaHdt), S15, S16."""
    r2, m = _m()
    res, _ = r2.run_steps(m, m.params, 3)
    for k, per_stage in enumerate(res):
        for st, r in per_stage.items():
            assert r, (k, st, "no fields compared")
            assert not r2.bad(r), (k, st, r2.bad(r))
    n = {st: len(r) for st, r in res[0].items()}
    assert n["T13_temp_impl"] == 2 and n["P02_rho_sigma_ivdc"] == 8 and n["T02_temp_integrate"] == 8, n


def test_negative_controls_bite():
    """Measured to bite (differing points summed over the stage's fields, steps 1/2/3; dev job 27831271):
    (a) diffKhT one ulp up (GAD_DIFF_X/Y): T11 0/61/1 (step 1 has no horizontal theta gradient; the ulp is absorbed
        by T + dt*gT, T13 and S16 unchanged);
    (b) diffKrNrT(5) one ulp up (GAD_IMPLICIT_R's tri-diagonal): T13 4900/4900/4900, T11 0;
    (c) the diagnostics request ignored (diagnostics_is_on = ()): doDiagsRho = 0, no GRAD_SIGMA at k=1 and no
        CALC_OCE_MXLAYER: S04 (hMixLayer) 4356 every step, P02 (sigma at k=1) 0/8448/12540, tracer stages 0;
    (d) tracForcingOutAB flipped (this run has 0, forcing inside AB, ini_parms.F:1099-1102 with forcing_In_AB):
        step 1 unchanged (abFac = 0 at nIter0), T11 bites from step 2."""
    r2, m = _m()
    st = ("P02_rho_sigma_ivdc", "S04_oceanic_phys", "T11_temp_gT", "T13_temp_impl", "S16_blocking_exchanges")
    fn = m.step_fn(st)

    def nd(params, n=2, f=fn):
        from mitjax.tests import r1_gate as rg
        state, ff, phi0, t, it = m.state0, m.ff, m.phi0surf, m.prm.time.startTime, m.prm.time.nIter0
        out = []
        for k in range(n):
            state, ff, phi0, t, it, _, pr = f(m.grid, params, m.eos, m.cg2dh, m.cg2d_params, m.ex, state, ff, phi0,
                                              jnp.int32(k + 1), jnp.float64(t), jnp.int32(it))
            out.append({s: sum(v[1] for v in rg.compare_stage(m.ds, k, s, r2.stage_values(s, pr[s])).values())
                        for s in st})
        return out
    p = m.params
    a = nd(p.replace(traced={"diffKhT": np.float64(np.nextafter(float(p.diffKhT), 1e9))}))
    assert a[0]["T11_temp_gT"] == 0 and a[1]["T11_temp_gT"] > 0, a
    kr = np.asarray(p.diffKrNrT.data).copy()
    kr[4] = np.nextafter(kr[4], 1.)
    b = nd(p.replace(traced={"diffKrNrT": type(p.diffKrNrT)(jnp.asarray(kr), "diffKrNrT", k=(1, m.cfg.size.Nr),
                                                             tiled=False)}), n=1)[0]
    assert b["T13_temp_impl"] > 0 and b["T11_temp_gT"] == 0, b
    c = nd(p.replace(static={"diagnostics_is_on": ()}), f=m.step_fn(st))
    assert c[0]["S04_oceanic_phys"] > 0 and c[1]["P02_rho_sigma_ivdc"] > 0 and c[1]["T13_temp_impl"] == 0, c
    assert p.tracForcingOutAB == 0
    d = nd(p.replace(static={"tracForcingOutAB": 1}), f=m.step_fn(st))
    assert d[0]["T11_temp_gT"] == 0 and d[1]["T11_temp_gT"] > 0, d


def test_ivdc_wired_on_a_planted_inversion():
    """The stable-path gate above would also pass if CALC_IVDC were never called (IVDConvCount starts at 0), so: plant
    a static inversion (theta at level 6 of one wet column warmer than level 5 by 1 K) in the initial state and run
    one step: IVDConvCount(k=6) = 1 there and 0 in the unplanted columns, and kappaRk of TEMP_INTEGRATE at that point
    = IVDConvCount*ivdc_kappa + diffKrNrT(6) (CALC_3D_DIFFUSIVITY), so the convective diffusivity reaches
    GAD_IMPLICIT_R (forward only; the oracle has no such run: the in-model gate against the Fortran is Task 16)."""
    r2, m = _m()
    sz = m.cfg.size
    p = (0, 5, sz.OLy + 15, sz.OLx + 15)            # storage (tile, k-1, j, i): level 6
    assert float(np.asarray(m.grid.maskC.data)[p]) == 1.
    th = m.state0.theta.data
    th = th.at[p].set(th[(0, 4) + p[2:]] + 1.0)
    st0 = m.state0.replace(theta=type(m.state0.theta)(th, "theta", _dims=m.state0.theta.dims))
    fn = m.step_fn(("P02_rho_sigma_ivdc", "T13_temp_impl"))
    out = fn(m.grid, m.params, m.eos, m.cg2dh, m.cg2d_params, m.ex, st0, m.ff, m.phi0surf, jnp.int32(1),
             jnp.float64(m.prm.time.startTime), jnp.int32(m.prm.time.nIter0))
    pr = out[-1]
    ivdc = np.asarray(pr["P02_rho_sigma_ivdc"]["IVDConvCount"].data)
    assert ivdc[p] == 1.0 and int(np.count_nonzero(ivdc)) == 1, int(np.count_nonzero(ivdc))
    kap = np.asarray(pr["T13_temp_impl"]["kappaRk"].data)
    assert kap[p] == 1.0*float(m.params.ivdc_kappa) + float(np.asarray(m.params.diffKrNrT.data)[5]), kap[p]


def test_p4_equals_p1_whole_run():
    """P=4 (shard_map(check_vma=True), 4 fake CPU devices, one 31x31 tile per device, ShardedExchanger; grid, cg2d
    operator, state and forcing placed once with TileSharding.put_tree) == the single-device run, bit for bit, on
    every leaf of the State, FFIELDS, phi0surf and the cg2d scalars after every one of the 10 steps."""
    from mitjax.eesupp import exch_maps as EM
    from mitjax.eesupp.shard import TileSharding
    r2, m = _m()
    sh = TileSharding(EM.load_maps(m.exp), 4)
    assert sh.layout.nTiles == 4 and sh.blocks.Tloc == 1
    step4, (s4, f4, p4) = r2.sharded_step_fn(m, sh)
    fn = m.step_fn(())
    s1, f1, p1, t1, it1 = m.state0, m.ff, m.phi0surf, m.prm.time.startTime, m.prm.time.nIter0
    t4, it4 = t1, it1
    for k in range(m.prm.time.nTimeSteps):
        s1, f1, p1, t1, it1, out1, _ = fn(m.grid, m.params, m.eos, m.cg2dh, m.cg2d_params, m.ex, s1, f1, p1,
                                          jnp.int32(k + 1), jnp.float64(t1), jnp.int32(it1))
        s4, f4, p4, t4, it4, out4 = step4(k, s4, f4, p4, t4, it4)
        d = r2.tree_bits_differ((s1, f1, p1, out1["cg2d"]), sh.unpad_tree((s4, f4, p4, out4["cg2d"])))
        assert not d and float(t1) == float(t4) and int(it1) == int(it4), (k, d)
        assert all(bool(np.all(np.isfinite(np.asarray(x)))) for x in jax.tree.leaves(sh.unpad_tree(s4))), k


def _cost_fn(m, nsteps=2, tiles=None):
    """J(theta0, uVel0) = sum of w * theta after `nsteps` steps (w > 0 only next to the tile edges: the interior rows
    and columns 1, 2, sNx-1, sNx (sNy-1, sNy) of every tile, or of `tiles` only), single device."""
    from mitjax.model.src.forward_step import forward_step
    cfg, fp, sz = m.cfg, m.fp, m.cfg.size
    shape = m.state0.theta.data.shape
    w = np.zeros(shape)
    OLx, OLy, sNx, sNy = sz.OLx, sz.OLy, sz.sNx, sz.sNy
    edge_i = [OLx + x - 1 for x in (1, 2, sNx - 1, sNx)]
    edge_j = [OLy + x - 1 for x in (1, 2, sNy - 1, sNy)]
    rng = np.random.default_rng(13)
    for ii in edge_i:
        w[:, :, OLy:OLy + sNy, ii] = rng.uniform(0.5, 1.5, w[:, :, OLy:OLy + sNy, ii].shape)
    for jj in edge_j:
        w[:, :, jj, OLx:OLx + sNx] = rng.uniform(0.5, 1.5, w[:, :, jj, OLx:OLx + sNx].shape)
    if tiles is not None:
        w[[t for t in range(shape[0]) if t not in tiles]] = 0.
    w = jnp.asarray(w)

    def cost(theta0, u0):
        st = m.state0.replace(theta=type(m.state0.theta)(theta0, "theta", _dims=m.state0.theta.dims),
                              uVel=type(m.state0.uVel)(u0, "uVel", _dims=m.state0.uVel.dims))
        ff, phi0, t, it = m.ff, m.phi0surf, jnp.float64(m.prm.time.startTime), jnp.int32(m.prm.time.nIter0)
        for k in range(nsteps):
            st, ff, phi0, t, it, _ = forward_step(jnp.int32(k + 1), t, it, cfg=cfg, grid=m.grid, params=m.params,
                                                  fp=fp, eos=m.eos, cg2dh=m.cg2dh, cg2d_params=m.cg2d_params,
                                                  state=st, ff=ff, phi0surf=phi0, ex=m.ex)
        return jnp.sum(w * st.theta.data)
    return cost


def test_gradient_finite_and_tile_edge_cotangents():
    """Reverse mode through two steps. (1) J = weighted sum of theta at the interior points next to the tile edges of
    all four tiles: d J / d (theta0, uVel0) finite on every lane (halos, land, all tiles). (2) J0 = the same sum on
    tile 1 (storage 0) only: its gradient w.r.t. theta0 in the first interior column of the east neighbour (storage
    1) and the first interior row of the north neighbour (storage 2) reaches them only through the exchanges'
    transposes: nonzero, and central FD at h = 0.3, 0.1, 0.03 within 1e-6 relative of the reverse-mode value at
    the best h (measured, dev job 27831271: best relative error 4e-9, 1.0e-7, 1.3e-7, 2.4e-8 at the four points;
    below h = 0.01 the FD error grows to 1e-6..2e-6, the forward noise of the cg2d stopping test at 1e-7)."""
    r2, m = _m()
    sz = m.cfg.size
    th0, u0 = m.state0.theta.data, m.state0.uVel.data
    gt, gu = jax.jit(jax.grad(_cost_fn(m), argnums=(0, 1)))(th0, u0)
    assert bool(jnp.all(jnp.isfinite(gt))) and bool(jnp.all(jnp.isfinite(gu)))
    cost0 = _cost_fn(m, tiles=(0,))
    g0 = jax.jit(jax.grad(cost0))(th0, u0)
    assert bool(jnp.all(jnp.isfinite(g0)))
    f = jax.jit(cost0)
    OLx, OLy, sNx, sNy = sz.OLx, sz.OLy, sz.sNx, sz.sNy
    mask = np.asarray(m.grid.maskC.data)
    # (storage tile, k, j, i): first interior column of tile 2 (east of tile 1), first interior row of tile 3
    # (north of tile 1), wet points away from the walls, levels 1 and 5
    pts = [(1, 0, OLy + 15, OLx), (1, 4, OLy + 5, OLx), (2, 0, OLy, OLx + 15), (2, 4, OLy, OLx + 5)]
    for p in pts:
        assert mask[p] == 1., p
        assert float(g0[p]) != 0., p
        rel = [abs(float((f(th0.at[p].add(h), u0) - f(th0.at[p].add(-h), u0)) / (2 * h)) - float(g0[p]))
               / abs(float(g0[p])) for h in (3e-1, 1e-1, 3e-2)]
        assert min(rel) <= 1e-6, (p, rel, float(g0[p]))


def test_gad_kernel_cfg_matches_replay_cfg():
    """The model's GAD kernel config (gad_calc_rhs.gad_kernel_cfg) == the GAD-A replay harness's KernelCfg for the
    baroclinic gyre (the kernels were gated with the latter)."""
    from mitjax.pkg.generic_advdiff.gad_calc_rhs import gad_kernel_cfg
    from mitjax.tests import gad_a_replay as ga
    r2, m = _m()
    assert tuple(gad_kernel_cfg(m.cfg)) == tuple(ga.kernel_cfg(*r2.EXP))
