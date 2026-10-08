"""M2 acceptance (plan Task 27), lane M2ACCEPT: the two P=N errors of the M2 forward variants fixed and gated.

(A) CALC_SURF_DR's global diagnostic (calc_surf_dr.F:200-212: _GLOBAL_SUM_RL of adjust_nb_pt and adjust_volum) from
    every real tile in tile order: the per-tile counts and the per-point rA*(Rmin_surf - rSurftmp) gathered with
    ex.all_tiles (calc_surf_dr.py), summed on the host (drivers/run._surf_adjust_lines).
(B) cost.h objf_tracer(nSx,nSy) of COST_TRACER (cost_tracer.F:51, every step from COST_TILE) kept replicated as
    COST_FINAL's tile_fc and every objf_*: each tile's locfc gathered with ex.all_tiles and added to its own
    objf_tracer(bi,bj), the one addition of the Fortran, the same on every device (cost_tracer.py).

Gates (fake CPU devices, jit(shard_map(check_vma=True)) of the run driver's own step, go_gate.run_steps_p):
1. CALC_SURF_DR with a planted etaH that clips 3 points on each of 5 tiles spread over the 4 devices
   (adjustment.cs-32x32x1/input.nlfs, 48 tiles of 16x8; the verification runs clip none, so their SURF_ADJUSTMENT
   diagnostic is zero): the State, both gathered `adjust` arrays and the host's SURF_ADJUSTMENT line at P = 4 ==
   P = 1 bitwise, and the line's count == the planted points.
2. The VJP of COST_TRACER (B) w.r.t. (objf_tracer, pTracer) at P = 3 (4 tiles: 2 per device, 2 padding tiles) ==
   P = 1 bitwise (each tile's cotangent is its own objf_tracer cotangent times the coefficients: the psum of
   ex.all_tiles transposes to the device's own block, no sum).
3. Whole runs P = N == P = 1 bitwise, every carry leaf and per-step output: adjustment.cs-32x32x1/input.nlfs (P = 4),
   global_ocean.90x40x15/input_ad forward (P = 4), tutorial_tracer_adjsens/input_ad forward (P = 3, padded).
Negative controls (measured): the pre-fix code planted back (the local-tile-only count and field of CALC_SURF_DR,
COST_TRACER's local-tile-only locfc) makes gates 1, 2 and 3 fail. Tier 1x (~15 min on a CPU compute node).
"""

import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from mitjax.farray import FArray  # noqa: E402
from mitjax.tests import go_gate as G  # noqa: E402
from mitjax.tests.ptracers_gate import planted  # noqa: E402

NLFS = ("adjustment.cs-32x32x1", "input.nlfs")
ADJSENS = ("tutorial_tracer_adjsens", "input_ad")
RUNS = [(NLFS, 4), (("global_ocean.90x40x15", "input_ad"), 4), (ADJSENS, 3)]       # (variant, P)

# the pre-fix code (branch m2-acceptance @ f6a0a84), planted back for the negative controls
SURF_DR_LOCAL = ("mitjax/model/src/calc_surf_dr.py",
                 """    adjust = (ex.all_tiles(jnp.sum(below & interior, axis=(1, 2), dtype=jnp.int32)),
              ex.all_tiles(jnp.where(below & interior, rA*(Rmin - rSurftmp), 0.)))""",
                 """    adjust = (jnp.sum(below & interior, axis=(1, 2), dtype=jnp.int32),
              jnp.where(below & interior, rA*(Rmin - rSurftmp), 0.))""")
COST_TRACER_LOCAL = ("mitjax/pkg/cost/cost_tracer.py", "return objf_tracer + ex.all_tiles(locfc)",
                     "return objf_tracer + locfc")


def model_maps(cfg, exp):
    """The exchange maps drivers/model.Model builds its Exchanger from (scripts/m1_acceptance.py model_maps)."""
    from mitjax.eesupp.exch_maps import load_cube_maps, load_maps
    from mitjax.pkg.exch2.w2_readparms import use_cubed_sphere_exchange
    if cfg.cpp.ALLOW_EXCH2 and use_cubed_sphere_exchange(exp):
        return load_cube_maps(cfg.experiment, cfg.input_dir)
    return load_maps(cfg.experiment, code=cfg.code_dir)


_MODELS = {}


def driver_model(exp, inp):
    """The run driver's Model of exp/inp (as `python -m mitjax run`), its experiment, built once per process."""
    if (exp, inp) not in _MODELS:
        from mitjax.drivers.model import Model
        from mitjax.drivers.run import load_experiment, make_rundir
        from mitjax.tests import advect_gate as ag
        exp_dir = G.paths_upstream() / "verification" / exp
        e = load_experiment(exp_dir, inp)
        _MODELS[(exp, inp)] = (Model(e, make_rundir(exp_dir, inp, ag.out_dir(f"m2accept-{exp}-{inp}"))), e)
    return _MODELS[(exp, inp)]


def _bits(x):
    a = np.asarray(x)
    return a.dtype, a.shape, np.ascontiguousarray(a).tobytes()


def _same(t1, t2):
    la, lb = jax.tree.leaves(t1), jax.tree.leaves(t2)
    return len(la) == len(lb) and all(_bits(a) == _bits(b) for a, b in zip(la, lb))


def _sharding(m, e, nproc):
    from mitjax.eesupp.shard import TileSharding
    return TileSharding(model_maps(m.cfg, e), nproc)


# ------------------------------------------------------------------------------------------------ gate 1 (A, unit)
PLANT_TILES = (0, 5, 13, 30, 47)        # tiles 1, 6, 14, 31, 48 (P = 4: 12 tiles per device; devices 0, 0, 1, 2, 3)


def planted_eta(m):
    """etaH = Rmin_surf - Ro_surf - (1 + t) on 3 interior points of each tile t in PLANT_TILES (rSurftmp below
    Rmin_surf: clipped, :112-139), the initial etaH elsewhere. Returns (State, number of planted points)."""
    st = m.initial_carry()[0]
    sz = m.cfg.size
    eta = np.array(st.etaH.data)
    rmin, ro = np.asarray(st.Rmin_surf.data), np.asarray(m.arrays.grid.Ro_surf.data)
    ks = np.asarray(m.arrays.grid.kSurfC.data)
    n = 0
    for t in PLANT_TILES:
        for (j, i) in ((3, 4), (5, 9), (sz.sNy, sz.sNx)):              # Fortran (j, i) in 1..sNy, 1..sNx
            jj, ii = j - 1 + sz.OLy, i - 1 + sz.OLx
            assert ks[t, jj, ii] <= sz.Nr                               # wet
            eta[t, jj, ii] = rmin[t, jj, ii] - ro[t, jj, ii] - (1.0 + t)
            n += 1
    return st.replace(etaH=FArray(jnp.asarray(eta), st.etaH.name, tiled=True, _dims=st.etaH.dims)), n


def surf_dr_gate(mod=None):
    """(P=1 and P=4 results equal, SURF_ADJUSTMENT line at P=1, at P=4, planted points); an exception of the P=4
    program (a check_vma error: a tile-local value declared replicated) counts as a failed gate."""
    from mitjax.drivers.run import _surf_adjust_lines
    from mitjax.drivers.sharded_grad import place_model, tile_specs
    from mitjax.model.src import calc_surf_dr as CS
    f_surf = (mod or CS).calc_surf_dr
    m, e = driver_model(*NLFS)
    st, npts = planted_eta(m)
    a = m.arrays

    def f(arrays, state):
        s, adj = f_surf(state.etaH, 0., 1, cfg=m.cfg, grid=arrays.grid, params=arrays.params, state=state,
                        ex=arrays.ex)
        return s, adj
    s1, adj1 = jax.tree.map(np.asarray, jax.jit(f)(a, st))
    line1 = _surf_adjust_lines(adj1, 1)
    sh = _sharding(m, e, 4)
    try:
        fN = sh.shard_map(f, in_specs=(tile_specs(sh, a), tile_specs(sh, st)),
                          out_specs=(tile_specs(sh, st), (sh.REP, sh.REP)))
        sN, adjN = fN(place_model(sh, a), place_model(sh, st))
    except Exception as err:                                            # noqa: BLE001 (the gate's verdict)
        return False, line1, repr(err)[:300], npts
    sN, adjN = sh.unpad_tree(sN), jax.tree.map(np.asarray, adjN)
    lineN = _surf_adjust_lines(adjN, 1)
    return _same(s1, sN) and _same(adj1, adjN) and line1 == lineN, line1, lineN, npts


def test_surf_dr_adjust_pn():
    ok, line1, lineN, npts = surf_dr_gate()
    print("SURF_ADJUSTMENT P=1:", line1, "P=4:", lineN)
    assert ok, (line1, lineN)
    assert len(line1) == 1 and int(line1[0].split()[-2]) == npts == 3 * len(PLANT_TILES), line1


def test_surf_dr_adjust_pn_control():
    """The pre-fix tile-local count and field planted back: the P=4 gate fails."""
    ok, line1, err, _ = surf_dr_gate(planted(*SURF_DR_LOCAL))
    print("control:", err)
    assert not ok


# ------------------------------------------------------------------------------------------------ gate 2 (B, VJP)
def cost_tracer_vjp_gate(mod=None):
    """VJP of objf_tracer = COST_TRACER(objf_tracer0, pTracer) w.r.t. (objf_tracer0, pTracer) for a fixed cotangent
    on objf_tracer, at P=1 and P=3 (padded): (equal bitwise, detail)."""
    from mitjax.drivers.sharded_grad import place_model, tile_specs
    from mitjax.pkg.cost import cost_tracer as CT
    from mitjax.pkg.ptracers.ptracers_fields_h import ptf_of_state
    f_ct = (mod or CT).cost_tracer
    m, e = driver_model(*ADJSENS)
    st = m.initial_carry()[0]
    obj0 = m.initial_carry()[4]["cost"].objf_tracer
    nT = obj0.shape[0]
    ct_obj = jnp.asarray(1.0 + np.arange(nT, dtype=np.float64) / 8.)
    a = m.arrays

    def f(arrays, obj, ptr01, ct):
        def J(obj, ptr01):
            ptf = ptf_of_state(st.replace(pTracer_01=ptr01), arrays.params)
            return f_ct(obj, cfg=m.cfg, grid=arrays.grid, params=arrays.params, ptr=arrays.params, ptf=ptf,
                        ex=arrays.ex)
        val, vjp = jax.vjp(J, obj, ptr01)
        return val, vjp(ct)
    p1 = jax.tree.map(np.asarray, jax.jit(f)(a, obj0, st.pTracer_01, ct_obj))
    sh = _sharding(m, e, 3)
    try:
        fN = sh.shard_map(f, in_specs=(tile_specs(sh, a), sh.REP, tile_specs(sh, st.pTracer_01), sh.REP),
                          out_specs=(sh.REP, (sh.REP, tile_specs(sh, st.pTracer_01))))
        pN = fN(place_model(sh, a), place_model(sh, obj0), place_model(sh, st.pTracer_01), place_model(sh, ct_obj))
    except Exception as err:                                            # noqa: BLE001 (the gate's verdict)
        return False, repr(err)[:300]
    pN = sh.unpad_tree(pN)
    nz = int(np.count_nonzero(np.asarray(p1[1][1].data)))
    return _same(p1, pN) and nz > 0, f"objf_tracer {p1[0]}, nonzero pTracer cotangent points {nz}"


def test_cost_tracer_vjp_pn():
    ok, info = cost_tracer_vjp_gate()
    print(info)
    assert ok, info


def test_cost_tracer_vjp_pn_control():
    ok, info = cost_tracer_vjp_gate(planted(*COST_TRACER_LOCAL))
    print("control:", info)
    assert not ok


# ------------------------------------------------------------------------------------------------ gate 3 (whole runs)
def whole_run_gate(exp, inp, nproc, n=None):
    """The whole run (n steps: all when None) at P=1 and P=nproc: (every carry leaf and per-step output bitwise,
    info); an exception of the P=nproc program counts as a failed gate."""
    m, e = driver_model(exp, inp)
    n = m.prm.time.nTimeSteps if n is None else n
    c1, o1 = G.run_steps_p(m, n)
    try:
        cN, oN = G.run_steps_p(m, n, nproc, maps=model_maps(m.cfg, e))
    except Exception as err:                                            # noqa: BLE001 (the gate's verdict)
        return False, repr(err)[:300]
    la, lb = jax.tree.leaves(c1), jax.tree.leaves(cN)
    ndiff = sum(_bits(x) != _bits(y) for x, y in zip(la, lb))
    same = len(o1) == len(oN) == n and all(_same(p_, q_) and jax.tree.structure(p_) == jax.tree.structure(q_)
                                           for p_, q_ in zip(o1, oN))
    return ndiff == 0 and same and len(la) == len(lb), f"{n} steps, {len(la)} leaves, {ndiff} differ, outputs {same}"


@pytest.mark.parametrize("run,nproc", RUNS, ids=[f"{r[0]}-{r[1]}-P{p}" for r, p in RUNS])
def test_whole_run_pn(run, nproc):
    ok, info = whole_run_gate(*run, nproc)
    print(run, f"P={nproc}", info)
    jax.clear_caches()
    assert ok, info


CONTROLS = [(NLFS, 4, SURF_DR_LOCAL, "mitjax.model.src.calc_surf_dr", "calc_surf_dr"),
            (("global_ocean.90x40x15", "input_ad"), 4, SURF_DR_LOCAL, "mitjax.model.src.calc_surf_dr",
             "calc_surf_dr"),
            (ADJSENS, 3, COST_TRACER_LOCAL, "mitjax.pkg.cost.cost_tracer", "cost_tracer")]


@pytest.mark.parametrize("run,nproc,plant,modname,fn", CONTROLS,
                         ids=[f"{r[0]}-{r[1]}-P{p}" for r, p, *_ in CONTROLS])
def test_whole_run_pn_control(run, nproc, plant, modname, fn, monkeypatch):
    """The pre-fix routine planted back (FORWARD_STEP imports it at call time: the module attribute is replaced):
    the P=N whole-run gate fails on its first step."""
    import importlib
    monkeypatch.setattr(importlib.import_module(modname), fn, getattr(planted(*plant), fn))
    ok, info = whole_run_gate(*run, nproc, n=1)
    print("control:", run, info)
    jax.clear_caches()
    assert not ok
