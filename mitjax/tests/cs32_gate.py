"""Helpers of the global_ocean.cs32x15/input gates (plan Task 25, GO lane session 8; not a test file): the run
driver's Model on the run directory of lane A's dumps-on run (cube_run_gate.CubeRun), its own step (Model.step's
forward_step with the pkg carry: GM/Redi, the forcing preload) with forward_step's probe on every dumped stage incl.
the per-level ones, and the comparison with the oracle dumps on every point of every tile (cube_run_gate.compare_stage).
"""

import jax
import jax.numpy as jnp

from mitjax.tests import cube_run_gate as C

EXP = ("global_ocean.cs32x15", "input")


def cs32_run():
    return C.cube_run(*EXP)


def probed_step_fn(r, stages, until=None):
    """jit(step)(arrays, carry, iloop, myTime, myIter) -> (carry, myTime, myIter, out, probes): the driver's step
    (go_gate.probed_steps) with forward_step's probe for the stage names `stages` (per-level stages keyed
    'name|k')."""
    from mitjax.farray import FArray
    from mitjax.model.src.forward_step import forward_step
    m = r.m
    cfg, fp, pks = m.cfg, m.fp, m.pks

    def step(a, carry, iloop, myTime, myIter):
        probes = {}

        def probe(stage, values):
            name = stage[0] if isinstance(stage, tuple) else stage
            if name in stages:
                probes[f"{stage[0]}|{stage[1]}" if isinstance(stage, tuple) else stage] = values
        state, ff, phi0surf = carry[:3]
        pk = carry[4] if len(carry) > 4 else None
        kw = {} if until is None else dict(until=until)
        state, ff, phi0surf, myTime, myIter, out = forward_step(
            iloop, myTime, myIter, cfg=cfg, grid=a.grid, params=a.params, fp=fp, eos=a.eos, cg2dh=a.cg2dh,
            cg2d_params=a.cg2d_params, state=state, ff=ff, phi0surf=phi0surf, ex=a.ex, probe=probe, pk=pk,
            pkc=a.pkc, pks=pks, **kw)
        if until is not None:
            return carry, myTime, myIter, out, probes
        flow = tuple(FArray(f.data, c.name, tiled=c.tiled, _dims=c.dims) for f, c in zip(out["flow"], carry[3]))
        return (state, ff, phi0surf, flow) + ((out["pk"],) if pk is not None else ()), myTime, myIter, out, probes
    return step


def trace(r, until=None):
    """Trace (no compile) one step with every dumped stage probed: raises where the port stops."""
    m = r.m
    step = probed_step_fn(r, tuple(r.stages(r.its[0])), until)
    t, it = m.start_counters()
    return jax.eval_shape(step, m.arrays, m.initial_carry(), jnp.int32(1), t, it)


def run_compare(r, n, until=None):
    """n steps from the initial carry, every dumped stage probed and compared with the oracle (bit patterns, every
    point of every tile): ({(it, stage): bad fields}, {(it, stage): fields compared}, final carry)."""
    m = r.m
    stages = tuple(sorted({s for i in r.its for s in r.stages(i)}))
    f = jax.jit(probed_step_fn(r, stages, until))
    carry = m.initial_carry()
    t, it = m.start_counters()
    out, ncmp = {}, {}
    for k in range(n):
        itn = int(it)
        carry, t, it, _, probes = f(m.arrays, carry, jnp.int32(k + 1), t, it)
        if itn not in r.its:
            continue
        for key, vals in probes.items():
            st = (key.split("|")[0], int(key.split("|")[1])) if "|" in key else key
            name = st[0] if isinstance(st, tuple) else st
            if name in r.stages(itn):
                if hasattr(vals, "__dataclass_fields__"):        # S18_cost_tile: cost.h (CostCommon) by field
                    vals = _cost_fields(vals, r.m.cfg.size)
                res = C.compare_stage(r, itn, st, vals)
                out[(itn, name)] = {**out.get((itn, name), {}), **C.bad(res)}
                ncmp[(itn, name)] = ncmp.get((itn, name), 0) + len(res)
    return out, ncmp, carry


# ---------------------------------------------------------------------------------------------------------------
# teacher-forced replays of the solver part of the step (inputs from the oracle's dumps of the same iteration)

def _s(r):
    from types import SimpleNamespace
    return SimpleNamespace(ds=r.ds, cfg=r.m.cfg)


def replay_cg2d(r, it, solve="literal"):
    """CG2D (solve="literal": cg2d; "rule": cg2d_solve) on the C01_cg2d_inputs of iteration `it` (cg2d_b, cg2d_x, the
    six operator arrays) with the Model's CG2D.h scalars (cg2dNorm, cg2dTolerance_sq of INI_CG2D in W units,
    cg2dNormaliseRHS = .FALSE.) and Cg2dParams, under jit. Returns {name: (n, n differing bits, n non-finite ours)}
    for C02's cg2d_x (every point) and its five scalars."""
    import numpy as np
    from mitjax.model.src import cg2d as cg2d_mod
    from mitjax.tests import cg2d_gate as cg
    m = r.m
    sz = m.cfg.size
    cg2dh = m.cg2dh.replace(**{n: cg.xy(cg.dump2d(r.ds, it, "C01_cg2d_inputs", n), n, sz) for n in cg.OPERATOR})
    b = cg.xy(cg.dump2d(r.ds, it, "C01_cg2d_inputs", "cg2d_b"), "cg2d_b", sz)
    x = cg.xy(cg.dump2d(r.ds, it, "C01_cg2d_inputs", "cg2d_x"), "cg2d_x", sz)
    p = m.cg2d_params
    fn = cg2d_mod.cg2d if solve == "literal" else cg2d_mod.cg2d_solve
    f = jax.jit(lambda b, x, c, ex: fn(b, x, p.cg2dMaxIters, p.cg2dUseMinResSol - 1, cfg=m.cfg, cg2dh=c, params=p,
                                       ex=ex))
    out = jax.tree.map(np.asarray, f(b, x, cg2dh, m.ex))
    ref = cg.dump2d(r.ds, it, "C02_cg2d_solution", "cg2d_x")
    ours = np.asarray(out[1].data)
    res = {"cg2d_x": (ref.size, int(np.sum(ours.view(np.int64) != ref.view(np.int64))),
                      int(np.sum(~np.isfinite(ours))))}
    for name, k in zip(cg.SCALARS, (5, 6, 2, 3, 4)):
        val, refv = np.float64(out[k]), np.float64(r.ds.scalar(it, "C02_cg2d_solution", name))
        res[name] = (1, int(val.view(np.int64) != refv.view(np.int64)), int(not np.isfinite(val)))
    return res


def replay_solve_for_pressure(r, it):
    """SOLVE_FOR_PRESSURE of iteration `it` on a teacher State (S06_dynamics, then S07_update_rstar_T's hFac / r*
    fields, then S08_update_cg2d's operator) and FFIELDS.h of S04 (EmPmR), the step's grid and CG2D.h taken from that
    State (forward_step.nlfs_load), under jit: S09_solve_for_pressure's etaN compared on every point. Returns
    ({field: compare tuple}, solver scalars)."""
    import numpy as np
    from mitjax.model.src.forward_step import nlfs_load
    from mitjax.model.src.solve_for_pressure import solve_for_pressure
    from mitjax.tests import go_gate as G
    m = r.m
    s = _s(r)
    st = G.teacher_state(s, it, [("S06_dynamics", None), ("S07_update_rstar_T", None), ("S08_update_cg2d", None)])
    ff, _ = G.teacher_ff(s, it, "S04_oceanic_phys")
    tp = m.prm.time
    k = it - int(tp.nIter0) + 1                                     # forward_step.F:807-808: the updated counters
    t1, i1 = jnp.float64(tp.startTime + tp.deltaTClock*k), jnp.int32(int(tp.nIter0) + k)

    def f(st, ff, t, i):
        grid, cg2dh = nlfs_load(cfg=m.cfg, grid=m.grid, cg2dh=m.cg2dh, state=st)
        return solve_for_pressure(t, i, cfg=m.cfg, grid=grid, params=m.params, state=st, ff=ff, cg2dh=cg2dh,
                                  cg2d_params=m.cg2d_params, ex=m.ex)
    st9, diag = jax.jit(f)(st, ff, t1, i1)
    res = C.compare_stage(r, it, "S09_solve_for_pressure", {"etaN": st9.etaN})
    return res, jax.tree.map(np.asarray, diag)


def _cost_fields(cost, sz):
    """cost.h scalars and per-tile values as the dump stores them (S18_cost_tile: every point of a tile's 2-D field
    holds the tile's value; fc, glofc the global scalar)."""
    import numpy as np
    from types import SimpleNamespace
    nt = sz.nSx * sz.nSy
    shp = (nt, 1, sz.sNy + 2 * sz.OLy, sz.sNx + 2 * sz.OLx)
    out = {}
    for f in ("fc", "glofc", "tile_fc"):
        v = np.asarray(getattr(cost, f), np.float64)
        out[f] = SimpleNamespace(data=np.broadcast_to(v.reshape((-1, 1, 1, 1) if v.ndim else (1, 1, 1, 1)), shp))
    return out

