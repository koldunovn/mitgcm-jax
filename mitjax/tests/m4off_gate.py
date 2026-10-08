"""Helpers of the offline_exf_seaice/input.thermo gates (M4 step 4, lane M4OFF session 1).

Oracle: lane A's registered dumps-on run job27855986-jdon (reference/reference_runs.py M4_TRIPLE; build
offline_exf_seaice-code-63cdc0b-704fd6b-jaxdump), iterations 0, 1, 2. The driver Model of the run is built from the
oracle's run directory; iteration 0 starts from the Model's own initial carry, iterations 1-2 are teacher-forced
(`teacher_carry`: the oracle's state at the start of the step; TICES: only category 1 is dumped, SEAICE_multDim = 1).
The sea-ice kernels are gated stage by stage, each from the oracle's values at the stage before (`stage_inputs`).
Session 2: SEAICE_GROWTH and SEAICE_MODEL from the oracle's fields before their stage (`inputs_at`), the whole step
(`steps`, through m4col_gate.step_fn), the 120-step run (`whole_run`), the restart (`restart_b`), P=N (`sharded_vs_single`)
and the gradient window (`window`, `window_cost`, `sharded_grad_vs_single`).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.farray import FArray

EXP = ("offline_exf_seaice", "input.thermo")
JDON = "job27855986-jdon"
UNTIL = "S02_load_fields"
FRONT_STAGES = ("S00_begin", "X01_exf_getffields", "X02_exf_radiation", "X03_exf_wind", "X04_exf_bulkformulae",
                "X05_exf_hflux_sflux", "X06_exf_mapfields", "S02_load_fields")


@functools.lru_cache(maxsize=None)
def model():
    """(Model, DumpSet) of input.thermo, from the oracle's dumps-on run directory."""
    from mitjax import paths
    from mitjax.config.params import load
    from mitjax.drivers.model import Model
    from mitjax.io.dump import DumpSet
    from mitjax.xla_flags import set_gate_xla_flags
    set_gate_xla_flags()
    top = paths.REFERENCE_RUNS / EXP[0] / EXP[1] / JDON
    return Model(load(*EXP), top / "rundir"), DumpSet(top / "dumps")


def like(old, data):
    """`data` (a dump record) as an FArray declared like `old`; a field with an nITD axis gets category 1 only."""
    a = np.asarray(data)
    if old.data.ndim == 4 and a.size != old.data.size:            # TICES: category 1 (SEAICE_multDim = 1)
        d = np.asarray(old.data).copy()
        d[:, 0] = a.reshape(d[:, 0].shape)
        a = d
    elif old.data.ndim == a.ndim - 1:                              # a 2-D field dumped with a k axis of 1
        a = a[:, 0]
    return FArray(jnp.asarray(a.reshape(old.data.shape), old.data.dtype), old.name, tiled=old.tiled, _dims=old.dims)


def _last(ds, it, name):
    for i in range(it - 1, -1, -1):
        keys = set(ds.keys(i))
        have = [st for st in ds.stages(i) if (i, st, name) in keys]
        if have:
            return np.asarray(ds.field(i, have[-1], name))
    return None


def teacher_carry(m, ds, it):
    """The Model's carry with the oracle's state at the start of iteration `it` (as m4col_gate.teacher_carry; the
    nITD axis of TICES by `like`)."""
    carry = m.initial_carry()
    if it == 0:
        return carry
    from mitjax.tests import r1_gate as rg
    state, ff, phi0surf, flow, pk = carry
    keys = set(ds.keys(it))
    sv = rg.state_fields(state)
    state = state.replace(**{n: like(sv[n], ds.field(it, "S00_begin", n)) for n in sv
                             if (it, "S00_begin", n) in keys and hasattr(sv[n], "data")})
    ffr = {}
    for n in ff.names():
        a = _last(ds, it, n)
        if a is not None:
            ffr[n] = like(getattr(ff, n), a)
    ff = ff.replace(**ffr)
    a = _last(ds, it, "phi0surf")
    if a is not None:
        phi0surf = like(phi0surf, a)
    pk = dict(pk)
    for key in ("exf", "seaice"):
        d = dict(pk[key])
        for n, v in d.items():
            if key == "exf" and n[-1:] in ("0", "1"):
                continue
            a = _last(ds, it, n)
            if a is not None and hasattr(v, "data") and v.data.dtype != jnp.int32:
                d[n] = like(v, a)
        pk[key] = d
    return (state, ff, phi0surf, flow, pk)


def front_fn(m, until=UNTIL):
    """jit(f)(arrays, carry, iloop, myTime, myIter) -> probes: FORWARD_STEP up to `until` with every probe kept."""
    from mitjax.model.src.forward_step import forward_step
    cfg, fp, pks = m.cfg, m.fp, m.pks

    def f(a, carry, iloop, myTime, myIter):
        probes = {}

        def probe(stage, values):
            probes[stage] = values
        state, ff, phi0surf = carry[:3]
        probes["_in"] = dict(ff=ff, phi0surf=phi0surf, gm=None, grid=a.grid)
        forward_step(iloop, myTime, myIter, cfg=cfg, grid=a.grid, params=a.params, fp=fp, eos=a.eos, cg2dh=a.cg2dh,
                     cg2d_params=a.cg2d_params, state=state, ff=ff, phi0surf=phi0surf, ex=a.ex, pk=carry[4],
                     pkc=a.pkc, pks=pks, probe=probe, until=until)
        return probes
    return jax.jit(f)


def run_front(m, ds, its=(0, 1, 2), arrays=None):
    """{it: {stage: {field: compare_field}}} of the front of FORWARD_STEP (through S02) at the dumped iterations."""
    from mitjax.tests import goadk_model_gate as M
    f = front_fn(m)
    a = m.arrays if arrays is None else arrays
    tp = m.prm.time
    out = {}
    for it in its:
        probes = f(a, teacher_carry(m, ds, it), jnp.int32(it + 1), jnp.float64(tp.startTime + tp.deltaTClock*it),
                   jnp.int32(tp.nIter0 + it))
        out[it] = M.compare_step(m, ds, it, probes)
    return out


def bad(res):
    """The (stage, field) entries of a compare_step result that differ (shape or any count)."""
    return {(s, n): v for s, d in res.items() for n, v in d.items() if v[0] == "shape" or any(v[1:])}


def differing(ds, it, stage, vals):
    """{field: number of points whose bit pattern differs} of `vals` ({name: FArray}) against the oracle's stage."""
    keys = set(ds.keys(it))
    out = {}
    for n, v in vals.items():
        if (it, stage, n) not in keys:
            continue
        r = np.asarray(ds.field(it, stage, n))
        o = np.asarray(v.data, np.float64)
        if o.ndim == 4 and r.size != o.size:
            o = o[:, 0]
        r = np.ascontiguousarray(r.reshape(o.shape), np.float64)
        out[n] = int(np.count_nonzero(np.ascontiguousarray(o).view(np.int64) != r.view(np.int64)))
    return out


def stage_inputs(m, ds, it, stage, names):
    """{name: FArray} of SEAICE.h / EXF_FIELDS.h / FFIELDS.h fields at the oracle's `stage` of iteration `it`."""
    sf, exf, ff = m.pk0["seaice"], m.pk0["exf"], m.ff0
    out = {}
    for n in names:
        old = sf.get(n) if n in sf else exf.get(n) if n in exf else getattr(ff, n)
        out[n] = like(old, ds.field(it, stage, n))
    return out


# ------------------------------------------------------------------- session 2: SEAICE_GROWTH and SEAICE_MODEL
SF_I04 = ("AREA", "HEFF", "HSNOW", "TICES", "UICE", "VICE", "d_HEFFbyNEG", "d_HSNWbyNEG", "frWtrIce", "saltWtrIce")
FF_I04 = ("EmPmR", "Qnet", "Qsw", "fu", "fv", "pLoad", "sIceLoad", "saltFlux")


def before(ds, it, stage, name):
    """The oracle's value of `name` at the last stage of iteration `it` before `stage` that dumps it, else at the
    last stage of an earlier iteration that dumps it (None if never dumped)."""
    keys = set(ds.keys(it))
    sts = ds.stages(it)
    for st in reversed(sts[:sts.index(stage)]):
        if (it, st, name) in keys:
            return np.asarray(ds.field(it, st, name))
    return _last(ds, it, name)


def inputs_at(m, ds, it, stage):
    """(sf, ff, exf, state) of the Model with every field the oracle dumps before `stage` of iteration `it` (each from
    the last stage that dumps it): the teacher-forced inputs of the routine that produces `stage`."""
    sf, exf = dict(m.pk0["seaice"]), dict(m.pk0["exf"])
    for d in (sf, exf):
        for n, v in d.items():
            if hasattr(v, "data") and v.data.dtype != jnp.int32:
                a = before(ds, it, stage, n)
                if a is not None:
                    d[n] = like(v, a)
    ffr = {}
    for n in m.ff0.names():
        a = before(ds, it, stage, n)
        if a is not None:
            ffr[n] = like(getattr(m.ff0, n), a)
    ff = m.ff0.replace(**ffr)
    st = m.state0
    st = st.replace(**{n: like(getattr(st, n), ds.field(it, "S00_begin", n)) for n in ("theta", "salt", "uVel",
                                                                                         "vVel", "etaN")})
    return sf, ff, exf, st


def growth_fn(m, cfg=None):
    """jit(f(sp, op, sf, ff, exf, st, myTime, myIter) -> (sf, ff)) of SEAICE_GROWTH (`cfg`: a planted CPP view,
    m4col_gate.cfg_with, for negative controls)."""
    from mitjax.pkg.seaice.seaice_growth import seaice_growth
    pkc = m.arrays.pkc
    cfg = m.cfg if cfg is None else cfg

    def f(sp, op, sf, ff, exf, st, myTime, myIter):
        return seaice_growth(myTime, myIter, sf, ff, exf, cfg=cfg, sp=sp, op=op, grid=m.arrays.grid, state=st,
                             exfp=pkc["exfp"])
    return jax.jit(f)


def run_growth(m, ds, its=(0, 1, 2), sp=None, op=None, fn=None):
    """{it: {field: differing points}} of SEAICE_GROWTH teacher-forced from I03_reg_ridge vs I04_growth."""
    pkc = m.arrays.pkc
    sp = pkc["sp"] if sp is None else sp
    op = pkc["op"] if op is None else op
    fn = growth_fn(m) if fn is None else fn
    tp = m.prm.time
    out = {}
    for it in its:
        sf, ff, exf, st = inputs_at(m, ds, it, "I04_growth")
        sf2, ff2 = fn(sp, op, sf, ff, exf, st, jnp.float64(tp.startTime + tp.deltaTClock*it),
                      jnp.int32(tp.nIter0 + it))
        vals = {n: sf2[n] for n in SF_I04}
        vals.update({n: getattr(ff2, n) for n in FF_I04})
        out[it] = differing(ds, it, "I04_growth", vals)
    return out


SEAICE_STAGES = ("I00_seaice_begin", "Y01_get_dynforcing", "Y09_ocean_stress", "I01_dynsolver", "I02_advdiff",
                 "I03_reg_ridge", "I04_growth")


def seaice_model_fn(m):
    """jit(f(sp, op, sf, ff, exf, st, myTime, myIter) -> {stage: {name: FArray}}) of SEAICE_MODEL with its probes;
    "P13_seaice_model" holds the returned sf, ff and exf."""
    from mitjax.pkg.seaice.seaice_model import seaice_model
    pkc = m.arrays.pkc

    def f(sp, op, sf, ff, exf, st, myTime, myIter):
        out = {}

        def probe(stage, sf_, ff_, exf_):
            out[stage] = {**exf_, **{n: getattr(ff_, n) for n in ff_.names()}, **sf_}
        sf2, ff2, exf2 = seaice_model(myTime, myIter, sf, ff, exf, cfg=m.cfg, sp=sp, op=op, grid=m.arrays.grid,
                                      state=st, exfp=pkc["exfp"], ex=m.ex, kgeo=None, probe=probe)
        out["P13_seaice_model"] = {**exf2, **{n: getattr(ff2, n) for n in ff2.names()}, **sf2}
        return out
    return jax.jit(f)


def run_seaice_model(m, ds, its=(0, 1, 2), sp=None):
    """{it: {stage: {field: differing points}}} of SEAICE_MODEL teacher-forced from the oracle's fields before
    I00_seaice_begin, compared at every dumped sea-ice stage and P13 (fields the stage dumps and the port carries)."""
    pkc = m.arrays.pkc
    sp = pkc["sp"] if sp is None else sp
    fn = seaice_model_fn(m)
    tp = m.prm.time
    out = {}
    for it in its:
        sf, ff, exf, st = inputs_at(m, ds, it, "I00_seaice_begin")
        res = fn(sp, pkc["op"], sf, ff, exf, st, jnp.float64(tp.startTime + tp.deltaTClock*it),
                 jnp.int32(tp.nIter0 + it))
        out[it] = {stage: differing(ds, it, stage, {n: v for n, v in vals.items() if hasattr(v, "data")})
                   for stage, vals in res.items()}
    return out


# ------------------------------------------------------------------------------------------- the whole step
def run():
    """cube_run_gate.CubeRun of input.thermo (the driver Model on the dumps-on run directory, the oracle dumps)."""
    from mitjax.tests import cube_run_gate as CR
    from mitjax.xla_flags import set_gate_xla_flags
    set_gate_xla_flags()
    return CR.cube_run(*EXP)


def _compare_probes(r, it, probes):
    from mitjax.tests import cube_run_gate as CR
    bad_, ncmp = {}, {}
    for key, vals in probes.items():
        st = (key.split("|")[0], int(key.split("|")[1])) if "|" in key else key
        name = st[0] if isinstance(st, tuple) else st
        if name in r.stages(it):
            if isinstance(vals, dict) and "TICES" in vals:     # the dump holds category 1 (SEAICE_multDim = 1)
                t = vals["TICES"]
                vals = {**vals, "TICES": FArray(t.data[:, :1], t.name, tiled=t.tiled,
                                                _dims=t.dims[:2] + ((t.dims[2][0], 1, 1),))}
            res = CR.compare_stage(r, it, st, vals)
            bad_[name] = {**bad_.get(name, {}), **CR.bad(res)}
            ncmp[name] = ncmp.get(name, 0) + len(res)
    return {s: v for s, v in bad_.items() if v}, ncmp


def steps(r, its=(0, 1, 2), teacher=False, cfg=None, arrays=None):
    """{it: ({stage: bad fields}, {stage: fields compared})}: the driver's whole FORWARD_STEP (m4col_gate.step_fn,
    every dumped stage probed) at the dumped iterations, free (from the Model's own initial carry, chained) or
    teacher-forced (teacher_carry at each iteration)."""
    from mitjax.tests import m4col_gate as C
    m = r.m
    f = C.step_fn(r, cfg)
    a = m.arrays if arrays is None else arrays
    tp = m.prm.time
    carry = m.initial_carry()
    out = {}
    for it in range(max(its) + 1):
        if teacher:
            carry = teacher_carry(m, r.ds, it)
        carry, probes = f(a, carry, jnp.int32(it + 1), jnp.float64(tp.startTime + tp.deltaTClock*it),
                          jnp.int32(tp.nIter0 + it))
        if it in its:
            out[it] = _compare_probes(r, it, probes)
    return out


def whole_run(tag="m4off-whole"):
    """(driver Model, forward result, oracle, run_verdict tuple, output_file_diffs tuple): the 120-step run through
    the run driver (cube_run_gate.whole_run: make_rundir + drivers.run.forward, as `python -m mitjax run`)."""
    from mitjax.tests import advect_gate as ag
    from mitjax.tests import cube_run_gate as CR
    from mitjax.xla_flags import set_gate_xla_flags
    set_gate_xla_flags()
    m, res, o = CR.whole_run(*EXP, tag=tag)
    return m, res, o, ag.run_verdict(*EXP, res, o), CR.output_file_diffs(m, o)


def monitor_records(records):
    """The MONITOR output of a run's STDOUT records: every %MON line and every MONITOR banner ("// Begin MONITOR ...",
    "// End MONITOR ..."), in order."""
    return [r for r in records if "%MON " in r or ("// " in r and " MONITOR " in r)]


def sharded_vs_single(r, nproc=4, nsteps=3, arrays=None):
    """{leaf: points that differ in bits} between `nsteps` of the driver's step (drivers.model.make_step: the whole
    carry incl. the sea-ice and EXF package state) at P=nproc (TileSharding over the 4 tiles of 40x21, jit(shard_map(
    check_vma=True)) with the ShardedExchanger) and at P=1 (the same step, jitted)."""
    from mitjax.drivers.sharded_grad import place_model, tile_specs
    from mitjax.eesupp import exch_maps as EM
    from mitjax.eesupp.shard import TileSharding
    from mitjax.tests import r2_gate as r2
    m = r.m
    a1 = m.arrays if arrays is None else arrays
    sh = TileSharding(EM.load_maps(EXP[0]), nproc)      # the registered map of the 4-tile layout (session 1)
    carry0 = m.initial_carry()

    def body(a, carry, iloop, myTime, myIter):
        carry, myTime, myIter, _ = m.step(a, carry, iloop, myTime, myIter)
        return carry, myTime, myIter
    cs = tile_specs(sh, carry0)
    f4 = sh.shard_map(body, in_specs=(tile_specs(sh, a1), cs, sh.REP, sh.REP, sh.REP), out_specs=(cs, sh.REP, sh.REP))
    f1 = jax.jit(body)
    a4, c4, c1 = place_model(sh, a1), sh.put_tree(carry0), carry0
    t1, it1 = m.start_counters()
    t4, it4 = t1, it1
    for k in range(nsteps):
        c1, t1, it1 = f1(a1, c1, jnp.int32(k + 1), t1, it1)
        c4, t4, it4 = f4(a4, c4, jnp.int32(k + 1), t4, it4)
    return r2.tree_bits_differ(sh.unpad_tree(c4), jax.tree.map(np.asarray, c1))


DT = 3600.0                      # input.thermo/data: deltaT = 3600.


def run_dir_model(tag, overrides=None, links=()):
    """A driver Model of input.thermo on a new run directory (make_rundir, never reused), namelist `overrides`
    (drivers.model.with_namelist), extra files linked into it (m4col_gate.run_dir_model for this experiment)."""
    import os
    import time
    from pathlib import Path
    from mitjax import paths as P
    from mitjax.drivers.model import Model as DriverModel, with_namelist
    from mitjax.drivers.run import load_experiment, make_rundir
    from mitjax.xla_flags import set_gate_xla_flags
    set_gate_xla_flags()
    exp_dir = P.UPSTREAM / "verification" / EXP[0]
    e = load_experiment(exp_dir, EXP[1])
    if overrides:
        e = with_namelist(e, overrides)
    rd = make_rundir(exp_dir, EXP[1], P.RUNS / "tests_m4off" / f"{tag}-{os.getpid()}-{time.time_ns()}")
    for s in links:
        os.symlink(s, rd / Path(s).name)
    return DriverModel(e, rd)


_RESTART = {}


def restart_pair(n=20, k=10):
    """Run A: `n` steps of input.thermo with permanent pickups every `k` steps (pChkptFreq = k*deltaT); cached."""
    if (n, k) not in _RESTART:
        from mitjax.drivers.run import forward
        a = run_dir_model("restartA", {("data", "PARM03", "pChkptFreq"): k*DT,
                                       ("data", "PARM03", "nTimeSteps"): n})
        _RESTART[(n, k)] = (a, forward(a))
    return _RESTART[(n, k)]


def restart_b(tag, mutate=None, n=20, k=10):
    """Run B: nIter0 = k, startTime = k*deltaT, nTimeSteps = n-k from copies of A's pickup.<k> and
    pickup_seaice.<k> (mutated by `mutate(dir)` if given). Returns (A, A's result, B, B's result)."""
    import os
    import time
    from pathlib import Path
    from mitjax import paths as P
    from mitjax.drivers.run import forward
    a, ra = restart_pair(n, k)
    d = P.RUNS / "tests_m4off" / f"{tag}_pickup-{os.getpid()}-{time.time_ns()}"
    d.mkdir(parents=True)
    links = []
    for pre in ("pickup", "pickup_seaice"):
        for suf in ("data", "meta"):
            src = Path(a.rundir) / f"{pre}.{k:010d}.{suf}"
            dst = d / src.name
            dst.write_bytes(src.read_bytes())
            links.append(dst)
    if mutate:
        mutate(d)
    b = run_dir_model(tag, {("data", "PARM03", "pChkptFreq"): k*DT, ("data", "PARM03", "nIter0"): (k, "int"),
                            ("data", "PARM03", "startTime"): k*DT, ("data", "PARM03", "nTimeSteps"): n - k},
                      links)
    return a, ra, b, forward(b)


# ----------------------------------------------------------------------------------------------- gradients
GX_SF = ("HEFF", "AREA", "HSNOW", "TICES")
GX_EXF = ("atemp", "lwdown", "swdown", "aqh", "wspeed", "evap")
GX_FF = ("Qnet", "Qsw")
GX_ST = ("theta", "salt")
GY_SF = ("HEFF", "AREA", "HSNOW", "TICES")
GY_FF = ("Qnet", "Qsw", "EmPmR", "saltFlux")
PT = (0, 13, 13)                 # an interior wet point [tile, j, i] (OLx = OLy = 3: i = j = 10 of tile 1)


def growth_gfun(m, ds, it, smooth=False):
    """(g, x0): g(x) -> {output: array} of SEAICE_GROWTH (jitted) at the oracle's inputs of iteration `it` before
    I04_growth, x the arrays of GX_* (every lane), outputs GY_*. `smooth`: the ice cover replaced on every lane by a
    thick, nearly closed, snow-covered one (HEFF = 1 m, AREA = 0.9, HSNOW = 0.1 m, TICES = 260 K), where every
    MAX/MIN the ice state reaches is decided by a margin far larger than the FD steps (the oracle's own state is
    thin ice, HEFF <= 0.2 m, with a snow-free cover: the sublimation MAX(MIN(r_FWbySublim, 0), 0) sits on its tie)."""
    sf, ff, exf, st = inputs_at(m, ds, it, "I04_growth")
    fn = growth_fn(m)
    pkc = m.arrays.pkc
    tp = m.prm.time
    myTime, myIter = jnp.float64(tp.startTime + tp.deltaTClock*it), jnp.int32(tp.nIter0 + it)
    x0 = {**{n: sf[n].data for n in GX_SF}, **{n: exf[n].data for n in GX_EXF},
          **{n: getattr(ff, n).data for n in GX_FF}, **{n: getattr(st, n).data for n in GX_ST}}
    if smooth:
        for n, v in (("HEFF", 1.0), ("AREA", 0.9), ("HSNOW", 0.1), ("TICES", 260.0)):
            x0[n] = jnp.full_like(x0[n], v)

    def rewrap(old, data):
        return FArray(data, old.name, tiled=old.tiled, _dims=old.dims)

    def g(x):
        sf2 = {**sf, **{n: rewrap(sf[n], x[n]) for n in GX_SF}}
        ex2 = {**exf, **{n: rewrap(exf[n], x[n]) for n in GX_EXF}}
        ff2 = ff.replace(**{n: rewrap(getattr(ff, n), x[n]) for n in GX_FF})
        st2 = st.replace(**{n: rewrap(getattr(st, n), x[n]) for n in GX_ST})
        osf, off = fn(pkc["sp"], pkc["op"], sf2, ff2, ex2, st2, myTime, myIter)
        return {**{n: osf[n].data for n in GY_SF}, **{n: getattr(off, n).data for n in GY_FF}}
    return g, x0


NSTEP_GRAD = 2


def cost_weights(m, seed=17):
    """TEST FIXTURE: weights uniform in [0.5, 1.5) (fixed seed) on the interior wet points of theta and of the sea-ice
    thickness, 0 on halos and land: {"theta": [tile, k, j, i], "HEFF": [tile, j, i]} (shardgrad_gate.weights)."""
    rng = np.random.default_rng(seed)
    mc = np.asarray(m.grid.maskC.data)
    sz = m.cfg.size
    inner = np.zeros(mc.shape)
    inner[..., sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] = 1.0
    return {"theta": jnp.asarray(rng.uniform(0.5, 1.5, mc.shape) * mc * inner),
            "HEFF": jnp.asarray(rng.uniform(0.5, 1.5, mc[:, 0].shape) * mc[:, 0] * inner[:, 0])}


def window(m, nstep=NSTEP_GRAD, it0=1):
    """(f, carry0): f(theta, arrays, carry0) -> (theta, HEFF) after `nstep` whole steps of the driver Model's step
    from the carry at the start of iteration `it0` with its theta replaced (theta: the data of State.theta)."""
    tp = m.prm.time

    def f(theta, a, carry0):
        th = carry0[0].theta
        carry = (carry0[0].replace(theta=FArray(theta, th.name, tiled=th.tiled, _dims=th.dims)),) + carry0[1:]
        t, it = jnp.float64(tp.startTime + tp.deltaTClock*it0), jnp.int32(tp.nIter0 + it0)
        for k in range(nstep):
            carry, t, it, _ = m.step(a, carry, jnp.int32(it0 + 1 + k), t, it)
        return carry[0].theta.data, carry[4]["seaice"]["HEFF"].data
    return f


def window_cost(m, f):
    """J(theta, arrays, carry0, w) = sum(w_theta * theta) + sum(w_HEFF * HEFF) after the window, each sum over the
    interior through the exchanger's global sum (ex.global_sum_rl: per-tile chains in Fortran order, then tile 1 +
    tile 2 + ...), so P=1 and P=4 add the same numbers in the same order."""
    sz = m.cfg.size
    J_ = slice(sz.OLy, sz.OLy + sz.sNy)
    I_ = slice(sz.OLx, sz.OLx + sz.sNx)

    def J(theta, a, carry0, w):
        th, heff = f(theta, a, carry0)
        x = (w["theta"] * th)[:, :, J_, I_]
        h = (w["HEFF"] * heff)[:, J_, I_]
        return a.ex.global_sum_rl(x.reshape(x.shape[0], -1, sz.sNx)) + a.ex.global_sum_rl(h)
    return J


def sharded_grad_vs_single(m, ds, nproc=4):
    """(J1, g1, J4, g4 unpadded, padding nonzero count): dJ/dtheta of window_cost at P=1 (jit) and at P=nproc
    (jit(shard_map(check_vma=True)) with the ShardedExchanger, jax.value_and_grad inside the shard_map body)."""
    from mitjax.drivers.sharded_grad import place_model, tile_specs
    from mitjax.eesupp import exch_maps as EM
    from mitjax.eesupp.shard import TileSharding
    J = window_cost(m, window(m))
    carry0 = teacher_carry(m, ds, 1)
    w = cost_weights(m)
    th0 = carry0[0].theta.data
    J1, g1 = jax.jit(jax.value_and_grad(J))(th0, m.arrays, carry0, w)
    sh = TileSharding(EM.load_maps(EXP[0]), nproc)
    wspec = {"theta": sh.TILES, "HEFF": sh.TILES}
    f4 = sh.shard_map(jax.value_and_grad(J), in_specs=(sh.TILES, tile_specs(sh, m.arrays), tile_specs(sh, carry0),
                                                       wspec), out_specs=(sh.REP, sh.TILES))
    put = lambda x: sh.put_tree(FArray(x, "x", tiled=True, _dims=carry0[0].theta.dims)).data if x.ndim == 4 else \
        sh.put_tree(FArray(x, "x", tiled=True, _dims=carry0[4]["seaice"]["HEFF"].dims)).data
    J4, g4 = f4(put(th0), place_model(sh, m.arrays), sh.put_tree(carry0), {k: put(v) for k, v in w.items()})
    g4 = np.asarray(g4)
    nT = g1.shape[0]
    return float(J1), np.asarray(g1), float(J4), g4[:nT], int(np.count_nonzero(g4[nT:]))
