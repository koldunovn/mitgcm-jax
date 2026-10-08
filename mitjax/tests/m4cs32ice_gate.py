"""Helpers of the global_ocean.cs32x15/input.seaice gates (M4 step 6, lane M4CS32ICE).

Oracle (session 3, plan decision 13): lane A's FTZ oracle, the registered dumps-on run job27878831-jdon (kind
`ftz_jdon`) of global_ocean.cs32x15/input.seaice: build global_ocean.cs32x15-code-63cdc0b-704fd6b-jaxdump-ftz = the
objects of the standard jaxdump build linked with gfortran 11.2.0's crtfastmath.o (FTZ/DAZ set at program start, no
object recompiled; reference/relink_ftz.sh), iterations 36000-36002 (nIter0 = 36000: the run starts from
pickup.0000036000 and pickup_seaice.0000036000). XLA:CPU computes with FTZ/DAZ (L-CONF-2); against the FTZ oracle
every gate is bitwise, with no band rule. The standard oracle job27855988-jdon (JDON_STD) stays the reference; its
difference from the FTZ oracle is lane A's measurement (reference/ftz_band.py: SEAICE_LSR's uIce/vIce at 36001-36002
and what carries them, all below 2**-1021), and ours against it is that same difference (test_m4cs32ice_seaice.py
test_standard_oracle_difference_is_the_ftz_difference).

Session 2: the Model is built from the experiment as it is (the session-1 stub SEAICEuseMetricTerms .FALSE. is gone:
SEAICE_INIT_FIXED's curvilinear metric arm is ported, gap S2). Teacher forcing (`teacher_carry`, `inputs_at`) takes
each field from the last oracle stage that dumps it, searching only the dumped iterations (the generic m4off/m4col
helpers walk every iteration down to 0, 36000 of them here)."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np

EXP = ("global_ocean.cs32x15", "input.seaice")
JDON = "job27878831-jdon"         # the FTZ oracle (registry kind ftz_jdon)
ORACLE_KIND = "ftz_jdon"
JDON_STD = "job27855988-jdon"     # the standard oracle (registry kind jdon): measurement only
S = "data.seaice"
X = "data.exf"


def _model(namelist=None):
    from mitjax import paths
    from mitjax.config.params import load
    from mitjax.drivers.model import Model, with_namelist
    import os
    from mitjax.xla_flags import set_gate_xla_flags
    if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
        set_gate_xla_flags()        # (a P=N subprocess has set the gate flags with its own device count)
    e = load(*EXP)
    if namelist:
        e = with_namelist(e, namelist)
    return Model(e, paths.REFERENCE_RUNS / EXP[0] / EXP[1] / JDON / "rundir")


@functools.lru_cache(maxsize=None)
def run():
    """r with .m (the Model on the oracle's dumps-on run directory), .ds, .its, .stages (as cube_run_gate.CubeRun)."""
    from types import SimpleNamespace
    from mitjax import paths
    from mitjax.io.dump import DumpSet
    m = _model()
    ds = DumpSet(paths.REFERENCE_RUNS / EXP[0] / EXP[1] / JDON / "dumps")
    return SimpleNamespace(m=m, ds=ds, its=ds.iterations(), stages=ds.stages)


def with_model(r, m):
    from types import SimpleNamespace
    return SimpleNamespace(m=m, ds=r.ds, its=r.its, stages=r.stages)


def planted_model(r, namelist):
    """r with a Model built from the experiment with `namelist` overrides (with_namelist: a bare value for a variable
    the file sets, (value, kind) for an unset one)."""
    return with_model(r, _model(namelist))


class _Planted:
    """The Model with a planted CPP view (negative controls): every attribute is the Model's except cfg."""

    def __init__(self, m, off=(), on=()):
        from mitjax.tests.m4col_gate import CppPlant
        self._m = m
        self.cfg = dataclasses.replace(m.cfg, cpp=CppPlant(m.cfg.cpp, off, on))

    def __getattr__(self, name):
        return getattr(self._m, name)


def planted_cpp(r, off=(), on=()):
    return with_model(r, _Planted(r.m, off, on))


# ---------------------------------------------------------------------------------------------------- oracle values
def last(ds, it, name):
    """[tile, ...] of `name` at its last dump in the dumped iterations before `it` (None if never dumped)."""
    for i in reversed([i for i in ds.iterations() if i < it]):
        keys = set(ds.keys(i))
        have = [st for st in ds.stages(i) if (i, st, name) in keys]
        if have:
            return np.asarray(ds.field(i, have[-1], name))
    return None


def before(ds, it, stage, name):
    """The oracle's `name` at the last stage of iteration `it` before `stage` that dumps it, else at the last dump of
    an earlier dumped iteration (None if never dumped)."""
    keys = set(ds.keys(it))
    sts = ds.stages(it)
    for st in reversed(sts[:sts.index(stage)]):
        if (it, st, name) in keys:
            return np.asarray(ds.field(it, st, name))
    return last(ds, it, name)


def clock(m, it):
    """(myTime, myIter, iloop) at the start of iteration `it` (myTime = startTime + deltaTClock*(it - nIter0))."""
    tp = m.prm.time
    k = it - int(tp.nIter0)
    return jnp.float64(tp.startTime + tp.deltaTClock*k), jnp.int32(it), jnp.int32(k + 1)


def teacher_carry(m, ds, it):
    """The Model's carry with the oracle's state at the start of iteration `it` (m4off_gate.teacher_carry): the State
    fields S00_begin dumps (the others from G00_geometry of `it`: rStarFacNm1C/W/S, etaHnm1 of NONLIN_FRSURF);
    FFIELDS.h, phi0surf, EXF_FIELDS.h and SEAICE.h at their last dump before `it` (the EXF record arrays <fld>0/1
    stay ours: the preload replaces them every step). nIter0: the initial carry itself."""
    from mitjax.tests import m4off_gate as O
    from mitjax.tests import r1_gate as rg
    carry = m.initial_carry()
    if it == int(m.prm.time.nIter0):
        return carry
    state, ff, phi0surf, flow, pk = carry
    keys = set(ds.keys(it))
    sv = rg.state_fields(state)
    rep = {}
    for n, v in sv.items():                 # S00_begin, else G00_geometry (the r* fields of the previous step that
        for st in ("S00_begin", "G00_geometry"):     # UPDATE_R_STAR(.FALSE.) reads: rStarFacNm1C/W/S, etaHnm1)
            if (it, st, n) in keys and hasattr(v, "data"):
                rep[n] = O.like(v, ds.field(it, st, n))
                break
    state = state.replace(**rep)
    ffr = {}
    for n in ff.names():
        a = last(ds, it, n)
        if a is not None:
            ffr[n] = O.like(getattr(ff, n), a)
    ff = ff.replace(**ffr)
    a = last(ds, it, "phi0surf")
    if a is not None:
        phi0surf = O.like(phi0surf, a)
    pk = dict(pk)
    for key in ("exf", "seaice"):
        d = dict(pk[key])
        for n, v in d.items():
            if key == "exf" and n[-1:] in ("0", "1"):
                continue
            a = last(ds, it, n)
            if a is not None and hasattr(v, "data") and v.data.dtype != jnp.int32:
                d[n] = O.like(v, a)
        pk[key] = d
    return (state, ff, phi0surf, flow, pk)


def inputs_at(m, ds, it, stage):
    """(sf, ff, exf, state) of the Model with every field the oracle dumps before `stage` of iteration `it` (each
    from the last stage that dumps it; the State fields too, so that S01's r* hFac reach the sea ice)."""
    from mitjax.tests import m4off_gate as O
    from mitjax.tests import r1_gate as rg
    sf, exf = dict(m.pk0["seaice"]), dict(m.pk0["exf"])
    for d in (sf, exf):
        for n, v in d.items():
            if hasattr(v, "data") and v.data.dtype != jnp.int32:
                a = before(ds, it, stage, n)
                if a is not None:
                    d[n] = O.like(v, a)
    ffr = {}
    for n in m.ff0.names():
        a = before(ds, it, stage, n)
        if a is not None:
            ffr[n] = O.like(getattr(m.ff0, n), a)
    ff = m.ff0.replace(**ffr)
    st = m.state0
    sv = rg.state_fields(st)
    rep = {}
    for n, v in sv.items():
        if hasattr(v, "data"):
            a = before(ds, it, stage, n)
            if a is not None:
                rep[n] = O.like(v, a)
    return sf, ff, exf, st.replace(**rep)


# ---------------------------------------------------------------------------------------------- probed steps
def step_compare(r, it, carry=None, until=None):
    """({stage: bad fields}, {stage: fields compared}) of FORWARD_STEP of iteration `it` from `carry` (default:
    teacher_carry) up to `until`, every dumped stage compared on every point of every tile (cube_run_gate)."""
    from mitjax.tests import cs32_gate as G
    from mitjax.tests import cube_run_gate as C
    m = r.m
    f = jax.jit(G.probed_step_fn(r, tuple(r.stages(it)), until))
    carry = teacher_carry(m, r.ds, it) if carry is None else carry
    t, i, k = clock(m, it)
    probes = f(m.arrays, carry, k, t, i)[4]
    out, ncmp = {}, {}
    for key, vals in probes.items():
        st = (key.split("|")[0], int(key.split("|")[1])) if "|" in key else key
        name = st[0] if isinstance(st, tuple) else st
        res = C.compare_stage(r, it, st, vals)
        out[name] = {**out.get(name, {}), **C.bad(res)}
        ncmp[name] = ncmp.get(name, 0) + len(res)
    return out, ncmp


def front(r, off=(), on=(), until="S02_load_fields", it=None):
    """({(it, stage): bad fields}, {(it, stage): fields compared}) of FORWARD_STEP of iteration `it` (default the
    first, from the Model's own initial carry; later ones teacher-forced) up to `until`, with CPP options planted
    `off` / `on` if given."""
    rr = r if not (off or on) else planted_cpp(r, off, on)
    it = r.its[0] if it is None else it
    out, ncmp = step_compare(rr, it, until=until)
    return {(it, s): v for s, v in out.items()}, {(it, s): v for s, v in ncmp.items()}


def bad_stages(out):
    return {s for (_, s), v in out.items() if v}


def seaice_init_diffs(m, ds, it):
    """{field: differing points} of SEAICE.h / SEAICE_GRID.h after SEAICE_INIT_FIXED + SEAICE_INIT_VARIA (the Model's
    initial sea-ice fields, from pickup_seaice.0000036000) against I00_seaice_begin of iteration `it`."""
    from mitjax.tests import m4lab_gate as L
    sf = m.pk0["seaice"]
    keys = set(ds.keys(it))
    return {n: L.ndiff(ds.field(it, "I00_seaice_begin", n), v) for n, v in sf.items()
            if (it, "I00_seaice_begin", n) in keys and hasattr(v, "data")}


def paths_rundir():
    from mitjax import paths
    return paths.REFERENCE_RUNS / EXP[0] / EXP[1] / JDON / "rundir"


# ---------------------------------------------------------------------------------------- sea-ice kernels (session 2)
def kernels(m, sp=None, op=None):
    """{stage: f(sf, ff, exf, st, myTime, myIter) -> {name: FArray}}: the port's routine that produces each dumped
    sea-ice stage of SEAICE_MODEL (the C-grid build of global_ocean.cs32x15), for teacher forcing from the oracle's
    fields before that stage (`kernel_diffs`). Y02/Y04: SEAICE_DYNSOLVER :145-302 (no free drift: Y04 = Y02);
    Y06: SEAICE_LSR (forward-only wrapper); Y09: SEAICE_OCEAN_STRESS; I01: the clip :388-410 (the only statement
    between Y09 and the end of SEAICE_DYNSOLVER here); I02: SEAICE_ADVDIFF; I03: SEAICE_REG_RIDGE; I04: SEAICE_GROWTH.
    Y02 .. Y09 read TAUX / TAUY of Y01 (sf["TAUX"], sf["TAUY"])."""
    from mitjax.ad.seaice_lsr_rule import seaice_lsr_forward_only
    from mitjax.pkg.seaice import seaice_dynsolver as D
    from mitjax.pkg.seaice.seaice_advdiff import seaice_advdiff
    from mitjax.pkg.seaice.seaice_get_dynforcing import seaice_get_dynforcing
    from mitjax.pkg.seaice.seaice_growth import seaice_growth
    from mitjax.pkg.seaice.seaice_ocean_stress import seaice_ocean_stress
    from mitjax.pkg.seaice.seaice_reg_ridge import seaice_reg_ridge
    pkc = m.arrays.pkc
    sp = pkc["sp"] if sp is None else sp
    op = pkc["op"] if op is None else op
    exfp, g, cfg = pkc["exfp"], m.arrays.grid, m.cfg

    def y01(sf, ff, exf, st, t, i):
        TAUX = _zeros_like(sf["UICE"], "TAUX")                                 # seaice_dynsolver.F:115-116
        TAUY = _zeros_like(sf["VICE"], "TAUY")
        TAUX, TAUY = seaice_get_dynforcing(sf["UICE"], sf["VICE"], sf["AREA"], sf["SIMaskU"], sf["SIMaskV"], TAUX,
                                           TAUY, t, i, cfg=cfg, sp=sp, exfp=exfp, exf=exf, grid=g, op=op, ff=ff)
        return {"TAUX": TAUX, "TAUY": TAUY}

    def y02(sf, ff, exf, st, t, i):
        return D._dynamics_forcing(sf["TAUX"], sf["TAUY"], t, i, sf, ff, cfg=cfg, sp=sp, op=op, grid=g, state=st)

    def y06(sf, ff, exf, st, t, i):
        sf2, out = seaice_lsr_forward_only(t, i, sf, cfg=cfg, sp=sp, op=op, grid=g, state=st, ex=m.ex)
        return {**sf2, "_lsr_out": out}

    def y09(sf, ff, exf, st, t, i):
        ff2 = seaice_ocean_stress(sf["TAUX"], sf["TAUY"], t, i, sf, ff, cfg=cfg, sp=sp, op=op, grid=g, state=st,
                                  ex=m.ex)
        return {n: getattr(ff2, n) for n in ff2.names()}

    def i01(sf, ff, exf, st, t, i):
        return D.clip_velocities(sf, cfg=cfg, sp=sp)

    def i02(sf, ff, exf, st, t, i):
        return seaice_advdiff(sf["UICE"], sf["VICE"], t, i, sf, cfg=cfg, sp=sp, op=op, grid=g)

    def i03(sf, ff, exf, st, t, i):
        return seaice_reg_ridge(t, i, sf, cfg=cfg, sp=sp, op=op)

    def i04(sf, ff, exf, st, t, i):
        sf2, ff2 = seaice_growth(t, i, sf, ff, exf, cfg=cfg, sp=sp, op=op, grid=g, state=st, exfp=exfp)
        return {**sf2, **{n: getattr(ff2, n) for n in ff2.names()}}
    return {"Y01_get_dynforcing": y01, "Y02_ice_strength": y02, "Y04_solver_inputs": y02, "Y06_lsr": y06,
            "Y09_ocean_stress": y09, "I01_dynsolver": i01, "I02_advdiff": i02, "I03_reg_ridge": i03,
            "I04_growth": i04}


def _zeros_like(a, name):
    from mitjax.farray import FArray
    return FArray(jnp.zeros_like(a.data), name, tiled=a.tiled, _dims=a.dims)


def kernel_diffs(r, stage, its=None, sp=None, op=None, fn=None, values=False):
    """{it: {field: differing points}} of the port's routine of `stage` (`kernels`) teacher-forced from the oracle's
    fields before `stage` (`inputs_at`; TAUX/TAUY from Y01 for the dynamics stages), every field the stage dumps,
    every point of every tile incl. halos and cube corners. values=True: {it: (diffs, ours, the oracle's stage)}."""
    from mitjax.tests import m4off_gate as O
    m, ds = r.m, r.ds
    f = jax.jit(kernels(m, sp, op)[stage]) if fn is None else fn
    out = {}
    for it in (r.its if its is None else its):
        sf, ff, exf, st = inputs_at(m, ds, it, stage)
        if stage[:2] == "Y0" and stage != "Y01_get_dynforcing":
            for n in ("TAUX", "TAUY"):
                sf[n] = O.like(sf["UICE"], ds.field(it, "Y01_get_dynforcing", n))
        t, i, _ = clock(m, it)
        res = f(sf, ff, exf, st, t, i)
        d = O.differing(ds, it, stage, {n: v for n, v in res.items() if hasattr(v, "data")})
        out[it] = (d, res) if values else d
    return out


def seaice_model_fn(m):
    """jit(f(sp, op, sf, ff, exf, st, myTime, myIter) -> {stage: {name: FArray}}) of SEAICE_MODEL with every probe;
    "P13_seaice_model" holds the returned SEAICE.h, FFIELDS.h and EXF_FIELDS.h (no SALT_PLUME here)."""
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


def seaice_model_diffs(r, its=None, values=False):
    """{it: {stage: {field: differing points}}} of SEAICE_MODEL teacher-forced from the oracle's fields before
    I00_seaice_begin, every dumped sea-ice stage and P13, every point incl. halos (values=True: {it: {stage: (diffs,
    ours)}})."""
    from mitjax.tests import m4off_gate as O
    m, ds = r.m, r.ds
    pkc = m.arrays.pkc
    fn = seaice_model_fn(m)
    out = {}
    for it in (r.its if its is None else its):
        sf, ff, exf, st = inputs_at(m, ds, it, "I00_seaice_begin")
        t, i, _ = clock(m, it)
        res = fn(pkc["sp"], pkc["op"], sf, ff, exf, st, t, i)
        out[it] = {}
        for stage in ds.stages(it):
            if stage not in res:
                continue
            vals = {n: v for n, v in res[stage].items() if hasattr(v, "data")}
            d = O.differing(ds, it, stage, vals)
            out[it][stage] = (d, vals) if values else d
    return out


def standard_dumps():
    """The standard oracle's dumps (job27855988-jdon, kind jdon): measurement only, never a gate's oracle."""
    from mitjax import paths
    from mitjax.io.dump import DumpSet
    return DumpSet(paths.REFERENCE_RUNS / EXP[0] / EXP[1] / JDON_STD / "dumps")


def bits_differ(ours, ref):
    """Number of points whose float64 bits differ (so -0. vs 0. counts)."""
    o = np.ascontiguousarray(np.asarray(ours, np.float64))
    r_ = np.ascontiguousarray(np.asarray(ref, np.float64).reshape(o.shape))
    return int((o.view(np.int64) != r_.view(np.int64)).sum())


def sharded_vs_single(m, maps, nproc, nsteps=3):
    """({carry leaf: points that differ in bits}, {LSR / cg2d output: steps that differ}, number of carry leaves
    compared, number of replicated outputs compared) between `nsteps` of the
    driver's step (the whole carry incl. EXF_FIELDS.h and SEAICE.h) at P=nproc (TileSharding over the 12 cube tiles,
    `maps` = exch_maps.load_cube_maps, jit(shard_map(check_vma=True)) with the ShardedExchanger) and at P=1 (the same
    step, jitted); m4lab_gate.sharded_vs_single for the cube. The step's replicated outputs (the LSR STDOUT values and
    the solver scalars, keys lsr_* / cg2d / sumRHS ...) are compared too; its tiled EXF monitor snapshot (exfmon_*)
    is not an output here."""
    from mitjax.drivers.sharded_grad import place_model, tile_specs
    from mitjax.eesupp.shard import TileSharding
    from mitjax.tests import r2_gate as r2
    a1 = m.arrays
    sh = TileSharding(maps, nproc)
    carry0 = m.initial_carry()
    t1, it1 = m.start_counters()
    c_shape = jax.eval_shape(m.step, a1, carry0, jnp.int32(1), t1, it1)[3]
    keys = tuple(k for k in (c_shape or {}) if not k.startswith("exfmon_"))

    def body(a, carry, iloop, myTime, myIter):
        carry, myTime, myIter, c = m.step(a, carry, iloop, myTime, myIter)
        return carry, myTime, myIter, {k: c[k] for k in keys}
    cs = tile_specs(sh, carry0)
    f4 = sh.shard_map(body, in_specs=(tile_specs(sh, a1), cs, sh.REP, sh.REP, sh.REP),
                      out_specs=(cs, sh.REP, sh.REP, {k: sh.REP for k in keys}))
    f1 = jax.jit(body)
    a4, c4, c1 = place_model(sh, a1), sh.put_tree(carry0), carry0
    t4, it4 = t1, it1
    od = {}
    for k in range(nsteps):
        c1, t1, it1, o1 = f1(a1, c1, jnp.int32(k + 1), t1, it1)
        c4, t4, it4, o4 = f4(a4, c4, jnp.int32(k + 1), t4, it4)
        for n in keys:
            if np.asarray(o1[n]).tobytes() != np.asarray(o4[n]).tobytes():
                od[n] = od.get(n, 0) + 1
    return (r2.tree_bits_differ(sh.unpad_tree(c4), jax.tree.map(np.asarray, c1)), od, len(jax.tree.leaves(c1)),
            len(keys))


# ------------------------------------------------------------------------------------ the whole run (session 2)
DT = 86400.0                     # input.seaice/data PARM03: deltaTClock = 86400. (nIter0 36000, nTimeSteps 10)


def whole_run(tag="m4cs32ice-whole"):
    """(driver Model, forward result, oracle, run_verdict tuple): the 10 steps of input.seaice through the run driver
    (cube_run_gate.whole_run: make_rundir with the experiment's prepare_run links + drivers.run.forward, as
    `python -m mitjax run`), against lane A's FTZ oracle run (kind ORACLE_KIND)."""
    from mitjax.tests import advect_gate as ag
    from mitjax.tests import cube_run_gate as CR
    from mitjax.xla_flags import set_gate_xla_flags
    set_gate_xla_flags()
    m, res, o = CR.whole_run(*EXP, tag=tag, kind=ORACLE_KIND)
    return m, res, o, ag.run_verdict(*EXP, res, o)


def solver_lines(records):
    """SEAICE_LSR's STDOUT lines and CG2D's (`cg2d` in the record, %MON excluded), without the timing summary."""
    return [r for r in records if "%MON" not in r and "Seconds in section" not in r
            and ("cg2d" in r.lower() or "SEAICE_LSR" in r)]


def run_dir_model(tag, overrides=None, links=()):
    """A driver Model of input.seaice on a new run directory (make_rundir, which runs the experiment's prepare_run;
    never reused), namelist `overrides`, extra files copied in beforehand are linked (`links`)."""
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
    rd = make_rundir(exp_dir, EXP[1], P.RUNS / "tests_m4cs32ice" / f"{tag}-{os.getpid()}-{time.time_ns()}")
    for s in links:
        dst = rd / Path(s).name
        if dst.is_symlink() or dst.exists():
            raise FileExistsError(dst)
        os.symlink(s, dst)
    return DriverModel(e, rd)


def restart_pair(k=5):
    """Run A: the 10 steps from pickup.0000036000 with pChkptFreq = k*deltaT (permanent pickups at 36000+k and
    36010); run B: nIter0 = 36000+k, nTimeSteps = 10-k from copies of A's pickup / pickup_seaice of 36000+k.
    Returns (A, A's result, B, B's result)."""
    import os
    import time
    from pathlib import Path
    from mitjax import paths as P
    from mitjax.drivers.run import forward
    a = run_dir_model("restartA", {("data", "PARM03", "pChkptFreq"): k*DT})
    ra = forward(a)
    d = P.RUNS / "tests_m4cs32ice" / f"restart_pickup-{os.getpid()}-{time.time_ns()}"
    d.mkdir(parents=True)
    links = []
    for pre in ("pickup", "pickup_seaice"):
        for suf in ("data", "meta"):
            src = Path(a.rundir) / f"{pre}.{36000 + k:010d}.{suf}"
            dst = d / src.name
            dst.write_bytes(src.read_bytes())
            links.append(dst)
    b = run_dir_model("restartB", {("data", "PARM03", "pChkptFreq"): k*DT, ("data", "PARM03", "nIter0"): 36000 + k,
                                   ("data", "PARM03", "nTimeSteps"): 10 - k}, links)
    return a, ra, b, forward(b)
