"""Helpers of the 1D_ocean_ice_column/input wiring gates (M4 step 3, lane M4COL session 3).

The driver Model of the column (pkg/cal + pkg/exf + pkg/seaice wired: Model._cal_exf_seaice, FORWARD_STEP's
LOAD_FIELDS_DRIVER -> EXF_GETFORCING and DO_OCEANIC_PHYS -> SEAICE_MODEL, EXTERNAL_FORCING_SURF's sIceLoad) runs
FORWARD_STEP up to P01_external_forcing_surf (`until`): KPP_CALC, the next routine, raises for this build's KPP
options (pkg/kpp/KPP_OPTIONS.h: KPP_SMOOTH_SHSQ, KPP_SMOOTH_DBLOC, no KPP_ESTIMATE_UREF; SHORTWAVE_HEATING), the
blocker of the whole-run gate (lane M4COL session 3).

Iteration 0 starts from the Model's own initial carry (INITIALISE_VARIA + the package set-up); iterations 1 and 2
are teacher-forced: the carry of the Model with the oracle's values at the start of that step (`teacher_carry`).
"""

import functools

import jax.numpy as jnp
import numpy as np

from mitjax.farray import FArray

EXP = ("1D_ocean_ice_column", "input")
JDON = "job27856057-jdon"
UNTIL = "P01_external_forcing_surf"
# the dump stages FORWARD_STEP probes up to P01 (S01: NONLIN_FRSURF only, not dumped in this build)
STAGES = ("S00_begin", "X01_exf_getffields", "X02_exf_radiation", "X03_exf_wind", "X04_exf_bulkformulae",
          "X05_exf_hflux_sflux", "X06_exf_mapfields", "S02_load_fields", "I00b_seaice_begin", "I01b_dynsolver",
          "I03_reg_ridge", "I04_growth", "P13_seaice_model", UNTIL)


@functools.lru_cache(maxsize=None)
def model():
    """(Model, DumpSet) of the column, from the oracle's dumps-on run directory."""
    from mitjax import paths
    from mitjax.config.params import load
    from mitjax.drivers.model import Model
    from mitjax.io.dump import DumpSet
    top = paths.REFERENCE_RUNS / EXP[0] / EXP[1] / JDON
    return Model(load(*EXP), top / "rundir"), DumpSet(top / "dumps")


def _last(ds, it, name):
    """[tile, ...] of `name` at its last dump of the iterations before `it` (None if never dumped)."""
    for i in range(it - 1, -1, -1):
        keys = set(ds.keys(i))
        have = [st for st in ds.stages(i) if (i, st, name) in keys]
        if have:
            return np.asarray(ds.field(i, have[-1], name))
    return None


def _like(old, data):
    a = np.asarray(data)
    if old.data.ndim == a.ndim - 1:              # a 2-D field dumped with a k axis of 1
        a = a[:, 0]
    return FArray(jnp.asarray(a.reshape(old.data.shape), old.data.dtype), old.name, tiled=old.tiled, _dims=old.dims)


def teacher_carry(m, ds, it):
    """The Model's carry with the oracle's state at the start of iteration `it`: every State field S00_begin of `it`
    dumps; FFIELDS.h, phi0surf, EXF_FIELDS.h, SEAICE.h and KPP.h fields at their last dump before it (the record arrays
    <fld>0/1 stay ours: the preload replaces them every step). Iteration 0: the initial carry itself."""
    carry = m.initial_carry()
    if it == 0:
        return carry
    from mitjax.tests import r1_gate as rg
    state, ff, phi0surf, flow, pk = carry
    keys = set(ds.keys(it))
    sv = rg.state_fields(state)
    rep = {n: _like(sv[n], ds.field(it, "S00_begin", n)) for n in sv
           if (it, "S00_begin", n) in keys and hasattr(sv[n], "data")}
    state = state.replace(**rep)
    ffr = {}
    for n in ff.names():
        a = _last(ds, it, n)
        if a is not None:
            ffr[n] = _like(getattr(ff, n), a)
    ff = ff.replace(**ffr)
    a = _last(ds, it, "phi0surf")
    if a is not None:
        phi0surf = _like(phi0surf, a)
    pk = dict(pk)
    for key in ("exf", "seaice", "kpp"):        # kpp: KPP.h (the points KPP_CALC does not write keep the prior)
        if key not in pk:
            continue
        d = dict(pk[key])
        for n, v in d.items():
            if key == "exf" and n[-1:] in ("0", "1"):
                continue
            a = _last(ds, it, n)
            if a is not None and hasattr(v, "data") and v.data.dtype != jnp.int32:
                d[n] = _like(v, a)
        pk[key] = d
    return (state, ff, phi0surf, flow, pk)


def front_step(m):
    """jit(f)(arrays, carry, iloop, myTime, myIter) -> (pk, out, probes): FORWARD_STEP up to UNTIL with every probe
    recorded (as goadk_model_gate.step_fn; the flow of THERMODYNAMICS does not exist yet)."""
    import jax
    from mitjax.model.src.forward_step import forward_step
    cfg, fp, pks = m.cfg, m.fp, m.pks

    def f(a, carry, iloop, myTime, myIter):
        probes = {}

        def probe(stage, values):
            probes[stage] = values
        state, ff, phi0surf = carry[:3]
        pk = carry[4]
        probes["_in"] = dict(ff=ff, phi0surf=phi0surf, gm=None, grid=a.grid)
        state, ff, phi0surf, myTime, myIter, out = forward_step(
            iloop, myTime, myIter, cfg=cfg, grid=a.grid, params=a.params, fp=fp, eos=a.eos, cg2dh=a.cg2dh,
            cg2d_params=a.cg2d_params, state=state, ff=ff, phi0surf=phi0surf, ex=a.ex, pk=pk, pkc=a.pkc,
            pks=pks, probe=probe, until=UNTIL)
        return out.get("pk"), {k: v for k, v in out.items() if k not in ("flow", "pk", "cg2d")}, probes
    return jax.jit(f)


def run_front(m, ds, its=(0, 1, 2), arrays=None, f=None):
    """[(it, compare_step result, out)]: the front of FORWARD_STEP at the dumped iterations (teacher_carry)."""
    from mitjax.tests import goadk_model_gate as M
    f = front_step(m) if f is None else f
    a = m.arrays if arrays is None else arrays
    tp = m.prm.time
    res = []
    for it in its:
        carry = teacher_carry(m, ds, it)
        myTime = jnp.float64(tp.startTime + tp.deltaTClock*it)
        myIter = jnp.int32(tp.nIter0 + it)
        _, o, probes = f(a, carry, jnp.int32(it + 1), myTime, myIter)
        res.append((it, M.compare_step(m, ds, it, probes), o))
    return res


def oracle_blocks(prefix):
    """{tsnumber: [records]} of the oracle STDOUT's monitor block whose records start with `%MON <prefix>`."""
    from mitjax import paths
    lines = (paths.REFERENCE_RUNS / EXP[0] / EXP[1] / JDON.replace("jdon", "plain") / "rundir"
             / "output.txt").read_text().splitlines()
    out, cur, ts = {}, None, None
    for ln in lines:
        body = ln.split(") ", 1)[1] if ln.startswith("(PID") else ln
        if body.startswith(f"%MON {prefix}"):
            if body.startswith(f"%MON {prefix}tsnumber") or body.startswith(f"%MON {prefix}_tsnumber"):
                ts = int(body.split("=")[1])
                cur = out.setdefault(ts, [])
            if cur is not None:
                cur.append(ln)
    return out


# ---------------------------------------------------------------------------------------------------------------
# lane M4COL session 4: the whole step (pkg/kpp arms of this build, JMD95Z FIND_ALPHA/FIND_BETA, GAD DST3)

class CppPlant:
    """A planted CPP view (negative controls only): `flag(name, header)` is False for the names in `off`, True for
    the names in `on`, the build's value otherwise; every other attribute is the build's."""

    def __init__(self, cpp, off=(), on=()):
        self._cpp, self._off, self._on = cpp, frozenset(off), frozenset(on)

    def flag(self, name, *a, **k):
        if name in self._off:
            return False
        if name in self._on:
            return True
        return self._cpp.flag(name, *a, **k)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._off:
            return False
        if name in self._on:
            return True
        return getattr(self._cpp, name)


def cfg_with(m, off=(), on=()):
    import dataclasses
    return dataclasses.replace(m.cfg, cpp=CppPlant(m.cfg.cpp, off, on))


def column_run():
    """cube_run_gate.CubeRun of the column (the driver Model on the dumps-on run directory, the oracle dumps)."""
    from mitjax.tests import cube_run_gate as CR
    return CR.cube_run(*EXP)


def step_fn(r, cfg=None):
    """jit(step)(arrays, carry, iloop, myTime, myIter) -> (carry, probes): the driver's whole FORWARD_STEP with every
    dumped stage probed (cs32_gate.probed_step_fn), with a planted `cfg` if given (negative controls)."""
    import jax
    from mitjax.farray import FArray
    from mitjax.model.src.forward_step import forward_step
    m = r.m
    cfg = m.cfg if cfg is None else cfg
    fp, pks = m.fp, m.pks
    stages = tuple(sorted({s for i in r.its for s in r.stages(i)}))

    def step(a, carry, iloop, myTime, myIter):
        probes = {}

        def probe(stage, values):
            name = stage[0] if isinstance(stage, tuple) else stage
            if name in stages:
                probes[f"{stage[0]}|{stage[1]}" if isinstance(stage, tuple) else stage] = values
        state, ff, phi0surf = carry[:3]
        state, ff, phi0surf, myTime, myIter, out = forward_step(
            iloop, myTime, myIter, cfg=cfg, grid=a.grid, params=a.params, fp=fp, eos=a.eos, cg2dh=a.cg2dh,
            cg2d_params=a.cg2d_params, state=state, ff=ff, phi0surf=phi0surf, ex=a.ex, probe=probe, pk=carry[4],
            pkc=a.pkc, pks=pks)
        flow = tuple(FArray(f.data, c.name, tiled=c.tiled, _dims=c.dims) for f, c in zip(out["flow"], carry[3]))
        return (state, ff, phi0surf, flow, out["pk"]), probes
    return jax.jit(step)


def teacher_step(r, it, f=None, cfg=None):
    """({stage: bad fields}, {stage: fields compared}): the whole step of iteration `it` from teacher_carry (the
    oracle's state at the start of `it`; the Model's own initial carry at it = 0), every dumped stage compared on
    every point (cube_run_gate.compare_stage, bit patterns)."""
    from mitjax.tests import cube_run_gate as CR
    m = r.m
    f = step_fn(r, cfg) if f is None else f
    tp = m.prm.time
    carry = teacher_carry(m, r.ds, it)
    _, probes = f(m.arrays, carry, jnp.int32(it + 1), jnp.float64(tp.startTime + tp.deltaTClock*it),
                  jnp.int32(tp.nIter0 + it))
    bad, ncmp = {}, {}
    for key, vals in probes.items():
        st = (key.split("|")[0], int(key.split("|")[1])) if "|" in key else key
        name = st[0] if isinstance(st, tuple) else st
        if name in r.stages(it):
            res = CR.compare_stage(r, it, st, vals)
            bad[name] = {**bad.get(name, {}), **CR.bad(res)}
            ncmp[name] = ncmp.get(name, 0) + len(res)
    return {s: v for s, v in bad.items() if v}, ncmp


def whole_run(tag="m4col-whole"):
    """(driver Model, forward result, oracle, run_verdict tuple, output_file_diffs tuple): the 10-step run through
    the run driver (cube_run_gate.whole_run: make_rundir + drivers.run.forward, as `python -m mitjax run`)."""
    from mitjax.tests import advect_gate as ag
    from mitjax.tests import cube_run_gate as CR
    m, res, o = CR.whole_run(*EXP, tag=tag)
    return m, res, o, ag.run_verdict(*EXP, res, o), CR.output_file_diffs(m, o)


def run_dir_model(tag, overrides=None, links=()):
    """A driver Model of the column on a new run directory (make_rundir, never reused), namelist `overrides`
    (drivers.model.with_namelist), extra files linked into it."""
    import os
    import time
    from pathlib import Path
    from mitjax import paths as P
    from mitjax.drivers.model import Model as DriverModel, with_namelist
    from mitjax.drivers.run import load_experiment, make_rundir
    exp_dir = P.UPSTREAM / "verification" / EXP[0]
    e = load_experiment(exp_dir, EXP[1])
    if overrides:
        e = with_namelist(e, overrides)
    rd = make_rundir(exp_dir, EXP[1], P.RUNS / "tests_m4col" / f"{tag}-{os.getpid()}-{time.time_ns()}")
    for s in links:
        os.symlink(s, rd / Path(s).name)
    return DriverModel(e, rd)


_RESTART = {}


def restart_pair():
    """Run A: the column's 10 steps with pChkptFreq = 5 steps (pickups at 5 and 10, ocean and sea ice); cached."""
    if "A" not in _RESTART:
        from mitjax.drivers.run import forward
        dt = 3600.0                                                    # data: deltaTClock = 3600.0
        a = run_dir_model("restartA", {("data", "PARM03", "pChkptFreq"): 5*dt})
        _RESTART["A"] = (a, forward(a))
    return _RESTART["A"]


def restart_b(tag, mutate=None):
    """Run B: nIter0 = 5, startTime = 5*deltaT, nTimeSteps = 5 from copies of A's pickup.0000000005 and
    pickup_seaice.0000000005 (mutated by `mutate(dir)` if given). Returns (A, A's result, B, B's result)."""
    import os
    import time
    from pathlib import Path
    from mitjax import paths as P
    from mitjax.drivers.run import forward
    a, ra = restart_pair()
    d = P.RUNS / "tests_m4col" / f"{tag}_pickup-{os.getpid()}-{time.time_ns()}"
    d.mkdir(parents=True)
    links = []
    for pre in ("pickup", "pickup_seaice"):
        for suf in ("data", "meta"):
            src = Path(a.rundir) / f"{pre}.0000000005.001.001.{suf}"
            dst = d / src.name
            dst.write_bytes(src.read_bytes())
            links.append(dst)
    if mutate:
        mutate(d)
    dt = 3600.0
    b = run_dir_model(tag, {("data", "PARM03", "pChkptFreq"): 5*dt, ("data", "PARM03", "nIter0"): (5, "int"),
                            ("data", "PARM03", "startTime"): 5*dt, ("data", "PARM03", "nTimeSteps"): 5}, links)
    return a, ra, b, forward(b)


def carry_diffs(ca, cb, OL=2):
    """{path: (points differing in bits, of which interior)} over every leaf of two carries (State by name)."""
    import jax
    out = {}
    sa, sb = ca[0], cb[0]
    for n in sa.names():
        x, y = getattr(sa, n), getattr(sb, n)
        if hasattr(x, "data"):
            d = _ndiff(x.data, y.data)
            if d.any():
                out[f"state.{n}"] = (int(d.sum()), int(d[..., OL:-OL, OL:-OL].sum()))
    for key in ("seaice", "exf"):
        for n, x in ca[4][key].items():
            if hasattr(x, "data"):
                d = _ndiff(x.data, cb[4][key][n].data)
                if d.any():
                    out[f"{key}.{n}"] = (int(d.sum()), int(d[..., OL:-OL, OL:-OL].sum()))
    for i, (x, y) in enumerate(zip(jax.tree.leaves(ca[1:4]), jax.tree.leaves(cb[1:4]))):
        d = _ndiff(x, y)
        if d.any():
            out[f"ff_phi0_flow[{i}]"] = (int(d.sum()), int(d[..., OL:-OL, OL:-OL].sum()))
    return out


def _ndiff(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.dtype.kind == "f":
        return x.view(np.int64) != y.view(np.int64)
    return x != y
