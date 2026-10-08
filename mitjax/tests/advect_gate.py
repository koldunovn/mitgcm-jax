"""Helpers of the R3 gates (advect_xy, advect_xz; plan Task 14): the R2 set-up of mitjax/tests/r2_gate.py with the
r* parameters (ini_parms_rstar: doResetHFactors for the NONLIN_FRSURF build of advect_xz) and GAD_INIT_VARIA (the
SOM moments, PACKAGES_INIT_VARIABLES -> GAD_INIT_VARIA, packages_init_variables.F:191-198), the substep stages the
advect oracles dump, and the later steps of the jaxdump3 runs (reference/reference_runs.py JD3).

The SOM moments are not dumped: where GAD_SOM_ADVECT replaces GAD_ADVECTION (temp_integrate.F:259-269) the oracle
writes no T10/T20 record, so the SOM tendency is gated through the following stages (T11-T13 / T21-T23, T02/T03,
S15, S16) and the moments themselves through every later step (they feed the next step's tendency).
"""

from functools import lru_cache

import jax
import jax.numpy as jnp

from mitjax.tests import grid_gate as gg
from mitjax.tests import r1_gate as rg
from mitjax.tests import r2_gate as r2

# the variants of plan Task 14 that this lane gates (input.nlfs: GO lane, mitjax/tests/test_r3_nlfs.py)
VARIANTS = (("advect_xy", "input"), ("advect_xy", "input.ab3_c4"), ("advect_xz", "input"), ("advect_xz", "input.pqm"))

# every probe stage FORWARD_STEP and its callees emit (mitjax/model/src/forward_step.py, thermodynamics.py,
# temp_integrate.py, salt_integrate.py, do_oceanic_phys.py); a variant compares those its oracle dumps
PROBES = ("S00_begin", "S02_load_fields", "P02_rho_sigma_ivdc", "P03_mxlayer", "S04_oceanic_phys",
          "T10_temp_adv", "T11_temp_gT", "T12_temp_step", "T13_temp_impl", "T02_temp_integrate",
          "T20_salt_adv", "T21_salt_gS", "T22_salt_step", "T23_salt_impl", "T03_salt_integrate",
          "S05_thermodynamics_sync", "S06_dynamics", "S09_solve_for_pressure", "S10_momentum_correction",
          "S11_integr_continuity", "S15_tracers_correction", "S16_blocking_exchanges")


class Model(r2.Model):
    """r2_gate.Model (INI_PARMS -> INITIALISE_VARIA with the tracer parameters) + ini_parms_rstar + GAD_INIT_VARIA."""

    def __init__(self, exp, inp):
        from mitjax.model.src.ini_parms_rstar import ini_parms_rstar
        from mitjax.pkg.generic_advdiff.gad_init_varia import gad_init_varia
        super().__init__(exp, inp)
        self.params = ini_parms_rstar(self.e, self.params)
        # CG2D.h is State under NONLIN_FRSURF (GO lane, mitjax/model/state.py NLFS_CG2D): r2_gate.Model passes
        # INI_CG2D's CG2D.h to INITIALISE_VARIA, which also runs UPDATE_CG2D (initialise_varia.F:325-327) when
        # nonlinFreeSurf > 2 (advect_xz/input.nlfs); no override here (it would undo that update)
        tp = self.prm.time
        # PACKAGES_INIT_VARIABLES (initialise_varia.F:263): GAD_INIT_VARIA under useGAD (packages_init_variables.F
        # :191-198); DIAGNOSTICS_INIT_VARIA (useDiagnostics) writes only diagnostics state (not ported)
        self.state0 = gad_init_varia(self.state0, cfg=self.cfg, params=self.params, startTime=tp.startTime,
                                     baseTime=tp.baseTime, nIter0=tp.nIter0,
                                     pickupSuff=getattr(self.prm.init, "pickupSuff", " "))
        self.stages = tuple(st for st in PROBES if st in self.ds.stages(self.it0))


@lru_cache(maxsize=None)
def model(exp, inp):
    return Model(exp, inp)


def oracle3(exp, inp):
    """(DumpSet, iterations) of the registered jaxdump3 dumps-on run (steps 0, 1, 2 and the later MONITOR step)."""
    from mitjax.io.dump import DumpSet
    reg = gg._registry()
    runs = [r for r in reg.RUNS if (r["exp"], r["input"], r["kind"]) == (exp, inp, "jdon3")]
    if len(runs) != 1:
        raise FileNotFoundError(f"{exp}/{inp}: {len(runs)} registered jaxdump3 runs")
    ds = DumpSet(reg.run_top(runs[0]) / "dumps")
    return ds, tuple(ds.iterations())


def run_steps(m, params, n, stages=None, ds=None, at=None, step_fn=None):
    """n steps from the initial state, every probe stage compared with the oracle (`ds`, default the jdon run) at the
    steps (0-based start iteration offsets) in `at` (default all). Returns ([{stage: compare_stage}] per compared
    step, final State)."""
    stages = m.stages if stages is None else stages
    ds = m.ds if ds is None else ds
    fn = m.step_fn(stages) if step_fn is None else step_fn
    state, ff, phi0, t, it = m.state0, m.ff, m.phi0surf, m.prm.time.startTime, m.prm.time.nIter0
    res = []
    for k in range(n):
        state, ff, phi0, t, it, out, pr = fn(m.grid, params, m.eos, m.cg2dh, m.cg2d_params, m.ex, state, ff, phi0,
                                             jnp.int32(k + 1), jnp.float64(t), jnp.int32(it))
        if at is None or k in at:
            itk = m.prm.time.nIter0 + k
            res.append({st: rg.compare_stage(ds, itk, st, r2.stage_values(st, pr[st])) for st in stages
                        if st in pr})
    return res, state


def bad(r):
    return r2.bad(r)


def ndiff(res):
    """Number of points differing in bit pattern per compared step and stage (summed over the stage's fields)."""
    return [{st: sum(v[3] for v in r[st].values()) for st in r} for r in res]


__all__ = ["VARIANTS", "PROBES", "Model", "model", "oracle3", "run_steps", "bad", "ndiff"]


def out_dir(tag):
    """A new directory for one whole run of this lane's gates (never reused, never removed)."""
    import os
    import time
    from mitjax import paths
    return paths.RUNS / "tests_r3_advection" / f"{tag}-{os.getpid()}-{time.time_ns()}"


def whole_run(exp, inp, tag="run"):
    """The whole run through the run driver (mitjax/drivers: load_experiment, make_rundir, Model, forward) with the
    pickups its schedule writes (end of run: pickup.ckptA, pickup_somT/S.ckptA). Returns (driver Model, forward
    result, monitor_gate Oracle of the jdon run)."""
    from mitjax import paths
    from mitjax.drivers.model import Model as DriverModel
    from mitjax.drivers.run import forward, load_experiment, make_rundir
    from mitjax.tests import monitor_gate as mg
    exp_dir = paths.UPSTREAM / "verification" / exp
    e = load_experiment(exp_dir, inp)
    rundir = make_rundir(exp_dir, inp, out_dir(f"{exp}-{inp}-{tag}"))
    m = DriverModel(e, rundir)
    res = forward(m)
    return m, res, mg.oracle(exp, inp)


def run_verdict(exp, inp, res, o):
    """(differing %MON records per kind, testreport rows [(name, ours vs results/, yardstick, ours vs oracle)],
    number of %MON blocks) of a whole run against the oracle STDOUT: every MONITOR record (%MON lines and their
    banners) in order, and tools/testreport_jax.py digits."""
    import importlib.util as iu
    from pathlib import Path
    from mitjax import paths
    kinds = {"mon": lambda r: "%MON " in r, "banner": lambda r: "MONITOR dynamic field statistics" in r}
    k0 = next(n for n, r in enumerate(o.raw) if "Begin MONITOR dynamic field statistics" in r) - 1
    diffs = {}
    for kind, sel in kinds.items():
        a, b = [r for r in res.records if sel(r)], [r for r in o.raw[k0:] if sel(r)]
        diffs[kind] = sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b)) + (0 if a else 1)
    nblocks = sum("%MON time_tsnumber" in r for r in res.records)
    out = Path(out_dir(f"{exp}-{inp}-records")) / "output.txt"
    out.parent.mkdir(parents=True)
    out.write_text("\n".join(res.records) + "\n")
    spec = iu.spec_from_file_location("_trj", paths.REPO / "tools" / "testreport_jax.py")
    trj = iu.module_from_spec(spec)
    spec.loader.exec_module(trj)
    ours_res = trj.compare(str(out), exp, inp)
    yard = trj.compare(str(o.stdout_path), exp, inp)
    vs_oracle = trj.compare(str(out), exp, inp, reference=str(o.stdout_path))
    rows = [(a.name, a.digits, y.digits, s.digits) for a, y, s in
            zip(ours_res.run.variables, yard.run.variables, vs_oracle.run.variables)]
    return diffs, rows, nblocks


def pickup_diffs(m, o):
    """{file: identical?} for every pickup file the oracle run directory holds (data and meta, byte for byte against
    ours in the run's rundir), and the files we wrote that the oracle did not."""
    import filecmp
    theirs = sorted(p.name for p in o.stdout_path.parent.iterdir() if p.name.startswith("pickup"))
    ours = sorted(p.name for p in m.rundir.iterdir() if p.name.startswith("pickup"))
    res = {f: (f in ours and filecmp.cmp(m.rundir / f, o.stdout_path.parent / f, shallow=False)) for f in theirs}
    extra = sorted(set(ours) - set(theirs))
    return res, extra


def sharded_step_fn(m, sh):
    """r2_gate.sharded_step_fn with FORWARD_STEP's current `out` dict (flow, cg2d, and rstar only when CALC_R_STAR
    runs); the out_specs follow the output's actual structure (jax.eval_shape of the P=1 step), so a variant without
    r* (advect_xy) and one with it (advect_xz.nlfs) both fit."""
    from mitjax.model.src.forward_step import forward_step
    cfg, fp = m.cfg, m.fp
    g4, eos4, cg2dh4 = sh.put_tree(m.grid), sh.put_tree(m.eos), sh.put_tree(m.cg2dh)
    prm4, cgp4 = sh.put_tree(m.params), sh.put_tree(m.cg2d_params)
    st0, ff0, ph0 = sh.put_tree(m.state0), sh.put_tree(m.ff), sh.put_tree(m.phi0surf)

    def body(grid, params, eos, cg2dh, cg2d_params, ex, state, ff, phi0surf, iloop, myTime, myIter):
        return forward_step(iloop, myTime, myIter, cfg=cfg, grid=grid, params=params, fp=fp, eos=eos, cg2dh=cg2dh,
                            cg2d_params=cg2d_params, state=state, ff=ff, phi0surf=phi0surf, ex=ex)

    T, R = sh.TILES, sh.REP
    sp = lambda tree: r2.specs(sh, tree)                                     # noqa: E731
    in_specs = (sp(m.grid), sp(m.params), sp(m.eos), sp(m.cg2dh), sp(m.cg2d_params), T, sp(m.state0), sp(m.ff),
                sp(m.phi0surf), R, R, R)
    out1 = jax.eval_shape(lambda st, f, ph: body(m.grid, m.params, m.eos, m.cg2dh, m.cg2d_params, m.ex, st, f, ph,
                                                    jnp.int32(1), jnp.float64(0.0), jnp.int32(0))[5],
                          m.state0, m.ff, m.phi0surf)
    out_specs = (sp(m.state0), sp(m.ff), sp(m.phi0surf), R, R, sp(out1))
    f = sh.shard_map(body, in_specs=in_specs, out_specs=out_specs)

    def step_fn(k, state, ff, phi0, t, it):
        return f(g4, prm4, eos4, cg2dh4, cgp4, sh.ex, state, ff, phi0, jnp.int32(k + 1), jnp.float64(t),
                 jnp.int32(it))
    return step_fn, (st0, ff0, ph0)
