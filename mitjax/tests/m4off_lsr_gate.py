"""Helpers of the offline_exf_seaice/input.dyn_lsr gates (M4 step 4, lane M4OFF session 3: the C-grid dynamics
with the LSR solver).

Oracle: lane A's registered dumps-on run job27855986-jdon of input.dyn_lsr (same build as input.thermo,
offline_exf_seaice-code-63cdc0b-704fd6b-jaxdump), iterations 0, 1, 2; the dump stages of SEAICE_DYNSOLVER
Y01_get_dynforcing, Y02_ice_strength (after SEAICE_CALC_ICE_STRENGTH), Y04_solver_inputs (before the solver),
Y06_lsr (after SEAICE_LSR), Y09_ocean_stress. Kernels are teacher-forced from the oracle's fields before their stage
(m4off_gate.inputs_at, which is generic in the Model)."""

import functools

import jax
import jax.numpy as jnp

from mitjax.tests import m4off_gate as G

EXP = ("offline_exf_seaice", "input.dyn_lsr")
JDON = "job27855986-jdon"
DYN_STAGES = ("Y02_ice_strength", "Y04_solver_inputs", "Y06_lsr")


@functools.lru_cache(maxsize=None)
def model():
    """(Model, DumpSet) of input.dyn_lsr from the oracle's dumps-on run directory."""
    from mitjax import paths
    from mitjax.config.params import load
    from mitjax.drivers.model import Model
    from mitjax.io.dump import DumpSet
    from mitjax.xla_flags import set_gate_xla_flags
    set_gate_xla_flags()
    top = paths.REFERENCE_RUNS / EXP[0] / EXP[1] / JDON
    return Model(load(*EXP), top / "rundir"), DumpSet(top / "dumps")


def clock(m, it):
    tp = m.prm.time
    return jnp.float64(tp.startTime + tp.deltaTClock*it), jnp.int32(tp.nIter0 + it)


def lsr_fn(m, sp=None):
    """jit(f(sp, op, sf, st, myTime, myIter) -> (sf, out)) of SEAICE_LSR (through the forward-only wrapper)."""
    from mitjax.ad.seaice_lsr_rule import seaice_lsr_forward_only

    def f(sp, op, sf, st, myTime, myIter):
        return seaice_lsr_forward_only(myTime, myIter, sf, cfg=m.cfg, sp=sp, op=op, grid=m.arrays.grid, state=st,
                                       ex=m.ex)
    return jax.jit(f)


def run_lsr(m, ds, its=(0, 1, 2), sp=None, fn=None):
    """{it: ({field: differing points} at Y06_lsr, out)}: SEAICE_LSR teacher-forced from the oracle's fields before
    Y06_lsr (Y04_solver_inputs)."""
    pkc = m.arrays.pkc
    sp = pkc["sp"] if sp is None else sp
    fn = lsr_fn(m) if fn is None else fn
    res = {}
    for it in its:
        sf, ff, exf, st = G.inputs_at(m, ds, it, "Y06_lsr")
        sf2, out = fn(sp, pkc["op"], sf, st, *clock(m, it))
        res[it] = (G.differing(ds, it, "Y06_lsr", {n: v for n, v in sf2.items() if hasattr(v, "data")}),
                   jax.tree_util.tree_map(lambda a: jax.device_get(a), out))
    return res


def lsr_lines(out, ipass_max=None):
    """The STDOUT lines SEAICE_LSR prints for one call (seaice_lsr.F:767-773, :1015-1035), from its `out`."""
    from mitjax.pkg.seaice.seaice_lsr import lsr_stdout_lines
    return lsr_stdout_lines(out)


def oracle_lsr_lines(it):
    """The oracle's SEAICE_LSR lines of iteration `it` (the 5*nonLinIterMax lines after the it-th block start)."""
    from mitjax import paths
    top = paths.REFERENCE_RUNS / EXP[0] / EXP[1] / JDON / "rundir" / "output.txt"
    lines = [ln.rstrip("\n") for ln in open(top) if ln.startswith(" SEAICE_LSR")]
    per = len(lines) // 12
    return lines[it*per:(it+1)*per]


# ------------------------------------------------------------------------------------------- the whole step / run
def run():
    """cube_run_gate.CubeRun of input.dyn_lsr (the driver Model on the dumps-on run directory, the oracle dumps); the
    whole-step helpers of m4off_gate (steps, sharded_vs_single) take it unchanged (same experiment, same tiles)."""
    from mitjax.tests import cube_run_gate as CR
    from mitjax.xla_flags import set_gate_xla_flags
    set_gate_xla_flags()
    return CR.cube_run(*EXP)


def whole_run(tag="m4off-lsr-whole"):
    """(driver Model, forward result, oracle, run_verdict tuple, output_file_diffs tuple): the 12-step run through the
    run driver (cube_run_gate.whole_run: make_rundir + drivers.run.forward, as `python -m mitjax run`)."""
    from mitjax.tests import advect_gate as ag
    from mitjax.tests import cube_run_gate as CR
    from mitjax.xla_flags import set_gate_xla_flags
    set_gate_xla_flags()
    m, res, o = CR.whole_run(*EXP, tag=tag)
    return m, res, o, ag.run_verdict(*EXP, res, o), CR.output_file_diffs(m, o)


def lsr_records(records):
    """The SEAICE_LSR lines of a run's STDOUT records, in order."""
    return [r for r in records if r.startswith(" SEAICE_LSR")]


DT = 1800.0                      # input.dyn_lsr/data: deltaT = 1800.0


def run_dir_model(tag, overrides=None, links=()):
    """A driver Model of input.dyn_lsr on a new run directory (as m4off_gate.run_dir_model for input.thermo)."""
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


def restart(n=12, k=6):
    """Run A: n steps with permanent pickups every k steps; run B: nIter0 = k from copies of A's pickup.<k> and
    pickup_seaice.<k>, n-k steps. Returns (A, A's result, B, B's result)."""
    import os
    import time
    from mitjax import paths as P
    from mitjax.drivers.run import forward
    a = run_dir_model("lsr-restartA", {("data", "PARM03", "pChkptFreq"): k*DT, ("data", "PARM03", "nTimeSteps"): n})
    ra = forward(a)
    d = P.RUNS / "tests_m4off" / f"lsr-restartB_pickup-{os.getpid()}-{time.time_ns()}"
    d.mkdir(parents=True)
    links = []
    from pathlib import Path
    for pre in ("pickup", "pickup_seaice"):
        for suf in ("data", "meta"):
            src = Path(a.rundir) / f"{pre}.{k:010d}.{suf}"
            dst = d / src.name
            dst.write_bytes(src.read_bytes())
            links.append(dst)
    b = run_dir_model("lsr-restartB", {("data", "PARM03", "pChkptFreq"): k*DT,
                                       ("data", "PARM03", "nIter0"): (k, "int"),
                                       ("data", "PARM03", "startTime"): k*DT,
                                       ("data", "PARM03", "nTimeSteps"): n - k}, links)
    return a, ra, b, forward(b)
