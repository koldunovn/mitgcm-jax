"""Helpers of the advect_cs gates (plan Task 25 part, ADVECT lane): the run driver's Model of advect_cs on the
cubed sphere (6 tiles of 32x32, W2 cube topology, exchanges from load_cube_maps: lane B), FORWARD_STEP under jit with
the substep probes, the oracle's dumps-on run (lane A, job 27832226: steps 0-2) and the whole run.

advect_cs: theta DST3FL (33) through GAD_ADVECTION's cube passes (three passes with corner fills, overlap-only and
interior-only updates), salt SOM (80) through GAD_SOM_ADVECT's cube arm (GAD_SOM_PREP_CS_CORNER /
GAD_SOM_FILL_CS_CORNER), its own ini_vel.F (a solid-body rotation from fCoriG), momStepping = .FALSE.
"""

import os
import time
from functools import lru_cache

import jax
import jax.numpy as jnp

from mitjax import paths
from mitjax.tests import r1_gate as rg

EXP = ("advect_cs", "input")
PROBES = ("S00_begin", "S02_load_fields", "S04_oceanic_phys", "T10_temp_adv", "T11_temp_gT", "T12_temp_step",
          "T13_temp_impl", "T02_temp_integrate", "T20_salt_adv", "T21_salt_gS", "T22_salt_step", "T23_salt_impl",
          "T03_salt_integrate", "S05_thermodynamics_sync", "S06_dynamics", "S09_solve_for_pressure",
          "S10_momentum_correction", "S11_integr_continuity", "S15_tracers_correction", "S16_blocking_exchanges")


def out_dir(tag):
    """A new directory for one run of this lane's gates (never reused, never removed)."""
    return paths.RUNS / "tests_advect_cs" / f"{tag}-{os.getpid()}-{time.time_ns()}"


def new_rundir(exp, inp, tag):
    from mitjax.drivers.run import make_rundir
    exp_dir = paths.UPSTREAM / "verification" / exp
    # the driver's make_rundir runs the experiment's prepare_run (grid_cs32 links) through lane A's make_rundir
    return make_rundir(exp_dir, inp, out_dir(f"{exp}-{inp}-{tag}"))


class CsModel:
    """The driver Model of advect_cs (mitjax/drivers/model.py) + the oracle dump set + a probing step function."""

    def __init__(self, exp=EXP[0], inp=EXP[1], tag="gate"):
        from mitjax.drivers.model import Model
        from mitjax.drivers.run import load_experiment
        from mitjax.tests import grid_gate as gg
        exp_dir = paths.UPSTREAM / "verification" / exp
        e = load_experiment(exp_dir, inp)
        self.exp, self.inp = exp, inp
        self.m = Model(e, new_rundir(exp, inp, tag))
        self.ds, self.it0, _ = gg.oracle(exp, inp)
        self.stages = tuple(st for st in PROBES if st in self.ds.stages(self.it0))

    def step_fn(self, probe_stages=()):
        """jit(step)(arrays, carry, iloop, myTime, myIter) -> (carry, myTime, myIter, out, probes)."""
        from mitjax.model.src.forward_step import forward_step
        m = self.m
        cfg, fp = m.cfg, m.fp

        def step(a, carry, iloop, myTime, myIter):
            probes = {}

            def probe(stage, values):
                if stage in probe_stages:
                    probes[stage] = values
            state, ff, phi0surf, _ = carry
            state, ff, phi0surf, myTime, myIter, out = forward_step(
                iloop, myTime, myIter, cfg=cfg, grid=a.grid, params=a.params, fp=fp, eos=a.eos, cg2dh=a.cg2dh,
                cg2d_params=a.cg2d_params, state=state, ff=ff, phi0surf=phi0surf, ex=a.ex, probe=probe)
            return (state, ff, phi0surf, out["flow"]), myTime, myIter, out, probes
        return jax.jit(step)


@lru_cache(maxsize=None)
def model(exp=EXP[0], inp=EXP[1]):
    return CsModel(exp, inp)


def run_steps(cm, n, stages=None, step=None, mutate=None):
    """n steps from the initial carry; per step {stage: r1_gate.compare_stage} for the probed stages."""
    from mitjax.tests import r2_gate as r2
    stages = cm.stages if stages is None else stages
    fn = cm.step_fn(stages) if step is None else step
    m = cm.m
    carry = m.initial_carry()
    t, it = m.start_counters()
    res = []
    for k in range(n):
        if mutate is not None:
            carry = mutate(k, carry)
        carry, t, it, out, pr = fn(m.arrays, carry, jnp.int32(k + 1), t, it)
        res.append({st: rg.compare_stage(cm.ds, cm.it0 + k, st, r2.stage_values(st, pr[st])) for st in stages
                    if st in pr})
    return res, carry


def bad(r):
    return {k: v for k, v in r.items() if v[0] == "shape" or any(v[1:])}


def sharded_step_fn(cm, sh):
    """FORWARD_STEP of the driver Model under jit(shard_map(check_vma=True)) on the TileSharding `sh`: returns
    (step(k, state, ff, phi0, t, it) on placed arrays, placed (state0, ff0, phi00))."""
    from mitjax.model.src.forward_step import forward_step
    from mitjax.tests import r2_gate as r2
    m = cm.m
    cfg, fp = m.cfg, m.fp
    a = m.arrays
    g4, prm4, eos4, cg2dh4, cgp4 = (sh.put_tree(a.grid), sh.put_tree(a.params), sh.put_tree(a.eos),
                                    sh.put_tree(a.cg2dh), sh.put_tree(a.cg2d_params))
    st0, ff0, ph0 = sh.put_tree(m.state0), sh.put_tree(m.ff0), sh.put_tree(m.phi0surf0)

    def body(grid, params, eos, cg2dh, cg2d_params, ex, state, ff, phi0surf, iloop, myTime, myIter):
        return forward_step(iloop, myTime, myIter, cfg=cfg, grid=grid, params=params, fp=fp, eos=eos, cg2dh=cg2dh,
                            cg2d_params=cg2d_params, state=state, ff=ff, phi0surf=phi0surf, ex=ex)

    T, R = sh.TILES, sh.REP
    sp = lambda tree: r2.specs(sh, tree)                                     # noqa: E731
    flow_spec = sp(m.state0.uVel)
    in_specs = (sp(a.grid), sp(a.params), sp(a.eos), sp(a.cg2dh), sp(a.cg2d_params), T, sp(m.state0), sp(m.ff0),
                sp(m.phi0surf0), R, R, R)
    # FORWARD_STEP's `out` keys (flow, cg2d, and whatever host-side outputs the lanes add): every one replicated
    # except the flow (tiled); the structure from an abstract evaluation of the unsharded step
    keys = jax.eval_shape(lambda *a: body(*a)[5], a.grid, a.params, a.eos, a.cg2dh, a.cg2d_params, a.ex, m.state0,
                          m.ff0, m.phi0surf0, jnp.int32(1), jnp.float64(0.), jnp.int32(0)).keys()
    out_specs = (sp(m.state0), sp(m.ff0), sp(m.phi0surf0), R, R,
                 {k: ((flow_spec, flow_spec, flow_spec) if k == "flow" else R) for k in keys})
    f = sh.shard_map(body, in_specs=in_specs, out_specs=out_specs)

    def step_fn(k, state, ff, phi0, t, it):
        return f(g4, prm4, eos4, cg2dh4, cgp4, sh.ex, state, ff, phi0, jnp.int32(k + 1), jnp.float64(t),
                 jnp.int32(it))
    return step_fn, (st0, ff0, ph0)


P6_SCRIPT = r"""
import os, sys
from mitjax.xla_flags import gate_xla_flags
os.environ["XLA_FLAGS"] = gate_xla_flags("").replace("device_count=4", "device_count=6")
import jax, jax.numpy as jnp, numpy as np
from mitjax.eesupp import exch_maps as EM
from mitjax.eesupp.shard import TileSharding
from mitjax.tests import advect_cs_gate as cg
from mitjax.tests import r2_gate as r2
assert len(jax.devices()) == 6
n = int(sys.argv[1])
cm = cg.CsModel(tag="p6")
m = cm.m
sh = TileSharding(EM.load_cube_maps("advect_cs", "input"), 6)
assert sh.layout.nTiles == 6 and sh.blocks.Tloc == 1
step6, (s6, f6, p6) = cg.sharded_step_fn(cm, sh)
fn = cm.step_fn(())
carry = m.initial_carry()
t, it = m.start_counters()
t6, it6 = t, it
for k in range(n):
    carry, t, it, out, _ = fn(m.arrays, carry, jnp.int32(k + 1), t, it)
    s6, f6, p6, t6, it6, out6 = step6(k, s6, f6, p6, t6, it6)
    d = r2.tree_bits_differ(carry[:3], sh.unpad_tree((s6, f6, p6)))
    assert not d and float(t) == float(t6) and int(it) == int(it6), (k, d)
print("P6 OK", n)
"""


def whole_run(tag="run"):
    """The whole run through the run driver's Model and forward, with the end-of-run pickups (WRITE_PICKUP and
    GAD_WRITE_PICKUP through MDS_WRITE_FIELD's exch2 I/O layout, GO lane). Returns (Model, forward result,
    monitor_gate Oracle of the jdon run)."""
    from mitjax.drivers.model import Model
    from mitjax.drivers.run import forward, load_experiment
    from mitjax.tests import monitor_gate as mg
    exp, inp = EXP
    m = Model(load_experiment(paths.UPSTREAM / "verification" / exp, inp), new_rundir(exp, inp, tag))
    res = forward(m)
    return m, res, mg.oracle(exp, inp)
