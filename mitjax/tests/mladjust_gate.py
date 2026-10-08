"""Helpers of the MLAdjust gates (M3 Task 30): verification/MLAdjust and its six variants through the driver `Model`
(mitjax/drivers/model.py), free-running from the Model's initial state, every dumped stage of steps 1-3 compared with
lane A's dumps-on run of the variant (`jdon`, three iterations from nIter0: 0-2, input.A4FlxF 36-38).

The step and its comparison are goadk_model_gate's (`step_fn`, `compare_step`: the model's own FORWARD_STEP with the
per-level DYNAMICS stages D00a / D00b / D00c recorded per level k; every point of every tile, halos included, element
equality on bit patterns). The exchanger is built in memory from the run's own exchange probe (X00_exch_probe) and
the exch2 vector tables (`maps`).
`compare_extra` adds the S00_begin / S04_oceanic_phys records the probes do not hold: the GRID.h hFac fields (fixed
in a linear free-surface run) against the Model's grid, the CG2D.h operator (aW2d ... pC) against its cg2dh, and,
in a run without useGMRedi, the GMREDI.h tensor and bolus stream function, which no routine writes (the zero of the
never-initialised common block).
"""

import functools
import time

import jax.numpy as jnp
import numpy as np

from mitjax.tests import goadk_model_gate as gmg
from mitjax.tests import r1_gate as rg

EXP = "MLAdjust"
VARIANTS = ("input", "input.A4FlxF", "input.AhFlxF", "input.AhStTn", "input.AhVrDv", "input.QGLeith", "input.QGLthGM")


@functools.lru_cache(maxsize=None)
def oracle(inp):
    from mitjax.tests import rstar_gate as rsg
    return rsg.oracle(EXP, inp, "jdon")


def model(inp):
    """A fresh driver Model of the variant (run directory of links under $MJX_RUNS/mladjust, never reused)."""
    import mitjax.drivers.model as dm
    from mitjax import paths
    from mitjax.drivers.run import load_experiment, make_rundir
    from mitjax.eesupp.exchange import Exchanger
    exp_dir = paths.UPSTREAM / "verification" / EXP
    e = load_experiment(exp_dir, inp)
    rd = make_rundir(exp_dir, inp, paths.RUNS / "mladjust" / f"{inp}-{time.time_ns()}")
    ds, _, _ = oracle(inp)
    return dm.Model(e, rd, ex=Exchanger(maps(inp, ds)))


def maps(inp, ds):
    """The exchange maps of the variant: the probed copies (exch_maps.build_maps of its X00_exch_probe) with the exch2
    C-grid vector tables (`rx2`: EXCH2_PUT_RX2's `sa1*A1 + sa2*A2`, which turns a -0 halo copy into +0 where the
    other component is >= +0) of the W2 set-up, as load_maps adds them to a registered map file."""
    from dataclasses import replace
    from mitjax.eesupp.exch_maps import build_maps, exch2_vector_tables, uses_exch2
    m = build_maps(ds)
    if uses_exch2(EXP, inp):
        m = replace(m, rx2=exch2_vector_tables(EXP, inp, m.layout))
    return m


def run_steps(m, ds, its, nsteps=3, arrays=None):
    """[(iteration, compare_step result)] of steps 1..nsteps from the Model's initial carry, the oracle's iteration
    its[0] + n compared with step n + 1 (FORWARD_STEP of iteration its[0] + n). `arrays`: the step's jit arguments
    (default m.arrays; a negative control passes changed Params)."""
    f = gmg.step_fn(m)
    a = m.arrays if arrays is None else arrays
    carry = m.initial_carry()
    myTime, myIter = m.start_counters()
    out = []
    for n in range(nsteps):
        carry, myTime, myIter, _, probes = f(a, carry, jnp.int32(n + 1), myTime, myIter)
        res = gmg.compare_step(m, ds, its[0] + n, probes)
        compare_extra(m, ds, its[0] + n, res)
        out.append((its[0] + n, res))
    return out, carry


GRID_FIELDS = ("hFacC", "hFacW", "hFacS", "recip_hFacC")
CG2D_FIELDS = ("aW2d", "aS2d", "aC2d", "pW", "pS", "pC")
GM_FIELDS = ("Kwx", "Kwy", "Kwz", "Kux", "Kvy", "Kuz", "Kvz", "GM_PsiX", "GM_PsiY")


def compare_extra(m, ds, it, res):
    """Adds to `res` (compare_step's {stage: {field: compare_field}}) the S00_begin / S04_oceanic_phys records the
    probes do not hold (module docstring)."""
    if m.cfg.cpp.NONLIN_FRSURF:
        raise NotImplementedError("mladjust_gate.compare_extra: the hFac fields of a NONLIN_FRSURF run")
    from mitjax.model.src.ini_parms import _use
    gm_on = _use(m.cfg, "useGMRedi")
    for s in ("S00_begin", "S04_oceanic_phys"):
        names = {n for (_, s_, n) in ds.keys(it) if s_ == s}
        for n in names - set(res.get(s, {})):
            ref = ds.field(it, s, n)
            if n in GRID_FIELDS:
                ours = np.asarray(getattr(m.grid, n).data)
            elif n in CG2D_FIELDS:
                ours = np.asarray(getattr(m.cg2dh, n).data)
            elif n in GM_FIELDS and not gm_on:
                ours = np.zeros_like(ref)
            else:
                continue
            res.setdefault(s, {})[n] = rg.compare_field(ours, ref)     # a 2-D field against its 1-level record
    return res
