"""Helpers of the R1 (tutorial_barotropic_gyre) gates, plan Task 12: the model of a variant set up as THE_MODEL_MAIN
orders it (INI_PARMS, INITIALISE_FIXED, INITIALISE_VARIA), one FORWARD_STEP under jit at the gate XLA flags, the
substep values of the step (FORWARD_STEP's `probe`) compared with the oracle's dump stages, and the whole run.

Setup: INI_PARMS -> ModelParams + Params (ini_parms_dyn) + ForcingParams (FORCING lane); INITIALISE_FIXED -> the
grid (grid_gate.build_grid), INI_LINEAR_PHISURF, INI_EOS (COL lane), INI_CG2D (CG2D lane); INITIALISE_VARIA ->
initialise_varia (INI_FFIELDS / INI_FORCING by the FORCING lane, INTEGR_CONTINUITY at nIter0 by this module, the
other PENDING routines exempt where the variant runs them). phi0surf (SURFACE.h) starts at 0 (INI_LINEAR_PHISURF
:200-208; no geoPotAnomFile in the M1 variants) and is written by EXTERNAL_FORCING_SURF.
"""

from functools import lru_cache
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.farray import FArray, loop_i, loop_j
from mitjax.model.grid import UNSET_RL, declare
from mitjax.model.src.cg2d_h import ini_parms_cg2d
from mitjax.model.src.ini_cg2d import ini_cg2d
from mitjax.model.src.ini_parms import ini_parms_dyn
from mitjax.model.src.ini_linear_phisurf import ini_linear_phisurf
from mitjax.tests import grid_gate as gg
from mitjax.tests import init_gate as ig

EXP = ("tutorial_barotropic_gyre", "input")


class Model:
    """Everything one forward run of a variant needs, built eagerly on the host."""

    def __init__(self, exp, inp):
        from mitjax.model.src.ini_eos import ini_eos
        from mitjax.model.src.ini_ffields import ini_ffields
        from mitjax.model.src.ini_forcing import ini_forcing
        from mitjax.model.src.ini_grid import ini_grid
        from mitjax.model.src.ini_parms_forcing import ini_parms_forcing
        from mitjax.model.src.initialise_varia import executed_pending, initialise_varia
        from mitjax.pkg.rw.read_rec import RW
        from mitjax.params_io import RunParams
        self.exp, self.inp = exp, inp
        e = self.e = gg.experiment(exp, inp)
        cfg = self.cfg = e.cfg
        self.prm = ig.params(exp, inp)
        self.params = ini_parms_dyn(e, self.prm.grid, self.prm.time, self.prm.init)
        self.fp = ini_parms_forcing(e)
        self.cg2d_params = ini_parms_cg2d(e)
        self.ex = gg.exchanger(exp)
        self.ds, self.it0, self.rundir = gg.oracle(exp, inp)
        self.rw = RW(self.rundir, self.prm.init.readBinaryPrec, cfg.size)
        grid = gg.build_grid(exp, inp, params=self.prm.grid, ex=self.ex)
        self.grid = ini_linear_phisurf(grid, cfg=cfg, params=self.params)                # initialise_fixed.F:226
        rp = RunParams(e.run)
        eos_p = SimpleNamespace(fluidIsWater=self.params.fluidIsWater, usingPCoords=self.params.usingPCoords,
                                eosType=self.params.eosType,
                                tAlpha=rp.get("data", "PARM01", "tAlpha") if rp.has("data", "PARM01", "tAlpha")
                                else UNSET_RL,                                           # ini_parms.F:421
                                sBeta=rp.get("data", "PARM01", "sBeta") if rp.has("data", "PARM01", "sBeta")
                                else UNSET_RL)                                           # ini_parms.F:422
        self.eos = ini_eos(cfg=cfg, params=eos_p)                                        # initialise_fixed.F:176
        self.cg2dh = jax.jit(lambda g, s, p, x: ini_cg2d(cfg=cfg, grid=g, surface=s, params=p, ex=x))(
            self.grid, self.grid, self.cg2d_params, self.ex)                            # initialise_fixed.F:246
        pend = tuple(r for r in executed_pending(cfg, self.prm) if r not in ("INTEGR_CONTINUITY",))
        st = initialise_varia(self.grid, cfg=cfg, params=self.prm, ex=self.ex, rw=self.rw, pending=pend,
                              dyn=self.params, cg2dh=self.cg2dh, cg2d_params=self.cg2d_params)   # GO: CG2D.h
        ff = ini_ffields(cfg=cfg)                                                        # initialise_varia.F:213
        _, _, _, lat = ini_grid(cfg=cfg, params=self.prm.grid)
        self.ff = ini_forcing(ff, cfg=cfg, grid=self.grid, fp=self.fp, rw=self.rw, ex=self.ex,
                              latBandClimRelax=lat)                                      # initialise_varia.F:242
        sz = cfg.size
        j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
        i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
        self.phi0surf = declare("Bo_surf", sz).at[i, j].set(0.)                          # ini_linear_phisurf.F:204
        self.state0 = st
        self.pending = pend

    def step_fn(self, probe_stages=()):
        """jit(step)(grid, params, eos, cg2dh, cg2d_params, ex, state, ff, phi0surf, iloop, myTime, myIter) ->
        (state, ff, phi0surf, myTime, myIter, out, probes); every float a traced argument, cfg/fp closed over."""
        from mitjax.model.src.forward_step import forward_step
        cfg, fp = self.cfg, self.fp

        def step(grid, params, eos, cg2dh, cg2d_params, ex, state, ff, phi0surf, iloop, myTime, myIter):
            probes = {}

            def probe(stage, values):
                if stage in probe_stages:
                    probes[stage] = values
            out = forward_step(iloop, myTime, myIter, cfg=cfg, grid=grid, params=params, fp=fp, eos=eos,
                               cg2dh=cg2dh, cg2d_params=cg2d_params, state=state, ff=ff, phi0surf=phi0surf, ex=ex,
                               probe=probe)
            return out + (probes,)
        return jax.jit(step)

    def run_step(self, fn, state, ff, phi0surf, iloop, myTime, myIter):
        return fn(self.grid, self.params, self.eos, self.cg2dh, self.cg2d_params, self.ex, state, ff, phi0surf,
                  jnp.int32(iloop), jnp.float64(myTime), jnp.int32(myIter))


@lru_cache(maxsize=None)
def model(exp=EXP[0], inp=EXP[1]):
    return Model(exp, inp)


def state_fields(state):
    from mitjax.tests.init_gate import _leaves
    return dict(_leaves(state))


def compare_field(ours, ref):
    """(n points, n differing (==), n non-finite ours, n differing bit patterns) on every point incl. halos."""
    o = np.asarray(ours)
    if o.ndim == 3:
        o = o[:, None]
    r = np.asarray(ref)
    if o.shape != r.shape:
        return ("shape", o.shape, r.shape, 0)
    o64, r64 = np.ascontiguousarray(o, np.float64), np.ascontiguousarray(r, np.float64)
    return (o.size, int(np.count_nonzero(~(o == r))), int(np.count_nonzero(~np.isfinite(o))),
            int(np.count_nonzero(o64.view(np.int64) != r64.view(np.int64))))


def compare_stage(ds, it, stage, values):
    """{field: compare_field} for every field the oracle dumps at (it, stage) that `values` holds (a State, an
    FFields, or a dict of FArrays)."""
    keys = {name for (i, s, name) in ds.keys(it) if s == stage}
    if hasattr(values, "names") and hasattr(values, "_f"):
        vals = state_fields(values)
    elif isinstance(values, dict):
        vals = values
    else:
        vals = {n: getattr(values, n) for n in values.names()}
    out = {}
    for n in sorted(keys):
        if n in vals:
            out[n] = compare_field(vals[n].data, ds.field(it, stage, n))
    return out
