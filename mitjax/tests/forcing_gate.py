"""Helpers of the forcing and SBO gates (M1 sub-lane FORCING): the forcing routines of an M1 variant run on the
oracle's own dumped inputs and compared with the oracle's dumped outputs, bitwise (element equality and equal bit
patterns, both sides finite) on every point of every tile, halos included.

Oracle: the registered dumps-on run of each variant (reference/reference_runs.py, kind "jdon"; grid_gate.oracle),
stages S00_begin (state and FFIELDS at the start of a step), S01_update_rstar_F / S07 / S12 (hFacC under r*),
S02_load_fields (after LOAD_FIELDS_DRIVER), S03_ctrl_map_forcing (after CTRL_MAP_FORCING, optim), P01_external_forcing_surf
(after EXTERNAL_FORCING_SURF), S04_oceanic_phys (theta after FREEZE_SURFACE: nothing else in DO_OCEANIC_PHYS writes
theta in these variants), S16_blocking_exchanges (state at the end of the step), S19_sbo_calc (SBO.h scalars).

Forcing time of a step (the_main_loop / forward_step counters, mitjax/drivers/the_main_loop.py): the step that starts
at iteration `it` has myIter = it and myTime = startTime + deltaTClock*(it - nIter0).
"""

import numpy as np
import jax.numpy as jnp

from mitjax.farray import FArray
from mitjax.model import state as stmod
from mitjax.model.src.ini_ffields import ini_ffields
from mitjax.model.src.ini_forcing import ini_forcing
from mitjax.model.src.ini_grid import ini_grid
from mitjax.model.src.ini_parms import ini_parms_time
from mitjax.model.src.ini_parms_forcing import ini_parms_forcing
from mitjax.model.state import State
from mitjax.pkg.rw.read_rec import RW
from mitjax.tests import grid_gate as gg

FORCING_F = ("fu", "fv", "Qnet", "Qsw", "EmPmR", "saltFlux", "pLoad", "sIceLoad")
SURF_F = ("surfaceForcingU", "surfaceForcingV", "surfaceForcingT", "surfaceForcingS", "EmPmR")

_CTX = {}


class Ctx:
    """Everything a forcing gate of one variant needs (built once per variant and cached)."""

    def __init__(self, exp, inp):
        self.exp, self.inp = exp, inp
        self.e = gg.experiment(exp, inp)
        self.cfg = self.e.cfg
        self.sz = self.cfg.size
        self.fp = ini_parms_forcing(self.e)
        self.tp = ini_parms_time(self.e)
        self.ds, self.it0, self.rundir = gg.oracle(exp, inp)
        self.ex = gg.exchanger(exp)
        self.rw = RW(self.rundir, 32 if self.fp is None else _read_prec(self.e), self.sz)
        self._grid = None

    @property
    def grid(self):
        if self._grid is None:
            self._grid = gg.build_grid(self.exp, self.inp, ex=self.ex)
        return self._grid

    def latBandClimRelax(self):
        _, _, _, lat = ini_grid(cfg=self.cfg, params=gg.grid_params(self.exp, self.inp))
        return lat

    def my_time(self, it):
        return self.tp.startTime + self.tp.deltaTClock * float(it - self.tp.nIter0)

    def iterations(self):
        return self.ds.iterations()

    def has(self, it, stage, fld):
        return (it, stage, fld) in self.ds.index

    def ref(self, it, stage, fld, occ=0):
        return self.ds.field(it, stage, fld, occ)

    def fa(self, it, stage, fld, like, occ=0):
        """An FArray with the declaration of `like` holding the oracle's dump of `fld`."""
        a = np.asarray(self.ref(it, stage, fld, occ))
        if a.ndim == 4 and like.data.ndim == 3:
            a = a[:, 0]
        return FArray(jnp.asarray(a), like.name, tiled=like.tiled, _dims=like.dims)

    def state_from(self, it, stage, names):
        d = {}
        for n in names:
            d[n] = self.fa(it, stage, n, stmod.declare(n, self.sz))
        return State(d)

    def initial_ffields(self):
        """INI_FFIELDS then INI_FORCING (initialise_varia.F order: INI_FFIELDS in INI_FIELDS, INI_FORCING after)."""
        ff = ini_ffields(cfg=self.cfg)
        return ini_forcing(ff, cfg=self.cfg, grid=self.grid, fp=self.fp, rw=self.rw, ex=self.ex,
                           latBandClimRelax=self.latBandClimRelax())


def _read_prec(e):
    from mitjax.model.src.ini_parms import ini_parms_init
    return ini_parms_init(e).readBinaryPrec


def ctx(exp, inp):
    if (exp, inp) not in _CTX:
        _CTX[(exp, inp)] = Ctx(exp, inp)
    return _CTX[(exp, inp)]


def compare(ours, ref):
    """(n points, n differing (==), n non-finite ours, n non-finite oracle, n differing bit patterns)."""
    o = np.asarray(ours.data if isinstance(ours, FArray) else ours, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    if o.ndim == 3 and r.ndim == 4:
        r = r[:, 0]
    if o.shape != r.shape:
        return ("shape", o.shape, r.shape)
    o, r = np.ascontiguousarray(o), np.ascontiguousarray(r)
    return (o.size, int(np.count_nonzero(~(o == r))), int(np.count_nonzero(~np.isfinite(o))),
            int(np.count_nonzero(~np.isfinite(r))), int(np.count_nonzero(o.view(np.int64) != r.view(np.int64))))


def bad(res):
    return {k: v for k, v in res.items() if v[0] == "shape" or any(v[1:])}


def ffields_with(c, ff, it, stage, names):
    """`ff` with the fields `names` replaced by the oracle's dump at (it, stage)."""
    return ff.replace(**{n: c.fa(it, stage, n, getattr(ff, n)) for n in names if c.has(it, stage, n)})


def forcing_input_stage(c, it):
    """The stage whose group f is the input of DO_OCEANIC_PHYS: S03_ctrl_map_forcing if the build dumps it, else
    S02_load_fields."""
    return "S03_ctrl_map_forcing" if c.has(it, "S03_ctrl_map_forcing", "Qnet") else "S02_load_fields"


def hfac_stage(c, it, before):
    """The last stage at or before `before` (in call order) that dumps hFacC in iteration it (r* builds rescale it at
    S01, S07): the hFacC a routine at `before` sees."""
    stages = c.ds.stages(it)
    best = "S00_begin"
    for s in stages:
        if c.has(it, s, "hFacC"):
            best = s
        if s == before:
            break
    return best
