"""EXF_GETFORCING in a traced time loop (M4 step 3, lane M4COL session 3): the record bookkeeping of EXF_GETCLIM /
EXF_GETFFIELDS preloaded on the host, the rest of the routine traced.

EXF_SET_FLD's record logic (exf_set_fld.F:120-297: EXF_GETFFIELDREC, the reads READ_REC_3D_RL, EXF_FILTER_RL,
EXF_SWAPFFIELDS) depends only on the clock and the input files, never on the model state; its result for one call is
(fac, fld0, fld1), on which the interpolation :300-314 runs. `exf_preload` runs EXF_GETCLIM and EXF_GETFFIELDS on the
host for the steps iloop = 1..nTimeSteps of a run (the start-of-step clock, forward_step.F:429-430: myIter =
nIter0 + iloop-1, myTime = startTime + deltaTClock*(iloop-1), the values LOAD_FIELDS_DRIVER sees) and keeps every
call's (fac, fld0, fld1); `exf_getforcing_preloaded` is EXF_GETFORCING (exf_getforcing.F:94-393) of step `iloop`
(traced) with those records: the same statements in the same order, the bookkeeping replaced by the lookup (the same
split at the host/traced boundary as external_fields_load.preload_periodic_forcing for EXTERNAL_FIELDS_LOAD).
"""

import dataclasses

import jax
import jax.numpy as jnp

from mitjax.farray import FArray
from mitjax.pkg.exf.exf_getclim import exf_getclim
from mitjax.pkg.exf.exf_getffields import exf_getffields
from mitjax.pkg.exf.exf_getforcing import exf_getforcing_fluxes, exf_stress_exch, exf_tsf


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ExfFP:
    """The PARAMS.h forcing values EXF reads (EXF_MAPFIELDS, EXF_GETFORCING's `fp`): REAL ones traced, the host flag
    of `temp_EvPrRn .NE. UNSET_RL` (exf_mapfields.F:134) static."""
    rhoConstFresh: object
    HeatCapacity_Cp: object
    temp_EvPrRn_set: bool = dataclasses.field(metadata=dict(static=True))


@jax.tree_util.register_pytree_node_class
class ExfPreload:
    """{fldName: (fac [n+1], fld0, fld1)} indexed by iloop (row 0 repeats row 1, unused); fld0 / fld1 are tiled
    FArrays [tile, n+1, j, i] (the step index as a k axis 0..n, the tile axis first, so the tile sharding of the
    P=N runs (drivers/sharded_grad.tile_specs, TileSharding.put_tree) splits them like every other tiled field;
    lane M4LAB session 4: a [n+1, tile, ...] array was replicated and broke shard_map at P=2, 4); `dims` (static)
    the FArray dims of one record."""

    def __init__(self, recs, dims):
        self.recs, self.dims = recs, dims

    def tree_flatten(self):
        keys = tuple(sorted(self.recs))
        return tuple(self.recs[k] for k in keys), (keys, self.dims)

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        keys, dims = aux
        return cls(dict(zip(keys, leaves)), dims)

    def step(self, iloop):
        """{fldName: (fac, fld0, fld1)} of step `iloop` (traced int)."""
        out = {}
        for n, (fac, a0, a1) in self.recs.items():
            out[n] = (fac[iloop], FArray(a0.data[:, iloop], n + "0", _dims=self.dims[n]),
                      FArray(a1.data[:, iloop], n + "1", _dims=self.dims[n]))
        return out


def exf_preload(f, nTimeSteps, *, cfg, exf, cal, grid, params, rw, tp, ex):
    """Host: EXF_GETCLIM + EXF_GETFFIELDS of steps 1..nTimeSteps from EXF_FIELDS.h `f` (after EXF_INIT_VARIA), every
    EXF_SET_FLD call's (fac, fld0, fld1) kept (module docstring). Returns ExfPreload."""
    import numpy as np
    rows = []
    for iloop in range(1, nTimeSteps + 1):
        myIter = tp.nIter0 + (iloop - 1)                                          # forward_step.F:429
        myTime = float(np.float64(tp.startTime) + np.float64(tp.deltaTClock) * np.float64(iloop - 1))   # :430
        rec = {}
        f = exf_getclim(myTime, myIter, f, cfg=cfg, exf=exf, cal=cal, grid=grid, params=params, rw=rw, tp=tp,
                        ex=ex, rec=rec)                                          # exf_getforcing.F:193
        f = exf_getffields(myTime, myIter, f, cfg=cfg, exf=exf, cal=cal, grid=grid, params=params, rw=rw, tp=tp,
                           rec=rec)                                              # :196
        rows.append(rec)
    rows = [rows[0]] + rows
    recs, dims = {}, {}
    for n in rows[0]:
        dims[n] = rows[0][n][1].dims
        kd = dims[n] + (("k", 0, len(rows) - 1),)                           # the step index iloop = 0..n
        recs[n] = (jnp.asarray([np.float64(r[n][0]) for r in rows], jnp.float64),
                   FArray(jnp.stack([r[n][1].data for r in rows], axis=1), n + "0", _dims=kd),
                   FArray(jnp.stack([r[n][2].data for r in rows], axis=1), n + "1", _dims=kd))
    return ExfPreload(recs, dims)


def exf_getforcing_preloaded(iloop, myTime, myIter, f, ff, *, pre, cfg, exf, cal, grid, params, fp, state, tp, ex,
                             probe=None, mon=None, ctrl=None):
    """EXF_GETFORCING( myTime, myIter, myThid ) @63cdc0b pkg/exf/exf_getforcing.F:94-393 at step `iloop` of a traced
    loop, with the records of `pre` (ExfPreload); else as exf_getforcing.exf_getforcing (the same calls in the same
    order: exf_Tsf :163-190, EXF_GETCLIM :193, EXF_GETFFIELDS :196, then exf_getforcing_fluxes :200-388). `mon` (a
    dict, optional): receives EXF_FIELDS.h as EXF_MONITOR sees it (:380). `ctrl` (lane M4ADCOL): the gentim2d
    controls of this step for EXF_GETFFIELDS' useCTRL block (exf_getffields.py). Returns (f, ff)."""
    step = pre.step(iloop)
    exf_Tsf = exf_tsf(f, cfg=cfg, exf=exf, grid=grid, params=params, state=state, ff=ff)   # :163-190
    f = exf_getclim(myTime, myIter, f, cfg=cfg, exf=exf, cal=cal, grid=grid, params=params, rw=None, tp=tp, ex=ex,
                    pre=step)                                                  # :193
    f = exf_getffields(myTime, myIter, f, cfg=cfg, exf=exf, cal=cal, grid=grid, params=params, rw=None, tp=tp,
                       pre=step, ctrl=ctrl)                                    # :196
    for n, (_, a0, a1) in step.items():           # the record arrays EXF_FIELDS.h holds after the call
        f[n + "0"], f[n + "1"] = a0, a1
    if probe is not None:
        probe("X01_exf_getffields", f, ff)
    f = exf_stress_exch(f, exf=exf, ex=ex)                                     # :200-204
    return exf_getforcing_fluxes(exf_Tsf, myTime, myIter, f, ff, cfg=cfg, exf=exf, grid=grid, params=params, fp=fp,
                                 state=state, tp=tp, ex=ex, probe=probe, mon=mon, ctrl=ctrl)
