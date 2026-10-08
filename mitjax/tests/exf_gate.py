"""Helpers of the EXF / CAL gates of 1D_ocean_ice_column/input (M4 step 3, lane M4COL).

Oracle: the registered dumps-on run of 1D_ocean_ice_column/input (reference/reference_runs.py, kind "jdon"; job
27856057, binary 1D_ocean_ice_column-code-63cdc0b-1ac79cb-jaxdump), iterations 0, 1, 2, stages X01_exf_getffields
.. X06_exf_mapfields (reference/jaxdump/SUBSTEPS.md "M4"), and its STDOUT (rundir/output.txt: the calendar summary
and the EXF field start times).

EXF_GETFORCING is replayed per dumped iteration, teacher-forced: the EXF fields before the call are EXF_INIT_VARIA's
(ours) at the first iteration and the oracle's X06_exf_mapfields of the previous iteration after it; gcmSST is
theta(ks) of S00_begin (LOAD_FIELDS_DRIVER, load_fields_driver.F:174-183); the record arrays fld0/fld1 (not dumped)
are carried by our own host reads from the first iteration on. Every stage X01..X06 is compared on every point of the
tile, halos included: element equality, both finite, plus the count of differing bit patterns.
REAL parameters are traced jit arguments (ExfParams.r, the forcing pytree FP) in the traced part (X02..X06); the
record reads and the time interpolation of X01 run on the host (eager, one XLA computation per operation).
"""

import dataclasses
import functools
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.farray import FArray
from mitjax.tests import grid_gate as gg
from mitjax.tests import init_gate as ig

EXP = "1D_ocean_ice_column"
INP = "input"
STAGES = ("X01_exf_getffields", "X02_exf_radiation", "X03_exf_wind", "X04_exf_bulkformulae", "X05_exf_hflux_sflux",
          "X06_exf_mapfields")
# EXF_FIELDS.h fields the dumps hold (jaxdump group x)
EXF_DUMPED = ("apressure", "aqh", "atemp", "climsss", "climsst", "cw", "evap", "hflux", "hl", "hs", "lwdown",
              "lwflux", "precip", "runoff", "saltflx", "sflux", "sh", "snowprecip", "sw", "swdown", "swflux",
              "ustress", "uwind", "vstress", "vwind", "wStress", "wspeed")
# FFIELDS.h fields EXF_MAPFIELDS writes (X06)
FF_MAPPED = ("Qnet", "Qsw", "EmPmR", "fu", "fv", "pLoad", "saltFlux")


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FP:
    """The PARAMS.h forcing values EXF reads: REAL ones traced, the host flag of `temp_EvPrRn .NE. UNSET_RL`
    (exf_mapfields.F:134) static."""
    rhoConstFresh: object
    HeatCapacity_Cp: object
    temp_EvPrRn_set: bool = dataclasses.field(metadata=dict(static=True))


@functools.lru_cache(maxsize=None)
def setup(planted_exf=None):
    """SimpleNamespace(e, cfg, sz, params, grid, ex, rw, tp, cal, exf, fp, ds) of 1D_ocean_ice_column/input.
    `planted_exf`: a tuple of (name, value) replacing EXF parameters (negative controls)."""
    from mitjax.eesupp.exch_maps import build_maps
    from mitjax.eesupp.exchange import Exchanger
    from mitjax.model.grid import UNSET_RL
    from mitjax.model.src.ini_parms import ini_parms_dyn, set_ref_state_eos
    from mitjax.model.src.ini_parms_forcing import ini_parms_forcing
    from mitjax.pkg.cal.cal_init_fixed import cal_init_fixed
    from mitjax.pkg.cal.cal_readparms import cal_readparms
    from mitjax.pkg.exf.exf_init_fixed import exf_init_fixed
    from mitjax.pkg.exf.exf_readparms import exf_readparms
    from mitjax.pkg.rw.read_rec import RW
    e = gg.experiment(EXP, INP)
    cfg = e.cfg
    prm = ig.params(EXP, INP)
    tp = prm.time
    params = ini_parms_dyn(e, prm.grid, prm.time, prm.init)
    ds, _, rundir = gg.oracle(EXP, INP)
    ex = Exchanger(build_maps(ds))
    grid = gg.build_grid(EXP, INP, params=prm.grid, ex=ex)
    params = set_ref_state_eos(params, grid, e)                                # surf_pRef (JMD95Z)
    rw = RW(rundir, prm.grid.readBinaryPrec, cfg.size)
    cal = cal_readparms(e)
    cal = cal_init_fixed(cal, startTime=tp.startTime, endTime=tp.endTime, deltaTClock=tp.deltaTClock,
                         nIter0=tp.nIter0, nEndIter=tp.nEndIter, nTimeSteps=tp.nTimeSteps)
    exf = exf_readparms(e, params=params)
    exf = exf_init_fixed(exf, cfg=cfg, cal=cal, nIter0=tp.nIter0, startTime=tp.startTime)
    if planted_exf:
        exf = exf.replace(**dict(planted_exf))
    fpp = ini_parms_forcing(e)
    fp = FP(rhoConstFresh=jnp.float64(fpp.rhoConstFresh), HeatCapacity_Cp=jnp.float64(fpp.HeatCapacity_Cp),
            temp_EvPrRn_set=bool(fpp.temp_EvPrRn != UNSET_RL))
    return SimpleNamespace(e=e, cfg=cfg, sz=cfg.size, params=params, grid=grid, ex=ex, rw=rw, tp=tp, cal=cal,
                           exf=exf, fp=fp, ds=ds, rundir=rundir)


def compare(ours, ref):
    """{name: (n points, n differing (==), n non-finite ours, n non-finite oracle, n differing bit patterns)}."""
    out = {}
    for n in ref:
        o, r = np.asarray(ours[n], np.float64), np.asarray(ref[n], np.float64)
        if o.shape != r.shape:
            out[n] = ("shape", o.shape, r.shape)
            continue
        o, r = np.ascontiguousarray(o), np.ascontiguousarray(r)
        out[n] = (o.size, int(np.count_nonzero(~(o == r))), int(np.count_nonzero(~np.isfinite(o))),
                  int(np.count_nonzero(~np.isfinite(r))), int(np.count_nonzero(o.view(np.int64) != r.view(np.int64))))
    return out


def failures(result):
    return {k: v for k, v in result.items() if v[0] == "shape" or any(v[1:])}


def _xy(name, data, sz):
    return FArray(jnp.asarray(data), name, i=(1-sz.OLx, sz.sNx+sz.OLx), j=(1-sz.OLy, sz.sNy+sz.OLy))


def init_fields(s):
    """EXF_FIELDS.h after EXF_INIT_VARIA (host)."""
    from mitjax.pkg.exf.exf_fields_h import exf_fields
    from mitjax.pkg.exf.exf_init_varia import exf_init_varia
    return exf_init_varia(exf_fields(s.sz), cfg=s.cfg, exf=s.exf, grid=s.grid, params=s.params, rw=s.rw)


def oracle_stage(s, it, stage):
    """{name: [tile, j, i]} of the EXF fields (and at X06 the FFIELDS.h fields of EXF_MAPFIELDS) in the oracle."""
    d = {n: s.ds.field(it, stage, n)[:, 0] for n in EXF_DUMPED}
    if stage == "X06_exf_mapfields":
        d.update({n: s.ds.field(it, stage, n)[:, 0] for n in FF_MAPPED})
    return d


def fluxes_fn(s):
    """jit(f(exf, params, fp, Tsf, fields, ff, myTime) -> ({stage: {name: array}})): EXF_GETFORCING after the reads
    (exf_getforcing_fluxes), every stage X02..X06 returned."""
    from mitjax.pkg.exf.exf_getforcing import exf_getforcing_fluxes
    cfg, sz, grid, ex, tp = s.cfg, s.sz, s.grid, s.ex, s.tp

    def f(exf, params, fp, Tsf, fields, ffd, myTime):
        from mitjax.model.src.ini_ffields import ini_ffields
        flds = {n: _xy(n, v, sz) for n, v in fields.items()}
        ff = ini_ffields(cfg=cfg)
        ff = ff.replace(**{n: _xy(n, v, sz) for n, v in ffd.items()})
        out = {}

        def probe(stage, fl, ff_):
            d = {n: fl[n].data for n in EXF_DUMPED}
            if stage == "X06_exf_mapfields":
                d.update({n: getattr(ff_, n).data for n in FF_MAPPED})
            out[stage] = d
        state = SimpleNamespace(theta=None, uVel=None, vVel=None)
        exf_getforcing_fluxes(_xy("exf_Tsf", Tsf, sz), myTime, 0, flds, ff, cfg=cfg, exf=exf, grid=grid,
                              params=params, fp=fp, state=state, tp=tp, ex=ex, probe=probe)
        return out
    return jax.jit(f)


def prior_fields(s, prev, last_stage=True):
    """{name: [tile, j, i]} of the EXF fields the oracle holds at the end of iteration `prev`: each field at the last
    stage of that iteration that dumps it (I00b_seaice_begin: SEAICE_MODEL exchanges uwind/vwind,
    seaice_model.F EXCH_UV_AGRID_3D_RL, which changes their halos after X06); `last_stage=False`: X06 (a negative
    control: the halos of uwind/vwind then differ)."""
    keys = set(s.ds.keys(prev))
    out = {}
    for n in EXF_DUMPED:
        sts = [st for st in s.ds.stages(prev) if (prev, st, n) in keys]
        st = sts[-1] if last_stage else "X06_exf_mapfields"
        out[n] = s.ds.field(prev, st, n)[:, 0]
    return out


def run_steps(s, its=(0, 1, 2), teacher=True, plant=None, last_stage=True):
    """[(it, {stage: comparison})] of EXF_GETFORCING at the dumped iterations. `teacher`: the EXF fields before each
    call are the oracle's at the end of the previous iteration (prior_fields; else our own chain). `plant(stage, d)`
    modifies our fields at a stage (negative controls; static)."""
    from mitjax.model.src.ini_ffields import ini_ffields
    from mitjax.pkg.exf.exf_getffields import exf_getffields
    from mitjax.pkg.exf.exf_getclim import exf_getclim
    from mitjax.pkg.exf.exf_getforcing import exf_tsf
    sz, tp = s.sz, s.tp
    fields = init_fields(s)
    ff0 = ini_ffields(cfg=s.cfg)
    ffd = {n: getattr(ff0, n).data for n in FF_MAPPED}
    fn = fluxes_fn(s)
    res = []
    prev = None
    for it in its:
        if teacher and prev is not None:
            o = prior_fields(s, prev, last_stage)
            fields = dict(fields)
            for n in EXF_DUMPED:
                fields[n] = _xy(n, o[n], sz)
        myTime = np.float64(tp.startTime + tp.deltaTClock*(it - tp.nIter0))
        theta = s.ds.field(it, "S00_begin", "theta")
        ks = 1
        gcm = _xy("gcmSST", theta[:, ks-1], sz)
        ff = ini_ffields(cfg=s.cfg).replace(gcmSST=gcm)
        Tsf = exf_tsf(fields, cfg=s.cfg, exf=s.exf, grid=s.grid, params=s.params, state=None, ff=ff)
        fields = exf_getclim(myTime, it, fields, cfg=s.cfg, exf=s.exf, cal=s.cal, grid=s.grid, params=s.params,
                             rw=s.rw, tp=tp, ex=s.ex)
        fields = exf_getffields(myTime, it, fields, cfg=s.cfg, exf=s.exf, cal=s.cal, grid=s.grid, params=s.params,
                                rw=s.rw, tp=tp)
        ours = {"X01_exf_getffields": {n: np.asarray(fields[n].data) for n in EXF_DUMPED}}
        din = {n: fields[n].data for n in EXF_DUMPED}
        if plant is not None:
            din = plant("in", din)
        out = jax.device_get(fn(s.exf, s.params, s.fp, Tsf.data, din, ffd, jnp.float64(myTime)))
        ours.update(out)
        if plant is not None:
            ours = {st: plant(st, d) for st, d in ours.items()}
        cmp = {st: compare(ours[st], oracle_stage(s, it, st)) for st in STAGES}
        # our own chain continues from our X06 fields (records stay ours), then SEAICE_MODEL's exchange of the
        # winds (seaice_model.F:124-126, `CALL EXCH_UV_AGRID_3D_RL( uwind, vwind, .TRUE., 1, myThid )`: pkg/exf
        # does not update their edges); compared with I00b_seaice_begin, the stage right after it
        fields = dict(fields)
        for n in EXF_DUMPED:
            fields[n] = _xy(n, ours["X06_exf_mapfields"][n], sz)
        u, v = s.ex.EXCH_UV_AGRID_3D_RL(fields["uwind"].data, fields["vwind"].data, True)
        fields["uwind"], fields["vwind"] = _xy("uwind", u, sz), _xy("vwind", v, sz)
        cmp["I00b_winds"] = compare({"uwind": np.asarray(u), "vwind": np.asarray(v)},
                                    {n: s.ds.field(it, "I00b_seaice_begin", n)[:, 0] for n in ("uwind", "vwind")})
        res.append((it, cmp))
        prev = it
    return res
