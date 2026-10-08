"""Helpers of the pkg/seaice gates of 1D_ocean_ice_column/input (M4 step 3, lane M4COL, session 2).

Oracle: the registered dumps-on run of 1D_ocean_ice_column/input (job 27856057, binary
1D_ocean_ice_column-code-63cdc0b-1ac79cb-jaxdump), iterations 0, 1, 2, stages I00b_seaice_begin (before DYNSOLVER,
seaice_model.F:182), I01b_dynsolver (after it), I03_reg_ridge (after SEAICE_REG_RIDGE, :249), I04_growth (after
SEAICE_GROWTH, :277), P13_seaice_model (after SEAICE_MODEL, do_oceanic_phys.F:453), and its STDOUT (rundir/output.txt:
the SEAICE_SUMMARY block).

Teacher forcing: every routine is replayed per dumped iteration from the oracle's fields at the stage before it (the
sea-ice state, the FFIELDS.h fields, the EXF fields, the ocean state theta/salt/uVel/vVel/etaN of S00_begin, which
SEAICE_MODEL sees unchanged: nothing between the start of the step and DO_OCEANIC_PHYS writes them); fields a stage
does not dump are taken from the last stage that dumps them. Every output is compared on every point of the tile,
halos included (exf_gate.compare: element equality, finiteness, differing bit patterns).
REAL parameters are traced jit arguments (SeaiceParams.r, OceanParams); host set-up (READPARMS, INIT_FIXED,
INIT_VARIA) runs eagerly.
"""

import functools
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.farray import FArray
from mitjax.tests import exf_gate as EG
from mitjax.tests import grid_gate as gg
from mitjax.tests import init_gate as ig

EXP, INP = EG.EXP, EG.INP
# SEAICE.h / SEAICE_GRID.h fields of the dumps (group S/B) that seaice_h carries
SF_DUMPED = ("AREA", "HEFF", "HSNOW", "UICE", "VICE", "DWATN", "uIceNm1", "vIceNm1", "ETA", "etaZ", "ZETA", "zetaZ",
             "PRESS", "tensileStrFac", "e11", "e22", "e12", "deltaC", "FORCEX", "FORCEY", "PRESS0", "FORCEX0",
             "FORCEY0", "SEAICE_zMax", "SEAICE_zMin", "AMASS", "DAIRN", "uIceB", "vIceB", "WINDX", "WINDY", "GWATX",
             "GWATY", "d_HEFFbyNEG", "d_HSNWbyNEG", "HSALT", "saltFluxAdjust", "saltWtrIce", "frWtrIce", "HEFFM",
             "SIMaskU", "SIMaskV", "k1AtC", "k2AtC", "k1AtU", "k1AtV", "k2AtU", "k2AtV", "TICES")
FF_DUMPED = ("fu", "fv", "Qnet", "Qsw", "EmPmR", "saltFlux", "sIceLoad", "pLoad")
STAGES = ("I00b_seaice_begin", "I01b_dynsolver", "I03_reg_ridge", "I04_growth", "P13_seaice_model")


@functools.lru_cache(maxsize=None)
def setup(planted=None):
    """SimpleNamespace(exf-gate set-up fields + sp, op, sf0, ff0, state0) of 1D_ocean_ice_column/input.
    `planted`: a tuple of (name, value) replacing SeaiceParams values after SEAICE_READPARMS (negative controls)."""
    from mitjax.model.grid import UNSET_RL
    from mitjax.model.src.ini_ffields import ini_ffields
    from mitjax.model.src.ini_parms_forcing import ini_parms_forcing
    from mitjax.pkg.exf.exf_readparms import params_celsius2K
    from mitjax.pkg.seaice.seaice_h import seaice_fields
    from mitjax.pkg.seaice.seaice_init_fixed import seaice_init_fixed
    from mitjax.pkg.seaice.seaice_init_varia import seaice_init_varia
    from mitjax.pkg.seaice.seaice_params_h import OceanParams
    from mitjax.pkg.seaice.seaice_readparms import seaice_readparms
    from mitjax.model.src.ini_linear_phisurf import ini_linear_phisurf
    s = SimpleNamespace(**vars(EG.setup()))
    s.grid = ini_linear_phisurf(s.grid, cfg=s.cfg, params=s.params)          # initialise_fixed.F:226 (Bo_surf)
    fpp = ini_parms_forcing(s.e)
    s.op = OceanParams(celsius2K=np.float64(params_celsius2K(s.e)), rhoConst=np.float64(fpp.rhoConst),
                       recip_rhoConst=np.float64(fpp.recip_rhoConst), rhoConstFresh=np.float64(fpp.rhoConstFresh),
                       HeatCapacity_Cp=np.float64(fpp.HeatCapacity_Cp), gravity=np.float64(fpp.gravity),
                       recip_gravity=np.float64(fpp.recip_gravity), sIceLoadFac=np.float64(fpp.sIceLoadFac),
                       temp_EvPrRn=np.float64(fpp.temp_EvPrRn), useRealFreshWaterFlux=bool(fpp.useRealFreshWaterFlux),
                       nonlinFreeSurf=int(fpp.nonlinFreeSurf), temp_EvPrRn_set=bool(fpp.temp_EvPrRn != UNSET_RL),
                       usingPCoords=bool(s.params.usingPCoords))
    s.sp = seaice_readparms(s.e, exf=s.exf, tp=s.tp, params=SimpleNamespace(
        recip_rhoConst=fpp.recip_rhoConst, usingCartesianGrid=s.params.usingCartesianGrid,
        monitorFreq=s.params.monitorFreq))                                   # SEAICE_monFreq (:559), as drivers/model.py
    if planted:
        s.sp = s.sp.replace(**dict(planted))
    sf = seaice_fields(s.cfg)
    s.sp, sf = seaice_init_fixed(sf, cfg=s.cfg, sp=s.sp, grid=s.grid, params=s.params, op=s.op)
    s.state0 = ocean_state(s, 0)
    from mitjax.model.src.ini_forcing import ini_forcing
    from mitjax.model.src.ini_grid import ini_grid
    ff = ini_ffields(cfg=s.cfg)                                                # initialise_varia.F / INI_FFIELDS
    _, _, _, lat = ini_grid(cfg=s.cfg, params=ig.params(EXP, INP).grid, ex=s.ex, rw=s.rw)
    ff = ini_forcing(ff, cfg=s.cfg, grid=s.grid, fp=fpp, rw=s.rw, ex=s.ex, latBandClimRelax=lat)   # SWFrac3D
    s.ff_forcing = ff
    s.sf0, s.ff0 = seaice_init_varia(sf, ff, cfg=s.cfg, sp=s.sp, op=s.op, state=s.state0, params=s.params,
                                     tp=s.tp, rw=s.rw, ex=s.ex)
    return s


def _xy(name, data, sz):
    return FArray(jnp.asarray(data), name, i=(1-sz.OLx, sz.sNx+sz.OLx), j=(1-sz.OLy, sz.sNy+sz.OLy))


def _xyz(name, data, sz):
    return FArray(jnp.asarray(data), name, i=(1-sz.OLx, sz.sNx+sz.OLx), j=(1-sz.OLy, sz.sNy+sz.OLy),
                  k=(1, data.shape[1]))


def ocean_state(s, it):
    """theta, salt, uVel, vVel ([tile, k, j, i]) and etaN of S00_begin of iteration `it` (FArrays)."""
    sz = s.sz
    d = {n: s.ds.field(it, "S00_begin", n) for n in ("theta", "salt", "uVel", "vVel", "etaN")}
    return SimpleNamespace(theta=_xyz("theta", d["theta"], sz), salt=_xyz("salt", d["salt"], sz),
                           uVel=_xyz("uVel", d["uVel"], sz), vVel=_xyz("vVel", d["vVel"], sz),
                           etaN=_xy("etaN", d["etaN"][:, 0], sz))


def last_value(s, it, stage, name):
    """[tile, (k,) j, i] of `name` at the last stage of iteration `it` up to `stage` that dumps it (else the last stage
    of the previous iterations)."""
    keys = set(s.ds.keys(it))
    sts = s.ds.stages(it)
    upto = sts[:sts.index(stage) + 1]
    have = [st for st in upto if (it, st, name) in keys]
    if have:
        return s.ds.field(it, have[-1], name)
    if it == 0:
        raise KeyError(f"{name} not dumped before {stage} at iteration 0")
    return last_value(s, it - 1, s.ds.stages(it - 1)[-1], name)


def oracle_fields(s, it, stage):
    """(sf, ffd, exf): the oracle's SEAICE.h dict, FFIELDS.h data dict and EXF fields at `stage` of iteration `it`
    (each field at its last dump up to that stage; at iteration 0 a field not dumped yet is our initial state)."""
    sz = s.sz
    sf = dict(s.sf0)
    for n in SF_DUMPED:
        try:
            a = np.asarray(last_value(s, it, stage, n))
        except KeyError:            # not dumped yet at iteration 0: our SEAICE_INIT_VARIA (gated vs I00b, it 0)
            continue
        sf[n] = _xyz(n, a, sz) if n == "TICES" else _xy(n, a[:, 0], sz)
    ffd = {}
    for n in FF_DUMPED:
        try:
            ffd[n] = np.asarray(last_value(s, it, stage, n))[:, 0]
        except KeyError:            # iteration 0 before its first dump: INI_FFIELDS / SEAICE_INIT_VARIA (ours)
            ffd[n] = np.asarray(getattr(s.ff0, n).data)
    exf = {n: np.asarray(last_value(s, it, stage, n))[:, 0] for n in EG.EXF_DUMPED}
    return sf, ffd, exf


def stage_values(s, it, stage, names):
    """{name: [tile, (k,) j, i]} of the oracle at `stage` (only the names the stage dumps)."""
    keys = set(s.ds.keys(it))
    out = {}
    for n in names:
        if (it, stage, n) in keys:
            a = np.asarray(s.ds.field(it, stage, n))
            out[n] = a if n == "TICES" else a[:, 0]
    return out


def ours_values(sf, ff, exf, names):
    out = {}
    for n in names:
        if n in sf:
            a = np.asarray(sf[n].data)
        elif n in exf:
            a = np.asarray(exf[n].data)
        else:
            a = np.asarray(getattr(ff, n).data)
        out[n] = a
    return out


def compare(ours, ref):
    return EG.compare(ours, ref)


def failures(result):
    return EG.failures(result)


def make_ff(s, ffd):
    """FFields of INI_FFIELDS + INI_FORCING (SWFrac3D) with the given {name: [tile, j, i]} fields."""
    return s.ff_forcing.replace(**{n: _xy(n, v, s.sz) for n, v in ffd.items()})


def make_exf(s, exf):
    return {n: _xy(n, v, s.sz) for n, v in exf.items()}


def dynsolver_fn(s):
    """jit(f(sp, op, sfd, ffd, exfd, state, myTime) -> (sfd, ffd)) of DYNSOLVER (+ OSTRES): dicts of data arrays."""
    from mitjax.pkg.seaice.dynsolver import dynsolver, kgeo_level
    kgeo = kgeo_level(s.sf0)

    def f(sp, op, sfd, ffd, exfd, st, myTime):
        sf = {n: (_xyz(n, v, s.sz) if n == "TICES" else _xy(n, v, s.sz)) for n, v in sfd.items()}
        state = SimpleNamespace(uVel=_xyz("uVel", st["uVel"], s.sz), vVel=_xyz("vVel", st["vVel"], s.sz),
                                etaN=_xy("etaN", st["etaN"], s.sz))
        sf, ff = dynsolver(myTime, 0, sf, make_ff(s, ffd), make_exf(s, exfd), cfg=s.cfg, sp=sp, op=op,
                           grid=s.grid, state=state, ex=s.ex, kgeo=kgeo)
        return {n: a.data for n, a in sf.items()}, {n: getattr(ff, n).data for n in FF_DUMPED}
    return jax.jit(f)


def stage_inputs(s, it, stage):
    """Data dicts (sfd, ffd, exfd, st) of the oracle at `stage` (the inputs of the routine after it)."""
    sf, ffd, exfd = oracle_fields(s, it, stage)
    sfd = {n: a.data for n, a in sf.items()}
    st = ocean_state(s, it)
    std = {"uVel": st.uVel.data, "vVel": st.vVel.data, "etaN": st.etaN.data, "theta": st.theta.data,
           "salt": st.salt.data}
    return sfd, ffd, exfd, std


def run_dynsolver(s, its=(0, 1, 2), sp=None):
    """[(it, comparison)] of DYNSOLVER teacher-forced from I00b_seaice_begin vs I01b_dynsolver."""
    fn = dynsolver_fn(s)
    sp = s.sp if sp is None else sp
    out = []
    for it in its:
        sfd, ffd, exfd, std = stage_inputs(s, it, "I00b_seaice_begin")
        myTime = jnp.float64(s.tp.startTime + s.tp.deltaTClock*(it - s.tp.nIter0))
        osf, off = jax.device_get(fn(sp, s.op, sfd, ffd, exfd, std, myTime))
        ref = stage_values(s, it, "I01b_dynsolver", SF_DUMPED + FF_DUMPED)
        ours = {n: (osf[n] if n in osf else off[n]) for n in ref}
        out.append((it, compare(ours, ref)))
    return out


def _sf_from(s, sfd):
    return {n: (_xyz(n, v, s.sz) if n == "TICES" else _xy(n, v, s.sz)) for n, v in sfd.items()}


def _state_from(s, st):
    return SimpleNamespace(**{n: (_xy(n, v, s.sz) if n == "etaN" else _xyz(n, v, s.sz)) for n, v in st.items()})


def reg_ridge_fn(s):
    """jit(f(sp, op, sfd, ffd, exfd, st, myTime) -> (sfd, ffd)) of SEAICE_REG_RIDGE."""
    from mitjax.pkg.seaice.seaice_reg_ridge import seaice_reg_ridge

    def f(sp, op, sfd, ffd, exfd, st, myTime):
        sf = seaice_reg_ridge(myTime, 0, _sf_from(s, sfd), cfg=s.cfg, sp=sp, op=op)
        return {n: a.data for n, a in sf.items()}, dict(ffd)
    return jax.jit(f)


def growth_fn(s):
    """jit(f(sp, op, sfd, ffd, exfd, st, myTime) -> (sfd, ffd)) of SEAICE_GROWTH."""
    from mitjax.pkg.seaice.seaice_growth import seaice_growth

    def f(sp, op, sfd, ffd, exfd, st, myTime):
        sf, ff = seaice_growth(myTime, 0, _sf_from(s, sfd), make_ff(s, ffd), make_exf(s, exfd), cfg=s.cfg, sp=sp,
                               op=op, grid=s.grid, state=_state_from(s, st), exfp=s.exf)
        return {n: a.data for n, a in sf.items()}, {n: getattr(ff, n).data for n in FF_DUMPED}
    return jax.jit(f)


def run_stage(s, fn, in_stage, out_stage, its=(0, 1, 2), sp=None, op=None, plant=None):
    """[(it, comparison)] of a routine teacher-forced from `in_stage` vs `out_stage` at the dumped iterations.
    `plant(sfd, ffd, exfd, std)` modifies the inputs (negative controls)."""
    sp = s.sp if sp is None else sp
    op = s.op if op is None else op
    out = []
    for it in its:
        sfd, ffd, exfd, std = stage_inputs(s, it, in_stage)
        if plant is not None:
            plant(sfd, ffd, exfd, std)
        myTime = jnp.float64(s.tp.startTime + s.tp.deltaTClock*(it - s.tp.nIter0))
        osf, off = jax.device_get(fn(sp, op, sfd, ffd, exfd, std, myTime))
        ref = stage_values(s, it, out_stage, SF_DUMPED + FF_DUMPED)
        ours = {n: (osf[n] if n in osf else off[n]) for n in ref}
        out.append((it, compare(ours, ref)))
    return out


def model_fn(s, kgeo=None, jit=True):
    """jit(f(sp, op, sfd, ffd, exfd, st, myTime) -> ({stage: (sfd, ffd, exfd)})) of SEAICE_MODEL with its probes.
    `kgeo`: a planted KGEO level (negative control)."""
    from mitjax.pkg.seaice.dynsolver import kgeo_level
    from mitjax.pkg.seaice.seaice_model import seaice_model
    kgeo = kgeo_level(s.sf0) if kgeo is None else kgeo

    def f(sp, op, sfd, ffd, exfd, st, myTime):
        out = {}

        def probe(stage, sf, ff, exf):
            out[stage] = ({n: a.data for n, a in sf.items()}, {n: getattr(ff, n).data for n in FF_DUMPED},
                          {n: a.data for n, a in exf.items()})
        sf, ff, exf = seaice_model(myTime, 0, _sf_from(s, sfd), make_ff(s, ffd), make_exf(s, exfd), cfg=s.cfg,
                                   sp=sp, op=op, grid=s.grid, state=_state_from(s, st), exfp=s.exf, ex=s.ex,
                                   kgeo=kgeo, probe=probe)
        probe("P13_seaice_model", sf, ff, exf)
        return out
    return jax.jit(f) if jit else f


def run_model(s, its=(0, 1, 2), teacher=True, sp=None, kgeo=None):
    """[(it, {stage: comparison})] of SEAICE_MODEL from the state before it (the oracle's at S02_load_fields:
    EXF fields of X06, sea ice and FFIELDS.h at their last dump; `teacher`=False: our own chain from SEAICE_INIT_VARIA,
    with the oracle's EXF fields and ocean state of each iteration) vs every sea-ice stage I00b..P13."""
    fn = model_fn(s, kgeo=kgeo)
    sp = s.sp if sp is None else sp
    res = []
    own = None
    for it in its:
        sfd, ffd, exfd, std = stage_inputs(s, it, "S02_load_fields")
        if not teacher and own is not None:
            osfd, offd = own
            sfd = dict(osfd)
            # FFIELDS.h fields SEAICE_MODEL rewrites come from our chain; the others (fu, fv, pLoad, and Qnet etc.
            # of EXF_MAPFIELDS this iteration) from the oracle's X06 / S02 of this iteration
            for n in ("sIceLoad",):
                ffd[n] = offd[n]
        myTime = jnp.float64(s.tp.startTime + s.tp.deltaTClock*(it - s.tp.nIter0))
        out = jax.device_get(fn(sp, s.op, sfd, ffd, exfd, std, myTime))
        cmp = {}
        for st in STAGES:
            osf, off, oexf = out[st]
            ref = stage_values(s, it, st, SF_DUMPED + FF_DUMPED + EG.EXF_DUMPED)
            ours = {n: (osf[n] if n in osf else off[n] if n in off else oexf[n]) for n in ref}
            cmp[st] = compare(ours, ref)
        res.append((it, cmp))
        own = out["P13_seaice_model"][:2]
    return res
