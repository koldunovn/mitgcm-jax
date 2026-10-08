"""Helpers of the lab_sea/input gates (M4 step 5, lane M4LAB session 1).

Oracle: lane A's registered dumps-on run job27855987-jdon of lab_sea/input (build lab_sea-code-63cdc0b, jaxdump),
iterations 1, 2, 3 (nIter0 = 1: the run starts from pickup.0000000001, pickup_cd.0000000001,
pickup_seaice.0000000001).

Session 1 scaffolding (`stub_model`): the Model of lab_sea/input could not be built while SEAICE_READPARMS /
SEAICE_INIT_VARIA raised on what was not ported (the lane's gap list L3-L11); `stub_model` built it with
those settings replaced (STUB_NAMELIST, STUB_CPP_OFF). Session 3 ported L4-L11, session 4 L3 (ALLOW_SITRACER): both
stubs are empty, `stub_model` is the plain Model of lab_sea/input on the oracle's run directory."""

import dataclasses
import functools

import jax.numpy as jnp
import numpy as np

EXP = ("lab_sea", "input")
JDON = "job27855987-jdon"
S = "data.seaice"
# the gap-list stubs (lane M4LAB session 1): run-time settings the port does not cover yet -> covered ones
# session 3: L4-L11 ported (the namelist stubs and the SEAICE_ALLOW_BOTTOMDRAG plant are gone)
STUB_NAMELIST = {}
STUB_CPP_OFF = ()                                              # L3 SITRACER ported (session 4): no plant


def stub_experiment():
    from mitjax.config.params import load
    from mitjax.drivers.model import with_namelist
    from mitjax.tests.m4col_gate import CppPlant
    e = with_namelist(load(*EXP), STUB_NAMELIST)
    return dataclasses.replace(e, cfg=dataclasses.replace(e.cfg, cpp=CppPlant(e.cfg.cpp, off=STUB_CPP_OFF)))


@functools.lru_cache(maxsize=None)
def stub_model():
    """(Model, DumpSet) of lab_sea/input with the session-1 stubs, on the oracle's dumps-on run directory."""
    from mitjax import paths
    from mitjax.drivers.model import Model
    from mitjax.io.dump import DumpSet
    from mitjax.xla_flags import set_gate_xla_flags
    set_gate_xla_flags()
    top = paths.REFERENCE_RUNS / EXP[0] / EXP[1] / JDON
    return Model(stub_experiment(), top / "rundir"), DumpSet(top / "dumps")


def ndiff(ref, got):
    """Number of points whose float64 bit pattern differs (ref: oracle array; got: FArray or array)."""
    o = np.asarray(got.data if hasattr(got, "data") else got, np.float64)
    r = np.asarray(ref, np.float64)
    if o.ndim == 4 and r.size != o.size:              # an nITD axis: the dump holds category 1
        o = o[:, 0]
    r = np.ascontiguousarray(r.reshape(o.shape))
    return int(np.count_nonzero(np.ascontiguousarray(o).view(np.int64) != r.view(np.int64)))


def seaice_init_diffs(m, ds, it=1):
    """{field: differing points} of SEAICE.h / SEAICE_GRID.h after SEAICE_INIT_FIXED + SEAICE_INIT_VARIA (the
    Model's initial sea-ice fields, from pickup_seaice.0000000001) against the oracle's I00_seaice_begin of the first
    iteration (nothing writes them between the initialisation and SEAICE_MODEL's start)."""
    sf = m.pk0["seaice"]
    keys = set(ds.keys(it))
    return {n: ndiff(ds.field(it, "I00_seaice_begin", n), v) for n, v in sf.items()
            if (it, "I00_seaice_begin", n) in keys and hasattr(v, "data")}


def external_forcing_surf_fn(m):
    """f(ff, area, theta, salt, phi0surf, myTime, myIter) -> ff: EXTERNAL_FORCING_SURF as DO_OCEANIC_PHYS calls it
    (do_oceanic_phys.F:560-575, the whole tile incl. halos), with SEAICE.h AREA."""
    import jax
    from mitjax.model.src.external_forcing_surf import external_forcing_surf
    sz = m.cfg.size
    iMin, iMax = 1-sz.OLx, sz.sNx+sz.OLx
    jMin, jMax = 1-sz.OLy, sz.sNy+sz.OLy

    def f(sp, ff, area, state, phi0surf, myTime, myIter):
        ff, _, _ = external_forcing_surf(iMin, iMax, jMin, jMax, myTime, myIter, ff, cfg=m.cfg, grid=m.grid,
                                         fp=m.fp, state=state, phi0surf=phi0surf, ptr=m.params, sp=sp, area=area)
        return ff
    return jax.jit(f)


def external_forcing_surf_diffs(m, ds, its=(1, 2, 3), sp=None):
    """{it: {field: differing points}} of P01_external_forcing_surf, teacher-forced: FFIELDS.h from the step's own
    front (m4off_gate.front_fn through S02_load_fields from the oracle's carry: SSS = climsss of EXF_MAPFIELDS, which
    is not dumped as SSS), then every FFIELDS.h field the oracle dumps at P13_seaice_model (after SEAICE_MODEL;
    FREEZE_SURFACE does not run: allowFreezing .FALSE.) and AREA from P13; theta/salt from S00_begin
    (THERMODYNAMICS has not run yet), phi0surf from the carry."""
    from mitjax.tests import m4off_gate as G
    f = external_forcing_surf_fn(m)
    front = G.front_fn(m)
    sp = m.arrays.pkc["sp"] if sp is None else sp
    tp = m.prm.time
    out = {}
    for it in its:
        keys = set(ds.keys(it))
        k = it - tp.nIter0
        myTime, myIter = jnp.float64(tp.startTime + tp.deltaTClock*k), jnp.int32(it)
        carry = G.teacher_carry(m, ds, it)
        ff = front(m.arrays, carry, jnp.int32(k + 1), myTime, myIter)["S02_load_fields"]
        ff = ff.replace(**{n: G.like(getattr(ff, n), ds.field(it, "P13_seaice_model", n)) for n in ff.names()
                           if (it, "P13_seaice_model", n) in keys})
        area = G.like(m.pk0["seaice"]["AREA"], ds.field(it, "P13_seaice_model", "AREA"))
        st = carry[0]
        ff2 = f(sp, ff, area, st, carry[2], myTime, myIter)
        out[it] = {n: ndiff(ds.field(it, "P01_external_forcing_surf", n), getattr(ff2, n))
                   for n in ff2.names() if (it, "P01_external_forcing_surf", n) in keys}
    return out


# ------------------------------------------------------------------- session 2: the ocean physics of lab_sea
TENSOR = ("Kwx", "Kwy", "Kwz", "Kux", "Kvy", "Kuz", "Kvz")
OCEAN_STAGES = ("P01_external_forcing_surf", "P02_rho_sigma_ivdc", "P03_mxlayer", "P07_kpp", "P05_gmredi_tensor",
                "P06_gmredi_exch", "P10_kpp_exch", "S04_oceanic_phys", "T01_residual_flow", "T11_temp_gT",
                "T12_temp_step", "T13_temp_impl", "T02_temp_integrate", "T21_salt_gS", "T22_salt_step",
                "T23_salt_impl", "T03_salt_integrate", "S05_thermodynamics_sync")


def ocean_inputs(m, ds, it):
    """The Model's carry with the oracle's state at the start of iteration `it` (m4off_gate.teacher_carry: State,
    FFIELDS.h, phi0surf, SEAICE.h and EXF_FIELDS.h from the oracle's last dump before the iteration), GMREDI.h's
    tensor from S00_begin, KPP.h from P10_kpp_exch of iteration it-1 (at the first iteration the Model's own
    KPP_INIT_VARIA values: KPP.h has no pickup). Session 4: the SEAICE_MODEL of the step is the port's own (L3-L12
    ported), so nothing after S00_begin is forced: no P13 values in place of SEAICE_MODEL and no SALT_PLUME.h from
    I04 (SEAICE_GROWTH writes saltPlumeFlux before KPP reads it). SEAICE_TRACER.h is not dumped: the Model's own
    initial SItracer (no dumped field reads it)."""
    from mitjax.tests import m4off_gate as G
    state, ff, phi0surf, flow, pk = G.teacher_carry(m, ds, it)
    pk = dict(pk)
    keys = set(ds.keys(it))
    gm = pk["gm"]
    pk["gm"] = gm.replace(**{n: G.like(getattr(gm, n), ds.field(it, "S00_begin", n)) for n in TENSOR
                             if n in gm and (it, "S00_begin", n) in keys})   # Kuz, Kvz: GM_EXTRA_DIAGONAL only
    if it > m.prm.time.nIter0:
        prev = set(ds.keys(it - 1))
        pk["kpp"] = dict(pk["kpp"], **{n: G.like(v, ds.field(it - 1, "P10_kpp_exch", n))
                                       for n, v in pk["kpp"].items() if (it - 1, "P10_kpp_exch", n) in prev})
    return (state, ff, phi0surf, flow, pk)


def ocean_step_fn(m, cfg=None, until="S05_thermodynamics_sync"):
    """jit(f)(arrays, carry, iloop, myTime, myIter) -> probes: FORWARD_STEP up to `until` with every probe kept, the
    port's own SEAICE_MODEL included (session 4; sessions 2-3 put the oracle's P13 values in its place). `cfg`: a
    planted CPP view for negative controls (m4col_gate.CppPlant)."""
    import jax
    from mitjax.model.src.forward_step import forward_step
    cfg = m.cfg if cfg is None else cfg
    fp, pks = m.fp, m.pks

    def f(a, carry, iloop, myTime, myIter):
        probes = {}

        def probe(stage, values):
            key = f"{stage[0]}@{stage[1]}" if isinstance(stage, tuple) else stage
            probes[key] = dict(values) if isinstance(stage, tuple) else values
        state, ff, phi0surf, flow, pk = carry
        probes["_in"] = dict(ff=ff, phi0surf=phi0surf, gm=pk.get("gm"), grid=a.grid)
        forward_step(iloop, myTime, myIter, cfg=cfg, grid=a.grid, params=a.params, fp=fp, eos=a.eos,
                     cg2dh=a.cg2dh, cg2d_params=a.cg2d_params, state=state, ff=ff, phi0surf=phi0surf, ex=a.ex,
                     pk=pk, pkc=a.pkc, pks=pks, probe=probe, until=until)
        return probes
    return jax.jit(f)


def ocean_diffs(m, ds, its=(1, 2, 3), cfg=None, arrays=None, carry_fix=None, stages=OCEAN_STAGES):
    """{it: {stage: {field: compare_field}}} of one step from the teacher carry (`ocean_inputs`) with the port's own
    sea ice: the ocean physics and thermodynamics (P01 .. S05) and whatever else `stages` names (DYN_STAGES + P13).
    `carry_fix(carry) -> carry`: a planted change of the inputs (negative controls)."""
    from mitjax.tests import goadk_model_gate as M
    f = ocean_step_fn(m, cfg=cfg)
    a = m.arrays if arrays is None else arrays
    tp = m.prm.time
    out = {}
    for it in its:
        carry = ocean_inputs(m, ds, it)
        if carry_fix is not None:
            carry = carry_fix(carry)
        k = it - tp.nIter0
        probes = f(a, carry, jnp.int32(k + 1), jnp.float64(tp.startTime + tp.deltaTClock*k), jnp.int32(it))
        res = M.compare_step(m, ds, it, probes)
        out[it] = {s: res[s] for s in stages if s in res}
    return out


def n_bad(res):
    """{(stage, field): differing points} of a compare_step result restricted to the differing entries."""
    return {(s, n): (v if v[0] == "shape" else sum(v[1:])) for s, d in res.items() for n, v in d.items()
            if v[0] == "shape" or any(v[1:])}


def gm_tensor_fn(m, cfg=None):
    """jit(f)(arrays, gm, state, sigmaX, sigmaY, sigmaR, kppf, myTime, myIter) -> GMREDI.h: GMREDI_CALC_TENSOR as
    DO_OCEANIC_PHYS calls it (the whole tile, :1031-1037)."""
    import jax
    from mitjax.pkg.gmredi.gmredi_calc_tensor import gmredi_calc_tensor
    cfg = m.cfg if cfg is None else cfg
    sz = cfg.size

    def f(a, gm, state, sx, sy, sr, kppf, myTime, myIter):
        return gmredi_calc_tensor(1-sz.OLx, sz.sNx+sz.OLx, 1-sz.OLy, sz.sNy+sz.OLy, sx, sy, sr, myTime, myIter,
                                  cfg=cfg, grid=a.grid, params=a.params, gm=gm, state=state, kppf=kppf)
    return jax.jit(f)


def gm_tensor_diffs(m, ds, its=(1, 2, 3), cfg=None, gm_fix=None, hbl_name="KPPhbl"):
    """{it: {field: differing points}} of P05_gmredi_tensor, GMREDI_CALC_TENSOR teacher-forced from the oracle's
    sigmaX/Y/R (P02), hMixLayer (P03), KPPhbl (P07) and the tensor before the step (S00_begin). `hbl_name`
    "hMixLayer" passes P03's hMixLayer as KPPhbl (the control of the locMixLayer arm)."""
    from mitjax.tests import m4off_gate as G
    f = gm_tensor_fn(m, cfg=cfg)
    tp = m.prm.time
    out = {}
    for it in its:
        state, ff, phi0surf, flow, pk = ocean_inputs(m, ds, it)
        gm = pk["gm"] if gm_fix is None else gm_fix(pk["gm"])
        st = state.replace(hMixLayer=G.like(state.hMixLayer, ds.field(it, "P03_mxlayer", "hMixLayer")))
        sig = [G.like(state.theta, ds.field(it, "P02_rho_sigma_ivdc", n)) for n in ("sigmaX", "sigmaY", "sigmaR")]
        src = ("P07_kpp", "KPPhbl") if hbl_name == "KPPhbl" else ("P03_mxlayer", "hMixLayer")
        kppf = {"KPPhbl": G.like(pk["kpp"]["KPPhbl"], ds.field(it, *src))}
        k = it - tp.nIter0
        gm2 = f(m.arrays, gm, st, *sig, kppf, jnp.float64(tp.startTime + tp.deltaTClock*k), jnp.int32(it))
        out[it] = {n: ndiff(ds.field(it, "P05_gmredi_tensor", n), getattr(gm2, n)) for n in TENSOR
                   if n in gm2 and (it, "P05_gmredi_tensor", n) in set(ds.keys(it))}
    return out


# ------------------------------------------------------------------- session 3: the sea-ice gaps
SF_I04 = ("AREA", "HEFF", "HSNOW", "TICES", "UICE", "VICE", "d_HEFFbyNEG", "d_HSNWbyNEG", "frWtrIce", "saltWtrIce")
FF_I04 = ("EmPmR", "Qnet", "Qsw", "fu", "fv", "pLoad", "sIceLoad", "saltFlux")
SP_I04 = ("saltPlumeFlux", "SaltPlumeDepth")


def clock(m, it):
    """(myTime, myIter) at the start of iteration `it` (lab_sea starts at nIter0 = 1, startTime = nIter0*deltaT)."""
    tp = m.prm.time
    return jnp.float64(tp.startTime + tp.deltaTClock*(it - tp.nIter0)), jnp.int32(it)


def growth_fn(m, cfg=None):
    """jit(f(sp, op, spp, sf, ff, exf, st, salt_plume, myTime, myIter) -> (sf, ff, salt_plume)) of SEAICE_GROWTH with
    ALLOW_SALT_PLUME (`cfg`: a planted CPP view for negative controls)."""
    import jax
    from mitjax.pkg.seaice.seaice_growth import seaice_growth
    pkc = m.arrays.pkc
    cfg = m.cfg if cfg is None else cfg

    def f(sp, op, spp, sf, ff, exf, st, salt_plume, myTime, myIter):
        return seaice_growth(myTime, myIter, sf, ff, exf, cfg=cfg, sp=sp, op=op, grid=m.arrays.grid, state=st,
                             exfp=pkc["exfp"], salt_plume=salt_plume, spp=spp)
    return jax.jit(f)


def growth_diffs(m, ds, its=(1, 2, 3), sp=None, spp=None, fn=None):
    """{it: {field: differing points}} of SEAICE_GROWTH teacher-forced from the oracle's fields before I04_growth
    (m4off_gate.inputs_at: I03_reg_ridge's SEAICE.h, the FFIELDS.h / EXF_FIELDS.h of earlier stages, theta/salt of
    S00_begin; SALT_PLUME.h from the previous iteration's I04, at the first iteration the Model's zero commons)
    against I04_growth, every point incl. halos (SEAICE.h, FFIELDS.h and SALT_PLUME.h fields)."""
    from mitjax.tests import m4off_gate as G
    pkc = m.arrays.pkc
    sp = pkc["sp"] if sp is None else sp
    spp = pkc["spp"] if spp is None else spp
    fn = growth_fn(m) if fn is None else fn
    out = {}
    for it in its:
        sf, ff, exf, st = G.inputs_at(m, ds, it, "I04_growth")
        spl = {}
        for n, v in m.pk0["salt_plume"].items():
            a = G.before(ds, it, "I04_growth", n)
            spl[n] = v if a is None else G.like(v, a)
        sf2, ff2, spl2 = fn(sp, pkc["op"], spp, sf, ff, exf, st, spl, *clock(m, it))
        vals = {n: sf2[n] for n in SF_I04}
        vals.update({n: getattr(ff2, n) for n in FF_I04})
        vals.update({n: spl2[n] for n in SP_I04})
        out[it] = G.differing(ds, it, "I04_growth", vals)
    return out


def advdiff_fn(m):
    """jit(f(sp, op, sf, myTime, myIter) -> sf) of SEAICE_ADVDIFF(UICE, VICE) as SEAICE_MODEL calls it
    (seaice_model.F:230)."""
    import jax
    from mitjax.pkg.seaice.seaice_advdiff import seaice_advdiff

    def f(sp, op, sf, myTime, myIter):
        return seaice_advdiff(sf["UICE"], sf["VICE"], myTime, myIter, sf, cfg=m.cfg, sp=sp, op=op,
                              grid=m.arrays.grid)
    return jax.jit(f)


def advdiff_diffs(m, ds, its=(1, 2, 3), sp=None, fn=None):
    """{it: {field: differing points}} of SEAICE_ADVDIFF teacher-forced from the oracle's SEAICE.h before I02_advdiff
    (I01_dynsolver: UICE, VICE, HEFF, AREA, HSNOW) against I02_advdiff, every point incl. halos."""
    from mitjax.tests import m4off_gate as G
    pkc = m.arrays.pkc
    sp = pkc["sp"] if sp is None else sp
    fn = advdiff_fn(m) if fn is None else fn
    out = {}
    for it in its:
        sf, _, _, _ = G.inputs_at(m, ds, it, "I02_advdiff")
        sf2 = fn(sp, pkc["op"], sf, *clock(m, it))
        out[it] = G.differing(ds, it, "I02_advdiff", {n: sf2[n] for n in ("AREA", "HEFF", "HSNOW", "TICES", "UICE",
                                                                          "VICE")})
    return out


def freedrift_fn(m):
    """jit(f(sp, op, sf, st, myTime, myIter) -> sf) of SEAICE_FREEDRIFT (seaice_dynsolver.F:307)."""
    import jax
    from mitjax.pkg.seaice.seaice_freedrift import seaice_freedrift

    def f(sp, op, sf, st, myTime, myIter):
        return seaice_freedrift(myTime, myIter, sf, cfg=m.cfg, sp=sp, op=op, grid=m.arrays.grid, state=st, ex=m.ex)
    return jax.jit(f)


def freedrift_diffs(m, ds, its=(1, 2, 3), sp=None, fn=None):
    """{it: {field: differing points}} of SEAICE_FREEDRIFT teacher-forced from the oracle's SEAICE.h at
    Y02_ice_strength (FORCEX0, FORCEY0, HEFF; the masks) and uVel/vVel of S00_begin, against Y03_freedrift
    (uice_fd, vice_fd), every point incl. halos."""
    from mitjax.tests import m4off_gate as G
    pkc = m.arrays.pkc
    sp = pkc["sp"] if sp is None else sp
    fn = freedrift_fn(m) if fn is None else fn
    out = {}
    for it in its:
        sf, _, _, st = G.inputs_at(m, ds, it, "Y03_freedrift")
        sf2 = fn(sp, pkc["op"], sf, st, *clock(m, it))
        out[it] = G.differing(ds, it, "Y03_freedrift", {n: sf2[n] for n in ("uice_fd", "vice_fd")})
    return out


DYN_STAGES = ("Y01_get_dynforcing", "Y02_ice_strength", "Y03_freedrift", "Y04_solver_inputs", "Y06_lsr",
              "Y09_ocean_stress", "I00_seaice_begin", "I01_dynsolver", "I02_advdiff", "I03_reg_ridge", "I04_growth")


def lsr_fn(m):
    """jit(f(sp, op, sf, st, myTime, myIter) -> (sf, out)) of SEAICE_LSR (the forward-only wrapper)."""
    import jax
    from mitjax.ad.seaice_lsr_rule import seaice_lsr_forward_only

    def f(sp, op, sf, st, myTime, myIter):
        return seaice_lsr_forward_only(myTime, myIter, sf, cfg=m.cfg, sp=sp, op=op, grid=m.arrays.grid, state=st,
                                       ex=m.ex)
    return jax.jit(f)


def lsr_diffs(m, ds, its=(1, 2, 3), sp=None, fn=None):
    """{it: ({field: differing points} at Y06_lsr, out)}: SEAICE_LSR teacher-forced from the oracle's SEAICE.h before
    Y06_lsr (Y04_solver_inputs) and uVel/vVel of S00_begin, every SEAICE.h field the stage dumps, incl. halos."""
    import jax
    from mitjax.tests import m4off_gate as G
    pkc = m.arrays.pkc
    sp = pkc["sp"] if sp is None else sp
    fn = lsr_fn(m) if fn is None else fn
    res = {}
    for it in its:
        sf, _, _, st = G.inputs_at(m, ds, it, "Y06_lsr")
        sf2, out = fn(sp, pkc["op"], sf, st, *clock(m, it))
        res[it] = (G.differing(ds, it, "Y06_lsr", {n: v for n, v in sf2.items() if hasattr(v, "data")}),
                   jax.tree_util.tree_map(jax.device_get, out))
    return res


def seaice_model_fn(m, cfg=None):
    """jit(f(sp, op, spp, sf, ff, exf, st, salt_plume, myTime, myIter) -> {stage: {name: FArray}}) of SEAICE_MODEL
    with every probe; "P13_seaice_model" holds the returned SEAICE.h, FFIELDS.h, EXF_FIELDS.h and SALT_PLUME.h."""
    import jax
    from mitjax.pkg.seaice.seaice_model import seaice_model
    pkc = m.arrays.pkc
    cfg = m.cfg if cfg is None else cfg

    def f(sp, op, spp, sf, ff, exf, st, salt_plume, myTime, myIter):
        out = {}

        def probe(stage, sf_, ff_, exf_):
            out[stage] = {**exf_, **{n: getattr(ff_, n) for n in ff_.names()}, **sf_}
        sf2, ff2, exf2, spl2 = seaice_model(myTime, myIter, sf, ff, exf, cfg=cfg, sp=sp, op=op, grid=m.arrays.grid,
                                            state=st, exfp=pkc["exfp"], ex=m.ex, kgeo=None, probe=probe,
                                            salt_plume=salt_plume, spp=spp)
        out["P13_seaice_model"] = {**exf2, **{n: getattr(ff2, n) for n in ff2.names()}, **sf2, **spl2}
        return out
    return jax.jit(f)


def seaice_model_diffs(m, ds, its=(1, 2, 3), sp=None, fn=None):
    """{it: {stage: {field: differing points}}} of SEAICE_MODEL teacher-forced from the oracle's fields before
    I00_seaice_begin (theta, salt, uVel, vVel, etaN of S00_begin; SALT_PLUME.h of the previous I04), compared at every
    dumped sea-ice stage and P13 (the fields the stage dumps and the port carries), every point incl. halos."""
    from mitjax.tests import m4off_gate as G
    pkc = m.arrays.pkc
    sp = pkc["sp"] if sp is None else sp
    fn = seaice_model_fn(m) if fn is None else fn
    out = {}
    for it in its:
        sf, ff, exf, st = G.inputs_at(m, ds, it, "I00_seaice_begin")
        spl = {}
        for n, v in m.pk0["salt_plume"].items():
            a = G.before(ds, it, "I00_seaice_begin", n)
            spl[n] = v if a is None else G.like(v, a)
        res = fn(sp, pkc["op"], pkc["spp"], sf, ff, exf, st, spl, *clock(m, it))
        out[it] = {stage: G.differing(ds, it, stage, {n: v for n, v in vals.items() if hasattr(v, "data")})
                   for stage, vals in res.items()}
    return out


# ------------------------------------------------------------------- session 4: lab_sea/input end to end
DT = 3600.0                      # lab_sea/input/data PARM03: deltaTClock = 3600.0 (startTime 3600., endTime 36000.)


def monitor_records(records):
    """Every %MON line and MONITOR banner of a run's STDOUT records, in order (m4off_gate.monitor_records)."""
    return [r for r in records if "%MON " in r or ("// " in r and " MONITOR " in r)]


def solver_records(records):
    """The solver lines of a run's STDOUT records: CG2D's (cg2d_init_res / Sum(rhs) / the cg2d_* monitor lines are
    %MON; the plain ones are 'cg2d:') and SEAICE_LSR's printed lines, in order."""
    return [r for r in records if "%MON" not in r and ("cg2d" in r.lower() or "SEAICE_LSR" in r)]


def whole_run(tag="m4lab-whole", plant_off=()):
    """(driver Model, forward result, oracle, run_verdict tuple): the 9-step run of lab_sea/input through the run
    driver (cube_run_gate.whole_run: make_rundir + drivers.run.forward, as `python -m mitjax run`), against lane A's
    run (monitor_gate.oracle: the dumps-on run, STDOUT identical to the plain one, invisibility-job27855987.txt).
    `plant_off`: CPP options planted off (m4col_gate.CppPlant; negative controls)."""
    import mitjax.drivers.run as RR
    from mitjax.tests import advect_gate as ag
    from mitjax.tests import cube_run_gate as CR
    from mitjax.tests.m4col_gate import CppPlant
    from mitjax.xla_flags import set_gate_xla_flags
    set_gate_xla_flags()
    orig = RR.load_experiment
    if plant_off:
        def planted(exp_dir, variant):
            e = orig(exp_dir, variant)
            return dataclasses.replace(e, cfg=dataclasses.replace(e.cfg, cpp=CppPlant(e.cfg.cpp, off=plant_off)))
        RR.load_experiment = planted              # read by cube_run_gate.whole_run's import at call time
    try:
        m, res, o = CR.whole_run(*EXP, tag=tag)
    finally:
        RR.load_experiment = orig
    return m, res, o, ag.run_verdict(*EXP, res, o)


def pickup_records(path):
    """{field name: bytes of its record(s)} of an MDS pickup file (the meta file's fldList; equal-size records)."""
    from pathlib import Path
    path = Path(path)
    meta = path.with_suffix(".meta").read_text()
    names = meta[meta.index("fldList"):].split("{")[1].split("}")[0].split("'")[1::2]
    raw = path.read_bytes()
    nrec = int(meta[meta.index("nrecords"):].split("[")[1].split("]")[0])
    size = len(raw) // nrec
    out, k = {}, 0
    nlev = (nrec - len(names)) + 1 if nrec != len(names) else 1        # one 3-D field (siTICES) holds nlev records
    for n in names:
        w = nlev if (nrec != len(names) and n.strip() == "siTICES") else 1
        out[n.strip()] = raw[k*size:(k + w)*size]
        k += w
    assert k == nrec, (names, nrec)
    return out


def run_dir_model(tag, overrides=None, links=()):
    """A driver Model of lab_sea/input on a new run directory (make_rundir, never reused), namelist `overrides`
    (drivers.model.with_namelist), extra files linked into it (m4off_gate.run_dir_model for this experiment)."""
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
    rd = make_rundir(exp_dir, EXP[1], P.RUNS / "tests_m4lab" / f"{tag}-{os.getpid()}-{time.time_ns()}")
    for s in links:
        os.symlink(s, rd / Path(s).name)
    return DriverModel(e, rd)


_RESTART = {}


def restart_a(k=5):
    """Run A: the 9 steps from pickup.0000000001 with pChkptFreq = k*deltaT: permanent pickups at the iterations
    whose time is a multiple of it (k = 5: iterations 5 and 10); cached."""
    if k not in _RESTART:
        from mitjax.drivers.run import forward
        a = run_dir_model("restartA", {("data", "PARM03", "pChkptFreq"): k*DT})
        _RESTART[k] = (a, forward(a))
    return _RESTART[k]


def restart_b(tag, k=5, mutate=None):
    """Run B: startTime = k*deltaT (nIter0 = k), to endTime 36000 (10 - k steps), from copies of A's pickup.<k>,
    pickup_cd.<k> and pickup_seaice.<k> (mutated by `mutate(dir)` if given). Returns (A, A's result, B, B's result).
    A ran k - 1 steps from nIter0 = 1: (k - 1) + pickup + (10 - k) = 9 steps."""
    import os
    import time
    from pathlib import Path
    from mitjax import paths as P
    from mitjax.drivers.run import forward
    a, ra = restart_a(k)
    d = P.RUNS / "tests_m4lab" / f"{tag}_pickup-{os.getpid()}-{time.time_ns()}"
    d.mkdir(parents=True)
    links = []
    for pre in ("pickup", "pickup_cd", "pickup_seaice"):
        for suf in ("data", "meta"):
            src = Path(a.rundir) / f"{pre}.{k:010d}.{suf}"
            dst = d / src.name
            dst.write_bytes(src.read_bytes())
            links.append(dst)
    if mutate:
        mutate(d)
    b = run_dir_model(tag, {("data", "PARM03", "pChkptFreq"): k*DT, ("data", "PARM03", "startTime"): k*DT}, links)
    return a, ra, b, forward(b)


def sharded_vs_single(m, nproc=4, nsteps=3):
    """{leaf: points that differ in bits} between `nsteps` of the driver's step (the whole carry incl. the sea-ice
    package state and SEAICE_TRACER.h) at P=nproc (TileSharding over the 4 tiles of 10x8 of the registered map
    lab_sea-t4_10x8_ol4x4.npz, jit(shard_map(check_vma=True)) with the ShardedExchanger) and at P=1 (the same step,
    jitted); m4off_gate.sharded_vs_single for this experiment."""
    import jax
    from mitjax.drivers.sharded_grad import place_model, tile_specs
    from mitjax.eesupp import exch_maps as EM
    from mitjax.eesupp.shard import TileSharding
    from mitjax.tests import r2_gate as r2
    a1 = m.arrays
    sh = TileSharding(EM.load_maps(EXP[0]), nproc)
    carry0 = m.initial_carry()

    def body(a, carry, iloop, myTime, myIter):
        carry, myTime, myIter, _ = m.step(a, carry, iloop, myTime, myIter)
        return carry, myTime, myIter
    cs = tile_specs(sh, carry0)
    f4 = sh.shard_map(body, in_specs=(tile_specs(sh, a1), cs, sh.REP, sh.REP, sh.REP), out_specs=(cs, sh.REP, sh.REP))
    f1 = jax.jit(body)
    a4, c4, c1 = place_model(sh, a1), sh.put_tree(carry0), carry0
    t1, it1 = m.start_counters()
    t4, it4 = t1, it1
    for k in range(nsteps):
        c1, t1, it1 = f1(a1, c1, jnp.int32(k + 1), t1, it1)
        c4, t4, it4 = f4(a4, c4, jnp.int32(k + 1), t4, it4)
    return r2.tree_bits_differ(sh.unpad_tree(c4), jax.tree.map(np.asarray, c1))


# ------------------------------------------------------------------- session 4: SITRACER kernel gradients
def smooth_sitracer_sf(m, seed=3):
    """The Model's initial SEAICE.h with a smooth sea-ice state on every lane (halos included), away from every
    switch of the SITRACER kernels: HEFF = 1 m (+ 5 % noise), AREA = 0.9 (+- 0.02), the tracers SItracer(:,:,1) =
    5000 s and (:,:,2) = 1 (+ 1 % noise, so ADVCAP's neighbourhood maximum is decided by margins), the SEAICE_GROWTH
    snapshots strictly monotone between the levels (SItrHEFF 1.00, 1.10, 1.05, 1.20, 1.30 m: growth, melt, growth,
    growth; SItrAREA 0.95, 0.85, 0.90: ridging then expansion) and small ice velocities (2 cm/s)."""
    rng = np.random.default_rng(seed)
    sf = dict(m.pk0["seaice"])

    def rep(name, f):
        v = sf[name]
        sf[name] = type(v)(jnp.asarray(f(np.asarray(v.data).shape)), v.name, tiled=v.tiled, _dims=v.dims)
    rep("HEFF", lambda s: 1.0 + 0.05*rng.random(s))
    rep("AREA", lambda s: 0.88 + 0.04*rng.random(s))
    rep("UICE", lambda s: 0.02*(rng.random(s) - 0.5))
    rep("VICE", lambda s: 0.02*(rng.random(s) - 0.5))
    rep("SItracer", lambda s: np.array([5000., 1., 0.])[None, :, None, None]*(1. + 0.01*rng.random(s)))
    rep("SItrHEFF", lambda s: np.array([1.0, 1.1, 1.05, 1.2, 1.3])[None, :, None, None]*(1. + 0.001*rng.random(s)))
    rep("SItrAREA", lambda s: np.array([0.95, 0.85, 0.9])[None, :, None, None]*(1. + 0.001*rng.random(s)))
    return sf


def sitracer_kernel_fn(m, kernel):
    """f(x, sf) -> y: `kernel` "tracer_phys" (SEAICE_TRACER_PHYS; x = (SItracer, SItrHEFF, SItrAREA), y =
    (SItracer, SItrBucket)) or "advdiff" (SEAICE_ADVDIFF with its SITRACER part; x = (SItracer, HEFF, AREA, UICE,
    VICE), y = (SItracer, SItrBucket, HEFF, AREA)), on the [tile, ...] data of the fields (every lane)."""
    from mitjax.pkg.seaice.seaice_advdiff import seaice_advdiff
    from mitjax.pkg.seaice.seaice_tracer_phys import seaice_tracer_phys
    pkc = m.arrays.pkc
    names = {"tracer_phys": (("SItracer", "SItrHEFF", "SItrAREA"), ("SItracer", "SItrBucket")),
             "advdiff": (("SItracer", "HEFF", "AREA", "UICE", "VICE"), ("SItracer", "SItrBucket", "HEFF", "AREA"))}
    xin, yout = names[kernel]
    myTime, myIter = clock(m, 1)

    def f(x, sf):
        sf = dict(sf)
        for n, d in zip(xin, x):
            v = sf[n]
            sf[n] = type(v)(d, v.name, tiled=v.tiled, _dims=v.dims)
        if kernel == "tracer_phys":
            out = seaice_tracer_phys(myTime, myIter, sf, cfg=m.cfg, sp=pkc["sp"])
        else:
            out = seaice_advdiff(sf["UICE"], sf["VICE"], myTime, myIter, sf, cfg=m.cfg, sp=pkc["sp"], op=pkc["op"],
                                 grid=m.arrays.grid)
        return tuple(out[n].data for n in yout)
    return f, xin
