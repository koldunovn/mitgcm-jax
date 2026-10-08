"""Helpers of the R5 (tutorial_global_oce_optim/input_ad, code_ad) gates, plan Task 16.

Oracle: the CTRL lane's forward-only code_ad run with dumps at every step 0..9 (`job27829092-ctrlzero`, the variant
as testreport runs it: zero first-guess control; jaxdump binary 5f16129). Stage gates are teacher-forced: each step
starts from the oracle's own S00_begin dump of that step (State, FFIELDS.h, phi0surf, the GM/Redi tensor), so a
stage is gated on exactly the Fortran's inputs and an error cannot hide behind (or be blamed on) an earlier step.
The step runs through `forward_step(until=...)` -- the model's own sequence of routines -- and stops after the last
stage this lane can run (THERMODYNAMICS' TEMP_INTEGRATE: SALT_INTEGRATE is the ADVECT lane's, the CD scheme the GO
lane's; SALT_INTEGRATE arrived with the ADVECT merge: the front now ends with THERMODYNAMICS, S05). Every point of
every tile is compared, halos included, bit patterns too (r1_gate.compare_field).

Set-up, in INITIALISE_FIXED / INITIALISE_VARIA order where it matters: INI_PARMS (+ the tracer and forcing groups),
the grid of the Task 10 port, INI_LINEAR_PHISURF, INI_EOS, GMREDI_READPARMS / GMREDI_INIT_FIXED / GMREDI_INIT_VARIA,
INI_FFIELDS + INI_FORCING, the host preload of the periodic forcing for steps 1..10 (external_fields_load.
preload_periodic_forcing), CTRL_MAP_INI_GENTIM2D on the run's control file (zero) and weight.
"""

from functools import lru_cache
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.farray import FArray, loop_i, loop_j
from mitjax.io.dump import DumpSet
from mitjax.model.grid import UNSET_RL, declare
from mitjax.tests import ctrl_cost_gate as cg
from mitjax.tests import grid_gate as gg
from mitjax.tests import init_gate as ig
from mitjax.tests import r1_gate as rg

EXP = ("tutorial_global_oce_optim", "input_ad")
NSTEPS = 10

# the stages of one step this lane's code reaches (teacher-forced from S00_begin), in the order of the step
FRONT = ("S02_load_fields", "S03_ctrl_map_forcing", "P01_external_forcing_surf", "P02_rho_sigma_ivdc",
         "P03_mxlayer", "P05_gmredi_tensor", "P06_gmredi_exch", "S04_oceanic_phys", "T01_residual_flow",
         "T11_temp_gT", "T12_temp_step", "T13_temp_impl", "T02_temp_integrate", "T21_salt_gS", "T22_salt_step",
         "T23_salt_impl", "T03_salt_integrate", "S05_thermodynamics_sync")
UNTIL = "S05_thermodynamics_sync"
# DYNAMICS (CD scheme in TIMESTEP, GO lane's pkg/cd_code) up to S06; its per-level stages D00a/D00b are not probed
# through forward_step (other lanes' gates key their probes by stage name), S06 holds DYNAMICS' outputs
DYN = ("S06_dynamics",)
# the rest of the step (C01/C02 are SOLVE_FOR_PRESSURE-internal stages forward_step does not probe: S09 holds their
# outputs, the cg2d scalars are compared from the step's out["cg2d"], `cg2d_scalars`)
BACK = ("S09_solve_for_pressure", "S10_momentum_correction", "S11_integr_continuity", "S15_tracers_correction",
        "S16_blocking_exchanges", "S18_cost_tile")
STEP = FRONT + DYN + BACK
# cost.h fields COST_INIT_VARIA does not initialise (NaN by design, cost_init_varia): first written by COST_FINAL
# (the oracle's 0 at S18 is the common block's load value); compared at E01 by the whole-run gate
S18_EXEMPT = ("objf_temp_tut", "objf_hflux_tut")
CD_FIELDS = ("uVelD", "vVelD", "uNM1", "vNM1", "etaNm1")
TENSOR = ("Kwx", "Kwy", "Kwz", "Kux", "Kvy", "Kuz", "Kvz")


class Model:
    """The R5 set-up (module docstring); `kind` names the ctrl oracle run (ctrl_cost_gate.RUNS)."""

    def __init__(self, kind="zero"):
        from mitjax.model.src.external_fields_load import preload_periodic_forcing
        from mitjax.model.src.ini_eos import ini_eos
        from mitjax.model.src.ini_ffields import ini_ffields
        from mitjax.model.src.ini_forcing import ini_forcing
        from mitjax.model.src.ini_grid import ini_grid
        from mitjax.model.src.ini_linear_phisurf import ini_linear_phisurf
        from mitjax.model.src.ini_parms import ini_parms_dyn
        from mitjax.model.src.ini_parms_forcing import ini_parms_forcing
        from mitjax.model.src.ini_parms_tracer import ini_parms_tracer
        from mitjax.params_io import RunParams
        from mitjax.pkg.gmredi.gmredi_init_fixed import gmredi_init_fixed
        from mitjax.pkg.gmredi.gmredi_init_varia import gmredi_init_varia
        from mitjax.pkg.gmredi.gmredi_readparms import gmredi_readparms
        from mitjax.pkg.ctrl.ctrl_readparms import ctrl_readparms_gentim2d
        from mitjax.pkg.rw.read_rec import RW
        self.kind = kind
        e = self.e = gg.experiment(*EXP)
        cfg = self.cfg = e.cfg
        self.prm = ig.params(*EXP)
        tp = self.prm.time
        params = ini_parms_dyn(e, self.prm.grid, tp, self.prm.init)
        self.params = ini_parms_tracer(e, params, tp, self.prm.init)
        self.fp = ini_parms_forcing(e)
        self.ex = gg.exchanger(EXP[0])
        self.top = cg.run_top(kind)
        self.ds = DumpSet(self.top / "dumps")
        self.rundir = self.top / "rundir"
        self.rw = RW(self.rundir, self.prm.init.readBinaryPrec, cfg.size)
        grid = gg.build_grid(*EXP, params=self.prm.grid, ex=self.ex)
        self.grid = ini_linear_phisurf(grid, cfg=cfg, params=self.params)               # initialise_fixed.F:226
        from mitjax.model.src.ini_parms import set_ref_state_eos
        self.params = set_ref_state_eos(self.params, self.grid, e)                     # set_ref_state.F:51-100
        rp = RunParams(e.run)
        eos_p = SimpleNamespace(fluidIsWater=self.params.fluidIsWater, usingPCoords=self.params.usingPCoords,
                                eosType=self.params.eosType,
                                tAlpha=rp.get("data", "PARM01", "tAlpha") if rp.has("data", "PARM01", "tAlpha")
                                else UNSET_RL,                                           # ini_parms.F:421
                                sBeta=rp.get("data", "PARM01", "sBeta") if rp.has("data", "PARM01", "sBeta")
                                else UNSET_RL)                                           # ini_parms.F:422
        self.eos = ini_eos(cfg=cfg, params=eos_p)                                        # initialise_fixed.F:176
        self.gm0 = gmredi_init_varia(gmredi_init_fixed(gmredi_readparms(e), cfg=cfg), cfg=cfg)
        ff = ini_ffields(cfg=cfg)                                                        # initialise_varia.F:213
        _, _, _, lat = ini_grid(cfg=cfg, params=self.prm.grid)
        self.ff0 = ini_forcing(ff, cfg=cfg, grid=self.grid, fp=self.fp, rw=self.rw, ex=self.ex,
                               latBandClimRelax=lat)                                     # initialise_varia.F:242
        self.stdout = []
        self.pre, _ = preload_periodic_forcing(self.ff0, NSTEPS, cfg=cfg, fp=self.fp, rw=self.rw, ex=self.ex,
                                               nIter0=tp.nIter0, startTime=tp.startTime,
                                               deltaTClock=tp.deltaTClock, stdout=self.stdout)
        self.gentim2d = ctrl_readparms_gentim2d(e.run)
        self.eff, _, self.genarr0 = cg.ctrl_init(kind)
        self.clock = cg.clock(kind)
        # CG2D.h (INI_CG2D, initialise_fixed.F:246) for SOLVE_FOR_PRESSURE; cost.h (COST_INIT_VARIA,
        # COST_DEPENDENT_INIT) for COST_TILE; data.cost lastinterval
        from mitjax.model.src.cg2d_h import ini_parms_cg2d
        from mitjax.model.src.ini_cg2d import ini_cg2d
        from mitjax.pkg.cost.cost_init_varia import cost_dependent_init, cost_init_varia
        self.cg2d_params = ini_parms_cg2d(e)
        self.cg2dh = jax.jit(lambda g, s_, p_, x: ini_cg2d(cfg=cfg, grid=g, surface=s_, params=p_, ex=x))(
            self.grid, self.grid, self.cg2d_params, self.ex)
        self.cost0 = cost_dependent_init(cost_init_varia(cfg=cfg, ntiles=cfg.size.nSx * cfg.size.nSy), cfg=cfg)
        self.lastinterval = float(cg.cost_params()["lastinterval"])
        sz = cfg.size
        j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
        i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
        self.phi0surf0 = declare("Bo_surf", sz).at[i, j].set(0.)

    # ---- teacher forcing: the start of step `it` (0-based) from the oracle's S00_begin dump
    def start(self, it, cd=None):
        """(state, ff, phi0surf, gm) at the start of step `it` from the oracle's S00_begin dump. CD_CODE_VARS.h
        (uVelD, vVelD, uNM1, vNM1, etaNm1) is not dumped: at it = 0 it is CD_CODE_INI_VARS' (zeros, nIter0 = 0),
        later it must be given (`cd`: the fields our own previous step left)."""
        from mitjax.model.state import empty_state
        cfg, sz = self.cfg, self.cfg.size
        st = empty_state(cfg)
        vals = {}
        for (i_, s_, n) in self.ds.keys(it):
            if s_ == "S00_begin" and n in st:
                f = getattr(st, n)
                vals[n] = FArray(jnp.asarray(self.ds.field(it, "S00_begin", n)).reshape(f.data.shape), n,
                                 tiled=f.tiled, _dims=f.dims)
        state = st.replace(**vals)
        if self.cfg.cpp.ALLOW_CD_CODE and self.params.useCDscheme:
            if cd is not None:
                state = state.replace(**cd)
            elif it == 0:
                from mitjax.pkg.cd_code.cd_code_ini_vars import cd_code_ini_vars
                state = cd_code_ini_vars(state, cfg=self.cfg, params=self.prm, ex=self.ex, rw=self.rw)
            # else: left NaN (empty_state) -- read first by DYNAMICS; a gate past S05 passes `cd`
        ff = self.ff0
        fv = {}
        for n in ff.names():
            if ("S00_begin", n) in {(s_, m) for (_, s_, m) in self.ds.keys(it)}:
                f = getattr(ff, n)
                fv[n] = FArray(jnp.asarray(self.ds.field(it, "S00_begin", n)).reshape(f.data.shape), n,
                               tiled=f.tiled, _dims=f.dims)
        ff = ff.replace(**fv)
        p = self.phi0surf0
        phi0surf = FArray(jnp.asarray(self.ds.field(it, "S00_begin", "phi0surf")).reshape(p.data.shape),
                          p.name, tiled=p.tiled, _dims=p.dims)
        gm = self.gm0.replace(**{n: _wrap(getattr(self.gm0, n), self.ds.field(it, "S00_begin", n))
                                 for n in TENSOR if hasattr(self.gm0, n) and getattr(self.gm0, n) is not None})
        return state, ff, phi0surf, gm

    def front_fn(self, until, stages=FRONT):
        """jit(step)(arrays..., state, ff, phi0surf, gm, genarr, iloop) -> (outputs, probes): FORWARD_STEP of the
        R5 build up to `until`; every float a traced argument (grid, params, eos, gm, preload, control records)."""
        from mitjax.model.src.forward_step import forward_step
        cfg, fp = self.cfg, self.fp
        pks = dict(gentim2d=self.gentim2d, clock=self.clock, endTime=float(self.prm.time.endTime),
                   lastinterval=float(self.lastinterval))

        def step(grid, params, eos, ex, pre, eff, cg2dh, cg2d_params, lastinterval, state, ff, phi0surf, gm, genarr,
                 cost, iloop):
            probes = {"_phi0surf_in": phi0surf}

            def probe(stage, values):
                if isinstance(stage, tuple):                    # DYNAMICS' per-level stages: (name, k)
                    name, k = stage
                    if name in stages:
                        d = probes.setdefault(name, {})
                        d.update({f"{n}_k{k:03d}": _level_of(v, k) for n, v in values.items()})
                elif stage in stages:
                    probes[stage] = values
            out = forward_step(iloop, params.startTime + params.deltaTClock*(iloop - 1).astype(jnp.float64),
                               params.nIter0 + iloop - 1, cfg=cfg, grid=grid, params=params, fp=fp, eos=eos,
                               cg2dh=cg2dh, cg2d_params=cg2d_params, state=state, ff=ff, phi0surf=phi0surf,
                               ex=ex, probe=probe, pk=dict(gm=gm, genarr=genarr, cost=cost),
                               pkc=dict(forcing=pre, effective=eff, lastinterval=lastinterval), pks=pks, until=until)
            return out, probes
        return jax.jit(step)

    def run_front(self, it, fn, *, params=None, pre=None, eff=None, gm_patch=None, genarr=None, cd=None,
                  cost=None):
        """Step `it` (0-based) teacher-forced; params / pre / eff / gm_patch(gm) replace the set-up's (planted
        errors of the negative controls). `genarr`: CTRL_GENARR.h at the start of the step (default: as after
        CTRL_MAP_INI_GENTIM2D, right for step 0; later steps of a nonzero control need the previous step's,
        out[5]["pk"]["genarr"])."""
        state, ff, phi0surf, gm = self.start(it, cd=cd)
        if gm_patch is not None:
            gm = gm_patch(gm)
        return fn(self.grid, self.params if params is None else params, self.eos, self.ex,
                  self.pre if pre is None else pre, self.eff if eff is None else eff, self.cg2dh, self.cg2d_params,
                  jnp.float64(self.lastinterval), state, ff, phi0surf, gm,
                  self.genarr0 if genarr is None else genarr, self.cost0 if cost is None else cost,
                  jnp.int32(it + 1))


def _level_of(v, k):
    """Level k of a 3-D FArray (the jaxdump per-level records hold `gU(1-OLx,1-OLy,k,bi,bj)`); 2-D as it is."""
    if len(v.dims) == 3:
        (_, ilo, ihi), (_, jlo, jhi), (_, klo, _) = v.dims
        return FArray(v.data[:, k - klo], v.name, i=(ilo, ihi), j=(jlo, jhi))
    return v


def _wrap(like, data):
    return FArray(jnp.asarray(data).reshape(like.data.shape), like.name, tiled=like.tiled, _dims=like.dims)


@lru_cache(maxsize=None)
def model(kind="zero"):
    return Model(kind)


def stage_values(stage, v):
    """The probe values of a stage as {name: FArray} for r1_gate.compare_stage."""
    if stage == "S04_oceanic_phys":
        s_, f_, p_ = v
        vals = rg.state_fields(s_)
        vals.update({n: getattr(f_, n) for n in f_.names()})
        vals["phi0surf"] = p_
        return vals
    if stage in ("P05_gmredi_tensor", "P06_gmredi_exch"):
        return {n: getattr(v, n) for n in TENSOR if getattr(v, n, None) is not None}
    if stage in ("S02_load_fields", "S03_ctrl_map_forcing", "P01_external_forcing_surf"):
        return {n: getattr(v, n) for n in v.names()}
    return v


def compare_front(m, it, probes, stages=FRONT):
    """{stage: {field: compare_field}} of step `it` against the oracle. S04 also compares the GMREDI.h tensor after
    GMREDI_DO_EXCH (the last writer inside DO_OCEANIC_PHYS) and the grid's hFacC/W/S, recip_hFacC (fixed fields the
    dump repeats)."""
    out = {}
    for st in stages:
        if st not in probes:
            continue
        vals = stage_values(st, probes[st])
        if st in ("S02_load_fields", "S03_ctrl_map_forcing"):
            vals = dict(vals, phi0surf=probes["_phi0surf_in"])       # not written before DO_OCEANIC_PHYS
        if st == "P01_external_forcing_surf" and "S04_oceanic_phys" in probes:
            vals = dict(vals, phi0surf=probes["S04_oceanic_phys"][2])  # EXTERNAL_FORCING_SURF: its only writer
        if st == "S04_oceanic_phys":
            vals = dict(vals)
            if "P06_gmredi_exch" in probes:
                vals.update(stage_values("P06_gmredi_exch", probes["P06_gmredi_exch"]))
            for n in ("hFacC", "hFacW", "hFacS", "recip_hFacC"):
                vals[n] = getattr(m.grid, n)
        if st == "S18_cost_tile":
            out[st] = _compare_cost(m, it, vals)
            continue
        out[st] = rg.compare_stage(m.ds, it, st, vals)
    return out


def _compare_cost(m, it, cost):
    """S18_cost_tile: the cost.h scalars the oracle dumps per tile (fc, glofc, tile_fc, objf_temp_tut,
    objf_hflux_tut; kind N records, one value per tile) against ours (fc, glofc scalars; the others [tile])."""
    out = {}
    for (_, s_, n) in m.ds.keys(it):
        if s_ != "S18_cost_tile" or not hasattr(cost, n) or n in S18_EXEMPT:
            continue
        recs = m.ds.tiles(it, s_, n)
        ref = np.array([recs[t].data.ravel()[0] for t in sorted(recs)])
        ours = np.broadcast_to(np.asarray(getattr(cost, n), np.float64), ref.shape)
        out[n] = rg.compare_field(ours[:, None, None, None], ref[:, None, None, None])
    return out


def missing(m, it, res, stages=FRONT):
    """The (stage, field) pairs the oracle dumps at `stages` of step `it` that `res` does not compare (S18_EXEMPT
    named above)."""
    return sorted((s_, n) for (_, s_, n) in m.ds.keys(it) if s_ in stages and n not in res.get(s_, {})
                  and not (s_ == "S18_cost_tile" and n in S18_EXEMPT))


def cg2d_scalars(m, it, cg2d):
    """{name: compare_field} of the CG2D solver scalars of the step's out["cg2d"] against C02_cg2d_solution."""
    out = {}
    for n in ("firstResidual", "minResidualSq", "lastResidual", "numIters", "nIterMin"):
        recs = m.ds.tiles(it, "C02_cg2d_solution", n)
        ref = np.array([recs[t].data.ravel()[0] for t in sorted(recs)])
        ours = np.broadcast_to(np.asarray(cg2d[n], np.float64), ref.shape)
        out[n] = rg.compare_field(ours[:, None, None, None], ref[:, None, None, None])
    return out


def run_steps(m, n, stages=STEP, fn=None):
    """Steps 0..n-1, each teacher-forced from the oracle's S00_begin dump, with the state the dumps do not hold
    (CD_CODE_VARS.h, CTRL_GENARR.h, cost.h) carried from our own previous step. Returns [(res, cg2d_res)]."""
    fn = fn or m.front_fn(None, stages=stages)
    cd = genarr = cost = None
    out = []
    for it in range(n):
        o, probes = m.run_front(it, fn, cd=cd, genarr=genarr, cost=cost)
        cd = {k: getattr(o[0], k) for k in CD_FIELDS}
        genarr, cost = o[5]["pk"]["genarr"], o[5]["pk"]["cost"]
        out.append((compare_front(m, it, probes, stages=stages), cg2d_scalars(m, it, o[5]["cg2d"])))
    return out


def bad(res):
    """The failing (stage, field) entries of compare_front's result."""
    out = {}
    for st, r in res.items():
        b = {k: v for k, v in r.items() if v[0] == "shape" or any(v[1:])}
        if b:
            out[st] = b
    return out


def n_fields(res):
    return sum(len(r) for r in res.values())


def n_oracle(m, it, stages=FRONT):
    """The number of fields the oracle dumps at `stages` of step `it` (every one must be compared)."""
    return sum(1 for (_, s_, _) in m.ds.keys(it) if s_ in stages)


# ---- whole run through the run driver (Model, forward) --------------------------------------------------------

def driver_model(tag="run", kind="zero"):
    """The run driver's Model (Model._packages incl. ctrl/cost) in a fresh run directory made by the driver's own
    make_rundir (linkdata + optim's prepare_run: *.bin from tutorial_global_oce_latlon/input, ones_64b.bin from
    isomip/input_ad). `kind` "zero" only (the variant as testreport runs it: first-guess control zeros)."""
    import os
    import time
    from mitjax import paths
    from mitjax.drivers.model import Model as DriverModel
    from mitjax.drivers.run import load_experiment, make_rundir
    if kind != "zero":
        raise ValueError("r5_gate.driver_model: only the zero-control variant")
    exp_dir = paths.UPSTREAM / "verification" / EXP[0]
    out = paths.RUNS / "tests_r5" / f"{tag}-{os.getpid()}-{time.time_ns()}"       # unique, never reused
    rundir = make_rundir(exp_dir, EXP[1], out)
    return DriverModel(load_experiment(exp_dir, EXP[1]), rundir)


def whole_run(tag="run", kind="zero"):
    """The whole 10-step run through mitjax/drivers (Model, forward incl. COST_FINAL), without pickups (none is
    due in 10 steps). Returns (driver Model, forward result, monitor_gate Oracle of the plain jdon run)."""
    from mitjax.drivers.run import forward
    from mitjax.tests import monitor_gate as mg
    m = driver_model(tag, kind)
    res = forward(m, write_pickups=False)
    return m, res, mg.oracle(*EXP)


# the grdchk points (pkg/grdchk/grdchk.py grdchk_points, test_ctrl_cost.py: the oracle's positions) as storage
# indices (tile, j, i) of the control record, and the TAF values of results/output_adm.txt (checkpoint68x)
GRDCHK_POINTS = ((0, 3, 44), (0, 3, 45), (0, 3, 46))
TAF_ADJOINT_GRADIENT = (-2.70384203444403E-06, -2.77397605795952E-06, -2.69091500991181E-06)
ORACLE_FD = (-2.70384161726867E-06, -2.77397556924797E-06, -2.69091464666360E-06)    # docs/YARDSTICK.md
REF_COST = 6.20023228182337E+00
