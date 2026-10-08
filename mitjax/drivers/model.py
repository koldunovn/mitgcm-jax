"""The host-side set-up of one forward run, in THE_MODEL_MAIN's order (plan Task 12), and the pure step function the
time loop scans.

    exp = mitjax.config.params.load(experiment, variant)       # the experiment's namelists and CPP options
    m = Model(exp, rundir)                                    # INI_PARMS, INITIALISE_FIXED, INITIALISE_VARIA
    carry0 = m.initial_carry()                                # (State, FFIELDS.h, phi0surf, THERMODYNAMICS' flow)
    step = m.step                                             # step(arrays, carry, iloop, myTime, myIter)

Fortran (@63cdc0b): the_model_main.F:452-470 INI_PARMS (in INITIALISE_FIXED), :528 INITIALISE_FIXED, :623-624 the
start counters, :628 THE_MAIN_LOOP; the_main_loop.F:386 INITIALISE_VARIA; initialise_fixed.F:156-246 (INI_GRID ...
INI_CORI, INI_EOS, INI_LINEAR_PHISURF, INI_CG2D); initialise_varia.F:199-334 (INI_FFIELDS :213, INI_FORCING :242,
INTEGR_CONTINUITY :334).

`Model.arrays` is everything FORWARD_STEP reads besides its carry -- GRID.h, the time-step PARAMS.h values, the EOS
coefficients, CG2D.h, the exchange maps -- as ONE pytree passed to the jitted programs as an ARGUMENT, never closed
over [F§1, L-ARCH-5]; the configuration (CPP options, SIZE.h, the forcing switches `fp`) is static and closed over by
`Model.step`. The carry is everything FORWARD_STEP carries from one call to the next: the State (DYNVARS.h,
SURFACE.h), FFIELDS.h (`ff`), phi0surf (SURFACE.h /SURF_FIXED/, written by EXTERNAL_FORCING_SURF) and THERMODYNAMICS'
flow (uFld, vFld, wFld: what MON_CALC_ADVCFL reads on the host at a monitor step, thermodynamics.F:263-283).

`with_namelist(exp, {(file, group, name): value})` is a changed copy of the experiment (the same as editing its
`data` file): the restart gate's nIter0 / nTimeSteps / pChkptFreq, the gradient gate's cg2dTargetResidual.
"""

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from mitjax.config.namelists import Var

# EEPARAMS.h:89-95 debLevZero=0, debLevA=1, debLevB=2 (the print levels the driver tests)
debLevZero, debLevA = 0, 1


def with_namelist(exp, overrides):
    """A copy of `exp` with namelist values replaced or added: {(file, group, name): value}; a value the file does
    not set needs the Fortran kind: {(file, group, name): (value, kind)} with kind in float/int/bool/str. The
    run's variable table (`exp.run.vars`), the static and float tables of `exp.cfg` / `exp.params` all change, as a
    changed `data` file would change them."""
    run, cfg = exp.run, exp.cfg
    vars_ = dict(run.vars)
    static = dict(cfg.static)
    floats = dict(exp.params.values)
    for (f, g, n), val in overrides.items():
        key = (f, g.lower(), n.lower())
        old = vars_.get(key)
        if old is None:
            if not isinstance(val, tuple):
                raise KeyError(f"{f}:{g}:{n} is not set in the experiment; pass (value, kind)")
            val, kind = val
            old = Var(f, g.lower(), n, kind, (), None, None, ())
        if old.shape:
            raise NotImplementedError(f"with_namelist: {f}:{g}:{n} is an array")
        vars_[key] = dataclasses.replace(old, value=val)
        if old.kind == "float":
            floats[f"{f}:{key[1]}:{key[2]}"] = jnp.float64(val)
        else:
            static[key] = val
    run2 = dataclasses.replace(run, vars=vars_)
    cfg2 = dataclasses.replace(cfg, static=tuple(sorted(static.items())))
    return dataclasses.replace(exp, cfg=cfg2, run=run2, params=dataclasses.replace(exp.params, values=floats))


@jax.tree_util.register_pytree_node_class
class Arrays:
    """What FORWARD_STEP reads besides its carry (module docstring): grid, params, eos, cg2dh, cg2d_params, ex, and
    `pkc` (R5: the traced package inputs of forward_step -- the periodic-forcing preload, the control records)."""

    names = ("grid", "params", "eos", "cg2dh", "cg2d_params", "ex", "pkc")

    def __init__(self, **kw):
        for n in self.names:
            object.__setattr__(self, n, kw.get(n, {}) if n == "pkc" else kw[n])

    def replace(self, **kw):
        return Arrays(**{n: kw.get(n, getattr(self, n)) for n in self.names})

    def tree_flatten(self):
        return tuple(getattr(self, n) for n in self.names), None

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        return cls(**dict(zip(cls.names, leaves)))


class Model:
    """One forward run of an experiment variant, set up eagerly on the host (module docstring).

    exp: mitjax.config.params.Experiment; rundir: the run directory (the experiment's input files linked in, as
    testreport's linkdata does; model input files are read relative to it, model output written there); ex: the
    exchanger (default: the experiment's registered exchange maps); exch2: Exch2Topology (default: the W2 set-up of
    the build, pkg/exch2, when ALLOW_EXCH2)."""

    def __init__(self, exp, rundir, *, ex=None, exch2=None, xx=None):
        rundir = Path(rundir)
        from mitjax.eesupp.exchange import Exchanger
        from mitjax.model.src.cg2d_h import ini_parms_cg2d
        from mitjax.model.src.ini_cg2d import ini_cg2d
        from mitjax.model.src.ini_eos import ini_eos
        from mitjax.model.src.ini_ffields import ini_ffields
        from mitjax.model.src.ini_forcing import ini_forcing
        from mitjax.model.src.ini_grid import ini_grid
        from mitjax.model.src.ini_linear_phisurf import ini_linear_phisurf
        from mitjax.model.src.ini_parms import ini_parms, ini_parms_dyn, ini_parms_io
        from mitjax.model.src.ini_parms_forcing import ini_parms_forcing
        from mitjax.model.src.initialise_varia import executed_pending, initialise_varia
        from mitjax.model.grid import UNSET_RL
        from mitjax.params_io import RunParams
        from mitjax.pkg.rw.read_rec import RW
        self.exp = exp
        self.w2 = None
        cfg = self.cfg = exp.cfg
        self.rundir = rundir
        if exch2 is None and cfg.cpp.ALLOW_EXCH2:
            from mitjax.pkg.exch2.w2_eeboot import exch2_topology, w2_eeboot
            self.w2 = w2_eeboot(exp)[0]                 # GO lane: W2 common blocks (MDS exch2 I/O layout)
            exch2 = exch2_topology(self.w2)
        # GO lane: a package switched on that the integrated model does not run is refused here, at set-up
        # (PORTING_RULES 1), not deep inside a step
        check_packages(cfg, exp=exp)
        # decision 11 (docs plan 20261006): an own main.F that skips the model (adjustment.cs-32x32x1/code_min) is
        # ported as its set-up only (mitjax/config/own_code.py MAIN_ONLY): no Model of that build
        from mitjax import paths
        from mitjax.config import own_code
        own = own_code.classify(cfg.exp_dir, cfg.code_dir, paths.UPSTREAM)
        if any(own.get(n, ("",))[0] == "ported" for n in own_code.MAIN_ONLY):
            raise own_code.UnportedRoutine(f"{cfg.exp_dir}/{cfg.code_dir}: its own main.F does not run the model "
                                           f"({own}); only its set-up is ported")
        # INI_PARMS (initialise_fixed.F:91) and the parameter groups of the other *_READPARMS
        self.prm = ini_parms(exp, exch2)
        self.params = ini_parms_dyn(exp, self.prm.grid, self.prm.time, self.prm.init)
        # ---- ADVECT lane (plan Task 14): the tracer-step values (TRACER lane's ini_parms_tracer, with the salinity
        # group and the AB3 weights) and the r* values (RSTAR lane's ini_parms_rstar), as the R2/R3 gates set them up
        from mitjax.model.src.ini_parms_rstar import ini_parms_rstar
        from mitjax.model.src.ini_parms_tracer import ini_parms_tracer
        self.params = ini_parms_tracer(exp, self.params, self.prm.time, self.prm.init)
        self.params = ini_parms_rstar(exp, self.params)
        self.params = ptracers_params(exp, self.params, self.prm)                   # PTRACERS lane
        self.params = rbcs_params(exp, self.params, self.prm)                       # M3 Task 30: RBCS_PARAMS.h
        self.fp = ini_parms_forcing(exp)
        self.io = ini_parms_io(exp)
        if dict(cfg.use).get("useMNC", False):            # GO lane (M1 acceptance): MNC_READPARMS pickup_write_mnc
            from mitjax.params_io import RunParams, fortran_default
            self.io = dataclasses.replace(self.io, pickup_write_mnc=bool(RunParams(exp.run).get(
                "data.mnc", "MNC_01", "pickup_write_mnc",
                default=fortran_default("pkg/mnc/mnc_readparms.F:97", "pickup_write_mnc", exp))))
        self.cg2d_params = ini_parms_cg2d(exp)
        self.exch_maps = None
        if ex is None:
            # the exchanges of the build replayed from the exchange code (docs plan 20261006 Task 3: exch1 / exch0 by
            # eesupp/exch1_tables.py, pkg/exch2 by pkg/exch2/exch2_cube_tables.py; cached by layout + topology in
            # $MJX_CACHE); lane A's probe maps (load_maps) are the replay's gate (test_exch_replay.py)
            from mitjax.eesupp.exch_maps import exchange_maps
            self.exch_maps = exchange_maps(exp)
            ex = Exchanger(self.exch_maps)
        self.ex = ex
        e2io = None
        w2c = self.w2                         # ADVECT lane (M2): the W2 set-up of the build (GO lane's self.w2)
        if w2c is not None and w2c.W2_useE2ioLayOut:
            from mitjax.io.mds import E2ioLayout   # MDS_READ_FIELD's exch2 global-file layout (mdsio_read_field.F:124-128)
            e2io = E2ioLayout.from_w2(w2c, cfg.size)
        self.rw = RW(rundir, self.prm.init.readBinaryPrec, cfg.size, e2io=e2io)
        # INITIALISE_FIXED (initialise_fixed.F:156-246)
        grid = initialise_fixed_grid(exp, self.prm.grid, ex=self.ex, rw=self.rw)
        self.grid = ini_linear_phisurf(grid, cfg=cfg, params=self.params)                  # :226
        if w2c is not None and self.params.useCubedSphereExchange:                         # ADVECT lane (M2)
            from mitjax.model.grid import w2_tile_view
            # W2_EXCH2_TOPOLOGY.h per tile (exch2_myFace, exch2_is*edge) for the cube passes of GAD_ADVECTION /
            # GAD_SOM_ADVECT, from the W2 set-up of the build (W2_EEBOOT)
            self.grid = w2_tile_view(self.grid, w2c, cfg.size)
        rp = RunParams(exp.run)
        eos_p = SimpleNamespace(fluidIsWater=self.params.fluidIsWater, usingPCoords=self.params.usingPCoords,
                                eosType=self.params.eosType,
                                tAlpha=rp.get("data", "PARM01", "tAlpha") if rp.has("data", "PARM01", "tAlpha")
                                else UNSET_RL,                                             # ini_parms.F:421
                                sBeta=rp.get("data", "PARM01", "sBeta") if rp.has("data", "PARM01", "sBeta")
                                else UNSET_RL)                                             # ini_parms.F:422
        self.eos = ini_eos(cfg=cfg, params=eos_p)                                          # :176
        if self.params.eosType.strip() in ("JMD95Z", "JMD95P", "UNESCO", "MDJWF"):          # R5 arm (Task 16)
            from mitjax.model.src.ini_parms import set_ref_state_eos
            self.params = set_ref_state_eos(self.params, self.grid, exp)    # set_ref_state.F:51-100, surf_pRef
        elif bool(dict(cfg.use).get("useEXF", False)):
            # lane M4OFF: EXF_MAPFIELDS reads PARAMS.h surf_pRef (pLoad, exf_mapfields.F); with a LINEAR EOS the
            # value of `data` / set_defaults.F:103 is carried here (set_ref_state_eos carries it for the others)
            from mitjax.pkg.exf.exf_readparms import params_surf_pRef
            self.params = self.params.replace(traced={"surf_pRef": jnp.float64(params_surf_pRef(exp))})
        self.cg2dh = jax.jit(lambda g, s, p, x: ini_cg2d(cfg=cfg, grid=g, surface=s, params=p, ex=x))(
            self.grid, self.grid, self.cg2d_params, self.ex)                              # :246
        # INITIALISE_VARIA (initialise_varia.F): INI_FFIELDS (:213) and INI_FORCING (:242) write FFIELDS.h only,
        # which no routine between them and INTEGR_CONTINUITY (:334) reads, so they run after it here
        # PACKAGES_INIT_VARIABLES (:263) is not ported; it writes nothing when every package switch it tests is
        # off and the CD scheme is not used (packages_init_variables.F:172-330: each call sits under IF (useX),
        # :204 IF (useCDscheme)), as in the barotropic gyre: skipped only then, raise otherwise
        # GO lane: the NONLIN_FRSURF sequence of INITIALISE_VARIA (CALC_R_STAR, UPDATE_R_STAR, UPDATE_CG2D) is ported
        # lane B (Task 25): CALC_SURF_DR / UPDATE_SURF_DR (NONLIN_FRSURF without r*) are ported too
        self.pending = tuple(r for r in executed_pending(cfg, self.prm)
                             if r not in ("INTEGR_CONTINUITY", "CALC_R_STAR", "UPDATE_R_STAR", "UPDATE_CG2D",
                                          "CALC_SURF_DR", "UPDATE_SURF_DR"))
        # ADVECT lane: PACKAGES_INIT_VARIABLES writes model state only through the calls of packages whose switch is
        # on; GAD_INIT_VARIA (useGAD, packages_init_variables.F:191-198) is ported (below) and
        # DIAGNOSTICS_INIT_VARIA (useDiagnostics, :172-176) writes only pkg/diagnostics state (not ported, output
        # only); any other active package still raises
        piv_ported = {"useGAD", "useDiagnostics",
                      # R5 arm: GMREDI_INIT_VARIA (:259-275) and CTRL_INIT_VARIABLES (:605-622) run in
                      # Model._packages; useAUTODIFF / useGrdchk have no PACKAGES_INIT_VARIABLES call
                      "useGMRedi", "useCTRL", "useAUTODIFF", "useGrdchk"}
        # PTRACERS lane: PTRACERS_INIT_VARIA (packages_init_variables.F:332-336) is ported; useMNC: the routine has no
        # pkg/mnc call (grep of packages_init_variables.F @63cdc0b), so useMNC makes it write nothing
        piv_ported |= {"usePTRACERS", "useMNC"}
        # GO lane: useSBO -- PACKAGES_INIT_VARIABLES has no pkg/sbo call (grep of packages_init_variables.F @63cdc0b)
        piv_ported |= {"useSBO"}
        # vermix lane (M3 Task 30): KPP_INIT_VARIA (packages_init_variables.F, useKPP) runs in Model._packages
        piv_ported |= {"useKPP", "useGGL90", "usePP81", "useMY82", "useOPPS"}
        # M3 Task 30: RBCS_INIT_VARIA (packages_init_variables.F, useRBCS) runs in Model._packages; LAYERS_INIT_VARIA
        # (useLayers) writes only LAYERS.h (output only, INTEGRATED_PACKAGES note)
        piv_ported |= {"useRBCS", "useLayers"}
        # lane M4COL (M4 step 3): EXF_INIT_VARIA (packages_init_variables.F:290-297) and SEAICE_INIT_VARIA (:426-441)
        # run in Model._cal_exf_seaice; pkg/cal has no PACKAGES_INIT_VARIABLES call
        piv_ported |= {"useEXF", "useSEAICE", "useCAL"}
        # GO lane: CD_CODE_INI_VARS (useCDscheme, packages_init_variables.F:200-210) is ported (initialise_varia)
        # lane M4ADCOL session 3: ECCO_INIT_VARIA (packages_init_variables.F:601, useECCO) runs in Model._ecco_init
        piv_ported |= {"useECCO"}
        # lane M4ADLAB session 3: DWNSLP_INIT_VARIA (packages_init_variables.F:286, useDOWN_SLOPE) in Model._packages
        piv_ported |= {"useDOWN_SLOPE"}
        piv_writes = any(v for k, v in cfg.use if k not in piv_ported)
        # R5 arm: COST_INIT_VARIA (:268) runs in Model._packages; AUTODIFF_INIT_VARIA (:246) writes TsurfCor,
        # SsurfCor (read only under linFSConserveTr) and the EfluxY/P diagnostics: nothing this forward reads
        unported = set(self.pending) - {"INI_FFIELDS", "INI_FORCING", "COST_INIT_VARIA", "AUTODIFF_INIT_VARIA"} - (
            set() if piv_writes else {"PACKAGES_INIT_VARIABLES"})
        if unported:
            raise NotImplementedError(f"Model: INITIALISE_VARIA routines not ported for this experiment: "
                                      f"{sorted(unported)}")
        self.init_host = []          # GO lane: INITIALISE_VARIA's CALC_R_STAR counters (host report in run.forward)
        # PTRACERS lane: CONVECTIVE_ADJUSTMENT_INI (initialise_varia.F:280-295) runs here, after the packages
        convect_ini = bool(cfg.cpp.INCLUDE_CONVECT_INI_CALL and self.prm.time.startTime == self.prm.time.baseTime
                           and self.params.cAdjFreq != 0.)
        self.convect_ini = convect_ini
        self.state0 = initialise_varia(self.grid, cfg=cfg, params=self.prm, ex=self.ex, rw=self.rw, eos=self.eos,
                                       pending=self.pending, dyn=self.params, cg2dh=self.cg2dh,
                                       cg2d_params=self.cg2d_params, host=self.init_host,
                                       convect_ini_by_caller=convect_ini)
        # GO lane: READ_PICKUP's CHECK_PICKUP (check_pickup.F:59-65, :173-198) sets RESTART.h's AB start levels
        # (mom_StartAB, tempStartAB, saltStartAB): they replace INI_MODEL_IO's values in the static Params
        for _, _, start in [h for h in self.init_host if h[0] == "CHECK_PICKUP"]:
            self.params = self.params.replace(static=dict(start))
        self.init_host = [h for h in self.init_host if h[0] != "CHECK_PICKUP"]
        if cfg.cpp.ALLOW_GENERIC_ADVDIFF and cfg.use_flag("useGAD"):                       # ADVECT lane
            from mitjax.pkg.generic_advdiff.gad_init_varia import gad_init_varia
            if rp_has_pickupSuff(exp):
                raise NotImplementedError("Model: pickupSuff (GAD_INIT_VARIA's restart test) is not ported")
            tp = self.prm.time
            self.state0 = gad_init_varia(self.state0, cfg=cfg, params=self.params, startTime=tp.startTime,
                                         baseTime=tp.baseTime, nIter0=tp.nIter0, pickupSuff=" ")   # :263
        # PTRACERS lane: PACKAGES_INIT_VARIABLES -> PTRACERS_INIT_VARIA (packages_init_variables.F:332-336, after
        # GAD_INIT_VARIA :191-198); the PTRACERS_START.h flags it sets go into params
        if cfg.cpp.ALLOW_PTRACERS and cfg.use_flag("usePTRACERS"):
            self.params, self.state0 = ptracers_init_state(self, self.params, self.state0)
        elif cfg.cpp.ALLOW_PTRACERS:
            # lane M4LAB (lab_sea: pkg/ptracers compiled, usePTRACERS = .FALSE.): PTRACERS_INIT_VARIA does not run
            # and nothing writes COMMON /PTRACERS_FIELDS/ (PTRACERS_FIELDS.h:23-29): the static common's zero, as
            # the oracle dumps it (pTracer_01, gpTrNm1_01, surfaceForcingPTr_01 = 0 at S00_begin, job27855987-jdon);
            # before, these unused fields held the declaration's NaN on every lane
            from mitjax.farray import FArray
            from mitjax.model.state import ptracers_fields_of

            def _zero(f):
                return FArray(jnp.zeros_like(f.data), f.name, tiled=f.tiled, _dims=f.dims)
            self.state0 = self.state0.replace(**{
                n: (tuple(_zero(f) for f in v) if isinstance(v, tuple) else _zero(v))
                for n in ptracers_fields_of(cfg) for v in (getattr(self.state0, n),)})
        ff = ini_ffields(cfg=cfg)                                                          # :213
        _, _, _, lat = ini_grid(cfg=cfg, params=self.prm.grid, ex=self.ex, rw=self.rw)
        self.ff0 = ini_forcing(ff, cfg=cfg, grid=self.grid, fp=self.fp, rw=self.rw, ex=self.ex,
                               latBandClimRelax=lat)                                       # :242
        from mitjax.farray import loop_i, loop_j
        from mitjax.model.grid import declare
        sz = cfg.size
        j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
        i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
        self.phi0surf0 = declare("Bo_surf", sz).at[i, j].set(0.)        # ini_linear_phisurf.F:200-208 (phi0surf)
        self.pk0, pkc, self.pks = self._packages(xx)                                        # R5 arm (Task 16)
        # PTRACERS lane: INITIALISE_VARIA's CONVECTIVE_ADJUSTMENT_INI (:280-295), after PACKAGES_INIT_VARIABLES (its
        # CTRL_MAP_INI_GENARR: `_ctrl_genarr`, xx_ptr included) and COST_INIT_VARIA, which writes no model state
        self.state0 = ptracers_convect_ini(self, self.state0, convect_ini)
        self.arrays = Arrays(grid=self.grid, params=self.params, eos=self.eos, cg2dh=self.cg2dh,
                             cg2d_params=self.cg2d_params, ex=self.ex, pkc=pkc)
        self.step = make_step(cfg, self.fp, self.pks)

    # ---- R5 arm (plan Task 16): package state and inputs of forward_step ----------------------------------------
    def _packages(self, xx):
        """(pk0, pkc, pks) of forward_step (its docstring): the periodic-forcing preload of the run's nTimeSteps
        (external_fields_load.preload_periodic_forcing: EXTERNAL_FIELDS_LOAD's reads and exchanges on the host, from
        INI_FORCING's FFIELDS.h); GMREDI.h after GMREDI_READPARMS / GMREDI_INIT_FIXED / GMREDI_INIT_VARIA (useGMRedi;
        packages_readparms.F, packages_init_fixed.F, packages_init_variables.F); with useCTRL the CTRL_GENARR.h state
        and the control records `effective` of CTRL_MAP_INI_GENTIM2D from the control vector `xx` ({iarr: [records]};
        default: zeros, the first guess CTRL_INIT_CTRLVAR writes with doInitXX); with ALLOW_COST the cost.h state
        (COST_INIT_VARIA, COST_DEPENDENT_INIT). Raises for a useCTRL / ALLOW_COST build until the driver's
        COST_WEIGHTS / COST_FINAL part is wired (pending)."""
        cfg, tp = self.cfg, self.prm.time
        pk0, pkc, pks = {}, {}, {}
        self._cal_exf_seaice(pk0, pkc, pks)                    # lane M4COL: pkg/cal, pkg/exf, pkg/seaice
        if self.fp.periodicExternalForcing:
            from mitjax.model.src.external_fields_load import preload_periodic_forcing
            pkc["forcing"], _ = preload_periodic_forcing(self.ff0, tp.nTimeSteps, cfg=cfg, fp=self.fp, rw=self.rw,
                                                         ex=self.ex, nIter0=tp.nIter0, startTime=tp.startTime,
                                                         deltaTClock=tp.deltaTClock)
        if cfg.cpp.ALLOW_GMREDI and self.params.useGMRedi:
            from mitjax.pkg.gmredi.gmredi_init_fixed import gmredi_init_fixed
            from mitjax.pkg.gmredi.gmredi_init_varia import gmredi_init_varia
            from mitjax.pkg.gmredi.gmredi_readparms import gmredi_readparms
            pk0["gm"] = gmredi_init_varia(gmredi_init_fixed(gmredi_readparms(self.exp), cfg=cfg), cfg=cfg)
        if cfg.cpp.ALLOW_DOWN_SLOPE and self.params.useDOWN_SLOPE:
            # lane M4ADLAB session 3 (lab_sea/input_ad): DWNSLP_READPARMS (packages_readparms.F), DWNSLP_INIT_FIXED
            # (packages_init_fixed.F:349-351, host: the site tables), DWNSLP_INIT_VARIA (packages_init_variables.F:
            # 286); the tables and DWNSLP_VARS.h ride in the carry (pk["dwnslp"]); the init log / STDOUT lines are kept
            from mitjax.pkg.down_slope.dwnslp_init_fixed import dwnslp_init_fixed
            from mitjax.pkg.down_slope.dwnslp_init_varia import dwnslp_init_varia
            from mitjax.pkg.down_slope.dwnslp_readparms import dwnslp_readparms
            dp = dwnslp_readparms(self.exp, tempStepping=bool(self.params.tempStepping),
                                  saltStepping=bool(self.params.saltStepping),
                                  debugLevel=int(self.params.debugLevel))
            fixed = dwnslp_init_fixed(dp, cfg=cfg, grid=self.grid, usingPCoords=bool(self.params.usingPCoords))
            self.dwnslp_log, self.dwnslp_stdout = fixed["log"], fixed["stdout"]
            pk0["dwnslp"] = dwnslp_init_varia(fixed, dp)
        if (cfg.cpp.ALLOW_MOM_VECINV and self.params.vectorInvariantMomentum) or (
                cfg.cpp.ALLOW_MOM_COMMON and self.params.momStepping and self.params.useVariableVisc):
            # GOADK lane; M3 lane MLAdjust: MOM_INIT_FIXED runs for every momStepping build (packages_init_fixed.F
            # :198-205); its MOM_VISC.h is read by MOM_VECINV and by MOM_FLUXFORM's MOM_CALC_VISC (useVariableVisc)
            from mitjax.pkg.mom_common.mom_init_fixed import mom_init_fixed
            v = mom_init_fixed(cfg=cfg, grid=self.grid, params=self.params)  # packages_init_fixed.F (MOM_INIT_FIXED)
            pkc["visc"] = PkgCommon(**{n: v[n] for n in ("L2_D", "L2_Z", "L3_D", "L3_Z", "L4rdt_D", "L4rdt_Z",
                                                         "deepFacAdv")})
        if cfg.cpp.ALLOW_KPP and self.params.useKPP:
            # vermix lane (M3 Task 30): KPP_READPARMS (packages_readparms.F), KPP_CHECK (packages_check.F),
            # KPP_INIT_FIXED (packages_init_fixed.F), KPP_INIT_VARIA (packages_init_variables.F) on KPP.h as the
            # static common holds it before (zero: KPPfrac keeps it, no SHORTWAVE_HEATING); KPP_PARAMS.h (traced
            # REALs, nzmax, tables) rides in pkc, KPP.h in the carry
            from mitjax.pkg.kpp.kpp_init_fixed import kpp_init_fixed
            from mitjax.pkg.kpp.kpp_init_varia import kpp_init_varia
            from mitjax.pkg.kpp.kpp_readparms import kpp_check, kpp_readparms
            if cfg.cpp.ALLOW_AUTODIFF:
                # lane M4ADCOL: the KPP arms of an ALLOW_AUTODIFF build are TAF directives, the ikey of KPP_CALC
                # (:215-219) and RI_IWMIX's inAdMode arm (kpp_routines.py); TAF's adjoint mode must equal the forward
                from mitjax.pkg.autodiff.autodiff_readparms import adjoint_mode_check
                adjoint_mode_check(self.exp)
            kp = kpp_readparms(self.exp, self.prm.time, self.io)
            kpp_check(kp, params=self.params, exp=self.exp)
            kp = kpp_init_fixed(kp, cfg=cfg, grid=self.grid)
            kp, pk0["kpp"] = kpp_init_varia(kp, kpp_h_zero(cfg.size), cfg=cfg, grid=self.grid, params=self.params)
            pkc["kpp_p"] = kp
            if cfg.cpp.flag("ALLOW_SALT_PLUME", "KPP_OPTIONS.h"):
                # lane M4LAB: KPP.h:35-38 /kpp_short1/ KPPplumefrac (#ifdef ALLOW_SALT_PLUME), written by KPP_CALC
                # only with useSALT_PLUME (kpp_calc.F:672-712) and by no init routine: the static common's zero
                pk0["kpp"] = dict(pk0["kpp"], KPPplumefrac=_zero2d(cfg.size, "KPPplumefrac"))
        if cfg.cpp.ALLOW_SALT_PLUME:
            # lane M4LAB (lab_sea: pkg/salt_plume compiled, useSALT_PLUME = .FALSE.): SALT_PLUME.h:89-90, 101, 109
            # SaltPlumeDepth (/DYNVARS_SALT_PLUME/) and saltPlumeFlux (/FFIELDS_saltPlumeFlux/), read by KPP_CALC
            # and KPP_TRANSPORT_S; SALT_PLUME_INIT_VARIA runs only with useSALT_PLUME (packages_init_variables.F:
            # 443-447): the static commons' zero. useSALT_PLUME itself is refused (DO_OCEANIC_PHYS, KPP).
            if cfg.use_flag("useSALT_PLUME"):
                raise NotImplementedError("Model: useSALT_PLUME (pkg/salt_plume physics) is not ported")
            pk0["salt_plume"] = {n: _zero2d(cfg.size, n) for n in ("saltPlumeFlux", "SaltPlumeDepth")}
            # session 3: SALT_PLUME_READPARMS (packages_readparms.F) returns before its defaults with useSALT_PLUME
            # off; SEAICE_GROWTH reads SPsalFRAC / SaltPlumeSouthernOcean as the never-written commons hold them
            from mitjax.pkg.salt_plume.salt_plume_readparms import salt_plume_readparms
            pkc["spp"] = salt_plume_readparms(cfg)
        if cfg.cpp.ALLOW_OPPS and cfg.use_flag("useOPPS"):
            from mitjax.pkg.opps.opps_readparms import opps_readparms                # vermix lane: OPPS_READPARMS
            pkc["opps"] = opps_readparms(self.exp)
        # vermix lane (M3 Task 30): GGL90 / PP81 / MY82 _READPARMS (packages_readparms.F), _INIT_FIXED
        # (packages_init_fixed.F) and _INIT_VARIA (packages_init_variables.F :239-276) on their package state
        if cfg.cpp.ALLOW_GGL90 and self.params.useGGL90:
            from mitjax.pkg.ggl90.ggl90_init_fixed import ggl90_init_fixed
            from mitjax.pkg.ggl90.ggl90_init_varia import ggl90_init_varia
            from mitjax.pkg.ggl90.ggl90_readparms import ggl90_readparms
            ggl = ggl90_readparms(self.exp)
            ggl90_init_fixed(cfg=cfg)
            pk0["ggl"] = ggl90_init_varia(ggl, cfg=cfg, grid=self.grid, nIter0=tp.nIter0, pickupSuff=" ", ex=self.ex,
                                          rw=self.rw)
        if cfg.cpp.ALLOW_PP81 and self.params.usePP81:
            from mitjax.pkg.pp81.pp81_init_varia import pp81_init_varia
            from mitjax.pkg.pp81.pp81_readparms import pp81_readparms
            pk0["pp81"] = pp81_init_varia(cfg=cfg, params=self.params, pp=pp81_readparms(self.exp, params=self.params))
        if cfg.cpp.ALLOW_MY82 and self.params.useMY82:
            from mitjax.pkg.my82.my82_init_varia import my82_init_varia
            from mitjax.pkg.my82.my82_readparms import my82_readparms
            pk0["my82"] = my82_init_varia(cfg=cfg, params=self.params, my=my82_readparms(self.exp))
        if cfg.cpp.ALLOW_RBCS and cfg.use_flag("useRBCS"):                          # M3 Task 30: pkg/rbcs
            from mitjax.pkg.rbcs.rbcs_fields_load import rbcs_preload
            from mitjax.pkg.rbcs.rbcs_init_fixed import rbcs_init_fixed
            from mitjax.pkg.rbcs.rbcs_init_varia import rbcs_init_varia
            fixed = rbcs_init_fixed(cfg=cfg, params=self.params, grid=self.grid, rw=self.rw, ex=self.ex)
            pkc["rbcs"] = dict(fixed, pre=rbcs_preload(cfg=cfg, params=self.params, rw=self.rw))  # packages_init_fixed
            pk0["rbcs"] = rbcs_init_varia(cfg=cfg)                                   # packages_init_variables
        if cfg.cpp.ALLOW_KPP and self.params.useKPP:
            pass
        elif cfg.cpp.ALLOW_KPP and cfg.cpp.ALLOW_AUTODIFF and not cfg.cpp.ALLOW_OFFLINE:
            # PTRACERS lane: KPP.h (KPP.h:28-41) for DO_OCEANIC_PHYS' KPP_CALC_DUMMY (useKPP = .FALSE.)
            pk0["kpp"] = kpp_h_zero(cfg.size)
        useCTRL = bool(cfg.cpp.ALLOW_CTRL) and any(k.lower() == "usectrl" and v for k, v in cfg.use)
        xx_genarr = None
        if isinstance(xx, dict) and "genarr" in xx:                   # GOADK lane: {(dim, iarr): record} controls
            xx = dict(xx)
            xx_genarr = xx.pop("genarr")
            xx = xx or None
        if useCTRL:
            self._ctrl_first_guess_zero()                  # lane API s2: the zero first guess below must be Fortran's
            pk0["genarr"], pkc["effective"], pks["gentim2d"], pks["clock"] = self._ctrl_init(xx)
            pkc["ctrl_weight"] = self._ctrl_weight()           # the weight records (CTRL_MAP_INI_GENTIM2D's input)
            self._ctrl_genarr(pk0, pkc, xx_genarr)                    # GOADK lane: CTRL_MAP_INI_GENARR
        elif xx is not None or xx_genarr is not None:
            raise ValueError("Model: a control vector `xx` needs a useCTRL build")
        if cfg.cpp.ALLOW_COST:
            pk0["cost"], pkc["lastinterval"], pks["endTime"], pks["lastinterval"] = self._cost_init()
        if cfg.cpp.ALLOW_ECCO and cfg.use_flag("useECCO"):          # lane M4ADCOL session 3: pkg/ecco
            self._ecco_init(pk0, pkc, pks)
        if cfg.cpp.ALLOW_COST:
            pkc["cost_fixed"] = self.cost_fixed              # COST_FINAL's inputs, a jit argument (adjoint_run)
        return pk0, pkc, pks

    def _ecco_init(self, pk0, pkc, pks):
        """Lane M4ADCOL session 3 (1D_ocean_ice_column/code_ad): pkg/ecco's set-up in THE_MODEL_MAIN's order --
        ECCO_READPARMS (packages_readparms.F:343), ECCO_INIT_FIXED -> ECCO_COST_INIT_FIXED (packages_init_fixed.F:381),
        ECCO_INIT_VARIA's ECCO_COST_INIT_VARIA -> COST_AVERAGESINIT (packages_init_variables.F:601,
        ecco_init_varia.F:65, ecco_cost_init_varia.F:60) -- and the host parts of COST_AVERAGESFIELDS (the per-call
        table) and COST_GENCOST_ALL (the data and uncertainty records). ECCO.h's averaged fields and bar records
        ride in the carry (pk0["ecco"]), the tables in pkc["ecco_tab"], the data in cost_fixed["ecco"] (jit
        arguments), the host structure in pks["ecco"] and self.ecco. ECCO_PHYS (ecco_init_varia.F:50 here, and
        forward_step.F:1169) writes ECCO.h fields this run never reads (pk0["ecco"]["phys"]; pkg/ecco/ecco_phys.py)."""
        import re

        import numpy as np
        from mitjax.params_io import RunParams
        from mitjax.pkg.ecco.cost_averagesfields import averages_table, check_customize, cost_averagesinit
        from mitjax.pkg.ecco.cost_gencost_all import gencost_setup
        from mitjax.pkg.ecco.ecco_cost_init_fixed import ecco_cost_init_fixed
        from mitjax.pkg.ecco.ecco_phys import check_ecco_phys, ecco_phys
        from mitjax.pkg.ecco.ecco_readparms import ecco_readparms
        cfg, tp, sz = self.cfg, self.prm.time, self.cfg.size
        if not cfg.cpp.ALLOW_COST or getattr(self, "cal", None) is None:
            raise NotImplementedError("Model: pkg/ecco without pkg/cost and pkg/cal is not ported")
        ep = ecco_readparms(self.exp.run, cfg)                                       # packages_readparms.F:343
        # OPTIMCYCLE.h optimcycle: data.optim &OPTIM optimcycle (the run directory's value, here 0)
        fo = Path(self.rundir) / "data.optim"
        mo = re.search(r"optimcycle\s*=\s*(-?\d+)", fo.read_text(), re.I) if fo.exists() else None
        if mo is None:
            raise NotImplementedError("Model: pkg/ecco needs data.optim's optimcycle")
        dTt1 = float(np.asarray(tp.dTtracerLev)[0])                                  # PARAMS.h dTtracerLev(1)
        fx = ecco_cost_init_fixed(ep, cal=self.cal, optimcycle=int(mo.group(1)), nTimeSteps=tp.nTimeSteps,
                                  dTtracerLev1=dTt1)
        check_customize(ep)
        table = averages_table(ep, cal=self.cal, nIter0=tp.nIter0, startTime=tp.startTime,
                               deltaTClock=tp.deltaTClock, nTimeSteps=tp.nTimeSteps, endTime=tp.endTime)
        setup = gencost_setup(ep, rw=self.rw, cal=self.cal, eccoiter=fx["eccoiter"], dTtracerLev1=dTt1, sz=sz)
        check_ecco_phys(ep)
        # ECCO_INIT_VARIA (packages_init_variables.F:601): ECCO_PHYS( startTime, -1 ) (ecco_init_varia.F:50), then
        # ECCO_COST_INIT_VARIA -> COST_AVERAGESINIT (:65)
        phys = ecco_phys(-1, cfg=cfg, grid=self.grid, params=self.params, eos=self.eos, state=self.state0,
                         ff=self.ff0, fp=self.fp)
        pk0["ecco"] = dict(cost_averagesinit(ep, sz=sz, ntiles=sz.nSx*sz.nSy), phys=phys)
        pkc["ecco_tab"] = {k: tuple(jnp.asarray(t[n]) for n in ("branch", "sum1", "rec")) for k, t in table.items()}
        pks["ecco"] = dict(ep=ep, nTimeSteps=int(tp.nTimeSteps))
        # PARAMS.h writeBinaryPrec (COST_GENLOOP's misfit files): data PARM01, else set_defaults.F:356
        # writeBinaryPrec = precFloat32 (EEPARAMS.h: 32)
        wbp = int(RunParams(self.exp.run).get("data", "PARM01", "writeBinaryPrec", default=32))
        self.ecco = SimpleNamespace(ep=ep, fixed=fx, table=table, setup=setup, writeBinaryPrec=wbp)
        self.cost_fixed["ecco"] = dict(
            obs={s["g"].k: [jnp.asarray(r["obs"]) for r in s["recs"]] for s in setup},
            weight={s["g"].k: [jnp.asarray(r["weight"]) for r in s["recs"]] for s in setup},
            mult={g.k: jnp.float64(g.mult_gencost) for g in ep.gencost})

    def _cal_exf_seaice(self, pk0, pkc, pks):
        """Lane M4COL (M4 step 3): pkg/cal, pkg/exf and pkg/seaice in THE_MODEL_MAIN's order. PACKAGES_READPARMS:
        CAL_READPARMS (packages_readparms.F:156), EXF_READPARMS (:161), SEAICE_READPARMS (after them; it reads the
        EXF values); PACKAGES_INIT_FIXED: CAL_INIT_FIXED (packages_init_fixed.F:162), EXF_INIT_FIXED (:253),
        SEAICE_INIT_FIXED (:514); PACKAGES_INIT_VARIABLES (initialise_varia.F:263, after INI_FORCING :242):
        EXF_INIT_VARIA (packages_init_variables.F:295), SEAICE_INIT_VARIA (:437; it writes sIceLoad of FFIELDS.h).
        The routines read no field another one writes in between, so they run here, after INITIALISE_VARIA's model
        part (as INI_FFIELDS / INI_FORCING do in this driver). Package state carried by FORWARD_STEP: pk0["exf"]
        (EXF_FIELDS.h), pk0["seaice"] (SEAICE.h); traced inputs: pkc["exf_pre"] (the EXF record preload of the
        run's steps, exf_preload), pkc["exfp"] / pkc["sp"] / pkc["op"] / pkc["exf_fp"] (EXF_PARAM.h, SEAICE_PARAMS.h
        and the PARAMS.h values both read); static: pks["exfs"] (cal.h, the time parameters, KGEO's level)."""
        cfg, tp, sz = self.cfg, self.prm.time, self.cfg.size
        useCAL, useEXF, useSEAICE = (bool(dict(cfg.use).get(n, False)) for n in ("useCAL", "useEXF", "useSEAICE"))
        if not (useCAL or useEXF or useSEAICE):
            return
        if not useEXF:
            raise NotImplementedError("Model: pkg/cal / pkg/seaice are wired only with pkg/exf (useEXF)")
        from mitjax.model.grid import UNSET_RL
        from mitjax.model.src.ini_parms_forcing import ini_parms_forcing
        from mitjax.pkg.cal.cal_init_fixed import cal_init_fixed
        from mitjax.pkg.cal.cal_readparms import cal_readparms
        from mitjax.pkg.exf.exf_fields_h import exf_fields
        from mitjax.pkg.exf.exf_init_fixed import exf_init_fixed
        from mitjax.pkg.exf.exf_init_varia import exf_init_varia
        from mitjax.pkg.exf.exf_preload import ExfFP, exf_preload
        from mitjax.pkg.exf.exf_readparms import exf_readparms, params_celsius2K
        fpp = ini_parms_forcing(self.exp)
        # lane M4OFF: a build without pkg/cal (offline_exf_seaice: packages.conf "-cal") has no ALLOW_CAL, so
        # useCAL stays .FALSE. (packages_boot.F) and CAL_READPARMS / CAL_INIT_FIXED are not called
        # (packages_readparms.F:153-157, packages_init_fixed.F:161-166 #ifdef ALLOW_CAL / IF (useCAL)); cal = None
        cal = cal_readparms(self.exp) if useCAL else None                        # packages_readparms.F:156
        exf = exf_readparms(self.exp, params=self.params)                        # :161
        sp = None
        if useSEAICE:
            from mitjax.pkg.seaice.seaice_readparms import seaice_readparms
            sp = seaice_readparms(self.exp, exf=exf, tp=tp, params=SimpleNamespace(
                recip_rhoConst=fpp.recip_rhoConst, usingCartesianGrid=self.params.usingCartesianGrid,
                monitorFreq=self.params.monitorFreq))                     # lane M4OFF: SEAICE_monFreq (:559)
        if useCAL:
            cal = cal_init_fixed(cal, startTime=tp.startTime, endTime=tp.endTime, deltaTClock=tp.deltaTClock,
                                 nIter0=tp.nIter0, nEndIter=tp.nEndIter, nTimeSteps=tp.nTimeSteps)   # init_fixed.F:162
        exf = exf_init_fixed(exf, cfg=cfg, cal=cal, nIter0=tp.nIter0, startTime=tp.startTime)       # :253
        f = exf_init_varia(exf_fields(sz, cfg), cfg=cfg, exf=exf, grid=self.grid, params=self.params,
                           rw=self.rw)                                           # packages_init_variables.F:295
        exf_fp = ExfFP(rhoConstFresh=jnp.float64(fpp.rhoConstFresh), HeatCapacity_Cp=jnp.float64(
            fpp.HeatCapacity_Cp), temp_EvPrRn_set=bool(fpp.temp_EvPrRn != UNSET_RL))
        pk0["exf"] = f
        pkc["exf_pre"] = exf_preload(f, tp.nTimeSteps, cfg=cfg, exf=exf, cal=cal, grid=self.grid,
                                     params=self.params, rw=self.rw, tp=tp, ex=self.ex)
        pkc["exfp"], pkc["exf_fp"] = exf, exf_fp
        kgeo = None
        if useSEAICE:
            import numpy as np
            from mitjax.pkg.seaice.dynsolver import kgeo_level
            from mitjax.pkg.seaice.seaice_h import seaice_fields
            from mitjax.pkg.seaice.seaice_init_fixed import seaice_init_fixed
            from mitjax.pkg.seaice.seaice_init_varia import seaice_init_varia
            from mitjax.pkg.seaice.seaice_params_h import OceanParams
            op = OceanParams(celsius2K=np.float64(params_celsius2K(self.exp)), rhoConst=np.float64(fpp.rhoConst),
                             recip_rhoConst=np.float64(fpp.recip_rhoConst),
                             rhoConstFresh=np.float64(fpp.rhoConstFresh),
                             HeatCapacity_Cp=np.float64(fpp.HeatCapacity_Cp), gravity=np.float64(fpp.gravity),
                             recip_gravity=np.float64(fpp.recip_gravity), sIceLoadFac=np.float64(fpp.sIceLoadFac),
                             temp_EvPrRn=np.float64(fpp.temp_EvPrRn),
                             useRealFreshWaterFlux=bool(fpp.useRealFreshWaterFlux),
                             nonlinFreeSurf=int(fpp.nonlinFreeSurf), temp_EvPrRn_set=bool(fpp.temp_EvPrRn != UNSET_RL),
                             usingPCoords=bool(self.params.usingPCoords),
                             useCubedSphereExchange=bool(self.params.useCubedSphereExchange),
                             selectBalanceEmPmR=int(fpp.selectBalanceEmPmR), balanceQnet=bool(fpp.balanceQnet),
                             **self._seaice_dyn_op(sp, fpp))
            sp, sf = seaice_init_fixed(seaice_fields(cfg), cfg=cfg, sp=sp, grid=self.grid, params=self.params,
                                       op=op)                                    # packages_init_fixed.F:514
            pickupSuff = " "
            if rp_has_pickupSuff(self.exp):
                raise NotImplementedError("Model: pickupSuff (SEAICE_INIT_VARIA's pickup) is not ported")
            sf, self.ff0 = seaice_init_varia(sf, self.ff0, cfg=cfg, sp=sp, op=op, state=self.state0,
                                             params=self.params, tp=tp, rw=self.rw, ex=self.ex,
                                             pickupSuff=pickupSuff,
                                             ip=self.prm.init)                   # packages_init_variables.F:437
            pk0["seaice"] = sf
            pkc["sp"], pkc["op"] = sp, op
            kgeo = kgeo_level(sf) if "KGEO" in sf else None    # KGEO: SEAICE_BGRID_DYNAMICS only (lane M4OFF)
        pks["exfs"] = dict(cal=cal, tp=tp, kgeo=kgeo, useSEAICE=useSEAICE)
        self.cal, self.exfp, self.sp = cal, exf, sp      # the host routines' copies (monitors, pickups)

    def _seaice_dyn_op(self, sp, fpp):
        """Lane M4OFF session 3: the PARAMS.h / GRID.h values SEAICE_LSR reads (debugLevel and deltaTClock for
        printResidual, seaice_lsr.F:173-174; globalArea for SEAICE_RESIDUAL :1282-1283, INI_GLOBAL_DOMAIN), only
        with SEAICEuseDYNAMICS (other runs keep OceanParams' defaults)."""
        if sp is None or not sp.SEAICEuseDYNAMICS:
            return {}
        import numpy as np
        from mitjax.model.src.ini_global_domain import ini_global_domain_2d
        globalArea = ini_global_domain_2d(cfg=self.cfg, grid=self.grid, ex=self.ex)[1]
        return dict(debugLevel=int(fpp.debugLevel), deltaTClock=np.float64(self.prm.time.deltaTClock),
                    globalArea=np.float64(globalArea))

    def _ctrl_genarr(self, pk0, pkc, xx_genarr):
        """GOADK lane: CTRL_INIT_VARIABLES' genarr part (ctrl_init_variables.F:83-91 bottomDragFld = 0 under
        ALLOW_BOTTOMDRAG_CONTROL; :104-106 CTRL_MAP_INI_GENARR): the generic init. controls (xx_theta, xx_salt,
        xx_kapgm, xx_kapredi, xx_bottomdrag) added to theta / salt (self.state0), GM_inpK3dGM / GM_inpK3dRedi
        (pk0["gm"]) and bottomDragFld (pkc["ctrlf"], CTRL_FIELDS.h for MOM_U/V_BOTDRAG_COEFF). `xx_genarr`:
        {(dim, iarr): FArray} the control records (default: zeros, the first guess CTRL_INIT_CTRLVAR writes with
        doInitXX); weights READ_REC_3D_RL from the run directory (precFloat64 global files). Applied after
        INITIALISE_VARIA's later calls, which do not read the mapped fields (INTEGR_CONTINUITY, the NLFS sequence).
        PTRACERS lane: with usePTRACERS the xx_ptr<n> controls map onto pTracer (ctrl_map_ini_genarr.F:378-387,
        :412-421; tutorial_tracer_adjsens/code_ad: xx_ptr1, weight ones_64b.bin)."""
        import numpy as np
        from mitjax.farray import FArray
        from mitjax.pkg.ctrl.ctrl_map_ini_genarr import ctrl_map_ini_genarr
        from mitjax.pkg.ctrl.ctrl_readparms import ctrl_readparms_genarr, fstr_blank
        cfg, sz = self.cfg, self.cfg.size
        g2, g3 = ctrl_readparms_genarr(self.exp.run, 2, cfg), ctrl_readparms_genarr(self.exp.run, 3, cfg)
        live = [g for g in g2 + g3 if not fstr_blank(g.weight)]
        bdrag = cfg.cpp.flag("ALLOW_BOTTOMDRAG_CONTROL", "CTRL_OPTIONS.h")
        b2 = dict(i=(1 - sz.OLx, sz.sNx + sz.OLx), j=(1 - sz.OLy, sz.sNy + sz.OLy))
        z2 = jnp.zeros((sz.nSx * sz.nSy, sz.sNy + 2 * sz.OLy, sz.sNx + 2 * sz.OLx), jnp.float64)
        fields = dict(theta=self.state0.theta, salt=self.state0.salt)
        if "diffKr" in self.state0:                                 # GO lane: xx_diffkr (ctrl_map_ini_genarr.F:404-407)
            fields["diffKr"] = self.state0.diffKr
        ptr = None
        if cfg.cpp.ALLOW_PTRACERS and cfg.use_flag("usePTRACERS"):   # PTRACERS lane: the xx_ptr<n> controls
            ptr = self.params
            fields.update({f"pTracer_{n:02d}": getattr(self.state0, f"pTracer_{n:02d}")
                           for n in range(1, ptr.PTRACERS_num + 1)})
        if "gm" in pk0:
            gm = pk0["gm"]
            fields.update({n: getattr(gm, n) for n in ("GM_inpK3dGM", "GM_inpK3dRedi") if n in gm})
        if bdrag:
            fields["bottomDragFld"] = FArray(z2, "bottomDragFld", **b2)               # ctrl_init_variables.F:83-91
        if "seaice" in pk0:     # lane M4ADLAB: xx_siarea / xx_siheff (ctrl_map_ini_genarr.F:217-222) on SEAICE.h,
            # after SEAICE_INIT_VARIA (packages_init_variables.F:437 before CTRL_INIT_VARIABLES :619)
            fields.update(AREA=pk0["seaice"]["AREA"], HEFF=pk0["seaice"]["HEFF"])
        if live:
            xx_in, w_in = {}, {}
            for g in live:
                nz = sz.Nr if g.dim == 3 else 1
                if g.weight.strip() == "wunit.data":    # written by CTRL_INIT_FIXED at set-up (ctrl_init_fixed.F:92-110)
                    from mitjax.pkg.ctrl.ctrl_init_fixed import ctrl_init_fixed_wunit
                    w = ctrl_init_fixed_wunit(sz=sz)[:, :nz]
                else:
                    w = _read_global_r8(self.rundir / g.weight.strip(), nz, sz)
                dims = dict(b2, k=(1, sz.Nr)) if g.dim == 3 else b2
                w_in[(g.dim, g.iarr)] = FArray(jnp.asarray(w if g.dim == 3 else w[:, 0]), "w", **dims)
                x = None if xx_genarr is None else xx_genarr.get((g.dim, g.iarr))
                xx_in[(g.dim, g.iarr)] = x if x is not None else FArray(jnp.zeros_like(w_in[(g.dim, g.iarr)].data),
                                                                       "xx", **dims)
            # PTRACERS lane (the adjoint of an initial-condition control, drivers/adjoint_run): the State and the
            # map's inputs before the map, so that the control can enter inside the differentiated function
            self.state0_pre_genarr = self.state0
            self.genarr_setup = dict(ptr=ptr)
            fields, self.genarr_effective = ctrl_map_ini_genarr(fields, cfg=cfg, genarr2d=g2, genarr3d=g3,
                                                                xx_in=xx_in, weight_in=w_in, maskC=self.grid.maskC,
                                                                ex=self.ex, ptr=ptr)
            # the adjoint driver re-applies the map to the control vector (drivers/adjoint_run.GenarrAdjoint): the
            # live controls, their record shapes and the weights (traced, in pkc)
            self.genarr_live = tuple((g.dim, g.iarr) for g in live)
            self.genarr_parms = (g2, g3)
            pkc["genarr_w"] = {f"{d}_{i}": w for (d, i), w in w_in.items()}
            self.genarr_w_in = w_in           # lane M4ADCOL: wgenarr3d (interior) for CTRL_COST_GEN3D
            self.genarr_xx0 = xx_in
            self.state0 = self.state0.replace(theta=fields["theta"], salt=fields["salt"],
                                              **{n: v for n, v in fields.items()
                                                 if n.startswith("pTracer_") or n == "diffKr"})
            if "gm" in pk0:
                pk0["gm"] = pk0["gm"].replace(**{n: fields[n] for n in ("GM_inpK3dGM", "GM_inpK3dRedi")
                                                 if n in fields})
            if "seaice" in pk0:                                       # lane M4ADLAB: AREA, HEFF after the map
                pk0["seaice"] = dict(pk0["seaice"], AREA=fields["AREA"], HEFF=fields["HEFF"])
        if bdrag:
            pkc["ctrlf"] = PkgCommon(bottomDragFld=fields["bottomDragFld"])

    def _ctrl_first_guess_zero(self):
        """Every first guess this driver builds is zero (xx_zero, the genarr records of _ctrl_genarr): CTRL_INIT_CTRLVAR
        writes the xx_<name> file as zeros only IF ( ( doInitXX .AND. optimcycle.EQ.0 ) .OR. doAdmTlm )
        (pkg/ctrl/ctrl_init_ctrlvar.F:140-151); otherwise the model reads xx_<name>.<optimcycle> from disk (and
        THE_MODEL_MAIN unpacks the control vector, the_model_main.F:637-642), which is not ported. doInitXX: data.ctrl
        CTRL_NML, default .TRUE. (ctrl_readparms.F:192); optimcycle: data.optim OPTIM, default 0
        (optim_readparms.F:79); doAdmTlm = #ifdef ALLOW_ADMTLM (ctrl_readparms.F:204-208). Lane API session 2."""
        from mitjax.params_io import RunParams, fortran_default
        rp = RunParams(self.exp.run)
        doInitXX = bool(rp.get("data.ctrl", "CTRL_NML", "doInitXX",
                               default=fortran_default("pkg/ctrl/ctrl_readparms.F:192", "doInitXX", self.exp)))
        optimcycle = int(rp.get("data.optim", "OPTIM", "optimcycle",
                                default=fortran_default("pkg/ctrl/optim_readparms.F:79", "optimcycle", self.exp)))
        doAdmTlm = self.cfg.cpp.flag("ALLOW_ADMTLM", "CTRL_OPTIONS.h")
        if not ((doInitXX and optimcycle == 0) or doAdmTlm):
            raise NotImplementedError(
                f"CTRL_INIT_CTRLVAR: doInitXX = {doInitXX}, optimcycle = {optimcycle}: the first guess is read from "
                "xx_<name>.<optimcycle> (pkg/ctrl/ctrl_init_ctrlvar.F:140-151 writes zeros only with doInitXX .AND. "
                "optimcycle.EQ.0); reading a non-zero first guess is not ported")

    def xx_zero(self):
        """The first-guess control vector {iarr: [record FArray]} = zeros (CTRL_INIT_CTRLVAR with doInitXX writes
        the xx_<name> file as zeros, ctrl_init_ctrlvar.F:143-151): one record per gentim2d control (period 0)."""
        from mitjax.pkg.ctrl.ctrl_readparms import ctrl_readparms_gentim2d, fstr_blank
        sz = self.cfg.size
        z = jnp.zeros((sz.nSx * sz.nSy, sz.sNy + 2 * sz.OLy, sz.sNx + 2 * sz.OLx), jnp.float64)
        b = dict(i=(1 - sz.OLx, sz.sNx + sz.OLx), j=(1 - sz.OLy, sz.sNy + sz.OLy))
        from mitjax.farray import FArray
        return {g.iarr: [FArray(z, "xx", **b)] * self._gentim2d_nrec(g)
                for g in ctrl_readparms_gentim2d(self.exp.run, self.cfg) if not fstr_blank(g.xx_gentim2d_weight)}

    def _gentim2d_nrec(self, g):
        """endrec - startrec + 1 of a gentim2d control (CTRL_INIT_REC, ctrl_map_ini_gentim2d.F:124-131): 1 for
        period 0 without pkg/cal (R5); with useCAL (lane M4ADCOL) the calendar arm (ctrl_init_rec.F:92-110)."""
        from mitjax.pkg.ctrl.ctrl_init_rec import ctrl_init_rec
        tp = self.prm.time
        _, _, startrec, endrec = ctrl_init_rec(g.xx_gentim2d_file, g.xx_gentim2d_startdate1, g.xx_gentim2d_startdate2,
                                               g.xx_gentim2d_period, 1, useCAL=self.cfg.use_flag("useCAL"),
                                               startTime=tp.startTime, endTime=tp.endTime,
                                               cal=getattr(self, "cal", None))
        return endrec - startrec + 1

    def ctrl_effective(self, xx, arrays=None):
        """CTRL_MAP_INI_GENTIM2D (ctrl_init_variables.F:112, PACKAGES_INIT_VARIABLES :605-622) on the control
        vector `xx` -> (effective records, wgentim2d). Traceable in `xx` (the adjoint driver differentiates it).
        `arrays`: the Arrays whose grid, exchanger and weight records to use (a placed copy inside shard_map: the
        device's tile block and its ShardedExchanger); default the Model's own."""
        from mitjax.pkg.ctrl.ctrl_map_ini_gentim2d import ctrl_map_ini_gentim2d
        from mitjax.pkg.ctrl.ctrl_readparms import ctrl_readparms_gentim2d
        tp = self.prm.time
        a = arrays
        if a is None:
            grid, ex, w = self.grid, self.ex, self._ctrl_weight()
        else:
            grid, ex, w = a.grid, a.ex, a.pkc["ctrl_weight"]
        eff, wg, _ = ctrl_map_ini_gentim2d(xx, w, cfg=self.cfg,
                                           gentim2d=ctrl_readparms_gentim2d(self.exp.run, self.cfg),
                                           maskC=grid.maskC, ex=ex, startTime=tp.startTime, endTime=tp.endTime,
                                           useCAL=self.cfg.use_flag("useCAL"), cal=getattr(self, "cal", None))
        return eff, wg

    def _ctrl_weight(self):
        """{iarr: FArray} record 1 of xx_gentim2d_weight(iarr) as READ_REC_3D_RL reads it (ctrl_map_ini_gentim2d.F
        :440-441: precFloat64, big-endian, global Nx*Ny; interior set, halos 0)."""
        from mitjax.pkg.ctrl.ctrl_readparms import ctrl_readparms_gentim2d, fstr_blank
        return {g.iarr: _read_global_xy_r8(self.rundir / g.xx_gentim2d_weight.strip(), self.cfg.size)
                for g in ctrl_readparms_gentim2d(self.exp.run, self.cfg) if not fstr_blank(g.xx_gentim2d_weight)}

    def _ctrl_init(self, xx):
        """CTRL_INIT_VARIABLES (PACKAGES_INIT_VARIABLES :605-622): CTRL_GENARR.h (xx_gentim2d, xx_gentim2d0/1 =
        zeros as the common block, wgentim2d), the effective records, the static gentim2d list and the clock of
        CTRL_GET_GEN_REC. bottomDragFld and the objf_gen* counters it also zeroes are not read by this forward."""
        from mitjax.farray import FArray
        from mitjax.pkg.ctrl.ctrl_readparms import ctrl_readparms_gentim2d
        cfg, sz, tp = self.cfg, self.cfg.size, self.prm.time
        xx = self.xx_zero() if xx is None else xx
        self.ctrl_xx_tim2d = xx           # lane M4ADCOL: CTRL_COST_GEN2D reads the control records (cost_final)
        eff, wg = self.ctrl_effective(xx)
        z = jnp.zeros((sz.nSx * sz.nSy, sz.sNy + 2 * sz.OLy, sz.sNx + 2 * sz.OLx), jnp.float64)
        b = dict(i=(1 - sz.OLx, sz.sNx + sz.OLx), j=(1 - sz.OLy, sz.sNy + sz.OLy))
        genarr = {k: {i: FArray(z, k, **b) for i in eff} for k in ("xx_gentim2d", "xx_gentim2d0", "xx_gentim2d1")}
        genarr["wgentim2d"] = {i: wg[i] for i in eff}
        clock = {"useCAL": False, "startTime": float(tp.startTime), "deltaTClock": float(tp.deltaTClock),
                 "externForcingCycle": float(self.fp.externForcingCycle)}
        gentim2d = ctrl_readparms_gentim2d(self.exp.run, self.cfg)
        # lane M4ADCOL: the gentim2d controls EXF_GETSURFACEFLUXES (exf_getsurfacefluxes.F:103-137: xx_tauu, xx_tauv,
        # xx_hflux, xx_sflux) and EXF_WIND (exf_wind.F:248-258: xx_wspeed) add are not ported: a run naming them
        # raises (for every other name these loops add nothing)
        from mitjax.pkg.ctrl.ctrl_readparms import fstr_prefix
        for g in gentim2d:
            for n, lit in ((7, "xx_tauu"), (7, "xx_tauv"), (8, "xx_hflux"), (8, "xx_sflux"), (9, "xx_wspeed")):
                if cfg.cpp.flag("ALLOW_EXF") and fstr_prefix(g.xx_gentim2d_file, n, lit):
                    raise NotImplementedError(f"Model: the gentim2d control {lit} (EXF_GETSURFACEFLUXES / EXF_WIND) "
                                              "is not ported")
        if cfg.use_flag("useCAL"):
            # lane M4ADCOL (1D_ocean_ice_column/code_ad): CTRL_GET_GEN_REC's useCAL arm (ctrl_get_gen_rec.F:80-166) on
            # the host calendar (cal.h): xx_gentim2d_startdate(1:4,iarr) from CTRL_INIT_REC (CAL_FULLDATE,
            # ctrl_init_rec.F:96-97) and the per-step table a traced clock reads
            from mitjax.pkg.cal.cal_fulldate import cal_FullDate
            from mitjax.pkg.ctrl.ctrl_get_gen_rec import gen_rec_table
            from mitjax.pkg.ctrl.ctrl_readparms import fstr_blank
            cal = self.cal
            live = [g for g in gentim2d if not fstr_blank(g.xx_gentim2d_weight)]
            startdate = {g.iarr: tuple(cal_FullDate(g.xx_gentim2d_startdate1, g.xx_gentim2d_startdate2, cal=cal))
                         for g in live}
            table = {}
            for g in live:
                key = (startdate[g.iarr], float(g.xx_gentim2d_period))
                if key not in table:
                    table[key] = gen_rec_table(startdate[g.iarr], g.xx_gentim2d_period, cal=cal,
                                               startTime=tp.startTime, deltaTClock=tp.deltaTClock,
                                               nIter0=tp.nIter0, nTimeSteps=tp.nTimeSteps)
            clock.update(useCAL=True, cal=cal, nIter0=int(tp.nIter0), startdate=startdate, table=table)
        return genarr, eff, gentim2d, clock

    def _cost_init(self):
        """COST_INIT_VARIA + COST_DEPENDENT_INIT (cost.h) and, once, COST_WEIGHTS with its file reads
        (Err_levitus_15layer.bin record 1, Nr REAL*8; Err_hflux.bin READ_REC_3D_RL precFloat64; lev_t_an.bin
        READ_FLD_XYZ_RL, readBinaryPrec 32) kept for COST_FINAL (`self.cost_fixed`). Returns (cost.h, lastinterval
        traced, endTime, lastinterval host)."""
        import numpy as np
        from mitjax.farray import FArray
        from mitjax.pkg.cost.cost_init_varia import cost_dependent_init, cost_init_varia
        from mitjax.pkg.cost.cost_weights import cost_weights
        cfg, sz = self.cfg, self.cfg.size
        cost = cost_dependent_init(cost_init_varia(cfg=cfg, ntiles=sz.nSx * sz.nSy), cfg=cfg)
        p = self.exp.params
        if not (cfg.cpp.ALLOW_COST_TEMP or cfg.cpp.ALLOW_COST_HFLUXM):
            return self._cost_init_generic(cost)    # GOADK lane (COST_ATLANTIC_HEAT), PTRACERS lane (COST_TRACER)
        prm = {n: p[f"data.cost:cost_nml:{n}"] for n in ("mult_temp_tut", "mult_hflux_tut", "lastinterval")}
        rd = self.rundir
        tmpwti = np.fromfile(rd / "Err_levitus_15layer.bin", ">f8", count=sz.Nr)            # cost_weights.F:70-84
        b = dict(i=(1 - sz.OLx, sz.sNx + sz.OLx), j=(1 - sz.OLy, sz.sNy + sz.OLy))
        errh = _read_global_xy_r8(rd / "Err_hflux.bin", sz)                                 # :104-105
        whfluxm, wtheta = cost_weights(cfg=cfg, tmpwti=tmpwti, Err_hflux=errh, ex=self.ex,
                                       xy=lambda d, n: FArray(jnp.asarray(d), n, **b))
        raw = np.fromfile(rd / "lev_t_an.bin", ">f4").astype(np.float64).reshape(sz.Nr, sz.Ny, sz.Nx)   # cost_temp.F:47
        tl = np.zeros((sz.nSx * sz.nSy, sz.Nr, sz.sNy + 2 * sz.OLy, sz.sNx + 2 * sz.OLx))
        for bj in range(sz.nSy):
            for bi in range(sz.nSx):
                tl[bi + bj * sz.nSx, :, sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] = \
                    raw[:, bj * sz.sNy:(bj + 1) * sz.sNy, bi * sz.sNx:(bi + 1) * sz.sNx]
        self.cost_fixed = dict(whfluxm=whfluxm, wtheta=wtheta, thetalev=FArray(jnp.asarray(tl), "thetalev",
                                                                               k=(1, sz.Nr), **b), params=prm)
        return cost, prm["lastinterval"], float(self.prm.time.endTime), float(prm["lastinterval"])

    def _cost_init_generic(self, cost):
        """The cost set-up of a build whose cost terms are COST_ATLANTIC_HEAT (GOADK lane: global_ocean.90x40x15/
        code_ad) and / or COST_TRACER (PTRACERS lane: tutorial_tracer_adjsens/code_ad), without COST_WEIGHTS:
        COST_READPARMS' mult_atl / mult_tracer (data.cost, default 0. _d 0, cost_readparms.F:50, :52) and lastinterval
        (default `2592000.`, a REAL*4 literal, exact, :66; then :90-92 IF ( MOD(lastinterval,deltaTClock) .GT. 0. )
        lastinterval = MAX(INT(lastinterval/deltaTClock-1)*deltaTClock, deltaTClock), on host floats); with
        ALLOW_COST_ATLANTIC_HEAT the GRID.h / PARAMS.h inputs of COST_ATLANTIC_HEAT (arrays and traced floats: a jit
        argument). Returns as _cost_init."""
        import math
        from mitjax.params_io import RunParams, fortran_default
        cfg, tp = self.cfg, self.prm.time
        atl, trc = cfg.cpp.ALLOW_COST_ATLANTIC_HEAT, cfg.cpp.ALLOW_COST_TRACER
        tst = cfg.cpp.ALLOW_COST_TEST                                           # GO lane (cs32x15/code_ad)
        # lane M4ADCOL (1D_ocean_ice_column/code_ad): the sea-ice cost (ALLOW_COST_ICE, SEAICE_COST_SENSI and
        # SEAICE_COST_FINAL) with pkg/ecco's and pkg/ctrl's COST_FINAL terms
        ice = cfg.cpp.flag("ALLOW_SEAICE") and cfg.cpp.flag("ALLOW_COST_ICE", "SEAICE_OPTIONS.h")
        # lane M4ADLAB session 3 (lab_sea/code_ad): no cost term of pkg/cost or pkg/seaice, only pkg/ecco's gencost
        # and pkg/ctrl's penalties (cost_final.F's package finals, `_pkg_final`)
        ecco = bool(cfg.cpp.ALLOW_ECCO)
        if not (atl or trc or tst or ice or ecco):
            raise NotImplementedError("Model: a cost build without ALLOW_COST_TEMP/HFLUXM/ATLANTIC_HEAT/TRACER is not "
                                      "wired")
        rp = RunParams(self.exp.run)
        lastinterval = float(rp.get("data.cost", "COST_NML", "lastinterval", default=fortran_default(
            "pkg/cost/cost_readparms.F:66", "lastinterval", self.exp)))
        dTC = float(tp.deltaTClock)
        if math.fmod(lastinterval, dTC) > 0.:                                   # :90-92
            lastinterval = max(int(lastinterval/dTC - 1)*dTC, dTC)    # MINMAX-RAW: host floats > 0, no NaN
        prm = {"lastinterval": jnp.float64(lastinterval)}
        self.cost_fixed = dict(params=prm)
        if trc:                                                                 # PTRACERS lane
            prm["mult_tracer"] = jnp.float64(rp.get("data.cost", "COST_NML", "mult_tracer", default=fortran_default(
                "pkg/cost/cost_readparms.F:52", "mult_tracer", self.exp)))
        if tst:                                                                 # GO lane: COST_TEST
            from mitjax.pkg.rw.read_rec import READ_FLD_XYZ_RL
            prm["mult_test"] = jnp.float64(rp.get("data.cost", "COST_NML", "mult_test", default=fortran_default(
                "pkg/cost/cost_readparms.F:51", "mult_test", self.exp)))
            # cost_test.F:51 READ_FLD_XYZ_RL( hydrogThetaFile, ' ', thetaLev, 0, myThid ): the interior of the local
            # (its halo is never written nor read: NaN here)
            hyd = self.prm.init.hydrogThetaFile                                 # PARAMS.h (set_defaults.F:365)
            self.cost_fixed["test"] = dict(thetaLev=READ_FLD_XYZ_RL(hyd, " ", self.state0.theta.local("thetaLev"),
                                                                    0, rw=self.rw))
        if atl:                                                                 # GOADK lane
            prm["mult_atl"] = self.exp.params["data.cost:cost_nml:mult_atl"]
            g = self.grid
            self.cost_fixed["atl"] = dict(
                grid=dict(dxG=g.dxG, maskS=g.maskS, maskC=g.maskC, drF=g.drF),
                params=dict(HeatCapacity_Cp=jnp.float64(self.fp.HeatCapacity_Cp), rhoConst=self.params.rhoConst))
        return cost, prm["lastinterval"], float(tp.endTime), lastinterval

    def cost_final(self, cost, genarr, fixed=None, maskC=None, state=None, out=None, xx=None, ecco=None,
                   arrays=None):
        """COST_FINAL (the_main_loop.F:767-775) on the cost.h state after the loop -> (cost, loc_fc); traced-safe.
        `fixed`: COST_WEIGHTS' fields and the data.cost values (default the set-up's; the adjoint passes the traced
        copy in model.pkc). Lane M4ADCOL: with ALLOW_COST_ICE / ALLOW_ECCO the package finals of cost_final.F:86-122
        (`_pkg_final`; `out` (a dict) receives their printed values, `xx` the control records: default the set-up's,
        {"tim2d": {iarr: [records]}, "arr3d": {iarr: record}}). Session 3: with useECCO `ecco` is the carried ECCO.h
        state (carry[4]["ecco"]) and `state` the final State (COST_AVERAGESFIELDS after the loop reads theta, salt).
        Lane M4COSTSHARD s2: `arrays` = the Arrays whose grid (maskC, R_low) and tables (pkc["ecco_tab"]) COST_FINAL
        reads (default the set-up's self.arrays, the same objects as self.grid); the adjoint drivers pass the traced
        model (adjoint_run.final_cost_gathered: its grid gathered to every tile), so no arm reads the set-up's grid.
        `maskC` (default arrays.grid.maskC) is the mask every arm passes on."""
        from mitjax.pkg.cost.cost_final import cost_final
        cf = self.cost_fixed if fixed is None else fixed
        A = self.arrays if arrays is None else arrays
        mC = A.grid.maskC if maskC is None else maskC
        if (self.cfg.cpp.flag("ALLOW_SEAICE") and self.cfg.cpp.flag("ALLOW_COST_ICE", "SEAICE_OPTIONS.h")) \
                or self.cfg.cpp.ALLOW_ECCO:     # ALLOW_SEAICE first: SEAICE_OPTIONS.h only in a pkg/seaice build
            return cost_final(cost, cfg=self.cfg, params=cf["params"], maskC=mC,
                              pkg_final=lambda c: self._pkg_final(c, genarr, out if out is not None else {}, xx,
                                                                  cf=cf, ecco=ecco, state=state, maskC=mC,
                                                                  arrays=A))
        xx = genarr["xx_gentim2d"] if genarr is not None and "xx_gentim2d" in genarr else {}
        if "atl" in cf:                                                       # GOADK lane: COST_ATLANTIC_HEAT
            # single process: myXGlobalLo = myYGlobalLo = 1 (eesupp ini_procs.F, nPx = nPy = 1; static ints)
            atl = dict(grid=SimpleNamespace(**cf["atl"]["grid"]), params=SimpleNamespace(**cf["atl"]["params"]),
                       myXGlobalLo=1, myYGlobalLo=1)
            return cost_final(cost, cfg=self.cfg, params=cf["params"], maskC=mC, atl=atl)
        if "test" in cf:                                                      # GO lane: COST_TEST (DYNVARS.h theta)
            if state is None:
                raise ValueError("Model.cost_final: COST_TEST reads theta: pass the final State (`state=`)")
            return cost_final(cost, cfg=self.cfg, params=cf["params"], maskC=mC,
                              test=dict(theta=state.theta, thetaLev=cf["test"]["thetaLev"]))
        if "wtheta" not in cf:                                                # PTRACERS lane: COST_TRACER alone
            return cost_final(cost, cfg=self.cfg, params=cf["params"], maskC=mC)
        return cost_final(cost, cfg=self.cfg, params=cf["params"], maskC=mC, wtheta=cf["wtheta"],
                          thetalev=cf["thetalev"], whfluxm=cf["whfluxm"], xx_gentim2d=xx)

    def _pkg_final(self, cost, genarr, out, xx, *, cf, ecco=None, state=None, maskC=None, arrays=None):
        """cost_final.F:86-122 for 1D_ocean_ice_column/code_ad (lane M4ADCOL): COST_AVERAGESFIELDS( endTime ) after the
        loop (the_main_loop.F:737-742, session 3), COST_DRIVER (the_main_loop.F:769: COST_GENCOST_ALL under useECCO,
        cost_driver.F:44-53, CTRL_COST_DRIVER under useCTRL, :57-64) and the finals ECCO_COST_FINAL (:97),
        CTRL_COST_FINAL (:102), SEAICE_COST_FINAL (:113) in their order. With useECCO, out["ecco_files"] carries the
        bar records and misfit fields the host writes (run.forward). `arrays`, `maskC`: as cost_final passes them
        (the grid, ecco_tab and mask of the traced model)."""
        from mitjax.pkg.ctrl.ctrl_readparms import ctrl_readparms_genarr, ctrl_readparms_gentim2d, ctrl_size
        cfg = self.cfg
        A = self.arrays if arrays is None else arrays
        if maskC is None:
            maskC = A.grid.maskC
        useECCO = cfg.cpp.ALLOW_ECCO and cfg.use_flag("useECCO")
        if useECCO:                                                              # cost_driver.F:44-53, :97
            from mitjax.pkg.ecco.cost_averagesfields import cost_averagesfields
            from mitjax.pkg.ecco.cost_gencost_all import cost_gencost_all
            from mitjax.pkg.ecco.ecco_cost_final import ecco_cost_final
            if ecco is None or state is None:
                raise ValueError("Model.cost_final: useECCO needs the carried ECCO.h state (`ecco=`) and `state=`")
            ep, sz = self.ecco.ep, cfg.size
            ecco = cost_averagesfields(ecco, jnp.int32(self.prm.time.nTimeSteps + 1), ep=ep,
                                       tab=A.pkc["ecco_tab"], theta=state.theta, salt=state.salt,
                                       maskC=maskC, sz=sz)                       # the_main_loop.F:737-742
            ce = cf["ecco"]
            setup = [dict(g=s["g"], nnz=s["nnz"], pp=s["pp"],
                          recs=[dict(irec=r["irec"], obs=ce["obs"][s["g"].k][n], weight=ce["weight"][s["g"].k][n])
                                for n, r in enumerate(s["recs"])]) for s in self.ecco.setup]
            res, mis, offs = cost_gencost_all(ecco["rec"], setup, maskC=maskC, sz=sz,
                                              R_low=A.grid.R_low)                # cost_driver.F:51
            cost, out["ecco"] = ecco_cost_final(cost, res, ep=ep, mult=ce["mult"])   # cost_final.F:97
            out["ecco_files"] = dict(rec=ecco["rec"], mis=mis, offs=offs)
        if cfg.cpp.ALLOW_CTRL and cfg.use_flag("useCTRL"):                       # cost_driver.F:57-64, :102
            from mitjax.pkg.ctrl.ctrl_cost_driver import ctrl_cost_driver
            from mitjax.pkg.ctrl.ctrl_cost_final import ctrl_cost_final
            cs = ctrl_size(cfg)
            g2 = ctrl_readparms_gentim2d(self.exp.run, cfg)
            g3 = ctrl_readparms_genarr(self.exp.run, 3, cfg)
            arr2d = cfg.cpp.flag("ALLOW_GENARR2D_CONTROL", "CTRL_OPTIONS.h")   # lane M4ADLAB: lab_sea/code_ad
            g2a = ctrl_readparms_genarr(self.exp.run, 2, cfg) if arr2d else []
            mult_default = 1.0 if useECCO else 0.0                               # ctrl_readparms.F:482-483
            m2 = {i: mult_default for i in range(1, cs["maxCtrlTim2D"] + 1)}     # :497-500 (UNSET: default)
            m22 = {i: mult_default for i in range(1, cs["maxCtrlArr2D"] + 1)}    # :485-488 (UNSET: default)
            m3 = {i: mult_default for i in range(1, cs["maxCtrlArr3D"] + 1)}     # :491-494
            use_cc = any(v > 0.0 for v in list(m2.values()) + list(m3.values())
                         + (list(m22.values()) if arr2d else []))                # :506-526
            xt = (xx or {}).get("tim2d", self.ctrl_xx_tim2d)
            xa = (xx or {}).get("arr3d", {i: v for (d, i), v in self.genarr_xx0.items() if d == 3})
            xa2 = (xx or {}).get("arr2d", {i: v for (d, i), v in getattr(self, "genarr_xx0", {}).items() if d == 2})
            recs, xtr = {}, {}
            for g in g2:
                if g.iarr in xt:
                    n = len(xt[g.iarr])
                    recs[g.iarr] = (1, n)       # ncvarrecstart / ncvarrecsend = startrec / endrec (CTRL_INIT_REC)
                    xtr[g.iarr] = {r: xt[g.iarr][r - 1] for r in range(1, n + 1)}
            w3 = {i: self.genarr_w_in[(3, i)] for (d, i) in self.genarr_w_in if d == 3}
            w2 = {i: self.genarr_w_in[(2, i)] for (d, i) in getattr(self, "genarr_w_in", {}) if d == 2}
            r = ctrl_cost_driver(cfg=cfg, useCtrlCostContribution=use_cc, gentim2d=g2, genarr3d=g3,
                                 xx_tim2d=xtr, recs_tim2d=recs, xx_arr3d=xa, wgentim2d=genarr["wgentim2d"],
                                 wgenarr3d=w3, maskC=maskC, genarr2d=g2a, xx_arr2d=xa2, wgenarr2d=w2)
            t2, a2, a3 = r if arr2d else (r[0], None, r[1])
            cost, out["ctrl"] = ctrl_cost_final(cost, t2, a3, useCtrlCostContribution=use_cc,
                                                maxCtrlTim2D=cs["maxCtrlTim2D"], maxCtrlArr3D=cs["maxCtrlArr3D"],
                                                mult_gentim2d=m2, mult_genarr3d=m3, a2=a2,
                                                maxCtrlArr2D=cs["maxCtrlArr2D"], mult_genarr2d=m22)
            out["ctrl_meta"] = dict(files_tim2d={g.iarr: g.xx_gentim2d_file for g in g2},
                                    files_arr3d={g.iarr: g.file for g in g3}, mult_gentim2d=m2, mult_genarr3d=m3,
                                    files_arr2d={g.iarr: g.file for g in g2a}, mult_genarr2d=m22)
        if cfg.cpp.flag("ALLOW_SEAICE") and cfg.use_flag("useSEAICE"):          # :113
            from mitjax.pkg.seaice.seaice_cost_final import seaice_cost_final
            cost, f_ice, no_ice = seaice_cost_final(cost, cfg=cfg, sp=self.sp)
            if f_ice is not None:           # lane M4ADLAB: without ALLOW_COST_ICE nothing (seaice_cost_final.F:45-97)
                out["seaice"] = (f_ice, no_ice, self.sp.mult_ice)
        return cost

    def initial_carry(self):
        """(State, ff, phi0surf, flow) after INITIALISE_VARIA; flow starts as (uVel, vVel, wVel) (any value: it is
        written by every THERMODYNAMICS call before the host reads it). With package state (R5 arm: GMREDI.h,
        CTRL_GENARR.h, cost.h) a fifth element `pk` (dict)."""
        s = self.state0
        flow = (s.uVel, s.vVel, s.wVel)
        if self.cfg.cpp.NONLIN_FRSURF and self.params.nonlinFreeSurf > 0 and not self.params.staggerTimeStep:
            flow = flow + (s.hFacW, s.hFacS, s.recip_hFacC)   # GO lane: THERMODYNAMICS' hFac (forward_step.py)
        carry = (s, self.ff0, self.phi0surf0, flow)
        return carry + ((dict(self.pk0),) if self.pk0 else ())   # R5 arm: package state

    def start_counters(self):
        """the_model_main.F:623-624: (myTime, myIter) = (startTime, nIter0)."""
        tp = self.prm.time
        return jnp.float64(tp.startTime), jnp.int32(tp.nIter0)


@jax.tree_util.register_pytree_node_class
class PkgCommon:
    """GOADK lane: a common block by Fortran name (`visc.L2_D`, `ctrlf.bottomDragFld`), a pytree whose arrays are the
    leaves (MOM_VISC.h for MOM_VECINV, CTRL_FIELDS.h for MOM_U/V_BOTDRAG_COEFF)."""

    def __init__(self, **fields):
        object.__setattr__(self, "_f", dict(fields))

    def __getattr__(self, name):
        f = object.__getattribute__(self, "_f")
        if name in f:
            return f[name]
        raise AttributeError(name)

    def tree_flatten(self):
        keys = tuple(sorted(self._f))
        return tuple(self._f[k] for k in keys), keys

    @classmethod
    def tree_unflatten(cls, keys, leaves):
        return cls(**dict(zip(keys, leaves)))


def _read_global_r8(path, nz, sz):
    """GOADK lane: a global big-endian REAL*8 record of nz levels (READ_REC_3D_RL precFloat64, one process) ->
    numpy [tile, nz, j, i], interior set, halos 0."""
    import numpy as np
    raw = np.fromfile(path, ">f8", count=sz.Nx * sz.Ny * nz).reshape(nz, sz.Ny, sz.Nx)
    out = np.zeros((sz.nSx * sz.nSy, nz, sz.sNy + 2 * sz.OLy, sz.sNx + 2 * sz.OLx))
    for bj in range(sz.nSy):
        for bi in range(sz.nSx):
            out[bi + bj * sz.nSx, :, sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] = \
                raw[:, bj * sz.sNy:(bj + 1) * sz.sNy, bi * sz.sNx:(bi + 1) * sz.sNx]
    return out


def _read_global_xy_r8(path, sz):
    """Record 1 of a global big-endian REAL*8 file of Nx*Ny values (READ_REC_3D_RL / READ_REC_XY_RL with
    precFloat64, one process) -> 2-D FArray [tile, j, i], interior set, halos 0 (the callers' zeroed targets)."""
    import numpy as np
    from mitjax.farray import FArray
    n = sz.Nx * sz.Ny
    raw = np.fromfile(path, ">f8", count=n).reshape(sz.Ny, sz.Nx)
    out = np.zeros((sz.nSx * sz.nSy, sz.sNy + 2 * sz.OLy, sz.sNx + 2 * sz.OLx))
    for bj in range(sz.nSy):
        for bi in range(sz.nSx):
            out[bi + bj * sz.nSx, sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] = \
                raw[bj * sz.sNy:(bj + 1) * sz.sNy, bi * sz.sNx:(bi + 1) * sz.sNx]
    return FArray(jnp.asarray(out), path.name, i=(1 - sz.OLx, sz.sNx + sz.OLx), j=(1 - sz.OLy, sz.sNy + sz.OLy))


def rp_has_pickupSuff(exp):
    """ADVECT lane: whether the run's data file sets PARM03 pickupSuff (set_defaults.F: pickupSuff = ' ')."""
    from mitjax.params_io import RunParams
    return RunParams(exp.run).has("data", "PARM03", "pickupSuff")


# GO lane: the packages (pkg/ directory names) whose `useX` switch the integrated Model runs when it is .TRUE.:
# generic_advdiff (useGAD: GAD_INIT_VARIA, the tracer step) and diagnostics (useDiagnostics: output only, pkg/diagnostics
# not ported, DIAGNOSTICS_IS_ON on the host, Nikolay 2026-10-01). Every other switched-on package is refused at set-up
# (sbo, gmredi, the forcing preload ... join this set when they are wired into Model / FORWARD_STEP).
# mnc (useMNC): output only and not ported, as decided on 2026-10-01 (Nikolay: "diagnostics/mnc not
# ported (gate: Fortran %MON unchanged with diagnostics off)"); gate: lane A's mncoff / diagoff oracle runs,
# scripts/tests/test_m2_oracle.py::test_output_request_runs_change_no_model_number (model lines identical, pickups
# byte-identical with useMNC / useDiagnostics off). A run that READS through MNC is refused (MNC_READ_SWITCHES).
INTEGRATED_PACKAGES = frozenset({"generic_advdiff", "diagnostics", "sbo", "mnc",    # sbo: run.SboHost (GO lane)
                                 # R5 arm: Model._packages / forward_step (GMREDI.h, CTRL_GENARR.h, cost.h);
                                 # autodiff: the forward arms; grdchk: GRDCHK_MAIN in drivers/adjoint_run.py
                                 "gmredi", "ctrl", "autodiff", "grdchk"})

# Every switch at `pinned` (63cdc0b) that makes the model READ through pkg/mnc (each read call site cited): with
# useMNC, a .TRUE. value is a hard error (UnsupportedOption). Defaults .FALSE. (pkg/mnc/mnc_readparms.F:98-113,
# pkg/diagnostics/diagnostics_readparms.F:158); the package-own pickup switches of ptracers, thsice, land, aim_v23,
# fizhi (derived from pickup_read_mnc or their own namelists) belong to packages the Model refuses anyway.
MNC_READ_SWITCHES = (
    ("data.mnc", "MNC_01", "pickup_read_mnc",
     "model/src/read_pickup.F:488, pkg/cd_code/cd_code_read_pickup.F:50, pkg/generic_advdiff/gad_read_pickup.F:51"),
    ("data.mnc", "MNC_01", "readgrid_mnc", "model/src/ini_curvilinear_grid.F:209"),
    ("data.mnc", "MNC_01", "mnc_read_bathy", "model/src/ini_depths.F:106"),
    ("data.mnc", "MNC_01", "mnc_read_salt", "model/src/ini_salt.F:67"),
    ("data.mnc", "MNC_01", "mnc_read_theta", "model/src/ini_theta.F:68"),
    ("data.diagnostics", "DIAGNOSTICS_LIST", "diag_pickup_read_mnc",
     "pkg/diagnostics/diagnostics_read_pickup.F:58 (diagnostics_readparms.F:269: .AND. diag_mnc)"),
)


# PTRACERS lane: pkg/ptracers (PTRACERS_INIT_VARIA in Model, PTRACERS_FORCING_SURF / _INTEGRATE / _CONVECT /
# _FIELDS_BLOCKING_EXCH in FORWARD_STEP, PTRACERS_MONITOR in run.forward)
INTEGRATED_PACKAGES = INTEGRATED_PACKAGES | {"ptracers"}
INTEGRATED_PACKAGES = INTEGRATED_PACKAGES | {"kpp", "opps", "ggl90", "pp81", "my82"}   # vermix lane (M3 Task 30)
# M3 Task 30 (channel): layers (useLayers) is output only: LAYERS_CALC runs in DO_THE_MODEL_IO (do_the_model_io.F:234-
# 238), LAYERS_WSURF_TR / LAYERS_FILL only under layers_useThermo (thermodynamics.F:158-163, impldiff.F:383-388), and
# every pkg/layers routine assigns only LAYERS.h arrays, its locals and its O arguments (all LAYERS.h fields; the
# model fields uVel, vVel, theta, salt enter as I arguments) -- not ported, as diagnostics; no layers switch feeds back.
# rbcs: RBCS_READPARMS / INIT_FIXED / INIT_VARIA / FIELDS_LOAD / ADD_TENDENCY (mitjax/pkg/rbcs).
INTEGRATED_PACKAGES = INTEGRATED_PACKAGES | {"layers", "rbcs"}
# lane M4COL (M4 step 3, 1D_ocean_ice_column): pkg/cal, pkg/exf, pkg/seaice (Model._cal_exf_seaice, FORWARD_STEP's
# LOAD_FIELDS_DRIVER / DO_OCEANIC_PHYS, run.forward's EXF / SEAICE monitors and the sea-ice pickup)
INTEGRATED_PACKAGES = INTEGRATED_PACKAGES | {"cal", "exf", "seaice"}
# lane M4ADCOL session 3 (M4 step 7, 1D_ocean_ice_column/code_ad): pkg/ecco's gencost path (Model._ecco_init,
# make_step's COST_AVERAGESFIELDS, Model._pkg_final; mitjax/pkg/ecco)
INTEGRATED_PACKAGES = INTEGRATED_PACKAGES | {"ecco"}
# lane M4ADLAB session 3 (lab_sea/input_ad): pkg/down_slope (Model._packages, DO_OCEANIC_PHYS, TEMP/SALT_INTEGRATE)
INTEGRATED_PACKAGES = INTEGRATED_PACKAGES | {"down_slope"}


def ptracers_params(exp, params, prm):
    """PTRACERS lane: PTRACERS_READPARMS (packages_readparms.F) and PTRACERS_INIT_FIXED (packages_init_fixed.F) of a
    build with ALLOW_PTRACERS and usePTRACERS; their PTRACERS_PARAMS.h values (all named PTRACERS_* or
    lambdaTr1ClimRelax) join the Params, which the ptracers routines receive as `ptr`. The PARAMS.h values they read
    come from INI_PARMS (params, prm) and, for doAB_onGtGs, from data / set_defaults.F:316; MNC_PARAMS.h's
    monitor_mnc as MonitorHost reads it (mnc_readparms.F:101)."""
    cfg = exp.cfg
    if not (cfg.cpp.ALLOW_PTRACERS and cfg.use_flag("usePTRACERS")):
        return params
    import numpy as np
    from mitjax.params_io import RunParams, fortran_default
    from mitjax.pkg.ptracers.ptracers_init_fixed import ptracers_init_fixed
    from mitjax.pkg.ptracers.ptracers_readparms import ptracers_readparms
    rp = RunParams(exp.run)
    useMNC = dict(cfg.use).get("useMNC", False)
    pp = dict(baseTime=prm.time.baseTime, saltAdvScheme=params.saltAdvScheme, diffKhS=params.diffKhS,
              diffK4S=params.diffK4S, diffKrNrS=np.asarray(params.diffKrNrS.data), useGMRedi=params.useGMRedi,
              useDOWN_SLOPE=params.useDOWN_SLOPE, useKPP=params.useKPP,
              doAB_onGtGs=rp.get("data", "PARM03", "doAB_onGtGs", default=fortran_default(
                  "model/src/set_defaults.F:316", "doAB_onGtGs", exp)),
              dTtracerLev=np.asarray(prm.time.dTtracerLev), monitorFreq=float(params.monitorFreq), useMNC=useMNC,
              monitor_mnc=(rp.get("data.mnc", "MNC_01", "monitor_mnc", default=fortran_default(
                  "pkg/mnc/mnc_readparms.F:101", "monitor_mnc", exp)) if useMNC else True),
              pickup_write_mnc=(rp.get("data.mnc", "MNC_01", "pickup_write_mnc", default=fortran_default(
                  "pkg/mnc/mnc_readparms.F:97", "pickup_write_mnc", exp)) if useMNC else False))
    ptr = ptracers_readparms(exp, pp)
    ptr = ptracers_init_fixed(ptr, cfg=cfg, multiDimAdvection=params.multiDimAdvection)
    # GO lane: ptracers_init_fixed.F:82-83 useMultiDimAdvec = useMultiDimAdvec .OR. PTRACERS_MultiDimAdv(iTracer)
    multi = any(ptr.PTRACERS_MultiDimAdv[:ptr.PTRACERS_numInUse])
    return params.replace(static={**ptr.static_items(), "useMultiDimAdvec": bool(params.useMultiDimAdvec or multi)},
                          traced=ptr.traced_items())


def rbcs_params(exp, params, prm):
    """M3 Task 30: RBCS_READPARMS (packages_readparms.F) of a build with ALLOW_RBCS and useRBCS; its RBCS_PARAMS.h
    values join the Params (tauRelaxU/V/T/S traced, the others static). deltaTClock and startTime from INI_PARMS."""
    cfg = exp.cfg
    if not (cfg.cpp.ALLOW_RBCS and cfg.use_flag("useRBCS")):
        return params
    from mitjax.pkg.rbcs.rbcs_readparms import rbcs_readparms
    static, traced = rbcs_readparms(exp, deltaTClock=float(prm.time.deltaTClock), startTime=float(prm.time.startTime))
    return params.replace(static=static, traced=traced)


def ptracers_init_state(m, params, state):
    """PTRACERS lane: PTRACERS_INIT_VARIA (packages_init_variables.F:332-336) on the Model `m`: returns (params with
    the PTRACERS_START.h flags, State with the PTRACERS_FIELDS.h fields)."""
    from mitjax.model.grid import declare
    from mitjax.pkg.ptracers.ptracers_fields_h import state_with_ptf
    from mitjax.pkg.ptracers.ptracers_init_varia import ptracers_init_varia
    from mitjax.pkg.ptracers.ptracers_params_h import PtracersParams
    if rp_has_pickupSuff(m.exp):
        raise NotImplementedError("Model: pickupSuff (PTRACERS_INIT_VARIA's restart test) is not ported")
    tp = m.prm.time
    ptr = PtracersParams(params.static_items(), params.traced_items())
    ptr, ptf = ptracers_init_varia(tp.nIter0, " ", cfg=m.cfg, grid=m.grid, ptr=ptr, ex=m.ex, rw=m.rw,
                                   like3=m.grid.hFacC, like2=declare("Bo_surf", m.cfg.size))
    params = params.replace(static={k: ptr.static_items()[k] for k in ("PTRACERS_StepFwd", "PTRACERS_startAB")})
    return params, state_with_ptf(state, ptf)


def _zero2d(sz, name):
    """A 2-D (1-OLx:sNx+OLx, 1-OLy:sNy+OLy, nSx, nSy) common-block array as a gfortran static common holds it before
    any write (zero), as kpp_h_zero's 2-D fields (lane M4LAB)."""
    from mitjax.farray import FArray
    b2 = dict(i=(1 - sz.OLx, sz.sNx + sz.OLx), j=(1 - sz.OLy, sz.sNy + sz.OLy))
    return FArray(jnp.zeros((sz.nSx * sz.nSy, sz.sNy + 2 * sz.OLy, sz.sNx + 2 * sz.OLx)), name, **b2)


def kpp_h_zero(sz):
    """PTRACERS lane: the KPP.h common blocks /kpp/, /kpp_short/ (KPP.h:28-41) as a gfortran static common holds them
    before any write (zero; KPP_INIT_VARIA runs only with useKPP, packages_init_variables.F): {name: FArray}."""
    from mitjax.farray import FArray
    b2 = dict(i=(1 - sz.OLx, sz.sNx + sz.OLx), j=(1 - sz.OLy, sz.sNy + sz.OLy))
    t = sz.nSx * sz.nSy
    n2 = (t, sz.sNy + 2 * sz.OLy, sz.sNx + 2 * sz.OLx)
    out = {n: FArray(jnp.zeros(n2), n, **b2) for n in ("KPPhbl", "KPPfrac")}
    out.update({n: FArray(jnp.zeros((t, sz.Nr) + n2[1:]), n, k=(1, sz.Nr), **b2)
                for n in ("KPPviscAz", "KPPdiffKzS", "KPPdiffKzT", "KPPghat")})
    return out


def ptracers_convect_ini(m, state, convect_ini):
    """PTRACERS lane: INITIALISE_VARIA's CONVECTIVE_ADJUSTMENT_INI (initialise_varia.F:280-295) on the Model `m` when
    `convect_ini`, after PACKAGES_INIT_VARIABLES (PTRACERS_INIT_VARIA, CTRL_MAP_INI_GENARR): on the State with the
    GRID.h values of that point (hFac = h0Fac by INI_NLFS_VARS' ALLOW_AUTODIFF reset, h0Fac = INI_MASKS_ETC's hFac:
    `m.grid`; INITIALISE_VARIA's NONLIN_FRSURF sequence :297-347, which ran before this call in the driver, reads
    none of theta, salt, pTracer). Returns the State."""
    if not convect_ini:
        return state
    from mitjax.model.src.convective_adjustment_ini import convective_adjustment_ini
    from mitjax.pkg.ptracers.ptracers_fields_h import ptf_of_state, state_with_ptf
    cfg, tp = m.cfg, m.prm.time
    usePTRACERS = bool(cfg.cpp.ALLOW_PTRACERS) and cfg.use_flag("usePTRACERS")
    ptf = ptf_of_state(state, m.params) if usePTRACERS else None
    state, ptf = convective_adjustment_ini(tp.startTime, tp.nIter0, cfg=cfg, grid=m.grid, params=m.params,
                                           eos=m.eos, state=state, ptr=m.params, ptf=ptf)
    return state_with_ptf(state, ptf) if usePTRACERS else state


def check_packages(cfg, ported=INTEGRATED_PACKAGES, exp=None):
    """config.params.require_ported with the packages the integrated Model runs (raises UnportedPackage); with
    `exp` and useMNC, every MNC read switch (MNC_READ_SWITCHES) must be .FALSE. (raises UnsupportedOption)."""
    from mitjax.config.params import UnsupportedOption, require_ported
    out = require_ported(cfg, ported)
    if exp is not None and dict(cfg.use).get("useMNC", False):
        from mitjax.params_io import RunParams
        rp = RunParams(exp.run)
        on = [f"{f}:{g}:{k} ({cite})" for f, g, k, cite in MNC_READ_SWITCHES if rp.has(f, g, k) and rp.get(f, g, k)]
        if on:
            raise UnsupportedOption(f"{cfg.experiment}/{cfg.input_dir}: reading through pkg/mnc is not ported: "
                                    + "; ".join(on))
    return out


def initialise_fixed_grid(exp, gp, *, ex, rw):
    """The GRID.h part of INITIALISE_FIXED: INI_GRID, SET_GRID_FACTORS, INI_DEPTHS, INI_MASKS_ETC, INI_CORI
    (initialise_fixed.F:156, 181, 190, 201, 236; LOAD_REF_FILES, INI_EOS, SET_REF_STATE and PACKAGES_INIT_FIXED in
    between write no GRID.h field)."""
    from mitjax.model.src.ini_cori import ini_cori
    from mitjax.model.src.ini_depths import ini_depths
    from mitjax.model.src.ini_grid import ini_grid
    from mitjax.model.src.ini_masks_etc import ini_masks_etc
    from mitjax.model.src.set_grid_factors import set_grid_factors
    cfg = exp.cfg
    grid, _, _, _ = ini_grid(cfg=cfg, params=gp, ex=ex, rw=rw)
    grid = set_grid_factors(grid, cfg=cfg, params=gp)
    grid = ini_depths(grid, cfg=cfg, params=gp, ex=ex, rw=rw)
    grid = ini_masks_etc(grid, cfg=cfg, params=gp, ex=ex)
    return ini_cori(grid, cfg=cfg, params=gp)


def make_step(cfg, fp, pks=None):
    """step(arrays, carry, iloop, myTime, myIter) -> (carry, myTime, myIter, cg2d): one FORWARD_STEP (traced; no
    I/O). cg2d: SOLVE_FOR_PRESSURE's solver scalars (printed by the driver on the host)."""
    from mitjax.model.src.forward_step import forward_step

    def step(a, carry, iloop, myTime, myIter):
        state, ff, phi0surf = carry[:3]
        pk = carry[4] if len(carry) > 4 else None                                   # R5 arm: package state
        if pk is not None and "ecco" in pk:
            # lane M4ADCOL session 3: COST_AVERAGESFIELDS( myTime ) before FORWARD_STEP, on the step-start state
            # (the_main_loop.F:663-673; call number iloop of the host table, cost_averagesfields.py)
            from mitjax.pkg.ecco.cost_averagesfields import cost_averagesfields
            pk = dict(pk, ecco=cost_averagesfields(pk["ecco"], iloop, ep=pks["ecco"]["ep"], tab=a.pkc["ecco_tab"],
                                                   theta=state.theta, salt=state.salt, maskC=a.grid.maskC,
                                                   sz=cfg.size))
        state, ff, phi0surf, myTime, myIter, out = forward_step(
            iloop, myTime, myIter, cfg=cfg, grid=a.grid, params=a.params, fp=fp, eos=a.eos, cg2dh=a.cg2dh,
            cg2d_params=a.cg2d_params, state=state, ff=ff, phi0surf=phi0surf, ex=a.ex, pk=pk, pkc=a.pkc, pks=pks)
        # ADVECT lane: THERMODYNAMICS' tracer arm returns its local copies uFld, vFld, wFld (the names are pytree
        # metadata); keep the carry's names (initial_carry) so that the scan carry keeps its structure
        from mitjax.farray import FArray
        flow = tuple(FArray(f.data, c.name, tiled=c.tiled, _dims=c.dims) for f, c in zip(out["flow"], carry[3]))
        c = out["cg2d"]
        if out.get("rstar") is not None:
            # GO lane: CALC_R_STAR's per-tile counters ride along with the solver scalars (keys rstar_*), stacked by
            # the scan; run.forward applies calc_r_star_host and the zero-denominator report on the host per step
            c = dict(c or {}, **{"rstar_" + k: v for k, v in out["rstar"].items()})
        if out.get("exf_mon") is not None:
            # lane M4COL: EXF_FIELDS.h as EXF_MONITOR sees it (exf_getforcing.F:380), keys exfmon_*; run.forward prints
            # the EXF monitor block of each step on the host (exf_monitor.exf_monitor)
            c = dict(c or {}, **{"exfmon_" + k: v for k, v in out["exf_mon"].items()})
        if out.get("lsr") is not None:
            # lane M4OFF session 3: SEAICE_LSR's per-pass STDOUT values (keys lsr_*), printed by run.forward
            c = dict(c or {}, **{"lsr_" + k: v for k, v in out["lsr"].items()})
        if out.get("opps") is not None:
            # vermix lane: OPPS_CALC's time-loop bound counters (keys opps_*), checked per step on the host by
            # run.forward (opps_calc.opps_calc_host)
            c = dict(c or {}, **{"opps_" + k: v for k, v in out["opps"].items()})
        return (state, ff, phi0surf, flow) + ((out["pk"],) if pk else ()), myTime, myIter, c   # R5: pk
    return step
