"""The adjoint of a cost-function experiment (plan Task 16, first TAF match: tutorial_global_oce_optim/input_ad):
the gradient of COST_FINAL's fc with respect to the control vector, the gradient check's finite differences, and
the `ADM` lines GRDCHK_MAIN prints.

    m = Model(exp, rundir)                         # useCTRL / ALLOW_COST build (drivers/model.py Model._packages)
    fc, g = gradient(m)                            # fc(xx) and dfc/dxx through the checkpointed time loop
    fd = grdchk_fd(m, points)                      # GRDCHK_MAIN's +-grdchk_eps forward runs at the check points

fc(xx) = COST_FINAL( THE_MAIN_LOOP( CTRL_MAP_INI_GENTIM2D(xx) ) ): the control vector `xx` (the record of
xx_<name>.<optimcycle>, {iarr: [FArray]}, here the array of iarr 1) enters through the effective control records
(CTRL_INIT_VARIABLES, packages_init_variables.F:605-622; the only place the control file is read), which the step's
CTRL_MAP_GENTIM2D / CTRL_MAP_FORCING add to Qnet; the time loop is drivers/checkpoint.integrate (one checkpoint per
step), COST_FINAL (the_main_loop.F:767-775) is the final cost. The model and its parameters are jit arguments
([F§1]); the control enters as `params_fn` (the effective records in model.pkc), inside the differentiated function.

GOADK lane: `GenarrAdjoint` is the same for a generic init. control (xx_theta / xx_kapgm / xx_kapredi /
xx_bottomdrag of global_ocean.90x40x15/input_ad*): fc(xx) = COST_FINAL(THE_MAIN_LOOP(CTRL_MAP_INI_GENARR(xx))),
the map re-applied to the zero-control Model's initial carry inside the differentiated function (`genarr_apply`).

The TAF side (grdchk_main.F @63cdc0b): `ADM  ref_cost_function` = fc of the unperturbed run (:276-283 after
THE_MAIN_LOOP :359 of the reference run), `ADM  adjoint_gradient` = the adjoint control vector at the point (adxx,
GRDCHK_GETADXX :274), `ADM  finite-diff_grad` = (fcpertplus - fcpertminus)/(grdchk_epsfac*grdchk_eps) with the
control perturbed by +-grdchk_eps at the point (:359-432; useCentralDiff = .TRUE.). The control vector here is
the unscaled record (ones weights: CTRL_PACK/UNPACK scale by sqrt(weight) = 1 in this experiment).
"""

import copy
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.farray import FArray

PREFIX = "(PID.TID 0000.0001) "


def _gather(tree, ex):
    """Every tiled FArray of `tree` with its tile axis gathered to all real tiles (ex.all_tiles); other leaves as
    they are."""
    if tree is None:
        return None

    def one(x):
        if isinstance(x, FArray) and x.tiled:
            return FArray(ex.all_tiles(x.data), x.name, tiled=True, _dims=x.dims)
        return x
    return jax.tree.map(one, tree, is_leaf=lambda v: isinstance(v, FArray))


def final_cost_gathered(m, model, st):
    """fc of COST_FINAL (Model.cost_final) after the loop, as one process computes it, on one device and on a
    TileSharding's blocks alike: every tiled field COST_FINAL reads -- cost.h's tiled accumulators (cMean*), the
    final State (COST_TEST's theta), CTRL_GENARR.h, ECCO.h, COST_WEIGHTS' fields (model.pkc["cost_fixed"]), the
    control records, the traced grid (maskC, R_low: `arrays` of Model.cost_final) -- gathered to every real tile in
    tile order (`_gather`: ex.all_tiles, the identity on one device; under shard_map a psum of the zero-padded
    blocks, exact), then the single-device
    routine. Its per-tile chains (objf_test, ...) and GLOBAL_SUM_TILE_RL's tile-order sum are those of the Fortran on
    every tile, so fc is P-independent bit for bit. cost.h's plain [tile] accumulators (tile_fc, objf_*) are full
    [nTiles] arrays on every device (replicated; a per-step accumulator joins them gathered, cost_tracer.py's
    pattern). Lane M4COSTSHARD: GenarrAdjoint's final cost (the global_ocean.cs32x15/code_ad COST_TEST builds) is
    this one too; before, it read the set-up's global maskC (Model.cost_final's default) and the local theta, which
    does not trace under shard_map (cost_test: objf_test [12] + the [Tloc] chains)."""
    ex = model.ex
    pk = st[0][4]
    cost, genarr = _gather(pk["cost"], ex), _gather(pk.get("genarr"), ex)
    fixed = _gather(model.pkc["cost_fixed"], ex)
    xt = model.pkc.get("xx_tim2d")
    # lane M4COSTSHARD s2: the grid (maskC, gencost's R_low) and tables (ecco_tab) from the traced model, its grid
    # gathered (only the fields COST_FINAL reads survive: XLA drops the unused gathers)
    arrays = model.replace(grid=_gather(model.grid, ex))
    cost, _ = m.cost_final(cost, genarr, fixed=fixed, arrays=arrays,
                           state=_gather(st[0][0], ex),                 # GO lane: COST_TEST reads theta
                           ecco=_gather(pk.get("ecco"), ex),            # lane M4ADCOL session 3: pkg/ecco
                           # lane M4ADLAB s5: CTRL_COST_GEN2D on the control records (read only by the
                           # ecco / ctrl cost build's COST_FINAL, Model._pkg_final)
                           xx=None if xt is None else {"tim2d": _gather(xt, ex)})
    return cost.fc


def problem(m, iarr=1, rec=1):
    """(step, model, st0, xs, theta0, params_fn, final_cost) of drivers/grad.value_and_grad for the Model `m`:
    st = (carry, myTime, myIter); theta = the data array [tile, j, i] of record `rec` of gentim2d control iarr (first
    guess: zeros, Model.xx_zero); every other record of every live gentim2d control is the first guess (zeros), as
    the grdchk perturbs one record of one control (lane M4ADLAB session 5: lab_sea/input_ad has nine gentim2d
    controls of two records each; grdchk on xx_atemp = iarr 1, record 1). params_fn also takes the record as a tiled
    FArray (the sharded drivers place and shard it on its tile axis: `sharded_problem`). Every function reads the
    grid, exchanger and inputs from `model`, so the same functions run on one device and on a TileSharding's blocks
    (drivers/sharded_grad.py). The control records also reach COST_FINAL's CTRL_COST_GEN2D (`xx` of
    Model.cost_final, the ecco / ctrl cost build of lab_sea/code_ad), so fc(theta) includes its xx**2 term."""
    if len(m.initial_carry()) < 5 or "effective" not in m.arrays.pkc:
        raise ValueError("adjoint_run: the Model has no control (useCTRL) / cost (ALLOW_COST) state")
    nrec = {i: len(v) for i, v in m.xx_zero().items()}
    if not 1 <= rec <= nrec[iarr]:
        raise ValueError(f"adjoint_run: gentim2d control {iarr} has records 1..{nrec[iarr]}, not {rec}")
    like = m.xx_zero()[iarr][0]
    tp = m.prm.time

    def xx_full(r):
        # the control vector {iarr: [records]}: record `rec` of control iarr is `r`, every other record zeros of
        # its shape and kind (a device's block under shard_map)
        z = FArray(jnp.zeros_like(r.data), r.name, tiled=r.tiled, _dims=r.dims)
        xx = {i: [z] * n for i, n in nrec.items()}
        xx[iarr] = [r if k == rec - 1 else z for k in range(nrec[iarr])]
        return xx

    def step(model, st, iloop):
        carry, t, it = st
        carry, t, it, _ = m.step(model, carry, iloop, t, it)
        return (carry, t, it)

    def params_fn(theta, model):
        # CTRL_MAP_INI_GENTIM2D with the model's own grid, exchanger and weight records (a device's block and its
        # ShardedExchanger under shard_map)
        r = theta if isinstance(theta, FArray) else FArray(theta, like.name, tiled=like.tiled, _dims=like.dims)
        xx = xx_full(r)
        eff, _ = m.ctrl_effective(xx, arrays=model)
        return model.replace(pkc={**model.pkc, "effective": eff, "xx_tim2d": xx})

    def final_cost(model, st):
        return final_cost_gathered(m, model, st)

    t0, it0 = m.start_counters()
    st0 = (m.initial_carry(), t0, it0)
    xs = jnp.arange(1, tp.nTimeSteps + 1, dtype=jnp.int32)
    return step, m.arrays, st0, xs, like.data, params_fn, final_cost


def sharded_problem(m, iarr=1, rec=1):
    """problem() with theta0 the control record as a tiled FArray (drivers/sharded_grad places theta with
    tile_specs: tiled FArrays are padded and sharded)."""
    step, model, st0, xs, th0, params_fn, final_cost = problem(m, iarr, rec)
    like = m.xx_zero()[iarr][0]
    return step, model, st0, xs, FArray(th0, like.name, tiled=True, _dims=like.dims), params_fn, final_cost


# the State fields ADMONITOR reports (monitor_ad.F:162-179) and the cost.h accumulators COST_TILE writes
AD_STATE = ("etaN", "uVel", "vVel", "wVel", "theta", "salt")
AD_COST = ("cMeanTheta", "cMeanUVel", "cMeanVVel", "cMeanThetaUVel", "cMeanThetaVVel")


def monitor_stats(model, ct):
    """stats_fn of grad.value_and_grad: the cotangent fields at a step boundary that the adjoint monitor needs
    (State: AD_STATE; cost.h: AD_COST). `ct` is the cotangent of (carry, myTime, myIter)."""
    carry = ct[0]
    out = {n: getattr(carry[0], n).data for n in AD_STATE}
    cost = carry[4]["cost"]
    out.update({n: getattr(cost, n).data for n in AD_COST})
    phys = (carry[4].get("ecco") or {}).get("phys")
    if phys is not None:                    # lane M4ADLAB s5: ECCO.h m_eta, m_bp, m_UE, m_VN (ECCO_PHYS's outputs)
        out.update({"ecco_phys_" + n: v.data for n, v in phys.items()})
    sea = carry[4].get("seaice")
    if sea is not None:                     # lane M4ADLAB s5: SEAICE.h for ADSEAICE_MONITOR (ad_seaice_records)
        out.update({"seaice_" + n: sea[n].data for n in AD_SEAICE if n in sea})
    return out


# the SEAICE.h fields ADSEAICE_MONITOR reports (pkg/seaice/seaice_monitor_ad.F:109-120, C-grid, VARIABLE_SALINITY)
AD_SEAICE = ("UICE", "VICE", "AREA", "HEFF", "HSNOW", "HSALT")


def ad_seaice_records(m, stats, *, adjMonitorFreq=None):
    """The `%MON ad_seaice_*` blocks of the reverse sweep, k = n .. 0 (lane M4ADLAB s5): ADSEAICE_MONITOR
    (pkg/seaice/seaice_monitor_ad.F:11-145) where the forward calls SEAICE_MONITOR, i.e. in SEAICE_OUTPUT inside
    DO_THE_MODEL_IO, the last call of FORWARD_STEP (forward_step.F:1182, after MONITOR :1154, COST_TILE :1163 and
    ECCO_PHYS :1169, none of which reads SEAICE.h): the adjoint SEAICE.h variables there are the step-boundary
    cotangents of the sea-ice carry (raw arrays: the routine has no ADEXCH). myIter = nIter0 + k (:95). The k = 0
    block is the reverse of INITIALISE_VARIA's monitor call (TAF prints it; included for comparison)."""
    from mitjax.drivers.run import MonitorHost
    from mitjax.params_io import RunParams, fortran_default
    from mitjax.pkg.monitor.mon_out import mon_out_i, mon_out_rl
    from mitjax.pkg.monitor.mon_set_pref import mon_set_pref
    from mitjax.pkg.monitor.mon_writestats_rl import mon_writestats_rl
    from mitjax.pkg.monitor.monitor import level_view, vec_head
    from mitjax.pkg.monitor.monitor_h import different_multiple, mon_string_none
    if adjMonitorFreq is None:
        rp = RunParams(m.exp.run)
        adjMonitorFreq = rp.get("data", "PARM03", "adjMonitorFreq", default=fortran_default(
            "model/src/set_defaults.F:352", "adjMonitorFreq", m.exp))
    mh = MonitorHost(m)
    mon, cfg, g, tp = mh.mon, mh.cfg, m.grid, m.prm.time
    n = np.asarray(next(iter(stats.values()))).shape[0] - 1
    n0 = len(mon.units.get(mon.mon_ioUnit, []))
    drF1 = vec_head(g.drF, 1)
    like = m.pk0["seaice"]
    for k in range(n, -1, -1):
        myTime = float(np.float64(tp.startTime) + np.float64(tp.deltaTClock) * np.float64(k))
        if not different_multiple(float(adjMonitorFreq), myTime, m.params.deltaTClock):     # :51
            continue
        mon.mon_write_stdout = bool(cfg.monitor_stdio)

        def fld(nm):
            x = like[nm]
            return level_view(FArray(jnp.asarray(np.asarray(stats["seaice_" + nm])[k]), nm, tiled=x.tiled,
                                     _dims=x.dims))
        mon_set_pref("ad_seaice", mon=mon)                                                  # :94
        mon_out_i("_tsnumber", tp.nIter0 + k, mon_string_none, cfg=cfg, mon=mon)            # :95
        mon_out_rl("_time_sec", myTime, mon_string_none, cfg=cfg, mon=mon)                  # :96
        d = [0.0] * 6
        for nm, lab, msk, area in (("UICE", "_aduice", g.maskInW, g.rAw), ("VICE", "_advice", g.maskInS, g.rAs),
                                   ("AREA", "_adarea", g.maskInC, g.rA), ("HEFF", "_adheff", g.maskInC, g.rA),
                                   ("HSNOW", "_adhsnow", g.maskInC, g.rA), ("HSALT", "_adhsalt", g.maskInC, g.rA)):
            if "seaice_" + nm in stats:                                                     # :109-120
                d = mon_writestats_rl(1, fld(nm), lab, level_view(msk), msk, area, drF1, d, cfg=cfg, mon=mon,
                                      ex=m.ex)
    return list(mon.units.get(mon.mon_ioUnit, []))[n0:]


AD_FORCING = ("Qnet", "Qsw", "EmPmR", "fu", "fv")      # monitor_ad.F:195-215 (adQsw under SHORTWAVE_HEATING)


# the FFIELDS.h fields ADEXF_MONITOR reports for iwhen = 3 (pkg/exf/exf_monitor_ad.F:207-218)
AD_EXF3 = ("fu", "fv", "Qnet", "EmPmR", "Qsw")


def ad_exf_records(m, stats, *, exf_adjMonFreq=None, exf_adjMonSelect=None):
    """The `%MON ad_exf_*` blocks of iwhen = 3 in the reverse sweep, step k = n-1 .. 0 (lane M4COSTSHARD):
    ADEXF_MONITOR( 3, myTime, myIter ) (pkg/exf/exf_monitor_ad.F:11-251), which ADEXF_ADJOINT_SNAPSHOTS
    (exf_adjoint_snapshots_ad.F:219) calls at the reverse of EXF_ADJOINT_SNAPSHOTS( 3, ...): with useSEAICE the last
    call of SEAICE_MODEL (seaice_model.F:405, after its _EXCH_XY_RS of EmPmR, Qnet, Qsw :333-337), otherwise the end
    of EXF_GETFORCING (exf_getforcing.F:386-388). The adjoint variables adfu, adfv, adQnet, adEmPmR (adQsw under
    SHORTWAVE_HEATING) there are the cotangents of FFIELDS.h at that point of step k (raw arrays: no ADEXCH copy).
    `stats`: {"fu", "fv", "Qnet", "EmPmR"[, "Qsw"]: [n, tile, j, i]} of those cotangents per step (a driver taps them;
    scripts/m4costshard_adexf.py). myIter = nIter0 + k, myTime = startTime + k*deltaTClock (the step's own, as
    SEAICE_MODEL gets them). exf_adjMonSelect, exf_adjMonFreq: data.exf EXF_NML_01 (defaults 1 and adjMonitorFreq,
    exf_readparms.F:305-306). Ported: the iwhen = 3 arm (:204-218) and its header (:62-107, :220-246; MNC raises);
    the iwhen = 1 / 2 arms (EXF_FIELDS.h cotangents inside EXF_GETFORCING, exf_adjMonSelect >= 2) are not ported:
    a run whose exf_adjMonSelect >= 2 gets the iwhen = 3 blocks only (the others are reported missing by the
    comparison)."""
    from mitjax.drivers.run import MonitorHost
    from mitjax.io.fortran_format import fortran_write
    from mitjax.params_io import RunParams, fortran_default
    from mitjax.pkg.monitor.mon_out import mon_out_i, mon_out_rl
    from mitjax.pkg.monitor.mon_set_pref import mon_set_pref
    from mitjax.pkg.monitor.mon_writestats_rs import mon_writestats_rs
    from mitjax.pkg.monitor.monitor import _banner, level_view, vec_head
    from mitjax.pkg.monitor.monitor_h import different_multiple, mon_string_none
    from mitjax.io.namelist import NULL
    rp = RunParams(m.exp.run)
    if exf_adjMonFreq is None:
        v = rp.nml.var("data.exf", "EXF_NML_01", "exf_adjMonFreq")
        if v is not None and v.value is not NULL:
            exf_adjMonFreq = v.value
        else:                                   # exf_readparms.F:305 exf_adjMonFreq = adjMonitorFreq (data PARM03)
            exf_adjMonFreq = rp.get("data", "PARM03", "adjMonitorFreq", default=fortran_default(
                "model/src/set_defaults.F:352", "adjMonitorFreq", m.exp))
    if exf_adjMonSelect is None:
        exf_adjMonSelect = int(rp.get("data.exf", "EXF_NML_01", "exf_adjMonSelect", default=fortran_default(
            "pkg/exf/exf_readparms.F:306", "exf_adjMonSelect", m.exp)))
    mh = MonitorHost(m)
    mon, cfg, g, tp = mh.mon, mh.cfg, m.grid, m.prm.time
    if cfg.ALLOW_MNC and cfg.useMNC and cfg.monitor_mnc:                                  # :83-97
        raise NotImplementedError("ADEXF_MONITOR: monitor output to MNC is not ported")
    n = np.asarray(stats["fu"]).shape[0]
    n0 = len(mon.units.get(mon.mon_ioUnit, []))
    drF1 = vec_head(g.drF, 1)
    like = m.ff0
    iwhen = 3
    for k in range(n - 1, -1, -1):
        myTime = float(np.float64(tp.startTime) + np.float64(tp.deltaTClock) * np.float64(k))
        if not (exf_adjMonSelect > 0
                and different_multiple(float(exf_adjMonFreq), myTime, m.params.deltaTClock)):   # :62-63
            continue
        mon.mon_write_stdout = bool(cfg.monitor_stdio)                                    # :73-80 (iwhen = 3)
        mon.mon_write_mnc = False                                                         # :81
        if mon.mon_write_stdout:                                                          # :99-107
            _banner(fortran_write("(A,I2)", "// Begin AD_MONITOR EXF statistics for iwhen = ", iwhen), mon)

        def fld(nm):
            x = getattr(like, nm)
            return level_view(FArray(jnp.asarray(np.asarray(stats[nm])[k]), nm, tiled=x.tiled, _dims=x.dims))
        mon_set_pref("ad_exf", mon=mon)                                                   # :112
        mon_out_i("_tsnumber", tp.nIter0 + k, mon_string_none, cfg=cfg, mon=mon)          # :113
        mon_out_rl("_time_sec", myTime, mon_string_none, cfg=cfg, mon=mon)                # :114
        d = [0.0] * 6
        mInC = level_view(g.maskInC)
        for nm, lab in (("fu", "_adfu"), ("fv", "_adfv"), ("Qnet", "_adqnet"), ("EmPmR", "_adempmr"),
                        ("Qsw", "_adqsw")):                                               # :207-218
            if nm == "Qsw" and not m.cfg.cpp.SHORTWAVE_HEATING:              # :214 #ifdef
                continue
            d = mon_writestats_rs(1, fld(nm), lab, mInC, g.maskInC, g.rA, drF1, d, cfg=cfg, mon=mon, ex=m.ex)
        if mon.mon_write_stdout:                                                          # :224-240
            _banner(fortran_write("(A,I2)", "// End AD_MONITOR EXF statistics for iwhen = ", iwhen), mon)
        mon.mon_write_stdout = False                                                      # :242
        mon.mon_write_mnc = False                                                         # :243
    return list(mon.units.get(mon.mon_ioUnit, []))[n0:]


def monitor_stats_ptr(model, ct):
    """PTRACERS lane: stats_fn for the adjoint monitor of a ptracer / COST_TRACER build (tutorial_tracer_adjsens):
    the State cotangents of AD_STATE and of every pTracer_NN (ADPTRACERS_MONITOR), the FFIELDS.h cotangents of
    AD_FORCING present in the build (the ad_forcing block of monitorSelect >= 4) and cost.h's objf_tracer cotangent
    (for COST_TILE's COST_TRACER transpose, which the reverse sweep runs before ADMONITOR)."""
    carry = ct[0]
    st, ff = carry[0], carry[1]
    out = {n: getattr(st, n).data for n in AD_STATE}
    out.update({n: getattr(st, n).data for n in st.names() if n.startswith("pTracer_")})
    out.update({"ff_" + n: getattr(ff, n).data for n in AD_FORCING if n in ff})
    out["objf_tracer"] = carry[4]["cost"].objf_tracer
    return out


class Adjoint:
    """The jitted fc(theta), (fc, dfc/dtheta) and tangent dfc.v of one Model, built once (one compile each; the
    model is an argument: a changed model -- e.g. a planted cost weight in model.pkc -- reuses the programs).
    `monitor=True`: value_and_grad also returns the step-boundary cotangent fields (monitor_stats at boundaries
    0..n through the stats hook of drivers/checkpoint) for the adjoint monitor (`ad_monitor_records`)."""

    def __init__(self, m, *, schedule="step", iarr=1, rec=1, monitor=False):
        from mitjax.drivers.checkpoint import make_sinks, n_steps, prepare_state
        from mitjax.drivers.grad import _objective
        step, self.model, self.st0, self.xs, self.theta0, params_fn, final_cost = problem(m, iarr, rec)
        self._stepfn, self._params_fn, self._final_cost = step, params_fn, final_cost
        kw = dict(segments=None, cost=None, final_cost=final_cost, init_fn=lambda t, s: s, params_fn=params_fn)
        self.monitor = monitor
        if monitor:
            J = _objective(step, schedule=schedule, stats_fn=monitor_stats, **kw)
            self._sinks = make_sinks(monitor_stats, self.model,
                                     prepare_state(step, self.model, self.st0, self.xs[0]), n_steps(self.xs))

            def vg(th, model, st0, xs, sinks):
                val, (g, stats) = jax.value_and_grad(lambda t, sk: J(t, model, st0, xs, sk), argnums=(0, 1))(
                    th, sinks)
                return val, g, stats
            self._vg = jax.jit(vg)
        else:
            J = _objective(step, schedule=schedule, **kw)
            self._vg = jax.jit(jax.value_and_grad(J))
        J0 = _objective(step, schedule="none", **kw)
        self._J = jax.jit(J0)
        self._jvp = jax.jit(lambda th, v, model, st0, xs: jax.jvp(lambda t: J0(t, model, st0, xs), (th,), (v,)))
        self.m = m

    def value_and_grad(self, theta=None, model=None):
        """(fc, dfc/dtheta), with monitor=True (fc, dfc/dtheta, stats)."""
        th, model = self.theta0 if theta is None else theta, self.model if model is None else model
        if self.monitor:
            return self._vg(th, model, self.st0, self.xs, self._sinks)
        return self._vg(th, model, self.st0, self.xs)

    def boundary_states(self, theta=None, model=None):
        """The carries (with myTime, myIter) at the step boundaries 0..n of the forward run (one jitted step)."""
        th, model = self.theta0 if theta is None else theta, self.model if model is None else model
        mdl = self._params_fn(th, model)
        stepj = jax.jit(self._stepfn)
        st = self.initial_state(th, model) if hasattr(self, "_init") else self.st0   # PTRACERS lane
        out = [st]
        for x in np.asarray(self.xs):
            st = stepj(mdl, st, jnp.int32(x))
            out.append(st)
        return out

    def cost(self, theta=None, model=None):
        return self._J(self.theta0 if theta is None else theta, self.model if model is None else model,
                       self.st0, self.xs)

    def jvp(self, v, theta=None, model=None):
        """(fc, dfc.v): the tangent-linear model in direction v (jax.jvp)."""
        return self._jvp(self.theta0 if theta is None else theta, v, self.model if model is None else model,
                         self.st0, self.xs)

    def grdchk_fd(self, points, *, eps=None, theta=None, model=None):
        """GRDCHK_MAIN's finite differences (grdchk_main.F:359-432): for each storage index (tile, j, i) the costs
        with the control at that point +eps and -eps (grdchk_eps of data.grdchk), gfd = (fp - fm)/(2*eps). Returns
        [(point, fcpertplus, fcpertminus, gfd)]."""
        from mitjax.pkg.grdchk.grdchk import fd_gradient, grdchk_epsfac
        eps = float(self.m.exp.params["data.grdchk:grdchk_nml:grdchk_eps"]) if eps is None else float(eps)
        th = self.theta0 if theta is None else theta
        out = []
        for p in points:
            fp = float(self.cost(th.at[p].add(eps), model))            # :359 the forward run with xx + eps
            fm = float(self.cost(th.at[p].add(-eps), model))           # :400 the forward run with xx - eps
            out.append((p, fp, fm, fd_gradient(fp, fm, eps, grdchk_epsfac(True))))
        return out


def gradient(m, theta=None, *, schedule="step", iarr=1):
    """(fc, dfc/dtheta) at `theta` (default: the first guess, zeros); one-off (builds an Adjoint)."""
    return Adjoint(m, schedule=schedule, iarr=iarr).value_and_grad(theta)


def grdchk_fd(m, points, *, eps=None, theta=None, iarr=1):
    """Adjoint.grdchk_fd, one-off."""
    return Adjoint(m, iarr=iarr).grdchk_fd(points, eps=eps, theta=theta)


def adm_lines(fc, adjoint_gradient, finite_diff_grad):
    """The `ADM` lines of one check point as grdchk_main.F:510-518 prints them: WRITE(msgBuf,'(A30,1PE22.14)')
    ' ADM  ref_cost_function      =', fcref (and adxxmemo, gfd), through PRINT_MESSAGE (SQUEEZE_RIGHT)."""
    from mitjax.io.fortran_format import fortran_write
    out = []
    for lit, v in ((" ADM  ref_cost_function      =", fc), (" ADM  adjoint_gradient       =", adjoint_gradient),
                   (" ADM  finite-diff_grad       =", finite_diff_grad)):
        out.append((PREFIX + fortran_write("(A30,1PE22.14)", lit, float(v))).rstrip())
    return out


def run(m, points, *, schedule="step"):
    """fc, the adjoint gradient and the finite differences at `points`; returns SimpleNamespace(fc, grad, fd, lines)."""
    a = Adjoint(m, schedule=schedule)
    fc, g = a.value_and_grad()
    fd = a.grdchk_fd(points)
    lines = []
    for (p, fp, fm, gfd) in fd:
        lines += adm_lines(fc, np.asarray(g)[p], gfd)
    return SimpleNamespace(fc=float(fc), grad=np.asarray(g), fd=fd, lines=lines)


def ad_monitor_fields(m, ct, carry, k):
    """The adjoint variables ADMONITOR sees at step k (0..n): the boundary cotangents `ct` (monitor_stats of
    boundary k) of the State, plus -- for k >= 1 -- COST_TILE's contribution (FORWARD_STEP calls COST_TILE right
    after MONITOR, forward_step.F:1151-1164, so in the reverse sweep ADCOST_TILE runs before ADMONITOR): the
    transpose of COST_TILE's accumulation at the boundary State `carry` applied to the cost.h cotangents. Then the
    copies with ADEXCH of mon_AdVarExch = 2 (monitor_ad.F:162-179). Returns a namespace adEtaN ... adSalt."""
    from mitjax.ad.adexch import copy_ad_uv_outp, copy_advar_outp
    from mitjax.pkg.cost.cost_tile import cost_tile
    st = carry[0][0]
    fld = {n: FArray(jnp.asarray(ct[n]), n, tiled=getattr(st, n).tiled, _dims=getattr(st, n).dims)
           for n in AD_STATE}
    if k >= 1 and m.cfg.cpp.ALLOW_COST:
        pk, pks, g = carry[0][4], m.pks, m.grid
        tp = m.prm.time
        myTime = float(np.float64(tp.startTime) + np.float64(tp.deltaTClock) * np.float64(k))
        myIter = tp.nIter0 + k
        cost0 = pk["cost"]

        def f(theta, u, v):
            # COST_ACCUMULATE_MEAN writes into the cost.h object it gets: hand it a copy (cost0 is the boundary
            # carry's, which must not hold the tracers of this vjp)
            c = cost_tile(copy.copy(cost0), myTime, myIter, cfg=m.cfg, endTime=pks["endTime"], lastinterval=pks["lastinterval"],
                          theta=theta, uVel=u, vVel=v, maskC=g.maskC, maskW=g.maskW, maskS=g.maskS,
                          deltaTClock=m.params.deltaTClock, lastinterval_traced=m.arrays.pkc["lastinterval"])
            return {n: getattr(c, n).data for n in AD_COST}
        _, vjp = jax.vjp(f, st.theta, st.uVel, st.vVel)
        dth, du, dv = vjp({n: jnp.asarray(ct[n]) for n in AD_COST})
        fld["theta"] = FArray(fld["theta"].data + dth.data, "theta", tiled=st.theta.tiled, _dims=st.theta.dims)
        fld["uVel"] = FArray(fld["uVel"].data + du.data, "uVel", tiled=st.uVel.tiled, _dims=st.uVel.dims)
        fld["vVel"] = FArray(fld["vVel"].data + dv.data, "vVel", tiled=st.vVel.tiled, _dims=st.vVel.dims)
    phys_ct = {n[len("ecco_phys_"):]: ct[n] for n in ct if n.startswith("ecco_phys_")}
    if k >= 1 and phys_ct:
        # lane M4ADLAB s5: FORWARD_STEP calls ECCO_PHYS after MONITOR and COST_TILE (forward_step.F:1154, :1163,
        # :1169), so in the reverse sweep the adjoint of ECCO_PHYS (m_eta, m_bp <- etaN; m_UE, m_VN <- uVel, vVel)
        # runs before ADMONITOR: its transpose applied to the ECCO.h cotangents at the boundary. Not at k = 0: the
        # initial ECCO_PHYS (ECCO_INIT_VARIA, packages_init_variables.F:601 inside INITIALISE_VARIA's :263) comes
        # before the initial MONITOR (initialise_varia.F:383), so its adjoint runs after ADMONITOR there.
        from mitjax.pkg.ecco.ecco_phys import ecco_phys
        g, myIter = m.grid, int(m.prm.time.nIter0 + k)

        def fp_(eta, u, v):
            ph = ecco_phys(myIter, cfg=m.cfg, grid=g, params=m.params, eos=m.eos,
                           state=st.replace(etaN=eta, uVel=u, vVel=v), ff=carry[0][1], fp=m.fp)
            return {n: ph[n].data for n in phys_ct}
        _, vjp = jax.vjp(fp_, st.etaN, st.uVel, st.vVel)
        de, du, dv = vjp({n: jnp.asarray(c) for n, c in phys_ct.items()})
        for n, d in (("etaN", de), ("uVel", du), ("vVel", dv)):
            fld[n] = FArray(fld[n].data + d.data, n, tiled=getattr(st, n).tiled, _dims=getattr(st, n).dims)
    ex = m.ex
    u, v = copy_ad_uv_outp(fld["uVel"], fld["vVel"], 34, ex=ex)              # monitor_ad.F:165-166
    return SimpleNamespace(adEtaN=copy_advar_outp(fld["etaN"], 12, ex=ex),    # :162
                           aduVel=u, advVel=v,
                           adwVel=copy_advar_outp(fld["wVel"], 12, ex=ex),    # :171
                           adTheta=copy_advar_outp(fld["theta"], 12, ex=ex),  # :174
                           adSalt=copy_advar_outp(fld["salt"], 12, ex=ex))    # :175


def ad_monitor_records(m, stats, carries, *, adjMonitorFreq=None):
    """The `%MON ad_*` blocks of the reverse sweep, k = n .. 0 as the adjoint build prints them (ADMONITOR at
    myIter = nIter0 + k, myTime = startTime + k*deltaTClock): `stats` from Adjoint(monitor=True).value_and_grad,
    `carries` from Adjoint.boundary_states. adjMonitorFreq: data PARM03 (set_defaults.F:352 `0.`: no block)."""
    from mitjax.drivers.run import MonitorHost
    from mitjax.params_io import RunParams, fortran_default
    from mitjax.pkg.autodiff.adjoint_monitor import admonitor
    if adjMonitorFreq is None:
        rp = RunParams(m.exp.run)
        adjMonitorFreq = rp.get("data", "PARM03", "adjMonitorFreq", default=fortran_default(
            "model/src/set_defaults.F:352", "adjMonitorFreq", m.exp))
    mh = MonitorHost(m)
    tp = m.prm.time
    n = len(carries) - 1
    n0 = len(mh.mon.units.get(mh.mon.mon_ioUnit, []))
    for k in range(n, -1, -1):
        ct = {key: np.asarray(v)[k] for key, v in stats.items()}
        ad = ad_monitor_fields(m, ct, carries[k], k)
        myTime = float(np.float64(tp.startTime) + np.float64(tp.deltaTClock) * np.float64(k))
        admonitor(myTime, tp.nIter0 + k, cfg=mh.cfg, params=mh.params, grid=mh._grid(carries[k][0]), ad=ad,
                  mon=mh.mon,
                  adjMonitorFreq=float(adjMonitorFreq), ex=m.ex)
    return list(mh.mon.units.get(mh.mon.mon_ioUnit, []))[n0:]


# ---- GOADK lane (M2 Task 24): generic init. controls (CTRL_MAP_INI_GENARR) ----------------------------------------

def genarr_apply(m, key, theta, model, st0):
    """CTRL_MAP_INI_GENARR (ctrl_init_variables.F:104-106) of the control vector `theta` (the record of
    xx_<name>.<optimcycle> of the genarr control `key` = (dim, iarr): [tile, Nr, j, i] or [tile, j, i]) on the
    Model's first-guess fields: theta / salt in the initial State, GM_inpK3dGM / GM_inpK3dRedi in the initial GMREDI.h
    (carry[4]["gm"]), bottomDragFld in model.pkc["ctrlf"]. The first-guess Model was built with the zero control, so
    its fields are the unmapped ones plus 0 (bit for bit the unmapped values): adding the map of `theta` to them is the
    Fortran's fld + xx*mask/sqrt(w). Returns (model, st0).
    PTRACERS lane (tutorial_tracer_adjsens: xx_ptr1): the pTracer fields join the map (`ptr=`), and when
    INITIALISE_VARIA runs CONVECTIVE_ADJUSTMENT_INI (Model.convect_ini) it follows the map here, as in the Fortran
    (initialise_varia.F:263-295) -- on model.grid / params / eos, the jit arguments; st0's State is then the State
    before the map (GenarrAdjoint takes Model.state0_pre_genarr), since the Model's initial State is adjusted."""
    from mitjax.pkg.ctrl.ctrl_map_ini_genarr import ctrl_map_ini_genarr
    carry, t0, it0 = st0
    state, pk = carry[0], dict(carry[4])
    g2, g3 = m.genarr_parms
    like = m.genarr_xx0[key]
    fields = dict(theta=state.theta, salt=state.salt)
    if "diffKr" in state:                                   # GO lane: xx_diffkr (ctrl_map_ini_genarr.F:404-407)
        fields["diffKr"] = state.diffKr
    ptr = m.genarr_setup["ptr"]                           # PTRACERS lane: the xx_ptr<n> fields (usePTRACERS)
    if ptr is not None:
        fields.update({f"pTracer_{n:02d}": getattr(state, f"pTracer_{n:02d}")
                       for n in range(1, ptr.PTRACERS_num + 1)})
    if "gm" in pk:
        fields.update({n: getattr(pk["gm"], n) for n in ("GM_inpK3dGM", "GM_inpK3dRedi") if n in pk["gm"]})
    if "ctrlf" in model.pkc:
        fields["bottomDragFld"] = model.pkc["ctrlf"].bottomDragFld
    w = {(int(k.split("_")[0]), int(k.split("_")[1])): v for k, v in model.pkc["genarr_w"].items()}
    # GO lane: only the control `key` is (re)mapped here; the other live controls' maps (their first guess) are
    # already in the zero-control Model's fields: they are passed as not in use (blank weight, ctrl_map_ini_genarr
    # .F:354 IF (xx_genarr3d_weight(iarr).NE.' '))
    import dataclasses
    g2 = [g if (2, g.iarr) == key else dataclasses.replace(g, weight=" ") for g in g2]
    g3 = [g if (3, g.iarr) == key else dataclasses.replace(g, weight=" ") for g in g3]
    out, _ = ctrl_map_ini_genarr(fields, cfg=m.cfg, genarr2d=g2, genarr3d=g3,
                                 xx_in={key: FArray(theta, like.name, tiled=like.tiled, _dims=like.dims)},
                                 weight_in={key: w[key]}, maskC=model.grid.maskC, ex=model.ex,
                                 ptr=model.params if ptr is not None else None)
    state = state.replace(theta=out["theta"], salt=out["salt"],
                          **{n: v for n, v in out.items() if n.startswith("pTracer_") or n == "diffKr"})
    if m.convect_ini:            # PTRACERS lane: INITIALISE_VARIA's CONVECTIVE_ADJUSTMENT_INI after the map (:280-295)
        from mitjax.model.src.convective_adjustment_ini import convective_adjustment_ini
        from mitjax.pkg.ptracers.ptracers_fields_h import ptf_of_state, state_with_ptf
        tp = m.prm.time
        ptf = ptf_of_state(state, model.params) if ptr is not None else None
        state, ptf = convective_adjustment_ini(tp.startTime, tp.nIter0, cfg=m.cfg, grid=model.grid,
                                               params=model.params, eos=model.eos, state=state, ptr=model.params,
                                               ptf=ptf)
        if ptr is not None:
            state = state_with_ptf(state, ptf)
    if "gm" in pk:
        pk["gm"] = pk["gm"].replace(**{n: out[n] for n in ("GM_inpK3dGM", "GM_inpK3dRedi") if n in out})
    if "ctrlf" in model.pkc:
        from mitjax.drivers.model import PkgCommon
        model = model.replace(pkc={**model.pkc, "ctrlf": PkgCommon(bottomDragFld=out["bottomDragFld"])})
    carry = (state,) + tuple(carry[1:4]) + (pk,)
    return model, (carry, t0, it0)


class GenarrAdjoint:
    """Adjoint for a Model whose control is a generic init. control (xx_genarr2d/3d: global_ocean.90x40x15/input_ad
    xx_theta, .kapgm xx_kapgm, .kapredi xx_kapredi, .bottomdrag xx_bottomdrag): fc(theta) =
    COST_FINAL(THE_MAIN_LOOP(CTRL_MAP_INI_GENARR(theta))); the same jitted value_and_grad / cost / jvp / grdchk_fd as
    Adjoint (the control enters at initialisation, inside the differentiated function). `monitor=True`: as Adjoint,
    value_and_grad also returns the step-boundary cotangents (monitor_stats) for ad_monitor_records, with
    boundary_states giving the forward carries."""

    def __init__(self, m, *, schedule="step", monitor=False, key=None):
        from mitjax.drivers.checkpoint import SAVE_NAMES, integrate, make_sinks, n_steps, prepare_state
        if key is None:
            if len(m.genarr_live) != 1:
                raise NotImplementedError(f"GenarrAdjoint: {len(m.genarr_live)} genarr controls: name the one to "
                                          "differentiate (`key` = (dim, iarr), e.g. the grdchk variable)")
            key = m.genarr_live[0]
        elif key not in m.genarr_live:
            raise ValueError(f"GenarrAdjoint: {key} is not a live genarr control ({m.genarr_live})")
        # GO lane: with several live controls the others stay at their first guess (zero: their maps are in the
        # Model's initial carry), as GRDCHK perturbs one control and TAF's adxx of the others is not compared
        self.key = key
        self.m = m
        tp = m.prm.time

        def step(model, st, iloop):
            carry, t, it = st
            carry, t, it, _ = m.step(model, carry, iloop, t, it)
            return (carry, t, it)

        def final_cost(model, st):
            # lane M4COSTSHARD: COST_FINAL on the gathered tiled fields of `model` and the final State (grid,
            # exchanger and cost_fixed from the argument, never the set-up's), so the objective also runs on a
            # TileSharding's blocks (sharded_value_and_grad_fn)
            return final_cost_gathered(m, model, st)

        def J(theta, model, st0, xs, sched, sinks=None):
            model, s0 = genarr_apply(m, key, theta, model, st0)
            s_n, _ = integrate(step, model, s0, xs, schedule=sched, save_names=SAVE_NAMES,
                               stats_fn=None if sinks is None else monitor_stats, sinks=sinks)
            return final_cost(model, s_n)

        t0, it0 = m.start_counters()
        c0 = m.initial_carry()
        if m.convect_ini:            # PTRACERS lane: the State before the map (CONVECTIVE_ADJUSTMENT_INI follows it)
            c0 = (m.state0_pre_genarr,) + tuple(c0[1:])
        self.st0 = (c0, t0, it0)
        self.xs = jnp.arange(1, tp.nTimeSteps + 1, dtype=jnp.int32)
        self.model = m.arrays
        self.theta0 = m.genarr_xx0[key].data
        self._stepfn, self._final_cost = step, final_cost
        self._Jfn, self.schedule = J, schedule
        # PTRACERS lane: what boundary_states / the reverse sweeps of the tests use (Adjoint's interface)
        self._params_fn = lambda th, model: genarr_apply(m, key, th, model, self.st0)[0]
        self._init = lambda th, model, st: genarr_apply(m, key, th, model, st)[1]
        self.monitor = monitor
        if monitor:                  # GOADK lane: the adjoint monitor's stats hook, as Adjoint(monitor=True)
            self._sinks = make_sinks(monitor_stats, self.model,
                                     prepare_state(step, self.model, self.st0, self.xs[0]), n_steps(self.xs))

            def vg(th, mo, s, x, sinks):
                val, (g, stats) = jax.value_and_grad(lambda t, sk: J(t, mo, s, x, schedule, sk), argnums=(0, 1))(
                    th, sinks)
                return val, g, stats
            self._vg = jax.jit(vg)
        else:
            self._vg = jax.jit(jax.value_and_grad(lambda th, mo, s, x: J(th, mo, s, x, schedule)))
        self._J = jax.jit(lambda th, mo, s, x: J(th, mo, s, x, "none"))
        self._jvp = jax.jit(lambda th, v, mo, s, x: jax.jvp(lambda t: J(t, mo, s, x, "none"), (th,), (v,)))

    value_and_grad = Adjoint.value_and_grad
    cost = Adjoint.cost
    jvp = Adjoint.jvp
    grdchk_fd = Adjoint.grdchk_fd
    boundary_states = Adjoint.boundary_states

    def initial_state(self, theta=None, model=None):
        """PTRACERS lane: the step state (carry, myTime, myIter) after the control map (and
        CONVECTIVE_ADJUSTMENT_INI)."""
        return jax.jit(self._init)(self.theta0 if theta is None else theta,
                                   self.model if model is None else model, self.st0)

    def theta_farray(self, theta=None):
        """The control record (default the first guess) as the tiled FArray the sharded drivers place."""
        like = self.m.genarr_xx0[self.key]
        return FArray(self.theta0 if theta is None else theta, like.name, tiled=True, _dims=like.dims)

    def sharded_value_and_grad_fn(self, sh, model=None):
        """Lane M4COSTSHARD: f(theta4, model4, st04, xs, seed) -> (fc, seed * dfc/dtheta4) on the TileSharding `sh`:
        jit(shard_map(check_vma=True)) of jax.vjp of this objective (the same J: CTRL_MAP_INI_GENARR, the loop with
        this instance's schedule, COST_FINAL on the gathered fields, `final_cost_gathered`), as drivers/sharded_grad.
        sharded_value_and_grad_fn does for grad._objective. Arguments placed as there: theta4 =
        sh.put_tree(self.theta_farray()), model4 = sharded_grad.place_model(sh, model), st04 = sh.put_tree(self.st0),
        xs and the seed replicated. The gradient comes back padded [Tpad, ...] with the tile axis sharded. `model`:
        the arrays the program is built for (default self.model; their static options -- e.g. a backward-only switch
        or another CG2D / LSR derivative -- are part of the specs, so pass the model the program is called with)."""
        from mitjax.drivers.sharded_grad import tile_specs
        if self.monitor:
            raise NotImplementedError("GenarrAdjoint.sharded_value_and_grad_fn: monitor=True is not sharded")
        J, sched = self._Jfn, self.schedule

        def body(th, mo, s, x, sd):
            val, pull = jax.vjp(lambda t: J(t.data, mo, s, x, sched), th)
            return val, pull(sd)[0]
        th_s = tile_specs(sh, self.theta_farray())
        mo_s = tile_specs(sh, self.model if model is None else model)
        return sh.shard_map(body, in_specs=(th_s, mo_s, tile_specs(sh, self.st0), sh.REP,
                                            sh.REP), out_specs=(sh.REP, th_s))

    def value_and_grad_seed_fn(self):
        """The single-device counterpart of sharded_value_and_grad_fn (the same body, jitted): f(theta, model, st0,
        xs, seed) -> (fc, seed * dfc/dtheta) with theta the tiled FArray (theta_farray)."""
        J, sched = self._Jfn, self.schedule

        def body(th, mo, s, x, sd):
            val, pull = jax.vjp(lambda t: J(t.data, mo, s, x, sched), th)
            return val, pull(sd)[0]
        return jax.jit(body)


def ad_monitor_fields_ptr(m, ct, carry, k):
    """PTRACERS lane: the adjoint variables ADMONITOR and ADPTRACERS_MONITOR see at step k (0..n) of a ptracer /
    COST_TRACER build (tutorial_tracer_adjsens), from the boundary cotangents `ct` (monitor_stats_ptr of boundary k):
    for k >= 1 COST_TILE's COST_TRACER transpose (cost_tile.F:146-152, which the reverse sweep runs before ADMONITOR:
    forward_step.F:1151-1164) at the boundary State `carry` applied to the objf_tracer cotangent is added to
    adpTracer; then the copies of mon_AdVarExch = 2: COPY_ADVAR_OUTP / COPY_AD_UV_OUTP with ADEXCH for the dynamics
    (vTypes 12, 34; monitor_ad.F:162-179) and the forcing (vTypes 11, 33; :196-215). adptracer is printed as it is
    (ptracers_monitor_ad.F:108-110). Returns (namespace for admonitor, [adptracer FArray per tracer])."""
    from mitjax.ad.adexch import copy_ad_uv_outp, copy_advar_outp
    from mitjax.pkg.cost.cost_tracer import cost_tracer
    from mitjax.pkg.ptracers.ptracers_fields_h import ptf_of_state
    st, ff = carry[0][0], carry[0][1]

    def fa(n, like):
        return FArray(jnp.asarray(ct[n]), like.name, tiled=like.tiled, _dims=like.dims)
    fld = {n: fa(n, getattr(st, n)) for n in AD_STATE}
    ptrn = sorted(n for n in ct if n.startswith("pTracer_"))
    adp = {n: fa(n, getattr(st, n)) for n in ptrn}
    if k >= 1 and m.cfg.cpp.ALLOW_COST_TRACER:
        grid = m.arrays.grid.replace(hFacC=st.hFacC) if m.cfg.cpp.NONLIN_FRSURF else m.arrays.grid
        p = m.params

        def f(ptr01):
            ptf = ptf_of_state(st.replace(pTracer_01=ptr01), p)
            return cost_tracer(jnp.zeros_like(jnp.asarray(ct["objf_tracer"])), cfg=m.cfg, grid=grid, params=p,
                               ptr=p, ptf=ptf, ex=m.ex)
        _, vjp = jax.vjp(f, st.pTracer_01)
        d, = vjp(jnp.asarray(ct["objf_tracer"]))
        adp["pTracer_01"] = FArray(adp["pTracer_01"].data + d.data, "pTracer_01", tiled=st.pTracer_01.tiled,
                                   _dims=st.pTracer_01.dims)
    phys_ct = {n[len("ecco_phys_"):]: ct[n] for n in ct if n.startswith("ecco_phys_")}
    if k >= 1 and phys_ct:
        # lane M4ADLAB s5: FORWARD_STEP calls ECCO_PHYS after MONITOR and COST_TILE (forward_step.F:1154, :1163,
        # :1169), so in the reverse sweep the adjoint of ECCO_PHYS (m_eta, m_bp <- etaN; m_UE, m_VN <- uVel, vVel)
        # runs before ADMONITOR: its transpose applied to the ECCO.h cotangents at the boundary. Not at k = 0: the
        # initial ECCO_PHYS (ECCO_INIT_VARIA, packages_init_variables.F:601 inside INITIALISE_VARIA's :263) comes
        # before the initial MONITOR (initialise_varia.F:383), so its adjoint runs after ADMONITOR there.
        from mitjax.pkg.ecco.ecco_phys import ecco_phys
        g, myIter = m.grid, int(m.prm.time.nIter0 + k)

        def fp_(eta, u, v):
            ph = ecco_phys(myIter, cfg=m.cfg, grid=g, params=m.params, eos=m.eos,
                           state=st.replace(etaN=eta, uVel=u, vVel=v), ff=carry[0][1], fp=m.fp)
            return {n: ph[n].data for n in phys_ct}
        _, vjp = jax.vjp(fp_, st.etaN, st.uVel, st.vVel)
        de, du, dv = vjp({n: jnp.asarray(c) for n, c in phys_ct.items()})
        for n, d in (("etaN", de), ("uVel", du), ("vVel", dv)):
            fld[n] = FArray(fld[n].data + d.data, n, tiled=getattr(st, n).tiled, _dims=getattr(st, n).dims)
    ex = m.ex
    u, v = copy_ad_uv_outp(fld["uVel"], fld["vVel"], 34, ex=ex)              # monitor_ad.F:165-166
    ns = SimpleNamespace(adEtaN=copy_advar_outp(fld["etaN"], 12, ex=ex),      # :162
                         aduVel=u, advVel=v,
                         adwVel=copy_advar_outp(fld["wVel"], 12, ex=ex),      # :171
                         adTheta=copy_advar_outp(fld["theta"], 12, ex=ex),    # :174
                         adSalt=copy_advar_outp(fld["salt"], 12, ex=ex))      # :175
    if "ff_Qnet" in ct:                                                       # :196-215 (monitorSelect >= 4)
        ns.adQnet = copy_advar_outp(fa("ff_Qnet", ff.Qnet), 11, ex=ex)        # :198
        ns.adQsw = copy_advar_outp(fa("ff_Qsw", ff.Qsw), 11, ex=ex) if m.cfg.cpp.SHORTWAVE_HEATING else None  # :202
        ns.adEmPmR = copy_advar_outp(fa("ff_EmPmR", ff.EmPmR), 11, ex=ex)     # :206
        ns.adfu, ns.adfv = copy_ad_uv_outp(fa("ff_fu", ff.fu), fa("ff_fv", ff.fv), 33, ex=ex)   # :209-210
    return ns, [adp[n] for n in ptrn]


def ad_monitor_records_ptr(m, stats, carries, *, adjMonitorFreq=None):
    """PTRACERS lane: the `%MON ad_*` blocks of the reverse sweep of a ptracer build, k = n .. 0 (ADMONITOR at
    myIter = nIter0 + k, then its ADPTRACERS_MONITOR, monitor_ad.F:262-264): `stats` from
    Adjoint(monitor=True).value_and_grad (monitor_stats_ptr), `carries` from Adjoint.boundary_states."""
    from mitjax.drivers.run import MonitorHost
    from mitjax.params_io import RunParams, fortran_default
    from mitjax.pkg.autodiff.adjoint_monitor import admonitor
    from mitjax.pkg.ptracers.ptracers_monitor_ad import adptracers_monitor
    if adjMonitorFreq is None:
        rp = RunParams(m.exp.run)
        adjMonitorFreq = rp.get("data", "PARM03", "adjMonitorFreq", default=fortran_default(
            "model/src/set_defaults.F:352", "adjMonitorFreq", m.exp))
    mh = MonitorHost(m)
    tp = m.prm.time
    n = len(carries) - 1
    n0 = len(mh.mon.units.get(mh.mon.mon_ioUnit, []))
    for k in range(n, -1, -1):
        ct = {key: np.asarray(v)[k] for key, v in stats.items()}
        ad, adp = ad_monitor_fields_ptr(m, ct, carries[k], k)
        myTime = float(np.float64(tp.startTime) + np.float64(tp.deltaTClock) * np.float64(k))
        g = mh._grid(carries[k][0])
        admonitor(myTime, tp.nIter0 + k, cfg=mh.cfg, params=mh.params, grid=g, ad=ad, mon=mh.mon,
                  adjMonitorFreq=float(adjMonitorFreq), ex=m.ex)
        adptracers_monitor(myTime, tp.nIter0 + k, cfg=mh.cfg, params=mh.params, grid=g, ptr=m.params, adptracer=adp,
                           mon=mh.mon, adjMonitorFreq=float(adjMonitorFreq), ex=m.ex)
    return list(mh.mon.units.get(mh.mon.mon_ioUnit, []))[n0:]
