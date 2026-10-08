"""`python -m mitjax run <experiment dir> --variant <v> --out <dir>`: a forward run as THE_MODEL_MAIN orders it, with
the time loop as `lax.scan` chunks and the host-side parts of FORWARD_STEP between the chunks (plan Task 12).

    rundir = make_rundir(exp_dir, variant, out)      # testreport's linkdata: the input files linked into out/rundir
    m = Model(exp, rundir)                           # INI_PARMS, INITIALISE_FIXED, INITIALISE_VARIA (model.py)
    res = forward(m)                                 # INITIALISE_VARIA's MONITOR, then THE_MAIN_LOOP

The traced part of every step is FORWARD_STEP (mitjax/model/src/forward_step.py), scanned by
mitjax/drivers/the_main_loop.py. The host decides, from the concrete clock (forward_step.F:807-808 computed in float64
on the host, the same operations as the traced counters), at which steps it must see the State:
  * MON_CALC_ADVCFL (thermodynamics.F:131-136: monOutputCFL = monitorSelect >= 2 .AND. DIFFERENT_MULTIPLE(
    monitorFreq, wrTime, deltaTClock); wrTime = myTime + deltaTClock with myTime the step's start time, or with
    staggerTimeStep wrTime = myTime, the step's end time: THERMODYNAMICS runs after forward_step.F:807-808 then)
    -- it reads the step's flow;
  * MONITOR (forward_step.F:1151-1156, monitor.F:48: DIFFERENT_MULTIPLE(monitorFreq, myTime, deltaTClock));
  * DO_WRITE_PICKUP (forward_step.F:1196, do_write_pickup.F:58-77: pChkPtFreq, chkPtFreq, modelEnd with
    writePickupAtEnd; modelEnd = myTime.EQ.endTime .OR. myIter.EQ.nEndIter, forward_step.F:1172).
A scan chunk ends exactly at each such step, so the host routines see the Fortran's states. Printed per step (in the
Fortran's order): CG2D's Sum(rhs) line (cg2d.F:196-201, debugLevel >= debLevZero), SOLVE_FOR_PRESSURE's solver lines
(solve_for_pressure.F:331-350, at monitorFreq, debugLevel >= debLevA), the MONITOR block, DO_WRITE_PICKUP's lines.
Not ported (output files only; no STDOUT record in the M1 runs): DO_THE_MODEL_IO (forward_step.F:1182), the snapshot
files of dumpFreq / dumpInitAndLast; STATE_SUMMARY and the parameter printout of INI_PARMS / CONFIG_SUMMARY.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mitjax.drivers.model import Model, debLevA, debLevZero
from mitjax.drivers.the_main_loop import the_main_loop
from mitjax.pkg.monitor.monitor_h import different_multiple

PREFIX = "(PID.TID 0000.0001) "     # PRINT_MESSAGE's prefix of a one-process, one-thread run (eesupp/src/print.F)


# ------------------------------------------------------------------------------------------------------ run directory

def make_rundir(exp_dir, variant, out_dir):
    """`<out_dir>/rundir` with the variant's input files linked as testreport's linkdata links them (MULTI_THREAD=f,
    MPI=0; mitjax/make_rundir.linkdata_plan). out_dir must not exist yet (unique per run, never reused).

    ADVECT lane (M2): a variant whose linkdata brings an executable `prepare_run` (advect_cs/input links
    grid_cs32.face00?.bin from ../../tutorial_held_suarez_cs/input) gets lane A's run directory instead: testreport's
    layout (a mirror of verification/ with the experiment directory, the run directory `run` / `tr_run.<v>`), linkdata,
    then prepare_run with lane A's safety rules (refused if it would remove a file or reports an Error:, the linkdata
    entries checked after it, gunzip shimmed) -- mitjax/make_rundir.main with its run root moved to out_dir (never
    $MJX_REFERENCE); `<out_dir>/rundir` links to that run directory."""
    from mitjax import make_rundir as mr
    from mitjax.config.namelists import link_sources
    exp_dir = Path(exp_dir).resolve()
    out = Path(out_dir)
    dirs, _ = mr.input_dirs(variant)
    if any(os.access(exp_dir / d / "prepare_run", os.X_OK) for d in dirs):
        out.mkdir(parents=True, exist_ok=False)
        rc = mr.main([exp_dir.name, variant, "--run-id", "run", "--runs", str(out / "testreport"),
                      "--exp-dir", str(exp_dir)])
        top = out / "testreport" / exp_dir.name / variant / "run"
        if rc != 0:
            raise RuntimeError(f"make_rundir: lane A's run directory was refused: "
                               f"{(top / 'REFUSED.txt').read_text() if (top / 'REFUSED.txt').exists() else rc}")
        rundir = out / "rundir"
        rundir.symlink_to((top / "rundir").resolve())
        return rundir
    out.mkdir(parents=True, exist_ok=False)
    rundir = out / "rundir"
    rundir.mkdir()
    for name, ldir in link_sources(exp_dir, variant).items():
        os.symlink(exp_dir / ldir / name, rundir / name)
    return rundir


def load_experiment(exp_dir, variant):
    """mitjax.config.params.load of the experiment directory `exp_dir` (anywhere on disk: it holds the code and input
    directories; the MITgcm sources come from MJX_UPSTREAM, docs plan 20261006 Task 2)."""
    from mitjax.config.params import load
    exp_dir = Path(exp_dir).resolve()
    return load(exp_dir.name, variant, exp_dir=exp_dir)


# ------------------------------------------------------------------------------------------------------- the monitor

class MonitorHost:
    """pkg/monitor on the host for one Model: MON_INIT (via mon_init), MON_CALC_ADVCFL, MONITOR. Its configuration
    comes from the run's own parameters (INI_PARMS), cross-checked against the oracle's printout in the tests."""

    def __init__(self, m):
        from mitjax.params_io import RunParams, fortran_default
        from mitjax.pkg.monitor.mon_init import mon_init
        from mitjax.pkg.monitor.monitor_h import MonitorCommon
        self.m = m
        cfg, sz, p = m.cfg, m.cfg.size, m.params
        rp = RunParams(m.exp.run)
        cpp = cfg.cpp
        known = lambda name, header=None: (name in cpp.known) and (     # noqa: E731
            cpp.flag(name, header) if header else bool(getattr(cpp, name)))
        use = dict(cfg.use)
        useMNC = use.get("useMNC", False)
        self.cfg = SimpleNamespace(
            sNx=sz.sNx, sNy=sz.sNy, OLx=sz.OLx, OLy=sz.OLy, nSx=sz.nSx, nSy=sz.nSy, Nr=sz.Nr,
            monitorSelect=p.monitorSelect,
            monitor_stdio=rp.get("data", "PARM03", "monitor_stdio",
                                 default=fortran_default("model/src/set_defaults.F:354", "monitor_stdio", m.exp)),
            usingPCoords=p.usingPCoords, usingSphericalPolarGrid=p.usingSphericalPolarGrid,
            fluidIsAir=p.fluidIsAir, fluidIsWater=p.fluidIsWater, useCoriolis=p.useCoriolis,
            selectCoriMap=m.prm.grid.selectCoriMap, nonHydrostatic=p.nonHydrostatic,
            select_rStar=m.prm.init.select_rStar,
            useCubedSphereExchange=rp.get("eedata", "EEPARMS", "useCubedSphereExchange", default=fortran_default(
                "eesupp/src/eeset_parms.F:106", "useCubedSphereExchange", m.exp)),
            useMNC=useMNC,
            monitor_mnc=(rp.get("data.mnc", "MNC_01", "monitor_mnc", default=fortran_default(
                "pkg/mnc/mnc_readparms.F:101", "monitor_mnc", m.exp)) if useMNC else True),
            useAIM=use.get("useAIM", False),
            ALLOW_MNC=known("ALLOW_MNC"), ALLOW_NONHYDROSTATIC=known("ALLOW_NONHYDROSTATIC"),
            ALLOW_AIM=known("ALLOW_AIM"), NONLIN_FRSURF=known("NONLIN_FRSURF"),
            MONITOR_TEST_HFACZ=known("MONITOR_TEST_HFACZ", "MONITOR_OPTIONS.h"),
            staggerTimeStep=p.staggerTimeStep)
        if self.cfg.useCubedSphereExchange:                                     # lane B: MON_VORT3's cube arm
            self.cfg.cs_vort3_corners = _cs_vort3_corners(getattr(m, "w2", None))
        tp = m.prm.time
        self.params = SimpleNamespace(
            monitorFreq=float(p.monitorFreq), deltaTClock=float(tp.deltaTClock), deltaTMom=float(tp.deltaTMom),
            dTtracerLev=[float(x) for x in tp.dTtracerLev], rUnit2mass=float(p.rUnit2mass),
            startTime=float(tp.startTime), nIter0=int(tp.nIter0), gBaro=float(p.gBaro),
            **({"atm_Cp": float(p.atm_Cp), "rC_Po_kappa": [float(x) for x in np.asarray(p.rC_Po_kappa.data)]}
               if p.fluidIsAir else {}))                                        # lane B: MON_SURFCOR's air terms
        if p.fluidIsAir and p.useCoriolis:     # lane B: MON_KE's angular momentum (mon_ke.F:166-322, mon_init.F:44)
            from mitjax.model.src.ini_global_domain import ini_global_domain_2d
            globalArea = ini_global_domain_2d(cfg=m.cfg, grid=m.grid, ex=m.ex)[1]       # GRID.h globalArea
            self.params.abEps, self.params.exactConserv = float(p.abEps), bool(p.exactConserv)
            self.params.rSphere, self.params.omega = float(m.prm.grid.rSphere), float(m.prm.grid.omega)
            self.params.freeSurfFac, self.params.globalArea = float(m.cg2d_params.freeSurfFac), float(globalArea)
        self.mon = MonitorCommon()
        mon_init(cfg=self.cfg, params=self.params, mon=self.mon)

    def _grid(self, carry=None):
        g, p = self.m.grid, self.m.params
        ns = SimpleNamespace(**{n: getattr(g, n) for n in g.names()})
        if self.m.cfg.cpp.NONLIN_FRSURF:
            # GO lane: under NONLIN_FRSURF GRID.h hFacC/W/S, recip_hFacC/W/S are State (mitjax/model/state.py
            # NLFS_GRID): MONITOR reads the values the step left
            from mitjax.model.state import NLFS_GRID
            if carry is None:
                raise ValueError("MonitorHost: under NONLIN_FRSURF the grid's hFac fields come from the carry")
            for n in NLFS_GRID:
                setattr(ns, n, getattr(carry[0], n))
        ns.rhoFacC, ns.rhoFacF, ns.recip_rhoFacC = p.rhoFacC, p.rhoFacF, p.recip_rhoFacC
        return ns

    @staticmethod
    def _state(carry):
        state, ff, phi0surf = carry[:3]
        ns = SimpleNamespace(**{n: getattr(state, n) for n in state.names() if not isinstance(getattr(state, n),
                                                                                               tuple)})
        for n in ("Qnet", "Qsw", "EmPmR", "fu", "fv"):
            setattr(ns, n, getattr(ff, n))
        ns.phi0surf = phi0surf
        return ns

    def calc_advcfl(self, flow, myIter, carry=None):
        """thermodynamics.F:278-283 MON_CALC_ADVCFL_TILE per tile, :387-390 MON_CALC_ADVCFL_GLOB."""
        from mitjax.pkg.monitor.mon_calc_advcfl import mon_calc_advcfl_glob, mon_calc_advcfl_tile
        grid = self._grid(carry)
        if len(flow) == 6:
            # GO lane: non-staggered NONLIN_FRSURF: THERMODYNAMICS (forward_step.F:733) runs before
            # UPDATE_R_STAR(.TRUE.); the step returns the hFacW, hFacS, recip_hFacC it used with the flow
            flow, (grid.hFacW, grid.hFacS, grid.recip_hFacC) = flow[:3], flow[3:]
        maxCFL = mon_calc_advcfl_tile(self.cfg.Nr, *flow, self.params.dTtracerLev, None, myIter, cfg=self.cfg,
                                      grid=grid)
        mon_calc_advcfl_glob(maxCFL, myIter, mon=self.mon)

    def monitor(self, myTime, myIter, carry):
        """MONITOR( myTime, myIter ): the records it prints (with the PRINT_MESSAGE prefix)."""
        from mitjax.pkg.monitor.monitor import monitor
        n0 = len(self.mon.units.get(self.mon.mon_ioUnit, []))
        monitor(myTime, myIter, cfg=self.cfg, params=self.params, grid=self._grid(carry), state=self._state(carry),
                mon=self.mon)
        return list(self.mon.units.get(self.mon.mon_ioUnit, []))[n0:]


# ---------------------------------------------------------------------------------------------------- the time loop

class SboHost:
    """GO lane: pkg/sbo as DO_THE_MODEL_IO calls it (do_the_model_io.F:178-184: IF (useSBO) SBO_CALC + SBO_OUTPUT),
    on the host at every call whose SBO_OUTPUT prints (myIter = nIter0, or DIFFERENT_MULTIPLE(sbo_monFreq, myTime,
    deltaTClock), sbo_output.F:103-105). SBO_CALC is traced (mitjax/pkg/sbo/sbo_calc.py, jitted once); under
    NONLIN_FRSURF it reads the GRID.h hFacC the step left (the State's, forward_step.nlfs_load). The SBO_global file
    record of SBO_OUTPUT (sbo_output.F:84-100) is output only and not ported. `m`: anything with cfg, exp, fp,
    grid, ex and prm.time (the driver's Model)."""

    def __init__(self, m):
        import jax
        from mitjax.model.src.forward_step import nlfs_load
        from mitjax.pkg.sbo.sbo_calc import sbo_calc
        from mitjax.pkg.sbo.sbo_readparms import sbo_readparms
        self.m = m
        cfg, fp = m.cfg, m.fp
        gp = m.prm.grid
        self.sp = sbo_readparms(m.exp, fp, usingCartesianGrid=gp.usingCartesianGrid,
                                usingCylindricalGrid=gp.usingCylindricalGrid)

        def calc(t, it, grid, state, ff, ex):
            grid, _ = nlfs_load(cfg=cfg, grid=grid, cg2dh=None, state=state)
            return sbo_calc(t, it, cfg=cfg, grid=grid, fp=fp, state=state, ff=ff, ex=ex)
        self._calc = jax.jit(calc)

    def prints(self, myTime, myIter):
        """SBO_OUTPUT's print condition (sbo_output.F:103-105)."""
        tp = self.m.prm.time
        return int(myIter) == int(tp.nIter0) or different_multiple(self.sp.sbo_monFreq, myTime, tp.deltaTClock)

    def records(self, myTime, myIter, carry):
        """SBO_CALC + SBO_OUTPUT of DO_THE_MODEL_IO( myTime, myIter ): the %SBO records it prints."""
        import jax.numpy as jnp
        from mitjax.pkg.sbo.sbo_output import new_stdout, sbo_output
        tp = self.m.prm.time
        state, ff = carry[0], carry[1]
        sbo = self._calc(jnp.float64(myTime), jnp.int32(myIter), self.m.grid, state, ff, self.m.ex)
        out = new_stdout()
        sbo_output(myTime, myIter, sbo, sp=self.sp, nIter0=tp.nIter0, deltaTClock=tp.deltaTClock, stdout=out)
        return list(out.units.get(6, []))


def host_clock(tp, iloop):
    """forward_step.F:807-808 on the host: (myTime, myIter) after step iloop, in float64 as the traced counters."""
    return float(np.float64(tp.startTime) + np.float64(tp.deltaTClock) * np.float64(iloop)), tp.nIter0 + iloop


def chunk_ends(m, nTimeSteps=None):
    """The steps (iloop) after which the host must see the State (module docstring), in order, with the last step
    of the run. Returns [(iloop, cfl, monitor, pickup)]."""
    from mitjax.model.src.do_write_pickup import pickup_due
    tp, p, io = m.prm.time, m.params, m.io
    n = tp.nTimeSteps if nTimeSteps is None else nTimeSteps
    out = []
    for iloop in range(1, n + 1):
        t_start = host_clock(tp, iloop - 1)[0] if iloop > 1 else float(tp.startTime)
        # M3 Task 30: THERMODYNAMICS' myTime is the step's start time (forward_step.F:430, the call at :733) without
        # staggerTimeStep, and the end time with it (the call at :1005 follows the counter update :807-808); wrTime =
        # myTime with staggerTimeStep, myTime + deltaTClock without (thermodynamics.F:132-134)
        wrTime = (host_clock(tp, iloop)[0] if p.staggerTimeStep
                  else t_start + float(tp.deltaTClock))                         # thermodynamics.F:132-134
        cfl = (p.monitorSelect >= 2 and different_multiple(p.monitorFreq, wrTime, tp.deltaTClock)
               and bool(m.cfg.cpp.ALLOW_GENERIC_ADVDIFF))     # lane B: thermodynamics.F:94-416 is compiled out without it
        myTime, myIter = host_clock(tp, iloop)
        mon = p.monitorFreq > 0. and different_multiple(p.monitorFreq, myTime, tp.deltaTClock)
        modelEnd = myTime == tp.endTime or myIter == tp.nEndIter                         # forward_step.F:1172
        pk = pickup_due(modelEnd, myTime, io=io, deltaTClock=tp.deltaTClock)[0]
        if cfl or mon or pk or iloop == n:
            out.append((iloop, bool(cfl), bool(mon), bool(pk)))
    return out


def solver_records(c, s, monitor_step, debugLevel, nsa=False):
    """The printed solver lines of step s (index into the stacked cg2d outputs c). nsa: the solver is CG2D_NSA
    (PTRACERS lane: its own Sum(rhs) line, cg2d_nsa.F:229-230)."""
    from mitjax.model.src.cg2d import cg2d_sum_rhs_message, solve_for_pressure_cg2d_messages
    out = []
    if debugLevel >= debLevZero and nsa:                                                 # cg2d_nsa.F:226-232
        from mitjax.model.src.cg2d_nsa import cg2d_nsa_sum_rhs_message
        out.append(cg2d_nsa_sum_rhs_message(c["sumRHS"][s], c["rhsMax"][s]))
    elif debugLevel >= debLevZero:                                                       # cg2d.F:196-201
        out.append(cg2d_sum_rhs_message(c["sumRHS"][s], c["rhsMax"][s]))
    if monitor_step and debugLevel >= debLevA:                                           # solve_for_pressure.F:332-350
        out += [PREFIX + ln for ln in solve_for_pressure_cg2d_messages(
            c["firstResidual"][s], c["minResidualSq"][s], c["lastResidual"][s], c["numIters"][s], c["nIterMin"][s])]
    return out


def sharded_step(m, sh, carry):
    """Lane APIF (docs plan 20261006 S5): the run's FORWARD_STEP (m.step) for a sharded forward -- jit(shard_map(
    check_vma=True)) over the TileSharding `sh` as the P=N whole-run gates run it (mitjax/tests/go_gate.run_steps_p,
    test_m2accept_pn.py): Arrays and carry placed with sharded_grad.place_model / tile_specs (tiled FArrays split by
    tile, padding tiles copies of tile 1, the exchanger the placed ShardedExchanger), the per-step scalars and the
    step's printed outputs replicated. Returns (step, placed Arrays, placed carry)."""
    import jax
    import jax.numpy as jnp
    from mitjax.drivers.sharded_grad import place_model, tile_specs
    t, it = m.start_counters()
    a_s, c_s = tile_specs(sh, m.arrays), tile_specs(sh, carry)
    c_shape = jax.eval_shape(m.step, m.arrays, carry, jnp.int32(1), t, it)[3]
    step = sh.shard_map(m.step, in_specs=(a_s, c_s, sh.REP, sh.REP, sh.REP),
                        out_specs=(c_s, sh.REP, sh.REP, jax.tree.map(lambda _: sh.REP, c_shape)))
    return step, place_model(sh, m.arrays), place_model(sh, carry)


def forward(m, *, nTimeSteps=None, mds=None, write_pickups=True, chunks=None, keep=None, sharding=None):
    """INITIALISE_VARIA's MONITOR (initialise_varia.F:383) and THE_MAIN_LOOP over nTimeSteps (default the run's).

    Returns SimpleNamespace(carry, myTime, myIter, records, pickups, kept): `records` the STDOUT records, `pickups`
    the files written, `kept` {iloop: carry} for the steps in `keep` (extra chunk ends, for the gates). `chunks`: extra
    chunk ends (tests compare chunkings: one step per chunk vs the host events only).
    `sharding` (lane APIF): a TileSharding (mitjax/eesupp/shard.py); the time loop then runs `sharded_step` on the
    placed carry, and every host event (monitor, pickups, kept carries, COST_FINAL, the returned carry) reads the
    carry gathered and unpadded (TileSharding.unpad_tree). None: the single-device run, unchanged."""
    import jax
    from mitjax.model.src.do_write_pickup import Restart, do_write_pickup
    from mitjax.pkg.mdsio.mdsio_write_field import MdsContext
    tp, p = m.prm.time, m.params
    n = tp.nTimeSteps if nTimeSteps is None else nTimeSteps
    pre = getattr(m.arrays, "pkc", {}).get("forcing")
    if pre is not None and n > pre.irec.shape[0] - 1:                   # R5 arm: the preload covers the run's steps
        raise ValueError(f"forward: {n} steps, but the periodic forcing is preloaded for {pre.irec.shape[0] - 1}")
    mh = MonitorHost(m)
    if mds is None:
        mds = MdsContext(m.rundir, m.cfg.size, exch2=bool(m.cfg.cpp.ALLOW_EXCH2), w2=getattr(m, "w2", None),
                         useSingleCpuIO=m.io.useSingleCpuIO,
                         mdsioLocalDir=m.io.mdsioLocalDir, the_run_name=m.io.the_run_name,
                         useCal=_mds_cal(m))                     # lane M4COL: MDS_WRITE_META's timeStepDate
    restart = Restart()
    carry = m.initial_carry()
    myTime, myIter = m.start_counters()
    records = []
    for name, it_, adj in getattr(m, "init_host", ()):     # lane B: INITIALISE_VARIA's CALC_SURF_DR (:316, :343)
        if name == "CALC_SURF_DR":
            records += _surf_adjust_lines(adj, it_)
    cpu, hc = _host_copy(carry)                         # GO lane: the host work below reads one host-side copy
    with jax.default_device(cpu):
        records += list(mh.monitor(float(tp.startTime), int(tp.nIter0), hc))             # initialise_varia.F:383
        sh = SboHost(m) if dict(m.cfg.use).get("useSBO", False) else None               # GO lane (useSBO)
        if sh is not None and sh.prints(float(tp.startTime), int(tp.nIter0)):           # initialise_varia.F:390
            records += sh.records(float(tp.startTime), int(tp.nIter0), hc)
        records += seaice_monitor_records(mh, float(tp.startTime), int(tp.nIter0), hc)     # lane M4COL (:192-196)
        records += ptracers_monitor_records(mh, float(tp.startTime), int(tp.nIter0), hc)   # PTRACERS lane (:213)
    events = {e[0]: e for e in chunk_ends(m, n)}
    sbo_ends = set() if sh is None else {k for k in range(1, n + 1) if sh.prints(*host_clock(tp, k))}
    si_ends = seaice_monitor_steps(m, n)                     # lane M4OFF: SEAICE_MONITOR's own schedule
    ends = sorted(set(events) | set(keep or ()) | set(chunks or ()) | sbo_ends | si_ends)
    pickups, kept, chunk_lengths = [], {}, []
    stderr, numbWrite = [], 0                     # GO lane: STDERR lines; CALC_R_STAR's SAVEd numbWrite (DATA 0)
    for name, it_, cnt in getattr(m, "init_host", ()):                  # INITIALISE_VARIA's CALC_R_STAR calls
        if name == "CALC_R_STAR":                                       # (CALC_SURF_DR's: above; lane B)
            numbWrite = _rstar_host(cnt, it_, numbWrite, m.cfg, stderr)
    iloop = 1
    step, arrays, gather = m.step, m.arrays, (lambda c: c)
    if sharding is not None:                            # lane APIF: the sharded forward (sharded_step)
        step, arrays, carry = sharded_step(m, sharding, carry)
        gather = sharding.unpad_tree
    for end in ends:
        nst = end - iloop + 1
        chunk_lengths.append(nst)
        carry, myTime, myIter, cg = the_main_loop(step, arrays, carry, myTime, myIter, nTimeSteps=nst,
                                                  iloop0=iloop)
        t_host, it_host = host_clock(tp, end)
        if float(myTime) != t_host or int(myIter) != it_host:
            raise AssertionError(f"clock: traced ({float(myTime)!r}, {int(myIter)}) != host ({t_host!r}, {it_host})")
        _, cfl, mon, pk = events.get(end, (end, False, False, False))
        if cg is not None:      # ADVECT lane: without momStepping SOLVE_FOR_PRESSURE does not run (forward_step.F:897)
            c = {k: np.asarray(v) for k, v in cg.items()}
            for s in range(nst):
                if "exfmon_hflux" in c:                 # lane M4COL: EXF_GETFORCING's EXF_MONITOR (step start)
                    with jax.default_device(cpu):
                        records += exf_monitor_records(mh, iloop + s, {k[7:]: v[s] for k, v in c.items()
                                                                       if k.startswith("exfmon_")})
                if "lsr_printFlex" in c:                # lane M4OFF session 3: SEAICE_LSR (in SEAICE_MODEL)
                    from mitjax.pkg.seaice.seaice_lsr import lsr_stderr_lines, lsr_stdout_lines
                    lo = {k[4:]: v[s] for k, v in c.items() if k.startswith("lsr_")}
                    records += lsr_stdout_lines(lo)
                    stderr += [PREFIX + ln for ln in lsr_stderr_lines(lo, iloop + s - 1 + int(tp.nIter0))]
                if "sumRHS" in c:
                    records += solver_records(c, s, mon and s == nst - 1, p.debugLevel,
                                              nsa=bool(m.cfg.cpp.ALLOW_CG2D_NSA and m.cg2d_params.useNSACGSolver))
                if "rstar_icntc1" in c:                 # GO lane: CALC_R_STAR's host part (forward_step.F:949)
                    numbWrite = _rstar_host({k[6:]: v[s] for k, v in c.items() if k.startswith("rstar_")},
                                            iloop + s + int(tp.nIter0), numbWrite, m.cfg, stderr)
                if "opps_ntimeOver" in c:               # vermix lane: OPPS_CALC's static time-loop bound (STOP)
                    from mitjax.pkg.opps.opps_calc import opps_calc_host
                    from mitjax.pkg.opps.opps_h import NTIME_MAX
                    opps_calc_host({"ntimeOver": c["opps_ntimeOver"][s]}, iloop + s + int(tp.nIter0), NTIME_MAX)
                if "rstar_surf_nb" in c:                # lane B / GOADK: CALC_SURF_DR's diagnostic (forward_step.F:957)
                    records += _surf_adjust_lines((c["rstar_surf_nb"][s], c["rstar_surf_vol"][s]),
                                                  iloop + s + int(tp.nIter0))
        if cfl or mon or end in sbo_ends or end in si_ends or (pk and write_pickups):
            _, hc = _host_copy(gather(carry))                   # GO lane: one host-side copy of the carry per host event
        with jax.default_device(cpu):
            if cfl:
                mh.calc_advcfl(hc[3], it_host - 1, hc)             # THERMODYNAMICS' myIter: the step's start
            if mon:
                records += mh.monitor(t_host, it_host, hc)
            if end in sbo_ends:                                # GO lane: forward_step.F:1182 DO_THE_MODEL_IO (SBO)
                records += sh.records(t_host, it_host, hc)
            if mon or end in si_ends:   # lane M4COL: DO_THE_MODEL_IO's SEAICE_OUTPUT (do_the_model_io.F:192-196);
                records += seaice_monitor_records(mh, t_host, it_host, hc)    # it prints on its own test (:55-56)
            if mon:                   # PTRACERS lane: DO_THE_MODEL_IO's PTRACERS_OUTPUT (do_the_model_io.F:213)
                records += ptracers_monitor_records(mh, t_host, it_host, hc)
            if pk and write_pickups:
                modelEnd = t_host == tp.endTime or it_host == tp.nEndIter
                lines, fn = do_write_pickup(modelEnd, t_host, it_host, cfg=m.cfg, params=p, ip=m.prm.init,
                                            io=m.io, state=hc[0], mds=mds, restart=restart,
                                            deltaTClock=tp.deltaTClock, exp=m.exp,       # exp: ADVECT lane
                                            pk=hc[4] if len(hc) > 4 else None,           # vermix lane: GGL90
                                            cal=getattr(m, "cal", None),                 # lane M4COL: useCAL
                                            sp=getattr(m, "sp", None))                   # lane M4COL: sea ice
                records += [PREFIX + ln for ln in lines]
                if fn:
                    pickups.append(fn)
        if keep and end in keep:
            kept[end] = gather(carry)
        iloop = end + 1
    if sharding is not None:
        carry = jax.device_put(gather(carry), cpu)      # lane APIF: the gathered carry, as _host_copy places it
    cost = None
    if m.cfg.cpp.ALLOW_COST and len(carry) > 4:            # R5 arm: COST_FINAL after the loop (the_main_loop.F:767-775)
        from mitjax.pkg.cost.cost_final import cost_final_lines
        pk = carry[4]
        early = pk["cost"].fc
        pout = {}                                       # lane M4ADCOL: the package finals' printed values
        cost, loc = m.cost_final(pk["cost"], pk.get("genarr"), state=carry[0],   # GO lane: COST_TEST reads theta
                                 out=pout, ecco=pk.get("ecco"))                  # lane M4ADCOL session 3: pkg/ecco
        if "ecco_files" in pout:        # lane M4ADCOL session 3: the bar files (COST_AVERAGESFIELDS after the loop,
            # the_main_loop.F:737-742) and the misfit files (COST_GENCOST_ALL, cost_driver.F:51), host writes
            from mitjax.pkg.ecco.cost_averagesfields import write_barfiles
            from mitjax.pkg.ecco.cost_gencost_all import write_misfits
            ef = pout.pop("ecco_files")
            write_barfiles(ef, m.ecco.ep, m.ecco.table, eccoiter=m.ecco.fixed["eccoiter"], mds=mds)
            write_misfits(ef["mis"], m.ecco.ep, eccoiter=m.ecco.fixed["eccoiter"],
                          writeBinaryPrec=m.ecco.writeBinaryPrec, globalFile=m.io.globalFiles, mds=mds,
                          setup=m.ecco.setup)
            if ef.get("offs"):          # lane M4ADLAB session 4: ECCO_OFFSET's prints (COST_DRIVER, before COST_FINAL)
                from mitjax.pkg.ecco.cost_gencost_all import ecco_offset_lines
                records += [PREFIX + ln for ln in ecco_offset_lines(ef["offs"], m.ecco.ep)]
        if pout:                                        # lane M4ADCOL: cost_final.F:86-122 and :228-242
            from mitjax.pkg.cost.cost_copy_file import package_final_records
            records += package_final_records(pout, rundir=m.rundir, prefix=PREFIX)
        for ln in cost_final_lines(cost, loc, cfg=m.cfg, params=m.cost_fixed["params"], early_fc=early,
                                   files=bool(pout), rundir=m.rundir):
            # early / local / global fc go through PRINT_MESSAGE (cost_final.F:155, :212, :244), the per-tile
            # objf lines are WRITE(ioUnit,...) (:164-184); the costfunction.0000 file (:229-236) is not written
            records.append(PREFIX + ln if ln.lstrip().startswith(("early fc", "local fc", "global fc", "Writing ",
                                                                     "Reading ")) else ln)
    return SimpleNamespace(carry=carry, myTime=myTime, myIter=myIter, records=records, pickups=pickups, kept=kept,
                           chunk_lengths=chunk_lengths, stderr=stderr, cost=cost)


def _mds_cal(m):
    """Lane M4COL: MDS_WRITE_META's ALLOW_CAL context (mdsio_write_meta.F:177-199: useCal, baseTime, deltaTClock,
    pkg/cal's common block), or False without useCAL."""
    if not dict(m.cfg.use).get("useCAL", False):
        return False
    return SimpleNamespace(cal=m.cal, baseTime=float(m.prm.time.baseTime), deltaTClock=float(m.prm.time.deltaTClock))


def _exf_mon_cfg(mh):
    """MonitorHost's monitor configuration with the CPP flags EXF_MONITOR / SEAICE_MONITOR test (lane M4COL)."""
    m = mh.m
    cpp = m.cfg.cpp
    c = SimpleNamespace(**vars(mh.cfg))
    c.exf_flags = {o: bool(cpp.flag(o, "EXF_OPTIONS.h")) for o in (
        "ALLOW_BULKFORMULAE", "ALLOW_ATM_TEMP", "ALLOW_DOWNWARD_RADIATION", "ALLOW_RUNOFF", "ALLOW_SALTFLX",
        "ALLOW_CLIMSST_RELAXATION", "ALLOW_CLIMSSS_RELAXATION", "EXF_SEAICE_FRACTION", "ALLOW_RUNOFTEMP",
        "ALLOW_CLIMSTRESS_RELAXATION")}
    c.exf_flags["ALLOW_BLING"] = bool(cpp.flag("ALLOW_BLING"))
    c.SHORTWAVE_HEATING = bool(cpp.flag("SHORTWAVE_HEATING"))
    c.ATMOSPHERIC_LOADING = bool(cpp.flag("ATMOSPHERIC_LOADING"))
    c.seaice_flags = {o: bool(cpp.flag(o, "SEAICE_OPTIONS.h")) for o in (
        "ALLOW_SITRACER", "SEAICE_CGRID", "SEAICE_BGRID_DYNAMICS", "SEAICE_VARIABLE_SALINITY")}
    return c


def _pkg_monfreq(m, file, group, name, cite):
    """A package monitor frequency: the namelist's value, else its default `= monitorFreq` (the cited line)."""
    from mitjax.params_io import RunParams
    rp = RunParams(m.exp.run)
    return float(rp.get(file, group, name)) if rp.has(file, group, name) else float(m.params.monitorFreq)   # cite


def exf_monitor_records(mh, iloop, fields):
    """Lane M4COL: EXF_MONITOR (exf_getforcing.F:380, inside LOAD_FIELDS_DRIVER at the start of step `iloop`: myTime
    = startTime + deltaTClock*(iloop-1), myIter = nIter0 + iloop-1) on the host from the step's EXF_FIELDS.h
    snapshot `fields` ({name: [tile, j, i]}): the records it prints. exf_monFreq: data.exf's, else monitorFreq
    (exf_readparms.F:304)."""
    from mitjax.farray import FArray
    from mitjax.pkg.exf.exf_monitor import exf_monitor
    m = mh.m
    tp, sz = m.prm.time, m.cfg.size
    myTime, myIter = host_clock(tp, iloop - 1)
    exf_monFreq = _pkg_monfreq(m, "data.exf", "EXF_NML_01", "exf_monFreq", "exf_readparms.F:304")
    from mitjax.eesupp.different_multiple import different_multiple
    if not different_multiple(exf_monFreq, myTime, float(tp.deltaTClock)):   # exf_monitor.F:64-65 (lane M4CS32ICE:
        return []         # nothing printed; under NONLIN_FRSURF the monitor grid would need the step-start carry)
    f = {n: FArray(np.asarray(v), n, i=(1 - sz.OLx, sz.sNx + sz.OLx), j=(1 - sz.OLy, sz.sNy + sz.OLy))
         for n, v in fields.items()}
    n0 = len(mh.mon.units.get(mh.mon.mon_ioUnit, []))
    exf_monitor(myTime, myIter, f, cfg=_exf_mon_cfg(mh), exf=m.exfp, exf_monFreq=exf_monFreq,
                deltaTClock=float(tp.deltaTClock), grid=mh._grid(), mon=mh.mon)
    return list(mh.mon.units.get(mh.mon.mon_ioUnit, []))[n0:]


def seaice_monitor_steps(m, n):
    """Lane M4OFF: the steps (iloop) at whose end SEAICE_MONITOR prints (SEAICE_OUTPUT is called by DO_THE_MODEL_IO
    every step, forward_step.F:1182, do_the_model_io.F:192-196; SEAICE_MONITOR prints when
    DIFFERENT_MULTIPLE(SEAICE_monFreq, myTime, deltaTClock), seaice_monitor.F:55-56): the driver stops a chunk there.
    Empty without useSEAICE or with SEAICE_monFreq <= 0."""
    if not (m.cfg.cpp.flag("ALLOW_SEAICE") and dict(m.cfg.use).get("useSEAICE", False)):
        return set()
    tp = m.prm.time
    freq = _pkg_monfreq(m, "data.seaice", "SEAICE_PARM01", "SEAICE_monFreq", "seaice_readparms.F:559")
    return {k for k in range(1, n + 1) if different_multiple(freq, host_clock(tp, k)[0], tp.deltaTClock)}


def seaice_monitor_records(mh, myTime, myIter, carry):
    """Lane M4COL: SEAICE_OUTPUT's SEAICE_MONITOR (seaice_output.F:171; DO_THE_MODEL_IO do_the_model_io.F:192-196) on
    the host: the records it prints. The driver calls it at every step where MONITOR or SEAICE_MONITOR prints (lane
    M4OFF: seaice_monitor_steps, SEAICE_monFreq's own schedule); SEAICE_MONITOR's own test decides (:55-56)."""
    m = mh.m
    if not (m.cfg.cpp.flag("ALLOW_SEAICE") and dict(m.cfg.use).get("useSEAICE", False)):
        return []
    from mitjax.pkg.seaice.seaice_monitor import seaice_monitor
    freq = _pkg_monfreq(m, "data.seaice", "SEAICE_PARM01", "SEAICE_monFreq", "seaice_readparms.F:559")
    n0 = len(mh.mon.units.get(mh.mon.mon_ioUnit, []))
    seaice_monitor(myTime, myIter, carry[4]["seaice"], cfg=_exf_mon_cfg(mh), sp=m.sp, SEAICE_monFreq=freq,
                   deltaTClock=float(m.prm.time.deltaTClock), grid=mh._grid(carry), mon=mh.mon,
                   useThSIce=bool(dict(m.cfg.use).get("useThSIce", False)))
    return list(mh.mon.units.get(mh.mon.mon_ioUnit, []))[n0:]


def _host_copy(carry):
    """GO lane (M1 acceptance): (the host CPU device, ONE copy of the carry on it) for a host event. MONITOR, SBO,
    MON_CALC_ADVCFL, PTRACERS_MONITOR and WRITE_PICKUP read that copy under jax.default_device(cpu), so their eager
    and jitted host computations run on the CPU backend whichever device ran the scan (a GPU carry is copied once,
    not field by field). On a CPU run the carry is already on that device: device_put returns it, the same values."""
    import jax
    cpu = jax.devices("cpu")[0]
    return cpu, jax.device_put(carry, cpu)


def ptracers_monitor_records(mh, myTime, myIter, carry):
    """PTRACERS lane: PTRACERS_MONITOR (pkg/ptracers/ptracers_monitor.F, called by PTRACERS_OUTPUT from
    DO_THE_MODEL_IO, forward_step.F:1182, after MONITOR; initially after INITIALISE_VARIA's MONITOR) on the host, with
    MonitorHost's MONITOR.h and namespaces: the records it prints. Its schedule is MONITOR's (the driver stops a
    chunk at the MONITOR steps): a PTRACERS_monitorFreq different from monitorFreq raises."""
    m = mh.m
    if not (m.cfg.cpp.ALLOW_PTRACERS and m.cfg.use_flag("usePTRACERS")):
        return []
    from mitjax.pkg.ptracers.ptracers_fields_h import ptf_of_state
    from mitjax.pkg.ptracers.ptracers_monitor import ptracers_monitor
    p = m.params
    if p.PTRACERS_monitorFreq != p.monitorFreq:
        raise NotImplementedError("run: PTRACERS_monitorFreq /= monitorFreq (a schedule of its own) is not wired")
    n0 = len(mh.mon.units.get(mh.mon.mon_ioUnit, []))
    ptracers_monitor(myTime, myIter, cfg=mh.cfg, params=mh.params, grid=mh._grid(carry), ptr=p,
                     ptf=ptf_of_state(carry[0], p), mon=mh.mon)
    return list(mh.mon.units.get(mh.mon.mon_ioUnit, []))[n0:]


def _cs_vort3_corners(w2):
    """Lane B (Task 25): MON_VORT3's corner switches per tile (W2_myTileList order, bi fastest) as mon_vort3.F:157-177
    sets them under ALLOW_EXCH2: SW = isWedge .AND. isSedge; SE = isEedge .AND. isSedge .AND. exch2_myFace = 2;
    NW = isWedge .AND. isNedge .AND. exch2_myFace = 1 (NE = .FALSE., :177). [tile, 3] bool, None without W2."""
    if w2 is None:
        return None
    from mitjax.eesupp.fill_cs_corner_tr_rl import cs_corner_flags
    fl = cs_corner_flags(w2)                                    # (SW, SE, NW, NE), fill_cs_corner_tr_rl.F:75-82
    sz = w2.size
    face = np.array([w2.exch2_myFace[w2.W2_myTileList[bi, bj]] for bj in range(1, sz.nSy + 1)
                     for bi in range(1, sz.nSx + 1)])
    return np.stack([fl[:, 0], fl[:, 1] & (face == 2), fl[:, 2] & (face == 1)], axis=1)   # :175-176


def _surf_adjust_lines(adjust, myIter):
    """Lane B (Task 25): CALC_SURF_DR's global diagnostic (calc_surf_dr.F:200-212) on the host, from its `adjust` =
    (per-tile numbers of clipped interior points [nTiles], their rA*(Rmin_surf - rSurftmp) on the C loop range, 0.
    elsewhere [nTiles, j, i]; every real tile in tile order, calc_surf_dr's ex.all_tiles): adjust_nb_pt the sum of
    the per-tile counts (whole numbers: exact in any order), adjust_volum summed in the Fortran loop order (tile, j,
    i; a 0. term leaves a sum unchanged), _GLOBAL_SUM_RL of one process; the line WRITE(standardMessageUnit,'(2(A,I10),1PE16.8)') (a direct WRITE: no PID.TID prefix) when
    adjust_nb_pt >= 1. `myIter`: CALC_SURF_DR's (in FORWARD_STEP the counter after :807-808)."""
    from mitjax.io.fortran_format import fortran_write
    nb, vals = adjust
    adjust_nb_pt = float(np.sum(np.asarray(nb, dtype=np.int64)))                # :200 _GLOBAL_SUM_RL
    if not adjust_nb_pt >= 1.:                                                  # :202
        return []
    adjust_volum = 0.
    for x in np.asarray(vals, dtype=np.float64).ravel():                       # [tile, j, i], row-major
        adjust_volum = adjust_volum + float(x)                                  # :136-137
    nTmp = int(round(adjust_nb_pt))                                             # :205 NINT
    return [fortran_write("(2(A,I10),1PE16.8)", " SURF_ADJUSTMENT: Iter=", int(myIter), " Nb_pts,Vol=", nTmp,
                          adjust_volum)]                                        # :206-208


def _rstar_host(counters, myIter, numbWrite, cfg, stderr):
    """GO lane: CALC_R_STAR's WRITE / STOP part (calc_r_star.F:201-253, calc_r_star_host: raises RuntimeError
    'ABNORMAL END: S/R CALC_R_STAR' where the Fortran STOPs) and the accepted guard's zero-denominator report
    (calc_r_star.F:306-311, Nikolay 2026-10-01) of one call, on the host; lines appended to `stderr`."""
    from mitjax.model.src.calc_r_star import calc_r_star_host, calc_r_star_zero_report
    counters = {k: np.asarray(v) for k, v in counters.items()}
    lines, numbWrite = calc_r_star_host(counters, int(myIter), numbWrite, cfg=cfg)
    _, zlines = calc_r_star_zero_report(counters, int(myIter))
    stderr += lines + zlines
    return numbWrite


def run(experiment_dir, variant, out_dir):
    """The CLI: link the run directory, set the model up, run, write `<out>/rundir/output.txt` (our STDOUT records,
    the lines mitjax/testreport_jax.py compares). Returns the path of output.txt."""
    from mitjax.xla_flags import set_gate_xla_flags
    set_gate_xla_flags()
    exp = load_experiment(experiment_dir, variant)
    rundir = make_rundir(experiment_dir, variant, out_dir)
    m = Model(exp, rundir)
    res = forward(m)
    out = Path(rundir) / "output.txt"
    with open(out, "x") as fh:
        fh.write("\n".join(res.records) + "\n")
    return out
