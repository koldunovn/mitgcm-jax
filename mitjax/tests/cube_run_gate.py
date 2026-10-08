"""Helpers of the cube dynamics gates (plan Task 25, lane B; not a test file): the production set-up
(mitjax/drivers/model.Model: INI_PARMS, INITIALISE_FIXED with the cube exchanger of load_cube_maps and the
curvilinear grid, INITIALISE_VARIA) on the run directory of lane A's dumps-on run, FORWARD_STEP under jit at the gate
XLA flags with its substep `probe`, and the comparison with the oracle's dump stages on every point of every tile
(halos included, bit patterns).
"""

import importlib.util
from functools import lru_cache
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from mitjax import paths
from mitjax.tests import r1_gate as rg


def _rr():
    spec = importlib.util.spec_from_file_location("_mjx_reference_runs", paths.REPO / "reference" / "reference_runs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_top(exp, inp, kind="jdon"):
    rr = _rr()
    runs = [r for r in rr.RUNS if (r["exp"], r["input"], r["kind"]) == (exp, inp, kind)]
    if len(runs) != 1:
        raise FileNotFoundError(f"{exp}/{inp}: {len(runs)} registered runs of kind {kind}")
    return rr.run_top(runs[0])


class CubeRun:
    """drivers.Model of a variant on the oracle's run directory, its oracle dumps, and a probing step function."""

    def __init__(self, exp, inp):
        from mitjax.config.params import load
        from mitjax.drivers.model import Model
        from mitjax.io.dump import DumpSet
        self.exp, self.inp = exp, inp
        self.top = run_top(exp, inp)
        self.e = load(exp, inp)
        self.m = Model(self.e, self.top / "rundir")
        self.ds = DumpSet(self.top / "dumps")
        self.its = self.ds.iterations()

    def step_fn(self, probe_stages=()):
        """jit(step)(grid, params, eos, cg2dh, cg2d_params, ex, state, ff, phi0surf, iloop, myTime, myIter) ->
        (state, ff, phi0surf, myTime, myIter, out, probes) (as r1_gate.Model.step_fn)."""
        from mitjax.model.src.forward_step import forward_step
        cfg, fp = self.m.cfg, self.m.fp

        pks = self.m.pks

        def step(grid, params, eos, cg2dh, cg2d_params, ex, pkc, state, ff, phi0surf, iloop, myTime, myIter):
            probes = {}

            def probe(stage, values):                                       # string keys: a jit output
                name = stage[0] if isinstance(stage, tuple) else stage
                if name in probe_stages:
                    probes[f"{stage[0]}|{stage[1]}" if isinstance(stage, tuple) else stage] = values
            out = forward_step(iloop, myTime, myIter, cfg=cfg, grid=grid, params=params, fp=fp, eos=eos,
                               cg2dh=cg2dh, cg2d_params=cg2d_params, state=state, ff=ff, phi0surf=phi0surf, ex=ex,
                               probe=probe, pkc=pkc, pks=pks)       # pkc, pks: the Model's package inputs (visc)
            return out + (probes,)
        return jax.jit(step)

    def run_step(self, fn, state, ff, phi0surf, iloop, myTime, myIter):
        m = self.m
        return fn(m.grid, m.params, m.eos, m.cg2dh, m.cg2d_params, m.ex, m.arrays.pkc, state, ff, phi0surf,
                  jnp.int32(iloop), jnp.float64(myTime), jnp.int32(myIter))

    def stages(self, it):
        return self.ds.stages(it)


@lru_cache(maxsize=None)
def cube_run(exp, inp):
    return CubeRun(exp, inp)


def stage_values(stage, v):
    """{dump name: array} of one probe: S04's (State, FFields, phi0surf) flattened (r2_gate.stage_values); the
    per-level stages (D00a_phi_hyd, D00b_mom_fluxform; key (stage, k)) renamed `<name>_k<kkk>`, a 3-D array cut to
    level k."""
    from mitjax.tests import r2_gate as r2
    if isinstance(stage, tuple):
        st, k = stage
        out = {}
        for n, a in v.items():
            d = a.data if hasattr(a, "data") else a
            out[f"{n}_k{k:03d}"] = SimpleNamespace(data=d[:, k - 1] if d.ndim == 4 else d)
        return out
    return r2.stage_values(stage, v)


def compare_stage(r, it, stage, values):
    name = stage[0] if isinstance(stage, tuple) else stage
    return rg.compare_stage(r.ds, it, name, stage_values(stage, values))


def bad(res):
    """{field: compare tuple} of the fields that differ in bits or are non-finite."""
    return {n: v for n, v in res.items() if v[0] == "shape" or v[2] or v[3]}


def run_steps(r, n, compare=True):
    """n steps from the initial carry with every dumped stage probed; returns {(it, stage): bad fields} and the
    final carry."""
    m = r.m
    state, ff, phi0, _ = m.initial_carry()
    t, it = m.start_counters()
    out, ncmp = {}, {}
    stages = tuple(sorted({s for i in r.its for s in r.stages(i)}))
    fn = r.step_fn(stages)
    for k in range(n):
        itn = int(it)
        state, ff, phi0, t, it, _, probes = r.run_step(fn, state, ff, phi0, k + 1, t, it)
        if compare and itn in r.its:
            for key, vals in probes.items():
                st = (key.split("|")[0], int(key.split("|")[1])) if "|" in key else key
                name = st[0] if isinstance(st, tuple) else st
                if name in r.stages(itn):
                    res = compare_stage(r, itn, st, vals)
                    out[(itn, name)] = {**out.get((itn, name), {}), **bad(res)}
                    ncmp[(itn, name)] = ncmp.get((itn, name), 0) + len(res)
    run_steps.ncompared = ncmp
    return out, (state, ff, phi0, t, it)


def front_and_phi_hyd(r, until="S05_thermodynamics_sync"):
    """Step 1 of FORWARD_STEP from the initial carry up to the dump stage `until` (forward_step's gate stop), every
    dumped stage probed, then DYNAMICS' level loop of CALC_PHI_HYD on the State it returns, as mitjax/model/src/
    dynamics.py runs it (dynamics.F:191-192 iMin..jMax, :314-315 phiHydF = phiHydC = 0 on every point, dPhiHydX/Y
    NaN locals, :422 DO k, :482-486) with its D00a_phi_hyd probe -- for runs whose DYNAMICS stops later (solid-body:
    MOM_VECINV is not wired into DYNAMICS). Returns ({(it, stage): bad fields}, {(it, stage): fields compared}) for
    the probed stages, the D00a_phi_hyd levels and S06_dynamics' totPhiHyd / phiHydLow (written in DYNAMICS only by
    CALC_PHI_HYD's DIAGS_PHI_HYD / DIAGS_PHI_RLOW)."""
    from mitjax.farray import loop_i, loop_j
    from mitjax.model.src.calc_phi_hyd import calc_phi_hyd
    from mitjax.model.src.forward_step import forward_step
    m = r.m
    cfg, fp = m.cfg, m.fp
    it0 = r.its[0]
    stages = tuple(r.stages(it0))

    def step(grid, params, eos, cg2dh, cg2d_params, ex, pkc, state, ff, phi0surf, iloop, myTime, myIter):
        probes = {}

        def probe(stage, values):                                               # string keys (a jit output)
            name = stage[0] if isinstance(stage, tuple) else stage
            if name in stages:
                probes[f"{stage[0]}|{stage[1]}" if isinstance(stage, tuple) else stage] = values
        state, ff, phi0surf, myTime, myIter, _ = forward_step(
            iloop, myTime, myIter, cfg=cfg, grid=grid, params=params, fp=fp, eos=eos, cg2dh=cg2dh,
            cg2d_params=cg2d_params, state=state, ff=ff, phi0surf=phi0surf, ex=ex, probe=probe, until=until,
            pkc=pkc, pks=m.pks)
        sz = cfg.size
        iMin, iMax, jMin, jMax = 0, sz.sNx+1, 0, sz.sNy+1                       # dynamics.F:191-192
        z2 = state.etaN.local("z2")
        jA, iA = loop_j(1-sz.OLy, sz.sNy+sz.OLy), loop_i(1-sz.OLx, sz.sNx+sz.OLx)
        phiHydF = z2.at[iA, jA].set(0.)                                         # :314
        phiHydC = z2.at[iA, jA].set(0.)                                         # :315
        dPhiHydX, dPhiHydY = z2.local("dPhiHydX"), z2.local("dPhiHydY")
        for k in range(1, sz.Nr+1):                                             # :422
            state, phiHydF, phiHydC, dPhiHydX, dPhiHydY = calc_phi_hyd(        # :482-486
                iMin, iMax, jMin, jMax, k, phiHydF, phiHydC, dPhiHydX, dPhiHydY, myTime, myIter,
                cfg=cfg, grid=grid, params=params, state=state, phi0surf=phi0surf)
            probe(("D00a_phi_hyd", k), dict(dPhiHydX=dPhiHydX, dPhiHydY=dPhiHydY, phiHydC=phiHydC,
                                            phiHydF=phiHydF))
        return dict(totPhiHyd=state.totPhiHyd, phiHydLow=state.phiHydLow), probes

    state, ff, phi0, _ = m.initial_carry()
    t, it = m.start_counters()
    s06, probes = r.run_step(jax.jit(step), state, ff, phi0, 1, t, it)
    out, ncmp = {}, {}
    for key, vals in list(probes.items()) + [("S06_dynamics", s06)]:
        st = (key.split("|")[0], int(key.split("|")[1])) if "|" in key else key
        name = st[0] if isinstance(st, tuple) else st
        res = compare_stage(r, it0, st, vals)
        out[(it0, name)] = {**out.get((it0, name), {}), **bad(res)}
        ncmp[(it0, name)] = ncmp.get((it0, name), 0) + len(res)
    return out, ncmp


def phiref_bad(r):
    """SET_REF_STATE's ATMOSPHERIC phiRef (set_ref_state.F:308-348, integr_GeoPot /= 1: the finite-difference form)
    rebuilt on the host from Params' Exner factors rF_Po_kappa / rC_Po_kappa (glibc pow, ini_parms._atm_traced) vs
    the oracle's PHrefF = phiRef(2k-1), PHrefC = phiRef(2k) (write_grid.F:148-156, float64 MDS vectors in the jdon run
    directory), bit for bit: the factors CALC_PHI_HYD's ddPIm / ddPIp use, checked through the oracle's own
    arithmetic of the same expressions. Returns {file: indices that differ}."""
    p = r.m.params
    if p.integr_GeoPot == 1:
        raise NotImplementedError("phiref_bad: the integr_GeoPot = 1 form (rHalf) is not rebuilt")
    if p.select_rStar >= 1 or p.selectSigmaCoord >= 1:
        raise NotImplementedError("phiref_bad: tLoc = thetaConst is not rebuilt")
    Nr = r.e.cfg.size.Nr
    atm_Cp = float(p.atm_Cp)
    pKF = [None] + [float(x) for x in np.asarray(p.rF_Po_kappa.data)]
    pKC = [None] + [float(x) for x in np.asarray(p.rC_Po_kappa.data)]
    tLoc = [None] + [float(x) for x in np.asarray(p.tRef.data)]                 # :316-318
    phiRef = [None] * (2*Nr + 2)
    phiRef[1] = r.m.prm.grid.seaLev_Z*float(p.gravity)                          # :308
    k = 1
    ddPI = atm_Cp*(pKF[k] - pKC[k])                                             # :334-335
    phiRef[2*k] = phiRef[1] + ddPI*tLoc[k]                                      # :336
    for k in range(1, Nr):                                                      # :337-343
        ddPI = atm_Cp*(pKC[k] - pKC[k+1])
        phiRef[2*k+1] = phiRef[2*k] + ddPI*0.5*tLoc[k]
        phiRef[2*k+2] = phiRef[2*k] + ddPI*0.5*(tLoc[k]+tLoc[k+1])
    k = Nr
    ddPI = atm_Cp*(pKC[k] - pKF[k+1])                                           # :345-346
    phiRef[2*k+1] = phiRef[2*k] + ddPI*tLoc[k]                                  # :347
    d = r.top / "rundir"
    want = {"PHrefF": np.fromfile(d / "PHrefF.data", ">f8"), "PHrefC": np.fromfile(d / "PHrefC.data", ">f8")}
    got = {"PHrefF": np.array([phiRef[2*k-1] for k in range(1, Nr+2)]), "PHrefC": np.array([phiRef[2*k] for k in
                                                                                             range(1, Nr+1)])}
    return {n: [int(i) for i in np.nonzero(got[n].view(np.int64) != want[n].astype(np.float64).view(np.int64))[0]]
            if got[n].shape == want[n].shape else "shape" for n in want}


# check-list name -> (%MON field, statistic) of the near-zero means / sds whose testreport digits vs results/ are
# round-off (plan "Decisions 2026-10-02": gated relative to the field size)
ROUNDOFF = {"Vav": ("vvel", "mean"), "Uav": ("uvel", "mean"), "Tsd": ("theta", "sd")}


def _mon_series(lines):
    """{%MON name: [values in record order]} of STDOUT records."""
    import re
    out = {}
    for ln in lines:
        m = re.search(r"%MON (\w+)\s*=\s*(\S+)", ln)
        if m:
            v = m.group(2).replace("D", "E")
            e3 = re.fullmatch(r"([+-]?\d*\.\d+)([+-]\d{3})", v)    # 1PE with a 3-digit exponent: no letter
            out.setdefault(m.group(1), []).append(float(f"{e3.group(1)}E{e3.group(2)}" if e3 else v))
    return out


def roundoff_rel(records, exp, inp, names):
    """{check-list name: max over the monitor times of |ours - results/| / field size} for near-zero statistics:
    ours = the run's %MON records, results/ = verification/<exp>/results/output[.<v>].txt (the testreport reference,
    the yardstick's), field size = max(|dynstat_<f>_min|, |dynstat_<f>_max|) of results/ at the same time (the
    corresponding max). A time with field size 0 counts 0 if the two values are equal, else inf."""
    from mitjax import paths
    v = inp.split(".", 1)[1] if "." in inp else None
    ref = paths.UPSTREAM / "verification" / exp / "results" / (f"output.{v}.txt" if v else "output.txt")
    a, b = _mon_series(records), _mon_series(ref.read_text().split("\n"))
    out = {}
    for name in names:
        f, stat = ROUNDOFF[name]
        key = f"dynstat_{f}_{stat}"
        if len(a[key]) != len(b[key]):
            raise ValueError(f"{key}: {len(a[key])} records here, {len(b[key])} in {ref}")
        worst = 0.0
        for t, (x, y) in enumerate(zip(a[key], b[key])):
            size = max(abs(b[f"dynstat_{f}_min"][t]), abs(b[f"dynstat_{f}_max"][t]))
            r = abs(x - y)/size if size > 0 else (0.0 if x == y else float("inf"))
            worst = max(worst, r)
        out[name] = worst
    return out


__all__ = ["CubeRun", "cube_run", "compare_stage", "bad", "run_steps", "np"]


def sharded_vs_single(r, nproc, nsteps=3):
    """{leaf: points that differ in bits} between nsteps of FORWARD_STEP at P=nproc (TileSharding, shard_map) and
    at P=1 (the same jit step without probes), for State, FFields and phi0surf."""
    from mitjax.eesupp.shard import TileSharding
    from mitjax.tests import r2_gate as r2
    m = r.m
    ns = SimpleNamespace(cfg=m.cfg, fp=m.fp, grid=m.grid, eos=m.eos, cg2dh=m.cg2dh, params=m.params,
                         cg2d_params=m.cg2d_params, state0=m.state0, ff=m.ff0, phi0surf=m.phi0surf0)
    sh = TileSharding(_maps(r), nproc)
    step_n, (st, ff, ph) = _sharded_step_fn(ns, sh, m.arrays.pkc, m.pks)
    fn = r.step_fn(())
    s1, f1, p1, _ = m.initial_carry()
    t, it = m.start_counters()
    tn, itn = t, it
    for k in range(nsteps):
        s1, f1, p1, t, it, _, _ = r.run_step(fn, s1, f1, p1, k + 1, t, it)
        st, ff, ph, tn, itn, _ = step_n(k, st, ff, ph, tn, itn)
    return r2.tree_bits_differ(sh.unpad_tree((st, ff, ph)), jax.tree.map(np.asarray, (s1, f1, p1)))


def _sharded_step_fn(m, sh, pkc, pks):
    """r2_gate.sharded_step_fn with the Model's package inputs (lane B, solid-body: MOM_VECINV's MOM_VISC.h `visc` in
    pkc, placed and sharded like the grid): FORWARD_STEP under jit(shard_map(check_vma=True)) on the TileSharding
    `sh`. Returns step_fn(k, state4, ff4, phi04, t, it) and the placed (state0, ff0, phi00)."""
    from mitjax.model.src.forward_step import forward_step
    from mitjax.tests.r2_gate import specs
    cfg, fp = m.cfg, m.fp
    g4, eos4, cg2dh4 = sh.put_tree(m.grid), sh.put_tree(m.eos), sh.put_tree(m.cg2dh)
    prm4, cgp4, pkc4 = sh.put_tree(m.params), sh.put_tree(m.cg2d_params), sh.put_tree(pkc)
    st0, ff0, ph0 = sh.put_tree(m.state0), sh.put_tree(m.ff), sh.put_tree(m.phi0surf)

    def body(grid, params, eos, cg2dh, cg2d_params, ex, pkc, state, ff, phi0surf, iloop, myTime, myIter):
        st, f, p, t, it, out = forward_step(iloop, myTime, myIter, cfg=cfg, grid=grid, params=params, fp=fp, eos=eos,
                                            cg2dh=cg2dh, cg2d_params=cg2d_params, state=state, ff=ff,
                                            phi0surf=phi0surf, ex=ex, pkc=pkc, pks=pks)
        return st, f, p, t, it, out

    T, R = sh.TILES, sh.REP
    flow_spec = specs(sh, m.state0.uVel)
    in_specs = (specs(sh, m.grid), specs(sh, m.params), specs(sh, m.eos), specs(sh, m.cg2dh),
                specs(sh, m.cg2d_params), T, specs(sh, pkc), specs(sh, m.state0), specs(sh, m.ff),
                specs(sh, m.phi0surf), R, R, R)
    out_specs = (specs(sh, m.state0), specs(sh, m.ff), specs(sh, m.phi0surf), R, R,
                 {"flow": (flow_spec, flow_spec, flow_spec), "cg2d": R})
    f = sh.shard_map(body, in_specs=in_specs, out_specs=out_specs)

    def step_fn(k, state, ff, phi0, t, it):
        return f(g4, prm4, eos4, cg2dh4, cgp4, sh.ex, pkc4, state, ff, phi0, jnp.int32(k + 1), jnp.float64(t),
                 jnp.int32(it))
    return step_fn, (st0, ff0, ph0)


def _maps(r):
    from mitjax.eesupp import exch_maps as EM
    return EM.load_cube_maps(r.exp, r.inp)


def fixed_grid_g00_bad(exp, inp, gp=None):
    """({field: points that differ in bits}, fields compared) of the model's INITIALISE_FIXED grid chain
    (drivers/model.initialise_fixed_grid) vs every G00_geometry field of lane A's dumps-on run that the Grid holds
    (all points; the vertical vectors through grid_gate.as_dump_shape)."""
    from mitjax.config.params import load
    from mitjax.drivers.model import initialise_fixed_grid
    from mitjax.eesupp import exch_maps as EM
    from mitjax.eesupp.exchange import Exchanger
    from mitjax.io.dump import DumpSet
    from mitjax.model.src.ini_parms import ini_parms_grid
    from mitjax.io.mds import E2ioLayout
    from mitjax.pkg.exch2.w2_eeboot import exch2_topology, w2_eeboot
    from mitjax.pkg.rw.read_rec import RW
    from mitjax.tests import grid_gate as gg
    e = load(exp, inp)
    w2, _ = w2_eeboot(e)
    gp = ini_parms_grid(e, exch2_topology(w2)) if gp is None else gp
    top = run_top(exp, inp)
    e2io = E2ioLayout.from_w2(w2, e.cfg.size) if w2.W2_useE2ioLayOut else None   # as drivers/model.py
    rw = RW(top / "rundir", gp.readBinaryPrec, e.cfg.size, e2io=e2io)
    g = initialise_fixed_grid(e, gp, ex=Exchanger(EM.load_cube_maps(exp, inp)), rw=rw)
    ds = DumpSet(top / "dumps")
    it = ds.iterations()[0]
    bad, n = {}, 0
    for nm in sorted({k[2] for k in ds.keys(it) if k[1] == "G00_geometry"}):
        if not hasattr(g, nm):
            continue
        a = getattr(g, nm)
        try:
            o, ref = gg.as_dump_shape(getattr(a, "data", a), ds.field(it, "G00_geometry", nm))
        except (ValueError, IndexError):
            continue
        if np.shape(o) != np.shape(ref):
            continue
        n += 1
        k = int(np.count_nonzero(np.ascontiguousarray(o, np.float64).view(np.int64)
                                 != np.ascontiguousarray(ref, np.float64).view(np.int64)))
        if k:
            bad[nm] = k
    return bad, n


def whole_run(exp, inp, tag="cube", kind="jdon"):
    """advect_gate.whole_run with the links the experiment's prepare_run makes (adjustment.cs: tile00N.mitgrid from
    ../../aim.5l_cs/input): the run driver's make_rundir links only linkdata's files, so the entries lane A's run
    directory records as `linked_by: prepare_run` (its MANIFEST.json) are linked to the same upstream files before the
    Model is set up. Returns (driver Model, forward result, monitor_gate Oracle of registry kind `kind`; the FTZ oracle
    ftz_jdon for the variants that meet subnormals, plan decision 13)."""
    import json
    import os
    from pathlib import Path
    from mitjax import paths as P
    from mitjax.drivers.model import Model as DriverModel
    from mitjax.drivers.run import forward, load_experiment, make_rundir
    from mitjax.tests import advect_gate as ag
    from mitjax.tests import monitor_gate as mg
    exp_dir = P.UPSTREAM / "verification" / exp
    e = load_experiment(exp_dir, inp)
    rundir = make_rundir(exp_dir, inp, ag.out_dir(f"{exp}-{inp}-{tag}"))
    man = json.loads((run_top(exp, inp, kind) / "MANIFEST.json").read_text())
    for name, ent in man["entries"].items():
        if ent.get("linked_by") == "prepare_run":
            if (rundir / name).exists():          # make_rundir now runs prepare_run itself (ADVECT s4): same target
                assert (rundir / name).resolve() == Path(ent["resolves_to"]).resolve(), name
            else:
                os.symlink(ent["resolves_to"], rundir / name)
    m = DriverModel(e, rundir)
    return m, forward(m), mg.oracle(exp, inp, kind)


def output_file_diffs(m, o):
    """({file: identical?} for every .data / .meta file the oracle run wrote that our run directory also holds, byte
    for byte; [files the oracle wrote that we did not]). Files linked into the run directory (inputs) are skipped."""
    import filecmp
    od = o.stdout_path.parent
    theirs = sorted(p.name for p in od.iterdir() if p.is_file() and not p.is_symlink()
                    and p.suffix in (".data", ".meta"))
    res = {f: filecmp.cmp(m.rundir / f, od / f, shallow=False) for f in theirs if (m.rundir / f).exists()}
    missing = [f for f in theirs if not (m.rundir / f).exists()]
    return res, missing
