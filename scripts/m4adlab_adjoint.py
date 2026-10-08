#!/usr/bin/env python3
"""The adjoint of lab_sea/input_ad vs TAF (lane M4ADLAB session 5; measurement script).

    m4adlab_adjoint.py OUT_DIR [PHASES]      PHASES: comma list of fwd, sweep, grad, tlm, mon (default all)
    run by $MJX_RUNS/m4adlab/dev/dev.sbatch <commit> scripts/m4adlab_adjoint.py $MJX_RUNS/m4adlab/<new dir> <phases>

The control is the grdchk variable of data.grdchk: xx_atemp (gentim2d iarr 1), record 1, tile (1,1), (i,j) =
(6..10, 8) (`grdchk pos: i,j,k= 6..10 8 1 ; bi,bj= 1 1 ; rec= 1`), eps 1e-3; every other record / control zero
(drivers/adjoint_run.problem with iarr 1, rec 1). The run's own settings: TAF's adjoint mode = the forward
(lane M4ADLAB session 1), the LSR derivative = A1 (SEAICE_LSR_ADJOINT_ITER: mitjax/ad/lsr_sweeps.taped_sweeps).
Phases (one compiled program each, jax.clear_caches() between):
  fwd    fc vs TAF's ADM ref_cost_function; our central FD at the five points (eps 1e-3) with fc+ / fc- / FD in
         printed digits vs the FD oracle job27856073-fdzero (= TAF's grdchk lines)
  sweep  central FD at eps 1e-3 * (10, 1e-1, 1e-2, 1e-3) at the five points (truncation vs rounding)
  grad   value_and_grad (with the adjoint-monitor sinks): admGrd at the five points vs TAF (digits), finite on every
         lane, zero off the interior, a repeat (bitwise); compile time, XLA memory analysis, peak RSS
  tlm    dot test jvp(v) vs <grad, v> for a random interior v; jvp at the five unit vectors vs the gradient
  mon    the `%MON ad_dynstat_*` records of the reverse sweep vs TAF's output_adm.txt (digits; no threshold)
Writes OUT_DIR/result.json after each step; refuses an OUT_DIR that already holds one.
"""
import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()
import jax  # noqa: E402

from mitjax import paths  # noqa: E402
from mitjax.io import stdout as so  # noqa: E402

OUT = Path(sys.argv[1])
phases = sys.argv[2].split(",") if len(sys.argv) > 2 else ["fwd", "sweep", "grad", "tlm", "mon"]
if (OUT / "result.json").exists():
    raise SystemExit(f"{OUT}/result.json exists: choose a new OUT_DIR (nothing is overwritten)")
OUT.mkdir(parents=True, exist_ok=True)
R = {"phases": phases, "t": {}}
EXP = ("lab_sea", "input_ad")


def save():
    (OUT / "result.json").write_text(json.dumps(R, indent=1, default=float))


def digits(a, b):
    if a == b:
        return 16
    if a == 0.0 or b == 0.0:
        return 0
    return int(np.floor(-np.log10(abs(a - b) / max(abs(a), abs(b)))))


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def clear():
    jax.clear_caches()
    gc.collect()


t0 = time.time()
from mitjax.drivers.adjoint_run import Adjoint, adm_lines  # noqa: E402
from mitjax.drivers.model import Model  # noqa: E402
from mitjax.drivers.run import load_experiment, make_rundir  # noqa: E402

exp_dir = paths.UPSTREAM / "verification" / EXP[0]
e = load_experiment(exp_dir, EXP[1])
rundir = make_rundir(exp_dir, EXP[1], OUT / "run")
m = Model(e, rundir)
a = Adjoint(m, iarr=1, rec=1, monitor="mon" in phases)
sz = m.cfg.size
P = [(0, sz.OLy + 8 - 1, sz.OLx + i - 1) for i in (6, 7, 8, 9, 10)]     # tile (1,1), (i, 8)
R["points"] = P
taf_path = exp_dir / "results" / "output_adm.txt"
taf = so.grdchk(so.read_stdout(taf_path))
orc = so.grdchk(so.read_stdout(paths.REFERENCE_RUNS / EXP[0] / EXP[1] / "job27856073-fdzero" / "rundir" / "output.txt"))
R["taf"] = dict(fcref=taf.fcref, adm=[p.adm for p in taf.points])
R["oracle"] = dict(fcref=orc.fcref, adm=[p.adm for p in orc.points],
                   fcpert=[(getattr(p, "fcpertplus", None), getattr(p, "fcpertminus", None)) for p in orc.points])
eps = float(m.exp.params["data.grdchk:grdchk_nml:grdchk_eps"])
R["eps"] = eps
R["t"]["setup"] = time.time() - t0
R["rss_setup_gb"] = rss_gb()
save()

if "fwd" in phases or "sweep" in phases:
    t1 = time.time()
    fc = float(a.cost())
    R["t"]["fwd_compile_run"] = time.time() - t1
    R["fc"] = fc
    R["fc_printed"] = f"{fc:.14E}"
    R["fc_digits_taf"] = digits(fc, R["taf"]["fcref"])
    save()
if "fwd" in phases:
    t1 = time.time()
    fd = a.grdchk_fd(P, eps=eps)
    R["fd"] = [(fp, fm, g) for (_, fp, fm, g) in fd]
    R["fd_printed"] = [(f"{fp:.14E}", f"{fm:.14E}", f"{g:.14E}") for (_, fp, fm, g) in fd]
    R["fd_printed_equal_oracle"] = [f"{g:.14E}" == f"{o['finite-diff_grad']:.14E}"
                                    for (_, _, _, g), o in zip(fd, R["oracle"]["adm"])]
    R["fcpert_printed_equal_oracle"] = [(f"{fp:.14E}" == f"{o[0]:.14E}", f"{fm:.14E}" == f"{o[1]:.14E}")
                                        for (fp, fm, _), o in zip(R["fd"], R["oracle"]["fcpert"])]
    R["t"]["fd"] = time.time() - t1
    save()
if "sweep" in phases:
    t1 = time.time()
    sweep = {}
    for n, p in enumerate(P):
        hs = []
        for f in (10.0, 0.1, 0.01, 0.001):
            (_, fp, fm, g), = a.grdchk_fd([p], eps=eps * f)
            hs.append((eps * f, fp, fm, g))
        sweep[n + 1] = hs
        R["hsweep"] = sweep
        save()
    R["t"]["sweep"] = time.time() - t1
    save()
clear()

g = None
if "grad" in phases:
    t1 = time.time()
    args = (a.theta0, a.model, a.st0, a.xs) + ((a._sinks,) if a.monitor else ())
    comp = a._vg.lower(*args).compile()
    R["t"]["grad_compile"] = time.time() - t1
    try:
        ma = comp.memory_analysis()
        R["grad_memory"] = {k: int(getattr(ma, k)) for k in ("temp_size_in_bytes", "argument_size_in_bytes",
                                                             "output_size_in_bytes", "generated_code_size_in_bytes")
                            if hasattr(ma, k)}
    except Exception as exc:                                    # measured, reported
        R["grad_memory"] = f"{type(exc).__name__}: {exc}"[:300]
    save()
    t2 = time.time()
    out = jax.block_until_ready(comp(*args))
    R["t"]["grad_run"] = time.time() - t2
    R["rss_after_grad_gb"] = rss_gb()
    fcg, g = out[0], np.asarray(out[1])
    stats = {k: np.asarray(v) for k, v in out[2].items()} if a.monitor else None
    if stats is not None:
        np.savez(OUT / "monitor_stats.npz", **stats)
    np.save(OUT / "grad.npy", g)
    R["fc_grad"] = float(fcg)
    R["grad_finite"] = bool(np.all(np.isfinite(g)))
    R["grad"] = [float(g[p]) for p in P]
    R["grad_printed"] = [f"{float(g[p]):.14E}" for p in P]
    R["grad_taf"] = [o["adjoint_gradient"] for o in R["taf"]["adm"]]
    R["grad_digits_taf"] = [digits(float(g[p]), o["adjoint_gradient"]) for p, o in zip(P, R["taf"]["adm"])]
    R["grad_rel_taf"] = [abs(float(g[p]) - o["adjoint_gradient"]) / abs(o["adjoint_gradient"])
                         for p, o in zip(P, R["taf"]["adm"])]
    fdl = R.get("fd")
    R["adm_lines"] = []
    for k, p in enumerate(P):
        R["adm_lines"] += adm_lines(float(fcg), g[p], fdl[k][2] if fdl else 0.0)
    interior = np.zeros(g.shape, bool)
    interior[..., sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] = True
    R["grad_nonzero_off_interior"] = int(np.count_nonzero(g[~interior]))
    R["grad_nonzero_interior"] = int(np.count_nonzero(g[interior]))
    save()
    t2 = time.time()
    out2 = jax.block_until_ready(comp(*args))
    g2 = np.asarray(out2[1])
    R["repeat_bitwise"] = bool(np.array_equal(g2.view(np.uint64), g.view(np.uint64)))
    R["t"]["grad_repeat"] = time.time() - t2
    save()
    del comp
    clear()

if "tlm" in phases and g is not None:
    t1 = time.time()
    rng = np.random.default_rng(20261005)
    v = np.zeros(g.shape)
    v[..., sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] = rng.standard_normal(
        v[..., sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx].shape)
    _, jv = a.jvp(jax.numpy.asarray(v))
    lhs, rhs = float(jv), float(np.sum(g * v))
    R["dot_test"] = dict(jvp=lhs, grad_dot_v=rhs, rel=abs(lhs - rhs) / max(abs(lhs), abs(rhs)))
    R["t"]["tlm_compile_run"] = time.time() - t1
    save()
    tl = []
    for p in P:
        e1 = np.zeros(g.shape)
        e1[p] = 1.0
        _, jp = a.jvp(jax.numpy.asarray(e1))
        tl.append((float(jp), float(g[p]), digits(float(jp), float(g[p]))))
        R["tlm_points"] = tl
        save()
    R["t"]["tlm"] = time.time() - t1
    save()
    clear()

if "mon" in phases and g is not None:
    from mitjax.drivers.adjoint_run import ad_monitor_records
    t1 = time.time()
    try:
        carries = a.boundary_states()
        recs = ad_monitor_records(m, stats, carries)

        def blocks(rs):
            out, cur = [], None
            for r in rs:
                if "%MON ad_time_tsnumber" in r:
                    cur = []
                    out.append(cur)
                if cur is not None and "%MON ad_" in r:
                    cur.append(r.rstrip())
            return out
        ours = blocks(recs)
        taf_b = blocks(taf_path.read_text(errors="replace").splitlines())
        R["mon_nblocks"] = (len(ours), len(taf_b))
        dig = {}
        for bo, bt in zip(ours, taf_b):
            no = {r.split("%MON")[1].split("=")[0].strip(): float(r.split("=")[1]) for r in bo}
            nt = {r.split("%MON")[1].split("=")[0].strip(): float(r.split("=")[1]) for r in bt}
            it = int(nt.get("ad_time_tsnumber", -1))
            for name in nt:
                if name in no and name.startswith("ad_dynstat"):
                    dig.setdefault(name, []).append((it, digits(no[name], nt[name]), no[name], nt[name]))
            R.setdefault("mon_missing", sorted(set(nt) - set(no)))
        R["mon_records"] = dig
        R["mon_min_digits"] = {n: min(d for _, d, _, _ in v) for n, v in dig.items()}
        (OUT / "monitor_ours.txt").write_text("\n".join(recs) + "\n")
        try:
            from mitjax.drivers.adjoint_run import ad_seaice_records
            srecs = ad_seaice_records(m, stats)
            (OUT / "monitor_seaice_ours.txt").write_text("\n".join(srecs) + "\n")

            def sblocks(rs):
                out, cur = [], None
                for r in rs:
                    if "%MON ad_seaice_tsnumber" in r:
                        cur = []
                        out.append(cur)
                    if cur is not None and "%MON ad_seaice" in r:
                        cur.append(r.rstrip())
                return out
            so_, st_ = sblocks(srecs), sblocks(taf_path.read_text(errors="replace").splitlines())
            sdig = {}
            for bo, bt in zip(so_, st_):
                no = {r.split("%MON")[1].split("=")[0].strip(): float(r.split("=")[1]) for r in bo}
                nt = {r.split("%MON")[1].split("=")[0].strip(): float(r.split("=")[1]) for r in bt}
                it = int(nt.get("ad_seaice_tsnumber", -1))
                for name in nt:
                    if name in no and "tsnumber" not in name and "time_sec" not in name:
                        sdig.setdefault(name, []).append((it, digits(no[name], nt[name]), no[name], nt[name]))
            R["seaice_nblocks"] = (len(so_), len(st_))
            R["seaice_records"] = sdig
        except Exception:                               # measured, reported (not a gate)
            import traceback
            R["seaice_error"] = traceback.format_exc()[-3000:]
    except Exception as exc:                                    # measured, reported (not a gate)
        import traceback
        R["mon_error"] = traceback.format_exc()[-3000:]
    R["t"]["mon"] = time.time() - t1
    save()
R["t"]["total"] = time.time() - t0
R["rss_end_gb"] = rss_gb()
save()
