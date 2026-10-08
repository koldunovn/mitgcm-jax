#!/usr/bin/env python3
"""The adjoint of global_ocean.cs32x15 input_ad.seaice / input_ad.seaice_dynmix vs TAF (lane M4ADCS32ICE session 2;
measurement script).

    m4adcs32ice_adjoint.py OUT_DIR VARIANT MODE [PHASES]
        VARIANT  seaice | dynmix
        MODE     exact  (no backward-only switch; the LSR derivative A1 by mjx_lsr_derivative = "sweeps")
                 run    (the run's own data.autodiff switches: drivers/ad_switches.py, the sea-ice approximate
                         advection at the "flux" level; + A1 where the LSR is differentiated)
                 run-routine  (as run, the sea-ice approximate advection at the "routine" level: O1)
        PHASES   comma list of fwd, sweep, grad, tlm, mon (default fwd,grad,tlm)
    run by $MJX_RUNS/m4adcs32ice/s2/dev/dev.sbatch <commit> scripts/m4adcs32ice_adjoint.py <new dir> ...

The control is the grdchk variable xx_theta (genarr3d), k = 1, tile (1,1), (i, j) = (1..4, 1) (the FD oracle's
`grdchk pos`), eps 1e-2 (data.grdchk); the other controls at their zero first guess (GenarrAdjoint(key=)).
The CG2D derivative follows the run's cg2dFullAdjoint = .FALSE. (operator passive, as TAF's CG2D_MAD and TAF's TLM).
Phases (one compiled program each, jax.clear_caches() between):
  fwd    fc of the default program and of MODE's program (bitwise equal?), vs TAF's ref_cost_function; our central FD
         at the four points (eps 1e-2) vs the FD oracle job27856074-fdzero's printed lines
  sweep  central FD at eps 1e-2 x (1e-1, 1e-2) at the four points
  grad   value_and_grad: admGrd at the four points vs TAF's TLM (tangent-lin_grad) and TAF's ADM (adjoint_gradient),
         digits; finite on every lane; zero off the interior; a repeat (bitwise); compile time, memory, RSS
  tlm    dot test jvp(v) vs <grad, v> (random interior v); jvp at the four unit vectors vs the gradient
  mon    with `grad`: the %MON ad_dynstat_* / ad_seaice_* records of the reverse sweep vs output_adm.<variant>.txt
Writes OUT_DIR/result.json after each step; refuses an OUT_DIR that already holds one.
"""
import gc
import gzip
import json
import re
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
VAR, MODE = sys.argv[2], sys.argv[3]
phases = sys.argv[4].split(",") if len(sys.argv) > 4 else ["fwd", "grad", "tlm"]
assert VAR in ("seaice", "dynmix") and MODE in ("exact", "run", "run-routine"), (VAR, MODE)
if MODE != "exact" and "tlm" in phases:
    raise SystemExit("tlm: the run-mode hooks are reverse-only (custom_vjp); the TLM is the exact mode's")
if (OUT / "result.json").exists():
    raise SystemExit(f"{OUT}/result.json exists: choose a new OUT_DIR (nothing is overwritten)")
OUT.mkdir(parents=True, exist_ok=True)
R = {"variant": VAR, "mode": MODE, "phases": phases, "t": {}}
INP = {"seaice": "input_ad.seaice", "dynmix": "input_ad.seaice_dynmix"}[VAR]
EXP = ("global_ocean.cs32x15", INP)


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
from mitjax.drivers.adjoint_run import GenarrAdjoint, adm_lines  # noqa: E402
from mitjax.drivers.model import Model  # noqa: E402
from mitjax.drivers.run import load_experiment, make_rundir  # noqa: E402
from mitjax.pkg.ctrl.ctrl_readparms import ctrl_readparms_genarr  # noqa: E402

exp_dir = paths.UPSTREAM / "verification" / EXP[0]
e = load_experiment(exp_dir, EXP[1])
rundir = make_rundir(exp_dir, EXP[1], OUT / "run")
m = Model(e, rundir)
key = next((3, g.iarr) for g in ctrl_readparms_genarr(m.exp.run, 3, m.cfg) if g.file.strip() == "xx_theta")
a = GenarrAdjoint(m, key=key, monitor="mon" in phases)
sz = m.cfg.size
fdo_path = paths.REFERENCE_RUNS / EXP[0] / EXP[1] / "job27856074-fdzero" / "rundir" / "output.txt"
orc = so.grdchk(so.read_stdout(fdo_path))
P = [(p.bi - 1 + (p.bj - 1) * sz.nSx, p.k - 1, p.j - 1 + sz.OLy, p.i - 1 + sz.OLx) for p in orc.points]
R["points"] = P
res = exp_dir / "results"
taf_path = res / f"output_adm.{VAR if VAR == 'seaice' else 'seaice_dynmix'}.txt"
tlm_path = res / f"output_tlm.{VAR if VAR == 'seaice' else 'seaice_dynmix'}.txt.gz"
taf = so.grdchk(so.read_stdout(taf_path))
tlm_txt = gzip.open(tlm_path, "rt", errors="replace").read()
R["taf"] = dict(fcref=taf.fcref, adm=[p.adm["adjoint_gradient"] for p in taf.points],
                fd=[p.adm["finite-diff_grad"] for p in taf.points],
                tlm=[float(x) for x in re.findall(r"TLM\s+tangent-lin_grad\s+=\s+(\S+)", tlm_txt)],
                tlm_fd=[float(x) for x in re.findall(r"TLM\s+finite-diff_grad\s+=\s+(\S+)", tlm_txt)])
R["oracle"] = dict(fcref=orc.fcref, fd=[p.adm["finite-diff_grad"] for p in orc.points])
eps = float(m.exp.params["data.grdchk:grdchk_nml:grdchk_eps"])
R["eps"] = eps

# MODE's program: the Model's arrays with the options (static: a new trace)
from mitjax.ad.modes import with_lsr_derivative  # noqa: E402

model = a.model.replace(pkc={**a.model.pkc, "sp": with_lsr_derivative(a.model.pkc["sp"], "sweeps")})
if MODE != "exact":
    from mitjax.drivers.ad_switches import with_run_switches  # noqa: E402
    model = with_run_switches(m, model, seaice_level="routine" if MODE == "run-routine" else "flux")
    R["switches"] = dict(params=model.params.static_items().get("mjx_approx_advection_in_ad"),
                         sp_approx=getattr(model.pkc["sp"], "mjx_approx_advection_in_ad", None),
                         sp_freedrift=getattr(model.pkc["sp"], "mjx_freedrift_in_ad", None))
R["t"]["setup"] = time.time() - t0
R["rss_setup_gb"] = rss_gb()
save()

if "fwd" in phases or "sweep" in phases:
    t1 = time.time()
    fc0 = float(a.cost())
    R["t"]["fwd_default_compile_run"] = time.time() - t1
    clear()
    t1 = time.time()
    fc = float(a.cost(model=model))
    R["t"]["fwd_mode_compile_run"] = time.time() - t1
    R["fc_default"], R["fc"] = fc0, fc
    R["fc_mode_bitwise_default"] = bool(np.float64(fc).view(np.uint64) == np.float64(fc0).view(np.uint64))
    R["fc_printed"] = f"{fc:.14E}"
    R["fc_printed_equal_taf"] = f"{fc:.14E}" == f"{taf.fcref:.14E}"
    save()
if "fwd" in phases:
    t1 = time.time()
    fd = a.grdchk_fd(P, eps=eps, model=model)
    R["fd"] = [(fp, fm, g) for (_, fp, fm, g) in fd]
    R["fd_printed_equal_oracle"] = [f"{g:.14E}" == f"{o:.14E}" for (_, _, _, g), o in zip(fd, R["oracle"]["fd"])]
    R["t"]["fd"] = time.time() - t1
    save()
if "sweep" in phases:
    t1 = time.time()
    sweep = {}
    for n, p in enumerate(P):
        hs = []
        for f in (0.1, 0.01):
            (_, fp, fm, g), = a.grdchk_fd([p], eps=eps * f, model=model)
            hs.append((eps * f, fp, fm, g))
        sweep[n + 1] = hs
        R["hsweep"] = sweep
        save()
    R["t"]["sweep"] = time.time() - t1
    save()
clear()

g = None
stats = None
if "grad" in phases:
    t1 = time.time()
    args = (a.theta0, model, a.st0, a.xs) + ((a._sinks,) if a.monitor else ())
    comp = a._vg.lower(*args).compile()
    R["t"]["grad_compile"] = time.time() - t1
    try:
        ma = comp.memory_analysis()
        R["grad_memory"] = {k: int(getattr(ma, k)) for k in ("temp_size_in_bytes", "argument_size_in_bytes",
                                                             "output_size_in_bytes", "generated_code_size_in_bytes")
                            if hasattr(ma, k)}
    except Exception as exc:                                    # measured, reported
        R["grad_memory"] = f"{type(exc).__name__}: {exc}"[:300]
    R["rss_after_compile_gb"] = rss_gb()
    save()
    t2 = time.time()
    out = jax.block_until_ready(comp(*args))
    R["t"]["grad_run"] = time.time() - t2
    R["rss_after_grad_gb"] = rss_gb()
    fcg, g = out[0], np.asarray(out[1])
    if a.monitor:
        stats = {k: np.asarray(v) for k, v in out[2].items()}
        np.savez(OUT / "monitor_stats.npz", **stats)
    np.save(OUT / "grad.npy", g)
    R["fc_grad"] = float(fcg)
    R["grad_finite"] = bool(np.all(np.isfinite(g)))
    R["grad"] = [float(g[p]) for p in P]
    R["grad_printed"] = [f"{float(g[p]):.14E}" for p in P]
    for k in ("tlm", "adm", "fd"):
        R[f"grad_digits_taf_{k}"] = [digits(float(g[p]), y) for p, y in zip(P, R["taf"][k])]
        R[f"grad_rel_taf_{k}"] = [abs(float(g[p]) - y) / abs(y) for p, y in zip(P, R["taf"][k])]
    R["adm_lines"] = []
    fdl = R.get("fd")
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
    del comp, out, out2
    clear()

if "tlm" in phases and g is not None:
    t1 = time.time()
    rng = np.random.default_rng(20261005)
    v = np.zeros(g.shape)
    sl = (Ellipsis, slice(sz.OLy, sz.OLy + sz.sNy), slice(sz.OLx, sz.OLx + sz.sNx))
    v[sl] = rng.standard_normal(v[sl].shape)
    v *= np.asarray(m.grid.maskC.data)
    _, jv = a.jvp(jax.numpy.asarray(v), model=model)
    lhs, rhs = float(jv), float(np.sum(g * v))
    R["dot_test"] = dict(jvp=lhs, grad_dot_v=rhs, rel=abs(lhs - rhs) / max(abs(lhs), abs(rhs)))
    R["t"]["tlm_compile_run"] = time.time() - t1
    save()
    tl = []
    for p, y in zip(P, R["taf"]["tlm"]):
        e1 = np.zeros(g.shape)
        e1[p] = 1.0
        _, jp = a.jvp(jax.numpy.asarray(e1), model=model)
        tl.append(dict(jvp=float(jp), grad=float(g[p]), digits_grad=digits(float(jp), float(g[p])),
                       digits_taf_tlm=digits(float(jp), y)))
        R["tlm_points"] = tl
        save()
    R["t"]["tlm"] = time.time() - t1
    save()
    clear()


def _blocks(rs, tag):
    out, cur = [], None
    for r in rs:
        if f"%MON {tag}_tsnumber" in r:
            cur = []
            out.append(cur)
        if cur is not None and (f"%MON {tag}" in r if tag == "ad_seaice" else
                                ("%MON ad_" in r and "%MON ad_seaice" not in r and "%MON ad_exf" not in r)):
            cur.append(r.rstrip())
    return out


def _compare(ours, theirs, tag, keep):
    dig = {}
    for bo, bt in zip(ours, theirs):
        no = {r.split("%MON")[1].split("=")[0].strip(): float(r.split("=")[1]) for r in bo}
        nt = {r.split("%MON")[1].split("=")[0].strip(): float(r.split("=")[1]) for r in bt}
        it = int(nt.get(f"{tag}_tsnumber", -1))
        for name in nt:
            if name in no and keep(name):
                dig.setdefault(name, []).append((it, digits(no[name], nt[name]), no[name], nt[name]))
    return dig


if "mon" in phases and stats is not None:
    from mitjax.drivers.adjoint_run import ad_monitor_records, ad_seaice_records
    t1 = time.time()
    taf_lines = taf_path.read_text(errors="replace").splitlines()
    try:
        carries = a.boundary_states(model=model)
        recs = ad_monitor_records(m, stats, carries)
        (OUT / "monitor_ours.txt").write_text("\n".join(recs) + "\n")
        ours, theirs = _blocks(recs, "ad_time"), _blocks(taf_lines, "ad_time")
        R["mon_nblocks"] = (len(ours), len(theirs))
        R["mon_records"] = _compare(ours, theirs, "ad_time", lambda n: n.startswith("ad_dynstat"))
    except Exception:                                    # measured, reported (not a gate)
        import traceback
        R["mon_error"] = traceback.format_exc()[-3000:]
    save()
    try:
        srecs = ad_seaice_records(m, stats)
        (OUT / "monitor_seaice_ours.txt").write_text("\n".join(srecs) + "\n")
        so_, st_ = _blocks(srecs, "ad_seaice"), _blocks(taf_lines, "ad_seaice")
        R["seaice_nblocks"] = (len(so_), len(st_))
        R["seaice_records"] = _compare(so_, st_, "ad_seaice",
                                       lambda n: "tsnumber" not in n and "time_sec" not in n)
    except Exception:                                    # measured, reported (not a gate)
        import traceback
        R["seaice_error"] = traceback.format_exc()[-3000:]
    R["t"]["mon"] = time.time() - t1
    save()
R["t"]["total"] = time.time() - t0
R["rss_peak_gb"] = rss_gb()
save()
print(json.dumps({k: R.get(k) for k in ("fc_printed", "fc_mode_bitwise_default", "fd_printed_equal_oracle",
                                         "grad_printed", "grad_digits_taf_tlm", "grad_digits_taf_adm",
                                         "grad_finite", "repeat_bitwise", "dot_test", "t", "grad_memory")},
                 indent=1, default=float))
