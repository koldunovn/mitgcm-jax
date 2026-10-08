#!/usr/bin/env python3
"""The adjoint of 1D_ocean_ice_column/input_ad vs TAF (lane M4ADCOL session 3; measurement script).

    m4adcol_adjoint.py OUT_DIR [PHASES]      PHASES: comma list of fwd, grad, tlm, mon (default fwd,grad,tlm,mon)
    sbatch -J mjx_m4adcol_adj -p compute -A ab0995 --time=01:00:00 -N1 --wrap \
        "JAX_PLATFORMS=cpu PYTHONPATH=<snapshot> python scripts/m4adcol_adjoint.py $MJX_RUNS/m4adcol/<new dir>"

Builds the driver Model of input_ad (useECCO as the run sets it, zero control, a fresh run directory) and
drivers/adjoint_run.GenarrAdjoint on xx_theta (the grdchk variable, genarr3d iarr 1), then, one compiled program per
phase with jax.clear_caches() between:
  fwd   fc vs TAF's `ADM ref_cost_function` and the FD oracle job27856073-fdzero; our central FD (eps 1e-7, the
        grdchk perturbations) at the four points vs the oracle's printed FD; an eps sweep at every point
  grad  value_and_grad: admGrd at the four points vs TAF (digits), finite everywhere, nonzero only on wet interior
        points, a repeat (bitwise)
  tlm   dot test: jvp(v) . w-free form <J'(x) v> vs <grad, v> for a random v on the interior (relative difference),
        and jvp at the four unit vectors vs the gradient entries
  mon   the `%MON ad_*` blocks of the reverse sweep (ad_monitor_records) vs TAF's (results/output_adm.txt): digits per
        record (no threshold; Nikolay's open decision)
Writes OUT_DIR/result.json after each phase; refuses an OUT_DIR that already holds one.
"""
import gc
import json
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
phases = sys.argv[2].split(",") if len(sys.argv) > 2 else ["fwd", "grad", "tlm", "mon"]
if (OUT / "result.json").exists():
    raise SystemExit(f"{OUT}/result.json exists: choose a new OUT_DIR (nothing is overwritten)")
OUT.mkdir(parents=True, exist_ok=True)
R = {"phases": phases, "t": {}}
EXP = ("1D_ocean_ice_column", "input_ad")


def save():
    (OUT / "result.json").write_text(json.dumps(R, indent=1, default=float))


def digits(a, b):
    if a == b:
        return 16
    if a == 0.0 or b == 0.0:
        return 0
    return int(np.floor(-np.log10(abs(a - b) / max(abs(a), abs(b)))))


def clear():
    jax.clear_caches()
    gc.collect()


t0 = time.time()
from mitjax.drivers.adjoint_run import GenarrAdjoint, adm_lines  # noqa: E402
from mitjax.drivers.model import Model  # noqa: E402
from mitjax.drivers.run import load_experiment, make_rundir  # noqa: E402

exp_dir = paths.UPSTREAM / "verification" / EXP[0]
e = load_experiment(exp_dir, EXP[1])
rundir = make_rundir(exp_dir, EXP[1], OUT / "run")
m = Model(e, rundir)
a = GenarrAdjoint(m, key=(3, 1), monitor="mon" in phases)
sz = m.cfg.size
P = [(0, k - 1, sz.OLy, sz.OLx) for k in (1, 2, 3, 4)]       # data.grdchk: (i,j) = (1,1), k = 1..4, tile 1
R["points"] = P
taf_path = paths.UPSTREAM / "verification" / EXP[0] / "results" / "output_adm.txt"
taf = so.grdchk(so.read_stdout(taf_path))
orc = so.grdchk(so.read_stdout(paths.REFERENCE_RUNS / EXP[0] / EXP[1] / "job27856073-fdzero" / "rundir" / "output.txt"))
R["taf"] = dict(fcref=taf.fcref, adm=[p.adm for p in taf.points])
R["oracle"] = dict(fcref=orc.fcref, adm=[p.adm for p in orc.points],
                   fcpert=[(getattr(p, "fcpertplus", None), getattr(p, "fcpertminus", None)) for p in orc.points])
eps = float(m.exp.params["data.grdchk:grdchk_nml:grdchk_eps"])
R["eps"] = eps
R["t"]["setup"] = time.time() - t0
save()

if "fwd" in phases:
    t1 = time.time()
    fc = float(a.cost())
    R["t"]["fwd_compile_run"] = time.time() - t1
    R["fc"] = fc
    R["fc_digits_taf"] = digits(fc, R["taf"]["fcref"])
    save()
    fd = a.grdchk_fd(P, eps=eps)
    R["fd"] = [(fp, fm, g) for (_, fp, fm, g) in fd]
    R["fd_printed_equal_oracle"] = [f"{g:.14E}" == f"{o['finite-diff_grad']:.14E}"
                                    for (_, _, _, g), o in zip(fd, R["oracle"]["adm"])]
    R["fcpert_printed_equal_oracle"] = [(f"{fp:.14E}" == f"{o[0]:.14E}", f"{fm:.14E}" == f"{o[1]:.14E}")
                                        for (fp, fm, _), o in zip(R["fd"], R["oracle"]["fcpert"])]
    save()
    sweep = {}
    for n, p in enumerate(P):
        hs = []
        for f in (1e3, 1e2, 10.0, 1.0, 0.1, 0.01):
            (_, fp, fm, g), = a.grdchk_fd([p], eps=eps * f)
            hs.append((eps * f, g))
        sweep[n + 1] = hs
        save()
    R["hsweep"] = sweep
    R["t"]["fwd"] = time.time() - t1
    save()
    clear()

g = None
if "grad" in phases:
    t1 = time.time()
    out = a.value_and_grad()
    fcg, g = out[0], np.asarray(out[1])
    stats = {k: np.asarray(v) for k, v in out[2].items()} if a.monitor else None
    if stats is not None:
        np.savez(OUT / "monitor_stats.npz", **stats)
    R["t"]["grad_compile_run"] = time.time() - t1
    np.save(OUT / "grad.npy", g)
    R["fc_grad"] = float(fcg)
    R["grad_finite"] = bool(np.all(np.isfinite(g)))
    R["grad"] = [float(g[p]) for p in P]
    R["grad_taf"] = [o["adjoint_gradient"] for o in R["taf"]["adm"]]
    R["grad_digits_taf"] = [digits(float(g[p]), o["adjoint_gradient"]) for p, o in zip(P, R["taf"]["adm"])]
    R["grad_full_column"] = [float(x) for x in g[0, :, sz.OLy, sz.OLx]]
    fdl = R.get("fd")
    R["adm_lines"] = []
    for k, p in enumerate(P):
        R["adm_lines"] += adm_lines(float(fcg), g[p], fdl[k][2] if fdl else 0.0)
    interior = np.zeros(g.shape, bool)
    interior[..., sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] = True
    wet = interior & (np.asarray(m.grid.maskC.data) != 0)
    R["grad_nonzero_off_wet"] = int(np.count_nonzero(g[~wet]))
    R["grad_nonzero_wet"] = int(np.count_nonzero(g[wet]))
    save()
    t2 = time.time()
    out2 = a.value_and_grad()
    g2 = np.asarray(out2[1])
    R["repeat_bitwise"] = bool(np.array_equal(g2.view(np.uint64), g.view(np.uint64)))
    R["t"]["grad_repeat"] = time.time() - t2
    save()
    clear()

if "tlm" in phases and g is not None:
    t1 = time.time()
    rng = np.random.default_rng(20261004)
    v = np.zeros(g.shape)
    v[..., sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] = rng.standard_normal(
        v[..., sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx].shape)
    _, jv = a.jvp(jax.numpy.asarray(v))
    lhs, rhs = float(jv), float(np.sum(g * v))
    R["dot_test"] = dict(jvp=lhs, grad_dot_v=rhs, rel=abs(lhs - rhs) / max(abs(lhs), abs(rhs)))
    save()
    tl = []
    for p in P:
        e1 = np.zeros(g.shape)
        e1[p] = 1.0
        _, jp = a.jvp(jax.numpy.asarray(e1))
        tl.append((float(jp), float(g[p]), digits(float(jp), float(g[p]))))
    R["tlm_points"] = tl
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
    except Exception as exc:                                    # measured, reported (not a gate)
        R["mon_error"] = f"{type(exc).__name__}: {exc}"[:2000]
    R["t"]["mon"] = time.time() - t1
    save()
R["t"]["total"] = time.time() - t0
save()
