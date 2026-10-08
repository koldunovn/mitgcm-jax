#!/usr/bin/env python3
"""The adjoint of global_ocean.cs32x15/input_ad vs TAF (GO lane session 9, M2 Task 25; measurement script, a copy of
scripts/goadk_adjoint.py with the experiment switched: goadk_gate / goadk_model_gate read their module EXP).

    cs32_adjoint.py input_ad OUT_DIR [PHASES [FROM_DIR]]  PHASES: comma list of fwd, grad, mon, tlm (default all),
                                                          plus the flag exact (below);
                                                          FROM_DIR: grad.npy / monitor_stats.npz of an earlier run
    sbatch -J mjx_cs32_adj -p compute -A ab0995 --time=01:30:00 --mem=0 -N1 --wrap \
        "JAX_PLATFORMS=cpu PYTHONPATH=$PWD python -u scripts/cs32_adjoint.py input_ad $MJX_RUNS/go_dev/<new dir>"

CG2D derivative (mitjax/ad/modes.py): by default the run's own setting, as TAF's CG2D_MAD (cg2dFullAdjoint = .FALSE.
in input_ad: the cg2d operator passive, pkg/autodiff/cg2d_mad.F:220-236); the flag `exact` selects the exact option
(Cg2dParams.mjx_cg2d_derivative = "exact": the operator's derivative included; nonlinFreeSurf = 4 updates the operator
every step, so the two differ here). Session 10's measurement flag tafcg2d (a test-side stop_gradient) is this switch's
run setting now.

VARIANT: input_ad, input_ad.kapgm, input_ad.kapredi or input_ad.bottomdrag. Builds the driver Model of the variant
(the run directory of lane A's dumps-on run, zero control) and drivers/adjoint_run.GenarrAdjoint, then, one compiled
program per phase with jax.clear_caches() between (PORTING_LESSONS "vm.max_map_count"):
  fwd   fc vs TAF's `ADM ref_cost_function` (results/output_adm*.txt) and the oracle's fcref (lane A's fdzero run);
        GRDCHK's FD at the 4 grdchk points (+-grdchk_eps) vs the printed FD and perturbed costs of both; an FD h-sweep
        at the first point (grdchk_eps x 100 .. 1e-3)
  grad  value_and_grad: admGrd digits vs TAF at the 4 points, the ADM lines (adm_lines), finite / nonzero only on
        the wet interior, a repeat (bitwise)
  mon   (after grad; the gradient program runs with the adjoint monitor's stats hook) the `%MON ad_*` blocks of
        the reverse sweep (ad_monitor_records on GenarrAdjoint.boundary_states) vs TAF's 11 blocks: digits per
        record and block (both zero: 16; one zero: 0)
  tlm   jvp vs adjoint dot test in a random direction, and jvp at the 4 points vs TAF's `TLM tangent-lin_grad`
        lines (results/output_tlm*.txt.gz)
Writes OUT_DIR/result.json after each phase and OUT_DIR/grad.npy; refuses an OUT_DIR that already holds a result
(nothing is overwritten). Measured in session 4: jobs 27838400-27838403; the input_ad case is gated
by mitjax/tests/test_goadk_adjoint.py.
"""
import gc
import json
import sys
import time
from pathlib import Path
import numpy as np
from mitjax.xla_flags import set_gate_xla_flags
set_gate_xla_flags()
import jax
import jax.numpy as jnp
from mitjax import paths
from mitjax.io import stdout as so
from mitjax.tests import goadk_gate as G
from mitjax.tests import goadk_model_gate as MG
from mitjax.drivers.adjoint_run import GenarrAdjoint, adm_lines

EXPNAME = "global_ocean.cs32x15"
G.EXP = MG.EXP = EXPNAME                    # the helpers read the experiment from their module constant
RESDIR = "verification/" + EXPNAME + "/results"

inp = sys.argv[1]
OUT = Path(sys.argv[2])
phases = sys.argv[3].split(",") if len(sys.argv) > 3 else ["fwd", "grad", "mon", "tlm"]
if (OUT / "result.json").exists():
    raise SystemExit(f"{OUT}/result.json exists: choose a new OUT_DIR (nothing is overwritten)")
OUT.mkdir(parents=True, exist_ok=True)
R = {"inp": inp, "phases": phases, "t": {}}


def save():
    (OUT / "result.json").write_text(json.dumps(R, indent=1, default=float))


def digits(a, b):
    if a == b:
        return 16
    return int(np.floor(-np.log10(abs(a - b) / max(abs(a), abs(b)))))


def clear():
    jax.clear_caches()
    gc.collect()


R["cg2d_derivative"] = "exact" if "exact" in phases else "run"     # docstring: mitjax/ad/modes.py

t0 = time.time()
m, _ = MG.model(inp, "jdon")
from mitjax.pkg.ctrl.ctrl_readparms import ctrl_readparms_genarr
KEY = next((3, g.iarr) for g in ctrl_readparms_genarr(m.exp.run, 3, m.cfg) if g.file.strip() == "xx_theta")
R["key"] = KEY                                       # the grdchk variable (data.grdchk grdchkvarname = xx_theta)
a = GenarrAdjoint(m, monitor="mon" in phases, key=KEY)
if R["cg2d_derivative"] == "exact":
    from mitjax.ad.modes import with_cg2d_derivative
    a.model = a.model.replace(cg2d_params=with_cg2d_derivative(a.model.cg2d_params, "exact"))
sz = m.cfg.size
pts, _, s = G.grdchk_case(inp, maskC=np.asarray(m.grid.maskC.data))      # the Model's mask (= the dumps')
dim = G.CONTROL[inp][0]
P = []
for _, r in pts:
    t = r.itile - 1 + (r.jtile - 1) * sz.nSx
    j, i = r.jtilepos - 1 + sz.OLy, r.itilepos - 1 + sz.OLx
    P.append((t, r.layer - 1, j, i) if dim == 3 else (t, j, i))
R["points"] = P
R["pos"] = [(r.itilepos, r.jtilepos, r.layer, r.itile, r.jtile) for _, r in pts]
res = "output_adm.txt" if inp == "input_ad" else f"output_adm.{inp.split('.', 1)[1]}.txt"
taf = so.grdchk(so.read_stdout(paths.UPSTREAM / RESDIR / res))
orc = so.grdchk(G.fd_stdout(inp))
R["taf"] = dict(fcref=taf.fcref, adm=[p.adm for p in taf.points],
                fcpert=[(getattr(p, "fcpertplus", None), getattr(p, "fcpertminus", None)) for p in taf.points],
                pos=[(p.i, p.j, p.k, p.bi, p.bj) for p in taf.points])
R["oracle"] = dict(fcref=orc.fcref, adm=[p.adm for p in orc.points],
                   fcpert=[(getattr(p, "fcpertplus", None), getattr(p, "fcpertminus", None)) for p in orc.points],
                   pos=[(p.i, p.j, p.k, p.bi, p.bj) for p in orc.points])
assert R["taf"]["pos"] == R["pos"] == R["oracle"]["pos"], (R["taf"]["pos"], R["pos"], R["oracle"]["pos"])
dyn = [ln.text for ln in so.read_stdout(paths.UPSTREAM / RESDIR / res)
       if "ad_dynstat" in ln.text]
R["taf_ad_dynstat_lines"] = len(dyn)
R["t"]["setup"] = time.time() - t0
eps = float(m.exp.params["data.grdchk:grdchk_nml:grdchk_eps"])
R["eps"] = eps
save()

if "fwd" in phases:
    t1 = time.time()
    fc = float(a.cost())
    R["t"]["fwd_compile_run"] = time.time() - t1
    R["fc"] = fc
    R["fc_digits_taf"] = digits(fc, R["taf"]["fcref"])
    R["fc_digits_oracle"] = digits(fc, R["oracle"]["fcref"])
    save()
    fd = a.grdchk_fd(P, eps=eps)
    R["fd"] = [(fp, fm, g) for (_, fp, fm, g) in fd]
    R["fd_digits_oracle"] = [digits(g, o["finite-diff_grad"]) for (_, _, _, g), o in zip(fd, R["oracle"]["adm"])]
    R["fd_digits_taf"] = [digits(g, o["finite-diff_grad"]) for (_, _, _, g), o in zip(fd, R["taf"]["adm"])]
    R["fd_printed_equal_oracle"] = [f"{g:.14E}" == f"{o['finite-diff_grad']:.14E}"
                                    for (_, _, _, g), o in zip(fd, R["oracle"]["adm"])]
    R["fcpert_digits_oracle"] = [(digits(fp, o[0]), digits(fm, o[1])) if o[0] is not None else None
                                 for (fp, fm, _), o in zip(R["fd"], R["oracle"]["fcpert"])]
    save()
    hs = []
    for f in (100.0, 10.0, 1.0, 0.1, 0.01, 1e-3):
        (_, fp, fm, g), = a.grdchk_fd([P[0]], eps=eps * f)
        hs.append((eps * f, g))
    R["hsweep_p1"] = hs
    R["hsweep"] = [[(eps * f, a.grdchk_fd([p], eps=eps * f)[0][3]) for f in (1.0, 0.1, 0.01)] for p in P]  # GO s10
    R["t"]["fwd"] = time.time() - t1
    save()
    clear()

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
    R["grad_digits_taf"] = [digits(float(g[p]), o["adjoint_gradient"]) for p, o in zip(P, R["taf"]["adm"])]
    fdl = R.get("fd")
    R["adm_lines"] = []
    for k, p in enumerate(P):
        R["adm_lines"] += adm_lines(float(fcg), g[p], fdl[k][2] if fdl else 0.0)
    interior = np.zeros(g.shape, bool)
    interior[..., sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] = True
    mC = np.asarray(m.grid.maskC.data)
    wet = interior & ((mC != 0) if dim == 3 else (mC[:, 0] != 0))
    R["grad_nonzero_off_wet"] = int(np.count_nonzero(g[~wet]))
    R["grad_nonzero_wet"] = int(np.count_nonzero(g[wet]))
    save()
    t2 = time.time()
    out2 = a.value_and_grad()
    fcg2, g2 = out2[0], np.asarray(out2[1])                                   # block: the dispatch returns before the run ends
    R["t"]["grad_repeat"] = time.time() - t2
    R["repeat_bitwise"] = bool(np.array_equal(g2.view(np.uint64), g.view(np.uint64))) and float(fcg2) == float(fcg)
    R["t"]["grad"] = time.time() - t1
    save()
    del g2
    clear()

if len(sys.argv) > 4 and "grad" not in phases:                 # mon / tlm from an earlier run's gradient
    src = Path(sys.argv[4])
    np.save(OUT / "grad.npy", np.load(src / "grad.npy"))
    if (src / "monitor_stats.npz").exists():
        stats = dict(np.load(src / "monitor_stats.npz"))
    R["from"] = str(src)

if "mon" in phases:
    from mitjax.drivers.adjoint_run import ad_monitor_records
    t1 = time.time()
    carries = a.boundary_states()
    recs = ad_monitor_records(m, stats, carries)        # adjMonitorFreq from the run's data (1.)
    clear()

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
    taf_b = blocks((paths.UPSTREAM / RESDIR / res)
                   .read_text(errors="replace").splitlines())[:len(ours)]
    R["mon_nblocks"] = (len(ours), len(taf_b))
    dig = {}
    for bo, bt in zip(ours, taf_b):
        assert [r.split("=")[0] for r in bo] == [r.split("=")[0] for r in bt], (bo[0], bt[0])
        it = int(bo[0].split("=")[1])
        for ro, rt in zip(bo[2:], bt[2:]):
            name = ro.split("%MON")[1].split("=")[0].strip()
            x, y = float(ro.split("=")[1]), float(rt.split("=")[1])
            d = 16 if x == y else (0 if (x == 0.0 or y == 0.0) else digits(x, y))
            dig.setdefault(name, []).append((it, d, x, y))
    R["mon_digits"] = {n: [d for _, d, _, _ in v] for n, v in dig.items()}
    R["mon_min_digits"] = {n: min(d for _, d, _, _ in v) for n, v in dig.items()}
    R["mon_records"] = {n: v for n, v in dig.items()}
    R["mon_time_lines_equal"] = all(bo[:2] == bt[:2] for bo, bt in zip(ours, taf_b))
    (OUT / "monitor_ours.txt").write_text("\n".join(recs) + "\n")
    R["t"]["mon"] = time.time() - t1
    save()

if "tlm" in phases:
    import gzip
    import re
    t1 = time.time()
    g = np.load(OUT / "grad.npy")
    rng = np.random.default_rng(20261002)
    v = rng.standard_normal(g.shape)
    fct, d = a.jvp(jnp.asarray(v))
    d = float(d)
    R["t"]["tlm_compile_run"] = time.time() - t1
    gv = float(np.sum(g * v))
    R["dot"] = dict(jvp=d, adj=gv, rel=abs(d - gv) / max(abs(d), abs(gv)), fc=float(fct))
    save()
    tres = res.replace("output_adm", "output_tlm") + ".gz"
    tl = gzip.open(paths.UPSTREAM / RESDIR / tres, "rt", errors="replace").read()
    taf_tl = [float(x) for x in re.findall(r"TLM  tangent-lin_grad\s*=\s*(\S+)", tl)]
    taf_tlfc = [float(x) for x in re.findall(r"TLM  ref_cost_function\s*=\s*(\S+)", tl)]
    jv = []
    for p in P:
        e = np.zeros(g.shape)
        e[p] = 1.0
        _, dp = a.jvp(jnp.asarray(e))
        jv.append(float(dp))
    R["tlm"] = dict(ours=jv, taf=taf_tl, taf_fc=taf_tlfc, digits=[digits(x, y) for x, y in zip(jv, taf_tl)],
                    digits_vs_adjoint=[digits(x, float(g[p])) for x, p in zip(jv, P)])
    R["t"]["tlm"] = time.time() - t1
    save()
    clear()
R["done"] = True
save()
print(json.dumps(R, indent=1, default=float))
