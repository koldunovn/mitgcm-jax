#!/usr/bin/env python3
"""The size of global_ocean.cs32x15/input_ad's programs (GO lane session 10, CS32AD; measurement script): one program
per process, lowered and compiled with the gate XLA flags while a sampler thread prints the process's memory mappings
and RSS, so a compile that dies leaves its last numbers in the log.

    python -u scripts/cs32_compile_study.py MODE OUT_DIR
    sbatch -J mjx_cs32_cs -p compute -A ab0995 --time=01:00:00 --mem=0 -N1 --wrap \
        "JAX_PLATFORMS=cpu PYTHONPATH=$PWD python -u scripts/cs32_compile_study.py grad $MJX_RUNS/go_dev/<new dir>"

MODE: grad     GenarrAdjoint(key = xx_theta)'s value_and_grad (monitor off), schedule "step"
      gradmon  the same with the adjoint monitor's stats hook (scripts/cs32_adjoint.py's grad phase)
      fwd      the cost fc(xx) (schedule "none")
      jvp      the tangent dfc.v
Prints a line per phase, a sampler line every 20 s (`/proc/self/maps` lines, VmRSS, VmHWM, threads), a faulthandler
dump on a fatal signal, and `RESULT {json}`; writes OUT_DIR/result.json (lower / compile time, mappings added by the
compile, fusions in the compiled HLO, peak RSS, the first and a warm run, fc and the gradient at the 4 grdchk points
vs TAF's `ADM adjoint_gradient`). vm.max_map_count is 65530 on Levante (PORTING_LESSONS "XLA:CPU compile aborts =
vm.max_map_count"). Refuses an OUT_DIR that already holds a result.
"""

import faulthandler
import gc
import json
import os
import re
import resource
import sys
import threading
import time
from pathlib import Path

faulthandler.enable(all_threads=True)
MODE, OUT = sys.argv[1], Path(sys.argv[2])
if (OUT / "result.json").exists():
    raise SystemExit(f"{OUT}/result.json exists: choose a new OUT_DIR (nothing is overwritten)")
OUT.mkdir(parents=True, exist_ok=True)

from mitjax.xla_flags import set_gate_xla_flags  # noqa: E402

set_gate_xla_flags()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

T0 = time.time()
PHASE = ["start"]
R = {"mode": MODE, "xla_flags": os.environ.get("XLA_FLAGS"), "samples": []}


def nmaps():
    with open("/proc/self/maps") as f:
        return sum(1 for _ in f)


def status():
    d = {}
    with open("/proc/self/status") as f:
        for ln in f:
            k, _, v = ln.partition(":")
            if k in ("VmRSS", "VmHWM", "Threads"):
                d[k] = int(v.split()[0])
    return d


def say(msg):
    print(f"[{time.time() - T0:8.1f} s] {msg}", flush=True)


def sampler():
    while True:
        s = status()
        n = nmaps()
        R["samples"].append((round(time.time() - T0, 1), PHASE[0], n, s.get("VmRSS", 0) // 1024))
        say(f"sample {PHASE[0]}: maps {n}  VmRSS {s.get('VmRSS', 0) / 2**20:.1f} GB  VmHWM "
            f"{s.get('VmHWM', 0) / 2**20:.1f} GB  threads {s.get('Threads')}")
        time.sleep(20)


def save():
    (OUT / "result.json").write_text(json.dumps(R, indent=1, default=float))


with open("/proc/sys/vm/max_map_count") as f:
    R["max_map_count"] = int(f.read())
R["rlimit_stack"] = resource.getrlimit(resource.RLIMIT_STACK)
say(f"mode {MODE}  max_map_count {R['max_map_count']}  RLIMIT_STACK {R['rlimit_stack']}  XLA_FLAGS {R['xla_flags']}")
threading.Thread(target=sampler, daemon=True).start()

from mitjax import paths  # noqa: E402
from mitjax.drivers.adjoint_run import GenarrAdjoint  # noqa: E402
from mitjax.io import stdout as so  # noqa: E402
from mitjax.pkg.ctrl.ctrl_readparms import ctrl_readparms_genarr  # noqa: E402
from mitjax.tests import goadk_gate as G  # noqa: E402
from mitjax.tests import goadk_model_gate as MG  # noqa: E402

EXPNAME = "global_ocean.cs32x15"
G.EXP = MG.EXP = EXPNAME
PHASE[0] = "setup"
m, _ = MG.model("input_ad", "jdon")
KEY = next((3, g.iarr) for g in ctrl_readparms_genarr(m.exp.run, 3, m.cfg) if g.file.strip() == "xx_theta")
a = GenarrAdjoint(m, monitor=(MODE == "gradmon"), key=KEY)
sz = m.cfg.size
pts, _, _ = G.grdchk_case("input_ad", maskC=np.asarray(m.grid.maskC.data))
P = [(r.itile - 1 + (r.jtile - 1) * sz.nSx, r.layer - 1, r.jtilepos - 1 + sz.OLy, r.itilepos - 1 + sz.OLx)
     for _, r in pts]
taf = so.grdchk(so.read_stdout(paths.UPSTREAM / "verification" / EXPNAME / "results" / "output_adm.txt"))
R.update(key=KEY, points=P, taf_adm=[p.adm["adjoint_gradient"] for p in taf.points], taf_fc=taf.fcref)
R["t_setup"] = time.time() - T0
say(f"model + GenarrAdjoint built in {R['t_setup']:.1f} s; key {KEY}; points {P}")
save()

args = (a.theta0, a.model, a.st0, a.xs)
if MODE in ("grad", "gradmon"):
    fn = a._vg
    if MODE == "gradmon":
        args = args + (a._sinks,)
elif MODE == "fwd":
    fn = a._J
else:
    v = jnp.asarray(np.random.default_rng(20261002).standard_normal(a.theta0.shape))
    fn, args = a._jvp, (a.theta0, v) + args[1:]

gc.collect()
PHASE[0] = "lower"
m0, t = nmaps(), time.time()
lowered = fn.lower(*args)
R["t_lower"] = time.time() - t
say(f"lower {R['t_lower']:.1f} s")
txt = lowered.as_text()
ops = re.findall(r"= (?:\"?)(stablehlo\.[a-z_]+|func\.call|call)\b", txt)
cnt = {}
for o in ops:
    cnt[o] = cnt.get(o, 0) + 1
R["stablehlo_ops"] = len(ops)
R["stablehlo_top"] = sorted(cnt.items(), key=lambda kv: -kv[1])[:25]
R["stablehlo_funcs"] = len(re.findall(r"^\s*func\.func ", txt, flags=re.M))
R["stablehlo_bytes"] = len(txt)
del txt
say(f"lowered module: {R['stablehlo_ops']} ops in {R['stablehlo_funcs']} funcs; top {R['stablehlo_top'][:8]}")
save()
PHASE[0] = "compile"
t = time.time()
comp = lowered.compile()
R["t_compile"] = time.time() - t
R["maps_delta"] = nmaps() - m0
R["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
ctxt = comp.as_text()
R["fusions"] = len(re.findall(r"= [^\n]*? fusion\(", ctxt))
R["hlo_instructions"] = sum(1 for ln in ctxt.splitlines() if " = " in ln)
del ctxt
say(f"compile {R['t_compile']:.1f} s  maps added {R['maps_delta']}  fusions {R['fusions']}  "
    f"peak RSS {R['peak_rss_gb']:.1f} GB")
save()
for k in ("run1", "run2"):
    PHASE[0] = k
    t = time.time()
    out = jax.block_until_ready(comp(*args))
    R[f"t_{k}"] = time.time() - t
    say(f"{k} {R[f't_{k}']:.2f} s")
if MODE in ("grad", "gradmon"):
    g = np.asarray(out[1])
    R["fc"] = float(out[0])
    R["grad"] = [float(g[p]) for p in P]
    R["admGrd_rel_taf"] = [abs(float(g[p]) - t) / abs(t) for p, t in zip(P, R["taf_adm"])]
    R["grad_finite"] = bool(np.all(np.isfinite(g)))
    np.save(OUT / "grad.npy", g)
elif MODE == "fwd":
    R["fc"] = float(out)
else:
    R["fc"], R["jvp"] = float(out[0]), float(out[1])
R["fc_rel_taf"] = abs(R["fc"] - taf.fcref) / abs(taf.fcref)
R["done"] = True
PHASE[0] = "done"
save()
print("RESULT " + json.dumps({k: v for k, v in R.items() if k != "samples"}, default=float), flush=True)
