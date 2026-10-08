"""Where the first call of the R2 gradient program spends its time (plan Task 17, lane SHARDGRAD): model build, trace,
lower, XLA compile, first and warm run, one process per XLA flag variant (XLA reads XLA_FLAGS once per process).

    python -u scripts/shardgrad/compile_study.py VARIANT [P]      (P = 1: the single-device program; P > 1: sharded)

VARIANT: gate        mitjax.xla_flags gate set (no FMA on CPU, algsimp disabled, 4 fake CPU devices)
         noalgsimp   the gate set without --xla_disable_hlo_passes=algsimp
         noalgsimp_at0  noalgsimp + --xla_gpu_autotune_level=0
         cputrace    the gate set, tracing under jax.default_device(cpu) (eager ops during tracing stay on the host)
Prints one line per phase (flushed) and a final `RESULT {json}`; a faulthandler stack dump every 240 s shows which phase
a long wait is in. Uses the default backend (GPU when present); the model is built on the host CPU as in
mitjax/tests/test_sharded_grad_gpu.py.
"""

import faulthandler
import json
import os
import sys
import time

VARIANT = sys.argv[1]
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 1
from mitjax.xla_flags import GATE_FLAGS  # noqa: E402  (stdlib only)

flags = list(GATE_FLAGS)
if VARIANT.startswith("noalgsimp"):
    flags = [f for f in flags if not f.startswith("--xla_disable_hlo_passes")]
if VARIANT == "noalgsimp_at0":
    flags.append("--xla_gpu_autotune_level=0")
os.environ["XLA_FLAGS"] = " ".join(flags)
faulthandler.dump_traceback_later(240, repeat=True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

T0 = time.time()


def say(msg):
    print(f"[{time.time() - T0:7.1f} s] {VARIANT} P={NP}: {msg}", flush=True)


say(f"XLA_FLAGS={os.environ['XLA_FLAGS']} devices={jax.devices()}")
from mitjax.tests import shardgrad_gate as G  # noqa: E402

res = {"variant": VARIANT, "P": NP, "xla_flags": os.environ["XLA_FLAGS"]}
t = time.time()
cpu = jax.devices("cpu")[0]
with jax.default_device(cpu):
    su = G.Setup(G.model())
res["build"] = time.time() - t
say(f"model built on the host in {res['build']:.1f} s")
gpus = [d for d in jax.devices() if d.platform != "cpu"] or jax.devices()
if NP == 1:
    f = su.p1_seed_fn()
    args = jax.device_put((su.theta, su.model, su.st0, su.xs), gpus[0]) + (jnp.float64(1.0),)
else:
    sd = G.Sharded(su, NP, devices=gpus[:NP])
    f, args = sd.vg, (sd.theta, sd.model, sd.st0, su.xs, G.SG.seed(1.0))
ctx = jax.default_device(cpu) if VARIANT == "cputrace" else jax.default_device(gpus[0])
t = time.time()
with ctx:
    tr = f.trace(*args)
res["trace"] = time.time() - t
say(f"trace {res['trace']:.1f} s")
t = time.time()
lo = tr.lower()
res["lower"] = time.time() - t
say(f"lower {res['lower']:.1f} s")
t = time.time()
co = lo.compile()
res["compile"] = time.time() - t
say(f"compile {res['compile']:.1f} s")
for k in ("run1", "run2"):
    t = time.time()
    out = jax.block_until_ready(co(*args))
    res[k] = time.time() - t
    say(f"{k} {res[k]:.2f} s, J = {float(out[0]):.17e}")
res["J"] = float(out[0])
faulthandler.cancel_dump_traceback_later()
print("RESULT " + json.dumps(res), flush=True)
