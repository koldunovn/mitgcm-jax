"""The R5 optim adjoint on GPUs (plan Task 17): `p1` -- the single-device program on one A100, `pN` -- the sharded
program (drivers/sharded_grad.py) on P A100s, `compare` -- the two saved runs against their measured repeat floors.

    python scripts/r5_shardgrad/gpu.py p1 OUT            (1 GPU)
    python scripts/r5_shardgrad/gpu.py pN OUT [P]         (P GPUs, default 4)
    python scripts/r5_shardgrad/gpu.py compare P1_OUT PN_OUT

Each program is lowered and compiled explicitly first (times printed), then run twice with seed 1 (the repeat floor:
a GPU run need not be bitwise reproducible), once with seed 0 and once with seed 2 (the linearity guard). The Model
is built on the host CPU and placed on the GPUs. Saves OUT/run.npz (J, gradient of each run) and OUT/run.json."""

import json
import sys

# The deterministic order class of the sharded vs single-device gradient program, measured on CPU where both are
# deterministic (dev job 27838618, test_r5_sharded_grad_cpu.py: P=4 vs single device 7.66e-15, max |d| / max |g|).
# GPU criterion (Nikolay 2026-10-02): P=N vs P=1 within floor(P=1) + floor(P=N) + this class.
ORDER_CLASS = 7.66e-15
import time
from pathlib import Path

import numpy as np


def _run(mode, out, nproc=4):
    from mitjax.xla_flags import set_gate_xla_flags
    set_gate_xla_flags()
    import jax
    from mitjax.drivers import sharded_grad as SG
    from mitjax.tests import r5_shardgrad as RS
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    cpu = jax.devices("cpu")[0]
    gpus = jax.devices("gpu")
    print("devices", gpus, flush=True)
    t0 = time.time()
    with jax.default_device(cpu):
        su = RS.Setup(f"gpu-{mode}")
    info = {"mode": mode, "setup_s": round(time.time() - t0, 1)}
    if mode == "p1":
        put = lambda x: jax.device_put(x, gpus[0])       # noqa: E731
        args = [put(a) for a in (su.theta, su.model, su.st0, su.xs)]
        f = SG.value_and_grad_seed_fn(su.step, **su.kw())

        def call(seed):
            J, g = f(*args, put(SG.seed(seed)))
            J, g = jax.block_until_ready((J, g))
            return float(J), np.asarray(g.data)
        lower_args = args + [put(SG.seed(1.0))]
    else:
        sd = RS.Sharded(su, nproc, devices=gpus[:nproc])
        f = sd.vg

        def call(seed):
            J, g = jax.block_until_ready(f(sd.theta, sd.model, sd.st0, su.xs, SG.seed(seed)))
            return float(J), np.asarray(g.data)[:sd.sh.layout.nTiles]
        lower_args = [sd.theta, sd.model, sd.st0, su.xs, SG.seed(1.0)]
        info["nproc"] = nproc
    t0 = time.time()
    lowered = f.lower(*lower_args)
    info["lower_s"] = round(time.time() - t0, 1)
    print("lowered", info["lower_s"], flush=True)
    t0 = time.time()
    lowered.compile()
    info["compile_s"] = round(time.time() - t0, 1)
    print("compiled", info["compile_s"], flush=True)
    runs = []
    for k in range(2):
        t0 = time.time()
        J, g = call(1.0)
        runs.append((J, g))
        info[f"run{k}_s"] = round(time.time() - t0, 1)
        print("run", k, info[f"run{k}_s"], J, flush=True)
    (J0, g0), (J2, g2) = call(0.0), call(2.0)
    a, b = runs[0][1], runs[1][1]
    info["J"] = [runs[0][0], runs[1][0]]
    info["floor"] = float(np.max(np.abs(a - b)) / np.max(np.abs(a)))
    info["finite"] = bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))
    info["seed0_zero"] = bool(np.all(g0 == 0.0))
    info["seed2_rel"] = float(np.max(np.abs(g2 - 2.0 * a)) / np.max(np.abs(2.0 * a)))
    from mitjax.tests import r5_gate as r5
    info["taf_rel"] = [float(abs(a[p] - v) / abs(v)) for p, v in zip(r5.GRDCHK_POINTS, r5.TAF_ADJOINT_GRADIENT)]
    np.savez(out / "run.npz", g0=a, g1=b, J=np.array(info["J"]))
    (out / "run.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


def _compare(p1, pn):
    a, b = np.load(Path(p1) / "run.npz"), np.load(Path(pn) / "run.npz")
    i1, iN = (json.loads((Path(d) / "run.json").read_text()) for d in (p1, pn))
    rel = float(np.max(np.abs(b["g0"] - a["g0"])) / np.max(np.abs(a["g0"])))
    bound = i1["floor"] + iN["floor"] + ORDER_CLASS
    res = {"J_p1": i1["J"], "J_pN": iN["J"], "rel_pN_vs_p1": rel, "floor_p1": i1["floor"], "floor_pN": iN["floor"],
           "order_class": ORDER_CLASS, "bound": bound, "within": rel <= bound}
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        _compare(sys.argv[2], sys.argv[3])
    else:
        _run(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
