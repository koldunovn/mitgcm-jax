#!/usr/bin/env python3
"""Steady-state cost of one model step (plan Task 18: the tier-3 twin cost estimate).

    step_timing.py EXPERIMENT INPUT OUT_DIR [--steps N] [--gpu]

On the host CPU backend, or with --gpu on the first CUDA device (the Model built on the CPU backend, its Arrays and
initial carry placed on the GPU, as scripts/gpu_smoke.py).

Builds the run driver's Model, times the compile of the one-step scan program run.forward executes
(the_main_loop._scan, n = 1: lower + compile), then N calls of it from the initial carry (each blocked until
ready; no host MONITOR / pickup work), and writes OUT_DIR/result.json (setup_s, compile_s, step_s list, the
median, the carry leaves holding non-finite values after the steps). Run on a compute node
(scripts/m1_acceptance.sbatch PN=timing; scripts/gpu_smoke.sbatch timing). Never deletes anything.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mitjax.xla_flags import set_gate_xla_flags  # noqa: E402

set_gate_xla_flags()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("input")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args(argv)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mitjax import paths
    from mitjax.drivers.model import Model
    from mitjax.drivers.run import load_experiment, make_rundir
    from mitjax.drivers.the_main_loop import _scan
    out = Path(a.out)
    out.mkdir(parents=True)
    exp_dir = paths.UPSTREAM / "verification" / a.experiment
    cpu = jax.devices("cpu")[0]
    dev = jax.devices("cuda")[0] if a.gpu else cpu
    t0 = time.time()
    with jax.default_device(cpu):
        m = Model(load_experiment(exp_dir, a.input), make_rundir(exp_dir, a.input, out / "run"))
        carry = m.initial_carry()
    res = {"experiment": a.experiment, "input": a.input, "setup_s": time.time() - t0, "device": str(dev),
           "device_kind": dev.device_kind}
    m.arrays, carry = jax.device_put((m.arrays, carry), dev)
    t, it = jax.device_put(m.start_counters(), dev)
    tc = time.time()
    with jax.default_device(dev):
        exe = _scan.lower(m.step, m.arrays, carry, t, it, jnp.int32(1), 1).compile()
    res["compile_s"] = time.time() - tc
    steps = []
    for k in range(1, a.steps + 1):
        ts = time.time()
        carry, t, it, _ = exe(m.arrays, carry, t, it, jax.device_put(jnp.int32(k), dev))
        jax.block_until_ready(carry)
        steps.append(time.time() - ts)
    res["step_s"] = steps
    paths_ = jax.tree_util.tree_flatten_with_path(carry)[0]
    res["nonfinite_leaves"] = {jax.tree_util.keystr(k): int(np.count_nonzero(~np.isfinite(np.asarray(x))))
                               for k, x in paths_ if np.issubdtype(np.asarray(x).dtype, np.floating)
                               and not np.all(np.isfinite(np.asarray(x)))}       # after the timed steps
    res["step_median_s"] = float(np.median(steps[1:] if len(steps) > 1 else steps))
    (out / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
