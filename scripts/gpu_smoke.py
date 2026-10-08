#!/usr/bin/env python3
"""M1 GPU smoke run (plan Task 18): global_ocean.90x40x15/input on ONE A100 (scripts/gpu_smoke.sbatch).

The Model is built on the host CPU backend exactly as on a CPU node (INI_PARMS, INITIALISE_FIXED, INITIALISE_VARIA,
the forcing preload), then its Arrays and initial carry are placed on the GPU and run.forward runs the whole run
(10 steps; the scan executes on the GPU; the host-side MONITOR / SBO / solver lines are computed from the GPU
results). Writes OUT/output.txt (the STDOUT records) and OUT/result.json (wall times: Model set-up, the compile of
the one-step scan program forward runs (lower + compile, timed alone), the whole run incl. its own compile;
finiteness and the carry leaves holding non-finite values); never deletes anything.

Comparison, defined BEFORE the first GPU run (session 6, docs/M1_ACCEPTANCE.md): pass if the run completes, every
%MON value is finite, and tools/testreport_jax.py gives >= 10 digits GPU vs our CPU run (the acceptance job's
output.txt) on every check-list variable; the measured digits (the CPU-vs-GPU spread of this run) are reported.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mitjax.xla_flags import set_gate_xla_flags  # noqa: E402

set_gate_xla_flags()


def main(out):
    import jax
    import numpy as np
    from mitjax import paths
    from mitjax.drivers.model import Model
    from mitjax.drivers.run import forward, load_experiment, make_rundir
    out = Path(out)
    out.mkdir(parents=True)
    exp, inp = "global_ocean.90x40x15", "input"
    exp_dir = paths.UPSTREAM / "verification" / exp
    cpu, gpu = jax.devices("cpu")[0], jax.devices("cuda")[0]
    res = {"gpu": str(gpu), "device_kind": gpu.device_kind}
    t0 = time.time()
    with jax.default_device(cpu):
        m = Model(load_experiment(exp_dir, inp), make_rundir(exp_dir, inp, out / "run"))
    res["setup_s"] = time.time() - t0
    m.arrays = jax.device_put(m.arrays, gpu)
    carry0 = jax.device_put(m.initial_carry(), gpu)
    m.initial_carry = lambda: carry0
    from mitjax.drivers.the_main_loop import _scan
    t, it = m.start_counters()
    tc = time.time()
    with jax.default_device(gpu):
        _scan.lower(m.step, m.arrays, carry0, t, it, 1, 1).compile()     # the one-step scan program forward runs
    res["compile_s"] = time.time() - tc
    t1 = time.time()
    with jax.default_device(gpu):
        r = forward(m, write_pickups=False, chunks=[1])          # chunk 1 alone: its wall time = compile + 1 step
    res["run_s"] = time.time() - t1
    res["chunk_lengths"] = r.chunk_lengths
    leaves = jax.tree.leaves(r.carry)
    res["carry_finite"] = bool(all(np.all(np.isfinite(np.asarray(x))) for x in leaves
                                   if np.issubdtype(np.asarray(x).dtype, np.floating)))
    res["nonfinite_leaves"] = {jax.tree_util.keystr(k): int(np.count_nonzero(~np.isfinite(np.asarray(x))))
                               for k, x in jax.tree_util.tree_flatten_with_path(r.carry)[0]
                               if np.issubdtype(np.asarray(x).dtype, np.floating)
                               and not np.all(np.isfinite(np.asarray(x)))}
    res["carry_platform"] = sorted({d.platform for x in leaves if hasattr(x, "devices") for d in x.devices()})
    (out / "output.txt").write_text("\n".join(r.records) + "\n")
    (out / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
