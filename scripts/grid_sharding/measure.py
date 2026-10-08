"""Measurement behind the grid-sharding decision (lane B, 2026-10-01): host build once + placement vs. building the
grid as one compiled program (what a per-device build needs). Run on a compute node (scripts/grid_sharding/run.sbatch).

Per ≥4-tile M1 variant: eager host build time (cold, warm), put_tree time on P=4 fake CPU devices and bytes placed,
then the grid builder under jax.jit on one device: trace+compile+run time (alarm-limited) and the number of
differing points vs the eager build (bit patterns)."""

import json
import signal
import sys
import time

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

import jax  # noqa: E402
import numpy as np  # noqa: E402

from mitjax.eesupp.exch_maps import load_maps  # noqa: E402
from mitjax.eesupp.shard import TileSharding  # noqa: E402
from mitjax.farray import FArray  # noqa: E402
from mitjax.tests import grid_gate as gg  # noqa: E402

VARIANTS = [("tutorial_baroclinic_gyre", "input"), ("global_ocean.90x40x15", "input"),
            ("tutorial_global_oce_optim", "input_ad")]
JIT_LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 600


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def leaves(tree):
    return jax.tree.leaves(tree, is_leaf=lambda x: isinstance(x, FArray))


def ndiff_bits(a, b):
    n = 0
    for x, y in zip(leaves(a), leaves(b)):
        xd = np.asarray(x.data if isinstance(x, FArray) else x, np.float64)
        yd = np.asarray(y.data if isinstance(y, FArray) else y, np.float64)
        n += int(np.count_nonzero(xd.view(np.int64) != yd.view(np.int64)))
    return n


out = {"devices": [str(d) for d in jax.devices()]}
for exp, inp in VARIANTS:
    r = {}
    t = time.perf_counter()
    g = jax.block_until_ready(gg.build_grid(exp, inp))
    r["eager_cold_s"] = time.perf_counter() - t
    t = time.perf_counter()
    g = jax.block_until_ready(gg.build_grid(exp, inp))
    r["eager_warm_s"] = time.perf_counter() - t
    r["bytes"] = int(sum(np.asarray(x.data if isinstance(x, FArray) else x).nbytes for x in leaves(g)))
    r["n_leaves"] = len(leaves(g))
    sh = TileSharding(load_maps(exp), nproc=4)
    t = time.perf_counter()
    g4 = jax.block_until_ready(sh.put_tree(g))
    r["put_tree_P4_s"] = time.perf_counter() - t
    r["put_tree_ndiff_bits"] = ndiff_bits(sh.unpad_tree(g4), g)
    params = gg.grid_params(exp, inp)
    ex = gg.exchanger(exp)
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(JIT_LIMIT)
    t = time.perf_counter()
    try:
        gj = jax.block_until_ready(jax.jit(lambda: gg.build_grid(exp, inp, params=params, ex=ex))())
        r["jit_build_s"] = time.perf_counter() - t
        r["jit_ndiff_bits"] = ndiff_bits(gj, g)
    except Timeout:
        r["jit_build_s"] = f">{JIT_LIMIT} (stopped by alarm)"
    except Exception as err:  # noqa: BLE001
        r["jit_build_s"] = time.perf_counter() - t
        r["jit_error"] = f"{type(err).__name__}: {str(err)[:400]}"
    finally:
        signal.alarm(0)
    out[f"{exp}/{inp}"] = r
    print(json.dumps({f"{exp}/{inp}": r}), flush=True)
print("RESULT " + json.dumps(out), flush=True)
