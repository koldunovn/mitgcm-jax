"""global_ocean.cs32x15/input (plan Task 25, M2), GO lane session 8: the whole run through the run driver and the
sharded step.

* the whole run (20 steps, python -m mitjax run's Model and run.forward; cube_run_gate.whole_run links what the
  experiment's prepare_run links): every %MON record and MONITOR banner identical to the oracle STDOUT (21 blocks),
  testreport digits 16 vs results/ (= the yardstick) and 16 vs the oracle on every check-list variable, the end-of-run
  pickup files (12 tiles, data and meta) byte-identical, nothing on STDERR; the oracle's PHrefC / PHrefF / RhoRef
  (SET_REF_STATE's output files) are not written by the driver;
* P=6 == P=1 (subprocess with 6 fake CPU devices, one cube face per device; cube maps of exch_maps.load_cube_maps):
  every carry leaf (incl. the GM/Redi pkg carry) and every per-step output bit for bit after 3 steps.
Costs: about 10 min on a CPU compute node.
"""

import os
import subprocess
import sys

from mitjax import paths
from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

EXP = ("global_ocean.cs32x15", "input")


def test_whole_run_monitor_digits_pickups():
    from mitjax.tests import advect_gate as ag
    from mitjax.tests import cube_run_gate as C
    m, res, o = C.whole_run(*EXP, tag="go-whole")
    diffs, rows, nblocks = ag.run_verdict(*EXP, res, o)
    print("cs32x15 digits (name, ours vs results/, yardstick, ours vs oracle):", rows)
    assert nblocks == 21 and diffs == {"mon": 0, "banner": 0}, (nblocks, diffs)
    assert len(rows) == 17 and all(a >= y and a == 16 and s == 16 for _, a, y, s in rows), rows
    files, missing = C.output_file_diffs(m, o)
    pk = {f: ok for f, ok in files.items() if f.startswith("pickup")}
    assert len(pk) == 24 and all(pk.values()), pk
    assert sorted(missing) == sorted(f"{n}.{x}" for n in ("PHrefC", "PHrefF", "RhoRef") for x in ("data", "meta"))
    assert res.stderr == [], res.stderr[:5]


P6 = r"""
import os
from mitjax.xla_flags import gate_xla_flags
os.environ["XLA_FLAGS"] = gate_xla_flags("").replace("device_count=4", "device_count=6")
import jax
import numpy as np
from mitjax.tests import cube_run_gate as C
from mitjax.tests import go_gate as G
assert len(jax.devices()) == 6
r = C.cube_run("global_ocean.cs32x15", "input")
c1, o1 = G.run_steps_p(r.m, 3)
c6, o6 = G.run_steps_p(r.m, 3, 6, maps=C._maps(r))
def bits(x):
    a = np.asarray(x)
    return a.dtype, a.shape, np.ascontiguousarray(a).tobytes()
la, lb = jax.tree.leaves(c1), jax.tree.leaves(c6)
same = all(jax.tree.structure(p) == jax.tree.structure(q) and all(bits(x) == bits(y) for x, y in zip(
    jax.tree.leaves(p), jax.tree.leaves(q))) for p, q in zip(o1, o6))
print("P6", len(la), len(lb), sum(bits(a) != bits(b) for a, b in zip(la, lb)), same)
"""


def test_p6_equals_p1():
    env = dict(os.environ)
    env.pop("XLA_FLAGS", None)
    env["PYTHONPATH"] = str(paths.REPO)
    r = subprocess.run([sys.executable, "-c", P6], env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("P6 ")][-1].split()
    assert int(line[1]) == int(line[2]) > 100 and line[3] == "0" and line[4] == "True", line
