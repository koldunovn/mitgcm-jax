"""global_ocean.cs32x15/input.seaice end to end (M4 step 6, lane M4CS32ICE session 2): the 10 steps from
pickup.0000036000 / pickup_seaice.0000036000 through the run driver (as `python -m mitjax run`) against lane A's FTZ
oracle run job27878831 (plan decision 13; session 3), a restart, and P=6 / P=2 == P=1.

* every %MON record and MONITOR banner (11 blocks, incl. the SEAICE monitor; exf_monFreq 0: no EXF block) identical
  to the oracle STDOUT; testreport digits vs the oracle 16 (22 = both zero) and vs results/ at least the yardstick;
  the SEAICE_LSR and CG2D lines identical; every pickup byte-identical to the FTZ oracle's. Measured against the
  standard oracle (gfortran keeps subnormals): pickup_seaice.ckptA's siUICE / siVICE differ at exactly the points
  where the FTZ oracle's do (session 2: 77 / 44, ours ~2.3e-308 where the standard oracle has 0.), nothing else;
* restart: 5 steps + pickup + 5 steps == 10 steps (every end-of-run pickup byte-identical);
* P=6 (two tiles per device) and P=2 (six) == P=1 after 3 steps on every carry leaf and every replicated output.
Costs: about 30 min on a CPU compute node."""

import os
import subprocess
import sys

import numpy as np
import pytest

from mitjax import paths
from mitjax.tests import m4cs32ice_gate as G


@pytest.fixture(autouse=True)
def _free_compiled_programs():
    """Every test here compiles whole-run programs: drop all compiled programs before each (PORTING_LESSONS "XLA:CPU
    compile aborts = vm.max_map_count"; gate 27878249 aborted in backend_compile_and_load at
    test_control_clip_off_run, after the exf / seaice files and the whole run and restart in the same process)."""
    import gc
    import jax
    jax.clear_caches()
    gc.collect()
    yield


def test_whole_run_monitor_digits_solver_pickups():
    from mitjax.tests import advect_gate as ag
    from mitjax.tests import m4lab_gate as L
    m, res, o, (diffs, rows, nblocks) = G.whole_run()
    assert nblocks == 11 and diffs == {"mon": 0, "banner": 0}, (nblocks, diffs)
    assert len(rows) == 25 and all(a >= y and s >= 16 for _, a, y, s in rows), rows
    k0 = next(n for n, r in enumerate(o.raw) if "Begin MONITOR dynamic field statistics" in r) - 1
    ours, ref = G.solver_lines(res.records), G.solver_lines(o.raw[k0:])
    assert len(ours) == len(ref) >= 100 and ours == ref
    pd, extra = ag.pickup_diffs(m, o)
    assert extra == [] and len(pd) >= 6 and all(pd.values()), (pd, extra)      # byte-identical to the FTZ oracle
    assert res.stderr == [], res.stderr[:5]
    # measurement (decision 13): against the standard oracle's end pickups our pickup_seaice differs at exactly the
    # points (siUICE / siVICE) where the FTZ oracle's does; every other pickup is byte-identical to both
    std = paths.REFERENCE_RUNS / G.EXP[0] / G.EXP[1] / G.JDON_STD / "rundir"
    a = L.pickup_records(m.rundir / "pickup_seaice.ckptA.data")
    b = L.pickup_records(o.stdout_path.parent / "pickup_seaice.ckptA.data")
    c = L.pickup_records(std / "pickup_seaice.ckptA.data")
    nd = {n: (G.bits_differ(np.frombuffer(a[n], ">f8"), np.frombuffer(c[n], ">f8")),
              G.bits_differ(np.frombuffer(b[n], ">f8"), np.frombuffer(c[n], ">f8"))) for n in c}
    assert {n for n, (x, y) in nd.items() if x or y} == {"siUICE", "siVICE"}, nd
    assert all(x == y for x, y in nd.values()), nd
    assert (std / "pickup.ckptA.data").read_bytes() == (m.rundir / "pickup.ckptA.data").read_bytes()


def test_restart_5_pickup_5_equals_10():
    import filecmp
    a, ra, b, rb = G.restart_pair()
    for pre in ("pickup", "pickup_seaice"):
        for suf in ("data", "meta"):
            f = f"{pre}.{36010:010d}.{suf}"
            assert filecmp.cmp(a.rundir / f, b.rundir / f, shallow=False), f


PN = r"""
import os, sys
N = int(sys.argv[1])
from mitjax.xla_flags import gate_xla_flags
os.environ["XLA_FLAGS"] = gate_xla_flags("").replace("device_count=4", f"device_count={N}")
import jax
from mitjax.eesupp import exch_maps as EM
from mitjax.tests import m4cs32ice_gate as G
assert len(jax.devices()) == N
bad, od, nleaves, nout = G.sharded_vs_single(G.run().m, EM.load_cube_maps(*G.EXP), N, 3)
print("PN", nleaves, len(bad), nout, len(od), bad, od)
"""


def _pn(n):
    env = dict(os.environ)
    env.pop("XLA_FLAGS", None)
    env["PYTHONPATH"] = str(paths.REPO)
    r = subprocess.run([sys.executable, "-c", PN, str(n)], env=env, capture_output=True, text=True, timeout=2400)
    assert r.returncode == 0, r.stderr[-3000:]
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("PN ")][-1]
    # carry leaves compared, of them differing; replicated outputs compared, of them differing (gate 27878249 read the
    # differing-leaf dict's length as the number compared: P=6 / P=2 were bitwise, dev jobs 27878974 / 27878972)
    nleaves, nbad, nout, nod = (int(t) for t in line.split()[1:5])
    assert nleaves > 50 and nbad == 0 and nout > 0 and nod == 0, line


def test_p6_equals_p1():
    _pn(6)


def test_p2_equals_p1():
    _pn(2)


def test_control_clip_off_run():
    """Planted: SEAICE_clipVelocities = .FALSE. (data.seaice sets .TRUE.) for the first 5 steps: the %MON records
    differ from the oracle's, first at 36004, where the clip binds (seaice_uice_max = 0.4 in the oracle)."""
    from mitjax.drivers.run import forward
    from mitjax.tests import monitor_gate as mg
    m = G.run_dir_model("clipoff", {("data.seaice", "SEAICE_PARM01", "SEAICE_clipVelocities"): False,
                                    ("data", "PARM03", "nTimeSteps"): 5})
    res = forward(m)
    o = mg.oracle(*G.EXP, G.ORACLE_KIND)
    k0 = next(n for n, r in enumerate(o.raw) if "Begin MONITOR dynamic field statistics" in r) - 1
    a = [r for r in res.records if "%MON " in r]
    b = [r for r in o.raw[k0:] if "%MON " in r][:len(a)]
    bad = [x for x, y in zip(a, b) if x != y]
    assert bad, "clip off: no %MON record differs"
    first = next(k for k, (x, y) in enumerate(zip(a, b)) if x != y)
    ts = [r for r in a[:first + 1] if "time_tsnumber" in r][-1]
    assert int(ts.split("=")[1].split()[0]) >= 36004, ts
