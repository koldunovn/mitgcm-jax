"""offline_exf_seaice/input.dyn_lsr end to end (M4 step 4, lane M4OFF session 3): the whole step at every dumped
stage, the 12-step run through the run driver, the restart, P=N == P=1. Helpers: mitjax/tests/m4off_lsr_gate.py and
the experiment-generic ones of m4off_gate.py."""

import pytest

from mitjax.tests import m4off_gate as G
from mitjax.tests import m4off_lsr_gate as L


@pytest.fixture(autouse=True)
def _release_executables():
    """Free the previous test's compiled programs (PORTING_LESSONS "vm.max_map_count"); changes no value."""
    import gc
    import jax
    jax.clear_caches()
    gc.collect()
    yield


def test_steps_0_2_every_dumped_stage_free_and_teacher_forced():
    """The driver's whole FORWARD_STEP (SEAICE_MODEL with SEAICE_DYNSOLVER's LSR, PPM advection, no thermodynamics)
    at iterations 0-2, free (chained from the Model's own initial carry) and teacher-forced: every dumped stage
    (25 stages, 543 fields per iteration, measured dev job 27870970) bitwise."""
    r = L.run()
    for teacher in (False, True):
        out = G.steps(r, its=(0, 1, 2), teacher=teacher)
        for it, (bad, ncmp) in out.items():
            assert not bad, (teacher, it, bad)
            assert len(ncmp) >= 25 and sum(ncmp.values()) >= 540, (teacher, it, ncmp)
            for st in ("Y02_ice_strength", "Y06_lsr", "I02_advdiff", "P13_seaice_model"):
                assert ncmp.get(st, 0) >= 6, (teacher, it, st, ncmp)


def test_whole_run_monitor_lsr_lines_digits_pickups():
    """The 12-step run through the run driver (as `python -m mitjax run`): every MONITOR record (the dynamics block
    at 0 and 12 (monitorFreq 864000 s: the first and last step), the SEAICE block every step (SEAICE_monFreq 1800 s),
    the EXF block at 0) and every SEAICE_LSR line (5 per pass, 20 passes, 12 steps: 1200) identical to the oracle's
    STDOUT, the testreport digits at the yardstick, pickup.ckptA and pickup_seaice.ckptA byte-identical, nothing on
    STDERR."""
    m, res, o, (diffs, rows, nblocks), (files, missing) = L.whole_run()
    print("offline_exf_seaice/input.dyn_lsr digits (name, ours vs results/, yardstick, ours vs oracle):", rows)
    assert diffs == {"mon": 0, "banner": 0}, (nblocks, diffs)
    k0 = next(n for n, r in enumerate(o.raw) if "Begin MONITOR dynamic field statistics" in r) - 1
    a, b = G.monitor_records(res.records), G.monitor_records(o.raw[k0:])
    assert a == b, (len(a), len(b), [(x, y) for x, y in zip(a, b) if x != y][:2])
    assert sum("%MON seaice_tsnumber" in r for r in a) == 13, sum("%MON seaice_tsnumber" in r for r in a)
    la, lb = L.lsr_records(res.records), L.lsr_records(o.raw)
    assert la == lb and len(la) == 1200, (len(la), len(lb), [(x, y) for x, y in zip(la, lb) if x != y][:2])
    assert rows and all(x >= y for _, x, y, _ in rows), rows
    pk = {f: ok for f, ok in files.items() if f.startswith("pickup")}
    assert sorted(pk) == ["pickup.ckptA.data", "pickup.ckptA.meta", "pickup_seaice.ckptA.data",
                          "pickup_seaice.ckptA.meta"], pk
    assert all(pk.values()), pk
    assert res.stderr == [], res.stderr[:5]


def test_restart_6_pickup_6_equals_12():
    """Run A: 12 steps with permanent pickups every 6; run B: nIter0 = 6 from A's pickup.0000000006 and
    pickup_seaice.0000000006 (uIce, vIce with the dynamics). B's pickup_seaice at 12 is byte-identical to A's and
    B's SEAICE_LSR lines equal A's last 6 steps. The ocean pickup differs in exactly one record, EtaH, and B's is 0:
    the Fortran's own restart behaviour (measured, dev job 27870970): INI_PSURF sets etaH = etaN on a cold start
    only (ini_psurf.F:78-87; etaN /= 0 here from pSurfInitFile eta_3c0.bin), READ_PICKUP reads EtaH only with
    nonlinFreeSurf > 0 (read_pickup.F:460-463; 0 here), INI_DYNVARS zeroes it (ini_dynvars.F:112) and nothing
    updates it without momStepping. input.thermo was blind to this (etaN = 0)."""
    import filecmp
    from pathlib import Path
    import numpy as np
    a, ra, b, rb = L.restart()
    f = f"pickup_seaice.{12:010d}.data"
    assert filecmp.cmp(Path(a.rundir) / f, Path(b.rundir) / f, shallow=False), f
    pa = np.fromfile(Path(a.rundir) / "pickup.0000000012.data", ">f8").reshape(7, -1)
    pb = np.fromfile(Path(b.rundir) / "pickup.0000000012.data", ">f8").reshape(7, -1)
    meta = (Path(a.rundir) / "pickup.0000000012.meta").read_text()
    import re
    names = re.findall(r"'([^']+)'", meta.split("fldList")[1].split("}")[0])
    assert [n.strip() for n in names] == ["Uvel", "Vvel", "Theta", "Salt", "EtaN", "dEtaHdt", "EtaH"], names
    diff = [k for k in range(7) if not np.array_equal(pa[k].view(np.int64), pb[k].view(np.int64))]
    assert diff == [6] and not pb[6].any() and np.array_equal(pa[6], pa[4]), diff     # EtaH; A: EtaH = EtaN
    la, lb = L.lsr_records(ra.records), L.lsr_records(rb.records)
    assert len(lb) == 600 and la[-600:] == lb


@pytest.mark.parametrize("nproc", (4, 2))
def test_sharded_equals_single(nproc):
    """P=4 (one tile per device) and P=2 (two tiles per device) == P=1: every leaf of the carry bitwise after 3
    steps (the LSR's running residual sums over all tiles in the oracle's order: eesupp all_tiles)."""
    r = L.run()
    d = G.sharded_vs_single(r, nproc=nproc, nsteps=3)
    assert not {k: v for k, v in d.items() if v}, d
