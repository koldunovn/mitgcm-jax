"""lab_sea/input end to end (M4 step 5, lane M4LAB session 4): the 9-step run through the run driver, the SITRACER
gate (L3), the restart, P=N == P=1.

Oracle: lane A's registered run job27855987 of the build lab_sea-code-63cdc0b (dumps-on STDOUT and pickups,
identical to the plain run's: invisibility-job27855987.txt). Helpers: mitjax/tests/m4lab_gate.py.

1. The whole run (drivers.run.forward, as `python -m mitjax run`, nIter0 = 1 from pickup.0000000001,
   pickup_cd.0000000001, pickup_seaice.0000000001, 9 steps to endTime 36000 s): every MONITOR record (%MON lines and
   banners: the dynamics, SEAICE (incl. seaice_sitracer01/02) and EXF blocks) and every solver STDOUT line identical to
   the oracle's, the testreport digits at the yardstick (vs results/) and vs the oracle, pickup.0000000010,
   pickup_cd.0000000010 and pickup_seaice.0000000010 byte-identical.
2. SITRACER (L3) is blind in the dumps; its gate is (1): the %MON seaice_sitracer lines of every step and the siTrac01
   ('age', mate AREA) and siTrac02 ('one', mate HEFF) records of pickup_seaice.0000000010. Negative control:
   ALLOW_SITRACER_ADVCAP planted off changes them (measured, dev job of session 4).
3. Restart: run A (pChkptFreq = 5 steps: pickups at 5 and 10) vs B = A's pickups at 5 + 5 steps: 4 + pickup + 5 = 9;
   B's three pickups at 10 byte-identical to A's (and to the oracle's), B's STDOUT records from iteration 5 on equal
   A's except exf_uwind_del2 / exf_vwind_del2 at B's first step, which the Fortran's own restart prints differently
   (halos; the Fortran values, oracle binary job 27874017, are asserted).
4. P=4 (one tile of 10x8 per device) and P=2 == P=1, every leaf of the carry bitwise after 3 steps.
"""

import filecmp
from pathlib import Path

import pytest

from mitjax.tests import m4lab_gate as L

PICKUPS = ("pickup.0000000010", "pickup_cd.0000000010", "pickup_seaice.0000000010")


@pytest.fixture(autouse=True)
def _release_executables():
    """Free the previous test's compiled programs (PORTING_LESSONS "vm.max_map_count"); changes no value."""
    import gc
    import jax
    jax.clear_caches()
    gc.collect()
    yield


def _k0(o):
    return next(n for n, r in enumerate(o.raw) if "Begin MONITOR dynamic field statistics" in r) - 1


def test_whole_run_monitor_solver_digits_pickups():
    """The 9-step run: every MONITOR record and solver line identical to the oracle's STDOUT, the digits at the
    yardstick, the three pickups at iteration 10 byte-identical (incl. the siTrac records: the SITRACER gate)."""
    m, res, o, (diffs, rows, nblocks) = L.whole_run()
    print("lab_sea/input digits (name, ours vs results/, yardstick, ours vs oracle):", rows)
    a, b = L.monitor_records(res.records), L.monitor_records(o.raw[_k0(o):])
    assert diffs == {"mon": 0, "banner": 0} and nblocks == 10, (diffs, nblocks)
    assert a == b, (len(a), len(b), [(x, y) for x, y in zip(a, b) if x != y][:2])
    assert sum("%MON seaice_tsnumber" in r for r in a) == 10
    assert sum("%MON seaice_sitracer" in r for r in a) == 10*2*5           # 10 blocks x 2 tracers x 5 statistics
    sa, sb = L.solver_records(res.records), L.solver_records(o.raw[_k0(o):])
    assert sa == sb and len(sa) > 0, (len(sa), len(sb), [(x, y) for x, y in zip(sa, sb) if x != y][:2])
    assert rows and all(x >= y for _, x, y, _ in rows), rows
    assert all(s >= y for _, _, y, s in rows), rows
    od = o.stdout_path.parent
    for pre in PICKUPS:
        for suf in ("data", "meta"):
            f = f"{pre}.{suf}"
            assert filecmp.cmp(Path(m.rundir) / f, od / f, shallow=False), f
    recs = L.pickup_records(od / "pickup_seaice.0000000010.data")
    assert {"siTrac01", "siTrac02"} <= set(recs), sorted(recs)


def test_sitracer_negative_control_advcap():
    """ALLOW_SITRACER_ADVCAP planted off (no cap of the advected tracer by its pre-advection neighbourhood
    maximum): the %MON seaice_sitracer records and the siTrac records of pickup_seaice.0000000010 differ."""
    m, res, o, _ = L.whole_run("m4lab-ctrl-advcap", plant_off=("ALLOW_SITRACER_ADVCAP",))
    a, b = L.monitor_records(res.records), L.monitor_records(o.raw[_k0(o):])
    nd = sum(x != y for x, y in zip(a, b) if "sitracer" in y)
    ra = L.pickup_records(Path(m.rundir) / "pickup_seaice.0000000010.data")
    rb = L.pickup_records(o.stdout_path.parent / "pickup_seaice.0000000010.data")
    print("ADVCAP off: differing sitracer records", nd, {k: ra[k] == rb[k] for k in rb})
    assert nd > 0
    assert ra["siTrac01"] != rb["siTrac01"] or ra["siTrac02"] != rb["siTrac02"]


def test_restart_4_pickup_5_equals_9():
    """Run A: 9 steps with pChkptFreq = 5*deltaT (pickups at 5 and 10); run B: startTime = 5*deltaT (nIter0 = 5) to
    endTime from A's pickup.0000000005, pickup_cd.0000000005, pickup_seaice.0000000005 (siTrac01/02 present now). B's
    three pickups at 10 byte-identical to A's and to the oracle's; B's STDOUT records from iteration 5's SEAICE block
    on equal A's except the two halo-dependent EXF del2 lines of B's first step, which equal the Fortran restart's."""
    a, ra, b, rb = L.restart_b("restartB")
    assert ra.pickups == ["pickup.0000000005", "pickup.0000000010"] and rb.pickups == ["pickup.0000000010"], \
        (ra.pickups, rb.pickups)
    from mitjax import paths
    od = paths.REFERENCE_RUNS / L.EXP[0] / L.EXP[1] / L.JDON / "rundir"
    for pre in PICKUPS:
        for suf in ("data", "meta"):
            f = f"{pre}.{suf}"
            assert filecmp.cmp(Path(a.rundir) / f, Path(b.rundir) / f, shallow=False), f
            assert filecmp.cmp(Path(b.rundir) / f, od / f, shallow=False), f

    def tail(recs):
        k = next(n for n, r in enumerate(recs) if "%MON seaice_tsnumber" in r and r.split("=")[1].strip() == "5")
        return [r for r in recs[k:] if "%CHECKPOINT" not in r]
    ta, tb = tail(ra.records), tail(rb.records)
    diff = [(x, y) for x, y in zip(ta, tb) if x != y]
    assert len(ta) == len(tb) and len(ta) > 100, (len(ta), len(tb))
    # The Fortran's own restart prints these two differently too (measured with the oracle binary, job 27874017:
    # A 8.0270959439526E-02 / 4.8609272056854E-02, B 1.6626227332346E-01 / 1.5093881355919E-01 at exf_tsnumber 5):
    # MON_STATS_RL's del2 reads the halos of uwind/vwind (mon_stats_rl.F:76-81), which EXF_GETFORCING does not exchange
    # and SEAICE_MODEL's EXCH_UV_AGRID_3D_RL (seaice_model.F:127) fills only from the first step on
    # (docs/ISSUES_UPSTREAM.md). Our B prints the Fortran B's values; every other record is A's.
    assert [(x.split("=")[0].split()[-1], y.split("=")[1].strip()) for x, y in diff] == [
        ("exf_uwind_del2", "1.6626227332346E-01"), ("exf_vwind_del2", "1.5093881355919E-01")], diff


@pytest.mark.parametrize("nproc", [4, 2])
def test_sharded_equals_single(nproc):
    """3 steps of the driver's step (the whole carry: State, FFIELDS.h, phi0surf, flow, KPP / GMREDI / SALT_PLUME,
    the sea-ice incl. SEAICE_TRACER.h and the EXF state) at P=nproc == P=1, every leaf bitwise."""
    m, _ = L.stub_model()
    assert L.sharded_vs_single(m, nproc, 3) == {}
