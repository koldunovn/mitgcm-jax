"""lab_sea/input with one tile of 20 x 16 instead of lab_sea/code's 2 x 2 tiles of 10 x 8 (docs plan 20261006 S4,
lane LSTILE). Helpers: mitjax/tests/lstile_gate.py.

The Fortran's result depends on the tiling here: SEAICE_LSR solves its tridiagonal lines within a tile, the line ends
taking the neighbour tile's iterate from the halo (pkg/seaice/seaice_lsr.F:2001-2002 in SEAICE_LSR_TRIDIAGU, the
same in TRIDIAGV; one EXCH_UV_XY_RL per sweep, :987; SEAICE_GLOBAL_3DIAG_SOLVER is #undef in
lab_sea/code/SEAICE_OPTIONS.h:117), and the sweeps stop at LSR_ERROR = 1.E-4 (input/data.seaice:29;
seaice_lsr.F:955), so another tiling stops at another iterate. The retile oracle (build
lab_sea-code-63cdc0b-07ce3bc-retile, run job27944614-retile) differs from results/ and from the four-tile oracle by
the same testreport digits (PS 2, T 8-9, S 11-12, U 3-5, V 2-6, sea ice 6-7). The tiling also changes what MONITOR
prints: the dynstat_sst/sss lines only with nSx = nSy = 1 (pkg/monitor/monitor.F:122-131).

1. mitjax's one-tile run (mitjax.load on a copy with the retile SIZE.h, Experiment.run) against the retile oracle with
   lab_sea's four-tile forward-gate criterion (test_m4lab_run::test_whole_run_monitor_solver_digits_pickups): every
   MONITOR record and solver line identical, the testreport digits those of identical STDOUTs, pickup.0000000010,
   pickup_cd.0000000010 and pickup_seaice.0000000010 byte-identical.
2. Negative control: the same run with lab_sea/code's own SIZE.h (four tiles) fails every part of (1).
3. The Fortran's own tiling dependence, read from the two oracle runs (no model run): the digits above.
Skipped without the oracle runs.
"""

import gc

import pytest

from mitjax.tests import lstile_gate as G

ORACLE_1 = G.oracle_dir(G.RETILE_RUN)
ORACLE_4 = G.oracle_dir(G.FOUR_TILE_RUN)
pytestmark = pytest.mark.skipif(ORACLE_1 is None or ORACLE_4 is None,
                                reason=f"oracle runs {G.RETILE_RUN} / {G.FOUR_TILE_RUN} not available")

# testreport digits (results/ order) of the retile oracle against the four-tile oracle (= against results/): measured,
# job27944614-retile vs job27855987-plain (the four-tile oracle reproduces results/ at 16 digits)
FORTRAN_TILING_DIGITS = {"PS": 2, "Tmn": 8, "Tmx": 9, "Tav": 9, "Tsd": 8, "Smn": 12, "Smx": 11, "Sav": 11, "Ssd": 8,
                         "Umn": 3, "Umx": 5, "Uav": 3, "Usd": 4, "Vmn": 6, "Vmx": 5, "Vav": 2, "Vsd": 4,
                         "aSImn": 22, "aSImx": 7, "aSIav": 7, "aSIsd": 6, "hSImn": 22, "hSImx": 7, "hSIav": 7,
                         "hSIsd": 6}


@pytest.fixture(autouse=True)
def _release_executables():
    """Free the previous test's compiled programs (PORTING_LESSONS "vm.max_map_count"); changes no value."""
    import jax
    jax.clear_caches()
    gc.collect()
    yield


def _report(plant=None, tag="one-tile"):
    run, warns, exp = G.run_mitjax(G.out_dir(tag), plant=plant)
    return G.report(run.output, run.rundir, ORACLE_1, exp.exp_dir), warns, exp, run


def test_one_tile_equals_retile_oracle():
    r, warns, exp, run = _report()
    same, _ = G.digits(ORACLE_1 / "output.txt", ORACLE_1 / "output.txt", exp.exp_dir)
    print("one tile vs retile oracle:", r["mon"][:3], r["solver"][:3], r["summary"], "warnings:", warns)
    assert r["mon"][2] == 0 and r["mon"][0] == r["mon"][1] > 0, r["mon"]
    assert sum("dynstat_sst_mean" in x for x in G.monitor_records(G.lines(run.output))) == 10   # nSx = nSy = 1
    assert r["solver"][2] == 0 and r["solver"][0] == r["solver"][1] > 0, r["solver"]
    assert r["digits"] == same, (r["digits"], same)
    assert {f: [n for n, v in rows.items() if not v[2]] for f, rows in r["pickups"].items()} == \
        {f: [] for f in G.PICKUPS}, r["pickups"]


def test_negative_control_four_tile_size_h():
    """lab_sea/code's own SIZE.h (four tiles) against the retile oracle: (1) must fail on every criterion."""
    r, _, _, _ = _report(plant=G.restore_four_tile_size_h, tag="ctrl-four-tile")
    print("four tiles vs retile oracle:", r["mon"][:3], r["solver"][:3], r["summary"])
    assert r["mon"][2] > 0 or r["mon"][0] != r["mon"][1], r["mon"]
    assert r["solver"][2] > 0, r["solver"]
    assert min(r["digits"].values()) < 16, r["digits"]
    assert all(not all(v[2] for v in rows.values()) for rows in r["pickups"].values()), r["pickups"]


def test_fortran_tiling_dependence():
    """The two oracle runs differ by FORTRAN_TILING_DIGITS (no model run)."""
    from mitjax import paths
    exp_dir = paths.UPSTREAM / "verification" / G.EXP
    d1, _ = G.digits(ORACLE_1 / "output.txt", ORACLE_4 / "output.txt", exp_dir)
    d4, _ = G.digits(ORACLE_4 / "output.txt", exp_dir / "results" / "output.txt", exp_dir)
    assert d1 == FORTRAN_TILING_DIGITS, d1
    assert all(v >= 16 for v in d4.values()), d4
    p = G.pickup_diffs(ORACLE_1, ORACLE_4)
    assert not all(v[2] for v in p["pickup_seaice.0000000010"].values())
