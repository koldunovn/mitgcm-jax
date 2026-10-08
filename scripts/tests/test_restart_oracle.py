"""The Fortran oracle's own restart (lane A session 5): tutorial_barotropic_gyre run A (10 steps, permanent pickups
at 5 and 10) vs run B (restart from A's pickup.0000000005: nIter0=5, nTimeSteps=5), both with the jaxdump2 binary,
dumps on at the start iterations 5-9 (reference/jobs/restart_runs.sbatch, job 27831497; registered as kinds restart_a,
restart_b: gate fixtures with namelist overlays, never yardsticks).

Measured (reference/restart_compare.py; the table below is the measurement, frozen):
* end of step 10 (S06_dynamics of iteration 9 for gU, gV, guNm1, gvNm1; S16 for the rest): every dumped field
  bitwise on every point except 180 halo points each of gU, gV, guNm1, gvNm1 (interiors bitwise);
* cause: READ_PICKUP exchanges GuNm1/GvNm1 (read_pickup.F:556, EXCH_UV_3D_RL; exch1 single tile: a periodic copy of
  the interior), while in a running model nothing ever writes their halos (+0 from INI_DYNVARS): B's guNm1/gvNm1 at
  S00_begin of iteration 5 equal the periodic copy of their interior at every point, A's halos are +0, and the 180
  points are exactly the halo points whose periodic source is nonzero. ADAMS_BASHFORTH2 runs over the halos
  (adams_bashforth2.F, i=1-OLx..sNx+OLx) and MOM_FLUXFORM/TIMESTEP do not overwrite them, so the difference moves
  between guNm1 and gU every step and never reaches the interior;
* transient: uVel/vVel differ at 59 halo points after MOMENTUM_CORRECTION_STEP (S10, S11), gone after the end-of-step
  exchanges (S16); at iteration 5 B has not yet computed gU, gV (interior +0 at S00/S05), phiHydLow, totPhiHyd,
  rhoInSitu, surfaceForcingU (not in the pickup; recomputed before use);
* STDOUT: the MONITOR block of iteration 5 differs only in trAdv_CFL_u/v/w_max (B prints MON_INIT's 0, mon_init.F);
  blocks 6-10 and the solver lines of steps 6-10 identical; pickup.0000000010 (data, meta) and every other file both
  runs write byte-identical.
Negative controls: a planted halo change and a planted interior change in B's dumps change the table; a planted
halo value breaks the periodic-copy identity.
Oracle-dependent: fails (never skips) while the registered runs are missing.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rr = _load("_restart_reference_runs", "reference/reference_runs.py")
rc = _load("_restart_compare", "reference/restart_compare.py")
dump = rc.dump

EXP, INP = "tutorial_barotropic_gyre", "input"
ITERS = (5, 6, 7, 8, 9)
G4 = ("gU", "gV", "guNm1", "gvNm1")


def expected_table():
    """{(iter, stage, field): (n_diff, n_interior)} of every differing record (measured, job 27831497)."""
    e = {}
    for f in ("gU", "gV"):
        e[(5, "S00_begin", f)] = e[(5, "S05_thermodynamics_sync", f)] = (3540, 3540)   # B: +0, not yet computed
        e[(5, "S06_dynamics", f)] = (180, 0)
    for f in ("guNm1", "gvNm1"):
        e[(5, "S00_begin", f)] = e[(5, "S05_thermodynamics_sync", f)] = (180, 0)      # READ_PICKUP's exchange
    e[(5, "S00_begin", "rhoInSitu")] = (4356, 3844)
    e[(5, "S00_begin", "surfaceForcingU")] = e[(5, "S02_load_fields", "surfaceForcingU")] = (4356, 3844)
    for st in ("S00_begin", "P02_rho_sigma_ivdc", "S04_oceanic_phys"):
        e[(5, st, "phiHydLow")] = (3600, 3600)
        e[(5, st, "totPhiHyd")] = (4096, 3844)
    for it in ITERS[1:]:
        for st in ("S00_begin", "S05_thermodynamics_sync"):
            for f in (G4 if it >= 7 else ("gU", "gV")):
                e[(it, st, f)] = (180, 0)
        e[(it, "D00b_mom_fluxform", "gU_k001")] = e[(it, "D00b_mom_fluxform", "gV_k001")] = (180, 0)
        for f in G4:
            e[(it, "S06_dynamics", f)] = (180, 0)
    for it in ITERS:
        for st in ("S10_momentum_correction", "S11_integr_continuity"):
            e[(it, st, "uVel")] = e[(it, st, "vVel")] = (59, 0)
    return e


_C = {}


def pair():
    if "pair" not in _C:
        a, b = (rr.run_top(rr.find_run(EXP, INP, k)) for k in ("restart_a", "restart_b"))
        _C["pair"] = (a, b, rc.compare(a, b, ITERS))
    return _C["pair"]


def table(rows):
    return {(r["iter"], r["stage"], r["field"]): (r["n_diff"], r["n_interior"]) for r in rows if r["n_diff"]}


def test_registered_restart_pair():
    a = rr.find_run(EXP, INP, "restart_a")
    b = rr.find_run(EXP, INP, "restart_b")
    assert a["steps"] == b["steps"] == ITERS and a["build"] == b["build"] == "jaxdump2"
    assert {"restart_a", "restart_b"} <= set(rr.FIXTURE_KINDS) and "restart_b" in rr.COPY_KINDS
    assert a["purpose"].startswith("GATE FIXTURE, NEVER A YARDSTICK") and b["purpose"].startswith("GATE FIXTURE")
    lock = rr.read_lock()
    assert rr.check_run(a, lock) == [] and rr.check_run(b, lock) == []


def test_fortran_restart_difference_pattern():
    _, _, res = pair()
    assert not res["only"]["A"] and not res["only"]["B"], res["only"]
    assert len(res["dumps"]) == 1605
    assert table(res["dumps"]) == expected_table()
    # end of step 10: only the four AB fields, 180 halo points each
    end = {k: v for k, v in table(res["dumps"]).items() if k[0] == 9 and k[1] in ("S06_dynamics",
                                                                                   "S16_blocking_exchanges")}
    assert end == {(9, "S06_dynamics", f): (180, 0) for f in G4}
    # interior differences only where B has not computed the field yet: B is +0 at every such point
    for r in res["dumps"]:
        if r["n_interior"]:
            assert r["iter"] == 5 and r["stage"] in ("S00_begin", "S02_load_fields", "S04_oceanic_phys",
                                                     "P02_rho_sigma_ivdc", "S05_thermodynamics_sync"), r["stage"]
            assert all(p[5] == "0000000000000000" for p in r["points"]), (r["stage"], r["field"])


def test_fortran_restart_halo_is_read_pickup_exchange():
    a_top, b_top, _ = pair()
    da, db = dump.DumpSet(a_top / "dumps"), dump.DumpSet(b_top / "dumps")
    for f in ("guNm1", "gvNm1"):
        a, b = da.field(5, "S00_begin", f)[0, 0], db.field(5, "S00_begin", f)[0, 0]
        assert periodic_copy_ok(b), f
        halo = np.pad(np.zeros((a.shape[0] - 4, a.shape[1] - 4), bool), 2, constant_values=True)
        assert np.all(rc.bits(a)[halo] == 0), f                                  # running model: +0 halos
        diff = rc.bits(a) != rc.bits(b)
        assert np.array_equal(diff, halo & (b != 0)) and diff.sum() == 180, f
        # the interior is the pickup file's (written by A at the end of step 5): A's interior bitwise
        assert np.array_equal(rc.bits(a)[~halo], rc.bits(b)[~halo]), f


def periodic_copy_ok(x, ol=2):
    """exch1 on one tile (nSx=nSy=nPx=nPy=1, no exch2): every halo point is the periodic copy of the interior."""
    return np.array_equal(rc.bits(np.pad(x[ol:-ol, ol:-ol], ol, mode="wrap")), rc.bits(x))


def test_fortran_restart_stdout_and_files():
    _, _, res = pair()
    so = res["stdout"]
    assert sorted(so["blocks"]) == [5, 6, 7, 8, 9, 10]
    d5 = so["blocks"][5]["diff"]
    assert [x.split("=")[0].split()[1] for x, _ in d5] == ["trAdv_CFL_u_max", "trAdv_CFL_v_max", "trAdv_CFL_w_max"]
    assert all(float(y.split("=")[1]) == 0.0 and float(x.split("=")[1]) != 0.0 for x, y in d5)
    assert all(not so["blocks"][ts]["diff"] and so["blocks"][ts]["n_lines"] == [51, 51] for ts in (6, 7, 8, 9, 10))
    assert sorted(so["solver"]) == [6, 7, 8, 9, 10] and all(s["equal"] for s in so["solver"].values())
    assert so["checkpoint"]["B"] == ["%CHECKPOINT        10 0000000010"]
    files = res["files"]
    assert files["pickup.0000000010.001.001.data"] == files["pickup.0000000010.001.001.meta"] == "identical"
    assert {f for f, v in files.items() if v == "DIFFERENT"} == {"data"}                      # the overlay
    assert {f for f, v in files.items() if v == "absent in A"} == {
        f"{v}.0000000005.001.001.{s}" for v in ("Eta", "S", "T", "U", "V", "W") for s in ("data", "meta")}


@pytest.mark.parametrize("where", ["halo", "interior"])
def test_negative_control_planted_change_is_detected(where):
    """A one-ulp change planted in B's guNm1 at S06_dynamics of iteration 9 (in memory) changes the table."""
    a_top, b_top, _ = pair()
    da, db = dump.DumpSet(a_top / "dumps"), dump.DumpSet(b_top / "dumps")
    rec = db.tiles(9, "S06_dynamics", "guNm1")[1]
    v = rec.data.copy()
    same = rc.bits(v) == rc.bits(da.tiles(9, "S06_dynamics", "guNm1")[1].data)
    inner = np.zeros_like(same)
    inner[:, 2:-2, 2:-2] = True
    k, j, i = np.argwhere(same & (inner if where == "interior" else ~inner))[0]
    v[k, j, i] = np.nextafter(v[k, j, i], np.inf)
    rec._data = v
    rows, _ = rc.compare_dumps(da, db, {9})
    got = table(rows)
    want = {k_: v_ for k_, v_ in expected_table().items() if k_[0] == 9}
    assert got != want
    assert got[(9, "S06_dynamics", "guNm1")] == (181, 1 if where == "interior" else 0)


def test_negative_control_periodic_copy_check():
    a_top, b_top, _ = pair()
    b = dump.DumpSet(b_top / "dumps").field(5, "S00_begin", "guNm1")[0, 0].copy()
    assert periodic_copy_ok(b)
    b[0, 10] = np.nextafter(b[0, 10], np.inf)
    assert not periodic_copy_ok(b)
