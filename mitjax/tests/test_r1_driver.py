"""R1 tutorial_barotropic_gyre through the run driver (plan Task 12): the scan driver, pickups, the restart gate.

* scan == step loop: the 10-step run as one `lax.scan` (the_main_loop), as scan chunks of 3+4+3 steps and as the
  driver's monitor chunking is bitwise equal (every leaf of the carry, all points incl. halos; the solver scalars of
  every step) to the step-by-step loop over the jitted FORWARD_STEP, which tier 1x gates against the oracle dumps
  (test_r1_barotropic_gyre.py); negative control: the scan started with myIter = nIter0 + 1 (the Adams-Bashforth
  start test of step 1 misses) differs. (A scan started at iloop0 = 2 does NOT differ in R1, measured: the clock only
  feeds the AB start test, which reads the carried myIter, and no time-dependent forcing.)
* pickup round trip: WRITE_PICKUP of the State at step 10 equals the oracle's pickup.ckptA byte for byte (data and
  meta, tier 1); here: READ_PICKUP of our pickup gives every pickup field back bitwise on the interior, and with the
  exchanges of READ_PICKUP on all points.
* restart gate: 10 steps == 5 steps + pickup.0000000005 + 5 steps (nIter0 = 5 in a copy of the experiment, as a
  restarted run's `data`), bitwise on every leaf of the carry on all points, except 180 halo points of each of gU,
  gV, guNm1, gvNm1 (READ_PICKUP exchanges GuNm1/GvNm1, the running model does not: the Fortran's own restart path
  by its code, read_pickup.F:556; interiors bitwise), the pickup.0000000010 files of both runs byte for byte, and B's STDOUT records equal to A's; negative controls: GuNm1 dropped from the pickup -> READ_PICKUP stops (as CHECK_PICKUP does with
  pickupStrictlyMatch = .TRUE.); GuNm1 zeroed in the pickup -> the gate fails (counted points).
"""

import filecmp
import os
import time
from pathlib import Path

import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from mitjax import paths  # noqa: E402

EXP, INP = "tutorial_barotropic_gyre", "input"
EXP_DIR = paths.UPSTREAM / "verification" / EXP


def out_dir(tag):
    """A new directory for this test's run (never reused, never removed)."""
    d = paths.RUNS / "tests_r1_driver" / f"{tag}-{os.getpid()}-{time.time_ns()}"
    return d


def model(tag, overrides=None, extra_links=()):
    from mitjax.drivers.model import Model, with_namelist
    from mitjax.drivers.run import load_experiment, make_rundir
    exp = load_experiment(EXP_DIR, INP)
    if overrides:
        exp = with_namelist(exp, overrides)
    rundir = make_rundir(EXP_DIR, INP, out_dir(tag))
    for src in extra_links:
        os.symlink(src, rundir / Path(src).name)
    return Model(exp, rundir)


_M = {}


def base():
    if "base" not in _M:
        _M["base"] = model("base")
    return _M["base"]


def leaves(carry):
    return jax.tree_util.tree_flatten_with_path(carry)[0]


def named_leaves(carry):
    """[(name, array)] of the carry (State, ff, phi0surf, flow) with Fortran names."""
    state, ff, phi0surf, flow = carry
    out = []
    for n in state.names():
        v = getattr(state, n)
        for k, f in enumerate(v if isinstance(v, tuple) else (v,)):
            out.append((n if not isinstance(v, tuple) else f"{n}_{k + 1}", f.data))
    out += [(f"ff.{n}", getattr(ff, n).data) for n in ff.names()]
    out.append(("phi0surf", phi0surf.data))
    out += [(f"flow.{n}", f.data) for n, f in zip(("uFld", "vFld", "wFld"), flow)]
    return out


def ndiff(a, b, where=False):
    """{field: number of points whose bit patterns differ} over every leaf of the carry (all points incl. halos);
    where=True: {field: (n, n of them in the interior)}."""
    out = {}
    for (n, x), (_, y) in zip(named_leaves(a), named_leaves(b)):
        x, y = np.asarray(x), np.asarray(y)
        m = (x.view(np.int64) != y.view(np.int64)) if x.dtype == np.float64 else (x != y)
        d = int(np.count_nonzero(m))
        if d:
            if where:
                sz = base_size()
                inner = m[..., sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx]
                out[n] = (d, int(np.count_nonzero(inner)))
            else:
                out[n] = d
    return out


def base_size():
    from mitjax.drivers.run import load_experiment
    return load_experiment(EXP_DIR, INP).cfg.size


def step_loop(m, n):
    """The step-by-step reference: the jitted FORWARD_STEP called once per iteration from Python."""
    fn = jax.jit(m.step)
    carry = m.initial_carry()
    t, it = m.start_counters()
    cgs = []
    for k in range(1, n + 1):
        carry, t, it, cg = fn(m.arrays, carry, jnp.int32(k), t, it)
        cgs.append({key: np.asarray(v) for key, v in cg.items()})
    return carry, t, it, cgs


def scan_chunks(m, lengths, iloop0=1, myIter_offset=0):
    from mitjax.drivers.the_main_loop import the_main_loop
    carry = m.initial_carry()
    t, it = m.start_counters()
    it = it + myIter_offset
    cgs = []
    for n in lengths:
        carry, t, it, cg = the_main_loop(m.step, m.arrays, carry, t, it, nTimeSteps=n, iloop0=iloop0)
        cgs += [{key: np.asarray(v)[s] for key, v in cg.items()} for s in range(n)]
        iloop0 += n
    return carry, t, it, cgs


def test_scan_equals_step_loop_bitwise():
    m = base()
    ref, t0, it0, cg0 = step_loop(m, 10)
    for lengths in ((10,), (3, 4, 3)):
        c, t, it, cg = scan_chunks(m, lengths)
        assert not ndiff(c, ref), (lengths, ndiff(c, ref))
        assert float(t) == float(t0) and int(it) == int(it0)
        for a, b in zip(cg, cg0):
            assert all(np.array_equal(a[k], b[k]) for k in a), lengths
    # the driver's chunking (host events: monitor every step) and its STDOUT records
    from mitjax.drivers.run import forward
    res = forward(m, write_pickups=False)
    assert res.chunk_lengths == [1] * 10
    assert not ndiff(res.carry, ref), ndiff(res.carry, ref)
    # negative control (a driver bug the gate must see): the scan started with the counter of the previous run's
    # last step (myIter = nIter0 + 1): the Adams-Bashforth start test (adams_bashforth2.F:61) misses step 1
    c, _, _, _ = scan_chunks(m, (10,), myIter_offset=1)
    d = ndiff(c, ref)
    assert sum(d.values()) > 1000, d


def test_pickup_round_trip():
    from mitjax.drivers.run import forward
    from mitjax.model.src.read_pickup import read_pickup
    from mitjax.pkg.mdsio.mdsio_write_field import MdsContext
    from mitjax.model.src.write_pickup import write_pickup
    m = base()
    res = forward(m, write_pickups=False)
    st = res.carry[0]
    d = out_dir("roundtrip")
    d.mkdir(parents=True)
    mds = MdsContext(d, m.cfg.size)
    fn = write_pickup(True, "0000000010", float(res.myTime), 10, cfg=m.cfg, params=m.params, ip=m.prm.init,
                      io=m.io, state=st, mds=mds)
    assert fn == "pickup.0000000010"
    from mitjax.pkg.rw.read_rec import RW
    rw = RW(d, m.prm.init.readBinaryPrec, m.cfg.size)
    empty = jax.tree_util.tree_map(lambda x: jnp.full_like(x, jnp.nan), st)
    back = read_pickup(empty, 10, cfg=m.cfg, params=m.prm, ex=m.ex, rw=rw)
    sz = m.cfg.size
    I = (Ellipsis, slice(sz.OLy, sz.OLy + sz.sNy), slice(sz.OLx, sz.OLx + sz.sNx))
    halo_diff = {}
    for name in ("uVel", "vVel", "theta", "salt", "guNm1", "gvNm1", "etaN"):
        a, b = np.asarray(getattr(back, name).data), np.asarray(getattr(st, name).data)
        assert np.array_equal(a[I].view(np.int64), b[I].view(np.int64)), name            # interior: the file
        halo_diff[name] = int(np.count_nonzero(a.view(np.int64) != b.view(np.int64)))
    # halos: READ_PICKUP exchanges every field it reads (read_pickup.F:538-562); the running model's uVel, vVel,
    # theta, salt, etaN halos are exchanged too (DO_FIELDS_BLOCKING_EXCHANGES), so all points agree; guNm1 / gvNm1
    # halos are never exchanged in a running model (TIMESTEP writes the interior; the halos keep INI_DYNVARS's 0),
    # so after a restart they hold the exchanged copies instead (measured: 248 / 248 halo points; the Fortran's
    # restart does the same: Fortran behaviour, not a port difference)
    assert {k: v for k, v in halo_diff.items() if k not in ("guNm1", "gvNm1")} == dict.fromkeys(
        ("uVel", "vVel", "theta", "salt", "etaN"), 0), halo_diff
    assert halo_diff["guNm1"] > 0 and halo_diff["gvNm1"] > 0, halo_diff


def _restart_pair():
    if "pair" in _M:
        return _M["pair"]
    dt = 1200.0
    a = model("restartA", {("data", "PARM03", "pChkptFreq"): 5 * dt})
    from mitjax.drivers.run import forward
    ra = forward(a)
    assert ra.pickups == ["pickup.0000000005", "pickup.0000000010"], ra.pickups
    _M["pair"] = (a, ra)
    return _M["pair"]


def _restart(tag, pick_src, mutate=None):
    """Run B: nIter0 = 5, nTimeSteps = 5 (a restarted run's data), pickup.0000000005 (a copy, mutated if asked)."""
    from mitjax.drivers.run import forward
    a, ra = _restart_pair()
    d = out_dir(tag + "_pickup")
    d.mkdir(parents=True)
    links = []
    for suf in ("data", "meta"):
        src = Path(a.rundir) / f"pickup.0000000005.001.001.{suf}"
        dst = d / src.name
        dst.write_bytes(src.read_bytes())
        links.append(dst)
    if mutate:
        mutate(d)
    b = model(tag, {("data", "PARM03", "pChkptFreq"): 5 * 1200.0, ("data", "PARM03", "nIter0"): 5,
                    ("data", "PARM03", "nTimeSteps"): 5}, extra_links=links)
    rb = forward(b)
    return a, ra, b, rb


def test_restart_gate_bitwise():
    a, ra, b, rb = _restart("restartB", None)
    assert b.prm.time.nIter0 == 5 and b.params.mom_StartAB == 5
    assert rb.pickups == ["pickup.0000000010"]
    d = ndiff(rb.carry, ra.carry, where=True)
    # every field bitwise on all points, except halo points of gU, gV, guNm1, gvNm1 (measured 180 each, interior 0):
    # READ_PICKUP exchanges GuNm1/GvNm1 (read_pickup.F:556) while the running model never exchanges them; the halo
    # values that TIMESTEP/MOM_FLUXFORM do not overwrite then live on in gU/guNm1 without reaching any interior
    # point (the Fortran's restart has the same property by its code, read_pickup.F:556 vs TIMESTEP; not measured with
    # a Fortran restart run; the pickup files hold interiors only)
    assert set(d) <= {"gU", "gV", "guNm1", "gvNm1"} and all(inner == 0 for _, inner in d.values()), d
    for suf in ("data", "meta"):
        f = f"pickup.0000000010.001.001.{suf}"
        assert filecmp.cmp(Path(a.rundir) / f, Path(b.rundir) / f, shallow=False), f
    # the STDOUT records: B's MONITOR block of iteration 5 (INITIALISE_VARIA's, after READ_PICKUP) equals A's block
    # after step 5 except the three trAdv_CFL lines (B: MON_INIT's 0, no THERMODYNAMICS has run yet; the Fortran's
    # restart prints the same, mon_init.F:46-48); from step 6 on B's records equal A's, apart from A's %CHECKPOINT of 5
    def block5(recs):
        k = next(n for n, r in enumerate(recs) if "%MON time_tsnumber" in r and r.split("=")[1].strip() == "5")
        e = next(n for n in range(k, len(recs)) if "End MONITOR" in recs[n])
        return k, recs[k:e]
    ka, a5 = block5(ra.records)
    kb, b5 = block5(rb.records)
    diff = [(x, y) for x, y in zip(a5, b5) if x != y]
    assert len(a5) == len(b5) and len(diff) == 3 and all("trAdv_CFL" in x and "trAdv_CFL" in y for x, y in diff), diff
    tail = lambda recs, k: [r for r in recs[next(n for n in range(k, len(recs)) if "cg2d: Sum(rhs)" in recs[n]):]   # noqa
                            if "%CHECKPOINT" not in r or r.rstrip().endswith("0000000010")]
    assert tail(rb.records, kb) == tail(ra.records, ka)


def _fields(meta_path):
    txt = meta_path.read_text()
    return txt[txt.index("fldList"):]


def test_restart_negative_controls():
    from mitjax.pkg.mdsio.mdsio_write_meta import mds_write_meta
    sz_rec = 62 * 62 * 8

    def drop_gunm1(d):
        """Rewrite the pickup without its 5th record (GuNm1) and without the name in the field list."""
        data = d / "pickup.0000000005.001.001.data"
        raw = data.read_bytes()
        data.write_bytes(raw[:4 * sz_rec] + raw[5 * sz_rec:])
        meta = d / "pickup.0000000005.001.001.meta"
        flds = ["Uvel", "Vvel", "Theta", "Salt", "GvNm1", "EtaN", "dEtaHdt", "EtaH"]
        mds_write_meta(meta, data, " ", " ", 64, 2, [[62, 1, 62], [62, 1, 62], [1, 1, 1]], (0, 1), len(flds),
                       flds, 1, [6000.0], 1.0, 8, 5)

    # a missing GuNm1 is recognised (mom_StartAB = 0) and, with pickupStrictlyMatch = .TRUE. (the default), stops
    # the run: CHECK_PICKUP, check_pickup.F:216-223 (ported in 42846d0, GO lane)
    with pytest.raises(ValueError, match="pickupStrictlyMatch=.FALSE."):
        _restart("negDrop", None, mutate=drop_gunm1)

    def zero_gunm1(d):
        data = d / "pickup.0000000005.001.001.data"
        raw = bytearray(data.read_bytes())
        raw[4 * sz_rec:5 * sz_rec] = bytes(sz_rec)
        data.write_bytes(bytes(raw))

    _, ra, _, rb = _restart("negZero", None, mutate=zero_gunm1)
    d = ndiff(rb.carry, ra.carry, where=True)
    assert sum(inner for _, inner in d.values()) > 1000 and "etaN" in d and "uVel" in d, d


def test_monitor_config_from_params_vs_printout():
    """The run driver's MONITOR configuration comes from our INI_PARMS (ini_parms_dyn, ini_parms_time): every value
    equals the oracle's parameter printout (monitor_gate.make_cfg / make_params read the STDOUT), R1. Cheap: no
    model step; the Model's set-up only."""
    from mitjax.drivers.run import MonitorHost
    from mitjax.tests import monitor_gate as mg
    m = base()
    mh = MonitorHost(m)
    o = mg.oracle(EXP, INP)
    ref_cfg = mg.make_cfg(o)
    ref_p = mg.make_params(o, ref_cfg)
    assert vars(mh.cfg) == vars(ref_cfg), {k: (v, vars(ref_cfg)[k]) for k, v in vars(mh.cfg).items()
                                           if vars(ref_cfg).get(k) != v}
    assert vars(mh.params) == vars(ref_p), {k: (v, vars(ref_p)[k]) for k, v in vars(mh.params).items()
                                            if vars(ref_p).get(k) != v}
