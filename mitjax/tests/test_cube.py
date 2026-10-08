"""Cubed-sphere W2 topology, exchanges and corner fills (plan Task 22; mitjax/pkg/exch2, mitjax/eesupp), tier 1x.

Oracle: lane A's registered dumps-on runs of all four cube experiments (W2 print-out, the jaxdump exchange probe), and
the gfortran cube replay harness (reference/replay_cube, runs in its CURRENT file; helpers in cube_gate.py)
of adjustment.cs-32x32x1 (48 tiles 16x8, OL 2, W2_CUMSUM_USE_MATRIX), solid-body.cs-32x32x1 (6 tiles 32x32, OL 2,
data.exch2 with W2_mapIO = 1), advect_cs (6 tiles 32x32 as nSx=2, nSy=3, OL 4) and global_ocean.cs32x15 (12 tiles 32x16,
OL 4). Gates:
* W2 print-out: w2_tile_topology.0000.log (1469 / 203 / 202 / 370 records) and the W2 STDOUT records (9 / 37 / 9 /
  9, incl. the OPEN_COPY_DATA_FILE echo of data.exch2) identical character for character;
* exchanges: every kind (XY, 3D, Z, S3D, SMs, SMn, As, An, Bs, Bn, UVs, UVn, UV3s, UV3n, Ds, Dn) built from the
  W2 topology alone (exch2_cube_tables.py) == the Fortran on every point, bitwise, for the index-coded probe and
  four signed-zero probes ((+0,+0), (-0,-0), (-0,+0), (+0,-0));
* FILL_CS_CORNER_TR_RL (fill4dir 0/1/2 x withSigns), _UV_RS, _UV_RL, _AG_RL (fill4dirX x withSigns) on every tile ==
  the Fortran, bitwise, same probes;
* sharded (P=4: padding on solid-body; P=6: one facet per device, in a subprocess with 6 fake devices) == single
  device bitwise on random data with -0, Inf and NaN, for every kind; transpose: dot test of every kind incl. the
  corners, and JAX's vjp == the hand-derived transpose for the C-grid kinds (passes + post statements).
Negative controls (each must bite): a facet link with the wrong edge, a missing D-grid sign flip, a corner fix from
the wrong neighbour, a plain copy instead of the sa1*u + sa2*v sum, corner flags of the wrong corner, withSigns
ignored in a corner fill.
"""

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax import paths
from mitjax.eesupp import exch_maps as EM
from mitjax.eesupp.exchange import Exchanger
from mitjax.eesupp.fill_cs_corner_ag_rl import fill_cs_corner_ag_rl
from mitjax.eesupp.fill_cs_corner_tr_rl import cs_corner_flags, fill_cs_corner_tr_rl
from mitjax.eesupp.fill_cs_corner_uv_rl import fill_cs_corner_uv_rl
from mitjax.eesupp.fill_cs_corner_uv_rs import fill_cs_corner_uv_rs
from mitjax.eesupp.print import MessageUnits
from mitjax.pkg.exch2 import w2_e2setup as E2, w2_eeboot as EB
from mitjax.tests import cube_gate as G

EXPECTED_PRINTOUT = {"adjustment.cs-32x32x1": (1469, 9), "solid-body.cs-32x32x1": (203, 37), "advect_cs": (202, 9),
                     "global_ocean.cs32x15": (370, 9)}


@pytest.mark.parametrize("exp", G.EXPS)
def test_cube_w2_printout_equals_replay(exp):
    e, w2, io = G.setup(exp)
    nlog, nout = EXPECTED_PRINTOUT[exp]
    assert G.printout_mismatches(exp, w2, io) == {"log": 0, "stdout": 0, "log_records": nlog,
                                                  "stdout_records": nout}


def test_cube_w2_negative_controls(monkeypatch):
    exp = "adjustment.cs-32x32x1"
    e, _, _ = G.setup(exp)

    def after(module, name, fn):
        orig = getattr(module, name)

        def wrapped(w2, **kw):
            out = orig(w2, **kw)
            fn(w2)
            return out
        monkeypatch.setattr(module, name, wrapped)

    # a facet link to the wrong edge (N of facet 1 to S of facet 3 instead of W): the set-up's own check stops
    after(E2, "w2_set_cs6_facets", lambda w2: w2.facet_link.__setitem__((1, 1), np.float32(3.2)))
    with pytest.raises(RuntimeError, match="Topology errors"):
        EB.w2_eeboot(e, MessageUnits())
    monkeypatch.undo()
    # a facet offset off by one after W2_SET_F2F_INDEX: W2_SET_TILE2TILES stops or the log differs
    after(E2, "w2_set_f2f_index", lambda w2: w2.facet_oi.__setitem__((1, 1), w2.facet_oi[1, 1] + 1))
    try:
        w2, io = EB.w2_eeboot(e, MessageUnits())
        assert G.printout_mismatches(exp, w2, io)["log"] > 0
    except RuntimeError as err:
        assert "connection" in str(err) or "out of" in str(err), err
    monkeypatch.undo()
    # tiles 1 and 2 swapped after W2_MAP_PROCS: the log differs
    def swap(w2):
        a, b = w2.W2_myTileList[1, 1], w2.W2_myTileList[2, 1]
        w2.W2_myTileList[1, 1], w2.W2_myTileList[2, 1] = b, a
    after(EB, "w2_map_procs", swap)
    w2, io = EB.w2_eeboot(e, MessageUnits())
    assert G.printout_mismatches(exp, w2, io)["log"] > 0


@pytest.mark.parametrize("exp", G.ORACLE_EXPS)
def test_cube_w2_and_probes_equal_oracle_runs(exp):
    """Lane A's dumps-on oracle runs (all four cube experiments, OLx 2 and 4): the W2 print-out character for
    character, and all 63 probe fields of the jaxdump exchange probe bitwise on every point."""
    e, w2, io = G.setup(exp)
    bad = G.oracle_printout_mismatches(exp, w2, io)
    assert bad["log"] == 0 and bad["stdout"] == 0, bad
    bad, n = G.oracle_probe_mismatches(exp)
    assert n == 63 and bad == {}, (n, bad)


@pytest.mark.parametrize("exp", G.EXPS)
def test_cube_exchanges_equal_replay(exp):
    assert G.exchange_mismatches(exp) == {}


def test_cube_exchange_negative_controls():
    exp = "adjustment.cs-32x32x1"

    def dgrid_no_sign(m):                       # D-grid sign changes dropped (exch2_uv_dgrid_3d_rx.template:91-196)
        swap, passes, post = m.rx2["Ds"]
        m.rx2["Ds"] = (swap, passes, {c: (s, k, np.ones_like(g), n) for c, (s, k, g, n) in post.items()})
    bad = G.exchange_mismatches(exp, G.planted_maps(exp, dgrid_no_sign))
    assert bad and all("Ds" in k for k in bad), bad

    def corner_wrong_neighbour(m):              # UVs corner fix uPhi(0,0) from vPhi(1,1) instead of vPhi(1,0)
        swap, passes, post = m.rx2["UVs"]
        s, k, g, n = (a.copy() for a in post[1])
        L = m.layout
        moved = np.nonzero(k == 2)[0]
        s[moved] = s[moved] + L.nx
        m.rx2["UVs"] = (swap, passes, {1: (s, k, g, n), 2: post[2]})
    bad = G.exchange_mismatches(exp, G.planted_maps(exp, corner_wrong_neighbour))
    assert bad and all(k.endswith("UVs_u") and not k.endswith("UV3s_u") for k in bad), bad

    # the copy control acts on the arithmetic, not the tables: evaluate the planted kinds as copies
    from mitjax.eesupp import exchange as X
    maps = EM.load_cube_maps(exp, "input")
    ex = Exchanger(maps)

    def as_copy(kind, u, v):
        if kind not in ("UVs", "UV3s"):
            return G.call(ex, kind, jnp.asarray(u), jnp.asarray(v))
        swap, passes, post = ex.rx2[kind]
        fu, fv = jnp.asarray(u).reshape(-1), jnp.asarray(v).reshape(-1)
        for p in passes:
            outs = []
            for c, own in ((1, fu), (2, fv)):
                src, wr, s1, s2 = p[c]
                val = jnp.where(s1 != 0, s1.astype(own.dtype) * fu[src], s2.astype(own.dtype) * fv[src])
                outs.append(jnp.where(wr, val, own))
            fu, fv = outs
        fu, fv = X.apply_post(post, fu, fv)
        return fu.reshape(maps.layout.shape2d), fv.reshape(maps.layout.shape2d)
    bad = G.exchange_mismatches(exp, apply=as_copy)
    assert bad and all(k[:2] != "z0" and ("UVs" in k or "UV3s" in k) for k in bad), bad   # only the zero probes

    # a facet link with the wrong orientation in the topology the tables are built from
    from mitjax.pkg.exch2.exch2_cube_tables import cube_exchange_programs
    from mitjax.pkg.exch2.w2_readparms import use_cubed_sphere_exchange
    import copy
    e, w2, _ = G.setup(exp)
    w = copy.deepcopy(w2)
    t = 1
    for N in range(1, w.exch2_nNeighbours[t] + 1):
        if w.exch2_pij[1, N, t] == 0:           # an orientation-changing link: flip its sign
            for k in range(1, 5):
                w.exch2_pij[k, N, t] = -w.exch2_pij[k, N, t]
            break
    try:
        maps2, rx2 = cube_exchange_programs(w, G.layout(exp), use_cubed_sphere_exchange(e), e.cfg.cpp)
        bad = G.exchange_mismatches(exp, EM.ExchangeMaps(G.layout(exp), maps2, {}, rx2))
        assert bad, "a flipped facet orientation was not detected"
    except RuntimeError as err:                 # or the PUT's source index leaves the array
        assert "out of bounds" in str(err)


def _fill_mismatches(exp, corners=None, ignore_signs=False):
    e, w2, _ = G.setup(exp)
    sz = e.cfg.size
    kw = dict(sNx=sz.sNx, sNy=sz.sNy, OLx=sz.OLx, OLy=sz.OLy)
    corners = cs_corner_flags(w2) if corners is None else corners
    R = G.replay(exp, "cube_fill.bin")
    F, cb = R["fields"], R["cbase"]
    bad = {}

    def cmp(key, got):
        n = int(np.count_nonzero(G.bits(got) != G.bits(F[key])))
        if n:
            bad[key] = n
    for iz in range(5):
        u, v = G.probe_inputs(exp, iz, cb)
        for s in (1, 2):
            ws = (s == 1) and not ignore_signs
            for d in range(3):
                cmp(f"z{iz}TR{d}s{s}_u", fill_cs_corner_tr_rl(d, ws, u, corners, True, **kw))
            for nm, fn in (("UVRS", fill_cs_corner_uv_rs), ("UVRL", fill_cs_corner_uv_rl)):
                gu, gv = fn(ws, u, v, corners, True, **kw)
                cmp(f"z{iz}{nm}s{s}_u", gu)
                cmp(f"z{iz}{nm}s{s}_v", gv)
            for d in (1, 2):
                gu, gv = fill_cs_corner_ag_rl(d == 1, ws, u, v, corners, True, **kw)
                cmp(f"z{iz}AG{d}s{s}_u", gu)
                cmp(f"z{iz}AG{d}s{s}_v", gv)
    return bad


@pytest.mark.parametrize("exp", G.EXPS)
def test_cube_fill_corners_equal_replay(exp):
    e, w2, _ = G.setup(exp)
    assert cs_corner_flags(w2).sum(axis=0).tolist() == [6, 6, 6, 6]      # every facet has its 4 corners once
    assert _fill_mismatches(exp) == {}


def test_cube_fill_corners_negative_controls():
    exp = "adjustment.cs-32x32x1"
    e, w2, _ = G.setup(exp)
    wrong = np.roll(cs_corner_flags(w2), 1, axis=1)                         # each corner flag on the wrong corner
    bad = _fill_mismatches(exp, corners=wrong)
    assert bad and any("TR1" in k for k in bad) and any("AG" in k for k in bad), sorted(bad)[:5]
    bad = _fill_mismatches(exp, ignore_signs=True)                           # withSigns ignored
    assert bad and all("s1" in k for k in bad) and all(k.startswith("z0") or k[:2] in ("z1", "z2", "z3", "z4")
                                                       for k in bad), sorted(bad)[:5]


def _random(L, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((2,) + L.shape2d)
    a[0].flat[::7] = -0.0
    a[1].flat[::11] = np.inf
    a[0].flat[5::13] = np.nan
    return a[0], a[1]


@pytest.mark.parametrize("exp", G.EXPS)
def test_cube_sharded_p4_equals_single(exp):
    from mitjax.eesupp.shard import TileSharding
    maps = EM.load_cube_maps(exp, "input")
    ex = Exchanger(maps)
    sh = TileSharding(maps, 4)
    u, v = _random(maps.layout, 1)
    for kind in G.SCALAR_KINDS + G.VECTOR_KINDS:
        f = sh.shard_map(lambda e, a, b, kind=kind: tuple(G.call(e, kind, a, b)),
                         in_specs=(sh.TILES, sh.TILES, sh.TILES), out_specs=sh.TILES)
        got = [sh.unpad(o) for o in f(sh.ex, sh.put_tiles(u), sh.put_tiles(v))]
        want = G.call(ex, kind, jnp.asarray(u), jnp.asarray(v))
        for g, w in zip(got, want):
            assert np.array_equal(G.bits(g), G.bits(w)), kind
    # the Fortran probes through the sharded path
    def apply(kind, uu, vv):
        f = sh.shard_map(lambda e, a, b: tuple(G.call(e, kind, a, b)), in_specs=(sh.TILES, sh.TILES, sh.TILES),
                         out_specs=sh.TILES)
        return [sh.unpad(o) for o in f(sh.ex, sh.put_tiles(uu), sh.put_tiles(vv))]
    assert G.exchange_mismatches(exp, apply=apply) == {}


P6_SCRIPT = r"""
import os, sys
from mitjax.xla_flags import gate_xla_flags
os.environ["XLA_FLAGS"] = gate_xla_flags("").replace("device_count=4", "device_count=6")
import jax, jax.numpy as jnp, numpy as np
from mitjax.eesupp import exch_maps as EM
from mitjax.eesupp.exchange import Exchanger
from mitjax.eesupp.shard import TileSharding
from mitjax.tests import cube_gate as G
from mitjax.tests.test_cube import _random
assert len(jax.devices()) == 6
for exp in G.EXPS:
    maps = EM.load_cube_maps(exp, "input"); ex = Exchanger(maps); sh = TileSharding(maps, 6)
    u, v = _random(maps.layout, 2)
    for kind in G.SCALAR_KINDS + G.VECTOR_KINDS:
        f = sh.shard_map(lambda e, a, b, kind=kind: tuple(G.call(e, kind, a, b)), in_specs=(sh.TILES,) * 3,
                         out_specs=sh.TILES)
        got = [sh.unpad(o) for o in f(sh.ex, sh.put_tiles(u), sh.put_tiles(v))]
        want = G.call(ex, kind, jnp.asarray(u), jnp.asarray(v))
        for g, w in zip(got, want):
            assert np.array_equal(G.bits(g), G.bits(w)), (exp, kind)
    print("P6 OK", exp, sh.blocks.Tloc, sum(sh.ex.rounds(k) for k in ("UVs", "Ds", "Z", "Bs")))
"""


def test_cube_sharded_p6_equals_single():
    """P=6 (solid-body, advect_cs: one facet per device; adjustment.cs: 8 tiles, cs32x15: 2 per device) in a
    subprocess with 6 fake CPU devices (the test session has 4, mitjax/xla_flags.py)."""
    env = dict(os.environ)
    env.pop("XLA_FLAGS", None)
    env["PYTHONPATH"] = str(paths.REPO)
    r = subprocess.run([sys.executable, "-c", P6_SCRIPT], env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    assert r.stdout.count("P6 OK") == len(G.EXPS), r.stdout


def _post_transpose(post, bu, bv):
    """Hand-derived transpose of the post gather y_c[p] = keep ? x_c[p] : (-1)^neg * sign * x_comp[src]."""
    xu = np.where(post[1][1] == 0, bu, 0.0)
    xv = np.where(post[2][1] == 0, bv, 0.0)
    for c, b in ((1, bu), (2, bv)):
        src, comp, sign, neg = post[c]
        w = np.nonzero(comp != 0)[0]
        val = np.where(neg[w], -1.0, 1.0) * sign[w].astype(np.float64) * b[w]
        for target, cc in ((xu, 1), (xv, 2)):
            m = w[comp[w] == cc]
            np.add.at(target, src[m], val[comp[w] == cc])
    return xu, xv


@pytest.mark.parametrize("exp", G.EXPS)
def test_cube_transpose(exp):
    from mitjax.tests.exch_gate import rx2_transpose_numpy
    maps = EM.load_cube_maps(exp, "input")
    ex = Exchanger(maps)
    L = maps.layout
    u, v = _random(L, 3)
    u, v = np.nan_to_num(u, posinf=1.0), np.nan_to_num(v, posinf=1.0)
    yu, yv = _random(L, 4)
    yu, yv = np.nan_to_num(yu, posinf=1.0), np.nan_to_num(yv, posinf=1.0)
    for kind in G.SCALAR_KINDS + G.VECTOR_KINDS:
        scalar = kind in G.SCALAR_KINDS

        def f(a, b, kind=kind):
            return tuple(G.call(ex, kind, a, b))
        outs, vjp = jax.vjp(f, jnp.asarray(u), jnp.asarray(v))
        ys = (jnp.asarray(yu),) if scalar else (jnp.asarray(yu), jnp.asarray(yv))
        xb = vjp(ys)
        lhs = sum(float(jnp.vdot(o, y)) for o, y in zip(outs, ys))
        rhs = float(jnp.vdot(xb[0], u)) + (0.0 if scalar else float(jnp.vdot(xb[1], v)))
        assert abs(lhs - rhs) <= 1e-12 * max(abs(lhs), 1.0), (kind, lhs, rhs)
        if kind in ex.rx2:
            swap, passes, post = maps.rx2[kind]
            bu, bv = _post_transpose(post, yu.reshape(-1), yv.reshape(-1))
            hu, hv = rx2_transpose_numpy(passes, swap, bu, bv)
            for h, j in ((hu, xb[0]), (hv, xb[1])):
                d = np.abs(np.asarray(j).reshape(-1) - h)
                assert np.max(d) <= 8 * np.finfo(float).eps * np.max(np.abs(np.concatenate([yu.ravel(),
                                                                                             yv.ravel()]))), kind


# ---------------------------------------------------------------------------------------------------------------------
# INI_CURVILINEAR_GRID

@pytest.mark.parametrize("exp", G.EXPS)
def test_cube_grid_equals_replay_and_oracle(exp):
    """The 35 GRID.h fields (xC .. rAz, angleCosC/SinC, u2zonDir/v2zonDir, the 12 reciprocals, fCori, fCoriG,
    fCoriCos) bitwise on every point incl. halos and corners vs the replay's cube_grid.bin (after INITIALISE_FIXED)
    and vs lane A's G00_geometry dumps. tile00N.mitgrid (adjustment.cs, solid-body: angles computed by
    CALC_GRID_ANGLES; solid-body also rescales from radius_fromHorizGrid 6370 km to rSphere 5500.4 km) and
    grid_cs32.face00N.bin (advect_cs, cs32x15: angles read)."""
    g = G.horizontal_grid(exp)
    bad, n = G.grid_replay_mismatches(exp, g)
    assert n == 35 and bad == {}, (n, bad)
    bad, n = G.grid_oracle_mismatches(exp, g)
    assert n >= 32 and bad == {}, (n, bad)


def test_cube_grid_negative_controls(monkeypatch):
    import dataclasses
    from mitjax.eesupp import exch_rs as XR
    from mitjax.model.src import ini_curvilinear_grid as ICG
    from mitjax.model.src.ini_parms import ini_parms_grid
    from mitjax.pkg.exch2.w2_eeboot import exch2_topology
    exp = "solid-body.cs-32x32x1"
    # records dxC and dyC swapped in the read
    rec = list(ICG.RECORDS)
    rec[10], rec[11] = rec[11], rec[10]
    monkeypatch.setattr(ICG, "RECORDS", tuple(rec))
    bad, _ = G.grid_replay_mismatches(exp, G.horizontal_grid(exp))
    assert "dxC" in bad and "dyC" in bad, bad
    monkeypatch.undo()
    # the radius rescaling skipped (radius_fromHorizGrid = rSphere)
    e, w2, _ = G.setup(exp)
    gp = ini_parms_grid(e, exch2_topology(w2))
    assert gp.rSphere != gp.radius_fromHorizGrid
    bad, _ = G.grid_replay_mismatches(exp, G.horizontal_grid(exp, gp=dataclasses.replace(
        gp, radius_fromHorizGrid=gp.rSphere)))
    assert {"dxC", "rA", "rAz", "recip_dxG"} <= set(bad) and "xC" not in bad, bad
    # signs on in the C-grid exchange of dxC/dyC (the Fortran passes .FALSE.)
    orig = XR.EXCH_UV_XY_RS
    monkeypatch.setattr(XR, "EXCH_UV_XY_RS", lambda u, v, ws, *, ex: orig(u, v, True, ex=ex))
    bad, _ = G.grid_replay_mismatches(exp, G.horizontal_grid(exp))
    assert {"dxC", "dyC", "rAw", "rAs", "dxG", "dyG"} & set(bad), bad
    assert "xC" not in bad and "rA" not in bad, bad


# ---------------------------------------------------------------------------------------------------------------------
# GAD_ADVECTION's cubed-sphere pass structure

def test_gad_cs_pass_schedule():
    """gad_cs_passes.gad_cs_pass_schedule == the gfortran replay of gad_advection.F's verbatim pass statements for
    every facet (1..6), every N/S/E/W edge combination (16) and pass (3): flags and the ordered fill/flux/update
    events. Negative control: a schedule with the after-flux fill directions swapped differs."""
    from mitjax.pkg.generic_advdiff.gad_cs_passes import gad_cs_pass_schedule, tile_schedules
    ref = G.gad_pass_replay()
    assert len(ref) == 6 * 16 * 3

    def ours(nCFace, iE, swap=False):
        sch = gad_cs_pass_schedule(nCFace, iE & 1 == 1, iE & 2 == 2, iE & 4 == 4, iE & 8 == 8)
        out = {}
        for ipass, p in enumerate(sch, start=1):
            ev = list(p["events"])
            if swap:
                ev = [("fill", 3 - e[1]) if e[0] == "fill" and k > 0 and ev[k - 1][0] == "flux" else e
                      for k, e in enumerate(ev)]
            out[(nCFace, iE, ipass)] = ((p["overlapOnly"], p["interiorOnly"], p["calc_fluxes_X"],
                                        p["calc_fluxes_Y"]), ev)
        return out
    got, planted = {}, {}
    for nCFace in range(1, 7):
        for iE in range(16):
            got.update(ours(nCFace, iE))
            planted.update(ours(nCFace, iE, swap=True))
    assert got == ref
    assert sum(planted[k] != ref[k] for k in ref) > 0
    # the experiments' tiles: each tile's schedule (from its facet and edge flags) is the replayed case
    for exp in G.EXPS:
        e, w2, _ = G.setup(exp)
        sz = e.cfg.size
        tiles = [w2.W2_myTileList[bi, bj] for bj in range(1, sz.nSy + 1) for bi in range(1, sz.nSx + 1)]
        for t, sch in zip(tiles, tile_schedules(w2)):
            iE = (w2.exch2_isNedge[t] + 2 * w2.exch2_isSedge[t] + 4 * w2.exch2_isEedge[t]
                  + 8 * w2.exch2_isWedge[t])
            for ipass, p in enumerate(sch, start=1):
                flags = (p["overlapOnly"], p["interiorOnly"], p["calc_fluxes_X"], p["calc_fluxes_Y"])
                assert (flags, list(p["events"])) == ref[(w2.exch2_myFace[t], iE, ipass)], (exp, t, ipass)
