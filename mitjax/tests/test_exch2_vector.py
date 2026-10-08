"""The exch2 C-grid vector exchanges as the Fortran computes them: halo values are sums `sa1*A1 + sa2*A2` of both
components at the same source index (EXCH2_PUT_RX2, pkg/exch2/exch2_put_rx2.template:229-230, 317-318), not copies
(lane A's finding from the exchange probes, reference/probe_maps/; tables: mitjax/pkg/exch2/exch2_rx2_cube.py;
application:
mitjax/eesupp/exchange.py apply_rx2, sharded_exchange.py). Tier1x.

Gates (global_ocean.90x40x15, the exch2 layout of M1; exch1 layouts advect_xy and advect_xz as the copy side):
* the tables from the W2 topology reproduce the measured maps of EXCH_UV_XY_RL (with and without signs),
  EXCH_UV_3D_RL and EXCH_UV_DGRID_3D_RL on every point (index, component and sign of the nonzero term);
* lane A's mixed signed-zero probes zu*/zv* (jdon3 runs) bitwise on every point for UVs, As, Bs, Ds, UV3s on all
  three layouts (exch2: the 5610 filled halo points of the -0 component become +0 in the C-grid exchanges, the A/B
  grids copy; exch1: copies), and the original probe set of the same runs unchanged;
* sharded P=4 == P=1 bitwise (zu/zv probes and random data with -0, Inf, NaN);
* the transpose: JAX's vjp == the hand-derived transpose (exch_gate.rx2_transpose_numpy), the dot test
  <Ex, y> = <x, E^T y>, and the 0*v terms visible in it (an Inf cotangent puts NaN into the other component).
Negative control: the plain copy (maps without the rx2 tables) fails the zu/zv gate on global_ocean (and passes the
code probes, which cannot see it).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.eesupp import exch_maps as EM
from mitjax.eesupp.exchange import Exchanger
from mitjax.eesupp.shard import TileSharding
from mitjax.tests.exch_gate import (JD3_PROBES, MIXED_ROUTINES, bits, jd3_probe_dumps, mixed_zero_mismatches,
                                    probe_mismatches, rx2_apply_numpy, rx2_transpose_numpy)

GO = "global_ocean.90x40x15"


def test_rx2_tables_reproduce_measured_maps():
    maps = EM.load_maps(GO)
    L = maps.layout
    assert set(maps.rx2) == {"UVs", "UVn", "UV3s", "Ds"}
    n = L.npoints
    own = np.arange(n)
    u = (1.0 + own).astype(np.float64)                 # distinct codes: a nonzero term identifies its source
    v = (1.0 + own + 10 * n).astype(np.float64)
    for kind, (swap, passes) in maps.rx2.items():
        assert len(passes) == 2 and swap == (kind == "Ds")
        uo, vo = rx2_apply_numpy(passes, swap, u, v)
        for c, o in (("u", uo), ("v", vo)):
            src, comp, sign = maps.maps[f"{kind}_{c}"]
            want = np.where(comp == 0, u if c == "u" else v, np.where(comp == 1, u[src], v[src]) * sign)
            assert int(np.count_nonzero(o != want)) == 0, (kind, c)
            assert int(np.count_nonzero(comp)) == 5610, (kind, c)
    for kind in ("UVs", "UVn", "UV3s", "Ds"):                           # every written point has one nonzero term
        for p in maps.rx2[kind][1]:
            for c in (1, 2):
                _, w, s1, s2 = p[c]
                assert np.all((np.abs(s1[w]) + np.abs(s2[w])) == 1)


def test_mixed_signed_zero_probes_bitwise():
    for exp in JD3_PROBES:
        maps = EM.load_maps(exp)
        ds, it = jd3_probe_dumps(exp)
        ex = Exchanger(maps)
        bad = mixed_zero_mismatches(ex, ds, it)
        assert len(bad) == 20 and sum(bad.values()) == 0, (exp, {k: v for k, v in bad.items() if v})
        assert sum(probe_mismatches(ex, ds, it).values()) == 0, exp
        if exp == GO:                                                  # the measured effect, exch2
            u, v = jax.jit(MIXED_ROUTINES["UVs"])(ex, jnp.full(maps.layout.shape2d, -0.0),
                                                  jnp.zeros(maps.layout.shape2d))
            assert int(np.signbit(np.asarray(u)).sum()) == 3600 + 6 and not np.signbit(np.asarray(v)).any()
        else:
            assert not maps.rx2


def test_negative_control_plain_copy():
    maps = EM.load_maps(GO)
    ds, it = jd3_probe_dumps(GO)
    copy = Exchanger(dataclasses.replace(maps, rx2={}))
    bad = mixed_zero_mismatches(copy, ds, it)
    nz = {k: v for k, v in bad.items() if v}
    print(f"plain copy, mixed probes: {nz}")
    assert set(nz) == {f"{t}{g}_{c}" for t, c in (("zu", "u"), ("zv", "v")) for g in ("UVs", "Ds", "UV3s")}, nz
    assert all(v == 5610 for v in nz.values()), nz
    assert sum(probe_mismatches(copy, ds, it).values()) == 0          # the code probes cannot see it


def _random_uv(L, seed, nlev=2):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((2, L.nTiles, nlev, L.ny, L.nx))
    a[rng.random(a.shape) < 0.2] = -0.0
    a[rng.random(a.shape) < 0.1] = 0.0
    a[0, 3, 0, 5, 5] = np.inf
    a[1, 7, 1, 4, 6] = np.nan
    return a[0], a[1]


def test_sharded_p4_equals_p1():
    maps = EM.load_maps(GO)
    L = maps.layout
    ex1 = Exchanger(maps)
    sh = TileSharding(maps, 4)
    u, v = _random_uv(L, 1)
    calls = {"UVs": lambda e, a, b: e.EXCH_UV_XY_RL(a, b, True), "UVn": lambda e, a, b: e.EXCH_UV_XY_RL(a, b, False),
             "UV3s": lambda e, a, b: e.EXCH_UV_3D_RL(a, b, True), "Ds": lambda e, a, b: e.EXCH_UV_DGRID_3D_RL(a, b, True)}
    for kind, fn in calls.items():
        ref = jax.jit(fn)(ex1, jnp.asarray(u), jnp.asarray(v))
        f = sh.shard_map(fn, in_specs=(sh.TILES,) * 3, out_specs=sh.TILES)
        got = f(sh.ex, sh.put_tiles(u), sh.put_tiles(v))
        for r, g in zip(ref, got):
            assert int(np.count_nonzero(bits(sh.unpad(g)) != bits(r))) == 0, kind
        hand = rx2_apply_numpy(maps.rx2[kind][1], maps.rx2[kind][0], np.moveaxis(u, 1, 0).reshape(2, -1),
                               np.moveaxis(v, 1, 0).reshape(2, -1))
        for r, h in zip(ref, hand):
            assert int(np.count_nonzero(bits(np.moveaxis(np.asarray(r), 1, 0).reshape(2, -1)) != bits(h))) == 0, kind
    ds, it = jd3_probe_dumps(GO)

    def apply(fn, ex, a, b):
        f = sh.shard_map(fn, in_specs=(sh.TILES,) * 3, out_specs=sh.TILES)
        return tuple(sh.unpad(o) for o in f(sh.ex, sh.put_tiles(a), sh.put_tiles(b)))
    bad = mixed_zero_mismatches(ex1, ds, it, apply=apply)
    assert sum(bad.values()) == 0, {k: v for k, v in bad.items() if v}
    print(f"rounds at P=4: UVs {sh.ex.rounds('UVs')}, Ds {sh.ex.rounds('Ds')}")


def test_transpose_derived_and_dot_test():
    maps = EM.load_maps(GO)
    L = maps.layout
    ex = Exchanger(maps)
    rng = np.random.default_rng(3)
    calls = {"UVs": lambda e, a, b: e.EXCH_UV_XY_RL(a, b, True), "UVn": lambda e, a, b: e.EXCH_UV_XY_RL(a, b, False),
             "UV3s": lambda e, a, b: e.EXCH_UV_3D_RL(a, b, True), "Ds": lambda e, a, b: e.EXCH_UV_DGRID_3D_RL(a, b, True)}
    for kind, fn in calls.items():
        x = [rng.standard_normal(L.shape2d) for _ in range(2)]
        y = [rng.standard_normal(L.shape2d) for _ in range(2)]
        Ex, vjp = jax.vjp(lambda a, b: fn(ex, a, b), *(jnp.asarray(t) for t in x))
        ETy = vjp(tuple(jnp.asarray(t) for t in y))
        hand = rx2_transpose_numpy(maps.rx2[kind][1], maps.rx2[kind][0], y[0].reshape(-1), y[1].reshape(-1))
        # the same terms; only the order in which a source point's contributions are summed differs (XLA's
        # scatter-add vs np.add.at): measured max |difference| 4.4e-16 with |y| ~ 1, so 8 ulp of the largest term
        scale = 8 * np.finfo(np.float64).eps * max(np.abs(y[0]).max(), np.abs(y[1]).max())
        for a, h in zip(ETy, hand):
            np.testing.assert_allclose(np.asarray(a).reshape(-1), h, rtol=0, atol=scale)
        lhs = sum(float(np.sum(np.asarray(a) * b)) for a, b in zip(Ex, y))
        rhs = sum(float(np.sum(a * np.asarray(b))) for a, b in zip(x, ETy))
        assert abs(lhs - rhs) <= 1e-12 * abs(lhs), (kind, lhs, rhs)
    # the zero-coefficient terms are in the transpose: an Inf cotangent on a u halo point written by the last pass
    # from an interior source puts 0*Inf = NaN into v-bar at that source (a copy's transpose leaves v-bar finite)
    swap, passes = maps.rx2["UVs"]
    src, w, s1, s2 = passes[1][1]
    p = int(np.nonzero(w & (s1 == 1) & ~passes[0][1][1][src] & ~passes[0][2][1][src])[0][0])
    yu = np.zeros(L.npoints)
    yu[p] = np.inf
    _, vjp = jax.vjp(lambda a, b: ex.EXCH_UV_XY_RL(a, b, True), jnp.zeros(L.shape2d), jnp.zeros(L.shape2d))
    bu, bv = vjp((jnp.asarray(yu.reshape(L.shape2d)), jnp.zeros(L.shape2d)))
    bv = np.asarray(bv).reshape(-1)
    assert np.isnan(bv[src[p]]), "the 0*v term is missing from the transpose"
    hu, hv = rx2_transpose_numpy(passes, swap, yu, np.zeros(L.npoints))
    assert np.array_equal(np.isnan(hv), np.isnan(bv)) and np.array_equal(np.isnan(hu), np.isnan(np.asarray(bu).reshape(-1)))
    _, vjp_copy = jax.vjp(lambda a, b: Exchanger(dataclasses.replace(maps, rx2={})).EXCH_UV_XY_RL(a, b, True),
                          jnp.zeros(L.shape2d), jnp.zeros(L.shape2d))
    assert not np.isnan(np.asarray(vjp_copy((jnp.asarray(yu.reshape(L.shape2d)), jnp.zeros(L.shape2d)))[1])).any()
