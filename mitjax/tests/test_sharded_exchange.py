"""mitjax/eesupp/{sharded_exchange,shard,global_sum}.py (plan Task 7b), tier 1x: the sharded exchanges and global sums
inside `shard_map(check_vma=True)` on fake CPU devices (forward only, [L-PAR-11]).

Gates: the Fortran probe reproduced bitwise through the sharded path at P=4 on every M1 layout (1, 2 and 4-tile
layouts are padded: replicas of tile 1); P=4 and P=3 == P=1 bitwise on random multi-level data with -0; the sharded
transpose == the single-device one, padding cotangents zero; global sums P=4 == gfortran (the reference of
test_global_sum.py) bitwise; only ppermute (one per colour round) and psum collectives.
"""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax import paths
from mitjax.eesupp import exch_maps as EM
from mitjax.eesupp.exchange import Exchanger
from mitjax.eesupp.shard import TileSharding
from mitjax.tests.exch_gate import ROUTINES, bits, probe_dumps, probe_mismatches

LAYOUTS = tuple(EM.M1_PROBES)
GSUM_REF = paths.REFERENCE / "exch_maps" / "global_sum_ref-job27827321"


def sharded_apply(sh):
    """apply(fn, ex, u, v) for exch_gate.probe_mismatches: fn on the padded, sharded arrays inside shard_map at sh.P."""
    def apply(fn, _ex, u, v):
        f = sh.shard_map(lambda e, u, v: fn(e, u, v), in_specs=(sh.TILES, sh.TILES, sh.TILES), out_specs=sh.TILES)
        return tuple(sh.unpad(o) for o in f(sh.ex, sh.put_tiles(u), sh.put_tiles(v)))
    return apply


@pytest.mark.parametrize("exp", LAYOUTS)
def test_sharded_probe_gate_p4(exp):
    """Every routine and signed-zero probe through the ShardedExchanger at P=4 == the Fortran probe, bitwise."""
    maps = EM.load_maps(exp)
    ds, it = probe_dumps(exp)
    sh = TileSharding(maps, 4)
    bad = probe_mismatches(Exchanger(maps), ds, it, apply=sharded_apply(sh))
    print(f"{exp}: Tloc={sh.blocks.Tloc} Tpad={sh.blocks.Tpad}, rounds XY={sh.ex.rounds('XY')} "
          f"UVs={sh.ex.rounds('UVs')}; mismatches {sum(bad.values())}")
    assert sum(bad.values()) == 0, {k: v for k, v in bad.items() if v}


def _random_fields(L, nk, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((L.nTiles, nk, L.ny, L.nx))
    a[rng.random(a.shape) < 0.05] = -0.0
    return a


@pytest.mark.parametrize("P", [3, 4])
def test_p_equals_p1_random(P):
    """Every routine at P (padding where nTiles is not a multiple of P) == P=1, bitwise (sign of zero included), on
    random [tile, k=2, j, i] fields (the 2-D routines on level 0) with 5 % -0, every M1 layout."""
    for exp in LAYOUTS:
        maps = EM.load_maps(exp)
        L = maps.layout
        u, v = _random_fields(L, 2, 1), _random_fields(L, 2, 2)
        sh = TileSharding(maps, P)
        apply_p = sharded_apply(sh)
        ex1 = Exchanger(maps)
        for flds, fn in ROUTINES:
            g2 = (lambda e, u, v, fn=fn: fn(e, u[:, 0], v[:, 0]))          # the probe's call on level 0
            one = jax.jit(g2)(ex1, u, v)
            many = apply_p(g2, None, u, v)
            for f, a, b in zip(flds, one, many):
                assert np.array_equal(bits(a), bits(b)), (exp, P, f)


def test_sharded_transpose_and_padding():
    """vjp of the sharded exchanges (P=4) == the single-device vjp (<= 1e-15 of the max: a point's copies are summed in
    another order), with the padding cotangent zeroed
    (`zero_padding`); the gradient on padding positions is exactly 0 (no map source is a padding tile). Layouts with
    padding at P=4: barotropic (1 tile + 3 replicas), advect_xy (2 + 2); without: global_ocean (36)."""
    for exp in ("tutorial_barotropic_gyre", "advect_xy", "global_ocean.90x40x15"):
        maps = EM.load_maps(exp)
        L = maps.layout
        sh = TileSharding(maps, 4)
        u, v = _random_fields(L, 1, 3)[:, 0], _random_fields(L, 1, 4)[:, 0]
        wu, wv = _random_fields(L, 1, 5)[:, 0], _random_fields(L, 1, 6)[:, 0]
        ex1 = Exchanger(maps)

        def loss1(u, v):
            a, b = ex1.EXCH_UV_XY_RL(u, v, True)
            c = ex1.EXCH_XY_RL(u)
            return jnp.vdot(a, wu) + jnp.vdot(b, wv) + jnp.vdot(c, wu)

        g1 = jax.jit(jax.grad(loss1, argnums=(0, 1)))(u, v)

        def body(e, u, v, wu, wv):
            def loc(u, v):
                a, b = e.EXCH_UV_XY_RL(u, v, True)
                c = e.EXCH_XY_RL(u)
                return a, b, c
            out, pull = jax.vjp(loc, u, v)
            cts = tuple(e.zero_padding(w) for w in (wu, wv, wu))
            return pull(cts)

        f = sh.shard_map(body, in_specs=(sh.TILES,) * 5, out_specs=sh.TILES)
        gp = f(sh.ex, *(sh.put_tiles(x) for x in (u, v, wu, wv)))
        for a, b in zip(g1, gp):
            full = np.asarray(b)
            a = np.asarray(a)
            err = np.max(np.abs(a - full[:L.nTiles])) / np.max(np.abs(a))
            ndiff = int(np.sum(bits(a) != bits(full[:L.nTiles])))
            print(f"{exp}: sharded vs single-device transpose: {ndiff} points differ, max {err:.2e} of the max")
            assert err <= 1e-15, (exp, err)      # the scatter-adds sum a point's copies in another order
            assert np.all(full[L.nTiles:] == 0.0), exp


def _gsum_ref():
    spec = importlib.util.spec_from_file_location("_gsumref_make_input", GSUM_REF / "make_input.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    cases = mk.cases()
    return cases, mk.decode_output((GSUM_REF / "output.bin").read_bytes(), cases)


def test_global_sums_p4_equal_fortran():
    """ShardedExchanger.global_sum_rl at P=4 on the six M1 tile shapes (data in the interior of full tile arrays) ==
    gfortran's ordered sums bitwise, plain and of products; global_max == the single-device max; the result is
    replicated (out_specs P())."""
    cases, ref = _gsum_ref()
    by_shape = {}
    for exp in LAYOUTS:
        maps = EM.load_maps(exp)
        by_shape[(maps.layout.nTiles, maps.layout.sNy, maps.layout.sNx)] = maps
    done = []
    for (a, b), (_, sA, _, sAB) in zip(cases, ref):
        if a.shape not in by_shape:
            continue
        maps = by_shape[a.shape]
        exp = maps.meta.get("exp")
        done.append(exp)
        L = maps.layout
        full = np.zeros(L.shape2d)
        fullb = np.zeros(L.shape2d)
        full[:, L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx] = a
        fullb[:, L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx] = b
        sh = TileSharding(maps, 4)
        inner = (slice(None), slice(L.OLy, L.OLy + L.sNy), slice(L.OLx, L.OLx + L.sNx))

        def body(e, x, y):
            return e.global_sum_rl(x[inner]), e.global_sum_rl((x * y)[inner]), e.global_max(x)

        f = sh.shard_map(body, in_specs=(sh.TILES,) * 3, out_specs=sh.REP)
        s1, s2, mx = f(sh.ex, sh.put_tiles(full), sh.put_tiles(fullb))
        assert bits(s1) == bits(sA) and bits(s2) == bits(sAB), exp
        assert float(mx) == float(np.max(full)), exp
    assert sorted(done) == sorted(LAYOUTS), done


def test_collectives_are_ppermute_and_psum():
    """The jaxpr of a sharded vector exchange plus a global sum at P=4 (global_ocean, 36 tiles) holds ppermutes (one
    per colour round of the exchange) and psums only: no all_to_all, all_gather or ragged_all_to_all."""
    maps = EM.load_maps("global_ocean.90x40x15")
    sh = TileSharding(maps, 4)
    L = maps.layout

    def body(e, u, v):
        a, b = e.EXCH_UV_XY_RL(u, v, True)
        return a, b, e.global_sum_rl(a[:, L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx])

    f = jax.shard_map(body, mesh=sh.mesh, in_specs=(sh.TILES,) * 3, out_specs=(sh.TILES, sh.TILES, sh.REP),
                      check_vma=True)
    z = sh.put_tiles(np.zeros(L.shape2d))
    txt = str(jax.make_jaxpr(f)(sh.ex, z, z))
    n_perm = txt.count("ppermute[")
    print(f"global_ocean P=4: {sh.ex.rounds('UVs')} rounds, {n_perm} ppermutes, {txt.count('psum')} psum")
    assert n_perm == sh.ex.rounds("UVs") > 0
    for banned in ("all_to_all", "all_gather", "ragged"):
        assert banned not in txt, banned
