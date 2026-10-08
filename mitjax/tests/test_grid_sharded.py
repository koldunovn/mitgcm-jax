"""P=4 == P=1 for the grid (plan Task 10's pending item; lane B, tier1x, fake CPU devices of the gate XLA flags).

Decision (measured, job 27829493; numbers in TileSharding.put_tree's docstring): the grid is built once on the host by the single-device builder
(mitjax/tests/grid_gate.build_grid: INI_GRID ... INI_CORI, eager, gated bitwise against the oracle in test_grid.py)
and then placed on the mesh by `TileSharding.put_tree` (mitjax/eesupp/shard.py): tiled FArrays padded and sharded on
the tile axis, untiled leaves replicated.

Gates, for every M1 variant with at least 4 tiles (baroclinic gyre 4 tiles, global_ocean 36, optim 2x2) at P=4, and
for the baroclinic gyre at P=3 (Tloc=2: two padding tiles):
* every device's shard of every tiled field is its block of the padded P=1 array, bit for bit (padding tiles are
  copies of tile 1, sharded_exchange.TileBlocks), every replicated leaf is the P=1 value on every device, and the
  gathered, unpadded tree is the P=1 grid bit for bit (every field, every point, halos included);
* inside the sharded program: the Grid passes through jit(shard_map(check_vma=True)) unchanged, and the sharded
  EXCH_XY_RL / EXCH_3D_RL of every 2-D / 3-D tiled grid field equal the single-device exchanges bit for bit (the
  placed tiles sit where the sharded exchanger expects them).
Negative controls (each measured to bite): tiles 1 and 2 swapped before placement; padding with zeros instead of
tile-1 copies (P=3); one field rounded to float32 on placement.
"""

import jax
import numpy as np
import pytest

from mitjax.eesupp.exch_maps import load_maps
from mitjax.eesupp.exchange import Exchanger
from mitjax.eesupp.shard import TileSharding
from mitjax.eesupp.sharded_exchange import TileBlocks
from mitjax.farray import FArray
from mitjax.tests import grid_gate as gg

VARIANTS = [("tutorial_baroclinic_gyre", "input"), ("global_ocean.90x40x15", "input"),
            ("tutorial_global_oce_optim", "input_ad")]
_GRIDS = {}


def host_grid(exp, inp):
    if (exp, inp) not in _GRIDS:
        _GRIDS[(exp, inp)] = jax.block_until_ready(gg.build_grid(exp, inp))
    return _GRIDS[(exp, inp)]


def _is_fa(x):
    return isinstance(x, FArray)


def _items(tree):
    """[(name, FArray or scalar)] of a Grid in its field order."""
    return [(k, getattr(tree, k)) for k in tree.names()]


def _ndiff(a, b):
    """Points where a and b differ in bits (dtype and shape must match; floats compared as bit patterns, so -0 and
    NaN payloads count)."""
    a, b = np.ascontiguousarray(np.asarray(a)), np.ascontiguousarray(np.asarray(b))
    if a.dtype != b.dtype or a.shape != b.shape:
        return max(a.size, b.size)
    if a.dtype.kind == "f":
        a, b = a.view(f"i{a.itemsize}"), b.view(f"i{b.itemsize}")
    return int(np.count_nonzero(a != b))


def placement_mismatches(sh, g1, g4):
    """{field: n differing points} between the placed tree g4 and the host tree g1: per-device shards against the
    padded P=1 blocks (tiled) or the whole P=1 value (replicated), the sharding of every leaf, and the gathered,
    unpadded tree against g1."""
    bad = {}
    T = sh.blocks.Tloc
    back = sh.unpad_tree(g4)
    for (name, x1), (_, x4), (_, xb) in zip(_items(g1), _items(g4), _items(back)):
        d1 = np.asarray(x1.data if _is_fa(x1) else x1)
        d4 = x4.data if _is_fa(x4) else x4
        db = np.asarray(xb.data if _is_fa(xb) else xb)
        n = 0
        tiled = _is_fa(x1) and x1.tiled
        padded = sh.blocks.pad(d1) if tiled else None
        devs = list(sh.mesh.devices.flat)
        for s in d4.addressable_shards:
            want = padded[devs.index(s.device) * T:(devs.index(s.device) + 1) * T] if tiled else d1
            n += _ndiff(s.data, want)
        if tiled != (not d4.sharding.is_fully_replicated):
            n += d1.size
        n += _ndiff(db, d1)
        if n:
            bad[name] = n
    return bad


def _specs(sh, g):
    return jax.tree.map(lambda x: sh.TILES if (_is_fa(x) and x.tiled) else sh.REP, g, is_leaf=_is_fa)


def program_mismatches(sh, g1, g4, maps):
    """{field: n differing points}: the Grid through an identity shard_map, and the sharded EXCH_XY_RL / EXCH_3D_RL
    of every 2-D / 3-D tiled field against the single-device Exchanger."""
    L = maps.layout
    bad = {}
    spec = _specs(sh, g4)
    gi = sh.unpad_tree(sh.shard_map(lambda g: g, in_specs=(spec,), out_specs=spec)(g4))
    for (name, x1), (_, xi) in zip(_items(g1), _items(gi)):
        a, b = np.asarray(x1.data if _is_fa(x1) else x1), np.asarray(xi.data if _is_fa(xi) else xi)
        n = _ndiff(a, b)
        if n:
            bad[f"identity:{name}"] = n
    names = [k for k, x in _items(g1) if _is_fa(x) and x.tiled and x.data.shape[-2:] == (L.ny, L.nx)
             and x.data.ndim in (3, 4)]

    def body(e, *fs):
        return tuple(e.EXCH_XY_RL(f) if f.ndim == 3 else e.EXCH_3D_RL(f) for f in fs)
    f = sh.shard_map(body, in_specs=(sh.TILES,) * (1 + len(names)), out_specs=(sh.TILES,) * len(names))
    outs = f(sh.ex, *(getattr(g4, k).data for k in names))
    ex1 = Exchanger(maps)
    for k, o in zip(names, outs):
        d = np.asarray(getattr(g1, k).data)
        ref = np.asarray(ex1.EXCH_XY_RL(d) if d.ndim == 3 else ex1.EXCH_3D_RL(d))
        n = _ndiff(sh.unpad(o), ref)
        if n:
            bad[f"exch:{k}"] = n
    assert len(names) >= 20, names
    return bad


@pytest.mark.parametrize("exp,inp", VARIANTS, ids=[e for e, _ in VARIANTS])
def test_grid_P4_equals_P1(exp, inp):
    g1 = host_grid(exp, inp)
    maps = load_maps(exp)
    sh = TileSharding(maps, 4)
    assert maps.layout.nTiles >= 4 and len(jax.devices()) >= 4
    g4 = sh.put_tree(g1)
    assert placement_mismatches(sh, g1, g4) == {}
    assert program_mismatches(sh, g1, g4, maps) == {}
    print(f"{exp}: {len(g1.names())} fields, Tloc={sh.blocks.Tloc}, Tpad={sh.blocks.Tpad}")


def test_grid_P3_padding_equals_P1():
    exp, inp = VARIANTS[0]
    g1 = host_grid(exp, inp)
    maps = load_maps(exp)
    sh = TileSharding(maps, 3)
    assert (sh.blocks.Tloc, sh.blocks.Tpad) == (2, 6)
    g4 = sh.put_tree(g1)
    assert placement_mismatches(sh, g1, g4) == {}
    assert program_mismatches(sh, g1, g4, maps) == {}


class ZeroPad(TileBlocks):
    """Planted error: padding tiles hold zeros instead of copies of tile 1."""

    def pad(self, a):
        p = np.array(super().pad(a), copy=True)
        p[self.nTiles:] = 0
        return p


def test_grid_sharding_negative_controls():
    exp, inp = VARIANTS[0]
    g1 = host_grid(exp, inp)
    maps = load_maps(exp)
    # 1. tiles 1 and 2 swapped before placement (P=4)
    sh = TileSharding(maps, 4)
    perm = jax.tree.map(lambda x: FArray(np.asarray(x.data)[[1, 0, 2, 3]], x.name, tiled=True, _dims=x.dims)
                        if (_is_fa(x) and x.tiled) else x, g1, is_leaf=_is_fa)
    g4 = sh.put_tree(perm)
    bad = placement_mismatches(sh, g1, g4)
    # measured: 22 fields differ (R_low, hFac*, masks, kSurf*/kLowC, xC, xG, ...; fields uniform across tiles do not)
    assert len(bad) >= 20 and {"R_low", "hFacC", "maskC", "xC"} <= set(bad), bad
    assert program_mismatches(sh, g1, g4, maps), "swapped tiles not seen by the sharded exchanges"
    # 2. padding tiles zero instead of copies of tile 1 (P=3)
    sh3 = TileSharding(maps, 3)
    sh3.blocks = ZeroPad(sh3.blocks.nTiles, sh3.blocks.P)
    bad = placement_mismatches(TileSharding(maps, 3), g1, sh3.put_tree(g1))
    # measured: 59 fields differ (every tiled field nonzero on tile 1)
    assert len(bad) >= 40, bad
    # 3. one field rounded to float32 on placement
    g32 = g1.replace(dxC=FArray(np.asarray(g1.dxC.data).astype(np.float32).astype(np.float64), "dxC", tiled=True,
                                _dims=g1.dxC.dims))
    bad = placement_mismatches(sh, g1, sh.put_tree(g32))
    assert set(bad) == {"dxC"}, bad
