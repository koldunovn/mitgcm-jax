"""Global sums in the Fortran order (plan Task 7b; [E§5], L-COPY-4).

A MITgcm global sum has two stages, both strictly sequential in the Fortran:
1. the CALLER forms one partial sum per tile with its own `DO j; DO i` loops, e.g. `model/src/cg2d.F:157, 161-175`
   @63cdc0b: `errTile(bi,bj) = 0. _d 0`, then `errTile(bi,bj) = errTile(bi,bj) + cg2d_r(i,j,bi,bj)*cg2d_r(i,j,bi,bj)`
   for j = 1..sNy (outer), i = 1..sNx (inner) -> `tile_sum_fortran`;
2. GLOBAL_SUM_TILE_RL (`eesupp/src/global_sum_tile.F`) adds the tile partials. The oracle is a single-process build
   (no MPI), so the path at `global_sum_tile.F:198-204` runs: `sumMyPr = 0.` and `DO bj = 1,nSy; DO bi = 1,nSx;
   sumMyPr = sumMyPr + shareBufGSR8(bi,bj)` (bi inner: tile order 1, 2, ..., eesupp/tiles.py) -> `global_sum_tile`.
   With MPI and GLOBAL_SUM_ORDER_TILES (`eesupp/inc/CPP_EEOPTIONS.h:127` @63cdc0b, defined) the MPI path
   `global_sum_tile.F:161-191` gives the same additions: a zeroed buffer over all tiles, each process's partials put in
   it, MPI_Allreduce(SUM), then `sumAllP = 0.` + tiles in the same order. The result depends on the tile size (the
   partials), not on the process count.

Both stages are explicit add chains in that order: `jnp.sum` is a tree reduction whose association is not Fortran's
(test_global_sum.py shows it differs on the reference data). Sharded (`all_tiles`): the MPI path's zeroed buffer and
Allreduce are a `psum` of the zero-padded per-tile vector (each entry is its tile's partial plus zeros: exact, only a
-0 partial becomes +0, which no sum starting from `0.` can see), then the same ordered chain on every device: the
result is the same for every P. Copied from the ECCO port's `parallel/global_sum.py` and `core/cg2d.py:tile_sum`
(the c66g -> master audit found only cosmetic hunks in global_sum_tile.F: docs/AUDIT_C66G_MASTER.md, `global_sum`).

The partial sums start from +0 and every addition is an IEEE double add (under the gate XLA flags no FMA contracts a
caller's product into the add: the caller forms the product array, then `tile_sum_fortran` adds).
"""

import jax
import jax.numpy as jnp
from jax import lax


def tile_sum_fortran(a):
    """Per-tile partial sums in the Fortran loop order: a [T, nj, ni] (the points the caller's `DO j; DO i` loops
    visit, j outer, i inner) -> [T], `s = 0; DO j; DO i; s = s + a(i,j)`.

    The first row as an unrolled chain starting with `0. + a(1,1)` (`_zero_plus`), the other rows as a `lax.scan`
    over j with the i chain unrolled: every tile's chain is the same sequence of adds, so the result per tile does
    not depend on how many tiles the array holds (P=1 and P=N bitwise). The carry derives from `a`, so under
    shard_map it is typed like `a` (varying), as the body's output is."""
    a = jnp.asarray(a)
    if a.ndim != 3:
        raise ValueError(f"tile_sum_fortran takes [tile, j, i] arrays, got shape {a.shape}")
    nj, ni = a.shape[1], a.shape[2]
    s = _zero_plus(a[:, 0, 0])
    for i in range(1, ni):
        s = s + a[:, 0, i]

    def row(s, arow):          # arow [T, ni]: i = 1..ni in order
        for i in range(ni):
            s = s + arow[:, i]
        return s, None

    if nj > 1:
        s, _ = lax.scan(row, s, jnp.moveaxis(a[:, 1:], 1, 0))
    return s


@jax.custom_jvp
def _zero_plus(x):
    """The IEEE value of `0. + x` (x itself, except -0 -> +0), derivative 1. XLA:CPU folds an addition of a constant 0
    away under jit even with algsimp disabled (measured 2026-10-01: jit(lambda x: 0.0 + x)(-0.0) is -0.0), so the
    first addition of an ordered sum is this select; its custom JVP is the identity (the select's own derivative
    would be 0 at x == 0, where d(0. + x)/dx = 1)."""
    return jnp.where(x == 0, jnp.zeros((), x.dtype), x)


_zero_plus.defjvps(lambda t, ans, x: t)


def global_sum_tile(phiTile):
    """GLOBAL_SUM_TILE_RL, path without MPI (global_sum_tile.F:199-204): 0. + tile 1 + tile 2 + ... in order.
    phiTile: [nTiles, ...] partial sums of every tile, tile order."""
    phiTile = jnp.asarray(phiTile)
    s = _zero_plus(phiTile[0])
    for t in range(1, phiTile.shape[0]):
        s = s + phiTile[t]
    return s


def all_tiles(per_tile, blocks, axis_name):
    """Inside shard_map: per_tile [Tloc, ...] values of this device's tiles -> [nTiles, ...] values of every real tile
    in tile order, the same (and typed invariant) on every device; padding tiles are dropped. global_sum_tile.F:164-183
    (zeroed buffer, own partials, MPI_Allreduce SUM) as a psum of the zero-padded block vector. Differentiable (psum
    transposes to a broadcast)."""
    d = lax.axis_index(axis_name)
    full = jnp.zeros((blocks.Tpad,) + per_tile.shape[1:], per_tile.dtype)
    full = lax.dynamic_update_slice_in_dim(full, per_tile, d * blocks.Tloc, axis=0)
    return lax.psum(full, axis_name)[:blocks.nTiles]


def global_sum_rl(a, ex=None):
    """_GLOBAL_SUM_RL of a field the caller sums with `DO j; DO i`: a [T, nj, ni] (the summed points of every tile
    held) -> scalar. Through the exchanger `ex` (eesupp/exchange.py, sharded_exchange.py) when given: every device and
    every P the same value; None: all tiles are in `a` (single device)."""
    part = tile_sum_fortran(a)
    return global_sum_tile(part) if ex is None else ex.global_sum_tile(part)
