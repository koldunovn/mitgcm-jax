"""Tile layout of an experiment and the tile identity of the arrays a kernel holds (plan Task 7b).

Arrays are `[tile, j, i]` or `[tile, k, j, i]` with halos (PORTING_RULES §4): MITgcm's
`(1-OLx:sNx+OLx, 1-OLy:sNy+OLy, [Nr,] nSx, nSy)` with the tile axis first; Fortran index i at array index i-1+OLx.
Tiles are in the experiment's order: the W2 tile number under pkg/exch2, `bi + (bj-1)*nSx` under the eesupp (exch1)
exchange. With one process (nPx = nPy = 1, every M1 build) both are the order of the `DO bj; DO bi` loops:
exch2's W2_myTileList(bi,bj) = bi + (bj-1)*nSx for the default ordering (`pkg/exch2/w2_map_procs.F:73-94` @63cdc0b
with no blank tiles), and the jaxdump header numbers exch1 tiles `1 + tBasex/sNx + (tBasey/sNy)*nSx*nPx`
(reference/jaxdump/SUBSTEPS.md). Global sums add the tiles in this order (eesupp/global_sum.py).

Kernels work on "the tiles in the arrays they are given": all tiles on one device, one block per device under
`shard_map` (eesupp/shard.py, sharded_exchange.py: padding tiles replicate tile 1). A kernel holding a per-tile table
indexed by the global tile number takes its own rows with `tile_rows(table, tile_index)`; `tile_index` is None on the
single-device path (all tiles in order), so that path is the same program as without sharding. Copied from the ECCO
port's `parallel/tiles.py` (L-COPY-3) without its Grid dependency; LLC/13-tile assumptions removed.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class TileLayout:
    """sNx, sNy, OLx, OLy from the experiment's SIZE.h; nTiles = nSx*nSy*nPx*nPy (or exch2_nTiles)."""
    sNx: int
    sNy: int
    OLx: int
    OLy: int
    nTiles: int

    @property
    def nx(self):
        """Padded extent in x (1-OLx .. sNx+OLx)."""
        return self.sNx + 2 * self.OLx

    @property
    def ny(self):
        return self.sNy + 2 * self.OLy

    @property
    def shape2d(self):
        return (self.nTiles, self.ny, self.nx)

    @property
    def npoints(self):
        """Points of one level of all tiles, halos included (the length of a flattened map)."""
        return self.nTiles * self.ny * self.nx

    def interior(self):
        """[ny, nx] bool: the Fortran points i = 1..sNx, j = 1..sNy."""
        m = np.zeros((self.ny, self.nx), bool)
        m[self.OLy:self.OLy + self.sNy, self.OLx:self.OLx + self.sNx] = True
        return m

    def tag(self):
        """File-name tag of the layout, e.g. t4_31x31_ol2x2."""
        return f"t{self.nTiles}_{self.sNx}x{self.sNy}_ol{self.OLx}x{self.OLy}"

    def as_array(self):
        return np.array([self.sNx, self.sNy, self.OLx, self.OLy, self.nTiles], np.int64)

    @classmethod
    def from_array(cls, a):
        return cls(*(int(x) for x in a))


def n_tiles(a):
    """Number of tiles in a [tile, ...] array: all tiles on one device, the local block (padding included) under
    shard_map."""
    return np.shape(a)[0]


def tile_rows(table, tiles):
    """Rows of a per-tile table [nTiles, ...] (indexed by the 0-based global tile number) for the tiles held:
    `tiles` = int [T] (sharded: the local block, a padding tile carries its donor's number) or None for all tiles in
    order (single device; returns `table` itself)."""
    if tiles is None:
        return table
    return jnp.asarray(table)[tiles]
