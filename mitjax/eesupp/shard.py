"""Host side of a tile-sharded run: mesh, padding and placement, and the shard_map wrapper (plan Task 7b, L-COPY-3).

    sh = TileSharding(maps, nproc=4)              # mesh, padded blocks, sharded exchanger placed on the mesh
    f = sh.shard_map(body, in_specs=(sh.TILES, sh.REP, sh.TILES), out_specs=sh.TILES)   # jit(shard_map(check_vma))
    out = sh.unpad(f(sh.put_tiles(a), p, sh.ex))
    g4 = sh.put_tree(grid)                        # a host-built pytree (the Grid) placed: tiled FArrays sharded

Every [tile, ...] array is padded from nTiles to Tpad = P*ceil(nTiles/P) tiles (padding = replicas of tile 1,
sharded_exchange.TileBlocks) and split into P contiguous blocks. Inside the body the kernels see the local block and
`ex` (the ShardedExchanger: ppermute exchanges, psum-based fixed-order global sums); everything else is the
single-device code (one code path, PORTING_RULES §4). With P=1 the same program runs on one device. Replicated inputs
(parameters, vertical profiles) must not carry a tile axis ([L-PAR-6]: `check_replicated`).

The ECCO port's `parallel/shard.py` also held the V4r4 sharded FORWARD_STEP driver (ShardedModel); mitjax has no
forward step yet, so only its model-independent part is copied; the driver arrives with the first forward step (M1)
and the gradient driver under shard_map with Task 17 ([L-PAR-11]: sharded gradients are gated on GPUs).
"""

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from mitjax.eesupp.sharded_exchange import AXIS, ShardedExchanger, TileBlocks
from mitjax.farray import FArray


def tile_mesh(nproc, devices=None, axis_name=AXIS):
    devs = list(jax.devices() if devices is None else devices)
    if len(devs) < nproc:
        raise ValueError(f"{nproc} devices requested, {len(devs)} available")
    return Mesh(np.array(devs[:nproc]), (axis_name,))


def is_tile_field(a, layout):
    """[nTiles, ..., ny, nx] array (a per-tile model field)."""
    s = np.shape(a)
    return len(s) >= 3 and s[0] == layout.nTiles and tuple(s[-2:]) == (layout.ny, layout.nx)


class TileSharding:
    """Mesh, tile blocks and the placed ShardedExchanger of a P-device run of one layout."""

    def __init__(self, maps, nproc, devices=None, axis_name=AXIS, names=None):
        self.layout = maps.layout
        self.P = nproc
        self.axis = axis_name
        self.blocks = TileBlocks(self.layout.nTiles, nproc)
        self.mesh = tile_mesh(nproc, devices, axis_name)
        self.TILES = PartitionSpec(axis_name)
        self.REP = PartitionSpec()
        self._tile = NamedSharding(self.mesh, self.TILES)
        self.ex = ShardedExchanger.build(maps, self.blocks, names=names, axis_name=axis_name).device_arrays(self.mesh)

    def put_tiles(self, a):
        """[nTiles, ...] -> padded [Tpad, ...] placed with the tile axis sharded."""
        return jax.device_put(self.blocks.pad(np.asarray(a)), self._tile)

    def put_padded(self, a):
        """An already padded [Tpad, ...] array placed with the tile axis sharded."""
        a = np.asarray(a)
        if a.shape[0] != self.blocks.Tpad:
            raise ValueError(f"leading axis {a.shape[0]} is not the padded tile axis ({self.blocks.Tpad})")
        return jax.device_put(a, self._tile)

    def unpad(self, a):
        """Padded [Tpad, ...] -> [nTiles, ...] numpy (padding tiles dropped)."""
        return np.asarray(a)[:self.layout.nTiles]

    def put_tree(self, tree):
        """A host-built pytree (the Grid of INITIALISE_FIXED, built once on the host: see below) placed on the mesh
        as the drivers take it: the storage of every tiled FArray padded and sharded on its tile axis (`put_tiles`;
        padding tiles are copies of tile 1), every other leaf (untiled FArrays such as the vertical profiles, scalars)
        replicated. The values are copied, never recomputed, so the placed tree is the host tree bit for bit.

        Why the grid is built once on the host and then placed (lane B, 2026-10-01, job 27829493,
        scripts/grid_sharding/measure.py, baroclinic gyre / global_ocean / optim): the eager host build takes
        0.61-0.62 s warm (10.9-15.1 s cold, the first-call compiles of the eager operations), placing the whole
        Grid on 4 devices 18-22 ms for 7.0-17.0 MB, with 0 differing bits. A per-device build would have to run the
        builder inside one compiled sharded program, and the builder does not trace: it is host code by design
        (ini_vertical_grid.py:85 takes Python floats of rC/rF; the builders allocate all nSx*nSy*nPx*nPy tiles of
        SIZE.h and read the input files on the host), so building per device means rewriting the core lane's
        builders for no gain. Gate: mitjax/tests/test_grid_sharded.py (P=4 and padded P=3 == P=1 bitwise)."""
        rep = NamedSharding(self.mesh, self.REP)

        def one(x):
            if isinstance(x, FArray):
                if x.tiled:
                    return FArray(self.put_tiles(x.data), x.name, tiled=True, _dims=x.dims)
                return FArray(jax.device_put(np.asarray(x.data), rep), x.name, tiled=False, _dims=x.dims)
            return jax.device_put(np.asarray(x), rep)
        return jax.tree.map(one, tree, is_leaf=lambda x: isinstance(x, FArray))

    def unpad_tree(self, tree):
        """The inverse of put_tree on the host: tiled FArrays gathered and unpadded to [nTiles, ...] numpy, the rest
        gathered (numpy)."""
        def one(x):
            if isinstance(x, FArray):
                d = self.unpad(x.data) if x.tiled else np.asarray(x.data)
                return FArray(d, x.name, tiled=x.tiled, _dims=x.dims)
            return np.asarray(x)
        return jax.tree.map(one, tree, is_leaf=lambda x: isinstance(x, FArray))

    def check_replicated(self, tree):
        bad = [np.shape(a) for a in jax.tree.leaves(tree) if is_tile_field(a, self.layout)]
        if bad:
            raise ValueError(f"replicated leaves with a tile axis (would be replicated, not sharded): {bad}")

    def shard_map(self, body, in_specs, out_specs):
        """jit(shard_map(body, check_vma=True)) on this mesh ([L-PAR-10]: never check_vma=False)."""
        return jax.jit(jax.shard_map(body, mesh=self.mesh, in_specs=in_specs, out_specs=out_specs, check_vma=True))
