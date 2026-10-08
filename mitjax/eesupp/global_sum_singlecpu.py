"""GLOBAL_SUM_SINGLECPU_RL: eesupp/src/global_sum_singlecpu.F @63cdc0b (lane M4ADLAB session 3: lab_sea/code_ad defines
CG2D_SINGLECPU_SUM, verification/lab_sea/code_ad/CPP_EEOPTIONS.h:129).

The routine copies the interior points (i = 1..sNx, j = 1..sNy) of every tile into one global array (GATHER_2D_R8,
the exch1 layout without pkg/exch2: tile (bi,bj) at global columns (bi-1)*sNx+1.. and rows (bj-1)*sNy+1..) and process
0 adds all of it in global order, `sumAll = 0. _d 0; DO ij = 1, xSize*ySize; sumAll = sumAll + xy_buffer_r8(ij)`
(:142-146), ij = iG + (jG-1)*xSize: one chain over the global rows (outer) and columns (inner), not tile partials.
The tile number of the arrays is `bi + (bj-1)*nSx*nPx` (eesupp/tiles.py), so the tile at array index t sits at tile
column t mod (nSx*nPx) and tile row t div (nSx*nPx). Under pkg/exch2 (zeroBuff and the W2 global layout) the
routine is not ported (raises).
"""

import jax.numpy as jnp

from mitjax.eesupp.global_sum import tile_sum_fortran


def global_sum_singlecpu_rl(phiLocal, *, cfg, ex):
    """GLOBAL_SUM_SINGLECPU_RL( phiLocal, sumPhi, oLi, oLj, myThid ) with oLi = oLj = 0 (the cg2d calls):
    `phiLocal` [T, sNy, sNx] interior values of the tiles held, -> the scalar sumPhi, identical on every device.
    The chain `0. + a(1) + a(2) + ...` over the global array is tile_sum_fortran of the [1, Ny, Nx] array (the same
    sequence of IEEE adds, `0. + x` first)."""
    if "exch2" in cfg.packages:
        raise NotImplementedError("GLOBAL_SUM_SINGLECPU_RL: the pkg/exch2 layout (zeroBuff, exch2_global_Nx/Ny) "
                                  "is not ported")
    sz = cfg.size
    a = ex.all_tiles(jnp.asarray(phiLocal))                                    # [nTiles, sNy, sNx] in tile order
    ntx, nty = sz.nSx*sz.nPx, sz.nSy*sz.nPy
    if a.shape[0] != ntx*nty:
        raise ValueError(f"GLOBAL_SUM_SINGLECPU_RL: {a.shape[0]} tiles for a {ntx} x {nty} layout")
    g = a.reshape(nty, ntx, sz.sNy, sz.sNx).transpose(0, 2, 1, 3).reshape(1, nty*sz.sNy, ntx*sz.sNx)
    return tile_sum_fortran(g)[0]                                              # :142-146
