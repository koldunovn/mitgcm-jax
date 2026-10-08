"""INI_GLOBAL_DOMAIN: model/src/ini_global_domain.F @63cdc0b (the 2-D domain sums, PTRACERS lane for INI_CG2D)."""

from mitjax.farray import loop_i, loop_j
from mitjax.eesupp.global_sum import tile_sum_fortran


def ini_global_domain_2d(*, cfg, grid, ex):
    """INI_GLOBAL_DOMAIN( myThid )   @63cdc0b model/src/ini_global_domain.F:7-190, its 2-D part (:71-89, :114, :116)

    C     | o Initialise domain (i.e., where there is fluid)
    C     |   related (global) quantities.
    C     | Compute global domain Area ;

    Returns the GRID.h values (n2dWetPts, globalArea) INI_CG2D reads for cg2dTargetResWunit > 0 (ini_cg2d.F:152-162).
    Reads maskInC and rA from `grid`; `ex`: the experiment's exchanger (GLOBAL_SUM_TILE_RL in tile order).

    Each tile partial (:76-87) is the `DO j; DO i` chain from 0. of maskInC and of rA*maskInC (`tile_sum_fortran`);
    GLOBAL_SUM_TILE_RL (:88-89) adds the partials in tile order. Not ported (nothing ported reads them): the 3-D sums
    n3dWetPts and rAc_3dMean (:91-110, :115, :117-118), the empty-tile message lines (:119-139, STDOUT only) and the
    cubed-sphere corner count that sets hasWetCSCorners (:143-182)."""
    sz = cfg.size
    g = grid
    j = loop_j(1, sz.sNy)                                                       # :78
    i = loop_i(1, sz.sNx)                                                       # :79
    tileNwet = tile_sum_fortran(g.maskInC[i, j])                                # :76, :80
    tileArea = tile_sum_fortran(g.rA[i, j]*g.maskInC[i, j])                     # :77, :81-82
    loc2dNwet = ex.global_sum_tile(tileNwet)                                    # :88
    loc2dArea = ex.global_sum_tile(tileArea)                                    # :89
    n2dWetPts = loc2dNwet                                                       # :114
    globalArea = loc2dArea                                                      # :116
    return n2dWetPts, globalArea
