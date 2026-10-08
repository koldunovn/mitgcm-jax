"""COST_TRACER: pkg/cost/cost_tracer.F @63cdc0b."""

from mitjax.eesupp.global_sum import tile_sum_fortran
from mitjax.farray import loop_i, loop_j

_OPT = "COST_OPTIONS.h"


def cost_tracer(objf_tracer, *, cfg, grid, params, ptr, ptf, ex):
    """cost_tracer( bi, bj, myThid )   @63cdc0b pkg/cost/cost_tracer.F:3-57

    C     | subroutine cost_tracer                                   |
    C     | o this routine computes the cost function for the tiles  |
    C     |   of this processor                                      |

    `objf_tracer`: cost.h objf_tracer(nSx,nSy) as a [tile] array of every real tile (replicated under shard_map,
    as COST_FINAL's tile_fc); returns it with this call's locfc added per tile (:51): the per-tile locfc of the tiles
    held, gathered to every real tile in tile order by `ex.all_tiles` (identity on one device; under shard_map a
    psum of the zero-padded block vector, exact), so each objf_tracer(bi,bj) gets the one addition of the Fortran.
    locfc (:41-49) is the `DO j=1,sNy; DO i=1,sNx` chain from 0 of
    hFacC(i,j,1)*lambdaTr1ClimRelax*ptracer(i,j,1,1)*rA(i,j)*drF(1)*dTtracerLev(1) (left to right, as written;
    `tile_sum_fortran`). Compiled under ALLOW_COST_TRACER and ALLOW_PTRACERS (:35-36; otherwise an empty routine)."""
    if not (cfg.cpp.flag("ALLOW_COST_TRACER", _OPT) and cfg.cpp.flag("ALLOW_PTRACERS", _OPT)):   # :35-36
        return objf_tracer
    sz = cfg.size
    g = grid
    k = 1                                                                       # :42
    j = loop_j(1, sz.sNy)                                                       # :43
    i = loop_i(1, sz.sNx)                                                       # :44
    locfc = tile_sum_fortran(g.hFacC[i, j, k]                                   # :41, :45-47
                             * ptr.lambdaTr1ClimRelax*ptf.pTracer[0][i, j, k]
                             * g.rA[i, j]*g.drF[k]*params.dTtracerLev[k])
    return objf_tracer + ex.all_tiles(locfc)                                    # :51
