"""SEAICE_COST_TEST   @63cdc0b pkg/seaice/seaice_cost_test.F:3-232 (lane M4ADCOL)"""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MIN


def seaice_cost_test(cost, myTime, myIter, *, cfg, sp, AREA, HEFF, rA, endTime, startTime, lastinterval,
                     deltaTClock, ex=None):
    """SEAICE_COST_TEST( myTime, myIter, myThid )

    `cost`: the cost.h pytree with SEAICE_COST.h objf_ice [tile] (seaice_cost_init_varia). `AREA`, `HEFF`: SEAICE.h
    after SEAICE_MODEL; `rA`: GRID.h. `endTime`, `startTime`, `deltaTClock` (PARAMS.h) and `lastinterval` (cost.h,
    COST_READPARMS) are traced floats; `myTime` the step's (traced). The body (:74-223) is compiled under
    ALLOW_COST and ALLOW_COST_ICE (:47, :65). Ported arms: cost_ice_flag 1 (HEFF, :88-99) and 2 (AREA, :101-112, the
    1D_ocean_ice_column/input_ad value); flags 3-7 raise (not executed by a ported run); any other value is the STOP
    of :217-221. usingPCoords would only change kSrf (:74-78), which flags 1 and 2 do not read.
    Each tile's sum is the Fortran's DO j / DO i add chain (objf_ice(bi,bj) + tempVar*rA*AREA, point by point).

    Lane M4COSTSHARD s2 (sharded runs): objf_ice is a [nTiles] array of every real tile (replicated under shard_map,
    as cost.h's other plain [tile] leaves), the fields are the tiles held. With the exchanger `ex`, each held tile's
    chain starts from its own objf_ice(bi,bj) (`ex.tile_index()`: the global tile of every local position; None on
    one device, all tiles in order) and the [Tloc] results are gathered to every real tile in tile order by
    `ex.all_tiles` (identity on one device; under shard_map a psum of the zero-padded block vector, exact), so each
    objf_ice(bi,bj) gets the Fortran's chain at every P (cost_tracer.py's gather; here the chain starts from the
    accumulator itself, not from 0). ex=None: the single-device form (the identity steps left out)."""
    if not cfg.cpp.flag("ALLOW_COST_ICE", "SEAICE_OPTIONS.h"):         # :47, :65 (lane M4ADLAB: lab_sea/code_ad
        return cost                                                    # undefines it, SEAICE_OPTIONS.h:264)
    flag = int(sp.cost_ice_flag)
    if flag in (3, 4, 5, 6, 7):
        raise NotImplementedError(f"SEAICE_COST_TEST: cost_ice_flag = {flag} (:135-215) is not ported")
    if flag not in (1, 2):
        raise RuntimeError("SEAICE_COST_TEST: invalid cost_ice_flag\nABNORMAL END: S/R SEAICE_COST_TEST")
    sz = cfg.size
    # :79 IF ( myTime .GT. (endTime - lastinterval) ) THEN   (traced clock: the update is selected)
    active = myTime > (endTime - lastinterval)
    # tempVar = 1. _d 0/( ( 1. _d 0 + min(endTime-startTime,lastinterval) ) / deltaTClock )
    tempVar = 1.0 / ((1.0 + MIN(endTime - startTime, lastinterval, p="b")) / deltaTClock)   # :80-82
    fld = HEFF if flag == 1 else AREA                                  # :95 HEFF / :108 AREA
    idx = None if ex is None else ex.tile_index()
    o = cost.objf_ice if idx is None else cost.objf_ice[idx]          # objf_ice(bi,bj) of the tiles held
    for j in range(1, sz.sNy + 1):                                     # :92 / :105  DO j = 1,sNy
        for i in range(1, sz.sNx + 1):                                 # :93 / :106  DO i = 1,sNx
            jj, ii = loop_j(j, j), loop_i(i, i)
            o = o + tempVar * rA[ii, jj][:, 0, 0] * fld[ii, jj][:, 0, 0]   # :94-95 / :107-108
    if ex is not None:
        o = ex.all_tiles(o)                                            # every real tile, tile order
    cost.objf_ice = jnp.where(active, o, cost.objf_ice)
    return cost
