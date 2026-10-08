"""PTRACERS_APPLY_FORCING of tutorial_global_oce_latlon: verification/tutorial_global_oce_latlon/code/
ptracers_apply_forcing.F @63cdc0b (the experiment's own version: age tracer)."""

from mitjax.farray import loop_i, loop_j
from mitjax.pkg.ptracers.ptracers_apply_forcing import check_unported, k_surface


def ptracers_apply_forcing(gPtracer, surfForcPtr, iMin, iMax, jMin, jMax, k, iTracer, myTime, myIter, *, cfg, grid,
                           params, ptr):
    """PTRACERS_APPLY_FORCING( gPtracer, surfForcPtr, iMin,iMax,jMin,jMax, k, bi, bj, iTracer, myTime, myIter,
    myThid )   @63cdc0b verification/tutorial_global_oce_latlon/code/ptracers_apply_forcing.F:7-136

    C     Apply passive tracer forcing, i.e., sources and sinks of tracer,
    C      by adding forcing terms to the tendency array

    pkg/ptracers' routine with the customized ELSE arm :100-109 (age tracer: in the interior, aging tendency one):
    on every level other than kSurface, gPtracer + 1. _d 0 * maskC(i,j,k) on 0:sNx+1, 0:sNy+1 (`1. _d 0 * maskC`
    kept literally). The surface level as pkg/ptracers' (:80-89)."""
    check_unported("PTRACERS_APPLY_FORCING", iTracer, cfg=cfg, params=params, ptr=ptr)
    kSurface = k_surface("PTRACERS_APPLY_FORCING", cfg=cfg, params=params)
    sz = cfg.size
    j = loop_j(0, sz.sNy+1)                                                     # :83, :103
    i = loop_i(0, sz.sNx+1)                                                     # :84, :104
    if k == kSurface:                                                           # :80
        gPtracer = gPtracer.at[i, j].set(gPtracer[i, j]                         # :85-87
                                         + surfForcPtr[i, j]
                                         * grid.recip_drF[k]*grid.recip_hFacC[i, j, k])
    else:                                                                       # :100-109
        gPtracer = gPtracer.at[i, j].set(gPtracer[i, j]                         # :105-106
                                         + 1.0*grid.maskC[i, j, k])
    return gPtracer
