"""PTRACERS_FORCING_SURF of tutorial_global_oce_latlon: verification/tutorial_global_oce_latlon/code/
ptracers_forcing_surf.F @63cdc0b (the experiment's own version: age tracer)."""

from mitjax.farray import loop_i, loop_j
from mitjax.pkg.ptracers.ptracers_forcing_surf import check_unported


def ptracers_forcing_surf(relaxForcingS, iMin, iMax, jMin, jMax, myTime, myIter, *, cfg, grid, params, ptr, ptf,
                          ff=None):
    """PTRACERS_FORCING_SURF( relaxForcingS, bi, bj, iMin, iMax, jMin, jMax, myTime, myIter, myThid )
    @63cdc0b verification/tutorial_global_oce_latlon/code/ptracers_forcing_surf.F:7-201

    C     Precomputes surface forcing term for pkg/ptracers.
    C     Precomputation is needed because of non-local KPP transport term,
    C     routine KPP_TRANSPORT_PTR.

    pkg/ptracers' routine with the initialisation :66-70 replaced (age tracer: at the surface, 10-day relaxation
    towards zero):
        surfaceForcingPTr(i,j,bi,bj,iTrc) =
            + 1. _d 0 / (10. _d 0 * 86400. _d 0)
                      * ( 0. _d 0 - pTracer(i,j,ks,bi,bj,iTrc) )
                      * drF(ks) * _hFacC(i,j,ks,bi,bj)
    the unary + applies to the whole product, which is ((c*(0-p))*drF)*hFacC with c = 1/(10*86400) (constants
    folded by gfortran and by Python in double: 864000 exact, one correctly rounded division). ks = 1
    (:55-59, usingPCoords raises). The later branches as pkg/ptracers' (`check_unported`)."""
    check_unported("PTRACERS_FORCING_SURF", params=params, ptr=ptr)
    if params.usingPCoords:                                                     # :55-56
        raise NotImplementedError("PTRACERS_FORCING_SURF: usingPCoords (ks = Nr) is not ported")
    ks = 1                                                                      # :58
    j = loop_j(jMin, jMax)                                                      # :64
    i = loop_i(iMin, iMax)                                                      # :65
    c = 1.0/(10.0*86400.0)                                                      # :68  1. _d 0 / (10. _d 0 * 86400. _d 0)
    for iTrc in range(1, ptr.PTRACERS_numInUse+1):                              # :62
        p = ptf.pTracer[iTrc-1]
        s = ptf.surfaceForcingPTr[iTrc-1]
        s = s.at[i, j].set(c                                                    # :67-70
                           * (0.0-p[i, j, ks])
                           * grid.drF[ks]*grid.hFacC[i, j, ks])
        ptf = ptf.set("surfaceForcingPTr", iTrc, s)
    return ptf
