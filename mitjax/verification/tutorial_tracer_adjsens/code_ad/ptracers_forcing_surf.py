"""PTRACERS_FORCING_SURF of tutorial_tracer_adjsens: verification/tutorial_tracer_adjsens/code_ad/
ptracers_forcing_surf.F @63cdc0b (the experiment's own version: the tracer imitates salt)."""

from mitjax.eesupp.global_sum import _zero_plus
from mitjax.farray import loop_i, loop_j
from mitjax.pkg.ptracers.ptracers_forcing_surf import check_unported


def ptracers_forcing_surf(relaxForcingS, iMin, iMax, jMin, jMax, myTime, myIter, *, cfg, grid, params, ptr, ptf,
                          ff):
    """PTRACERS_FORCING_SURF( relaxForcingS, bi, bj, iMin, iMax, jMin, jMax, myTime, myIter, myThid )
    @63cdc0b verification/tutorial_tracer_adjsens/code_ad/ptracers_forcing_surf.F:7-198

    C     Precomputes surface forcing term for pkg/ptracers.
    C     Precomputation is needed because of non-local KPP transport term,
    C     routine KPP_TRANSPORT_PTR.

    pkg/ptracers' routine with line :67 live (the comment of pkg/ptracers' :67 is code here):
        surfaceForcingPTr(i,j,bi,bj,iTrc) = 0. _d 0
     &                        + surfaceForcingS(i,j,bi,bj)
    (`0. + x`: +0 where x = -0, kept literally: XLA folds a constant `0. +` away under jit, so the IEEE sum is
    mitjax/eesupp/global_sum._zero_plus, derivative 1). surfaceForcingS: FFIELDS.h (`ff`). The later branches as
    pkg/ptracers' (`check_unported`)."""
    check_unported("PTRACERS_FORCING_SURF", params=params, ptr=ptr)
    j = loop_j(jMin, jMax)                                                      # :64
    i = loop_i(iMin, iMax)                                                      # :65
    for iTrc in range(1, ptr.PTRACERS_numInUse+1):                              # :62
        s = ptf.surfaceForcingPTr[iTrc-1]
        s = s.at[i, j].set(_zero_plus(                                          # :66-67  0. _d 0 + x
                           ff.surfaceForcingS[i, j]))
        ptf = ptf.set("surfaceForcingPTr", iTrc, s)
    return ptf
