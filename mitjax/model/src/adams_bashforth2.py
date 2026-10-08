"""ADAMS_BASHFORTH2: model/src/adams_bashforth2.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j


def adams_bashforth2(kArg, kSize, gTracer, gTrNm1, AB_gTr, startAB, myIter, *, cfg, params):
    """ADAMS_BASHFORTH2( bi, bj, kArg, kSize, gTracer, gTrNm1, AB_gTr, startAB, myIter, myThid )
    @63cdc0b model/src/adams_bashforth2.F:6-92

    C     | S/R ADAMS_BASHFORTH2
    C     | o Extrapolate tendencies forward in time using
    C     |   quasi-second order Adams-Bashforth method.
    C     kArg    :: if >0: level number to process ; =0: process all levels
    C     kSize   :: 3rd dimension of tracer and tendency arrays
    C     gTracer :: Tendency at current time  ( output: Extrapolated Tendency )
    C     gTrNm1  :: Tendency at previous time ( output: Tendency at current time )
    C     AB_gTr  :: Adams-Bashforth tracer tendency increment
    C     startAB :: number of previous time level available to start/restart AB
    C     myIter  :: Current time step number

    Returns (gTracer, gTrNm1, AB_gTr). gTracer, gTrNm1: FArrays (1-OLx:sNx+OLx,1-OLy:sNy+OLy,kSize) (the caller's
    gU(1-OLx,1-OLy,1,bi,bj) etc.: the whole 3-D field); AB_gTr: (1-OLx:sNx+OLx,1-OLy:sNy+OLy). kArg, kSize, startAB
    static; myIter traced (int32): the start test `myIter.EQ.nIter0 .AND. startAB.EQ.0` (:61) selects abFac as a
    `where` of two finite scalars. `0.5 _d 0` exact. Only the kArg > 0 branch (:80-88, what TIMESTEP calls) is ported;
    kArg = 0 (:69-78) raises. The i,j points are independent (each reads and writes its own point)."""
    if kArg == 0:
        raise NotImplementedError("ADAMS_BASHFORTH2: kArg = 0 (all levels, :69-78) is not ported")
    sz = cfg.size
    abFac = jnp.where((myIter == params.nIter0) & (startAB == 0),              # :61
                      0.,                                                       # :62  0. _d 0
                      0.5 + params.abEps)                                       # :64  0.5 _d 0 + abEps
    k = kArg                                                                    # :81
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)                                         # :82
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)                                         # :83
    AB_gTr = AB_gTr.at[i, j].set(abFac*(gTracer[i, j, k] - gTrNm1[i, j, k]))   # :84
    gTrNm1 = gTrNm1.at[i, j, k].set(gTracer[i, j, k])                           # :85
    gTracer = gTracer.at[i, j, k].set(gTracer[i, j, k] + AB_gTr[i, j])          # :86
    return gTracer, gTrNm1, AB_gTr
