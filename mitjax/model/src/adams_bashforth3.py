"""ADAMS_BASHFORTH3: model/src/adams_bashforth3.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j


def adams_bashforth3(kArg, kSize, gTracer, gTrNm, AB_gTr, startAB, myIter, *, cfg, params):
    """ADAMS_BASHFORTH3( bi, bj, kArg, kSize, gTracer, gTrNm, AB_gTr, startAB, myIter, myThid )
    @63cdc0b model/src/adams_bashforth3.F:6-132

    C     | S/R ADAMS_BASHFORTH3
    C     | o Extrapolate forward in time using third order
    C     |   Adams-Bashforth method.
    C     | Either apply to tendency (kArg>0) at level k=kArg,
    C     |     or apply to state variable (kArg=0) for all levels
    C Extrapolate forward in time using 2 A.B. parameters (alpha,beta),
    C either tendency gX :
    C gX^{n+1/2} = (1 + \\alpha + \\beta) gX^{n} - (\\alpha + 2 \\beta) gX^{n-1} + \\beta  gX^{n-2}
    C     kArg    :: if >0: apply AB on tendency at level k=kArg
    C             :: if =0: apply AB on state variable and process all levels
    C     kSize   :: 3rd dimension of gTracer
    C     gTracer ::  in: Tendency/State at current time
    C             :: out(kArg >0): Extrapolated Tendency at current time
    C     gTrNm   ::  in: Tendency/State at previous time
    C             :: out(kArg >0): Save tendency at current time
    C     AB_gTr  :: Adams-Bashforth tendency increment
    C     startAB :: number of previous time level available to start/restart AB
    C     myIter  :: Current time step number

    Returns (gTracer, gTrNm, AB_gTr). gTracer: (1-OLx:sNx+OLx,1-OLy:sNy+OLy,kSize); gTrNm: the DYNVARS.h pair
    gtNm(...,Nr,nSx,nSy,2) as a tuple of two FArrays (Fortran m = 1, 2 at positions 0, 1; mitjax/model/state.py);
    AB_gTr: (1-OLx:sNx+OLx,1-OLy:sNy+OLy). kArg, kSize, startAB static; myIter traced (int32), so the slot indices
    m1 = 1+MOD(myIter+1,2), m2 = 1+MOD(myIter,2) (:80-81) select the pair's members with a `where` on the parity
    of myIter (myIter >= nIter0 >= 0 in every forward run, where MOD equals the non-negative remainder), and the
    weights (:84-97) are `where`s of finite scalars on the traced start tests. Only the kArg > 0 branch (:113-127,
    what TEMP_INTEGRATE / SALT_INTEGRATE call with AdamsBashforthGt/Gs) is ported; kArg = 0 (:101-112, AB on the
    tracer, AdamsBashforth_T/S) raises. `0. _d 0` exact, `2.` REAL*4 exact. The i,j points are independent."""
    if kArg == 0:
        raise NotImplementedError("ADAMS_BASHFORTH3: kArg = 0 (AB on the state variable, :101-112) is not ported")
    sz = cfg.size
    even = (myIter % 2) == 0                    # m2 = 1+MOD(myIter,2) = 1, m1 = 1+MOD(myIter+1,2) = 2 when even
    nIter0 = params.nIter0
    alph_AB, beta_AB = params.alph_AB, params.beta_AB
    start0 = (myIter == nIter0) & (startAB == 0)                                # :84
    start1 = ((myIter == nIter0) & (startAB == 1)) | ((myIter == 1+nIter0) & (startAB == 0))   # :88-89
    ab0 = jnp.where(start0, 0., jnp.where(start1, alph_AB, alph_AB + beta_AB))                 # :85, :90, :94
    ab1 = jnp.where(start0, 0., jnp.where(start1, -alph_AB, -alph_AB - 2.*beta_AB))            # :86, :91, :95
    ab2 = jnp.where(start0, 0., jnp.where(start1, 0., beta_AB))                                # :87, :92, :96

    kl = kArg                                                                   # :115
    k = min(kArg, kSize)                                                        # :116; MINMAX-INT: integer (no tie or NaN case)
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)                                         # :117
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)                                         # :118
    g1, g2 = gTrNm
    gm1 = jnp.where(even, g2[i, j, kl], g1[i, j, kl])                           # gTrNm(i,j,kl,bi,bj,m1)
    gm2 = jnp.where(even, g1[i, j, kl], g2[i, j, kl])                           # gTrNm(i,j,kl,bi,bj,m2)
    AB_gTr = AB_gTr.at[i, j].set(ab0*gTracer[i, j, k]                           # :119-121
                                 + ab1*gm1
                                 + ab2*gm2)
    new = gTracer[i, j, k]                                                      # :122  gTrNm(i,j,kl,bi,bj,m2) = gTracer
    g1 = g1.at[i, j, kl].set(jnp.where(even, new, g1[i, j, kl]))
    g2 = g2.at[i, j, kl].set(jnp.where(even, g2[i, j, kl], new))
    gTracer = gTracer.at[i, j, k].set(gTracer[i, j, k] + AB_gTr[i, j])          # :123
    return gTracer, (g1, g2), AB_gTr
