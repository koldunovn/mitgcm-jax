"""CONVECTIVE_ADJUSTMENT: model/src/convective_adjustment.F @63cdc0b (and the traced DIFFERENT_MULTIPLE it calls)."""

import jax.numpy as jnp

from mitjax.farray import FArray
from mitjax.model.src.convective_weights import convective_weights
from mitjax.model.src.convectively_mixtracer import convectively_mixtracer
from mitjax.model.src.find_rho import find_rho_2d
from mitjax.ops.safe import safe_div
from mitjax.pkg.ptracers.ptracers_convect import ptracers_convect


def different_multiple(freq, val1, step):
    """LOGICAL FUNCTION DIFFERENT_MULTIPLE( freq, val1, step )   @63cdc0b eesupp/src/different_multiple.F:7-65, on
    traced values (the host version on Python floats is mitjax/eesupp/different_multiple.py; the same operations in
    the same order). Used where the model clock is traced (KERNEL_GUIDE §4: a run-time test of the clock is a
    `where` over finite values).

    C     | o Checks if a multiple of freq exist
    C     |   around val1 +/- step/2

    NINT (:54) rounds half away from zero: floor(|q|) + (|q| - floor(|q|) >= 0.5), with the sign of q (exact in
    float64 for |q| < 2**52)."""
    nz = freq != 0.                                                             # :43
    q = safe_div(val1, freq, nz)                                                # :54  v1/freq
    a = jnp.abs(q)
    n = jnp.floor(a)
    n = jnp.where(a-n >= 0.5, n+1., n)
    n = jnp.where(q >= 0., n, -n)                                               # NINT(v1/freq)
    v1 = val1                                                                   # :49-51
    v2 = val1-step
    v3 = val1+step
    v4 = n*freq                                                                 # :54
    d1 = v1-v4                                                                  # :55-57
    d2 = v2-v4
    d3 = v3-v4
    near = (jnp.abs(d1) < jnp.abs(d2)) & (jnp.abs(d1) <= jnp.abs(d3))           # :58-59
    return nz & ((jnp.abs(step) > freq) | near)                                 # :44-46, :58-60


def _level(A, k):
    """A(1-OLx,1-OLy,k,bi,bj) passed to a 2-D dummy argument (sequence association): level k as (i, j)."""
    (_, ilo, ihi), (_, jlo, jhi), (_, klo, _) = A.dims
    kk = k - klo                                                    # k: a Python int or a traced level (KIdx)
    return FArray(A.data[:, getattr(kk, "value", kk)], A.name, i=(ilo, ihi), j=(jlo, jhi), tiled=A.tiled)


def convective_adjustment(myTime, myIter, *, cfg, grid, params, eos, state, ptr=None, ptf=None):
    """CONVECTIVE_ADJUSTMENT( bi, bj, myTime, myIter, myThid )   @63cdc0b model/src/convective_adjustment.F:10-184

    C     | SUBROUTINE CONVECTIVE_ADJUSTMENT
    C     | o Driver for vertical mixing or similar parameterization

    Returns (state, ptf): State with theta, salt adjusted; PTRACERS_FIELDS.h `ptf` with pTracer adjusted (ptf None
    without ptracers). Compiled under INCLUDE_CONVECT_CALL (:42; raise otherwise). `params`: PARAMS.h (cAdjFreq and
    deltaTClock traced, usingZCoords and usePTRACERS static); `eos`: EOS.h for FIND_RHO_2D; `ptr`: PTRACERS_PARAMS.h.

    The level loop DO k=kTop,kBottom,kDir (:110-169) is a recursion (level k-1 is mixed at iteration k and read
    again, with level k's mixed value, by FIND_RHO_2D at iteration k+1): a level scan in the Fortran order, calling
    the per-level routines with k a traced level index (KERNEL_GUIDE §4). The test
    `IF ( DIFFERENT_MULTIPLE(cAdjFreq,myTime,deltaTClock) )` (:67-68, :179) reads the traced clock: every field is
    computed and the mixed values are selected where it holds (a `where`, both branches finite). ConvectCount
    (:83-89, :146) only feeds DIAGNOSTICS_FILL 'CONVADJ ' (:171-176, output only, not ported: useDiagnostics raises
    in the kernels' parameters); it is computed and dropped. The CADJ STOREs are TAF directives."""
    if not cfg.cpp.INCLUDE_CONVECT_CALL:                                        # :42
        raise NotImplementedError("CONVECTIVE_ADJUSTMENT: compiled without INCLUDE_CONVECT_CALL (an empty routine)")
    if params.useDiagnostics:
        raise NotImplementedError("CONVECTIVE_ADJUSTMENT: DIAGNOSTICS_FILL 'CONVADJ ' (useDiagnostics) is not ported")
    doIt = different_multiple(params.cAdjFreq, myTime, params.deltaTClock)     # :67-68
    sz = cfg.size
    Nr = sz.Nr
    iMin = 1-sz.OLx                                                             # :71-74
    iMax = sz.sNx+sz.OLx
    jMin = 1-sz.OLy
    jMax = sz.sNy+sz.OLy
    # :95-107  IF ( rkSign*gravitySign .GT. 0. ) "<=> usingZCoords" (the code's comment, :96): decided on the host
    # from usingZCoords (rkSign = gravitySign = -1 in z coordinates): kTop=2, kBottom=Nr, kDir=1, deltaK=-1
    if not params.usingZCoords:
        raise NotImplementedError("CONVECTIVE_ADJUSTMENT: the pressure-coordinate branch (:101-107) is not ported")
    kTop, kBottom, kDir, deltaK = 2, Nr, 1, -1
    theta0, salt0, ptf0 = state.theta, state.salt, ptf
    theta, salt = theta0, salt0
    lev = _level(theta, 1)
    rhoKm1 = lev.local("rhoKm1")
    rhoK = lev.local("rhoK")
    weightA = lev.local("weightA")
    weightB = lev.local("weightB")
    ConvectCount = FArray(jnp.zeros_like(theta.data), "ConvectCount", tiled=theta.tiled, _dims=theta.dims)  # :83-89
    usePTRACERS = cfg.cpp.ALLOW_PTRACERS and params.usePTRACERS                 # :160-166
    # :110  DO k=kTop,kBottom,kDir as a level scan (KERNEL_GUIDE §4; no level branches)
    def level_k(k, c):
        rhoKm1, rhoK, weightA, weightB, ConvectCount, theta, salt, ptf = c
        rhoKm1 = find_rho_2d(iMin, iMax, jMin, jMax, k+deltaK,                  # :123-128
                             _level(theta, k-1), _level(salt, k-1), rhoKm1, k-1,
                             cfg=cfg, grid=grid, params=params, eos=eos, state=state)
        rhoK = find_rho_2d(iMin, iMax, jMin, jMax, k+deltaK,                    # :131-136
                           _level(theta, k), _level(salt, k), rhoK, k,
                           cfg=cfg, grid=grid, params=params, eos=eos, state=state)
        weightA, weightB, ConvectCount = convective_weights(                    # :143-146
            k, rhoKm1, rhoK, weightA, weightB, ConvectCount, cfg=cfg, grid=grid)
        theta = convectively_mixtracer(k, weightA, weightB, theta, cfg=cfg)    # :149-152
        salt = convectively_mixtracer(k, weightA, weightB, salt, cfg=cfg)      # :155-158
        if usePTRACERS:                                                         # :160-166
            ptf = ptracers_convect(k, weightA, weightB, cfg=cfg, ptr=ptr, ptf=ptf)
        return rhoKm1, rhoK, weightA, weightB, ConvectCount, theta, salt, ptf

    from mitjax.ops.scan_k import scan_levels
    c = scan_levels(level_k, (rhoKm1, rhoK, weightA, weightB, ConvectCount, theta, salt, ptf), kTop, kBottom)
    rhoKm1, rhoK, weightA, weightB, ConvectCount, theta, salt, ptf = c
    # :67-68 / :179  only where DIFFERENT_MULTIPLE holds
    theta = FArray(jnp.where(doIt, theta.data, theta0.data), theta.name, tiled=theta.tiled, _dims=theta.dims)
    salt = FArray(jnp.where(doIt, salt.data, salt0.data), salt.name, tiled=salt.tiled, _dims=salt.dims)
    if usePTRACERS:
        ptf = ptf.replace(pTracer=tuple(FArray(jnp.where(doIt, a.data, b.data), a.name, tiled=a.tiled, _dims=a.dims)
                                        for a, b in zip(ptf.pTracer, ptf0.pTracer)))
    return state.replace(theta=theta, salt=salt), ptf
