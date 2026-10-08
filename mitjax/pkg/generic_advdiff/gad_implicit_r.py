"""GAD_IMPLICIT_R: pkg/generic_advdiff/gad_implicit_r.F @63cdc0b."""

from mitjax.farray import loops_kji
from mitjax.model.src.solve_tridiagonal import solve_tridiagonal

_OPT = "GAD_OPTIONS.h"      # gad_implicit_r.F:1  #include "GAD_OPTIONS.h"


def gad_implicit_r(implicitAdvection, advectionScheme, tracerIdentity, deltaTLev, kappaRX, recip_hFac, wFld,
                   tracer, gTracer, myTime, myIter, *, cfg, grid, params):
    """GAD_IMPLICIT_R( implicitAdvection, advectionScheme, tracerIdentity, deltaTLev, kappaRX, recip_hFac, wFld,
    tracer, gTracer, bi, bj, myTime, myIter, myThid )   @63cdc0b pkg/generic_advdiff/gad_implicit_r.F:6-458

    C     Solve implicitly vertical advection and diffusion for one tracer.
    C implicitAdvection :: if True, treat vertical advection implicitly
    C advectionScheme   :: advection scheme to use
    C tracerIdentity    :: Identifier for the tracer
    C kappaRX           :: 3-D array for vertical diffusion coefficient
    C recip_hFac        :: inverse of cell open-depth factor
    C wFld              :: Advection velocity field, vertical component
    C tracer            :: tracer field at current time step
    C gTracer           :: future tracer field

    Returns gTracer. kappaRX, recip_hFac, wFld, tracer, gTracer: (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr); deltaTLev(Nr) not
    tiled. iMin..jMax are the PARAMETERs 1, sNx, 1, sNy (:78-79). Ported: the tri-diagonal implicit diffusion
    (implicitDiffusion, :121-160) and its solve (SOLVE_TRIDIAGONAL, COL lane, :261-271; the STOP on errCode >= 1, :269-271, cannot be raised inside a
    traced program: the solve guards its zero pivots, the error code is dropped).
    The three k loops of :125-157 are independent across k (each level reads only inputs and, for c5d, b5d/d5d
    of the same level written before): k-vectorised nests (KERNEL_GUIDE §4). GO lane: implicitAdvection (:162-259)
    with GAD_DST2U1_IMPL_R (ENUM_UPWIND_1RST, ENUM_DST2), GAD_FLUXLIMIT_IMPL_R (ENUM_FLUX_LIMIT), GAD_U3C4_IMPL_R
    (ENUM_UPWIND_3RD/CENTERED_4TH/DST3) and SOLVE_PENTADIAGONAL (:272-282); the k loop :185-256 is a level scan in
    the Fortran order (each interface adds to matrix lines k and k-1; KERNEL_GUIDE §4). Raise: GAD_C2_IMPL_R and GAD_DST3FL_IMPL_R (no
    M1/M2 caller), ALLOW_AIM, useDiagnostics (:287-452; pkg/diagnostics is not ported). `0. _d 0`, `1. _d 0` exact. The Fortran's unary minus applies to the whole
    product (-(a*b) = (-a)*b bit for bit)."""
    if cfg.cpp.flag("ALLOW_DIAGNOSTICS", _OPT) and params.useDiagnostics:
        raise NotImplementedError("GAD_IMPLICIT_R: the diagnostics block (:287-452, useDiagnostics) is not ported")
    sz = cfg.size
    Nr = sz.Nr
    g = grid
    if not Nr > 1:                                                              # :105
        return gTracer
    iMin, iMax, jMin, jMax = 1, sz.sNx, 1, sz.sNy                               # :78-79

    k, j, i = loops_kji((1, Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))   # :108-118
    a5d = gTracer.local("a5d").at[i, j, k].set(0.)
    b5d = gTracer.local("b5d").at[i, j, k].set(0.)
    c5d = gTracer.local("c5d").at[i, j, k].set(1.)
    d5d = gTracer.local("d5d").at[i, j, k].set(0.)
    e5d = gTracer.local("e5d").at[i, j, k].set(0.)
    diagonalNumber = 1                                                          # :119

    if params.implicitDiffusion:                                                # :121-160
        diagonalNumber = 3                                                      # :123
        k, j, i = loops_kji((2, Nr), (jMin, jMax), (iMin, iMax))                # :125-135
        b5d = b5d.at[i, j, k].set(-(deltaTLev[k]*g.maskC[i, j, k-1]
                                    * recip_hFac[i, j, k]*g.recip_drF[k]
                                    * g.recip_deepFac2C[k]*params.recip_rhoFacC[k]
                                    * kappaRX[i, j, k]*g.recip_drC[k]
                                    * g.deepFac2F[k]*params.rhoFacF[k]))
        k, j, i = loops_kji((1, Nr-1), (jMin, jMax), (iMin, iMax))              # :137-147
        d5d = d5d.at[i, j, k].set(-(deltaTLev[k]*g.maskC[i, j, k+1]
                                    * recip_hFac[i, j, k]*g.recip_drF[k]
                                    * g.recip_deepFac2C[k]*params.recip_rhoFacC[k]
                                    * kappaRX[i, j, k+1]*g.recip_drC[k+1]
                                    * g.deepFac2F[k+1]*params.rhoFacF[k+1]))
        k, j, i = loops_kji((1, Nr), (jMin, jMax), (iMin, iMax))                # :149-157
        c5d = c5d.at[i, j, k].set(1. - (b5d[i, j, k] + d5d[i, j, k]))

    if implicitAdvection:                                                       # :162-259 (GO lane)
        from mitjax.pkg.generic_advdiff.gad_calc_rhs import gad_kernel_cfg
        from mitjax.pkg.generic_advdiff.gad_h import (ENUM_CENTERED_2ND, ENUM_CENTERED_4TH, ENUM_DST2, ENUM_DST3,
                                                      ENUM_DST3_FLUX_LIMIT, ENUM_FLUX_LIMIT, ENUM_UPWIND_1RST,
                                                      ENUM_UPWIND_3RD)
        kcfg = gad_kernel_cfg(cfg)
        localTr = None
        # :165-183  Non-Linear Advection scheme: keep a local copy of tracer field
        if advectionScheme == ENUM_FLUX_LIMIT or advectionScheme == ENUM_DST3_FLUX_LIMIT:
            k, j, i = loops_kji((1, Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
            src = gTracer if params.multiDimAdvection else tracer                 # :166 / :174
            localTr = gTracer.local("localTr").at[i, j, k].set(src[i, j, k])      # :170 / :178
        from mitjax.farray import loop_i, loop_j
        from mitjax.ops.scan_k import level
        rTrans0 = level(gTracer.local("rTrans"), 1)                             # rTrans(1-OLx:..,1-OLy:..), :89
        jA = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
        iA = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
        # :185  DO k=Nr,1,-1 as a level scan (KERNEL_GUIDE §4): k = 1 static (its branch), and with the
        # FLUXLIMIT / U3C4 kernels also k = 2, 3, Nr-1, Nr (their MAX(1,k-2), MIN(Nr,k+1), k.GE.Nr tests)
        def level_k(k, c):
            nonlocal diagonalNumber
            a5d, b5d, c5d, d5d, e5d = c
            if k == 1:                                                          # :188-193
                rTrans = rTrans0.at[iA, jA].set(0.)                             # :191  0. _d 0
            else:                                                               # :194-201
                rTrans = rTrans0.at[iA, jA].set(wFld[iA, jA, k]*g.rA[iA, jA]     # :197-199
                                                * g.deepFac2F[k]*params.rhoFacF[k]
                                                * g.maskC[iA, jA, k-1])
            if cfg.cpp.flag("ALLOW_AIM", _OPT):                                 # :203-209
                raise NotImplementedError("GAD_IMPLICIT_R: the ALLOW_AIM test (:203-206) is not ported")
            if k >= 2:                                                          # :208
                if advectionScheme == ENUM_CENTERED_2ND:                        # :211-216
                    raise NotImplementedError("GAD_IMPLICIT_R: GAD_C2_IMPL_R (ENUM_CENTERED_2ND) is not ported "
                                              "(no M1/M2 caller)")
                elif advectionScheme == ENUM_UPWIND_1RST or advectionScheme == ENUM_DST2:   # :217-224
                    from mitjax.pkg.generic_advdiff.gad_dst2u1_impl_r import gad_dst2u1_impl_r
                    diagonalNumber = 3
                    b5d, c5d, d5d = gad_dst2u1_impl_r(k, iMin, iMax, jMin, jMax, advectionScheme, deltaTLev,
                                                      rTrans, recip_hFac, b5d, c5d, d5d, cfg=kcfg, grid=g,
                                                      params=params)
                elif advectionScheme == ENUM_FLUX_LIMIT:                        # :225-230
                    from mitjax.pkg.generic_advdiff.gad_fluxlimit_impl_r import gad_fluxlimit_impl_r
                    diagonalNumber = 3
                    b5d, c5d, d5d = gad_fluxlimit_impl_r(k, iMin, iMax, jMin, jMax, deltaTLev, rTrans, recip_hFac,
                                                         localTr, b5d, c5d, d5d, cfg=kcfg, grid=g, params=params)
                elif advectionScheme in (ENUM_UPWIND_3RD, ENUM_CENTERED_4TH, ENUM_DST3):   # :231-239
                    from mitjax.pkg.generic_advdiff.gad_u3c4_impl_r import gad_u3c4_impl_r
                    diagonalNumber = 5
                    a5d, b5d, c5d, d5d, e5d = gad_u3c4_impl_r(k, iMin, iMax, jMin, jMax, advectionScheme,
                                                              deltaTLev, rTrans, recip_hFac, a5d, b5d, c5d, d5d,
                                                              e5d, cfg=kcfg, grid=g, params=params)
                elif advectionScheme == ENUM_DST3_FLUX_LIMIT:                   # :240-245
                    raise NotImplementedError("GAD_IMPLICIT_R: GAD_DST3FL_IMPL_R (ENUM_DST3_FLUX_LIMIT) is not "
                                              "ported (no M1/M2 caller)")
                else:                                                           # :246-247
                    raise RuntimeError("GAD_IMPLICIT_R: Adv.Scheme in Impl form not yet coded")
            return a5d, b5d, c5d, d5d, e5d

        from mitjax.ops.scan_k import scan_levels
        lin = advectionScheme == ENUM_UPWIND_1RST or advectionScheme == ENUM_DST2
        a5d, b5d, c5d, d5d, e5d = scan_levels(level_k, (a5d, b5d, c5d, d5d, e5d), 1, Nr, down=True,
                                              peel=(1, 0) if lin else (3, 2))

    if diagonalNumber == 3:                                                     # :261-271
        errCode = -1                                                            # :263
        gTracer, errCode = solve_tridiagonal(iMin, iMax, jMin, jMax, b5d, c5d, d5d, gTracer, errCode, cfg=cfg)
        # :269-271 IF (errCode.GE.1) STOP 'GAD_IMPLICIT_R: error when solving 3-Diag problem'
    elif diagonalNumber == 5:                                                   # :272-282 (GO lane)
        from mitjax.model.src.solve_pentadiagonal import solve_pentadiagonal
        errCode = -1                                                            # :274
        gTracer, errCode = solve_pentadiagonal(iMin, iMax, jMin, jMax, a5d, b5d, c5d, d5d, e5d, gTracer, errCode,
                                               cfg=cfg)                         # :275-279
        # :280-282 IF (errCode.GE.1) STOP 'GAD_IMPLICIT_R: error when solving 5-Diag problem
    elif diagonalNumber != 1:                                                   # :283-285
        raise RuntimeError("GAD_IMPLICIT_R: no solver available")
    return gTracer
