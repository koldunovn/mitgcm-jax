"""GGL90_MIXINGLENGTH: pkg/ggl90/ggl90_mixinglength.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j, loops_kji
from mitjax.ops.fortran_minmax import MAX, MIN
from mitjax.ops.safe import safe_sqrt
from mitjax.ops.scan_k import level, scan_k, set_levels


def ggl90_mixinglength(GGL90mixingLength, LCmixingLength, rMixingLength, iMin, iMax, jMin, jMax, myTime, myIter,
                       *, cfg, grid, params, ggl):
    """GGL90_MIXINGLENGTH( GGL90mixingLength, [LCmixingLength,] rMixingLength, iMin, iMax, jMin, jMax, bi, bj,
    myTime, myIter, myThid )   @63cdc0b pkg/ggl90/ggl90_mixinglength.F:10-421

    C     | SUBROUTINE GGL90_MIXINGLENGTH                            |
    C     | o Compute GGL90mixingLength (and LCmixingLength)         |
    C     | Equation numbers refer to                                |
    C     |  Gaspar et al. (1990), JGR 95 (C9), pp 16,179            |
    C     | Some parts of the implementation follow Blanke and       |
    C     |  Delecuse (1993), JPO, and OPA code, in particular the   |
    C     |  computation of the                                      |
    C     |  mixing length = max(min(lk,depth),lkmin)                |
    C     | Note: Only call this S/R if Nr > 1 (no use if Nr=1)      |
    C     GGL90mixingLength :: mixing length (m) following Banke+Delecuse
    C     rMixingLength     :: inverse of mixing length
    C     iMin,iMax,jMin,jMax :: index boundaries of computation domain

    GGL90mixingLength, LCmixingLength, rMixingLength: FArrays (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr) of the caller
    (LCmixingLength is an argument only under ALLOW_GGL90_LANGMUIR: pass None otherwise); iMin..jMax Python ints.
    Returns (GGL90mixingLength, LCmixingLength, rMixingLength) (LCmixingLength None without ALLOW_GGL90_LANGMUIR).

    Ported: z coordinates (kSrf = 1, kTop = 2, coordFac = 1, :99-109), mxlMaxFlag 0, 1, 2 and 3 (:166-307), the
    Langmuir branch (ALLOW_GGL90_LANGMUIR with useLANGMUIR, :309-376) and the default minimum (no
    GGL90_REGULARIZE_MIXINGLENGTH, :381-416). The downward sweep (DO k=2,Nr, :246-258) and the upward sweep (DO
    k=Nr-1,2,-1, :268-283) are recursions in k (scan_k in the Fortran order); every other k loop has independent
    iterations (each level reads only that level) and is vectorised. Raise: pressure coordinates (:99-101 and the
    usingPcoords arms), ALLOW_SHELFICE with useShelfIce (:151-156), GGL90_REGULARIZE_MIXINGLENGTH (:390, :405),
    ALLOW_AUTODIFF with adMxlMaxFlag /= mxlMaxFlag (:112-114: the AD-mode flag is a backward-only switch, not ported).
    SQRT( GGL90mixingLength*mxLength_Dn ) (:394) is guarded on a product > 0 (both factors are >= +0, so the value
    is Fortran's; the derivative of SQRT at 0 never reaches the backward pass).
    """
    opt = "GGL90_OPTIONS.h"
    sz = cfg.size
    Nr = sz.Nr
    if params.usingPCoords:                                             # :99-101
        raise NotImplementedError("GGL90_MIXINGLENGTH: pressure coordinates are not ported")
    kSrf, kTop = 1, 2                                                   # :103-104
    coordFac = 1.0                                                      # :107  1. _d 0
    recip_coordFac = 1.0/coordFac                                       # :109  1./coordFac
    locMxlMaxFlag = ggl.mxlMaxFlag                                      # :111
    if cfg.cpp.flag("ALLOW_AUTODIFF", opt) and ggl.adMxlMaxFlag != ggl.mxlMaxFlag:     # :112-114
        raise NotImplementedError("GGL90_MIXINGLENGTH: adMxlMaxFlag /= mxlMaxFlag (an AD-mode switch) is not ported")
    if cfg.cpp.flag("GGL90_REGULARIZE_MIXINGLENGTH", opt):
        raise NotImplementedError("GGL90_MIXINGLENGTH: GGL90_REGULARIZE_MIXINGLENGTH is not ported")
    langmuir = cfg.cpp.flag("ALLOW_GGL90_LANGMUIR", opt)
    Lmin = ggl.GGL90mixingLengthMin
    fj, fi = (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx)

    # :120-127  initialize local fields
    k, j, i = loops_kji((1, Nr), fj, fi)
    rMixingLength = rMixingLength.at[i, j, k].set(0.0)                  # 0. _d 0
    mxLength_Dn = GGL90mixingLength.local("mxLength_Dn").at[i, j, k].set(0.0)
    j, i = loop_j(*fj), loop_i(*fi)
    mxLength_Dn = mxLength_Dn.at[i, j, 1].set(Lmin)                     # :128-133
    if langmuir and ggl.useLANGMUIR:                                    # :135-145
        k, j, i = loops_kji((1, Nr), fj, fi)
        LCmixingLength = LCmixingLength.at[i, j, k].set(Lmin)

    L = GGL90mixingLength
    jj, ii = loop_j(jMin, jMax), loop_i(iMin, iMax)
    if ggl.mxlSurfFlag:                                                 # :147-160
        if cfg.cpp.ALLOW_SHELFICE and cfg.use_flag("useShelfIce"):
            raise NotImplementedError("GGL90_MIXINGLENGTH: ALLOW_SHELFICE with useShelfIce is not ported")
        L = L.at[ii, jj, kTop].set(grid.drF[kSrf]*recip_coordFac)      # :157

    if locMxlMaxFlag == 0:                                              # :166-179
        k, j, i = loops_kji((2, Nr), (jMin, jMax), (iMin, iMax))
        MaxLength = (grid.Ro_surf[i, j] - grid.R_low[i, j])*recip_coordFac      # :173-174
        L = L.at[i, j, k].set(MIN(L[i, j, k], MaxLength, p="b"))        # :175-176
    elif locMxlMaxFlag == 1:                                            # :181-193
        k, j, i = loops_kji((2, Nr), (jMin, jMax), (iMin, iMax))
        MaxLength = MIN(grid.Ro_surf[i, j]-grid.rF[k], grid.rF[k]-grid.R_low[i, j], p="a")*recip_coordFac  # :186-187
        L = L.at[i, j, k].set(MIN(L[i, j, k], MaxLength, p="b"))        # :189-190
    elif locMxlMaxFlag in (2, 3):                                       # :195-299
        # :241-258  z coordinates, downward sweep: mxLength_Dn(k) needs mxLength_Dn(k-1)
        def down(dn_km1, x):
            dn_k = x["Dn"].at[ii, jj].set(MIN(x["L"][ii, jj],          # :254-255
                                              dn_km1[ii, jj]+x["drF_km1"]*recip_coordFac, p="a"))
            return dn_k, dn_k

        _, out = scan_k(down, level(mxLength_Dn, 1), range(2, Nr+1),
                        lambda k: {"L": level(L, k), "Dn": level(mxLength_Dn, k), "drF_km1": grid.drF[k-1]})
        mxLength_Dn = set_levels(mxLength_Dn, out, range(2, Nr+1))
        # :262-267  upward sweep, k = Nr
        L = L.at[ii, jj, Nr].set(MIN(L[ii, jj, Nr], Lmin+grid.drF[Nr]*recip_coordFac, p="a"))   # :264-265
        # :268-283  DO k=Nr-1,2,-1: GGL90mixingLength(k) needs GGL90mixingLength(k+1)
        if Nr >= 3:
            def up(l_kp1, x):
                l_k = x["L"].at[ii, jj].set(MIN(x["L"][ii, jj],      # :279-280
                                                l_kp1[ii, jj]+x["drF_k"]*recip_coordFac, p="a"))
                return l_k, l_k

            _, out = scan_k(up, level(L, Nr), range(Nr-1, 1, -1),
                            lambda k: {"L": level(L, k), "drF_k": grid.drF[k]})
            L = set_levels(L, out, range(Nr-1, 1, -1))
        # :291-299  impose minimum from downward sweep
        k, j, i = loops_kji((2, Nr), (jMin, jMax), (iMin, iMax))
        L = L.at[i, j, k].set(MIN(L[i, j, k], mxLength_Dn[i, j, k], p="b"))     # :295-296
    else:                                                               # :301-307
        raise RuntimeError(f"GGL90_MIXINGLENGTH: mxlMaxFlag={locMxlMaxFlag:5d} not implemented; "
                           "ABNORMAL END: S/R GGL90_MIXINGLENGTH")

    if langmuir and ggl.useLANGMUIR:                                    # :309-376
        k, j, i = loops_kji((2, Nr), (jMin, jMax), (iMin, iMax))
        if locMxlMaxFlag == 1:                                          # :317-333
            MaxLength = (grid.Ro_surf[i, j]-grid.rF[k]) * recip_coordFac          # :324
            LCmixingLength = LCmixingLength.at[i, j, k].set(jnp.where(            # :326-330
                L[i, j, k] == MaxLength, ggl.LC_Gamma * L[i, j, k], L[i, j, k]))
        elif locMxlMaxFlag in (2, 3):                                   # :335-350
            LCmixingLength = LCmixingLength.at[i, j, k].set(jnp.where(            # :343-347
                L[i, j, k] == mxLength_Dn[i, j, k], ggl.LC_Gamma * L[i, j, k], L[i, j, k]))
        else:                                                           # :352-359
            raise RuntimeError(f"GGL90_MIXINGLENGTH: Langmuir Circ. Parameterization with mxlMaxFlag="
                               f"{locMxlMaxFlag:5d} not implemented; ABNORMAL END: S/R GGL90_MIXINGLENGTH")
        if locMxlMaxFlag in (1, 2):                                     # :361-374
            MLtmp = MAX(LCmixingLength[i, j, k], Lmin, p="a")          # :369
            LCmixingLength = LCmixingLength.at[i, j, k].set(MLtmp)      # :370

    # :381-416  impose minimum mixing length to avoid division by zero and compute inverse
    k, j, i = loops_kji((2, Nr), (jMin, jMax), (iMin, iMax))
    if locMxlMaxFlag == 3:
        prod = L[i, j, k]*mxLength_Dn[i, j, k]
        MLtmp = safe_sqrt(prod, prod > 0.0)                             # :394  SQRT( L*mxLength_Dn )
        MLtmp = MAX(MLtmp, Lmin, p="a")                                 # :395
        rMixingLength = rMixingLength.at[i, j, k].set(1.0/MLtmp)        # :397  1. _d 0 / MLtmp
    else:
        MLtmp = MAX(L[i, j, k], Lmin, p="a")                            # :409
        L = L.at[i, j, k].set(MLtmp)                                    # :411
        rMixingLength = rMixingLength.at[i, j, k].set(1.0/MLtmp)        # :412
    return L, (LCmixingLength if langmuir else None), rMixingLength
