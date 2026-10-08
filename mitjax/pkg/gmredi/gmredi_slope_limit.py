"""GMREDI_SLOPE_LIMIT: pkg/gmredi/gmredi_slope_limit.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.model.grid import oneRL, zeroRL
from mitjax.ops.fortran_minmax import MIN
from mitjax.ops.libm import glibc_tanh
from mitjax.ops.safe import safe_div, safe_sqrt
from mitjax.pkg.gmredi.gmredi_h import op5

# :101-102  PARAMETER( fpi = PI ); PARAMS.h:16  PARAMETER ( PI = 3.14159265358979323844D0 )
fpi = 3.14159265358979323844

# the Fortran STOP of a taper scheme excluded at compile time (gmredi_slope_limit.F:174-176, 232-234, 395-397,
# 443-445, 634-636)
_EXCLUDED = {"GM_EXCLUDE_CLIPPING": 'Need to compile without "#define GM_EXCLUDE_CLIPPING"',
             "GM_EXCLUDE_FM07_TAP": 'Need to compile without "#define GM_EXCLUDE_FM07_TAP"',
             "GM_EXCLUDE_AC02_TAP": 'Need to compile without "#define GM_EXCLUDE_AC02_TAP"',
             "GM_EXCLUDE_TAPERING": 'Need to compile without "#define GM_EXCLUDE_TAPERING"'}


def gmredi_slope_limit(SlopeX, SlopeY, SlopeSqr, taperFct, hTransLay, baseSlope, recipLambda, dSigmaDr,
                       dSigmaDx, dSigmaDy, Lrho, hMixLay, rDepth, depthZ, kLow, kPos, k, myTime, myIter,
                       *, cfg, params, gm):
    """GMREDI_SLOPE_LIMIT( SlopeX, SlopeY, SlopeSqr, taperFct, hTransLay, baseSlope, recipLambda, dSigmaDr,
                           dSigmaDx, dSigmaDy, Lrho, hMixLay, rDepth, depthZ, kLow, kPos, k, bi, bj, myTime,
                           myIter, myThid )
    @63cdc0b pkg/gmredi/gmredi_slope_limit.F:9-651

    C     | SUBROUTINE GMREDI_SLOPE_LIMIT
    C     | o Calculate slopes for use in GM/Redi tensor
    C     | On entry:
    C     |            dSigmaDr     contains the downward d/dz Sigma
    C     |            dSigmaDx/Dy  contains X/Y gradients of sigma
    C     |            Lrho
    C     |            hMixLay      mixed layer depth (> 0)
    C     |            rDepth       depth (> 0) in r-Unit from the surface
    C     |            depthZ       contains the depth (< 0 !) [m]
    C    U             hTransLay    transition layer depth (> 0)
    C    U             baseSlope, recipLambda,
    C     | On exit:
    C     |            dSigmaDr     contains the effective downward dSig/dz
    C     |            SlopeX/Y     contains X/Y slopes
    C     |            SlopeSqr     contains Sx^2+Sy^2
    C     |            taperFct     contains tapering funct. value ;
    C     |                         = 1 when using no tapering
    C    U             hTransLay    transition layer depth (> 0)
    C    U             baseSlope, recipLambda
    C     kPos     :: grid-cell location: 1,2,3 : at U,V,W location
    C     k        :: level index

    Returns (SlopeX, SlopeY, SlopeSqr, taperFct, hTransLay, baseSlope, recipLambda, dSigmaDr). `kPos`, `k`: Python
    ints. `params`: PARAMS.h (usingZCoords is not read here; wUnit2rVel, rVel2wUnit, rUnit2z, z2rUnit as FArrays);
    `gm`: GMREDI.h (gmredi_h.Gmredi).

    Ported: the no-clipping slope (:461-518) and the taper schemes 'gkw91' (:539-553, global_ocean.90x40x15) and
    'dm95' (:555-568, tutorial_global_oce_optim; TANH is glibc's, mitjax/ops/libm.glibc_tanh), M3 lane MLAdjust
    'linear' (:522-537; SQRT on the steep lanes only, safe_sqrt of a guarded quotient), lane M4LAB 'ldd97'
    (:570-592, lab_sea: TANH glibc's, SIN XLA's = glibc's; Lrho = the caller's ldd97_Lrho*, rDepth > 0; SQRT and
    the quotient rDepth/(Lrho*Smod) guarded off the selected lanes); a blank scheme (no taper beyond the cut-off,
    :640) has no code of its own. Not ported (raise): 'orig'/'clipping' (:171-227), 'fm07' (:229-391), 'ac02'
    (:393-439), 'stableGmAdjTap' (:594-638);
    where the build excludes a scheme, the Fortran STOP message; any other name, the Fortran STOP 'Bad
    GM_taper_scheme' (:640-641). hTransLay, baseSlope, recipLambda, Lrho, hMixLay, rDepth, depthZ, kLow are read only
    by fm07 (Lrho and rDepth also by ldd97) and pass through unchanged. Under ALLOW_AUTODIFF_TAMC the loop
    :500-518 is split in two (:505-512, store directives between): every point reads only its own values, so both forms compute the same; the
    statements below are the per-point ones in that order.

    Vectorisation: every point loop runs on the whole (i,j) range at once (each point reads only inputs or its own
    earlier values). A pointwise IF/ELSE is a `where` of its two branch values, each computed with guarded operands
    (mitjax/ops/safe.py) so that no lane forms 1/0, 0/0 or sqrt'(0): `1. _d 0 / dSigmaDr` where dSigmaDr = 0 (land,
    neutral), `maxSlopeSqr/SlopeSqr` and SQRT(SlopeSqr) outside their IF. A branch value that the Fortran stores in a
    local and then reads (dRdSigmaLtd, :484-486) is used as the branch temporary itself, so a lane where the branch
    is not taken never reads the local's unwritten (NaN) value [E§6]. SIGN(a, b) is copysign(|a|, b) (gfortran,
    -fsign-zero default; b /= 0 wherever it is evaluated here).
    """
    sz = cfg.size
    sNx, sNy, OLx, OLy = sz.sNx, sz.sNy, sz.OLx, sz.OLy
    scheme = gm.GM_taper_scheme.rstrip()

    GM_bigSlope = 1.e+02                                            # :124-125  GM_bigSlope = 1. _d +02
    if kPos == 3:                                                   # :128-138
        GM_bigSlope = GM_bigSlope*params.wUnit2rVel[k]
        maxSlopeSqr = (gm.GM_maxSlope*gm.GM_maxSlope
                       * params.wUnit2rVel[k]*params.wUnit2rVel[k])
        convSlopeUnit = params.rVel2wUnit[k]
    else:
        GM_bigSlope = GM_bigSlope*params.z2rUnit[k]
        maxSlopeSqr = (gm.GM_maxSlope*gm.GM_maxSlope
                       * params.z2rUnit[k]*params.z2rUnit[k])
        convSlopeUnit = params.rUnit2z[k]
    loc_rMaxSlope = gm.GM_rMaxSlope*convSlopeUnit                   # :139 (read only by 'orig'/'clipping')

    dSigmMod = SlopeX.local("dSigmMod")                             # :107-109 locals
    tmpFld = SlopeX.local("tmpFld")
    j = loop_j(1-OLy+1, sNy+OLy-1)                                  # :164-169
    i = loop_i(1-OLx+1, sNx+OLx-1)
    dSigmMod = dSigmMod.at[i, j].set(0.)                            # 0. _d 0
    tmpFld = tmpFld.at[i, j].set(0.)                                # 0. _d 0
    # (loc_rMaxSlope, dSigmMod and tmpFld are read only by the 'orig'/'clipping' and 'fm07' branches)

    if scheme in ("orig", "clipping"):                              # :171-227
        if cfg.cpp.GM_EXCLUDE_CLIPPING:
            raise RuntimeError(_EXCLUDED["GM_EXCLUDE_CLIPPING"])
        raise NotImplementedError("GMREDI_SLOPE_LIMIT: GM_taper_scheme 'orig'/'clipping' is not ported")
    elif scheme == "fm07":                                          # :229-391
        if cfg.cpp.GM_EXCLUDE_FM07_TAP:
            raise RuntimeError(_EXCLUDED["GM_EXCLUDE_FM07_TAP"])
        raise NotImplementedError("GMREDI_SLOPE_LIMIT: GM_taper_scheme 'fm07' is not ported")
    elif scheme == "ac02":                                          # :393-439
        if cfg.cpp.GM_EXCLUDE_AC02_TAP:
            raise RuntimeError(_EXCLUDED["GM_EXCLUDE_AC02_TAP"])
        raise NotImplementedError("GMREDI_SLOPE_LIMIT: GM_taper_scheme 'ac02' is not ported")
    if cfg.cpp.GM_EXCLUDE_TAPERING:                                 # :441-445
        raise RuntimeError(_EXCLUDED["GM_EXCLUDE_TAPERING"])
    if scheme not in ("", "linear", "gkw91", "dm95", "ldd97"):      # :522-641
        if scheme == "stableGmAdjTap":
            if not cfg.cpp.GMREDI_WITH_STABLE_ADJOINT:
                raise RuntimeError('Need to compile wth "#define GMREDI_WITH_STABLE_ADJOINT"')
            raise NotImplementedError("GMREDI_SLOPE_LIMIT: GM_taper_scheme 'stableGmAdjTap' is not ported")
        raise RuntimeError("GMREDI_SLOPE_LIMIT: Bad GM_taper_scheme")

    # :449-452  Compute the slope, no clipping, but avoid reverse slope in negatively stratified (dSigmaDr < 0) region
    j = loop_j(1-OLy+1, sNy+OLy-1)                                  # :461-468
    i = loop_i(1-OLx+1, sNx+OLx-1)
    dSigmaDr = dSigmaDr.at[i, j].set(
        jnp.where((dSigmaDr[i, j] != zeroRL) & (dSigmaDr[i, j] <= gm.GM_Small_Number),
                  gm.GM_Small_Number, dSigmaDr[i, j]))

    # :470-491
    isZero = dSigmaDr[i, j] == zeroRL                               # :472  IF ( dSigmaDr(i,j) .EQ. zeroRL ) THEN
    absBig = jnp.abs(GM_bigSlope)
    SlopeX_zero = jnp.where(dSigmaDx[i, j] != zeroRL,               # :473-477
                            jnp.copysign(absBig, dSigmaDx[i, j]), 0.)
    SlopeY_zero = jnp.where(dSigmaDy[i, j] != zeroRL,               # :478-482
                            jnp.copysign(absBig, dSigmaDy[i, j]), 0.)
    dRdSigmaLtd_else = safe_div(1., dSigmaDr[i, j], ~isZero)        # :484  1. _d 0 / dSigmaDr(i,j)
    SlopeX_else = dSigmaDx[i, j]*dRdSigmaLtd_else                   # :485
    SlopeY_else = dSigmaDy[i, j]*dRdSigmaLtd_else                   # :486
    SlopeX = SlopeX.at[i, j].set(jnp.where(isZero, SlopeX_zero, SlopeX_else))
    SlopeY = SlopeY.at[i, j].set(jnp.where(isZero, SlopeY_zero, SlopeY_else))

    # :500-518
    SlopeSqr = SlopeSqr.at[i, j].set(SlopeX[i, j]*SlopeX[i, j]
                                     + SlopeY[i, j]*SlopeY[i, j])
    taperFct = taperFct.at[i, j].set(1.)                            # 1. _d 0
    isCut = SlopeSqr[i, j] >= gm.GM_slopeSqCutoff                   # :513
    SlopeSqr = SlopeSqr.at[i, j].set(jnp.where(isCut, gm.GM_slopeSqCutoff, SlopeSqr[i, j]))
    taperFct = taperFct.at[i, j].set(jnp.where(isCut, 0., taperFct[i, j]))   # 0. _d 0

    # :520  Compute the tapering function for the GM+Redi tensor
    if scheme == "linear":                                          # :522-537 (M3 lane MLAdjust)
#-      Simplest adiabatic tapering = Smax/Slope (linear)
        isZeroS = SlopeSqr[i, j] == zeroRL
        isSteep = (SlopeSqr[i, j] > maxSlopeSqr) & (SlopeSqr[i, j] < gm.GM_slopeSqCutoff)
        lin = safe_sqrt(safe_div(maxSlopeSqr, SlopeSqr[i, j], isSteep), isSteep)   # :532 SQRT(maxSlopeSqr/SlopeSqr)
        taperFct = taperFct.at[i, j].set(
            jnp.where(isZeroS, 1.,                                  # 1. _d 0
                      jnp.where(isSteep, lin, taperFct[i, j])))
        SlopeSqr = SlopeSqr.at[i, j].set(                           # :533
            jnp.where(~isZeroS & isSteep, MIN(SlopeSqr[i, j], GM_bigSlope*GM_bigSlope, p="a"), SlopeSqr[i, j]))
    elif scheme == "gkw91":                                           # :539-553  Gerdes, Koberle and Willebrand 1991
        isZeroS = SlopeSqr[i, j] == zeroRL
        isSteep = (SlopeSqr[i, j] > maxSlopeSqr) & (SlopeSqr[i, j] < gm.GM_slopeSqCutoff)
        taperFct = taperFct.at[i, j].set(
            jnp.where(isZeroS, 1.,                                  # 1. _d 0
                      jnp.where(isSteep, safe_div(maxSlopeSqr, SlopeSqr[i, j], isSteep), taperFct[i, j])))
    elif scheme == "dm95":                                          # :555-568  Danabasoglu and McWilliams 1995
        isZeroS = SlopeSqr[i, j] == zeroRL
        isBelow = SlopeSqr[i, j] < gm.GM_slopeSqCutoff
        Smod = safe_sqrt(SlopeSqr[i, j], ~isZeroS & isBelow)*convSlopeUnit           # :564
        taperFct_dm95 = op5*(oneRL + glibc_tanh((gm.GM_Scrit-Smod)/gm.GM_Sd))       # :565
        taperFct = taperFct.at[i, j].set(
            jnp.where(isZeroS, 1.,                                  # 1. _d 0
                      jnp.where(isBelow, taperFct_dm95, taperFct[i, j])))
    elif scheme == "ldd97":                                         # :570-592  Large, Danabasoglu and Doney 1997
        isZeroS = SlopeSqr[i, j] == zeroRL                          # :576
        isBelow = SlopeSqr[i, j] < gm.GM_slopeSqCutoff              # :578
        sel = ~isZeroS & isBelow
        Smod = safe_sqrt(SlopeSqr[i, j], sel)                       # :579  SQRT(SlopeSqr(i,j))
        f1 = op5*(oneRL                                             # :580-581
                  + glibc_tanh((gm.GM_Scrit-Smod*convSlopeUnit)/gm.GM_Sd))
        Rnondim = safe_div(rDepth, Lrho[i, j]*Smod, sel)            # :582  rDepth/(Lrho(i,j)*Smod)
        big = Rnondim >= 1.                                         # :583  1. _d 0
        f2 = jnp.where(big, 1.,                                     # :584  1. _d 0
                       op5*(1. + jnp.sin(fpi*(Rnondim-op5))))       # :586  1. _d 0; SIN: XLA's = glibc's
        taperFct = taperFct.at[i, j].set(
            jnp.where(isZeroS, 1.,                                  # :577  1. _d 0
                      jnp.where(isBelow, f1*f2, taperFct[i, j])))   # :588

    return SlopeX, SlopeY, SlopeSqr, taperFct, hTransLay, baseSlope, recipLambda, dSigmaDr
