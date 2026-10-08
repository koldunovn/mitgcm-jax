"""GMREDI_SLOPE_PSI: pkg/gmredi/gmredi_slope_psi.F @63cdc0b (GOADK lane, M2: global_ocean.90x40x15/code_ad)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.model.grid import oneRL
from mitjax.ops.libm import glibc_tanh
from mitjax.pkg.gmredi.gmredi_h import op5


def gmredi_slope_psi(taperX, taperY, SlopeX, SlopeY, dSigmaDrW, dSigmaDrS, LrhoW, LrhoS, rDepth, k,
                     *, cfg, params, gm):
    """GMREDI_SLOPE_PSI( taperX, taperY, SlopeX, SlopeY, dSigmaDrW, dSigmaDrS, LrhoW, LrhoS, rDepth, k, bi, bj,
                         myThid )
    @63cdc0b pkg/gmredi/gmredi_slope_psi.F:9-429

    C     | SUBROUTINE GMREDI_SLOPE_PSI
    C     | o Calculate slopes for use in GM advective form
    C     | On entry:
    C     |     dSigmaDrW,S  contains the -d/dz Sigma if Z-coords
    C     |                           but  d/dp Sigma if P-coords
    C     |     SlopeX/Y     contains X/Y gradients of sigma
    C     |     rDepth       depth (> 0) in r-Unit from the surface
    C     | On exit:
    C     |     dSigmaDrW,S  contains the effective dSig/dz
    C     |     SlopeX/Y     contains X/Y slopes
    C     |     taperFct     contains tapering funct. value ;
    C     |                  = 1 when using no tapering

    Returns (taperX, taperY, SlopeX, SlopeY, dSigmaDrW, dSigmaDrS) (outputs first, then the updated arguments, in
    the Fortran argument order). `k`: Python int. `params`: PARAMS.h (wUnit2rVel, rVel2wUnit); `gm`: GMREDI.h.

    Ported (global_ocean.90x40x15/code_ad: GM_EXCLUDE_CLIPPING, GM_EXCLUDE_FM07_TAP defined, GM_EXCLUDE_TAPERING
    undefined, GMREDI_WITH_STABLE_ADJOINT undefined): the no-clipping slope of the main ELSE branch (:198-263: the
    stratification floor GM_Small_Number, the slope, the cut-off at SQRT(GM_slopeSqCutoff) for every scheme but
    'stableGmAdjTap') and the taper scheme 'dm95' (:314-328; TANH is glibc's, mitjax/ops/libm.glibc_tanh); a blank
    scheme has no taper code of its own. The Fortran STOPs of the excluded schemes ('orig'/'clipping' :95-101,
    'fm07' :181-183, GM_EXCLUDE_TAPERING :187-189, 'stableGmAdjTap' without GMREDI_WITH_STABLE_ADJOINT :376-380)
    and of a bad scheme name (:417-418) raise RuntimeError with the Fortran message. GO lane (global_ocean.cs32x15):
    'gkw91'/'ac02' (:290-312; the denominator SlopeX**2 + GM_Small_Number > 0 on every lane). M3 lane MLAdjust:
    'linear' (:270-288; Smod + GM_Small_Number > 0, so the division is finite on every lane). Not ported (raise):
    'ldd97' (:330-374, reads LrhoW/S and rDepth), the scheme branches of a
    build without GM_EXCLUDE_CLIPPING / GM_EXCLUDE_FM07_TAP. LrhoW, LrhoS and rDepth are read only by 'ldd97'.

    Vectorisation: every DO j / DO i nest runs on its whole (i,j) range at once (each point reads only its own
    values). A pointwise IF is a `where` of the two values; both are finite: the division SlopeX/dSigmaDrW comes
    after the floor `dSigmaDrW >= GM_Small_Number` (GM_Small_Number > 0 in every run, so no lane divides by 0).
    SIGN(slopeCutoff, SlopeX) is copysign(slopeCutoff, SlopeX) (gfortran -fsign-zero; slopeCutoff > 0). The
    IF ( ABS(Slope).GE.slopeCutoff ) clip is differentiated as written (JAX's derivative of `where`).
    """
    sz = cfg.size
    sNx, sNy, OLx, OLy = sz.sNx, sz.sNy, sz.OLx, sz.OLy
    scheme = gm.GM_taper_scheme.rstrip()

    slopeCutoff = jnp.sqrt(gm.GM_slopeSqCutoff)                     # :80  SQRT( GM_slopeSqCutoff )
    # :87-88  loc_maxSlope = GM_maxSlope*wUnit2rVel(k) (read by 'linear', 'gkw91'/'ac02'), loc_rMaxSlope =
    # GM_rMaxSlope*rVel2wUnit(k): read only by the 'orig'/'clipping' branches (not ported)
    loc_maxSlope = gm.GM_maxSlope*params.wUnit2rVel[k]              # :87

    if scheme in ("orig", "clipping"):                              # :95-180
        if cfg.cpp.GM_EXCLUDE_CLIPPING:
            raise RuntimeError('Need to compile without "#define GM_EXCLUDE_CLIPPING"')       # :100
        raise NotImplementedError("GMREDI_SLOPE_PSI: GM_taper_scheme 'orig'/'clipping' is not ported")
    if scheme == "fm07":                                            # :181-183
        raise RuntimeError("GMREDI_SLOPE_PSI: AdvForm not yet implemented for fm07")
    if cfg.cpp.GM_EXCLUDE_TAPERING:                                 # :187-189
        raise RuntimeError('Need to compile without "#define GM_EXCLUDE_TAPERING"')
    if scheme not in ("", "linear", "dm95", "gkw91", "ac02"):       # :270-419
        if scheme == "ldd97":
            raise NotImplementedError(f"GMREDI_SLOPE_PSI: GM_taper_scheme {scheme!r} is not ported")
        if scheme == "stableGmAdjTap":
            if not cfg.cpp.GMREDI_WITH_STABLE_ADJOINT:
                raise RuntimeError('Need to compile wth "#define GMREDI_WITH_STABLE_ADJOINT"')  # :380
            raise NotImplementedError("GMREDI_SLOPE_PSI: GM_taper_scheme 'stableGmAdjTap' is not ported")
        raise RuntimeError("GMREDI_SLOPE_PSI: Bad GM_taper_scheme")                         # :418

    # :196-227  X-direction
    j = loop_j(1-OLy, sNy+OLy)                                      # :200-205
    i = loop_i(1-OLx+1, sNx+OLx)
    dSigmaDrW = dSigmaDrW.at[i, j].set(jnp.where(dSigmaDrW[i, j] <= gm.GM_Small_Number,
                                                 gm.GM_Small_Number, dSigmaDrW[i, j]))
    SlopeX = SlopeX.at[i, j].set(SlopeX[i, j]/dSigmaDrW[i, j])      # :209-214
    taperX = taperX.at[i, j].set(1.)                                # 1. _d 0
    if scheme != "stableGmAdjTap":                                  # :218-227
        isCut = jnp.abs(SlopeX[i, j]) >= slopeCutoff
        SlopeX = SlopeX.at[i, j].set(jnp.where(isCut, jnp.copysign(slopeCutoff, SlopeX[i, j]), SlopeX[i, j]))
        taperX = taperX.at[i, j].set(jnp.where(isCut, 0., taperX[i, j]))   # 0. _d 0

    # :230-261  Y-direction
    j = loop_j(1-OLy+1, sNy+OLy)                                    # :234-239
    i = loop_i(1-OLx, sNx+OLx)
    dSigmaDrS = dSigmaDrS.at[i, j].set(jnp.where(dSigmaDrS[i, j] <= gm.GM_Small_Number,
                                                 gm.GM_Small_Number, dSigmaDrS[i, j]))
    SlopeY = SlopeY.at[i, j].set(SlopeY[i, j]/dSigmaDrS[i, j])      # :243-248
    taperY = taperY.at[i, j].set(1.)                                # 1. _d 0
    if scheme != "stableGmAdjTap":                                  # :252-261
        isCut = jnp.abs(SlopeY[i, j]) >= slopeCutoff
        SlopeY = SlopeY.at[i, j].set(jnp.where(isCut, jnp.copysign(slopeCutoff, SlopeY[i, j]), SlopeY[i, j]))
        taperY = taperY.at[i, j].set(jnp.where(isCut, 0., taperY[i, j]))   # 0. _d 0

    if scheme == "linear":                                          # :270-288 (M3 lane MLAdjust)
#-      Simplest adiabatic tapering = Smax/Slope (linear)
        j = loop_j(1-OLy, sNy+OLy)                                  # :273-280
        i = loop_i(1-OLx+1, sNx+OLx)
        Smod = jnp.abs(SlopeX[i, j])
        cond = (Smod > loc_maxSlope) & (Smod < slopeCutoff)
        taperX = taperX.at[i, j].set(jnp.where(cond, loc_maxSlope/(Smod+gm.GM_Small_Number), taperX[i, j]))
        j = loop_j(1-OLy+1, sNy+OLy)                                # :281-288
        i = loop_i(1-OLx, sNx+OLx)
        Smod = jnp.abs(SlopeY[i, j])
        cond = (Smod > loc_maxSlope) & (Smod < slopeCutoff)
        taperY = taperY.at[i, j].set(jnp.where(cond, loc_maxSlope/(Smod+gm.GM_Small_Number), taperY[i, j]))
    elif scheme in ("gkw91", "ac02"):                               # :290-312 (GO lane, global_ocean.cs32x15)
        # Gerdes, Koberle and Willebrand, Clim. Dyn. 1991
        maxSlopeSqr = loc_maxSlope*loc_maxSlope                     # :294
        j = loop_j(1-OLy, sNy+OLy)                                  # :295-303
        i = loop_i(1-OLx+1, sNx+OLx)
        Smod = jnp.abs(SlopeX[i, j])
        steep = (Smod > loc_maxSlope) & (Smod < slopeCutoff)
        taperX = taperX.at[i, j].set(jnp.where(steep, maxSlopeSqr/(SlopeX[i, j]*SlopeX[i, j] + gm.GM_Small_Number),
                                               taperX[i, j]))
        j = loop_j(1-OLy+1, sNy+OLy)                                # :304-312
        i = loop_i(1-OLx, sNx+OLx)
        Smod = jnp.abs(SlopeY[i, j])
        steep = (Smod > loc_maxSlope) & (Smod < slopeCutoff)
        taperY = taperY.at[i, j].set(jnp.where(steep, maxSlopeSqr/(SlopeY[i, j]*SlopeY[i, j] + gm.GM_Small_Number),
                                               taperY[i, j]))

    if scheme == "dm95":                                            # :314-328  Danabasoglu and McWilliams 1995
        j = loop_j(1-OLy, sNy+OLy)                                  # :317-322
        i = loop_i(1-OLx+1, sNx+OLx)
        Smod = jnp.abs(SlopeX[i, j])*params.rVel2wUnit[k]
        taperX = taperX.at[i, j].set(op5*(oneRL + glibc_tanh((gm.GM_Scrit-Smod)/gm.GM_Sd)))
        j = loop_j(1-OLy+1, sNy+OLy)                                # :323-328
        i = loop_i(1-OLx, sNx+OLx)
        Smod = jnp.abs(SlopeY[i, j])*params.rVel2wUnit[k]
        taperY = taperY.at[i, j].set(op5*(oneRL + glibc_tanh((gm.GM_Scrit-Smod)/gm.GM_Sd)))

    return taperX, taperY, SlopeX, SlopeY, dSigmaDrW, dSigmaDrS
