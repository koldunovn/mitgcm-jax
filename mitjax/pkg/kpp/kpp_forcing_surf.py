"""KPP_FORCING_SURF: pkg/kpp/kpp_forcing_surf.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j, loops_kji
from mitjax.ops.fortran_minmax import MAX
from mitjax.ops.safe import safe_div, safe_sqrt


def kpp_forcing_surf(rhoSurf, surfForcU, surfForcV, surfForcT, surfForcS, surfForcTice, Qsw, dbloc, ttalpha,
                     ssbeta, ustar, bo, bosol, dVsq, ikey, iMin, iMax, jMin, jMax, myTime, *, cfg, grid, params, fp,
                     kpp, state, SPforcS=None, SPforcT=None, boplume=None):
    """KPP_FORCING_SURF( rhoSurf, surfForcU, surfForcV, surfForcT, surfForcS, surfForcTice, Qsw, dbloc, ttalpha,
    ssbeta, ustar, bo, bosol, dVsq, ikey, iMin, iMax, jMin, jMax, bi, bj, myTime, myThid )
    @63cdc0b pkg/kpp/kpp_forcing_surf.F:10-513

    C     | SUBROUTINE KPP_FORCING_SURF
    C     | o Compute all surface related KPP fields:
    C     |   - friction velocity ustar
    C     |   - turbulent and radiative surface buoyancy forcing,
    C     |     bo and bosol, and surface haline buoyancy forcing
    C     |     boplume
    C     |   - velocity shear relative to surface squared (this is
    C     |     not really a surface affected quantity unless it is
    C     |     computed with respect to some resolved near surface
    C     |     velocity, but this is computed here to keep KPP_CALC tidy)

    (i,j[,k]) FArrays; returns (ustar, bo, bosol, dVsq). uVel, vVel from `state`; nzmax from `kpp`; maskC, rF, drF,
    recip_drF, recip_drC from `grid`; HeatCapacity_Cp, selectPenetratingSW from `fp` (forcing parameters). Both
    KPP_ESTIMATE_UREF arms: defined (:309-461, the vermix/code arm; `dbloc` is then an argument, :14-16) and
    undefined (:463-503, lane M4COL: 1D_ocean_ice_column/code compiles the default pkg/kpp/KPP_OPTIONS.h, :39
    `#undef KPP_ESTIMATE_UREF`; the argument list has no dbloc and the caller passes `dbloc=None`). No
    KPP_SMOOTH_DVSQ (raise otherwise). ALLOW_SALT_PLUME (lane M4LAB, lab_sea: useSALT_PLUME = .FALSE.): the
    arguments SPforcS, SPforcT (:17-19, keyword-only) and boplume (:22-24, output: then the return is (ustar, bo,
    bosol, boplume, dVsq)); boplume = p0 on every point (:177-185); the IF ( useSALT_PLUME ) arm (:247-277) and
    SALT_PLUME_VOLUME raise, so SPforcS, SPforcT are not read.
    `selectPenetratingSW .GE. 1` is a static INTEGER. The points (i,j) are independent; per point the Fortran runs two
    loops whose trip depends on data:
      * the DO k = Nr,1,-1 search for kTmp (:328-334): a Python loop over k in the Fortran order (the last k that
        satisfies the condition wins, as the Fortran's repeated assignment);
      * the DO WHILE ( ABS(rF(k+1)) .LE. zRef(i,j) ) accumulation (:404-411) in the zRef >= drF(1) arm: a Python loop
        over k = 2, ..., Nr in which a point keeps accumulating while its condition has held at every k so far; the
        final k of each point is the first k whose condition fails. zRef < |rF(Nr+1)| on every point (zRef =
        MAX(epsilon*zRef, z0) with zRef <= |rF(Nr+1)| and z0 <= drF(1)*zFac), so the loop ends at k <= Nr and the
        reads of level k (:413-416) are in bounds; they are gathers along k at the per-point final k.
    Both arms of every pointwise IF are computed and selected; the divisions, SQRT and LOG of an arm are guarded on the
    lanes that do not select it (mitjax/ops/safe.py). LOG is XLA's (glibc's bit for bit, libm.MEASURED["log"])."""
    if cfg.cpp.flag("KPP_SMOOTH_DVSQ", "KPP_OPTIONS.h"):
        raise NotImplementedError("KPP_FORCING_SURF: KPP_SMOOTH_DVSQ is not ported")
    salt_plume = cfg.cpp.flag("ALLOW_SALT_PLUME", "KPP_OPTIONS.h")
    if salt_plume != (SPforcS is not None and SPforcT is not None and boplume is not None):   # :17-24
        raise TypeError("KPP_FORCING_SURF: SPforcS, SPforcT, boplume are arguments exactly when ALLOW_SALT_PLUME "
                        "is defined")
    if salt_plume and cfg.cpp.flag("SALT_PLUME_VOLUME", "KPP_OPTIONS.h"):
        raise NotImplementedError("KPP_FORCING_SURF: SALT_PLUME_VOLUME is not ported")
    if salt_plume and cfg.use_flag("useSALT_PLUME"):                           # :247-277
        raise NotImplementedError("KPP_FORCING_SURF: the useSALT_PLUME arm (boplume) is not ported")
    estimate_uref = cfg.cpp.flag("KPP_ESTIMATE_UREF", "KPP_OPTIONS.h")
    if estimate_uref != (dbloc is not None):                                   # :14-16 the argument list
        raise TypeError("KPP_FORCING_SURF: dbloc is an argument exactly when KPP_ESTIMATE_UREF is defined")
    sz = cfg.size
    Nr = sz.Nr
    g = grid
    p0, p5 = 0.0, 0.5                                                          # :126-127 PARAMETER (REAL*4, exact)
    uVel, vVel = state.uVel, state.vVel
    recip_Cp = 1.0 / fp.HeatCapacity_Cp                                        # :168  1. _d 0 / HeatCapacity_Cp
    ja = loop_j(1-sz.OLy, sz.sNy+sz.OLy)                                       # :170-176
    ia = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    ustar = ustar.at[ia, ja].set(p0)
    bo = bo.at[ia, ja].set(p0)
    bosol = bosol.at[ia, ja].set(p0)
    if salt_plume:                                                             # :177-185  DO k = 1, Nrp1
        kk, ja3, ia3 = loops_kji((1, Nr + 1), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
        boplume = boplume.at[ia3, ja3, kk].set(p0)

    def _ret(ustar, bo, bosol, dVsq):                                          # the argument list :20-26
        return (ustar, bo, bosol, boplume, dVsq) if salt_plume else (ustar, bo, bosol, dVsq)
    j = loop_j(jMin, jMax)                                                     # :187-197
    i = loop_i(iMin, iMax)
    work3 = ((surfForcU[i, j] + surfForcU[i+1, j]) *
             (surfForcU[i, j] + surfForcU[i+1, j]) +
             (surfForcV[i, j] + surfForcV[i, j+1]) *
             (surfForcV[i, j] + surfForcV[i, j+1]))
    epsLocSq = kpp.phepsi*kpp.phepsi*g.drF[1]*g.drF[1]                         # :198
    epsLoc = jnp.sqrt(p5*kpp.phepsi*g.drF[1])                                  # :199
    small = work3 < epsLocSq                                                   # :203
    tempVar2 = safe_sqrt(work3, ~small) * p5                                   # :206
    ustar = ustar.at[i, j].set(jnp.where(small, epsLoc, safe_sqrt(tempVar2, ~small)))   # :204, :207
    bo = bo.at[i, j].set(- params.gravity *                                    # :225-229
                         (ttalpha[i, j, 1] * (surfForcT[i, j] +
                                              surfForcTice[i, j]) +
                          ssbeta[i, j, 1] * surfForcS[i, j])
                         / rhoSurf[i, j])
    if fp.selectPenetratingSW >= 1:                                            # :233-242
        bosol = bosol.at[i, j].set(params.gravity * ttalpha[i, j, 1] * Qsw[i, j]
                                   * recip_Cp*params.recip_rhoConst
                                   / rhoSurf[i, j])
    kk, ja3, ia3 = loops_kji((1, Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))   # :299-305
    dVsq = dVsq.at[ia3, ja3, kk].set(p0)
    if not estimate_uref:                                                      # :463-503  #else KPP_ESTIMATE_UREF
        kk, j3, i3 = loops_kji((1, Nr), (jMin, jMax), (iMin, iMax))            # :465-480 (levels independent)
        dVsq = dVsq.at[i3, j3, kk].set(p5 * (
            (uVel[i3, j3, 1]-uVel[i3, j3, kk]) *
            (uVel[i3, j3, 1]-uVel[i3, j3, kk]) +
            (uVel[i3+1, j3, 1]-uVel[i3+1, j3, kk]) *
            (uVel[i3+1, j3, 1]-uVel[i3+1, j3, kk]) +
            (vVel[i3, j3, 1]-vVel[i3, j3, kk]) *
            (vVel[i3, j3, 1]-vVel[i3, j3, kk]) +
            (vVel[i3, j3+1, 1]-vVel[i3, j3+1, kk]) *
            (vVel[i3, j3+1, 1]-vVel[i3, j3+1, kk])))
        return _ret(ustar, bo, bosol, dVsq)
    # :320  zFac = ABS(rF(3)) * LOG ( rF(3) / rF(2) ) * recip_drF(2)
    zFac = jnp.abs(g.rF[3]) * jnp.log(g.rF[3] / g.rF[2]) * g.recip_drF[2]
    nzm = kpp.nzmax[i, j]                                                      # :328-334
    kTmp = nzm
    for k in range(Nr, 0, -1):
        cond = ((k < nzm) & (g.maskC[i, j, k] > 0.)
                & (dbloc[i, j, k] * g.recip_drC[k+1] > kpp.dB_dz))
        kTmp = jnp.where(cond, k, kTmp)
    k = kTmp                                                                   # :337
    zr0 = (k == 0) | (nzm == 1)                                                # :338
    zr1 = ~zr0 & (k == 1)                                                      # :340
    zr2 = ~zr0 & ~zr1 & (k < nzm)                                              # :343
    rF_, drF_, rdrC = g.rF.data, g.drF.data, g.recip_drC.data                  # level vectors (k = 1 at index 0)
    kc = jnp.clip(k, 1, Nr)                    # MINMAX-RAW: a valid level index on lanes whose arm reads no level
    dBdz2_1 = dbloc[i, j, 1] * g.recip_drC[2]                                  # :341
    zRef_1 = safe_div(g.drF[1] * kpp.dB_dz, dBdz2_1, zr1)                      # :342  drF(1) * dB_dz / dBdz2
    km1 = jnp.clip(kc - 1, 1, Nr)              # MINMAX-RAW: a valid level index on lanes that do not select :344
    dbl_i = lambda kx: jnp.take_along_axis(                                    # noqa: E731  dbloc(i,j,kx), i,j loops
        dbloc.data[:, :, j.first - dbloc.dims[1][1]:j.last - dbloc.dims[1][1] + 1,
                   i.first - dbloc.dims[0][1]:i.last - dbloc.dims[0][1] + 1], (kx - 1)[:, None], axis=1)[:, 0]
    dBdz1 = dbl_i(km1) * rdrC[kc - 1]                                          # :344  dbloc(k-1) * recip_drC(k)
    dBdz2 = dbl_i(kc) * rdrC[kc]                                               # :345  dbloc(k) * recip_drC(k+1)
    zRef_2 = (jnp.abs(rF_[kc - 1]) + drF_[kc - 1] * (kpp.dB_dz - dBdz1) /      # :346-347
              MAX(kpp.phepsi, dBdz2 - dBdz1, p="a"))
    zRef_3 = jnp.abs(rF_[jnp.clip(k, 0, Nr)])                                  # :349  ABS(rF(k+1)) MINMAX-RAW: index
    zRef = jnp.where(zr0, p0, jnp.where(zr1, zRef_1, jnp.where(zr2, zRef_2, zRef_3)))
    tempVar1 = p5 * (                                                          # :353-361
        (uVel[i, j, 1]-uVel[i, j, 2]) *
        (uVel[i, j, 1]-uVel[i, j, 2]) +
        (uVel[i+1, j, 1]-uVel[i+1, j, 2]) *
        (uVel[i+1, j, 1]-uVel[i+1, j, 2]) +
        (vVel[i, j, 1]-vVel[i, j, 2]) *
        (vVel[i, j, 1]-vVel[i, j, 2]) +
        (vVel[i, j+1, 1]-vVel[i, j+1, 2]) *
        (vVel[i, j+1, 1]-vVel[i, j+1, 2]))
    lo = tempVar1 < (kpp.epsln*kpp.epsln)                                      # :362
    tempVar2 = jnp.where(lo, kpp.epsln, safe_sqrt(tempVar1, ~lo))              # :363, :365
    us = ustar[i, j]
    z0 = g.drF[1] * (zFac - tempVar2 * kpp.vonk / us)                          # :370
    z0 = MAX(z0, kpp.phepsi, p="a")                                            # :372
    zRef = MAX(kpp.epsilon * zRef, z0, p="b")                                  # :375
    uRef = p5 * (uVel[i, j, 1] + uVel[i+1, j, 1])                              # :381
    vRef = p5 * (vVel[i, j, 1] + vVel[i, j+1, 1])                              # :382
    shallow = zRef < g.drF[1]                                                  # :383
    # :384-399 (zRef < drF(1))
    ustarX = (surfForcU[i, j] +
              surfForcU[i+1, j]) * p5 * g.recip_drF[1]
    ustarY = (surfForcV[i, j] +
              surfForcV[i, j+1]) * p5 * g.recip_drF[1]
    tv1 = ustarX * ustarX + ustarY * ustarY                                    # :388
    lo2 = tv1 < (kpp.epsln*kpp.epsln)                                          # :389
    tv2 = jnp.where(lo2, kpp.epsln, safe_sqrt(tv1, ~lo2))                      # :390, :392
    tv2 = (us *                                                                # :394-397
           (jnp.log(zRef * g.recip_drF[1]) +
            z0 / zRef - z0 * g.recip_drF[1]) /
           kpp.vonk / tv2)
    uRef_s = uRef + ustarX * tv2                                               # :398
    vRef_s = vRef + ustarY * tv2                                               # :399
    # :400-418 (zRef >= drF(1))
    uRef_d = uRef*g.drF[1]                                                     # :402
    vRef_d = vRef*g.drF[1]                                                     # :403
    active = jnp.ones_like(shallow)                                            # :404  k = 2
    kfin = jnp.full(shallow.shape, 2, jnp.int32)
    for kw in range(2, Nr + 1):                                                # :405-411  DO WHILE
        active = active & (jnp.abs(g.rF[kw+1]) <= zRef)
        uRef_d = jnp.where(active, uRef_d + g.drF[kw] * p5
                           * (uVel[i, j, kw] + uVel[i+1, j, kw]), uRef_d)
        vRef_d = jnp.where(active, vRef_d + g.drF[kw] * p5
                           * (vVel[i, j, kw] + vVel[i, j+1, kw]), vRef_d)
        kfin = jnp.where(active, kw + 1, kfin)                                 # :410  k = k+1
    kf = jnp.clip(kfin, 1, Nr)                 # MINMAX-RAW: in bounds already (docstring); keeps the gather valid

    def at_k(A, di, dj):
        lo_i, lo_j = A.dims[0][1], A.dims[1][1]
        s = A.data[:, :, j.first + dj - lo_j:j.last + dj - lo_j + 1, i.first + di - lo_i:i.last + di - lo_i + 1]
        return jnp.take_along_axis(s, (kf - 1)[:, None], axis=1)[:, 0]
    dz = MAX(0., zRef - jnp.abs(rF_[kf - 1]), p="b")                           # :413, :415  MAX(0.,zref-ABS(rF(k)))
    uRef_d = uRef_d + dz * p5 * (at_k(uVel, 0, 0) + at_k(uVel, 1, 0))          # :413-414
    vRef_d = vRef_d + MAX(0., zRef - jnp.abs(rF_[kf - 1]), p="b") * p5 * (     # :415-416
        at_k(vVel, 0, 0) + at_k(vVel, 0, 1))
    uRef_d = uRef_d/zRef                                                       # :417
    vRef_d = vRef_d/zRef                                                       # :418
    uRef = jnp.where(shallow, uRef_s, uRef_d)                                  # :383-419
    vRef = jnp.where(shallow, vRef_s, vRef_d)
    kk, j3, i3 = loops_kji((1, Nr), (jMin, jMax), (iMin, iMax))                # :424-461
    uR, vR = uRef[:, None], vRef[:, None]
    dVsq = dVsq.at[i3, j3, kk].set(p5 * (
        (uR - uVel[i3, j3, kk]) *
        (uR - uVel[i3, j3, kk]) +
        (uR - uVel[i3+1, j3, kk]) *
        (uR - uVel[i3+1, j3, kk]) +
        (vR - vVel[i3, j3, kk]) *
        (vR - vVel[i3, j3, kk]) +
        (vR - vVel[i3, j3+1, kk]) *
        (vR - vVel[i3, j3+1, kk])))
    return _ret(ustar, bo, bosol, dVsq)
