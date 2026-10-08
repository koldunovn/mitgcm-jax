"""KPP_CALC: pkg/kpp/kpp_calc.F @63cdc0b, routine KPP_CALC (kpp_calc.F:19-719; KPP_CALC_DUMMY of the same file is
mitjax/pkg/kpp/kpp_calc_dummy.py)."""

import jax.numpy as jnp

from mitjax.farray import FArray, loop_i, loop_j, loops_kji
from mitjax.model.src.calc_3d_diffusivity import calc_3d_diffusivity
from mitjax.pkg.generic_advdiff.gad_h import GAD_SALINITY, GAD_TEMPERATURE
from mitjax.pkg.kpp.kpp_forcing_surf import kpp_forcing_surf
from mitjax.pkg.kpp.kpp_params_h import MDIFF, imt as _imt_of
from mitjax.ops.fortran_minmax import MAX
from mitjax.ops.scan_k import scan_levels
from mitjax.pkg.kpp.kpp_routines import _swfrac, imt_view, kpp_doublediff, kppmix, smooth_horiz, statekpp, xy_view

MINUSONE = -1.0                                                                # kpp_calc.F:149
p0, p5, p25, p125, p0625 = 0.0, 0.5, 0.25, 0.125, 0.0625                       # kpp_calc.F:151 (REAL*4, exact)


def kpp_calc(myTime, myIter, *, cfg, grid, params, fp, eos, state, ff, kpp, kppf, salt_plume=None):
    """KPP_CALC( bi, bj, myTime, myIter, myThid )   @63cdc0b pkg/kpp/kpp_calc.F:19-719

    C     | SUBROUTINE KPP_CALC                                      |
    C     | o Compute all KPP fields defined in KPP.h                |
    C     | This subroutine serves as an interface between MITGCMUV  |
    C     | code and NCOM 1-D routines in kpp_routines.F             |
    c     compute vertical mixing coefficients based on the k-profile
    c     and oceanic planetary boundary layer scheme by large & mcwilliams.
    c     KPP_CALC computes vertical viscosity and diffusivity for region
    c     (-2:sNx+3,-2:sNy+3) as required by CALC_DIFFUSIVITY and requires
    c     values of uVel, vVel, surfaceForcingU, surfaceForcingV in the
    c     region (-2:sNx+4,-2:sNy+4).

    `state`: DYNVARS.h (theta, salt, uVel, vVel, IVDConvCount); `ff`: FFIELDS.h (surfaceForcingU/V/T/S,
    adjustColdSST_diag, Qsw); `fp`: forcing parameters (HeatCapacity_Cp, selectPenetratingSW); `eos`: EOS.h;
    `kpp`: KPP_PARAMS.h (Kpp); `kppf`: the KPP.h fields {name: FArray}. Returns the new `kppf` (KPPfrac set only by
    the SHORTWAVE_HEATING arm :635-671, unchanged otherwise).

    Arms: the vermix/code options (KPP_OPTIONS.h: KPP_ESTIMATE_UREF, KPP_GHAT; no KPP_SMOOTH_*, no SHORTWAVE_HEATING)
    and, lane M4COL, those of 1D_ocean_ice_column/code (the default pkg/kpp/KPP_OPTIONS.h: KPP_SMOOTH_SHSQ :18,
    KPP_SMOOTH_DBLOC :22, KPP_ESTIMATE_UREF undefined :39, KPP_GHAT :63; CPP_OPTIONS.h:21 SHORTWAVE_HEATING):
      * KPP_SMOOTH_DBLOC (:270-283): `DO k = 1, Nr-1: CALL SMOOTH_HORIZ(k+1, ..., ghat(1-OLx,1-OLy,k))`, a per-level
        caller: a level scan (KERNEL_GUIDE §4) of SMOOTH_HORIZ on the level section of ghat;
      * KPP_SMOOTH_SHSQ (:471-489): the second statement of the shsq loop, in the same k-vectorised nest;
      * no KPP_ESTIMATE_UREF: KPP_FORCING_SURF without dbloc (:423-425), its :463-503 arm;
      * SHORTWAVE_HEATING: KPPMIX gets swatt = SWFrac3D(1-OLx,1-OLy,1,bi,bj) (FFIELDS.h, :556-558) and returns
        kbl (:565-567, INTEGER (i,j) local); KPPfrac = 1 - worka (:635-671) with worka interpolated in SWFrac3D at
        kbl (KPPuseSWfrac3D, static) or, lane M4LAB (lab_sea: KPPuseSWfrac3D = .FALSE.), the SWFRAC arm :652-663
        (worka = KPPhbl, SWFRAC(imt, minusone, worka), minusone = -1.0 :149). worka is written on every point
        (i,j) before each read, so it is a value, not a declared local.
    Lane M4LAB (lab_sea/code: ALLOW_SALT_PLUME compiled, useSALT_PLUME = .FALSE.): `salt_plume` = the SALT_PLUME.h
    fields {saltPlumeFlux, SaltPlumeDepth}; temp1 = saltPlumeFlux at k = 1, 0 below, temp2 = 0 (:399-416, jMin..jMax,
    iMin..iMax; the outer ring of the locals is never written: NaN) go to KPP_FORCING_SURF as SPforcS, SPforcT,
    which returns boplume (zero); boplume and SaltPlumeDepth go to KPPMIX. All of them are read only by the
    IF ( useSALT_PLUME ) arms, which raise (as KPPplumefrac, :672-712); SALT_PLUME_VOLUME / _SPLIT_BASIN raise.
    Each other option raises. The test of :222-223,
    `DIFFERENT_MULTIPLE(kpp_freq,myTime,deltaTClock) .OR. myTime .EQ. startTime`, is decided on the host: with
    kpp_freq = deltaTClock (data.kpp default, kpp_readparms.F:81; every M3 variant) DIFFERENT_MULTIPLE(f, t, f) is
    .TRUE. for every t >= 0 (eesupp/src/different_multiple.F:49-56: v4 = NINT(t/f)*f, d1 = t - v4 in [-f/2, f/2)
    for t >= 0, so ABS(d1) < ABS(d1-f) and ABS(d1) <= ABS(d1+f)), and the model time of a forward run is >= 0; any
    other kpp_freq raises (the fields would keep their previous values between updates). Loops: :262-268, :313-346,
    :445-451, :454-492, :574-588 have independent levels (each reads only inputs at its own and the neighbouring
    levels and writes its own level): k-vectorised (the kp1 = MIN(Nr,k+1) / km1 = MAX(1,k-1) edge level written
    separately). The (i,j) arrays are passed to KPPMIX as `(imt)` arrays (kpp_routines.imt_view). The diagnostics
    (KPPshsq) are output only."""
    for opt in ("KPP_SMOOTH_DENS", "KPP_SMOOTH_VISC", "KPP_SMOOTH_DIFF", "ALLOW_SHELFICE",
                "SALT_PLUME_VOLUME", "SALT_PLUME_SPLIT_BASIN"):
        if cfg.cpp.flag(opt, "KPP_OPTIONS.h"):
            raise NotImplementedError(f"KPP_CALC: {opt} is not ported")
    smooth_dbloc = cfg.cpp.flag("KPP_SMOOTH_DBLOC", "KPP_OPTIONS.h")
    smooth_shsq = cfg.cpp.flag("KPP_SMOOTH_SHSQ", "KPP_OPTIONS.h")
    estimate_uref = cfg.cpp.flag("KPP_ESTIMATE_UREF", "KPP_OPTIONS.h")
    sw_heating = cfg.cpp.flag("SHORTWAVE_HEATING", "KPP_OPTIONS.h")
    salt_plume_on = cfg.cpp.flag("ALLOW_SALT_PLUME", "KPP_OPTIONS.h")
    if salt_plume_on and salt_plume is None:
        raise ValueError("KPP_CALC: ALLOW_SALT_PLUME needs the SALT_PLUME.h fields `salt_plume`")
    if salt_plume_on and cfg.use_flag("useSALT_PLUME"):                        # :672-712
        raise NotImplementedError("KPP_CALC: useSALT_PLUME (KPPplumefrac, SALT_PLUME_FRAC) is not ported")
    if not kpp.kpp_freq_eq_deltaTClock:                                        # :222-223 (docstring)
        raise NotImplementedError("KPP_CALC: kpp_freq /= deltaTClock (updates at a lower frequency) is not ported")
    sz = cfg.size
    Nr = sz.Nr
    n = _imt_of(sz)
    g = grid
    iMin, iMax, jMin, jMax = 2-sz.OLx, sz.sNx+sz.OLx-1, 2-sz.OLy, sz.sNy+sz.OLy-1   # :153
    # :215-219 ikey: with ALLOW_AUTODIFF_TAMC the tile's tape key (:216, lane M4ADCOL: 1D_ocean_ice_column/code_ad),
    # else 0 (:218); KPPMIX and its callees read it only in CADJ keys and the tape keys kkey (kpp_routines.F:480,
    # :1588), no value: the port passes 0 in both builds
    ikey = 0                                                                   # :218
    kppf = dict(kppf)

    # :257-260  STATEKPP
    sdens, dbloc, Ritop, TTALPHA, SSBETA = statekpp(ikey, cfg=cfg, grid=grid, params=params, eos=eos, state=state)
    k3, j3, i3 = loops_kji((1, Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))   # :262-268
    ghat = FArray(dbloc.data, "ghat", tiled=True, _dims=dbloc.dims).at[i3, j3, k3].set(dbloc[i3, j3, k3])
    if smooth_dbloc and Nr > 1:                                                # :270-283  KPP_SMOOTH_DBLOC
        ia, ja = loop_i(1-sz.OLx, sz.sNx+sz.OLx), loop_j(1-sz.OLy, sz.sNy+sz.OLy)

        def smooth_k(k, ghat):                                                 # :276-281  DO k = 1, Nr-1
            kk = k - 1
            fld = FArray(ghat.data[:, getattr(kk, "value", kk)], "ghat", i=(1-sz.OLx, sz.sNx+sz.OLx),
                         j=(1-sz.OLy, sz.sNy+sz.OLy))                          # ghat(1-OLx,1-OLy,k) as a 2-D dummy
            fld = smooth_horiz(k+1, fld, cfg=cfg, grid=grid)
            return ghat.at[ia, ja, k].set(fld[ia, ja])
        ghat = scan_levels(smooth_k, ghat, 1, Nr-1)
    # :312-346  kSurf = 1; DO k = 1, Nr: kp1 = MIN(Nr,k+1) -> k+1 for k < Nr, Nr for k = Nr
    kSurf = 1
    nzm = kpp.nzmax[loop_i(1-sz.OLx, sz.sNx+sz.OLx), loop_j(1-sz.OLy, sz.sNy+sz.OLy)][:, None]   # [tile,1,j,i]
    zg = kpp.zgrid
    for (klo, khi, kp1off) in ((1, Nr-1, 1), (Nr, Nr, 0)):
        if khi < klo:
            continue
        k, j, i = loops_kji((klo, khi), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
        kv = jnp.arange(klo, khi + 1, dtype=jnp.int32)[None, :, None, None]
        mk = g.maskC[i, j, k]
        mkp1 = g.maskC[i, j, k + kp1off]
        db = dbloc[i, j, k] * mk * mkp1                                        # :324-325
        gh = ghat[i, j, k] * mk * mkp1                                         # :326-327
        rt = Ritop[i, j, k] * mk * g.maskC[i, j, kSurf]                        # :331-332
        bot = kv == nzm                                                        # :333-337
        db = jnp.where(bot, p0, db)
        gh = jnp.where(bot, p0, gh)
        rt = jnp.where(bot, p0, rt)
        zdif = (zg[1] - zg.data[klo:khi + 1])[None, :, None, None]            # (zgrid(1)-zgrid(k)); zgrid(0:Nr+1)
        rt = zdif * rt                                                         # :342
        dbloc = dbloc.at[i, j, k].set(db)
        ghat = ghat.at[i, j, k].set(gh)
        Ritop = Ritop.at[i, j, k].set(rt)

    sp_kw = {}
    if salt_plume_on:                                                          # :399-416  #ifndef SALT_PLUME_VOLUME
        j, i = loop_j(jMin, jMax), loop_i(iMin, iMax)
        k2, j2, i2 = loops_kji((2, Nr), (jMin, jMax), (iMin, iMax))
        temp1 = dbloc.local("temp1").at[i, j, 1].set(salt_plume["saltPlumeFlux"][i, j])
        temp1 = temp1.at[i2, j2, k2].set(0.)                                   # 0. _d 0
        temp2 = dbloc.local("temp2").at[i, j, 1].set(0.)
        temp2 = temp2.at[i2, j2, k2].set(0.)
        sp_kw = dict(SPforcS=temp1, SPforcT=temp2, boplume=TTALPHA.local("boplume"))   # boplume (.., Nrp1)

    # :419-435  KPP_FORCING_SURF (locals ustar, bo, bosol, dVsq: every point written)
    two = sdens.local("ustar")
    dVsq = dbloc.local("dVsq")
    fs = kpp_forcing_surf(
        sdens, ff.surfaceForcingU, ff.surfaceForcingV, ff.surfaceForcingT, ff.surfaceForcingS,
        ff.adjustColdSST_diag, ff.Qsw, dbloc if estimate_uref else None,       # :423-425  #ifdef KPP_ESTIMATE_UREF
        TTALPHA, SSBETA, two, sdens.local("bo"), sdens.local("bosol"), dVsq,
        ikey, iMin, iMax, jMin, jMax, myTime, cfg=cfg, grid=grid, params=params, fp=fp, kpp=kpp, state=state,
        **sp_kw)                                                               # :426-431  #ifdef ALLOW_SALT_PLUME
    if salt_plume_on:
        ustar, bo, bosol, boplume, dVsq = fs
        mix_sp = dict(boplume=imt_view(boplume), SPDepth=imt_view(salt_plume["SaltPlumeDepth"]))   # :549-555
    else:
        ustar, bo, bosol, dVsq = fs
        mix_sp = {}

    # :445-451  shsq = p0; :454-492 (k = 1..Nrm1, independent levels)
    shsq = dbloc.local("shsq").at[i3, j3, k3].set(p0)
    uVel, vVel = state.uVel, state.vVel
    if Nr > 1:
        k, j, i = loops_kji((1, Nr-1), (jMin, jMax), (iMin, iMax))
        shsq = shsq.at[i, j, k].set(p5 * (                                     # :462-470
            (uVel[i, j, k]-uVel[i, j, k+1]) *
            (uVel[i, j, k]-uVel[i, j, k+1]) +
            (uVel[i+1, j, k]-uVel[i+1, j, k+1]) *
            (uVel[i+1, j, k]-uVel[i+1, j, k+1]) +
            (vVel[i, j, k]-vVel[i, j, k+1]) *
            (vVel[i, j, k]-vVel[i, j, k+1]) +
            (vVel[i, j+1, k]-vVel[i, j+1, k+1]) *
            (vVel[i, j+1, k]-vVel[i, j+1, k+1])))
        if smooth_shsq:                                                        # :471-489  KPP_SMOOTH_SHSQ
            shsq = shsq.at[i, j, k].set(p5 * shsq[i, j, k] + p125 * (
                (uVel[i, j-1, k]-uVel[i, j-1, k+1]) *
                (uVel[i, j-1, k]-uVel[i, j-1, k+1]) +
                (uVel[i+1, j-1, k]-uVel[i+1, j-1, k+1]) *
                (uVel[i+1, j-1, k]-uVel[i+1, j-1, k+1]) +
                (uVel[i, j+1, k]-uVel[i, j+1, k+1]) *
                (uVel[i, j+1, k]-uVel[i, j+1, k+1]) +
                (uVel[i+1, j+1, k]-uVel[i+1, j+1, k+1]) *
                (uVel[i+1, j+1, k]-uVel[i+1, j+1, k+1]) +
                (vVel[i-1, j, k]-vVel[i-1, j, k+1]) *
                (vVel[i-1, j, k]-vVel[i-1, j, k+1]) +
                (vVel[i-1, j+1, k]-vVel[i-1, j+1, k+1]) *
                (vVel[i-1, j+1, k]-vVel[i-1, j+1, k+1]) +
                (vVel[i+1, j, k]-vVel[i+1, j, k+1]) *
                (vVel[i+1, j, k]-vVel[i+1, j, k+1]) +
                (vVel[i+1, j+1, k]-vVel[i+1, j+1, k+1]) *
                (vVel[i+1, j+1, k]-vVel[i+1, j+1, k+1])))

    # :517-526  background diffusivities
    kppf["KPPdiffKzS"] = calc_3d_diffusivity(1-sz.OLx, sz.sNx+sz.OLx, 1-sz.OLy, sz.sNy+sz.OLy, GAD_SALINITY,
                                             False, False, kppf["KPPdiffKzS"], cfg=cfg, grid=grid, params=params,
                                             state=state)
    kppf["KPPdiffKzT"] = calc_3d_diffusivity(1-sz.OLx, sz.sNx+sz.OLx, 1-sz.OLy, sz.sNy+sz.OLy, GAD_TEMPERATURE,
                                             False, False, kppf["KPPdiffKzT"], cfg=cfg, grid=grid, params=params,
                                             state=state)
    if kpp.KPPuseDoubleDiff:                                                   # :528-538
        if cfg.cpp.flag("EXCLUDE_KPP_DOUBLEDIFF", "KPP_OPTIONS.h"):
            raise NotImplementedError("KPP_CALC: EXCLUDE_KPP_DOUBLEDIFF with KPPuseDoubleDiff")
        kppf["KPPdiffKzT"], kppf["KPPdiffKzS"] = kpp_doublediff(
            TTALPHA, SSBETA, kppf["KPPdiffKzT"], kppf["KPPdiffKzS"], ikey, 1-sz.OLx, sz.sNx+sz.OLx, 1-sz.OLy,
            sz.sNy+sz.OLy, cfg=cfg, kpp=kpp, state=state)

    # :545-568  KPPMIX on (imt) views (sequence association)
    T = dbloc.data.shape[0]
    vd = {md: FArray(jnp.full((T, Nr + 2, n), jnp.nan), "vddiff", i=(1, n), k=(0, Nr + 1))
          for md in range(1, MDIFF + 1)}
    cor = FArray(g.fCori.data, "fCori", i=(1-sz.OLx, sz.sNx+sz.OLx), j=(1-sz.OLy, sz.sNy+sz.OLy))
    msk = FArray(g.maskC.data[:, 0], "maskC", i=(1-sz.OLx, sz.sNx+sz.OLx), j=(1-sz.OLy, sz.sNy+sz.OLy))
    hbl0 = sdens.local("hbl")
    mx = kppmix(
        imt_view(kpp.nzmax), imt_view(shsq), imt_view(dVsq), imt_view(ustar), imt_view(msk), imt_view(bo),
        imt_view(bosol), imt_view(dbloc), imt_view(Ritop), imt_view(cor), imt_view(kppf["KPPdiffKzS"]),
        imt_view(kppf["KPPdiffKzT"]), ikey, vd, imt_view(ghat), imt_view(hbl0), myTime, myIter,
        cfg=cfg, kpp=kpp, params=params,
        swatt=imt_view(ff.SWFrac3D) if sw_heating else None, fp=fp,           # :556-558 swatt, :565-567 kbl
        **mix_sp)
    vd, ghat_imt, hbl_imt = mx[:3]
    ghat = xy_view(ghat_imt, ghat)
    hbl_v = hbl_imt.reshape(hbl0.data.shape)                                   # hbl (imt) -> hbl(i,j)
    vdx = {md: FArray(vd[md].data.reshape((T, Nr + 2) + dbloc.data.shape[-2:]), "vddiff",
                      i=(1-sz.OLx, sz.sNx+sz.OLx), j=(1-sz.OLy, sz.sNy+sz.OLy), k=(0, Nr + 1))
           for md in vd}

    # :574-588  zero out land values and transfer to global variables (km1 = MAX(1,k-1): k = 1 and k = 2..Nr)
    for (klo, khi, km1off) in ((1, 1, 0), (2, Nr, -1)):
        if khi < klo:
            continue
        k, j, i = loops_kji((klo, khi), (jMin, jMax), (iMin, iMax))
        mk, mkm1 = g.maskC[i, j, k], g.maskC[i, j, k + km1off]
        kppf["KPPviscAz"] = kppf["KPPviscAz"].at[i, j, k].set(vdx[1][i, j, k-1] * mk * mkm1)    # :578-579
        kppf["KPPdiffKzS"] = kppf["KPPdiffKzS"].at[i, j, k].set(vdx[2][i, j, k-1] * mk * mkm1)  # :580-581
        kppf["KPPdiffKzT"] = kppf["KPPdiffKzT"].at[i, j, k].set(vdx[3][i, j, k-1] * mk * mkm1)  # :582-583
        kppf["KPPghat"] = kppf["KPPghat"].at[i, j, k].set(ghat[i, j, k] * mk * mkm1)            # :584-585
    j = loop_j(jMin, jMax)                                                     # :589-597
    i = loop_i(iMin, iMax)
    hblf = FArray(hbl_v, "hbl", tiled=True, _dims=hbl0.dims)
    kppf["KPPhbl"] = kppf["KPPhbl"].at[i, j].set(hblf[i, j] * g.maskC[i, j, kSurf])   # :595

    if sw_heating and fp.selectPenetratingSW >= 1 and not kpp.KPPuseSWfrac3D:  # :635-670 (lane M4LAB)
        ja, ia = loop_j(1-sz.OLy, sz.sNy+sz.OLy), loop_i(1-sz.OLx, sz.sNx+sz.OLx)   # :652-663  ELSE
        worka = _swfrac(MINUSONE, kppf["KPPhbl"][ia, ja])                      # worka(i,j) = KPPhbl; SWFRAC
        kppf["KPPfrac"] = kppf["KPPfrac"].at[ia, ja].set(1. - worka)           # :665-669  1. _d 0
    elif sw_heating and fp.selectPenetratingSW >= 1:                           # :635-670
        kbl = mx[3].reshape(hbl0.data.shape)                                   # kbl (imt) -> kbl(i,j), int32
        ja, ia = loop_j(1-sz.OLy, sz.sNy+sz.OLy), loop_i(1-sz.OLx, sz.sNx+sz.OLx)   # :644-651 KPPuseSWfrac3D
        k = kbl[:, None]                                                       # :646  k = kbl(i,j)  [tile,1,j,i]
        rF_, rdrF = g.rF.data, g.recip_drF.data                                # level vectors (k = 1 at index 0)
        rFac = MAX((kppf["KPPhbl"][ia, ja][:, None]+rF_[k - 1])*rdrF[k - 1], 0.0, p="a")   # :647  zeroRL
        sw = ff.SWFrac3D.data                                                  # [tile, k=1..Nr+1, j, i]
        sw_k = jnp.take_along_axis(sw, k - 1, axis=1)                          # SWFrac3D(i,j,k)
        sw_kp1 = jnp.take_along_axis(sw, k, axis=1)                            # SWFrac3D(i,j,k+1)
        worka = (sw_k
                 + rFac*(sw_kp1 - sw_k))[:, 0]                                 # :648-649
        kppf["KPPfrac"] = kppf["KPPfrac"].at[ia, ja].set(1. - worka)           # :665-669  1. _d 0
    return kppf

