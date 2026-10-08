"""SEAICE_ADVECTION: pkg/seaice/seaice_advection.F @63cdc0b (lane M4OFF: the multi-dimensional advection of one
sea-ice field, the non-cube arm with the flux-limited scheme 77; session 3: the PPM schemes 40-42, GAD_PPM_ADV_X/Y,
for offline_exf_seaice/input.dyn_lsr's scheme 41; lane M4LAB session 3: OS7MP, scheme 7, GAD_OS7MP_ADV_X/Y, for
lab_sea/input; lane M4ADCS32ICE: DST3 / DST3FL, schemes 30 / 33, GAD_DST3(FL)_ADV_X/Y, and the ALLOW_AUTODIFF arms,
for global_ocean.cs32x15/input_ad.seaice(_dynmix))."""

from mitjax.ad.approx_advection import ad_scheme, flux_call, seaice_level
from mitjax.farray import loop_i, loop_j
from mitjax.pkg.generic_advdiff.gad_calc_rhs import gad_kernel_cfg
from mitjax.pkg.generic_advdiff.gad_dst3_adv_x import gad_dst3_adv_x
from mitjax.pkg.generic_advdiff.gad_dst3_adv_y import gad_dst3_adv_y
from mitjax.pkg.generic_advdiff.gad_dst3fl_adv_x import gad_dst3fl_adv_x
from mitjax.pkg.generic_advdiff.gad_dst3fl_adv_y import gad_dst3fl_adv_y
from mitjax.pkg.generic_advdiff.gad_fluxlimit_adv_x import gad_fluxlimit_adv_x
from mitjax.pkg.generic_advdiff.gad_fluxlimit_adv_y import gad_fluxlimit_adv_y
from mitjax.pkg.generic_advdiff.gad_h import (ENUM_DST3, ENUM_DST3_FLUX_LIMIT, ENUM_FLUX_LIMIT, ENUM_OS7MP,
                                              ENUM_PPM_MONO_LIMIT, ENUM_PPM_NULL_LIMIT, ENUM_PPM_WENO_LIMIT)
from mitjax.pkg.generic_advdiff.gad_os7mp_adv_x import gad_os7mp_adv_x
from mitjax.pkg.generic_advdiff.gad_os7mp_adv_y import gad_os7mp_adv_y
from mitjax.pkg.generic_advdiff.gad_ppm_adv_x import gad_ppm_adv_x
from mitjax.pkg.generic_advdiff.gad_ppm_adv_y import gad_ppm_adv_y

extensiveFld = True        # seaice_advection.F:69-70  PARAMETER ( extensiveFld = .TRUE. )


def seaice_advection(tracerIdentity, advectionSchArg, uFld, vFld, uTrans, vTrans, iceFld, r_hFld, gFld, afx, afy,
                     myTime, myIter, *, cfg, sp, grid, SIMaskU, SIMaskV, useCubedSphereExchange):
    """SEAICE_ADVECTION( tracerIdentity, advectionSchArg, uFld, vFld, uTrans, vTrans, iceFld, r_hFld, gFld, afx,
    afy, bi, bj, myTime, myIter, myThid )   @63cdc0b pkg/seaice/seaice_advection.F:14-796

    C Calculates the tendency of a sea-ice field due to advection.
    C It uses the multi-dimensional method given in \\ref{sect:multiDimAdvection}
    C and can only be used for the non-linear advection schemes such as the
    C direct-space-time method and flux-limiters.

    `SIMaskU`, `SIMaskV` the SEAICE_GRID.h masks; `useCubedSphereExchange` EEPARAMS.h's (static). The tile loop of the caller is the leading axis of the FArrays (every tile at once). Returns (gFld, afx, afy).
    Ported: the non-cube arm (useCubedSphereExchange = .FALSE.: nipass = 2, X then Y, :232-239, :320-324) and (lane
    M4CS32ICE) the cube arm with ALLOW_EXCH2 (`_cs_passes`: nipass = 3, the masks' FILL_CS_CORNER_UV_RS, the
    seaice flags of :303-318 and the corner fills of :356-359, :424-427, :570-573, :638-641), no ALLOW_OBCS (maskLocW/S = SIMaskU/V, :271-274),
    advectionScheme = ENUM_FLUX_LIMIT (GAD_FLUXLIMIT_ADV_X/Y, :381-384, :595-598), ENUM_DST3 / ENUM_DST3_FLUX_LIMIT
    (GAD_DST3_ADV_X/Y, GAD_DST3FL_ADV_X/Y, :385-392, :599-606; lane M4ADCS32ICE), ENUM_OS7MP
    (GAD_OS7MP_ADV_X/Y, :393-396, :607-610; lane M4LAB) or a PPM scheme (the other schemes raise), extensiveFld = .TRUE. (:507-516, :721-730).
    ALLOW_AUTODIFF (lane M4ADCS32ICE): localTij = 0. before its copy (:207-213, overwritten on the same range),
    afx = afy = 0. on every point (:278-286); the scheme swap of :176-186 (inAdMode .AND. useApproxAdvectionInAdMode)
    never runs in a forward step (inAdMode is .FALSE.: AUTODIFF_INADMODE_UNSET, forward_step.F:433-435; the backward
    pass: mitjax/ad/approx_advection.py); with ALLOW_AUTODIFF_TAMC the STOPs tracerIdentity > maxpass
    (:164-173) and nipass > maxcube (:245-251), maxpass / maxcube the build's tamc.h PARAMETERs (SEAICE_READPARMS
    reads them: sp.mjx_tamc_maxpass / sp.mjx_tamc_maxcube). The debug prints (dBug, :195-205, :325-326, ...) and the
    diagnostics (:774-781) are output only. `tracerIdentity` keys TAF tapes and diagnostics only. Locals not written
    by the Fortran before a read are not read: af is zeroed on every point before each flux (:348-352, :562-566),
    localTij is iceFld on every point (:265-276); afx/afy are returned with the points the Fortran writes (others
    as given)."""
    del myTime, myIter
    if cfg.cpp.flag("ALLOW_OBCS"):
        raise NotImplementedError("SEAICE_ADVECTION: ALLOW_OBCS is not ported")
    if useCubedSphereExchange and not cfg.cpp.ALLOW_EXCH2:                    # :216-231
        raise NotImplementedError("SEAICE_ADVECTION: the cube without ALLOW_EXCH2 (nCFace = bi) is not ported")
    ad = bool(cfg.cpp.flag("ALLOW_AUTODIFF"))
    tamc = bool(ad and cfg.cpp.flag("ALLOW_AUTODIFF_TAMC", "AUTODIFF_OPTIONS.h"))
    advectionScheme = advectionSchArg                                          # :163
    if tamc and getattr(sp, "mjx_tamc_maxpass", None) is None:
        raise NotImplementedError("SEAICE_ADVECTION: tamc.h maxpass of a build with pkg/ptracers is not read")
    if tamc and tracerIdentity > sp.mjx_tamc_maxpass:                          # :164-173
        raise RuntimeError(f"SEAICE_ADVECTION: tracerIdentity > maxpass {tracerIdentity} {sp.mjx_tamc_maxpass}: "
                           "ABNORMAL END: S/R SEAICE_ADVECTION")
    # :176-186 (ALLOW_AUTODIFF) IF ( inAdMode .AND. useApproxAdvectionInAdMode ): inAdMode is .FALSE. in every
    # forward step (AUTODIFF_INADMODE_UNSET, forward_step.F:433-435), so the scheme swap never runs here; lane
    # M4ADCS32ICE session 2: in the reverse sweep the swap is the backward-only switch of
    # mitjax/ad/approx_advection.py ("flux" level: each flux call's derivative is that of the AD-mode scheme)
    ad_sch = ad_scheme(advectionScheme) if (ad and seaice_level(sp) == "flux") else None
    nipass = 3 if useCubedSphereExchange else 2                                # :217, :233
    if tamc and nipass > sp.mjx_tamc_maxcube:                                  # :245-251
        raise RuntimeError(f"S/R SEAICE_ADVECTION: nipass = {nipass} > {sp.mjx_tamc_maxcube} = maxcube, ==> check "
                           '"tamc.h": ABNORMAL END: S/R SEAICE_ADVECTION')
    ppm = advectionScheme in (ENUM_PPM_NULL_LIMIT, ENUM_PPM_MONO_LIMIT, ENUM_PPM_WENO_LIMIT)   # :398-403, :612-617
    if ppm and cfg.cpp.flag("ALLOW_AUTODIFF"):                                 # :397 #ifndef ALLOW_AUTODIFF
        raise NotImplementedError("SEAICE_ADVECTION: PPM schemes are not compiled with ALLOW_AUTODIFF")
    if advectionScheme not in (ENUM_FLUX_LIMIT, ENUM_DST3, ENUM_DST3_FLUX_LIMIT, ENUM_OS7MP) and not ppm:   # :370-417
        raise NotImplementedError(f"SEAICE_ADVECTION: adv. scheme {advectionScheme} is not ported (77, 30, 33, 7, "
                                  "40-42)")
    sz = cfg.size
    OLx, OLy, sNx, sNy = sz.OLx, sz.OLy, sz.sNx, sz.sNy
    kc = gad_kernel_cfg(cfg)
    deltaT = sp.SEAICE_deltaTtherm
    k = 1                                                                      # :254
    localTij = iceFld.local("localTij")
    jA = loop_j(1-OLy, sNy+OLy)
    iA = loop_i(1-OLx, sNx+OLx)
    if ad:                                                                     # :207-213
        localTij = localTij.at[iA, jA].set(0.)                                 # :210  0. _d 0
    localTij = localTij.at[iA, jA].set(iceFld[iA, jA])                        # :267
    maskLocW = grid_mask_like(iceFld, "maskLocW", SIMaskU)                # :272
    maskLocS = grid_mask_like(iceFld, "maskLocS", SIMaskV)                # :273
    if ad:                                                                     # :278-286
        afx = afx.at[iA, jA].set(0.)                                           # :282
        afy = afy.at[iA, jA].set(0.)                                           # :283
    maskInC, recip_rA = grid.maskInC, grid.recip_rA
    if useCubedSphereExchange:                                                 # lane M4CS32ICE (cs32)
        localTij, afx, afy = _cs_passes(advectionScheme, ppm, k, deltaT, uFld, vFld, uTrans, vTrans, localTij,
                                        maskLocW, maskLocS, afx, afy, kc=kc, grid=grid, ad_sch=ad_sch)
        gFld = gFld.at[iA, jA].set((localTij[iA, jA]-iceFld[iA, jA])/deltaT)    # :760-764
        return gFld, afx, afy
    for ipass in range(1, nipass + 1):                                         # :298
        calc_fluxes_X = (ipass % 2) == 1                                       # :322
        calc_fluxes_Y = not calc_fluxes_X                                      # :323
        if calc_fluxes_X:                                                      # :340-540
            af = iceFld.local("af").at[iA, jA].set(0.)                         # :348-352
            if ppm:                                                            # :398-403 (session 3)
                af = gad_ppm_adv_x(advectionScheme, k, True, deltaT, uFld, uTrans, localTij, af, cfg=kc,
                                   grid=grid)
            else:
                af = _adv_x(advectionScheme, k, deltaT, uTrans, uFld, maskLocW, localTij, af, kc=kc, grid=grid,
                           ad_sch=ad_sch)
            j = loop_j(1-OLy, sNy+OLy)                                         # :503-506 (jMinUpd, jMaxUpd)
            i = loop_i(1-OLx+1, sNx+OLx-1)                                     # :508-516 extensiveFld
            localTij = localTij.at[i, j].set(
                localTij[i, j]
                - deltaT*maskInC[i, j]
                * recip_rA[i, j]
                * (af[i+1, j]-af[i, j]))
            i = loop_i(1-OLx+1, sNx+OLx)                                       # :530-534
            afx = afx.at[i, j].set(af[i, j])
        if calc_fluxes_Y:                                                      # :554-754
            af = iceFld.local("af").at[iA, jA].set(0.)                         # :562-566
            if ppm:                                                            # :612-617 (session 3)
                af = gad_ppm_adv_y(advectionScheme, k, True, deltaT, vFld, vTrans, localTij, af, cfg=kc,
                                   grid=grid)
            else:
                af = _adv_y(advectionScheme, k, deltaT, vTrans, vFld, maskLocS, localTij, af, kc=kc, grid=grid,
                           ad_sch=ad_sch)
            i = loop_i(1-OLx, sNx+OLx)                                         # :717-720 (iMinUpd, iMaxUpd)
            j = loop_j(1-OLy+1, sNy+OLy-1)                                     # :722-730 extensiveFld
            localTij = localTij.at[i, j].set(
                localTij[i, j]
                - deltaT*maskInC[i, j]
                * recip_rA[i, j]
                * (af[i, j+1]-af[i, j]))
            j = loop_j(1-OLy+1, sNy+OLy)                                       # :744-748
            afy = afy.at[i, j].set(af[i, j])
    gFld = gFld.at[iA, jA].set((localTij[iA, jA]-iceFld[iA, jA])/deltaT)        # :760-764
    del r_hFld
    return gFld, afx, afy


def _adv_x(advectionScheme, k, deltaT, uTrans, uFld, maskLocW, localTij, af, *, kc, grid, ad_sch=None):
    """The X flux call of seaice_advection.F:381-396; `ad_sch` (lane M4ADCS32ICE session 2): the AD-mode scheme of the
    backward-only switch useApproxAdvectionInAdMode at the "flux" level (mitjax/ad/approx_advection.flux_call: the
    value is advectionScheme's, the reverse derivative ad_sch's), None = off."""
    return flux_call(lambda sch, dt, uT, uF, mW, lT, a, g: _adv_x_scheme(sch, k, dt, uT, uF, mW, lT, a, kc=kc, grid=g),
                     advectionScheme, ad_sch, deltaT, uTrans, uFld, maskLocW, localTij, af, grid)


def _adv_x_scheme(advectionScheme, k, deltaT, uTrans, uFld, maskLocW, localTij, af, *, kc, grid):
    """The X flux of a non-PPM scheme, seaice_advection.F:381-396 (calcCFL = .TRUE., SEAICE_deltaTtherm)."""
    if advectionScheme == ENUM_FLUX_LIMIT:                                     # :381-384
        return gad_fluxlimit_adv_x(k, True, deltaT, uTrans, uFld, maskLocW, localTij, af, cfg=kc, grid=grid)
    if advectionScheme == ENUM_DST3:                                           # :385-388 (lane M4ADCS32ICE)
        return gad_dst3_adv_x(k, True, deltaT, uTrans, uFld, maskLocW, localTij, af, cfg=kc, grid=grid)
    if advectionScheme == ENUM_DST3_FLUX_LIMIT:                                # :389-392 (lane M4ADCS32ICE)
        return gad_dst3fl_adv_x(k, True, deltaT, uTrans, uFld, maskLocW, localTij, af, cfg=kc, grid=grid)
    return gad_os7mp_adv_x(k, True, deltaT, uTrans, uFld, maskLocW, localTij, af, cfg=kc, grid=grid)   # :393-396


def _adv_y(advectionScheme, k, deltaT, vTrans, vFld, maskLocS, localTij, af, *, kc, grid, ad_sch=None):
    """The Y flux call of seaice_advection.F:595-610; `ad_sch` as _adv_x."""
    return flux_call(lambda sch, dt, vT, vF, mS, lT, a, g: _adv_y_scheme(sch, k, dt, vT, vF, mS, lT, a, kc=kc, grid=g),
                     advectionScheme, ad_sch, deltaT, vTrans, vFld, maskLocS, localTij, af, grid)


def _adv_y_scheme(advectionScheme, k, deltaT, vTrans, vFld, maskLocS, localTij, af, *, kc, grid):
    """The Y flux of a non-PPM scheme, seaice_advection.F:595-610 (calcCFL = .TRUE., SEAICE_deltaTtherm)."""
    if advectionScheme == ENUM_FLUX_LIMIT:                                     # :595-598
        return gad_fluxlimit_adv_y(k, True, deltaT, vTrans, vFld, maskLocS, localTij, af, cfg=kc, grid=grid)
    if advectionScheme == ENUM_DST3:                                           # :599-602 (lane M4ADCS32ICE)
        return gad_dst3_adv_y(k, True, deltaT, vTrans, vFld, maskLocS, localTij, af, cfg=kc, grid=grid)
    if advectionScheme == ENUM_DST3_FLUX_LIMIT:                                # :603-606 (lane M4ADCS32ICE)
        return gad_dst3fl_adv_y(k, True, deltaT, vTrans, vFld, maskLocS, localTij, af, cfg=kc, grid=grid)
    return gad_os7mp_adv_y(k, True, deltaT, vTrans, vFld, maskLocS, localTij, af, cfg=kc, grid=grid)   # :607-610


def grid_mask_like(like, name, mask):
    """A local `name(1-OLx:sNx+OLx,1-OLy:sNy+OLy)` holding `mask` (the SEAICE_GRID.h mask passed to the routine)."""
    return like.local(name).at[loop_i(*like.dims[0][1:]), loop_j(*like.dims[1][1:])].set(
        mask[loop_i(*like.dims[0][1:]), loop_j(*like.dims[1][1:])])


def _cs_pass_flags(nCFace, ipass):
    """seaice_advection.F:303-318 (useCubedSphereExchange) on every tile: (overlapOnly, interiorOnly, calc_fluxes_X,
    calc_fluxes_Y) as [T,1,1] bool arrays (nCFace >= 1: MOD is the non-negative remainder). interiorOnly is set in
    pass 1 only (gad_advection._cs_pass_flags, gad_advection.F:357/:362, also sets it in passes 2 and 3)."""
    import jax.numpy as jnp
    m3 = jnp.mod(nCFace, 3)
    interiorOnly = jnp.zeros_like(nCFace, bool)                                # :303
    overlapOnly = jnp.zeros_like(nCFace, bool)                                 # :304
    if ipass == 1:                                                             # :307-311
        overlapOnly = m3 == 0
        interiorOnly = m3 != 0
        calc_fluxes_X = (nCFace == 6) | (nCFace == 1) | (nCFace == 2)
        calc_fluxes_Y = (nCFace == 3) | (nCFace == 4) | (nCFace == 5)
    elif ipass == 2:                                                           # :312-315
        overlapOnly = m3 == 2
        calc_fluxes_X = (nCFace == 2) | (nCFace == 3) | (nCFace == 4)
        calc_fluxes_Y = (nCFace == 5) | (nCFace == 6) | (nCFace == 1)
    else:                                                                      # :316-318
        calc_fluxes_X = (nCFace == 5) | (nCFace == 6)
        calc_fluxes_Y = (nCFace == 2) | (nCFace == 3)
    return overlapOnly, interiorOnly, calc_fluxes_X, calc_fluxes_Y


def _cs_passes(advectionScheme, ppm, k, deltaT, uFld, vFld, uTrans, vTrans, localTij, maskLocW, maskLocS, afx, afy,
               *, kc, grid, ad_sch=None):
    """seaice_advection.F:216-231, :289-757 with useCubedSphereExchange and ALLOW_EXCH2 (lane M4CS32ICE), every tile
    at once. Per tile (gad_advection._cs_tile_flags: nCFace = exch2_myFace(myTile), [N,S,E,W]_edge): FILL_CS_CORNER_UV_RS
    of the masks without signs (:289-294), then three passes with SEAICE_ADVECTION's own flags (:303-318; NOT those
    of GAD_ADVECTION: interiorOnly is set in pass 1 only, gad_advection.F:357 / :362 also set it in passes 2 and 3;
    and the corner fill before the fluxes runs for `overlapOnly .OR. ipass.EQ.1`, :356-359 / :570-573, the one after
    them for `overlapOnly .AND. ipass.EQ.1` whether or not the fluxes were computed, :424-427 / :638-641). Whether a
    tile computes its fluxes (:345, :559) selects, per tile, af (zeroed then the scheme) or the af of before (the
    Fortran local keeps its value; read only by updates the same flags switch off). The update statements
    (extensiveFld, :441-484 / :507-516, :655-698 / :721-730) are evaluated on their full DO range and written where
    the Fortran's loops run (overlapOnly: the S/N (W/E) overlap rows of an edge tile between iMinUpd..iMaxUpd
    (jMinUpd..jMaxUpd); otherwise jMinUpd..jMaxUpd (iMinUpd..iMaxUpd) with the interiorOnly limits), as [T,j,i]
    masks: each point reads af at itself and its i+1 (j+1) neighbour and its own localTij. afx / afy likewise
    (:486-499, :530-534, :700-713, :744-748). Returns (localTij, afx, afy)."""
    import jax.numpy as jnp
    from mitjax.eesupp.fill_cs_corner_uv_rs import fill_cs_corner_uv_rs
    from mitjax.pkg.generic_advdiff.gad_advection import _cs_fill, _cs_tile_flags, _with_data
    sNx, sNy, OLx, OLy = kc.sNx, kc.sNy, kc.OLx, kc.OLy
    cs = _cs_tile_flags(grid)                                                  # :218-223
    N_edge, S_edge, E_edge, W_edge = cs["N_edge"], cs["S_edge"], cs["E_edge"], cs["W_edge"]
    nCFace = cs["nCFace"]
    nipass = 3                                                                 # :217
    withSigns = False                                                          # :290
    mW, mS = fill_cs_corner_uv_rs(withSigns, maskLocW.data, maskLocS.data, cs["corners"], True,   # :291-292
                                  sNx=sNx, sNy=sNy, OLx=OLx, OLy=OLy)
    maskLocW, maskLocS = _with_data(maskLocW, mW), _with_data(maskLocS, mS)
    maskInC, recip_rA = grid.maskInC, grid.recip_rA
    jj = jnp.arange(1-OLy, sNy+OLy+1)[None, :, None]                            # Fortran j of each storage row
    ii = jnp.arange(1-OLx, sNx+OLx+1)[None, None, :]
    shape = localTij.data.shape                                                # [T, j, i]
    jA, iA = loop_j(1-OLy, sNy+OLy), loop_i(1-OLx, sNx+OLx)
    af = localTij.local("af").at[iA, jA].set(0.)     # finite lanes; a tile's af is read only after its own write
    for ipass in range(1, nipass + 1):                                         # :298
        overlapOnly, interiorOnly, calc_fluxes_X, calc_fluxes_Y = _cs_pass_flags(nCFace, ipass)   # :303-318

#--   X direction                                                         # :340-551
        doFlux = calc_fluxes_X & (~overlapOnly | N_edge | S_edge)               # :340, :345
        localTij = _cs_fill(1, localTij, doFlux & (overlapOnly | (ipass == 1)), cs, kc)   # :356-359
        afX = af.at[iA, jA].set(0.)                                            # :348-352
        if ppm:                                                                # :398-403
            afX = gad_ppm_adv_x(advectionScheme, k, True, deltaT, uFld, uTrans, localTij, afX, cfg=kc, grid=grid)
        else:                                                                  # :381-396
            afX = _adv_x(advectionScheme, k, deltaT, uTrans, uFld, maskLocW, localTij, afX, kc=kc, grid=grid,
                         ad_sch=ad_sch)
        af = _with_data(af, jnp.where(doFlux, afX.data, af.data))
        localTij = _cs_fill(2, localTij, calc_fluxes_X & overlapOnly & (ipass == 1), cs, kc)   # :424-427
        iMinUpd = jnp.where(W_edge, 1, 1-OLx+1)                                # :434, :438
        iMaxUpd = jnp.where(E_edge, sNx, sNx+OLx-1)                            # :435, :439
        ovl = ((S_edge & (jj <= 0)) | (N_edge & (jj >= sNy+1))) & (ii >= iMinUpd) & (ii <= iMaxUpd)   # :441-484
        jMinUpd = jnp.where(interiorOnly & S_edge, 1, 1-OLy)                   # :503, :505
        jMaxUpd = jnp.where(interiorOnly & N_edge, sNy, sNy+OLy)               # :504, :506
        full = (jj >= jMinUpd) & (jj <= jMaxUpd) & (ii >= 1-OLx+1) & (ii <= sNx+OLx-1)   # :508-509
        upd = jnp.broadcast_to(calc_fluxes_X & jnp.where(overlapOnly, ovl, full), shape)
        j = loop_j(1-OLy, sNy+OLy)
        i = loop_i(1-OLx+1, sNx+OLx-1)
        newTij = (localTij[i, j]                                               # :444-448, :510-514
                  - deltaT*maskInC[i, j]
                  * recip_rA[i, j]
                  * (af[i+1, j]-af[i, j]))
        localTij = localTij.at[i, j].set(jnp.where(upd[:, :, 1:-1], newTij, localTij[i, j]))
        ovlF = (S_edge & (jj <= 0)) | (N_edge & (jj >= sNy+1))                 # :486-499 (i = 1-OLx+1..sNx+OLx)
        fullF = (jj >= jMinUpd) & (jj <= jMaxUpd)                              # :530-534
        keep = jnp.broadcast_to(calc_fluxes_X & jnp.where(overlapOnly, ovlF, fullF), shape)
        i = loop_i(1-OLx+1, sNx+OLx)
        afx = afx.at[i, j].set(jnp.where(keep[:, :, 1:], af[i, j], afx[i, j]))

#--   Y direction                                                         # :554-754
        doFlux = calc_fluxes_Y & (~overlapOnly | E_edge | W_edge)               # :554, :559
        localTij = _cs_fill(2, localTij, doFlux & (overlapOnly | (ipass == 1)), cs, kc)   # :570-573
        afY = af.at[iA, jA].set(0.)                                            # :562-566
        if ppm:                                                                # :612-617
            afY = gad_ppm_adv_y(advectionScheme, k, True, deltaT, vFld, vTrans, localTij, afY, cfg=kc, grid=grid)
        else:                                                                  # :595-610
            afY = _adv_y(advectionScheme, k, deltaT, vTrans, vFld, maskLocS, localTij, afY, kc=kc, grid=grid,
                         ad_sch=ad_sch)
        af = _with_data(af, jnp.where(doFlux, afY.data, af.data))
        localTij = _cs_fill(1, localTij, calc_fluxes_Y & overlapOnly & (ipass == 1), cs, kc)   # :638-641
        jMinUpd = jnp.where(S_edge, 1, 1-OLy+1)                                # :648, :652
        jMaxUpd = jnp.where(N_edge, sNy, sNy+OLy-1)                            # :649, :653
        ovl = ((W_edge & (ii <= 0)) | (E_edge & (ii >= sNx+1))) & (jj >= jMinUpd) & (jj <= jMaxUpd)   # :655-698
        iMinUpd = jnp.where(interiorOnly & W_edge, 1, 1-OLx)                   # :717, :719
        iMaxUpd = jnp.where(interiorOnly & E_edge, sNx, sNx+OLx)               # :718, :720
        full = (ii >= iMinUpd) & (ii <= iMaxUpd) & (jj >= 1-OLy+1) & (jj <= sNy+OLy-1)   # :722-723
        upd = jnp.broadcast_to(calc_fluxes_Y & jnp.where(overlapOnly, ovl, full), shape)
        j = loop_j(1-OLy+1, sNy+OLy-1)
        i = loop_i(1-OLx, sNx+OLx)
        newTij = (localTij[i, j]                                               # :658-662, :724-728
                  - deltaT*maskInC[i, j]
                  * recip_rA[i, j]
                  * (af[i, j+1]-af[i, j]))
        localTij = localTij.at[i, j].set(jnp.where(upd[:, 1:-1, :], newTij, localTij[i, j]))
        ovlF = (W_edge & (ii <= 0)) | (E_edge & (ii >= sNx+1))                 # :700-713 (j = 1-OLy+1..sNy+OLy)
        fullF = (ii >= iMinUpd) & (ii <= iMaxUpd)                              # :744-748
        keep = jnp.broadcast_to(calc_fluxes_Y & jnp.where(overlapOnly, ovlF, fullF), shape)
        j = loop_j(1-OLy+1, sNy+OLy)
        afy = afy.at[i, j].set(jnp.where(keep[:, 1:, :], af[i, j], afy[i, j]))
    return localTij, afx, afy
