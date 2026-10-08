"""SEAICE_ADVDIFF: pkg/seaice/seaice_advdiff.F @63cdc0b (lane M4OFF: the multi-dimensional C-grid arm)."""

from mitjax.ad.approx_advection import seaice_advection_call
from mitjax.farray import loop_i, loop_j
from mitjax.pkg.seaice.seaice_advection import seaice_advection
# the tracer identities of the sea-ice fields (SEAICE_PARAMS.h:670-672; TAF tape keys, diagnostics and SEAICE_ADVECTION's
# maxpass STOP)
from mitjax.pkg.seaice.seaice_params_h import GAD_AREA, GAD_HEFF, GAD_SNOW


def seaice_advdiff(uc, vc, myTime, myIter, sf, *, cfg, sp, op, grid, ex=None):
    """SEAICE_ADVDIFF( uc, vc, myTime, myIter, myThid )   @63cdc0b pkg/seaice/seaice_advdiff.F:10-663

    C     | o driver for different advection routines
    C     |   calls an adaption of gad_advection to call different
    C     |   advection routines of pkg/generic_advdiff

    `uc`, `vc` the C-grid ice velocities (SEAICE_MODEL passes uIce, vIce), `sf` SEAICE.h (HEFF, AREA, HSNOW, HEFFM,
    SIMaskU, SIMaskV). Returns sf with HEFF, AREA, HSNOW advected (:297-375) and (ALLOW_SITRACER) SItracer. The tile loop is the leading axis.
    Ported: SEAICEmultiDimAdvection with ALLOW_GENERIC_ADVDIFF, no SEAICE_ITD (:295-376), SEAICEdiffKh* = 0 (the
    SEAICE_DIFFUSION calls :304-312, :331-339, :358-366 raise in SEAICE_READPARMS otherwise), ALLOW_SITRACER (lane
    M4LAB session 4, `_sitracer_advdiff`: SItracer and SItrBucket too), no SEAICE_VARIABLE_SALINITY (raises), no SEAICE_BGRID_DYNAMICS (the uc/vc averaging :112-127 raises); lane M4ADCS32ICE
    (global_ocean.cs32x15/code_ad): ALLOW_AUTODIFF (gFld = 0. on every point of the tile before the first advection,
    :142-149; the rest are CADJ STORE directives).
    Lane M4ADCOL (1D_ocean_ice_column/input_ad): the single-dimension arm (:565-660): ADVECT (pkg/seaice/advect.py) of
    HEFF, AREA, HSNOW (`ex` the exchanger, for ADVECT's EXCH_XY_RL), with ALLOW_AUTODIFF (its arms here are CADJ
    STORE directives only) and SEAICE_VARIABLE_SALINITY with SEAICEadvSalt = .FALSE. (:637-656 not entered);
    SEAICEdiffKh* > 0 (SEAICE_DIFFUSION) and HSALT advection raise in SEAICE_READPARMS."""
    if not sp.SEAICEmultiDimAdvection:                                         # :135 / :565 ELSE (lane M4ADCOL)
        return _advdiff_single_dim(uc, vc, sf, cfg=cfg, sp=sp, grid=grid, ex=ex)
    for o in ("SEAICE_ITD", "SEAICE_VARIABLE_SALINITY", "SEAICE_BGRID_DYNAMICS", "ALLOW_SITRACER_DEBUG_DIAG"):
        if cfg.cpp.flag(o, "SEAICE_OPTIONS.h"):
            raise NotImplementedError(f"SEAICE_ADVDIFF: {o} is not ported")
    if not (sp.SEAICEmultiDimAdvection and cfg.cpp.flag("ALLOW_GENERIC_ADVDIFF")):   # :135, :565
        raise NotImplementedError("SEAICE_ADVDIFF: SEAICEmultiDimAdvection = .FALSE. (ADVECT, :565-660) is not ported")
    # SEAICEdiffKhHeff/Area/Snow > 0 (SEAICE_DIFFUSION) raise in SEAICE_READPARMS (REAL parameters, traced here)
    sz = cfg.size
    OLx, OLy, sNx, sNy = sz.OLx, sz.OLy, sz.sNx, sz.sNy
    sf = dict(sf)
    HEFFM, SIMaskU, SIMaskV = sf["HEFFM"], sf["SIMaskU"], sf["SIMaskV"]
    jA = loop_j(1-OLy, sNy+OLy)
    iA = loop_i(1-OLx, sNx+OLx)
    xA = HEFFM.local("xA").at[iA, jA].set(grid.dyG[iA, jA]*SIMaskU[iA, jA])   # :106 (_RS)
    yA = HEFFM.local("yA").at[iA, jA].set(grid.dxG[iA, jA]*SIMaskV[iA, jA])   # :107
    sitracer = cfg.cpp.flag("ALLOW_SITRACER", "SEAICE_OPTIONS.h")              # lane M4LAB session 4
    if sitracer:                                                               # :161-169
        hEffNm1 = sf["HEFF"].local("hEffNm1").at[iA, jA].set(sf["HEFF"][iA, jA])   # :164
        areaNm1 = sf["AREA"].local("areaNm1").at[iA, jA].set(sf["AREA"][iA, jA])   # :165
    recip_heff = HEFFM.local("recip_heff").at[iA, jA].set(1.)                 # :167
    uTrans = HEFFM.local("uTrans").at[iA, jA].set(uc[iA, jA]*xA[iA, jA])      # :174
    vTrans = HEFFM.local("vTrans").at[iA, jA].set(vc[iA, jA]*yA[iA, jA])      # :175
    j = loop_j(1, sNy)
    i = loop_i(1, sNx)
    kw = dict(cfg=cfg, sp=sp, grid=grid, SIMaskU=SIMaskU, SIMaskV=SIMaskV,
              useCubedSphereExchange=op.useCubedSphereExchange)
    gFld = HEFFM.local("gFld")
    if cfg.cpp.flag("ALLOW_AUTODIFF"):                                         # :142-149 (lane M4ADCS32ICE)
        gFld = gFld.at[iA, jA].set(0.)                                         # :146  0. _d 0
    for flag, name, sch, ident in (("SEAICEadvHeff", "HEFF", "SEAICEadvSchHeff", GAD_HEFF),    # :297-321
                                   ("SEAICEadvArea", "AREA", "SEAICEadvSchArea", GAD_AREA),    # :324-348
                                   ("SEAICEadvSnow", "HSNOW", "SEAICEadvSchSnow", GAD_SNOW)):  # :351-375
        if not getattr(sp, flag):
            continue
        afx = HEFFM.local("afx")
        afy = HEFFM.local("afy")
        gFld, afx, afy = seaice_advection_call(ident, getattr(sp, sch), uc, vc, uTrans, vTrans, sf[name],
                                               recip_heff, gFld, afx, afy, myTime, myIter, **kw)   # :298-303
        sf[name] = sf[name].at[i, j].set(HEFFM[i, j] * (                       # :314-320
            sf[name][i, j] + sp.SEAICE_deltaTtherm * gFld[i, j]))
    if sitracer:                                                               # :407-548
        sf = _sitracer_advdiff(uc, vc, uTrans, vTrans, recip_heff, hEffNm1, areaNm1, myTime, myIter, sf, cfg=cfg,
                               sp=sp, kw=kw)
    return sf


def _sitracer_advdiff(uc, vc, uTrans, vTrans, recip_heff, hEffNm1, areaNm1, myTime, myIter, sf, *, cfg, sp, kw):
    """seaice_advdiff.F:407-548 (#ifdef ALLOW_SITRACER, lane M4LAB session 4): the sea-ice tracers in use whose mate
    (SItrMate 'HEFF' / 'AREA') is advected: SItrExt = HEFFM*SItracer*mate before advection (hEffNm1 / areaNm1,
    :164-165), advected with the mate's scheme (SEAICE_ADVECTION, tracer GAD_SITR+iTr-1), the explicit step, then
    the value per unit of the advected mate: 'HEFF' (:470-509): SItracer = SItrExt/HEFF where HEFF >= siEps, else 0
    with SItrExt into SItrBucket; 'AREA' (:511-540): SItracer = SItrExt/AREA where AREA >= SEAICE_area_floor, else 0,
    SItrBucket = 0. ALLOW_SITRACER_ADVCAP (lab_sea/code): the new value is capped by the largest of the five
    pre-advection values around the point (:480-490, :520-526; the excess goes to SItrBucket for 'HEFF'), then
    negative values are removed where the mate is above its floor (:492-497, :528-531). The point loops are
    independent (they read only SItrPrev and SItrExt): vectorised; the divisions are guarded on their arm
    (safe_div). SEAICE_DIFFUSION (:453-461, IF SEAICEdiffKhHeff > 0) is not ported (SEAICE_READPARMS refuses
    SEAICEdiffKh* > 0).
    ALLOW_SITRACER_DEBUG_DIAG (diagnostics only) raises in SEAICE_ADVDIFF."""
    import jax.numpy as jnp
    from mitjax.ops.fortran_minmax import MAX, MIN
    from mitjax.ops.safe import safe_div
    from mitjax.pkg.seaice.seaice_params_h import GAD_SITR, ZERO, siEps
    advcap = cfg.cpp.flag("ALLOW_SITRACER_ADVCAP", "SEAICE_OPTIONS.h")
    sz = cfg.size
    OLx, OLy, sNx, sNy = sz.OLx, sz.OLy, sz.sNx, sz.sNy
    HEFFM = sf["HEFFM"]
    jA = loop_j(1-OLy, sNy+OLy)
    iA = loop_i(1-OLx, sNx+OLx)
    j = loop_j(1, sNy)
    i = loop_i(1, sNx)
    HEFF, AREA = sf["HEFF"], sf["AREA"]
    for iTr in range(1, sp.SItrNumInUse + 1):                                  # :409
        mate = sp.SItrMate[iTr-1]
        if not ((sp.SEAICEadvHeff and mate == "HEFF") or (sp.SEAICEadvArea and mate == "AREA")):   # :410-411
            continue
        SItracer, SItrBucket = sf["SItracer"], sf["SItrBucket"]
        if mate == "HEFF":                                                     # :413-421
            SEAICEadvSchSItr = sp.SEAICEadvSchHeff                             # :414
            mateNm1 = hEffNm1                                                  # :419 hEffNm1
        else:                                                                  # :423-431
            SEAICEadvSchSItr = sp.SEAICEadvSchArea                             # :424
            mateNm1 = areaNm1                                                  # :429 areaNm1
        SItrExt = HEFFM.local("SItrExt").at[iA, jA].set(                       # :418-419, :428-429
            HEFFM[iA, jA] * SItracer[iA, jA, iTr] * mateNm1[iA, jA])
        if advcap:
            SItrPrev = HEFFM.local("SItrPrev").at[iA, jA].set(SItracer[iA, jA, iTr])   # :438
        gFld = HEFFM.local("gFld")
        afx = HEFFM.local("afx")
        afy = HEFFM.local("afy")
        gFld, afx, afy = seaice_advection(GAD_SITR+iTr-1, SEAICEadvSchSItr, uc, vc, uTrans, vTrans, SItrExt,
                                          recip_heff, gFld, afx, afy, myTime, myIter, **kw)   # :446-452
        # :453-461 IF ( SEAICEdiffKhHeff .GT. 0. ): SEAICE_DIFFUSION; SEAICEdiffKh* > 0 raise in SEAICE_READPARMS
        # (REAL parameters, traced here), so the tendency is the advective one
        ext = HEFFM[i, j] * (SItrExt[i, j] + sp.SEAICE_deltaTtherm * gFld[i, j])   # :465-466
        if mate == "HEFF":                                                     # :470-509
            thick = HEFF[i, j] >= siEps                                        # :473
            tr = jnp.where(thick, safe_div(ext, HEFF[i, j], thick), 0.)        # :474, :477
            bucket = jnp.where(thick, 0., ext)                                 # :475, :478
            if advcap:                                                         # :480-490
                tmpscal1 = MAX(SItrPrev[i, j],                                 # :483-485
                               SItrPrev[i+1, j], SItrPrev[i-1, j],
                               SItrPrev[i, j+1], SItrPrev[i, j-1], p="bbbb")
                tmpscal2 = MAX(ZERO, tr-tmpscal1, p="a")                       # :486
                tr = tr-tmpscal2                                               # :487
                bucket = (bucket                                               # :488-489
                          + tmpscal2*HEFF[i, j])
            tmpscal1 = MIN(0., tr, p="b")                                      # :493
            tr = jnp.where(thick, tr-tmpscal1, tr)                             # :494
            bucket = jnp.where(thick, bucket                                   # :495-496
                               + HEFF[i, j]*tmpscal1, bucket)
        else:                                                                  # :511-540
            covered = AREA[i, j] >= sp.SEAICE_area_floor                       # :514
            tr = jnp.where(covered, safe_div(ext, AREA[i, j], covered), 0.)    # :515, :517
            bucket = jnp.zeros_like(ext)                                       # :519
            if advcap:                                                         # :520-526
                tmpscal1 = MAX(SItrPrev[i, j],                                 # :521-523
                               SItrPrev[i+1, j], SItrPrev[i-1, j],
                               SItrPrev[i, j+1], SItrPrev[i, j-1], p="bbbb")
                tmpscal2 = MAX(ZERO, tr-tmpscal1, p="a")                       # :524
                tr = tr-tmpscal2                                               # :525
            tmpscal1 = MIN(0., tr, p="b")                                      # :529
            tr = jnp.where(covered, tr-tmpscal1, tr)                           # :530
        sf["SItracer"] = SItracer.at[i, j, iTr].set(tr)
        sf["SItrBucket"] = SItrBucket.at[i, j, iTr].set(bucket)
    return sf


def _advdiff_single_dim(uc, vc, sf, *, cfg, sp, grid, ex):
    """seaice_advdiff.F:565-660 (lane M4ADCOL): `IF ( SEAICEadvHEff ) CALL ADVECT( uc, vc, hEff, fldNm1, HEFFM )`
    (:579-583), the same for AREA (:598-602) and HSNOW (:617-621). Lane M4ADLAB session 2 (lab_sea/input_ad,
    SEAICEdiffKhArea = 200): after each ADVECT, `IF ( SEAICEdiffKh* .GT. 0. )` SEAICE_DIFFUSION( ..., diffKh,
    SEAICE_deltaTtherm, fldNm1, HEFFM, xA, yA, fld ) per tile (:584-596, :603-615, :622-634; the IF on a traced REAL
    is the routine's own `where`, seaice_diffusion.py), with xA = dyG*SIMaskU, yA = dxG*SIMaskV on every point
    (:102-111)."""
    from mitjax.pkg.seaice.advect import advect
    from mitjax.pkg.seaice.seaice_diffusion import seaice_diffusion
    for o in ("SEAICE_ITD", "SEAICE_BGRID_DYNAMICS"):                          # :567-578 STOP; :113-127
        if cfg.cpp.flag(o, "SEAICE_OPTIONS.h"):
            raise NotImplementedError(f"SEAICE_ADVDIFF: {o} is not ported")
    if sp.SEAICEadvSalt:                                                       # :637-656 (raises in READPARMS)
        raise NotImplementedError("SEAICE_ADVDIFF: SEAICEadvSalt is not ported")
    sf = dict(sf)
    sz = cfg.size
    HEFFM = sf["HEFFM"]
    jA = loop_j(1-sz.OLy, sz.sNy+sz.OLy)                                       # :104-105
    iA = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    xA = HEFFM.local("xA").at[iA, jA].set(grid.dyG[iA, jA]*sf["SIMaskU"][iA, jA])   # :106 (_RS)
    yA = HEFFM.local("yA").at[iA, jA].set(grid.dxG[iA, jA]*sf["SIMaskV"][iA, jA])   # :107
    for flag, name, kh, ident in (("SEAICEadvHeff", "HEFF", "SEAICEdiffKhHeff", GAD_HEFF),   # :579-597
                                  ("SEAICEadvArea", "AREA", "SEAICEdiffKhArea", GAD_AREA),   # :598-616
                                  ("SEAICEadvSnow", "HSNOW", "SEAICEdiffKhSnow", GAD_SNOW)):   # :617-635
        if getattr(sp, flag):
            sf[name], fldNm1 = advect(uc, vc, sf[name], HEFFM, cfg=cfg, sp=sp, grid=grid, ex=ex)
            sf[name] = seaice_diffusion(ident, getattr(sp, kh), sp.SEAICE_deltaTtherm, fldNm1, HEFFM, xA, yA,
                                        sf[name], cfg=cfg, grid=grid)          # :584-596 / :603-615 / :622-634
    return sf

