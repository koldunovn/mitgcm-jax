"""SEAICE_GROWTH: pkg/seaice/seaice_growth.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.eesupp.global_sum import _zero_plus
from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX, MIN, build_winner
from mitjax.ops.safe import safe_div, safe_sqrt
from mitjax.pkg.seaice.seaice_budget_ocean import seaice_budget_ocean
from mitjax.pkg.seaice.seaice_params_h import HALF, ONE, ZERO, nITD
from mitjax.pkg.seaice.seaice_solve4temp import seaice_solve4temp

# Plan decision 15: seaice_growth.F:1847-1848 (SEAICE_areaLossFormula 3) compile to build-dependent winners
# (mitjax/tests/test_minmax_sites.py KNOWN_DIFFER); the builds that execute them and their winners from
# $MJX_REFERENCE/minmax_sites/<build>.json (only lab_sea/input_ad and input_ad.noseaicedyn set the formula, both on
# lab_sea/code_ad). Any other measured build refuses formula 3 (build_winner); a build the oracle never compiled takes
# MINMAX_DEFAULT, the winner of most measured builds that compile the site (MAX "a" in 6 of 8, MIN "b" in 6 of 8;
# plan decision 7, audited by test_minmax_sites.py), with one warning per site.
MINMAX_BUILD = {"pkg/seaice/seaice_growth.F:1847": {"lab_sea-code_ad-63cdc0b-704fd6b": "a"},
                "pkg/seaice/seaice_growth.F:1848": {"lab_sea-code_ad-63cdc0b-704fd6b": "b"}}
MINMAX_DEFAULT = {"pkg/seaice/seaice_growth.F:1847": "a", "pkg/seaice/seaice_growth.F:1848": "b"}


def seaice_growth(myTime, myIter, sf, ff, exf, *, cfg, sp, op, grid, state, exfp, salt_plume=None, spp=None):
    """SEAICE_GROWTH( myTime, myIter, myThid )   @63cdc0b pkg/seaice/seaice_growth.F:15-2687

    C     | o Updata ice thickness and snow depth
    C     | (thermodynamic growth and melt)

    `sf` SEAICE.h / SEAICE_GRID.h (dict), `ff` FFields (Qnet, Qsw, saltFlux, EmPmR, sIceLoad, SWFrac3D), `exf`
    EXF_FIELDS.h (dict: wspeed, evap, precip, snowprecip, runoff, and SEAICE_SOLVE4TEMP's lwdown, atemp, aqh, swdown),
    `exfp` ExfParams (useRelativeWind, useAtmWind, snowprecipfile), `state` (theta, salt), `grid` (drF, yC). Returns
    (sf, ff): HEFF, HSNOW, AREA, HSALT, TICES; Qnet, Qsw, saltFlux, EmPmR, sIceLoad. With ALLOW_SALT_PLUME (lane M4LAB
    session 3, lab_sea/code): `salt_plume` SALT_PLUME.h (dict: saltPlumeFlux, SaltPlumeDepth) and `spp`
    SaltPlumeParams (SPsalFRAC, SaltPlumeSouthernOcean; pkg/salt_plume/salt_plume_readparms.py) are required and
    (sf, ff, salt_plume) is returned. ALLOW_SITRACER (lane M4LAB session 4): the SEAICE_TRACER.h snapshots of HEFF
    (SItrHEFF levels 1-5: :520-522, :1335-1337 (inside .NOT.SEAICE_growMeltByConv), :1448-1450, :1676-1688,
    :2148-2155) and AREA (SItrAREA levels 2, 3: :522, :1861-1863), interior; the diagnostics fill :571-584 is output
    only. saltPlumeFlux of :2008-2045 without SALT_PLUME_SPLIT_BASIN (localSPfrac =
    SPsalFRAC, :2020) and without SALT_PLUME_IN_LEADS (both raise), interior only, as the Fortran.
    SEAICE_multDim 1..nITD (lane M4LAB session 3: lab_sea's 7 categories; the non-ITD category loops :776-801,
    :819-831, :849-888 are static Python loops over IT, SEAICE_PDF static).
    Ported for the builds 1D_ocean_ice_column/code and (lane M4OFF) offline_exf_seaice/code: ALLOW_EXF +
    ALLOW_ATM_TEMP, no SEAICE_ITD, ALLOW_RUNOFF, no SEAICE_CAP_SUBLIM / SEAICE_CAP_ICELOAD /
    ALLOW_BALANCE_FLUXES with its switches on (lane M4CS32ICE: off is no-op) /
    SEAICE_MODIFY_GROWTH_ADJ (each raises when compiled); EXF_SEAICE_FRACTION (lane M4ADLAB session 3: d_HEFFbyRLX in
    the sums :1999-2001, :2214-2216, :2375-2377); lane M4ADCOL: ALLOW_AUTODIFF (no value changes, below) and
    SEAICE_EXCLUDE_FOR_EXACT_AD_TESTING (:1522-1570 not compiled); by the build's options: SHORTWAVE_HEATING
    (:343-347, :1606; else :1608), SEAICE_DISABLE_SUBLIM (:967), SEAICE_DISABLE_HEATCONSFIX (else :2239-2300),
    SEAICE_VARIABLE_SALINITY (:2057-2145; else :1979-2048 with SEAICE_salt0), ALLOW_DIAGNOSTICS (the d_AREAby*
    locals :1864-1875, output only: computed and not returned; the DIAGNOSTICS_FILL calls are not ported).
    Run-time: SEAICE_multDim (the IT loops are static Python loops), SEAICE_growMeltByConv (static: :1299-1342
    and :1528-1568 skipped), SEAICE_areaGainFormula 1 or 2 (:1820-1826) and SEAICE_areaLossFormula 1, 2 or 3
    (:1833-1849; lane M4ADLAB session 3: formula 3's MAX/MIN winners at :1847-1848 are build-dependent, plan decision
    15: the executing build's winners from MINMAX_BUILD, any other measured build refuses, a build the oracle never
    compiled takes MINMAX_DEFAULT with a warning, plan decision 7), useRelativeWind = .FALSE. (raises),
    SEAICEheatConsFix / useRealFreshWaterFlux / nonlinFreeSurf (static).
    The (i,j) loops are independent per point: vectorised over the interior (1:sNx,1:sNy); the locals are interior
    arrays [tile, j, i]. Pointwise IFs are wheres with both arms finite: :651 (HEFFpreTH > 0; the regularised
    divisions and SQRTs guarded on that mask, safe_div/safe_sqrt), :1030, :1036-1045, :1482, :1807, :1852, :2065
    (tmpscal1 >= 0; the division by tmpscal2 guarded on the other arm), :2345-2358 (static flags). SIatmQnt, SItflux,
    SIatmFW (:2311-2361, :2396-2403) are locals only ALLOW_DIAGNOSTICS reads (not compiled): computed as the Fortran
    does and not returned. REAL*4 literals `0.0` exact; `0.2 _d 0`, `2.0 _d 0`, `1.0 _d 0` double."""
    for o in ("SEAICE_ITD", "SEAICE_CAP_SUBLIM", "SEAICE_CAP_ICELOAD", "SEAICE_MODIFY_GROWTH_ADJ",
              "SEAICE_USE_GROWTH_ADX", "SEAICE_GREASE"):
        if cfg.cpp.flag(o, "SEAICE_OPTIONS.h"):
            raise NotImplementedError(f"SEAICE_GROWTH: {o} is not ported")
    disable_sublim = cfg.cpp.flag("SEAICE_DISABLE_SUBLIM", "SEAICE_OPTIONS.h")
    heatconsfix = not cfg.cpp.flag("SEAICE_DISABLE_HEATCONSFIX", "SEAICE_OPTIONS.h")
    variable_salinity = cfg.cpp.flag("SEAICE_VARIABLE_SALINITY", "SEAICE_OPTIONS.h")
    shortwave_heating = cfg.cpp.flag("SHORTWAVE_HEATING")
    allow_diagnostics = cfg.cpp.flag("ALLOW_DIAGNOSTICS")
    # lane M4ADCOL (1D_ocean_ice_column/code_ad): ALLOW_AUTODIFF alone changes no value here: its live lines are the
    # AUTODIFF_OPTIONS.h include (:8-9) and, with ALLOW_AUTODIFF_TAMC, tamc.h (:45) and the tape key tkey (:168,
    # :413; measured: dev job 27880281's live-line diff code vs code_ad); every other ALLOW_AUTODIFF arm
    # (:588, :673, :988, :1111, :1927) also needs SEAICE_MODIFY_GROWTH_ADJ, which raises above
    exclude_exact_ad = cfg.cpp.flag("SEAICE_EXCLUDE_FOR_EXACT_AD_TESTING", "SEAICE_OPTIONS.h")
    for o, want in (("ALLOW_EXF", True),):
        if cfg.cpp.flag(o) != want:
            raise NotImplementedError(f"SEAICE_GROWTH: {o} {'undefined' if want else 'defined'} is not ported")
    if cfg.cpp.flag("ALLOW_BALANCE_FLUXES") and (op.selectBalanceEmPmR == 1 or op.balanceQnet):
        # lane M4CS32ICE: compiled with selectBalanceEmPmR /= 1 and balanceQnet = .FALSE. (global_ocean.cs32x15/
        # input.seaice: 0 and .FALSE.), the tile integrals :2440-2475 write zeros to locals and the global sums,
        # means and adjustments :2572-2664 are not taken (only selectBalanceEmPmR.EQ.1 / balanceQnet arms write
        # empmr, SIatmFW, SItflux, qnet, SIatmQnt): nothing of SEAICE.h / FFIELDS.h changes, so they are not
        # carried. The balancing arms raise.
        raise NotImplementedError("SEAICE_GROWTH: ALLOW_BALANCE_FLUXES with selectBalanceEmPmR = 1 or balanceQnet "
                                  "(:2440-2475, :2572-2664) is not ported")
    allow_salt_plume = cfg.cpp.flag("ALLOW_SALT_PLUME", "SEAICE_OPTIONS.h")    # :5-6, :41-42, :2008
    if allow_salt_plume:
        if salt_plume is None or spp is None:
            raise ValueError("SEAICE_GROWTH: ALLOW_SALT_PLUME needs SALT_PLUME.h `salt_plume` and `spp`")
        for o in ("SALT_PLUME_SPLIT_BASIN", "SALT_PLUME_IN_LEADS"):
            if cfg.cpp.flag(o, "SALT_PLUME_OPTIONS.h"):
                raise NotImplementedError(f"SEAICE_GROWTH: {o} (:2009-2018, :2022-2035) is not ported")
    elif salt_plume is not None:
        raise ValueError("SEAICE_GROWTH: `salt_plume` given without ALLOW_SALT_PLUME")
    # EXF_SEAICE_FRACTION (lane M4ADLAB session 3): d_HEFFbyRLX of SEAICE_REG_RIDGE (SEAICE.h:219-225) in the sums
    # :1999-2001, :2214-2216, :2375-2377
    seaice_fraction = cfg.cpp.flag("EXF_SEAICE_FRACTION", "EXF_OPTIONS.h")
    for o, want in (("ALLOW_ATM_TEMP", True), ("ALLOW_RUNOFF", True)):
        if cfg.cpp.flag(o, "EXF_OPTIONS.h") != want:
            raise NotImplementedError(f"SEAICE_GROWTH: {o} {'undefined' if want else 'defined'} is not ported")
    if not 1 <= sp.SEAICE_multDim <= nITD(cfg):
        raise ValueError("SEAICE_GROWTH: SEAICE_multDim outside 1..nITD")
    if sp.SEAICE_areaGainFormula not in (1, 2) or sp.SEAICE_areaLossFormula not in (1, 2, 3):
        raise NotImplementedError("SEAICE_GROWTH: areaGainFormula not in 1..2 / areaLossFormula not in 1..3 "
                                  "(seaice_check.F:95-107 stops)")
    if sp.SEAICE_areaLossFormula == 3:                                         # plan decision 15
        w = tuple(build_winner(cfg, MINMAX_BUILD, f"pkg/seaice/seaice_growth.F:{n}", MINMAX_DEFAULT)
                  for n in (1847, 1848))
        if w != ("a", "b"):
            raise NotImplementedError(f"SEAICE_GROWTH: seaice_growth.F:1847-1848 winners {w} of this build are not "
                                      "the ported ('a', 'b') (plan decision 15)")
    sz = cfg.size
    sf = dict(sf)
    # :335-394
    if op.usingPCoords:                                                        # :335-341
        raise NotImplementedError("SEAICE_GROWTH: usingPCoords is not ported")
    kSurface = 1                                                               # :339
    dzSurf = grid.drF.data[kSurface-1]                                         # :340
    if shortwave_heating:                                                      # :342-348
        kSrfS = 2                                                              # :346 (usingPCoords raised above)
    recip_multDim = jnp.float64(sp.SEAICE_multDim)                            # :351
    recip_multDim = ONE / recip_multDim                                        # :352
    recip_deltaTtherm = ONE / sp.SEAICE_deltaTtherm                            # :355
    recip_rhoIce = ONE / sp.SEAICE_rhoIce                                      # :356
    heffTooHeavy = dzSurf * 0.2                                                # :359 (read with SEAICE_CAP_ICELOAD)
    del heffTooHeavy
    ICE2SNOW = sp.SEAICE_rhoIce/sp.SEAICE_rhoSnow                              # :361
    SNOW2ICE = ONE / ICE2SNOW                                                  # :362
    QI = sp.SEAICE_rhoIce*sp.SEAICE_lhFusion                                   # :365
    recip_QI = ONE / QI                                                        # :366
    area_reg_sq = sp.SEAICE_area_reg * sp.SEAICE_area_reg                      # :374
    hice_reg_sq = sp.SEAICE_hice_reg * sp.SEAICE_hice_reg                      # :375
    convertQ2HI = sp.SEAICE_deltaTtherm/QI                                     # :378
    convertHI2Q = ONE/convertQ2HI                                              # :379
    convertPRECIP2HI = sp.SEAICE_deltaTtherm*op.rhoConstFresh/sp.SEAICE_rhoIce   # :381
    convertHI2PRECIP = ONE/convertPRECIP2HI                                    # :382
    denominator = 0.0                                                          # :385
    for IT in range(1, sp.SEAICE_multDim + 1):                                 # :386-388
        denominator = denominator + IT * sp.SEAICE_PDF[IT-1]
    denominator = (2.0 * denominator) - 1.0                                    # :389
    recip_denominator = 1. / denominator                                       # :390
    areaPDFfac = denominator * recip_multDim                                   # :394
    j = loop_j(1, sz.sNy)
    i = loop_i(1, sz.sNx)
    HEFF, HSNOW, AREA, HEFFM = sf["HEFF"], sf["HSNOW"], sf["AREA"], sf["HEFFM"]
    zero = jnp.zeros_like(HEFF[i, j])
    # :430-500 locals
    a_QbyATM_cover = zero                                                      # :432
    a_QbyATM_open = zero                                                       # :433
    r_QbyATM_cover = zero                                                      # :434
    r_QbyATM_open = zero                                                       # :435
    a_QSWbyATM_open = zero                                                     # :437
    a_QSWbyATM_cover = zero                                                    # :438
    a_QbyOCN = zero                                                            # :440
    r_QbyOCN = zero                                                            # :441
    d_HEFFbyOCNonICE = zero                                                    # :449
    d_HEFFbyATMonOCN = zero                                                    # :450
    d_HEFFbyFLOODING = zero                                                    # :451
    d_HEFFbyATMonOCN_open = zero                                               # :453
    d_HEFFbyATMonOCN_cover = zero                                              # :454
    d_HSNWbyATMonSNW = zero                                                    # :456
    d_HSNWbyOCNonSNW = zero                                                    # :457
    d_HSNWbyRAIN = zero                                                        # :458
    a_FWbySublim = zero                                                        # :459
    r_FWbySublim = zero                                                        # :460
    d_HEFFbySublim = zero                                                      # :461
    d_HSNWbySublim = zero                                                      # :462
    d_HFRWbyRAIN = zero                                                        # :466
    tmparr1 = zero                                                             # :467
    del tmparr1, a_QbyOCN, d_HSNWbySublim
    if allow_diagnostics:                                                      # :443-447
        d_AREAbyATM = zero                                                     # :444
        d_AREAbyICE = zero                                                     # :445
        d_AREAbyOCN = zero                                                     # :446
        del d_AREAbyATM, d_AREAbyICE, d_AREAbyOCN
    IT_ = range(1, sp.SEAICE_multDim + 1)
    ticeInMult = {IT: zero for IT in IT_}                                      # :473
    ticeOutMult = {IT: zero for IT in IT_}                                     # :474
    a_QbyATMmult_cover = {IT: zero for IT in IT_}                              # :475
    a_QSWbyATMmult_cover = {IT: zero for IT in IT_}                            # :476
    a_FWbySublimMult = {IT: zero for IT in IT_}                                # :477
    # :510-525
    HEFFpreTH = HEFF[i, j]                                                     # :512
    HSNWpreTH = HSNOW[i, j]                                                    # :513
    AREApreTH = AREA[i, j]                                                     # :514
    sitracer = cfg.cpp.flag("ALLOW_SITRACER", "SEAICE_OPTIONS.h")              # lane M4LAB session 4
    if sitracer:                                                               # :520-523
        SItrHEFF = {1: HEFF[i, j]}                                             # :521 SItrHEFF(i,j,bi,bj,1)=HEFF
        SItrAREA = {2: AREA[i, j]}                                             # :522 SItrAREA(i,j,bi,bj,2)=AREA
    # :649-670 effective thickness (regularised)
    ice = HEFFpreTH > ZERO                                                     # :651
    tmpscal1 = safe_sqrt(AREApreTH*AREApreTH + area_reg_sq, ice)               # :653
    tmpscal2 = safe_div(HEFFpreTH, tmpscal1, ice)                              # :655
    heffActual = jnp.where(ice, safe_sqrt(tmpscal2 * tmpscal2 + hice_reg_sq, ice), ZERO)   # :657, :665
    hsnowActual = jnp.where(ice, safe_div(HSNWpreTH, tmpscal1, ice), ZERO)     # :659, :666
    recip_heffActual = jnp.where(ice, safe_div(                                # :661-662, :667
        AREApreTH, safe_sqrt(HEFFpreTH*HEFFpreTH + hice_reg_sq, ice), ice), ZERO)
    # :722-729
    TmixLoc = state.theta[i, j, kSurface]+op.celsius2K                         # :725
    UG = MAX(sp.SEAICE_EPS, exf["wspeed"][i, j], p="b")                        # :727
    a_QbyATM_open, a_QSWbyATM_open = seaice_budget_ocean(                      # :738-742
        UG, TmixLoc, myTime, myIter, cfg=cfg, ff=ff)
    if exfp.useRelativeWind and exfp.useAtmWind:                               # :747-763
        raise NotImplementedError("SEAICE_GROWTH: useRelativeWind (:747-763) is not ported")
    # :776-801
    TICES = sf["TICES"]
    heffActualMult, hsnowActualMult = {}, {}
    for IT in IT_:
        ticeInMult[IT] = TICES[i, j, IT]                                       # :779
        ticeOutMult[IT] = TICES[i, j, IT]                                      # :780
        TICES = TICES.at[i, j, IT].set(ZERO)                                   # :781
        pFac = (2.0*IT - 1.0)*recip_denominator                                # :788
        pFacSnow = 1.                                                          # :789
        if sp.SEAICE_useMultDimSnow:                                           # :790
            pFacSnow = pFac
        heffActualMult[IT] = heffActual*pFac                                   # :793
        hsnowActualMult[IT] = hsnowActual*pFacSnow                             # :794
    # :819-831
    for IT in IT_:
        ticeOutMult[IT], a_QbyATMmult_cover[IT], a_QSWbyATMmult_cover[IT], a_FWbySublimMult[IT] = \
            seaice_solve4temp(UG, heffActualMult[IT], hsnowActualMult[IT], ticeInMult[IT], myTime, myIter,
                              cfg=cfg, sp=sp, op=op, exf=exf, grid=grid, state=state)
    # :849-888
    # the first addition to the zeroed sums (:432, :438, :459) is `0. + x` (_zero_plus: XLA folds an added constant 0
    # away, which keeps a -0.; measured: Qsw -0. for the Fortran's +0. on 8/5/4 ice-covered points, dev 27873180)
    acc = lambda a, x, IT: _zero_plus(x) if IT == 1 else a + x                 # noqa: E731
    for IT in IT_:
        TICES = TICES.at[i, j, IT].set(ticeOutMult[IT])                        # :867
        a_QbyATM_cover = acc(a_QbyATM_cover,                                   # :879-880
                             a_QbyATMmult_cover[IT]*sp.SEAICE_PDF[IT-1], IT)
        a_QSWbyATM_cover = acc(a_QSWbyATM_cover,                               # :881-882
                               a_QSWbyATMmult_cover[IT]*sp.SEAICE_PDF[IT-1], IT)
        a_FWbySublim = acc(a_FWbySublim,                                       # :883-884
                           a_FWbySublimMult[IT]*sp.SEAICE_PDF[IT-1], IT)
    sf["TICES"] = TICES
    # :950-973
    a_QbyATM_cover = (a_QbyATM_cover                                           # :952-953
                      * convertQ2HI * AREApreTH)
    a_QSWbyATM_cover = (a_QSWbyATM_cover                                       # :954-955
                        * convertQ2HI * AREApreTH)
    a_QbyATM_open = (a_QbyATM_open                                             # :956-957
                     * convertQ2HI * (ONE - AREApreTH))
    a_QSWbyATM_open = (a_QSWbyATM_open                                         # :958-959
                       * convertQ2HI * (ONE - AREApreTH))
    r_QbyATM_cover = a_QbyATM_cover                                            # :961
    r_QbyATM_open = a_QbyATM_open                                              # :962
    if disable_sublim:                                                         # :965-968
        a_FWbySublim = jnp.full_like(zero, ZERO)                               # :967
    a_FWbySublim = (sp.SEAICE_deltaTtherm*recip_rhoIce                         # :969-970
                    * a_FWbySublim*AREApreTH)
    r_FWbySublim = a_FWbySublim                                                # :971
    # :1024-1055 heat flux from the ocean
    theta1 = state.theta[i, j, kSurface]
    tempFrz = (sp.SEAICE_tempFrz0                                              # :1027-1028
               + sp.SEAICE_dTempFrz_dS*state.salt[i, j, kSurface])
    tmpscal1 = jnp.where(theta1 >= tempFrz, sp.SEAICE_mcPheePiston,            # :1030-1034
                         sp.SEAICE_frazilFrac*dzSurf/sp.SEAICE_deltaTtherm)
    if not sp.SEAICE_mcPheeStepFunc:                                           # :1036-1045
        MixedLayerTurbulenceFactor = jnp.where(AREApreTH > 0., ONE - sp.SEAICE_mcPheeTaper * AREApreTH, ONE)
    else:
        MixedLayerTurbulenceFactor = jnp.where(AREApreTH > 0., ONE - sp.SEAICE_mcPheeTaper, ONE)
    tmpscal2 = (- (op.HeatCapacity_Cp*op.rhoConst * recip_QI)                  # :1047-1049
                * (theta1-tempFrz)
                * sp.SEAICE_deltaTtherm * HEFFM[i, j])
    a_QbyOCN = (tmpscal1 * tmpscal2 * MixedLayerTurbulenceFactor)              # :1051-1052
    r_QbyOCN = a_QbyOCN                                                        # :1053
    # :1228-1285 sublimation
    HSNOW_ = HSNOW[i, j]
    tmpscal2 = MAX(MIN(r_FWbySublim, HSNOW_*SNOW2ICE, p="b"), ZERO, p="a")    # :1231-1241
    d_HSNWbySublim = - tmpscal2 * ICE2SNOW                                     # :1242
    HSNOW_ = HSNOW_ - tmpscal2*ICE2SNOW                                        # :1243
    r_FWbySublim = r_FWbySublim - tmpscal2                                     # :1244
    HEFF_ = HEFF[i, j]
    tmpscal2 = MAX(MIN(r_FWbySublim, HEFF_, p="a"), ZERO, p="a")               # :1255-1263
    d_HEFFbySublim = - tmpscal2                                                # :1264
    HEFF_ = HEFF_ - tmpscal2                                                   # :1265
    r_FWbySublim = r_FWbySublim - tmpscal2                                     # :1266
    del d_HSNWbySublim
    a_QbyATM_cover = a_QbyATM_cover-r_FWbySublim                               # :1281
    r_QbyATM_cover = r_QbyATM_cover-r_FWbySublim                               # :1282
    # :1299-1342 ocean heat on ice
    if not sp.SEAICE_growMeltByConv:                                           # :1299
        d_HEFFbyOCNonICE = MAX(r_QbyOCN, -HEFF_, p="a")                        # :1332
        r_QbyOCN = r_QbyOCN-d_HEFFbyOCNonICE                                   # :1333
        HEFF_ = HEFF_ + d_HEFFbyOCNonICE                                       # :1334
        if sitracer:
            SItrHEFF[2] = HEFF_                                                # :1335-1337
    # :1374-1388 atmosphere on snow
    tmpscal1 = MAX(r_QbyATM_cover, -HSNOW_*SNOW2ICE, p="a")                    # :1378
    tmpscal2 = MIN(tmpscal1, 0., p="a")                                        # :1379
    d_HSNWbyATMonSNW = tmpscal2*ICE2SNOW                                       # :1384
    HSNOW_ = HSNOW_ + tmpscal2*ICE2SNOW                                        # :1385
    r_QbyATM_cover = r_QbyATM_cover - tmpscal2                                 # :1386
    # :1436-1452 atmosphere on ice (covered part)
    tmpscal2 = MAX(-HEFF_, r_QbyATM_cover                                      # :1439-1441
                   + AREApreTH * r_QbyOCN, p="b")
    d_HEFFbyATMonOCN_cover = tmpscal2                                          # :1443
    d_HEFFbyATMonOCN = _zero_plus(tmpscal2)                                    # :1444 (zero + tmpscal2: _zero_plus)
    r_QbyATM_cover = r_QbyATM_cover-tmpscal2                                   # :1445
    HEFF_ = HEFF_ + tmpscal2                                                   # :1446
    if sitracer:
        SItrHEFF[3] = HEFF_                                                    # :1448-1450
    # :1462-1515 precipitation
    if exfp.snowprecipfile.strip():                                            # :1462-1473
        raise NotImplementedError("SEAICE_GROWTH: snowPrecipFile (:1463-1473) is not ported")
    PRECIP = exf["precip"][i, j]
    covered = a_QbyATM_cover >= 0.                                             # :1482
    d_HFRWbyRAIN = jnp.where(covered, 0.,                                      # :1484, :1489-1490
                             -convertPRECIP2HI*PRECIP*AREApreTH)
    d_HSNWbyRAIN = jnp.where(covered, convertPRECIP2HI*ICE2SNOW               # :1485-1486, :1491
                             * PRECIP*AREApreTH, 0.)
    HSNOW_ = HSNOW_ + d_HSNWbyRAIN                                             # :1507
    # :1528-1568 ocean heat on snow (#ifndef SEAICE_EXCLUDE_FOR_EXACT_AD_TESTING, :1522-1570: lane M4ADCOL,
    # 1D_ocean_ice_column/code_ad defines it and the block is not compiled; d_HSNWbyOCNonSNW keeps its zero of :457)
    if not exclude_exact_ad and not sp.SEAICE_growMeltByConv:                  # :1528
        tmpscal1 = MAX(r_QbyOCN*ICE2SNOW, -HSNOW_, p="a")                      # :1554
        tmpscal2 = MIN(tmpscal1, 0., p="a")                                    # :1555
        d_HSNWbyOCNonSNW = tmpscal2                                            # :1560
        r_QbyOCN = (r_QbyOCN                                                   # :1561-1562
                    - d_HSNWbyOCNonSNW*SNOW2ICE)
        HSNOW_ = HSNOW_+d_HSNWbyOCNonSNW                                       # :1563
    # :1583-1674 open-water growth
    tmpscal4 = HEFF_                                                           # :1594
    tmpscal1 = r_QbyATM_open+r_QbyOCN * (1.0 - AREApreTH)                      # :1598-1599
    if shortwave_heating:                                                      # :1602-1609
        tmpscal2 = ff.SWFrac3D[i, j, kSrfS] * a_QSWbyATM_open                  # :1606
    else:
        tmpscal2 = 0.                                                          # :1608
    tmpscal3 = sp.facOpenGrow*MAX(tmpscal1-tmpscal2,                           # :1612-1613
                                  -tmpscal4*sp.facOpenMelt, p="b")*HEFFM[i, j]
    d_HEFFbyATMonOCN_open = tmpscal3                                           # :1669
    d_HEFFbyATMonOCN = d_HEFFbyATMonOCN+tmpscal3                               # :1670
    r_QbyATM_open = r_QbyATM_open-tmpscal3                                     # :1671
    HEFF_ = HEFF_ + tmpscal3                                                   # :1672
    if sitracer:
        SItrHEFF[4] = HEFF_                                                    # :1676-1688 (ndef SEAICE_ITD :1684)
    # :1698-1733 flooding
    if sp.SEAICEuseFlooding:
        tmpscal0 = (HSNOW_*sp.SEAICE_rhoSnow                                   # :1723-1724
                    + HEFF_*sp.SEAICE_rhoIce)*op.recip_rhoConst
        tmpscal1 = MAX(0., tmpscal0 - HEFF_, p="a")                            # :1725
        d_HEFFbyFLOODING = tmpscal1                                            # :1726
        HEFF_ = HEFF_+d_HEFFbyFLOODING                                         # :1727
        HSNOW_ = HSNOW_ - d_HEFFbyFLOODING*ICE2SNOW                            # :1728-1729
    # :1798-1877 ice cover
    recip_HO = jnp.where(grid.yC[i, j] < ZERO, 1. / sp.HO_south, 1. / sp.HO)  # :1807-1811
    recip_HH = recip_heffActual                                                # :1813
    if sp.SEAICE_areaGainFormula == 1:                                         # :1822
        tmpscal4 = MAX(ZERO, d_HEFFbyATMonOCN_open, p="b")                     # :1823
    else:
        tmpscal4 = MAX(ZERO, a_QbyATM_open, p="b")                             # :1825
    if sp.SEAICE_areaLossFormula == 1:                                         # :1833
        tmpscal3 = (MIN(0., d_HEFFbyATMonOCN_cover, p="a")                     # :1834-1836
                    + MIN(0., d_HEFFbyATMonOCN_open, p="b")
                    + MIN(0., d_HEFFbyOCNonICE, p="b"))
    elif sp.SEAICE_areaLossFormula == 2:                                       # :1837
        tmpscal3 = MIN(0., d_HEFFbyATMonOCN_cover                              # :1838-1839
                       + d_HEFFbyATMonOCN_open + d_HEFFbyOCNonICE, p="b")
    else:                                                                      # :1840 (formula 3)
        tmpscal0 = HEFF_ - d_HEFFbyATMonOCN                                    # :1842
        tmpscal1 = (a_QbyATM_open+a_QbyATM_cover                               # :1844-1845
                    - d_HSNWbyATMonSNW*SNOW2ICE)
        tmpscal2 = MAX(-tmpscal0, tmpscal1, p="a")       # :1847; MINMAX-BUILD: lab_sea-code_ad-63cdc0b-704fd6b
        tmpscal3 = MIN(ZERO, tmpscal2, p="b")            # :1848; MINMAX-BUILD: lab_sea-code_ad-63cdc0b-704fd6b
    AREA_ = jnp.where((HEFF_ > 0.) | (HSNOW_ > 0.),                            # :1852-1860
                      MAX(0., MIN(sp.SEAICE_area_max, AREA[i, j]                # :1854-1857
                                  + recip_HO*tmpscal4+HALF*recip_HH*tmpscal3
                                  * areaPDFfac, p="b"), p="b"),
                      0.)
    if sitracer:
        SItrAREA[3] = AREA_                                                    # :1861-1863
    if allow_diagnostics:                                                      # :1864-1875 (output only)
        d_AREAbyATM = (recip_HO*MAX(ZERO, d_HEFFbyATMonOCN_open, p="b")        # :1865-1868
                       + HALF*recip_HH*MIN(0., d_HEFFbyATMonOCN_open, p="a")
                       * areaPDFfac)
        d_AREAbyICE = (HALF*recip_HH*MIN(0., d_HEFFbyATMonOCN_cover, p="a")    # :1869-1871
                       * areaPDFfac)
        d_AREAbyOCN = (HALF*recip_HH*MIN(0., d_HEFFbyOCNonICE, p="a")          # :1872-1874
                       * areaPDFfac)
        del d_AREAbyATM, d_AREAbyICE, d_AREAbyOCN
    salt1 = state.salt[i, j, kSurface]
    if not variable_salinity:                                                  # :1979-2048
        tmpscal1 = (sf["d_HEFFbyNEG"][i, j] + d_HEFFbyOCNonICE                 # :1996-1998
                    + d_HEFFbyATMonOCN + d_HEFFbyFLOODING
                    + d_HEFFbySublim)
        if seaice_fraction:
            tmpscal1 = tmpscal1 + sf["d_HEFFbyRLX"][i, j]                      # :1999-2001
        tmpscal3 = MAX(0., MIN(sp.SEAICE_salt0, salt1, p="a"), p="b")          # :2003-2004
        tmpscal2 = (tmpscal1 * tmpscal3 * HEFFM[i, j]                          # :2005-2006
                    * recip_deltaTtherm * sp.SEAICE_rhoIce)
        saltFlux = tmpscal2                                                    # :2007
        if allow_salt_plume:                                                   # :2008-2045
            localSPfrac = spp.SPsalFRAC                                        # :2020 (ndef SALT_PLUME_SPLIT_BASIN)
            tmpscal3 = (tmpscal1*salt1*HEFFM[i, j]                             # :2037-2038
                        * recip_deltaTtherm * sp.SEAICE_rhoIce)
            saltPlumeFlux = (MAX(tmpscal3-tmpscal2, 0., p="b")                 # :2039-2040
                             * localSPfrac)
            if not spp.SaltPlumeSouthernOcean:                                 # :2042-2045
                saltPlumeFlux = jnp.where(grid.yC[i, j] < 0.0, 0.0, saltPlumeFlux)   # :2043-2044
    else:                                                                      # :2057-2145
        HSALT = sf["HSALT"][i, j]
        tmpscal1 = d_HEFFbyOCNonICE+d_HEFFbyATMonOCN                           # :2060
        tmpscal2 = HEFF_-tmpscal1-d_HEFFbyFLOODING                             # :2063
        grow = tmpscal1 >= 0.0                                                 # :2065
        saltFlux = jnp.where(
            grow,
            HEFFM[i, j]*recip_deltaTtherm                                      # :2066-2069
            * sp.SEAICE_saltFrac*salt1
            * tmpscal1*sp.SEAICE_rhoIce,
            safe_div(HEFFM[i, j]*recip_deltaTtherm                             # :2096-2099
                     * HSALT
                     * tmpscal1, tmpscal2, ~grow))
        if allow_salt_plume:                    # :2070-2091, :2100-2104, :2133-2138 (lane M4ADLAB session 2)
            localSPfrac = spp.SPsalFRAC                                        # :2082 (ndef SALT_PLUME_SPLIT_BASIN)
            saltPlumeFlux = jnp.where(                                         # (ndef SALT_PLUME_IN_LEADS)
                grow,
                HEFFM[i, j]*recip_deltaTtherm                                  # :2087-2091
                * (ONE-sp.SEAICE_saltFrac)*salt1
                * tmpscal1*sp.SEAICE_rhoIce
                * localSPfrac,
                0.0)                                                           # :2102 0.0 _d 0
            if not spp.SaltPlumeSouthernOcean:                                 # :2134-2137
                saltPlumeFlux = jnp.where(grid.yC[i, j] < 0.0, 0.0, saltPlumeFlux)   # :2135-2136
        HSALT = (HSALT                                                         # :2140-2141
                 + saltFlux * sp.SEAICE_deltaTtherm)
        saltFlux = (saltFlux                                                   # :2142-2143
                    + sf["saltFluxAdjust"][i, j])
    if sitracer:
        SItrHEFF[5] = HEFF_                                                    # :2148-2155
    # :2207-2237 heat fluxes to the ocean
    d_HEFFbyNEG = sf["d_HEFFbyNEG"][i, j]
    d_HSNWbyNEG = sf["d_HSNWbyNEG"][i, j]
    snowPrecip = exf["snowprecip"][i, j]
    sumNEG = d_HEFFbyOCNonICE + d_HSNWbyOCNonSNW*SNOW2ICE + d_HEFFbyNEG        # :2212-2213
    if seaice_fraction:
        sumNEG = sumNEG + sf["d_HEFFbyRLX"][i, j]                              # :2214-2216
    QNET = (r_QbyATM_cover + r_QbyATM_open                                     # :2209-2220
            + a_QSWbyATM_cover
            - (sumNEG
               + d_HSNWbyNEG*SNOW2ICE
               - convertPRECIP2HI
               * snowPrecip * (ONE-AREApreTH)
               ) * HEFFM[i, j])
    QSW = a_QSWbyATM_cover + a_QSWbyATM_open                                   # :2225
    QNET = QNET*convertHI2Q                                                    # :2234
    QSW = QSW*convertHI2Q                                                      # :2235
    if heatconsfix:                                                            # :2239-2300
        tmpscal1 = jnp.full_like(zero, ZERO)                                   # :2267
        tmpscal3 = (op.rhoConstFresh*HEFFM[i, j]*(                             # :2269-2275
            (d_HSNWbyATMonSNW*SNOW2ICE
             + d_HSNWbyOCNonSNW*SNOW2ICE
             + d_HEFFbyOCNonICE + d_HEFFbyATMonOCN
             + d_HEFFbyNEG + d_HSNWbyNEG*SNOW2ICE)
            * convertHI2PRECIP
            - snowPrecip * (ONE-AREApreTH)))
        if op.temp_EvPrRn_set and op.useRealFreshWaterFlux and op.nonlinFreeSurf != 0:   # :2277-2278
            tmpscal1 = (- tmpscal3                                             # :2279-2280
                        * op.HeatCapacity_Cp * op.temp_EvPrRn)
        elif (not op.temp_EvPrRn_set) and op.useRealFreshWaterFlux and op.nonlinFreeSurf != 0:   # :2281-2282
            tmpscal1 = (- tmpscal3                                             # :2283-2284
                        * op.HeatCapacity_Cp * state.theta[i, j, kSurface])
        elif op.temp_EvPrRn_set:                                               # :2285
            tmpscal1 = (- tmpscal3*op.HeatCapacity_Cp                          # :2286-2287
                        * (op.temp_EvPrRn - state.theta[i, j, kSurface]))
        else:                                                                  # :2288-2289
            tmpscal1 = jnp.full_like(zero, ZERO)
        if op.useRealFreshWaterFlux and op.nonlinFreeSurf > 0 and sp.SEAICEheatConsFix:   # :2296-2297
            QNET = QNET+tmpscal1                                               # :2298
    # :2311-2361 diagnostics locals (SIatmQnt, SItflux)
    EVAP, RUNOFF = exf["evap"][i, j], exf["runoff"][i, j]
    SIatmQnt = (HEFFM[i, j]*convertHI2Q*(                                      # :2321-2324
        a_QSWbyATM_cover
        + a_QbyATM_cover + a_QbyATM_open))
    tmpscal1 = (op.rhoConstFresh*HEFFM[i, j]                                   # :2328-2330
                * convertHI2PRECIP * (- d_HSNWbyRAIN*SNOW2ICE
                                      + a_FWbySublim - r_FWbySublim))
    tmpscal2 = (op.rhoConstFresh*HEFFM[i, j]                                   # :2332-2339
                * ((EVAP-PRECIP)
                   * (ONE - AREApreTH)
                   - RUNOFF
                   + (d_HFRWbyRAIN + r_FWbySublim)
                   * convertHI2PRECIP))
    tmpscal1 = (- tmpscal1                                                     # :2343-2344
                * (-sp.SEAICE_lhFusion + op.HeatCapacity_Cp * ZERO))
    if op.temp_EvPrRn_set and op.useRealFreshWaterFlux and op.nonlinFreeSurf != 0:   # :2345-2348
        tmpscal2 = - tmpscal2*(ZERO + op.HeatCapacity_Cp * op.temp_EvPrRn)
    elif (not op.temp_EvPrRn_set) and op.useRealFreshWaterFlux and op.nonlinFreeSurf != 0:   # :2349-2352
        tmpscal2 = - tmpscal2*(ZERO + op.HeatCapacity_Cp * theta1)
    elif op.temp_EvPrRn_set:                                                   # :2353-2355
        tmpscal2 = - tmpscal2*op.HeatCapacity_Cp*(op.temp_EvPrRn - theta1)
    else:                                                                      # :2356-2357
        tmpscal2 = jnp.full_like(tmpscal2, ZERO)
    SItflux = SIatmQnt-tmpscal1-tmpscal2                                       # :2359
    del SItflux
    # :2367-2406 fresh water flux to the ocean
    tmpscal1 = (d_HSNWbyATMonSNW*SNOW2ICE                                      # :2369-2374
                + d_HFRWbyRAIN
                + d_HSNWbyOCNonSNW*SNOW2ICE
                + d_HEFFbyOCNonICE
                + d_HEFFbyATMonOCN
                + d_HEFFbyNEG)
    if seaice_fraction:
        tmpscal1 = tmpscal1 + sf["d_HEFFbyRLX"][i, j]                          # :2375-2377
    tmpscal1 = (tmpscal1                                                       # :2378-2380
                + d_HSNWbyNEG*SNOW2ICE
                + r_FWbySublim)
    EmPmR = (HEFFM[i, j]*(                                                     # :2381-2388
        (EVAP-PRECIP)
        * (ONE - AREApreTH)
        - RUNOFF
        + tmpscal1*convertHI2PRECIP
        )*op.rhoConstFresh)
    SIatmFW = (HEFFM[i, j]*(                                                   # :2396-2403
        EVAP*(ONE - AREApreTH)
        - PRECIP
        - RUNOFF
        )*op.rhoConstFresh
        + a_FWbySublim * sp.SEAICE_rhoIce * recip_deltaTtherm)
    del SIatmFW
    ff = ff.replace(Qnet=ff.Qnet.at[i, j].set(QNET), Qsw=ff.Qsw.at[i, j].set(QSW),
                    saltFlux=ff.saltFlux.at[i, j].set(saltFlux), EmPmR=ff.EmPmR.at[i, j].set(EmPmR))
    # :2424-2438 sea-ice load
    if op.useRealFreshWaterFlux:
        tmpscal2 = (HEFF_*sp.SEAICE_rhoIce                                     # :2432-2433
                    + HSNOW_*sp.SEAICE_rhoSnow)
        ff = ff.replace(sIceLoad=ff.sIceLoad.at[i, j].set(tmpscal2))           # :2435
    sf["HEFF"] = HEFF.at[i, j].set(HEFF_)
    sf["HSNOW"] = HSNOW.at[i, j].set(HSNOW_)
    sf["AREA"] = AREA.at[i, j].set(AREA_)
    if variable_salinity:
        sf["HSALT"] = sf["HSALT"].at[i, j].set(HSALT)
    if sitracer:      # SEAICE_TRACER.h, interior; SItrHEFF(:,:,2) keeps its value with SEAICE_growMeltByConv
        for jTh, v in SItrHEFF.items():
            sf["SItrHEFF"] = sf["SItrHEFF"].at[i, j, jTh].set(v)
        for jTh, v in SItrAREA.items():
            sf["SItrAREA"] = sf["SItrAREA"].at[i, j, jTh].set(v)
    if allow_salt_plume:
        salt_plume = dict(salt_plume, saltPlumeFlux=salt_plume["saltPlumeFlux"].at[i, j].set(saltPlumeFlux))
        return sf, ff, salt_plume
    return sf, ff
