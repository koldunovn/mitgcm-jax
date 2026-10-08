"""SEAICE_INIT_VARIA: pkg/seaice/seaice_init_varia.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.eesupp.exch_rs import EXCH_UV_XY_RL, EXCH_XY_RL, EXCH_XY_RS
from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX, MIN
from mitjax.ops.libm import glibc_exp
from mitjax.pkg.rw.read_rec import READ_FLD_XY_RL
from mitjax.pkg.seaice.seaice_params_h import ONE, SItrMaxNum, ZERO, nITD

# the fields :60-158 set to 0. _d 0 on every point, by the option that compiles the statement
_ZEROED = ("HEFF", "AREA", "HSNOW", "UICE", "VICE", "uIceNm1", "vIceNm1", "DWATN")            # :64-79
_ZEROED_C = ("stressDivergenceX", "stressDivergenceY")                                        # :80-82 SEAICE_CGRID
_ZEROED_EVP = ("seaice_sigma1", "seaice_sigma2", "seaice_sigma12")                            # :83-87 +ALLOW_EVP
_ZEROED_CB = ("e11", "e22", "e12", "deltaC", "PRESS", "ETA", "etaZ", "ZETA", "FORCEX", "FORCEY",   # :89-99
              "tensileStrFac", "PRESS0", "FORCEX0", "FORCEY0", "SEAICE_zMax", "SEAICE_zMin")      # :100-105
_ZEROED_C2 = ("seaiceMassC", "seaiceMassU", "seaiceMassV")                                    # :107-110 SEAICE_CGRID
_ZEROED_FD = ("uice_fd", "vice_fd")                                                           # :111-114 +FREEDRIFT
_ZEROED_BD = ("CbotC",)                                                                       # :115-117 +BOTTOMDRAG
_ZEROED_B = ("uIceB", "vIceB", "AMASS", "DAIRN", "WINDX", "WINDY", "GWATX", "GWATY")          # :123-132
_ZEROED_SAL = ("HSALT",)                                                                      # :133-135


def seaice_init_varia(sf, ff, *, cfg, sp, op, state, params, tp, rw, ex, pickupSuff=" ", ip=None):
    """SEAICE_INIT_VARIA( myThid )   @63cdc0b pkg/seaice/seaice_init_varia.F:7-501

    C     | o Initialization of sea ice model.

    `sf` SEAICE.h / SEAICE_GRID.h after SEAICE_INIT_FIXED, `ff` FFields (sIceLoad), `state` (salt), `tp` time
    parameters (startTime, baseTime, nIter0), `pickupSuff` PARAMS.h's (as the driver passes it), `rw` the run
    directory reader (HsnowFile, the pickup), `ip` PARAMS.h's init group (rwSuffixType, pickupStrictlyMatch: the
    pickup start only). Host side (eager, once). Returns (sf, ff).
    Ported: the B-grid, no-ITD, SEAICE_VARIABLE_SALINITY build from a cold start (:256-408): HEFF =
    SEAICE_initialHEFF*HEFFM, AREA, HSNOW (+ HsnowFile), HSALT, or from a pickup (:251-254 SEAICE_READ_PICKUP, lane
    M4COL); then ZETA/ETA/PRESS0/zMax/zMin (:432-445) and sIceLoad (:447-455). The uIce/vIce/Heff/Area/Hsalt files
    (:271-332, :391-394; raised by SEAICE_READPARMS), SItrFile (:397-406), ALLOW_AUTODIFF and ALLOW_OBCS parts raise
    when compiled / used. ALLOW_SITRACER (lane M4LAB session 4): SEAICE_TRACER.h zeroed with 'one' tracers at 1
    (:136-149); on a pickup start SEAICE_READ_PICKUP reads siTrac<nn>.
    PLOT_FIELD_* (:235-248) is print-out. EXP is libm's (glibc_exp)."""
    for o in ("SEAICE_ITD", "SEAICE_ALLOW_SIDEDRAG"):
        if cfg.cpp.flag(o, "SEAICE_OPTIONS.h"):
            raise NotImplementedError(f"SEAICE_INIT_VARIA: {o} is not ported")
    sitracer = cfg.cpp.flag("ALLOW_SITRACER", "SEAICE_OPTIONS.h")              # lane M4LAB session 4
    # ALLOW_AUTODIFF (lane M4ADCOL): :160-166 / :495-498 wrap :167-494 in `IF ( useSEAICE )` (PACKAGES_INIT_VARIABLES
    # then calls the routine unconditionally); the driver calls it only with useSEAICE, so the arm changes nothing
    if cfg.cpp.flag("ALLOW_AUTODIFF") and not cfg.use_flag("useSEAICE"):
        raise NotImplementedError("SEAICE_INIT_VARIA: ALLOW_AUTODIFF without useSEAICE (:1-159 only) is not ported")
    if cfg.cpp.flag("ALLOW_OBCS") and cfg.use_flag("useOBCS"):
        raise NotImplementedError("SEAICE_INIT_VARIA: ALLOW_OBCS is not ported")
    sz = cfg.size
    OLx, OLy, sNx, sNy, Nr = sz.OLx, sz.OLy, sz.sNx, sz.sNy, sz.Nr
    if params.usingPCoords:                                                    # :53-57
        kSrf = Nr
    else:
        kSrf = 1
    sf = dict(sf)
    j = loop_j(1-OLy, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx)
    opt = lambda o: cfg.cpp.flag(o, "SEAICE_OPTIONS.h")                       # noqa: E731
    cgrid, bgrid = opt("SEAICE_CGRID"), opt("SEAICE_BGRID_DYNAMICS")
    zeroed = _ZEROED + (_ZEROED_C if cgrid else ()) + (_ZEROED_EVP if cgrid and opt("SEAICE_ALLOW_EVP") else ()) \
        + (_ZEROED_CB if cgrid or bgrid else ()) + (_ZEROED_C2 if cgrid else ()) \
        + (_ZEROED_FD if cgrid and opt("SEAICE_ALLOW_FREEDRIFT") else ()) + (_ZEROED_B if bgrid else ()) \
        + (_ZEROED_BD if cgrid and opt("SEAICE_ALLOW_BOTTOMDRAG") else ()) \
        + (_ZEROED_SAL if opt("SEAICE_VARIABLE_SALINITY") else ())
    for n in zeroed:                                                           # :60-158
        sf[n] = sf[n].at[i, j].set(0.0)
    if sitracer:                                                               # :136-149
        for iTr in range(1, SItrMaxNum + 1):                                   # :137-142
            sf["SItracer"] = sf["SItracer"].at[i, j, iTr].set(
                1.0 if sp.SItrName[iTr-1] == "one" else 0.0)                   # :138, :141 'one': 1. _d 0
            sf["SItrBucket"] = sf["SItrBucket"].at[i, j, iTr].set(0.0)         # :139
        for jTh in range(1, 5 + 1):                                            # :143-145
            sf["SItrHEFF"] = sf["SItrHEFF"].at[i, j, jTh].set(0.0)
        for jTh in range(1, 3 + 1):                                            # :146-148
            sf["SItrAREA"] = sf["SItrAREA"].at[i, j, jTh].set(0.0)
    for k in range(1, nITD(cfg) + 1):                                          # :150-152
        sf["TICES"] = sf["TICES"].at[i, j, k].set(0.0)
    sf["saltWtrIce"] = sf["saltWtrIce"].at[i, j].set(0.0)                      # :153
    sf["frWtrIce"] = sf["frWtrIce"].at[i, j].set(0.0)                          # :154
    HEFFM = sf["HEFFM"]
    if cgrid:                                                                  # :175-185
        jj = loop_j(1-OLy+1, sNy+OLy)
        ii = loop_i(1-OLx+1, sNx+OLx)
        sf["seaiceMaskU"] = sf["seaiceMaskU"].at[ii, jj].set(0.0)              # :177
        sf["seaiceMaskV"] = sf["seaiceMaskV"].at[ii, jj].set(0.0)              # :178
        mask_uice = HEFFM[ii, jj]+HEFFM[ii-1, jj]                              # :179
        sf["seaiceMaskU"] = sf["seaiceMaskU"].at[ii, jj].set(                  # :180
            jnp.where(mask_uice > 1.5, 1.0, sf["seaiceMaskU"][ii, jj]))
        mask_uice = HEFFM[ii, jj]+HEFFM[ii, jj-1]                              # :181
        sf["seaiceMaskV"] = sf["seaiceMaskV"].at[ii, jj].set(                  # :182
            jnp.where(mask_uice > 1.5, 1.0, sf["seaiceMaskV"][ii, jj]))
    for k in range(1, nITD(cfg) + 1):                                          # :207-211
        sf["TICES"] = sf["TICES"].at[i, j, k].set(273.0)                       # :210
    if cgrid:                                                                  # :212-216
        for n in ("seaiceMassC", "seaiceMassU", "seaiceMassV"):
            sf[n] = sf[n].at[i, j].set(1000.0)
    if bgrid:
        sf["AMASS"] = sf["AMASS"].at[i, j].set(1000.0)                         # :218
    if cgrid:                                                                  # :227-229
        sf["seaiceMaskU"], sf["seaiceMaskV"] = EXCH_UV_XY_RL(sf["seaiceMaskU"], sf["seaiceMaskV"], False, ex=ex)
    if bgrid:
        sf["UVM"] = EXCH_XY_RS(sf["UVM"], ex=ex)                               # :231
    if not (tp.startTime == tp.baseTime and tp.nIter0 == 0 and pickupSuff.strip() == ""):   # :251-252
        from mitjax.pkg.seaice.seaice_read_pickup import seaice_read_pickup
        if ip is None:
            raise ValueError("SEAICE_INIT_VARIA: a pickup start needs ip= (PARAMS.h rwSuffixType, "
                             "pickupStrictlyMatch)")
        sf, changed = seaice_read_pickup(sf, cfg=cfg, sp=sp, tp=tp, ip=ip, rw=rw, ex=ex,   # :254
                                         useThSIce=cfg.use_flag("useThSIce"), pickupSuff=pickupSuff)
        if changed:             # SEAICE_CHECK_PICKUP's SEAICEmomStartBDF = 0 (read by SEAICEuseBDF2 only)
            raise NotImplementedError(f"SEAICE_INIT_VARIA: SEAICE_CHECK_PICKUP changed {changed}: not carried")
    else:                                                                      # :256-408
        sf["HEFF"] = sf["HEFF"].at[i, j].set(sp.SEAICE_initialHEFF*HEFFM[i, j])   # :262
        sf["UICE"] = sf["UICE"].at[i, j].set(ZERO)                             # :263
        sf["VICE"] = sf["VICE"].at[i, j].set(ZERO)                             # :264
        if sp.uIceFile.strip() or sp.vIceFile.strip():                         # :269-289
            raise NotImplementedError("SEAICE_INIT_VARIA: uIceFile / vIceFile (:269-289) are not ported")
        if sp.HeffFile.strip():                                                # :292-304
            sf["HEFF"] = READ_FLD_XY_RL(sp.HeffFile, " ", sf["HEFF"], 0, rw=rw)   # :293
            sf["HEFF"] = EXCH_XY_RL(sf["HEFF"], ex=ex)                         # :294
            sf["HEFF"] = sf["HEFF"].at[i, j].set(MAX(sf["HEFF"][i, j], ZERO, p="a"))   # :299
        HEFF = sf["HEFF"]
        sf["AREA"] = sf["AREA"].at[i, j].set(jnp.where(HEFF[i, j] > ZERO, ONE, sf["AREA"][i, j]))   # :310
        if sp.AreaFile.strip():                                                # :317-333
            sf["AREA"] = READ_FLD_XY_RL(sp.AreaFile, " ", sf["AREA"], 0, rw=rw)   # :318
            sf["AREA"] = EXCH_XY_RL(sf["AREA"], ex=ex)                         # :319
            AREA = MAX(sf["AREA"][i, j], ZERO, p="a")                          # :324
            AREA = MIN(AREA, ONE, p="a")                                       # :325
            HEFF = jnp.where(AREA <= ZERO, ZERO, sf["HEFF"][i, j])             # :326
            AREA = jnp.where(HEFF <= ZERO, ZERO, AREA)                         # :327
            sf["AREA"] = sf["AREA"].at[i, j].set(AREA)
            sf["HEFF"] = sf["HEFF"].at[i, j].set(HEFF)
            HEFF = sf["HEFF"]
        sf["HSNOW"] = sf["HSNOW"].at[i, j].set(0.2*sf["AREA"][i, j])           # :338
        if sp.HsnowFile.strip():                                               # :345-357
            sf["HSNOW"] = READ_FLD_XY_RL(sp.HsnowFile, " ", sf["HSNOW"], 0, rw=rw)   # :346
            sf["HSNOW"] = EXCH_XY_RL(sf["HSNOW"], ex=ex)                       # :347
            sf["HSNOW"] = sf["HSNOW"].at[i, j].set(MAX(sf["HSNOW"][i, j], ZERO, p="a"))   # :352
        if opt("SEAICE_VARIABLE_SALINITY"):                                    # :376-396
            sf["HSALT"] = sf["HSALT"].at[i, j].set(HEFF[i, j]*state.salt[i, j, kSrf]   # :381-382
                                                   * sp.SEAICE_rhoIce*sp.SEAICE_saltFrac)
            if sp.HsaltFile.strip():                                           # :391-394
                raise NotImplementedError("SEAICE_INIT_VARIA: HsaltFile (:391-394) is not ported")
        if sitracer and any(f.strip() for f in sp.SItrFile):                   # :397-406
            raise NotImplementedError("SEAICE_INIT_VARIA: SItrFile (READ_FLD_XY_RL of SItracer, :397-406) is not "
                                      "ported")
    HEFF = sf["HEFF"]
    # :432-445
    AREA = sf["AREA"]
    if cgrid or bgrid:                    # :434 (lane M4ADCOL: the column's code_ad compiles neither: block absent)
        sf["ZETA"] = sf["ZETA"].at[i, j].set(HEFF[i, j]*(1.0e11))              # :437
        sf["ETA"] = sf["ETA"].at[i, j].set(sf["ZETA"][i, j]/sp.SEAICE_eccen**2)   # :438
        PRESS0 = (sp.SEAICE_strength*HEFF[i, j]                                # :439-440
                  * glibc_exp(-sp.SEAICE_cStar*(ONE-AREA[i, j])))
        sf["SEAICE_zMax"] = sf["SEAICE_zMax"].at[i, j].set(sp.SEAICE_zetaMaxFac*PRESS0)   # :441
        sf["SEAICE_zMin"] = sf["SEAICE_zMin"].at[i, j].set(sp.SEAICE_zetaMin*jnp.ones_like(PRESS0))   # :442
        sf["PRESS0"] = sf["PRESS0"].at[i, j].set(PRESS0*HEFFM[i, j])           # :443
    if op.useRealFreshWaterFlux and not cfg.use_flag("useThSIce"):             # :447-455
        ff = ff.replace(sIceLoad=ff.sIceLoad.at[i, j].set(HEFF[i, j]*sp.SEAICE_rhoIce
                                                          + sf["HSNOW"][i, j]*sp.SEAICE_rhoSnow))
    if cgrid and sp.SEAICE_tensilFac != 0.:                                    # :462-477
        raise NotImplementedError("SEAICE_INIT_VARIA: SEAICE_tensilFac /= 0 (tensileStrFac, :462-477) is not ported")
    # :478-490 (SEAICE_ALLOW_JFNK / KRYLOV): scalarProductMetric of the JFNK / Krylov solvers is not carried (read
    # only with SEAICEuseJFNK / SEAICEuseKrylov, which this port raises on)
    return sf, ff
