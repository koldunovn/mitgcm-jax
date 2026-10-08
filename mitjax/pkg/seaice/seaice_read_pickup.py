"""SEAICE_READ_PICKUP: pkg/seaice/seaice_read_pickup.F @63cdc0b (host side). Lane M4COL (the restart of
1D_ocean_ice_column: SEAICE_INIT_VARIA :251-254)."""

from mitjax.eesupp.exch_rs import EXCH_3D_RL, EXCH_UV_XYZ_RL, EXCH_XY_RL
from mitjax.farray import loop_i, loop_j
from mitjax.pkg.rw.read_mflds import READ_MFLDS_3D_RL, READ_MFLDS_CHECK, READ_MFLDS_LEV_RL, READ_MFLDS_SET
from mitjax.pkg.seaice.seaice_check_pickup import seaice_check_pickup
from mitjax.pkg.seaice.seaice_params_h import nITD

PRECFLOAT64 = 64       # EEPARAMS.h:65  PARAMETER ( precFloat64 = 64 )
missFldDim = 20        # seaice_read_pickup.F:51  PARAMETER( missFldDim = 20 )


def _exch_uv_xy_rl(u, v, withSigns, *, ex):
    """EXCH_UV_XY_RL( u, v, withSigns, myThid ): the 2-D vector exchange, as EXCH_UV_XYZ_RL on one level."""
    from mitjax.farray import FArray
    (_, ilo, ihi), (_, jlo, jhi) = u.dims
    u3 = FArray(u.data[:, None], u.name, i=(ilo, ihi), j=(jlo, jhi), k=(1, 1))
    v3 = FArray(v.data[:, None], v.name, i=(ilo, ihi), j=(jlo, jhi), k=(1, 1))
    u3, v3 = EXCH_UV_XYZ_RL(u3, v3, withSigns, ex=ex)
    return (FArray(u3.data[:, 0], u.name, tiled=u.tiled, _dims=u.dims),
            FArray(v3.data[:, 0], v.name, tiled=v.tiled, _dims=v.dims))


def seaice_read_pickup(sf, *, cfg, sp, tp, ip, rw, ex, useThSIce=False, pickupSuff=" "):
    """SEAICE_READ_PICKUP( myThid )   @63cdc0b pkg/seaice/seaice_read_pickup.F:6-340

    C     | o Read sea ice pickup file for restarting.

    `sf` SEAICE.h (dict of FArrays), `sp` SEAICE_PARAMS.h (SEAICE_multDim), `tp` the time parameters
    (nIter0), `ip` PARAMS.h's init group (rwSuffixType, pickupStrictlyMatch), `rw` the run directory reader.
    Returns (sf, {SEAICE_PARAMS.h values SEAICE_CHECK_PICKUP changed}). Ported: the pickup_seaice.<nIter0> name
    (:63-72; rwSuffixType 0), the precision check (:88-96), the new format with a field list (:182-278: siTICE (or
    siTICES) / siAREA / siHEFF / siHSNOW / siHSALT (SEAICE_VARIABLE_SALINITY) / siTrac<nn> (ALLOW_SITRACER, lane M4LAB session 4: a missing
    record leaves SEAICE_INIT_VARIA's value, read_mflds.F:308-321, then the exchange :249) / siUICE / siVICE),
    READ_MFLDS_CHECK + SEAICE_CHECK_PICKUP (:280-295), the copy of TICES(k=1) into
    k = 2..nITD on the interior (:302-315, doMapTice) and the exchanges (:317-322, :330-332). Raise: RW_GET_SUFFIX
    (rwSuffixType /= 0), the old format without a field list (nbFields <= 0, :101-180), SEAICE_ITD,
    SEAICEuseEVP (:168-175, :267-274, :324-328; lane M4OFF: SEAICE_CGRID with SEAICE_ALLOW_EVP compiled in
    offline_exf_seaice/code, SEAICEuseEVP stays .FALSE. in a ported run, seaice_readparms.py). useThSIce: the TICES/AREA/HEFF/
    HSNOW reads are skipped (:186), as the Fortran (an argument, the driver's value)."""
    if cfg.cpp.flag("SEAICE_ITD", "SEAICE_OPTIONS.h"):
        raise NotImplementedError("SEAICE_READ_PICKUP: SEAICE_ITD is not ported")
    if cfg.cpp.flag("SEAICE_CGRID", "SEAICE_OPTIONS.h") and cfg.cpp.flag("SEAICE_ALLOW_EVP", "SEAICE_OPTIONS.h") \
            and sp.SEAICEuseEVP:                                                # :167-175, :266-274, :323-328
        raise NotImplementedError("SEAICE_READ_PICKUP: SEAICEuseEVP (seaice_sigma1/2/12) is not ported")
    nIter0 = tp.nIter0
    if str(pickupSuff).strip() == "":                                           # :63-72
        if ip.rwSuffixType == 0:
            fn = f"pickup_seaice.{nIter0:010d}"                                 # '(A,I10.10)'
        else:
            raise NotImplementedError("SEAICE_READ_PICKUP: rwSuffixType /= 0 (RW_GET_SUFFIX) is not ported")
    else:
        fn = f"pickup_seaice.{str(pickupSuff):>10.10s}"                         # '(A,A10)'
    fp = PRECFLOAT64                                                            # :73
    doMapTice = False                                                           # :74
    n_itd = nITD(cfg)
    mf, nbFields, filePrec = READ_MFLDS_SET(fn, n_itd, nIter0, rw=rw)           # :82-85
    if nbFields >= 0 and filePrec != fp:                                        # :88-96
        raise ValueError(f"SEAICE_READ_PICKUP: pickup-file binary precision do not match !\n"
                         f"SEAICE_READ_PICKUP: file prec.={filePrec:4d} but expecting prec.={fp:4d}\n"
                         "ABNORMAL END: S/R SEAICE_READ_PICKUP (data-prec Pb)")
    if nbFields <= 0:                                                           # :101-180
        raise NotImplementedError("SEAICE_READ_PICKUP: a pickup without a field list (old format, :101-180) is "
                                  "not ported")
    sf = dict(sf)
    nj = 0                                                                      # :184

    def rd3(name, key, nNz):
        nonlocal mf, nj
        sf[key], nj, mf = READ_MFLDS_3D_RL(name, sf[key], nj, fp, nNz, nIter0, mf=mf, rw=rw)

    def rdlev(name):
        nonlocal mf, nj
        sf["TICES"], nj, mf = READ_MFLDS_LEV_RL(name, sf["TICES"], nj, fp, n_itd, 1, 1, nIter0, mf=mf, rw=rw)

    if not useThSIce:                                                           # :186
        if sp.SEAICE_multDim > 1:                                               # :187-195
            rd3("siTICES ", "TICES", n_itd)
            nj = nj*n_itd                                                       # :190
            if nj == 0:
                doMapTice = True                                                # :192
                rdlev("siTICE  ")
        else:                                                                   # :196-205
            doMapTice = True                                                    # :198
            rdlev("siTICE  ")
            if nj == 0:
                rdlev("siTICES ")
        rd3("siAREA  ", "AREA", 1)                                              # :214-215
        rd3("siHEFF  ", "HEFF", 1)                                              # :216-217
        rd3("siHSNOW ", "HSNOW", 1)                                             # :218-219
        if cfg.cpp.flag("SEAICE_VARIABLE_SALINITY", "SEAICE_OPTIONS.h"):        # :239-242
            rd3("siHSALT ", "HSALT", 1)
        if cfg.cpp.flag("ALLOW_SITRACER", "SEAICE_OPTIONS.h"):                  # :243-251 (lane M4LAB session 4)
            from mitjax.pkg.seaice.seaice_h import level, set_level
            for iTrac in range(1, sp.SItrNumInUse + 1):                         # :244
                fldName = f"siTrac{iTrac:02d}"                                  # :245 '(A6,I2.2)'
                lev = level(sf["SItracer"], iTrac)                              # SItracer(1-OLx,1-OLy,1,1,iTrac)
                lev, nj, mf = READ_MFLDS_3D_RL(fldName, lev, nj, fp, 1, nIter0, mf=mf, rw=rw)   # :246-248
                lev = EXCH_XY_RL(lev, ex=ex)                                    # :249
                sf["SItracer"] = set_level(sf["SItracer"], iTrac, lev)
    rd3("siUICE  ", "UICE", 1)                                                  # :256-257
    rd3("siVICE  ", "VICE", 1)                                                  # :258-259
    # :260-265 IF ( SEAICEuseBDF2 ): not carried by SeaiceParams (SEAICE_READPARMS raises if data.seaice sets it,
    # so it holds its default .FALSE., seaice_readparms.F:342; as SEAICE_WRITE_PICKUP)
    if getattr(sp, "SEAICEuseBDF2", False):
        raise NotImplementedError("SEAICE_READ_PICKUP: SEAICEuseBDF2 (siUicNm1, siVicNm1) is not ported")
    nMissing = missFldDim                                                       # :281
    missFldList, nMissing, mf = READ_MFLDS_CHECK(nMissing, nIter0, mf=mf)      # :282-285
    if nMissing > missFldDim:                                                   # :286-291
        raise ValueError(f"SEAICE_READ_PICKUP: missing fields list has been truncated to{missFldDim:4d}\n"
                         "ABNORMAL END: S/R SEAICE_READ_PICKUP (list-size Pb)")
    changed = seaice_check_pickup(missFldList, nMissing, nbFields, nIter0, cfg=cfg, sp=sp,   # :292-295
                                  pickupStrictlyMatch=ip.pickupStrictlyMatch)
    if doMapTice:                                                               # :302-315
        sz = cfg.size
        j, i = loop_j(1, sz.sNy), loop_i(1, sz.sNx)
        for k in range(2, n_itd + 1):
            sf["TICES"] = sf["TICES"].at[i, j, k].set(sf["TICES"][i, j, 1])
    sf["UICE"], sf["VICE"] = _exch_uv_xy_rl(sf["UICE"], sf["VICE"], True, ex=ex)   # :318
    sf["HEFF"] = EXCH_XY_RL(sf["HEFF"], ex=ex)                                  # :319
    sf["AREA"] = EXCH_XY_RL(sf["AREA"], ex=ex)                                  # :320
    sf["TICES"] = EXCH_3D_RL(sf["TICES"], n_itd, ex=ex)                         # :321
    sf["HSNOW"] = EXCH_XY_RL(sf["HSNOW"], ex=ex)                                # :322
    if cfg.cpp.flag("SEAICE_VARIABLE_SALINITY", "SEAICE_OPTIONS.h"):            # :330-332
        sf["HSALT"] = EXCH_XY_RL(sf["HSALT"], ex=ex)
    return sf, changed
