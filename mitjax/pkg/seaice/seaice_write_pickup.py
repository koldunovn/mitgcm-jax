"""SEAICE_WRITE_PICKUP: pkg/seaice/seaice_write_pickup.F @63cdc0b (host side)."""

from mitjax.pkg.mdsio.mdsio_write_field import mds_wr_metafiles, mds_write_field
from mitjax.pkg.seaice.seaice_params_h import nITD

PRECFLOAT64 = 64       # EEPARAMS.h:65  PARAMETER ( precFloat64 = 64 )
oneRL = 1.0            # EEPARAMS.h  oneRL = 1.0 _d 0


def seaice_write_pickup(permPickup, suff, myTime, myIter, *, cfg, sp, sf, io, mds, useThSIce=False):
    """SEAICE_WRITE_PICKUP( permPickup, suff, myTime, myIter, myThid )   @63cdc0b
    pkg/seaice/seaice_write_pickup.F:6-216

    C     | o Write sea ice pickup file for restarting.

    `sf` SEAICE.h (concrete arrays), `io` (globalFiles), `mds` the run's MdsContext. 'pickup_seaice.'//suff (:68),
    precFloat64 (:72), records written with negative numbers (no meta file each), then one meta file
    (MDS_WR_METAFILES :205-213). Ported: the no-ITD build (:96-126), SEAICE_VARIABLE_SALINITY (:128-134), ALLOW_SITRACER's siTrac<nn> (:135-144, lane M4LAB session 4), UICE/VICE
    (:148-157), SEAICE_CGRID with SEAICE_ALLOW_EVP (:170-189: compiled; lane M4OFF: written only with SEAICEuseEVP,
    which stays .FALSE. in a ported run, seaice_readparms.py). Raises: SEAICE_ITD, SEAICEuseBDF2
    (:159-168), SEAICEuseEVP (:171-189). Returns the file name."""
    if cfg.cpp.flag("SEAICE_ITD", "SEAICE_OPTIONS.h"):
        raise NotImplementedError("SEAICE_WRITE_PICKUP: SEAICE_ITD is not ported")
    fn = "pickup_seaice." + suff                                                # :68  '(A,A)'
    fp = PRECFLOAT64                                                            # :72
    glf_w = io.globalFiles

    def wr(kSize, kLo, kHi, fld, nj):
        mds_write_field(fn, fp, glf_w, False, "RL", kSize, kLo, kHi, fld, nj, myIter, mds=mds)

    j = 0                                                                       # :73
    nj = 0                                                                      # :74
    wrFldList = []
    n_itd = nITD(cfg)
    if not useThSIce:                                                           # :78
        j = j + 1                                                               # :97
        nj = nj - 1                                                             # :98
        if sp.SEAICE_multDim > 1:                                               # :101-106
            wr(n_itd, 1, n_itd, sf["TICES"], nj)                                # WRITE_REC_3D_RL
            wrFldList.append("siTICES ")
            nj = nj - n_itd + 1                                                 # :105
        else:                                                                   # :107-110
            wr(n_itd, 1, 1, sf["TICES"], nj)                                    # WRITE_REC_LEV_RL( .., nITD, 1, 1, ..)
            wrFldList.append("siTICE  ")
        for name, label in (("AREA", "siAREA  "), ("HEFF", "siHEFF  "), ("HSNOW", "siHSNOW ")):   # :112-125
            j = j + 1
            nj = nj - 1
            wr(1, 1, 1, sf[name], nj)
            wrFldList.append(label)
        if cfg.cpp.flag("SEAICE_VARIABLE_SALINITY", "SEAICE_OPTIONS.h"):        # :128-134
            j = j + 1
            nj = nj - 1
            wr(1, 1, 1, sf["HSALT"], nj)
            wrFldList.append("siHSALT ")
        if cfg.cpp.flag("ALLOW_SITRACER", "SEAICE_OPTIONS.h"):                 # :135-144 (lane M4LAB session 4)
            from mitjax.pkg.seaice.seaice_h import level
            for iTrac in range(1, sp.SItrNumInUse + 1):                        # :136
                fldName = f"siTrac{iTrac:02d}"                                 # :137 '(A6,I2.2)'
                j = j + 1                                                      # :138
                nj = nj - 1                                                    # :139
                wr(1, 1, 1, level(sf["SItracer"], iTrac), nj)                  # :140-142 WRITE_REC_3D_RL
                wrFldList.append(fldName)                                      # :143
    for name, label in (("UICE", "siUICE  "), ("VICE", "siVICE  ")):           # :148-157
        j = j + 1
        nj = nj - 1
        wr(1, 1, 1, sf[name], nj)
        wrFldList.append(label)
    # SEAICEuseBDF2: not carried by SeaiceParams; SEAICE_READPARMS raises if data.seaice sets it, so it holds its
    # default .FALSE. (seaice_readparms.F:342)
    if getattr(sp, "SEAICEuseBDF2", False):                                     # :159-168
        raise NotImplementedError("SEAICE_WRITE_PICKUP: SEAICEuseBDF2 (uIceNm1, vIceNm1) is not ported")
    if cfg.cpp.flag("SEAICE_CGRID", "SEAICE_OPTIONS.h") and cfg.cpp.flag("SEAICE_ALLOW_EVP", "SEAICE_OPTIONS.h"):
        if sp.SEAICEuseEVP:                                                     # :170-189 (lane M4OFF)
            raise NotImplementedError("SEAICE_WRITE_PICKUP: SEAICEuseEVP (seaice_sigma1/2/12, :171-189) is not "
                                      "ported")
    nWrFlds = j                                                                 # :191
    nj = abs(nj)                                                                # :205
    glf = io.globalFiles                                                        # :206
    timList = [float(myTime)]                                                   # :207
    mds_wr_metafiles(fn, fp, glf, False, 0, 0, 1, " ", nWrFlds, wrFldList, 1, timList, oneRL, nj, myIter,
                     mds=mds)                                                   # :208-212
    return fn
