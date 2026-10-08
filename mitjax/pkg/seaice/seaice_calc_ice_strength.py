"""SEAICE_CALC_ICE_STRENGTH: pkg/seaice/seaice_calc_ice_strength.F @63cdc0b (lane M4OFF session 3, the C-grid build
without SEAICE_ITD)."""

from mitjax.farray import loop_i, loop_j
from mitjax.ops.libm import glibc_exp


def seaice_calc_ice_strength(myTime, myIter, sf, *, cfg, sp):
    """SEAICE_CALC_ICE_STRENGTH( bi, bj, myTime, myIter, myThid )
    @63cdc0b pkg/seaice/seaice_calc_ice_strength.F:9-200

    C     | o compute ice strength PRESS0 (and its limits SEAICE_zMax, SEAICE_zMin)

    `sf` SEAICE.h (HEFF, AREA, HEFFM; PRESS0, SEAICE_zMax, SEAICE_zMin are written). Returns sf.
    Ported: no SEAICE_ITD (the `IF ( .TRUE. )` arm :100-122, Hibler 1979 strength, on every point 1-OLx..sNx+OLx,
    1-OLy..sNy+OLy :83-86); SEAICE_ITD raises. The (i,j) loop is independent per point: vectorised. The power
    arms (:105-112, SEAICEpresPow0/1 /= 1) are static integer tests; `**` with an INTEGER exponent is
    raised on (not executed by offline_exf_seaice/input.dyn_lsr, Pow0 = Pow1 = 1). EXP is glibc's (libm.py)."""
    del myTime, myIter
    if cfg.cpp.flag("SEAICE_ITD", "SEAICE_OPTIONS.h"):
        raise NotImplementedError("SEAICE_CALC_ICE_STRENGTH: SEAICE_ITD is not ported")
    if sp.SEAICEpresPow0 != 1 or sp.SEAICEpresPow1 != 1:                       # :105-112
        # the arms need HEFF <= / > SEAICEpresH0 per point and tmpscal1**SEAICEpresPow (gfortran's integer power)
        raise NotImplementedError("SEAICE_CALC_ICE_STRENGTH: SEAICEpresPow0/1 /= 1 (:105-112) is not ported")
    sz = cfg.size
    HEFF, AREA, HEFFM = sf["HEFF"], sf["AREA"], sf["HEFFM"]
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)                                        # :85-86, :102
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)                                        # :83-84, :103
    tmpscal2 = HEFF[i, j]                                                      # :114
    PRESS0 = sf["PRESS0"].at[i, j].set(sp.SEAICE_strength*tmpscal2             # :116-117
                                       * glibc_exp(-sp.SEAICE_cStar*(sp.SEAICE_area_max-AREA[i, j])))
    zMax = sf["SEAICE_zMax"].at[i, j].set(sp.SEAICE_zetaMaxFac*PRESS0[i, j])   # :118
    zMin = sf["SEAICE_zMin"].at[i, j].set(sp.SEAICE_zetaMin)                   # :119
    PRESS0 = PRESS0.at[i, j].set(PRESS0[i, j]*HEFFM[i, j])                     # :120
    return {**sf, "PRESS0": PRESS0, "SEAICE_zMax": zMax, "SEAICE_zMin": zMin}
