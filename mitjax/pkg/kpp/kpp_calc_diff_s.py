"""KPP_CALC_DIFF_S: pkg/kpp/kpp_calc_diff_s.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j, loops_kji


def kpp_calc_diff_s(iMin, iMax, jMin, jMax, kArg, kSize, KappaRS, *, cfg, kppf):
    """KPP_CALC_DIFF_S( bi, bj, iMin, iMax, jMin, jMax, kArg, kSize, KappaRS, myThid )
    @63cdc0b pkg/kpp/kpp_calc_diff_s.F:3-67

    C     | SUBROUTINE KPP_CALC_DIFF_S
    C     | o Add contrubution to net diffusivity from KPP mixing

    KappaRS: (1-OLx:sNx+OLx,1-OLy:sNy+OLy,kSize) FArray; returns it. kArg = 0: levels 1..MIN(Nr,kSize) from
    KPPdiffKzS (:45-53, independent levels: k-vectorised); kArg > 0: level MIN(kArg,kSize) from level kArg
    (:56-61). kArg, kSize static ints."""
    Nr = cfg.size.Nr
    T = kppf["KPPdiffKzS"]
    if kArg == 0:                                                              # :45
        k, j, i = loops_kji((1, min(Nr, kSize)), (jMin, jMax), (iMin, iMax))   # :47  MINMAX-INT: static level count
        return KappaRS.at[i, j, k].set(T[i, j, k])                             # :50
    k = min(kArg, kSize)                                                       # :56  MINMAX-INT: static level index
    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    return KappaRS.at[i, j, k].set(T[i, j, kArg])                              # :59
