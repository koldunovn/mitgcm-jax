"""KPP_TRANSPORT_T: pkg/kpp/kpp_transport_t.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j


def kpp_transport_t(iMin, iMax, jMin, jMax, k, km1, df, myTime, myIter, *, cfg, grid, params, fp, ff, kppf,
                    kpp=None, gm=None):
    """KPP_TRANSPORT_T( iMin, iMax, jMin, jMax, bi, bj, k, km1, df, myTime, myIter, myThid )
    @63cdc0b pkg/kpp/kpp_transport_t.F:6-107

    C     | SUBROUTINE KPP_TRANSPORT_T
    C     | o Add non local KPP transport term (ghat) to diffusive
    C     |   temperature flux.
    C     | The nonlocal transport term is nonzero only for scalars
    C     | in unstable (convective) forcing conditions.
    C     | Note: KPPdiffKzT(-,k) is defined at the top of grid cell k
    C     |       while KPPghat(-,k) is defined at the bottom of grid
    C     |       cell k (-> Kpp index k-1 in MITgcm)

    df: (1-OLx:sNx+OLx,1-OLy:sNy+OLy) FArray; returns it. surfaceForcingT, adjustColdSST_diag, Qsw from `ff`
    (FFIELDS.h), HeatCapacity_Cp from `fp`; k, km1 static. Without ALLOW_GMREDI: :89-100. With ALLOW_GMREDI (lane
    M4LAB, lab_sea): :66-85, tmpFac = 1. _d 0 if useGMRedi .AND. KPP_ghatUseTotalDiffus (both static), else
    0. _d 0, times Kwz of GMREDI.h (`gm`); KPP_ghatUseTotalDiffus from KPP_PARAMS.h (`kpp`). The term
    tmpFac*Kwz is formed also when tmpFac = 0, as the Fortran forms it (the sign of a zero sum)."""
    recip_Cp = 1.0 / fp.HeatCapacity_Cp                                        # :65  1. _d 0 / HeatCapacity_Cp
    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    if cfg.cpp.flag("ALLOW_GMREDI", "KPP_OPTIONS.h"):                          # :66-85
        if kpp is None or gm is None:
            raise ValueError("KPP_TRANSPORT_T: ALLOW_GMREDI needs KPP_PARAMS.h `kpp` and GMREDI.h `gm`")
        if cfg.use_flag("useGMRedi") and kpp.KPP_ghatUseTotalDiffus:           # :68-72
            tmpFac = 1.                                                        # 1. _d 0
        else:
            tmpFac = 0.                                                        # 0. _d 0
        return df.at[i, j].set(- grid.rA[i, j]                                 # :73-84
                               * (kppf["KPPdiffKzT"][i, j, k] + tmpFac*gm.Kwz[i, j, k])
                               * kppf["KPPghat"][i, j, km1]
                               * (ff.surfaceForcingT[i, j]
                                  + ff.adjustColdSST_diag[i, j]
                                  - ff.Qsw[i, j]*recip_Cp*params.recip_rhoConst
                                  * kppf["KPPfrac"][i, j]))
    return df.at[i, j].set(- grid.rA[i, j]                                     # :91-98
                           * kppf["KPPdiffKzT"][i, j, k]
                           * kppf["KPPghat"][i, j, km1]
                           * (ff.surfaceForcingT[i, j]
                              + ff.adjustColdSST_diag[i, j]
                              - ff.Qsw[i, j]*recip_Cp*params.recip_rhoConst
                              * kppf["KPPfrac"][i, j]))
