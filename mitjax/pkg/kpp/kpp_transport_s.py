"""KPP_TRANSPORT_S: pkg/kpp/kpp_transport_s.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j


def kpp_transport_s(iMin, iMax, jMin, jMax, k, km1, df, myTime, myIter, *, cfg, grid, ff, kppf, params=None,
                    kpp=None, gm=None, salt_plume=None):
    """KPP_TRANSPORT_S( iMin, iMax, jMin, jMax, bi, bj, k, km1, df, myTime, myIter, myThid )
    @63cdc0b pkg/kpp/kpp_transport_s.F:9-126

    C     | SUBROUTINE KPP_TRANSPORT_S
    C     | o Add non local KPP transport term (ghat) to diffusive
    C     |   salt flux.
    C     | The nonlocal transport term is nonzero only for scalars
    C     | in unstable (convective) forcing conditions.

    df: (1-OLx:sNx+OLx,1-OLy:sNy+OLy) FArray; returns it. surfaceForcingS from `ff`; k, km1 static. Without
    ALLOW_GMREDI and ALLOW_SALT_PLUME: :107-119. Lane M4LAB (lab_sea compiles both): ALLOW_GMREDI :84-104 with
    tmpFac = 1. _d 0 if useGMRedi .AND. KPP_ghatUseTotalDiffus (KPP_PARAMS.h `kpp`), else 0. _d 0, times Kwz
    (GMREDI.h `gm`); ALLOW_SALT_PLUME (:96-100, :112-116): + tmpFac1*saltPlumeFlux*recip_rhoConst
    *(1.-KPPplumefrac), tmpFac1 = 1. _d 0 if useSALT_PLUME (and not SALT_PLUME_VOLUME), else 0. _d 0 (:72-82);
    saltPlumeFlux from SALT_PLUME.h (`salt_plume`), KPPplumefrac from KPP.h (`kppf`), recip_rhoConst from PARAMS.h
    (`params`). The products with a zero tmpFac / tmpFac1 are formed as the Fortran forms them: they decide the sign
    of a zero sum (surfaceForcingS = -0. + (+0.) = +0.)."""
    gmredi = cfg.cpp.flag("ALLOW_GMREDI", "KPP_OPTIONS.h")
    sp_on = cfg.cpp.flag("ALLOW_SALT_PLUME", "KPP_OPTIONS.h")
    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    srf = ff.surfaceForcingS[i, j]
    if sp_on:                                                                  # :72-82
        if salt_plume is None or params is None:
            raise ValueError("KPP_TRANSPORT_S: ALLOW_SALT_PLUME needs SALT_PLUME.h `salt_plume` and `params`")
        if cfg.use_flag("useSALT_PLUME") and not cfg.cpp.flag("SALT_PLUME_VOLUME", "KPP_OPTIONS.h"):
            tmpFac1 = 1.                                                       # 1. _d 0
        else:
            tmpFac1 = 0.                                                       # 0. _d 0
        srf = (srf                                                             # :96-100 / :112-116
               + tmpFac1*salt_plume["saltPlumeFlux"][i, j]*params.recip_rhoConst
               * (1.-kppf["KPPplumefrac"][i, j]))                              # 1. (REAL*4, exact)
    if gmredi:                                                                 # :84-104
        if kpp is None or gm is None:
            raise ValueError("KPP_TRANSPORT_S: ALLOW_GMREDI needs KPP_PARAMS.h `kpp` and GMREDI.h `gm`")
        if cfg.use_flag("useGMRedi") and kpp.KPP_ghatUseTotalDiffus:           # :86-90
            tmpFac = 1.                                                        # 1. _d 0
        else:
            tmpFac = 0.                                                        # 0. _d 0
        return df.at[i, j].set(- grid.rA[i, j]                                 # :93-101
                               * (kppf["KPPdiffKzS"][i, j, k] + tmpFac*gm.Kwz[i, j, k])
                               * kppf["KPPghat"][i, j, km1]
                               * (srf))
    return df.at[i, j].set(- grid.rA[i, j]                                     # :109-117
                           * kppf["KPPdiffKzS"][i, j, k]
                           * kppf["KPPghat"][i, j, km1]
                           * (srf))
