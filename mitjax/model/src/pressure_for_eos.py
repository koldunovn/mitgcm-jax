"""PRESSURE_FOR_EOS: model/src/pressure_for_eos.F @63cdc0b (called by FIND_RHO_2D for the pressure-dependent EOS)."""

from mitjax.farray import loop_i, loop_j


def pressure_for_eos(iMin, iMax, jMin, jMax, k, dpRef, locPres, *, cfg, grid, params, state):
    """PRESSURE_FOR_EOS( bi, bj, iMin, iMax, jMin, jMax, k, dpRef, locPres, myThid )
    @63cdc0b model/src/pressure_for_eos.F:3-117

    C     *==========================================================*
    C     | SUBROUTINE PRESSURE_FOR_EOS
    C     | o Provide a local copy of the total pressure
    C     |   at cell center (level k) for use in EOS funct. of P
    C     | Note: Since most seawater EOS are formulated as function
    C     |   of pressure anomaly relative to a reference P, this
    C     |   S/R allows to account for this reference Pressure (or
    C     |   different ref P) by adding a pressure shift "dpRef"
    C     |   to the output pressure.
    C     *==========================================================*

    Returns locPres; every point 1-OLx:sNx+OLx, 1-OLy:sNy+OLy is written whatever iMin..jMax are (the Fortran loops
    run over the full tile). Ported: z coordinates with selectP_inEOS_Zc = 2 (totPhiHyd + phiRef(2k), JMD95P),
    <= 1 (pRef4EOS, JMD95Z) and the "simplest case" else-branch; p coordinates (rC(k)). The ALLOW_NONHYDROSTATIC
    branch selectP_inEOS_Zc = 3 (:61-70) raises. selectP_inEOS_Zc and usingZCoords/usingPCoords are static values
    of `params` (as SET_PARMS leaves them, set_parms.F:272-301).
    """
    sz = cfg.size
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    sel = params.selectP_inEOS_Zc
    if params.usingZCoords:                                             # :58
        if cfg.cpp.ALLOW_NONHYDROSTATIC and sel == 3:                   # :60-70
            raise NotImplementedError("PRESSURE_FOR_EOS: selectP_inEOS_Zc = 3 (non-hydrostatic pressure) is not "
                                      "ported")
        if sel == 2:                                                    # :71-86
            locPres = locPres.at[i, j].set(
                params.rhoConst*(
                    state.totPhiHyd[i, j, k]
                    + params.phiRef[2*k]) + dpRef)
        elif sel <= 1:                                                  # :90-97
            locPres = locPres.at[i, j].set(params.pRef4EOS[k] + dpRef)
        else:                                                           # :98-105
            locPres = locPres.at[i, j].set(params.rhoConst*params.phiRef[2*k] + dpRef)
    elif params.usingPCoords:                                           # :106-113
        locPres = locPres.at[i, j].set(grid.rC[k] + dpRef)
    return locPres
