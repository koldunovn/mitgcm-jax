"""SEAICE_CALC_STRAINRATES: pkg/seaice/seaice_calc_strainrates.F @63cdc0b (lane M4OFF session 3, the C-grid build)."""

from mitjax.farray import loop_i, loop_j

third = 0.333333333333333333333333333       # :68 PARAMETER ( third = 0.333333333333333333333333333 _d 0 )


def seaice_calc_strainrates(uFld, vFld, e11Loc, e22Loc, e12Loc, iStep, myTime, myIter, *, cfg, sp, grid, sf):
    """SEAICE_CALC_STRAINRATES( uFld, vFld, e11Loc, e22Loc, e12Loc, iStep, myTime, myIter, myThid )
    @63cdc0b pkg/seaice/seaice_calc_strainrates.F:11-204

    C     | o compute strain rates from ice velocities

    `sf` SEAICE.h / SEAICE_GRID.h (k1AtC, k2AtC, k1AtZ, k2AtZ, SIMaskU, SIMaskV, HEFFM). Returns (e11Loc, e22Loc,
    e12Loc): the points of :103-108 / :136-157 written, every other point as given (the Fortran arguments are the
    SEAICE.h fields e11, e22, e12). Ported without ALLOW_OBCS (OBCS_UVICE_OLD, :2-5) and without ALLOW_AUTODIFF
    (:193-199). The (i,j) loops are independent per point: vectorised; the locals dudx, uave, dvdy, vave, dudy, dvdx
    are read only where written. SEAICE_no_slip .AND. SEAICE_2ndOrderBC (:158-189) cannot hold with the LSR solver
    (SEAICE_READPARMS :906-907): it raises.
    Lane M4ADLAB session 2 (lab_sea/code_ad): ALLOW_AUTODIFF compiles only :194-199, ZERO_ADJ of e11Loc, e12Loc,
    e22Loc under SEAICE_DYN_STABLE_ADJOINT (an adjoint-only switch, undefined in code_ad/SEAICE_OPTIONS.h:230): no
    forward value; SEAICE_DYN_STABLE_ADJOINT raises."""
    del iStep, myTime, myIter
    if cfg.cpp.flag("ALLOW_OBCS"):
        raise NotImplementedError("SEAICE_CALC_STRAINRATES: ALLOW_OBCS is not ported")
    if cfg.cpp.flag("ALLOW_AUTODIFF") and cfg.cpp.flag("SEAICE_DYN_STABLE_ADJOINT", "SEAICE_OPTIONS.h"):   # :193-200
        raise NotImplementedError("SEAICE_CALC_STRAINRATES: SEAICE_DYN_STABLE_ADJOINT (ZERO_ADJ, :194-199) is not "
                                  "ported")
    sz = cfg.size
    OLx, OLy, sNx, sNy = sz.OLx, sz.OLy, sz.sNx, sz.sNy
    k1AtC, k2AtC, k1AtZ, k2AtZ = sf["k1AtC"], sf["k2AtC"], sf["k1AtZ"], sf["k2AtZ"]
    SIMaskU, SIMaskV, HEFFM = sf["SIMaskU"], sf["SIMaskV"], sf["HEFFM"]
    noSlipFac = 0.0                                                            # :78
    if sp.SEAICE_no_slip:                                                      # :79
        noSlipFac = 1.0
    j = loop_j(1-OLy, sNy+OLy-1)                                               # :88-89
    i = loop_i(1-OLx, sNx+OLx-1)
    dudx = grid.recip_dxF[i, j] * (uFld[i+1, j]-uFld[i, j])                   # :90-91
    uave = 0.5 * (uFld[i, j]+uFld[i+1, j])                                     # :92
    dvdy = grid.recip_dyF[i, j] * (vFld[i, j+1]-vFld[i, j])                   # :97-98
    vave = 0.5 * (vFld[i, j]+vFld[i, j+1])                                     # :99
    e11Loc = e11Loc.at[i, j].set(dudx + vave * k2AtC[i, j])                   # :105
    e22Loc = e22Loc.at[i, j].set(dvdy + uave * k1AtC[i, j])                   # :106
    j = loop_j(1-OLy+1, sNy+OLy)                                               # :121-122
    i = loop_i(1-OLx+1, sNx+OLx)
    dudy = (uFld[i, j] - uFld[i, j-1]) * grid.recip_dyU[i, j]                 # :123-124
    uave = 0.5 * (uFld[i, j]+uFld[i, j-1])                                     # :125
    dvdx = (vFld[i, j] - vFld[i-1, j]) * grid.recip_dxV[i, j]                 # :130-131
    vave = 0.5 * (vFld[i, j]+vFld[i-1, j])                                     # :132
    hFacU = SIMaskU[i, j] - SIMaskU[i, j-1]                                    # :138
    hFacV = SIMaskV[i, j] - SIMaskV[i-1, j]                                    # :139
    e12Loc = e12Loc.at[i, j].set(0.5 * (                                       # :140-150
        dudy + dvdx
        - k1AtZ[i, j] * vave
        - k2AtZ[i, j] * uave
    )
        * HEFFM[i, j]*HEFFM[i-1, j]
        * HEFFM[i, j-1]*HEFFM[i-1, j-1]
        + noSlipFac * (
        2.0 * uave * grid.recip_dyU[i, j] * hFacU
        + 2.0 * vave * grid.recip_dxV[i, j] * hFacV
    ))
    if sp.SEAICE_no_slip and sp.SEAICE_2ndOrderBC:                             # :158-189
        raise NotImplementedError("SEAICE_CALC_STRAINRATES: SEAICE_2ndOrderBC (:158-189) is not ported")
    return e11Loc, e22Loc, e12Loc
