"""pkg/mom_vecinv/mom_vi_u_coriolis.F: flux (in Y) of vorticity at U points (MOM_VI_U_CORIOLIS)."""

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX


def mom_vi_u_coriolis(k, selectVortScheme, useJamartMomAdv, vFld, omega3, hFacZ, r_hFacZ, uCoriolisTerm,
                      *, cfg, grid):
    """MOM_VI_U_CORIOLIS(bi, bj, k, selectVortScheme, useJamartMomAdv, vFld, omega3, hFacZ, r_hFacZ,
                         uCoriolisTerm, myThid)   @63cdc0b pkg/mom_vecinv/mom_vi_u_coriolis.F:6-193

    C     | S/R MOM_VI_U_CORIOLIS
    C     | o Calculate flux (in Y-dir.) of vorticity at U point
    C     |   using 2nd order interpolation

    Returns uCoriolisTerm. selectVortScheme and useJamartMomAdv are static (arguments that select branches); an
    unknown scheme raises with the Fortran message (:170-176). epsil (`_RS`) = `1. _d -9` (:52); oneThird =
    `1. _d 0 / 3. _d 0` (:55, the division in double); tmpFac is only read in commented-out lines.
    `0.25 _d 0`, `4. _d 0`, halfRL (EEPARAMS.h:73) are exact. MAX winners from the oracle's site table. The point
    loops run on the whole (i,j) range; the Jamart rescaling (:179-190) reads the values of the loop before it.
    """
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    dxG, recip_dxC = grid.dxG, grid.recip_dxC
    hFacW, hFacS, recip_hFacW, maskW = grid.hFacW, grid.hFacS, grid.recip_hFacW, grid.maskW
    halfRL = 0.5                                                    # EEPARAMS.h:73

    epsil = 1.e-9                                                   # :52
    oneThird = 1. / 3.                                              # :55

    if selectVortScheme == 0:                                       # :57-73
#--   using enstrophy conserving scheme (Shallow-Water Eq.) by Sadourny, JAS 75
        j = loop_j(1-OLy, sNy+OLy-1)
        i = loop_i(2-OLx, sNx+OLx)
        vBarXY = 0.25*(
            (vFld[i, j]*dxG[i, j]*hFacS[i, j, k]
             + vFld[i-1, j]*dxG[i-1, j]*hFacS[i-1, j, k])
            +(vFld[i, j+1]*dxG[i, j+1]*hFacS[i, j+1, k]
              + vFld[i-1, j+1]*dxG[i-1, j+1]*hFacS[i-1, j+1, k])
                       )
        vort3u = halfRL*(omega3[i, j]*r_hFacZ[i, j]
                         + omega3[i, j+1]*r_hFacZ[i, j+1])
        uCoriolisTerm = uCoriolisTerm.at[i, j].set(+vort3u*vBarXY*recip_dxC[i, j]
                                                   * maskW[i, j, k])

    elif selectVortScheme == 1:                                     # :75-90
#--   same as above, with different formulation (relatively to hFac)
        j = loop_j(1-OLy, sNy+OLy-1)
        i = loop_i(2-OLx, sNx+OLx)
        vBarXY = halfRL*(
            (vFld[i, j]*dxG[i, j]*hFacZ[i, j]
             + vFld[i-1, j]*dxG[i-1, j]*hFacZ[i, j])
            +(vFld[i, j+1]*dxG[i, j+1]*hFacZ[i, j+1]
              + vFld[i-1, j+1]*dxG[i-1, j+1]*hFacZ[i, j+1])
                         )/MAX(epsil, hFacZ[i, j]+hFacZ[i, j+1], p="a")   # :80-85
        vort3u = halfRL*(omega3[i, j] + omega3[i, j+1])
        uCoriolisTerm = uCoriolisTerm.at[i, j].set(+vort3u*vBarXY*recip_dxC[i, j]
                                                   * maskW[i, j, k])

    elif selectVortScheme == 2:                                     # :92-109
#--   using energy conserving scheme (used by Sadourny in JAS 75 paper)
        j = loop_j(1-OLy, sNy+OLy-1)
        i = loop_i(2-OLx, sNx+OLx)
        vBarXm = halfRL*(
            vFld[i, j]*dxG[i, j]*hFacS[i, j, k]
            + vFld[i-1, j]*dxG[i-1, j]*hFacS[i-1, j, k])
        vBarXp = halfRL*(
            vFld[i, j+1]*dxG[i, j+1]*hFacS[i, j+1, k]
            + vFld[i-1, j+1]*dxG[i-1, j+1]*hFacS[i-1, j+1, k])
        vort3u = (vBarXm*r_hFacZ[i, j]*omega3[i, j]
                  + vBarXp*r_hFacZ[i, j+1]*omega3[i, j+1]
                  )*halfRL
        uCoriolisTerm = uCoriolisTerm.at[i, j].set(+vort3u*recip_dxC[i, j]
                                                   * maskW[i, j, k])

    elif selectVortScheme == 3:                                     # :111-149
#--   using energy & enstrophy conserving scheme
#     (from Sadourny, described by Burridge & Haseler, ECMWF Rep.4, 1977)
        j = loop_j(1-OLy, sNy+OLy-1)
        i = loop_i(2-OLx, sNx+OLx-1)
        vort3mj = ((r_hFacZ[i, j]*omega3[i, j]
                    +(r_hFacZ[i, j+1]*omega3[i, j+1]
                      +r_hFacZ[i-1, j]*omega3[i-1, j]
                      ))*oneThird
                   *vFld[i-1, j]*dxG[i-1, j]*hFacS[i-1, j, k])
        vort3ij = ((r_hFacZ[i, j]*omega3[i, j]
                    +(r_hFacZ[i, j+1]*omega3[i, j+1]
                      +r_hFacZ[i+1, j]*omega3[i+1, j]
                      ))*oneThird
                   *vFld[i, j]*dxG[i, j]*hFacS[i, j, k])
        vort3mp = ((r_hFacZ[i, j+1]*omega3[i, j+1]
                    +(r_hFacZ[i, j]*omega3[i, j]
                      +r_hFacZ[i-1, j+1]*omega3[i-1, j+1]
                      ))*oneThird
                   *vFld[i-1, j+1]*dxG[i-1, j+1]*hFacS[i-1, j+1, k])
        vort3ip = ((r_hFacZ[i, j+1]*omega3[i, j+1]
                    +(r_hFacZ[i, j]*omega3[i, j]
                      +r_hFacZ[i+1, j+1]*omega3[i+1, j+1]
                      ))*oneThird
                   *vFld[i, j+1]*dxG[i, j+1]*hFacS[i, j+1, k])
        uCoriolisTerm = uCoriolisTerm.at[i, j].set(+((vort3mj+vort3ij)+(vort3mp+vort3ip))
                                                   *0.25*recip_dxC[i, j]
                                                   * maskW[i, j, k])

    elif selectVortScheme == 4:                                     # :151-168
#--   using energy conserving scheme, no hFac weighting
        j = loop_j(1-OLy, sNy+OLy-1)
        i = loop_i(2-OLx, sNx+OLx)
        vBarXm = halfRL*(
            vFld[i, j]*dxG[i, j]*hFacS[i, j, k]
            + vFld[i-1, j]*dxG[i-1, j]*hFacS[i-1, j, k])
        vBarXp = halfRL*(
            vFld[i, j+1]*dxG[i, j+1]*hFacS[i, j+1, k]
            + vFld[i-1, j+1]*dxG[i-1, j+1]*hFacS[i-1, j+1, k])
        vort3u = (vBarXm*omega3[i, j]
                  + vBarXp*omega3[i, j+1]
                  )*halfRL
        uCoriolisTerm = uCoriolisTerm.at[i, j].set(+vort3u*recip_dxC[i, j]
                                                   *recip_hFacW[i, j, k])

    else:                                                           # :170-176
        raise ValueError(f"MOM_VI_U_CORIOLIS: selectVortScheme={selectVortScheme:5d} not implemented")

    if useJamartMomAdv:                                             # :179-190
        j = loop_j(1-OLy, sNy+OLy-1)
        i = loop_i(2-OLx, sNx+OLx-1)
        uCoriolisTerm = uCoriolisTerm.at[i, j].set(
            uCoriolisTerm[i, j]
            * 4. * hFacW[i, j, k]
            / MAX(epsil,                                            # :182-187
                  (hFacS[i, j, k]+hFacS[i-1, j, k])
                  +(hFacS[i, j+1, k]+hFacS[i-1, j+1, k]), p="a"))
    return uCoriolisTerm
