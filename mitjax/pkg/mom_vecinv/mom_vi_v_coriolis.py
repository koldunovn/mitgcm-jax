"""pkg/mom_vecinv/mom_vi_v_coriolis.F: flux (in X) of vorticity at V points (MOM_VI_V_CORIOLIS)."""

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX


def mom_vi_v_coriolis(k, selectVortScheme, useJamartMomAdv, uFld, omega3, hFacZ, r_hFacZ, vCoriolisTerm,
                      *, cfg, grid):
    """MOM_VI_V_CORIOLIS(bi, bj, k, selectVortScheme, useJamartMomAdv, uFld, omega3, hFacZ, r_hFacZ,
                         vCoriolisTerm, myThid)   @63cdc0b pkg/mom_vecinv/mom_vi_v_coriolis.F:6-193

    C     | S/R MOM_VI_V_CORIOLIS
    C     | o Calculate flux (in X-dir.) of vorticity at V point
    C     |   using 2nd order interpolation

    Returns vCoriolisTerm. selectVortScheme and useJamartMomAdv are static (arguments that select branches); an
    unknown scheme raises with the Fortran message (:170-176). epsil (`_RS`) = `1. _d -9` (:52); oneThird =
    `1. _d 0 / 3. _d 0` (:55); tmpFac is only read in commented-out lines. `0.25 _d 0`, `4. _d 0`, halfRL
    (EEPARAMS.h:73) are exact. MAX winners from the oracle's site table. The point loops run on the whole (i,j)
    range; the Jamart rescaling (:179-190) reads the values of the loop before it.
    """
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    dyG, recip_dyC = grid.dyG, grid.recip_dyC
    hFacW, hFacS, recip_hFacS, maskS = grid.hFacW, grid.hFacS, grid.recip_hFacS, grid.maskS
    halfRL = 0.5                                                    # EEPARAMS.h:73

    epsil = 1.e-9                                                   # :52
    oneThird = 1. / 3.                                              # :55

    if selectVortScheme == 0:                                       # :57-73
#--   using enstrophy conserving scheme (Shallow-Water Eq.) by Sadourny, JAS 75
        j = loop_j(2-OLy, sNy+OLy)
        i = loop_i(1-OLx, sNx+OLx-1)
        uBarXY = 0.25*(
            (uFld[i, j]*dyG[i, j]*hFacW[i, j, k]
             + uFld[i, j-1]*dyG[i, j-1]*hFacW[i, j-1, k])
            +(uFld[i+1, j]*dyG[i+1, j]*hFacW[i+1, j, k]
              + uFld[i+1, j-1]*dyG[i+1, j-1]*hFacW[i+1, j-1, k])
                       )
        vort3v = halfRL*(omega3[i, j]*r_hFacZ[i, j]
                         + omega3[i+1, j]*r_hFacZ[i+1, j])
        vCoriolisTerm = vCoriolisTerm.at[i, j].set(-vort3v*uBarXY*recip_dyC[i, j]
                                                   * maskS[i, j, k])

    elif selectVortScheme == 1:                                     # :75-90
#--   same as above, with different formulation (relatively to hFac)
        j = loop_j(2-OLy, sNy+OLy)
        i = loop_i(1-OLx, sNx+OLx-1)
        uBarXY = halfRL*(
            (uFld[i, j]*dyG[i, j]*hFacZ[i, j]
             + uFld[i, j-1]*dyG[i, j-1]*hFacZ[i, j])
            +(uFld[i+1, j]*dyG[i+1, j]*hFacZ[i+1, j]
              + uFld[i+1, j-1]*dyG[i+1, j-1]*hFacZ[i+1, j])
                         )/MAX(epsil, hFacZ[i, j]+hFacZ[i+1, j], p="a")   # :80-85
        vort3v = halfRL*(omega3[i, j] + omega3[i+1, j])
        vCoriolisTerm = vCoriolisTerm.at[i, j].set(-vort3v*uBarXY*recip_dyC[i, j]
                                                   * maskS[i, j, k])

    elif selectVortScheme == 2:                                     # :92-109
#--   using energy conserving scheme (used by Sadourny in JAS 75 paper)
        j = loop_j(2-OLy, sNy+OLy)
        i = loop_i(1-OLx, sNx+OLx-1)
        uBarYm = halfRL*(
            uFld[i, j]*dyG[i, j]*hFacW[i, j, k]
            + uFld[i, j-1]*dyG[i, j-1]*hFacW[i, j-1, k])
        uBarYp = halfRL*(
            uFld[i+1, j]*dyG[i+1, j]*hFacW[i+1, j, k]
            + uFld[i+1, j-1]*dyG[i+1, j-1]*hFacW[i+1, j-1, k])
        vort3v = (uBarYm*r_hFacZ[i, j]*omega3[i, j]
                  + uBarYp*r_hFacZ[i+1, j]*omega3[i+1, j]
                  )*halfRL
        vCoriolisTerm = vCoriolisTerm.at[i, j].set(-vort3v*recip_dyC[i, j]
                                                   * maskS[i, j, k])

    elif selectVortScheme == 3:                                     # :111-149
#--   using energy & enstrophy conserving scheme
#     (from Sadourny, described by Burridge & Haseler, ECMWF Rep.4, 1977)
        j = loop_j(2-OLy, sNy+OLy-1)
        i = loop_i(1-OLx, sNx+OLx-1)
        vort3im = ((r_hFacZ[i, j]*omega3[i, j]
                    +(r_hFacZ[i+1, j]*omega3[i+1, j]
                      +r_hFacZ[i, j-1]*omega3[i, j-1]
                      ))*oneThird
                   *uFld[i, j-1]*dyG[i, j-1]*hFacW[i, j-1, k])
        vort3ij = ((r_hFacZ[i, j]*omega3[i, j]
                    +(r_hFacZ[i+1, j]*omega3[i+1, j]
                      +r_hFacZ[i, j+1]*omega3[i, j+1]
                      ))*oneThird
                   *uFld[i, j]*dyG[i, j]*hFacW[i, j, k])
        vort3pm = ((r_hFacZ[i+1, j]*omega3[i+1, j]
                    +(r_hFacZ[i, j]*omega3[i, j]
                      +r_hFacZ[i+1, j-1]*omega3[i+1, j-1]
                      ))*oneThird
                   *uFld[i+1, j-1]*dyG[i+1, j-1]*hFacW[i+1, j-1, k])
        vort3pj = ((r_hFacZ[i+1, j]*omega3[i+1, j]
                    +(r_hFacZ[i, j]*omega3[i, j]
                      +r_hFacZ[i+1, j+1]*omega3[i+1, j+1]
                      ))*oneThird
                   *uFld[i+1, j]*dyG[i+1, j]*hFacW[i+1, j, k])
        vCoriolisTerm = vCoriolisTerm.at[i, j].set(-((vort3im+vort3ij)+(vort3pm+vort3pj))
                                                   *0.25*recip_dyC[i, j]
                                                   * maskS[i, j, k])

    elif selectVortScheme == 4:                                     # :151-168
#--   using energy conserving scheme, no hFac weighting
        j = loop_j(2-OLy, sNy+OLy)
        i = loop_i(1-OLx, sNx+OLx-1)
        uBarYm = halfRL*(
            uFld[i, j]*dyG[i, j]*hFacW[i, j, k]
            + uFld[i, j-1]*dyG[i, j-1]*hFacW[i, j-1, k])
        uBarYp = halfRL*(
            uFld[i+1, j]*dyG[i+1, j]*hFacW[i+1, j, k]
            + uFld[i+1, j-1]*dyG[i+1, j-1]*hFacW[i+1, j-1, k])
        vort3v = (uBarYm*omega3[i, j]
                  + uBarYp*omega3[i+1, j]
                  )*halfRL
        vCoriolisTerm = vCoriolisTerm.at[i, j].set(-vort3v*recip_dyC[i, j]
                                                   *recip_hFacS[i, j, k])

    else:                                                           # :170-176
        raise ValueError(f"MOM_VI_V_CORIOLIS: selectVortScheme={selectVortScheme:5d} not implemented")

    if useJamartMomAdv:                                             # :179-190
        j = loop_j(2-OLy, sNy+OLy-1)
        i = loop_i(1-OLx, sNx+OLx-1)
        vCoriolisTerm = vCoriolisTerm.at[i, j].set(
            vCoriolisTerm[i, j]
            * 4. * hFacS[i, j, k]
            / MAX(epsil,                                            # :182-187
                  (hFacW[i, j, k]+hFacW[i, j-1, k])
                  +(hFacW[i+1, j, k]+hFacW[i+1, j-1, k]), p="a"))
    return vCoriolisTerm
