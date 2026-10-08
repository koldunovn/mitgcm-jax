"""pkg/mom_common/mom_v_coriolis_nh.F: 3-D Coriolis term of the meridional momentum equation (MOM_V_CORIOLIS_NH)."""

from mitjax.farray import loop_i, loop_j


def mom_v_coriolis_nh(k, wFld, vCoriolisTerm, *, cfg, grid, params):
    """MOM_V_CORIOLIS_NH(bi, bj, k, wFld, vCoriolisTerm, myThid)   @63cdc0b pkg/mom_common/mom_v_coriolis_nh.F:7-84

    C Calculates the 3.D Coriolis term in the meridional momentum equation:
    C - \\overline{ f_prime \\overline{w}^{k} }^{i}
    C consistent with Non-Hydrostatic (or quasi-hydrostatic) formulation
    C  k                    :: vertical level
    C  wFld                 :: vertical flow
    C  vCoriolisTerm        :: Coriolis term

    Returns vCoriolisTerm. The V counterpart of the MOM lane's mom_u_coriolis_nh.py. `k` and select3dCoriScheme
    (PARAMS.h) are static. wFld is the 3-D field `wFld(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy)`. kp1 and wMsk
    (:45-47) are trace-time values; `1.` and `0.` are REAL*4 literals (exact). `-gravitySign*halfRL*(...)` negates
    gravitySign first (Python precedence), which gives the same bits as Fortran's -(gravitySign*halfRL*(...)).
    The point loops run on the whole (i,j) range.
    """
    sNx, sNy, OLx, OLy, Nr = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy, cfg.size.Nr
    fCoriCos, angleSinC, rA, deepFac2F = grid.fCoriCos, grid.angleSinC, grid.rA, grid.deepFac2F
    recip_rAs, recip_deepFac2C, recip_hFacS, gravitySign = (grid.recip_rAs, grid.recip_deepFac2C,
                                                             grid.recip_hFacS, grid.gravitySign)
    rVel2wUnit = params.rVel2wUnit
    halfRL = 0.5                                                    # EEPARAMS.h:73

    kp1 = min(k+1, Nr)                                              # :45-47; MINMAX-INT: integer (no tie or NaN case)
    wMsk = 1.
    if k == Nr:
        wMsk = 0.

    j = loop_j(2-OLy, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx)
    if params.select3dCoriScheme == 1:                              # :49-63
#-    Original discretization of 2*Omega*cos(phi)*w
        vCoriolisTerm = vCoriolisTerm.at[i, j].set(
            -gravitySign*halfRL
            *(fCoriCos[i, j]*angleSinC[i, j]*halfRL
              *(wFld[i, j, k]*rVel2wUnit[k]
                + wFld[i, j, kp1]*rVel2wUnit[kp1]*wMsk)
              + fCoriCos[i, j-1]*angleSinC[i, j-1]*halfRL
              *(wFld[i, j-1, k]*rVel2wUnit[k]
                + wFld[i, j-1, kp1]*rVel2wUnit[kp1]*wMsk)
              ))
    else:                                                           # :64-81
#-    Using averaged transport:
        vCoriolisTerm = vCoriolisTerm.at[i, j].set(
            -gravitySign*halfRL
            *(fCoriCos[i, j]*angleSinC[i, j]
              *(wFld[i, j, k]*rVel2wUnit[k]*deepFac2F[k]
                + wFld[i, j, kp1]*rVel2wUnit[kp1]*deepFac2F[kp1]*wMsk
                )*rA[i, j]*halfRL
              + fCoriCos[i, j-1]*angleSinC[i, j-1]
              *(wFld[i, j-1, k]*rVel2wUnit[k]*deepFac2F[k]
                + wFld[i, j-1, kp1]*rVel2wUnit[kp1]*deepFac2F[kp1]*wMsk
                )*rA[i, j-1]*halfRL
              )*recip_rAs[i, j]*recip_deepFac2C[k]
            *recip_hFacS[i, j, k])
    return vCoriolisTerm
