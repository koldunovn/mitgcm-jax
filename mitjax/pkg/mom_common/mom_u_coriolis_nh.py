"""pkg/mom_common/mom_u_coriolis_nh.F: 3-D Coriolis term of the zonal momentum equation (MOM_U_CORIOLIS_NH)."""

from mitjax.farray import loop_i, loop_j


def mom_u_coriolis_nh(k, wFld, uCoriolisTerm, *, cfg, grid, params):
    """MOM_U_CORIOLIS_NH(bi, bj, k, wFld, uCoriolisTerm, myThid)   @63cdc0b pkg/mom_common/mom_u_coriolis_nh.F:3-84

    C Calculates the 3.D Coriolis term in the zonal momentum equation:
    C - \\overline{ f_prime \\overline{w}^{k} }^{i}
    C consistent with Non-Hydrostatic (or quasi-hydrostatic) formulation
    C  k                    :: vertical level
    C  wFld                 :: vertical flow
    C  uCoriolisTerm        :: Coriolis term

    Returns uCoriolisTerm. `k` and select3dCoriScheme (PARAMS.h) are static. wFld is the 3-D field
    `wFld(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy)`. kp1 and wMsk (:45-47) are trace-time values; `1.` and `0.` are
    REAL*4 literals (exact). The point loops run on the whole (i,j) range.
    """
    sNx, sNy, OLx, OLy, Nr = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy, cfg.size.Nr
    fCoriCos, angleCosC, rA, deepFac2F = grid.fCoriCos, grid.angleCosC, grid.rA, grid.deepFac2F
    recip_rAw, recip_deepFac2C, recip_hFacW, gravitySign = (grid.recip_rAw, grid.recip_deepFac2C,
                                                             grid.recip_hFacW, grid.gravitySign)
    rVel2wUnit = params.rVel2wUnit
    halfRL = 0.5                                                    # EEPARAMS.h:73

    kp1 = min(k+1, Nr)                                              # :45-47; MINMAX-INT: integer (no tie or NaN case)
    wMsk = 1.
    if k == Nr:
        wMsk = 0.

    j = loop_j(1-OLy, sNy+OLy)
    i = loop_i(2-OLx, sNx+OLx)
    if params.select3dCoriScheme == 1:                              # :49-63
#-    Original discretization of 2*Omega*cos(phi)*w
        uCoriolisTerm = uCoriolisTerm.at[i, j].set(
            gravitySign*halfRL
            *(fCoriCos[i, j]*angleCosC[i, j]*halfRL
              *(wFld[i, j, k]*rVel2wUnit[k]
                + wFld[i, j, kp1]*rVel2wUnit[kp1]*wMsk)
              + fCoriCos[i-1, j]*angleCosC[i-1, j]*halfRL
              *(wFld[i-1, j, k]*rVel2wUnit[k]
                + wFld[i-1, j, kp1]*rVel2wUnit[kp1]*wMsk)
              ))
    else:                                                           # :64-81
#-    Using averaged transport:
        uCoriolisTerm = uCoriolisTerm.at[i, j].set(
            gravitySign*halfRL
            *(fCoriCos[i, j]*angleCosC[i, j]
              *(wFld[i, j, k]*rVel2wUnit[k]*deepFac2F[k]
                + wFld[i, j, kp1]*rVel2wUnit[kp1]*deepFac2F[kp1]*wMsk
                )*rA[i, j]*halfRL
              + fCoriCos[i-1, j]*angleCosC[i-1, j]
              *(wFld[i-1, j, k]*rVel2wUnit[k]*deepFac2F[k]
                + wFld[i-1, j, kp1]*rVel2wUnit[kp1]*deepFac2F[kp1]*wMsk
                )*rA[i-1, j]*halfRL
              )*recip_rAw[i, j]*recip_deepFac2C[k]
            * recip_hFacW[i, j, k])
    return uCoriolisTerm
