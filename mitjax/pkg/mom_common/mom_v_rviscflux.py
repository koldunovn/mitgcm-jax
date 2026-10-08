"""pkg/mom_common/mom_v_rviscflux.F: vertical viscous flux of V at interface k (MOM_V_RVISCFLUX)."""

from mitjax.farray import loop_i, loop_j


def mom_v_rviscflux(k, vFld, KappaRV, rViscFluxV, *, cfg, grid, params):
    """MOM_V_RVISCFLUX(bi,bj,k, vFld, KappaRV, rViscFluxV, myThid)   @63cdc0b pkg/mom_common/mom_v_rviscflux.F:3-69

    C Calculates the area integrated vertical viscous fluxes of V
    C  at vertical interface k (between level k & k-1):
    C F^r = - \\frac{ {\\cal A}_s }{\\Delta r_c} A_r \\delta_k v
    C  k                    :: vertical level
    C  vFld                 :: meridional flow
    C  KappaRV              :: vertical viscosity
    C  rViscFluxV           :: viscous fluxes

    Returns rViscFluxV. As MOM_U_RVISCFLUX (static `k`; KappaRV declared `(..,Nr)` here, `(..,Nr+1)` by the caller;
    levels 2..Nr read).
    """
    sNx, sNy, OLx, OLy, Nr = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy, cfg.size.Nr
    rAs, deepFac2F, recip_drC, maskS, rkSign = grid.rAs, grid.deepFac2F, grid.recip_drC, grid.maskS, grid.rkSign
    rhoFacF = params.rhoFacF

#     - Vertical viscous flux
    if k <= 1 or k > Nr:                                            # :48-53
        j = loop_j(1-OLy, sNy+OLy)
        i = loop_i(1-OLx, sNx+OLx)
        rViscFluxV = rViscFluxV.at[i, j].set(0.)
    else:                                                           # :54-66
        j = loop_j(1-OLy, sNy+OLy-1)
        i = loop_i(1-OLx, sNx+OLx-1)
        rViscFluxV = rViscFluxV.at[i, j].set(
            -KappaRV[i, j, k]
            *rAs[i, j]*deepFac2F[k]*rhoFacF[k]
            *(vFld[i, j, k]-vFld[i, j, k-1]
              )*rkSign*recip_drC[k]
            *maskS[i, j, k]
            *maskS[i, j, k-1])
    return rViscFluxV
