"""pkg/mom_common/mom_u_rviscflux.F: vertical viscous flux of U at interface k (MOM_U_RVISCFLUX)."""

from mitjax.farray import loop_i, loop_j


def mom_u_rviscflux(k, uFld, KappaRU, rViscFluxU, *, cfg, grid, params):
    """MOM_U_RVISCFLUX(bi,bj,k, uFld, KappaRU, rViscFluxU, myThid)   @63cdc0b pkg/mom_common/mom_u_rviscflux.F:3-69

    C Calculates the area integrated vertical viscous fluxes of U
    C  at vertical interface k (between level k & k-1):
    C F^r = - \\frac{ {\\cal A}_w }{\\Delta r_c} A_r \\delta_k u
    C  k                    :: vertical level
    C  uFld                 :: zonal flow
    C  KappaRU              :: vertical viscosity
    C  rViscFluxU           :: viscous fluxes

    Returns rViscFluxU. `k` is static (the caller passes k and k+1, :48 selects the branch). uFld is the 3-D field
    `uFld(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy)`; KappaRU is declared `(..,Nr)` here and `(..,Nr+1)` by the caller
    (MOM_FLUXFORM): the routine reads levels 2..Nr only, valid in either declaration. `_maskW` is maskW (no
    ALLOW_DEPTH_CONTROL). The point loop runs on the whole (i,j) range (each point reads inputs only).
    """
    sNx, sNy, OLx, OLy, Nr = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy, cfg.size.Nr
    rAw, deepFac2F, recip_drC, maskW, rkSign = grid.rAw, grid.deepFac2F, grid.recip_drC, grid.maskW, grid.rkSign
    rhoFacF = params.rhoFacF

#     - Vertical viscous flux
    if k <= 1 or k > Nr:                                            # :48-53
        j = loop_j(1-OLy, sNy+OLy)
        i = loop_i(1-OLx, sNx+OLx)
        rViscFluxU = rViscFluxU.at[i, j].set(0.)
    else:                                                           # :54-66
        j = loop_j(1-OLy, sNy+OLy-1)
        i = loop_i(1-OLx, sNx+OLx-1)
        rViscFluxU = rViscFluxU.at[i, j].set(
            -KappaRU[i, j, k]
            *rAw[i, j]*deepFac2F[k]*rhoFacF[k]
            *(uFld[i, j, k]-uFld[i, j, k-1]
              )*rkSign*recip_drC[k]
            *maskW[i, j, k]
            *maskW[i, j, k-1])
    return rViscFluxU
