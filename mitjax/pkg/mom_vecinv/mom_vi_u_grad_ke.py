"""pkg/mom_vecinv/mom_vi_u_grad_ke.F: zonal gradient of the kinetic energy (Bernoulli term) (MOM_VI_U_GRAD_KE)."""

from mitjax.farray import loop_i, loop_j


def mom_vi_u_grad_ke(k, KE, dKEdx, *, cfg, grid):
    """MOM_VI_U_GRAD_KE(bi,bj,k, KE, dKEdx, myThid)   @63cdc0b pkg/mom_vecinv/mom_vi_u_grad_ke.F:3-36

    C     | S/R MOM_U_GRAD_KE

    Returns dKEdx (points outside DO j=1-OLy,sNy+OLy / DO i=2-OLx,sNx+OLx keep their prior values). The point loop
    runs on the whole (i,j) range.
    """
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    recip_dxC, maskW, recip_deepFacC = grid.recip_dxC, grid.maskW, grid.recip_deepFacC

    j = loop_j(1-OLy, sNy+OLy)                                      # :28-33
    i = loop_i(2-OLx, sNx+OLx)
    dKEdx = dKEdx.at[i, j].set(-recip_dxC[i, j]*(KE[i, j]-KE[i-1, j])
                               *maskW[i, j, k]*recip_deepFacC[k])
    return dKEdx
