"""pkg/mom_vecinv/mom_vi_v_grad_ke.F: meridional gradient of the kinetic energy (Bernoulli term)
(MOM_VI_V_GRAD_KE)."""

from mitjax.farray import loop_i, loop_j


def mom_vi_v_grad_ke(k, KE, dKEdy, *, cfg, grid):
    """MOM_VI_V_GRAD_KE(bi,bj,k, KE, dKEdy, myThid)   @63cdc0b pkg/mom_vecinv/mom_vi_v_grad_ke.F:3-36

    C     | S/R MOM_V_GRAD_KE

    Returns dKEdy (points outside DO j=2-OLy,sNy+OLy / DO i=1-OLx,sNx+OLx keep their prior values). The point loop
    runs on the whole (i,j) range.
    """
    sNx, sNy, OLx, OLy = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy
    recip_dyC, maskS, recip_deepFacC = grid.recip_dyC, grid.maskS, grid.recip_deepFacC

    j = loop_j(2-OLy, sNy+OLy)                                      # :28-33
    i = loop_i(1-OLx, sNx+OLx)
    dKEdy = dKEdy.at[i, j].set(-recip_dyC[i, j]*(KE[i, j]-KE[i, j-1])
                               *maskS[i, j, k]*recip_deepFacC[k])
    return dKEdy
