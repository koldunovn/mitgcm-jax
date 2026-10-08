"""pkg/mom_vecinv/mom_vi_v_vertshear.F: vertical shear term -w dv/dr at V points (MOM_VI_V_VERTSHEAR)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j


def mom_vi_v_vertshear(k, deepFacA, vFld, wFld, vShearTerm, *, cfg, grid, params):
    """MOM_VI_V_VERTSHEAR(bi, bj, k, deepFacA, vFld, wFld, vShearTerm, myThid)
    @63cdc0b pkg/mom_vecinv/mom_vi_v_vertshear.F:7-140

    C     | S/R MOM_V_VERTSHEAR
    C  deepFacA             :: deep-model grid factor at level center

    Returns vShearTerm. `k` is a Python int; kp1, km1, mask_Kp1, mask_Km1 (:49-54) are trace-time values (oneRL,
    zeroRL exact; MIN/MAX of integers). vFld and wFld are 3-D fields (Nr levels). selectKEscheme and upwindShear
    (PARAMS.h) are static; rAdvAreaWeight (:44-47) follows selectKEscheme. recip_rhoFacC (PARAMS.h) and rhoFacF
    are `params` arrays; the rest is GRID.h. The commented-out variants (c1/c2 lines) are not ported. The point
    loop runs on the whole (i,j) range.
    """
    sNx, sNy, OLx, OLy, Nr = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy, cfg.size.Nr
    rA, recip_rAs, maskC, recip_hFacS = grid.rA, grid.recip_rAs, grid.maskC, grid.recip_hFacS
    recip_drF, recip_deepFac2C, deepFac2F, rkSign = grid.recip_drF, grid.recip_deepFac2C, grid.deepFac2F, grid.rkSign
    recip_rhoFacC, rhoFacF = params.recip_rhoFacC, params.rhoFacF
    halfRL, oneRL, zeroRL = 0.5, 1.0, 0.0                           # EEPARAMS.h:72-73

    rAdvAreaWeight = True                                           # :44-47
    if params.selectKEscheme == 1 or params.selectKEscheme == 3:
        rAdvAreaWeight = False

    kp1 = min(k+1, Nr)                                              # :49; MINMAX-INT: integer level index
    mask_Kp1 = oneRL
    if k == Nr:
        mask_Kp1 = zeroRL
    km1 = max(k-1, 1)                                               # :52; MINMAX-INT: integer level index
    mask_Km1 = oneRL
    if k == 1:
        mask_Km1 = zeroRL

    recip_drDeepRho = (recip_drF[k]/deepFacA[k]                     # :56-57
                       * recip_deepFac2C[k]*recip_rhoFacC[k])

    j = loop_j(2-OLy, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx)
    if rAdvAreaWeight:                                              # :69-82
#     Transport at interface k : Area weighted average
        wBarYm = (halfRL*(
            wFld[i, j, k]*rA[i, j]*maskC[i, j, km1]
            + wFld[i, j-1, k]*rA[i, j-1]*maskC[i, j-1, km1]
                          )*mask_Km1*deepFac2F[k]*rhoFacF[k]
                  *recip_rAs[i, j])
#     Transport at interface k+1 (here wFld is already masked)
        wBarYp = (halfRL*(
            wFld[i, j, kp1]*rA[i, j]
            + wFld[i, j-1, kp1]*rA[i, j-1]
                          )*mask_Kp1*deepFac2F[kp1]*rhoFacF[kp1]
                  *recip_rAs[i, j])
    else:                                                           # :83-95
#     Velocity at interface k : Simple average
        wBarYm = (halfRL*(
            wFld[i, j, k]*maskC[i, j, km1]
            + wFld[i, j-1, k]*maskC[i, j-1, km1]
                          )*mask_Km1*deepFac2F[k]*rhoFacF[k])
#     Velocity at interface k+1 (here wFld is already masked)
        wBarYp = (halfRL*(
            wFld[i, j, kp1]
            + wFld[i, j-1, kp1]
                          )*mask_Kp1*deepFac2F[kp1]*rhoFacF[kp1])

#     Shear at interface k
    vZm = (vFld[i, j, k]*deepFacA[k]                                # :99-100
           - vFld[i, j, km1]*deepFacA[km1]*mask_Km1)*rkSign
#     Shear at interface k+1
    vZp = (vFld[i, j, kp1]*deepFacA[kp1]*mask_Kp1                   # :107-108
           - vFld[i, j, k]*deepFacA[k])*rkSign

    if params.upwindShear:                                          # :127-135
        vShearTerm = vShearTerm.at[i, j].set(
            -halfRL*
            ((wBarYp*vZp + wBarYm*vZm)
             + (jnp.abs(wBarYp)*vZp - jnp.abs(wBarYm)*vZm)
             )*recip_hFacS[i, j, k]*recip_drDeepRho)
    else:
        vShearTerm = vShearTerm.at[i, j].set(
            -halfRL*(wBarYp*vZp + wBarYm*vZm)
            *recip_hFacS[i, j, k]*recip_drDeepRho)
    return vShearTerm
