"""pkg/generic_advdiff/gad_pqm_flx_y.F: PQM fluxes on grid-cell edges in Y (GAD_PQM_FLX_Y).

Row routine (see gad_osc_hat_y.py): the row arrays (along j) carry the index i of the caller's `do ix` loop, and `ix` is that loop.
"""

import jax.numpy as jnp

from mitjax.farray import loop_j
from mitjax.ops.fortran_minmax import MAX


def gad_pqm_flx_y(kk, ix, calc_CFL, delT, vvel, vfac, fhat, flux, *, cfg, grid):
    """GAD_PQM_FLX_Y(bi,bj,kk,ix, calc_CFL, delT, vvel, vfac, fhat, flux, myThid)
    @63cdc0b pkg/generic_advdiff/gad_pqm_flx_y.F:3-153

    C     | PQM_FLX_Y: evaluate PQM flux on grid-cell edges.               |
    C       calc_CFL  :: TRUE to calc. CFL from vel.
    C       delT      :: time-step.
    C       vvel      :: vel.-comp in y-direction.
    C       vfac      :: vel.-flux in y-direction.
    C       fhat      :: row of poly. coeff.
    C       flux      :: adv.-flux in y-direction.

    Returns flux. As GAD_PPM_FLX_Y (gad_ppm_flx_y.py): the point loop (:63-148) on all its points at once, both
    branches of the IF on the sign of vvel (:71-137) evaluated and selected, then the zero-velocity IF (:65-67);
    the divisor of :141 has magnitude >= 1.d-20. `ss11 ** n` (n = 2..5) are integer powers (lax.integer_pow;
    libgcc's __powidf2 for n >= 3, x*x for n = 2).
    """
    sNy, OLy = cfg.sNy, cfg.OLy
    recip_dyF, recip_deepFacC = grid.recip_dyF, grid.recip_deepFacC

    iy = loop_j(1-OLy+4, sNy+OLy-3)                                     # :63

#     ==================== integrate PQM profile over upwind cell IY-1   (:71-102, vvel > 0)
    if calc_CFL:                                                        # :74-80
        vCFL_p = (vvel[ix, iy] * delT
                  * recip_dyF[ix, iy-1]
                  * recip_deepFacC[kk])
    else:
        vCFL_p = vvel[ix, iy]

    ss11 = +1. - 2. * vCFL_p                                            # :82-83
    ss22 = +1.

#     ==================== integrate profile over region swept by face
    ivec = {}
    ivec[1] = ss22 - ss11                                               # :86-94
    ivec[2] = (ss22 ** 2
               - ss11 ** 2)*(1. / 2.)
    ivec[3] = (ss22 ** 3
               - ss11 ** 3)*(1. / 3.)
    ivec[4] = (ss22 ** 4
               - ss11 ** 4)*(1. / 4.)
    ivec[5] = (ss22 ** 5
               - ss11 ** 5)*(1. / 5.)

    intF_p = (ivec[1] * fhat[1][ix, iy-1]                               # :96-100
              + ivec[2] * fhat[2][ix, iy-1]
              + ivec[3] * fhat[3][ix, iy-1]
              + ivec[4] * fhat[4][ix, iy-1]
              + ivec[5] * fhat[5][ix, iy-1])

#     ==================== integrate PQM profile over upwind cell IY+0   (:104-135, vvel < 0)
    if calc_CFL:                                                        # :107-113
        vCFL_m = (vvel[ix, iy] * delT
                  * recip_dyF[ix, iy-0]
                  * recip_deepFacC[kk])
    else:
        vCFL_m = vvel[ix, iy]

    ss11 = -1. - 2. * vCFL_m                                            # :115-116
    ss22 = -1.

#     ==================== integrate profile over region swept by face
    ivec = {}
    ivec[1] = ss22 - ss11                                               # :119-127
    ivec[2] = (ss22 ** 2
               - ss11 ** 2)*(1. / 2.)
    ivec[3] = (ss22 ** 3
               - ss11 ** 3)*(1. / 3.)
    ivec[4] = (ss22 ** 4
               - ss11 ** 4)*(1. / 4.)
    ivec[5] = (ss22 ** 5
               - ss11 ** 5)*(1. / 5.)

    intF_m = (ivec[1] * fhat[1][ix, iy-0]                               # :129-133
              + ivec[2] * fhat[2][ix, iy-0]
              + ivec[3] * fhat[3][ix, iy-0]
              + ivec[4] * fhat[4][ix, iy-0]
              + ivec[5] * fhat[5][ix, iy-0])

    pos = vvel[ix, iy] > 0.                                             # :71
    vCFL = jnp.where(pos, vCFL_p, vCFL_m)
    intF = jnp.where(pos, intF_p, intF_m)

#-    to avoid potential underflow:
    intF = 0.5 * intF / jnp.copysign(MAX(jnp.abs(vCFL), 1.e-20, p="b"), vCFL)   # :141

#     ==================== calc. flux = upwind tracer * face-transport
    flux = flux.at[ix, iy].set(jnp.where(vvel[ix, iy] == 0.,            # :65-67, :144
                                         0., + vfac[ix, iy] * intF))
    return flux
