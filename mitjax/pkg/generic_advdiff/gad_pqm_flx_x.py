"""pkg/generic_advdiff/gad_pqm_flx_x.F: PQM fluxes on grid-cell edges in X (GAD_PQM_FLX_X).

Row routine (see gad_osc_hat_x.py): the row arrays carry the row index j and `iy` is the caller's row loop.
"""

import jax.numpy as jnp

from mitjax.farray import loop_i
from mitjax.ops.fortran_minmax import MAX


def gad_pqm_flx_x(kk, iy, calc_CFL, delT, uvel, ufac, fhat, flux, *, cfg, grid):
    """GAD_PQM_FLX_X(bi,bj,kk,iy, calc_CFL, delT, uvel, ufac, fhat, flux, myThid)
    @63cdc0b pkg/generic_advdiff/gad_pqm_flx_x.F:3-153

    C     | PQM_FLX_X: evaluate PQM flux on grid-cell edges.               |
    C       calc_CFL  :: TRUE to calc. CFL from vel.
    C       delT      :: time-step.
    C       uvel      :: vel.-comp in x-direction.
    C       ufac      :: vel.-flux in x-direction.
    C       fhat      :: row of poly. coeff.
    C       flux      :: adv.-flux in x-direction.

    Returns flux. As GAD_PPM_FLX_X (gad_ppm_flx_x.py): the point loop (:63-148) on all its points at once, both
    branches of the IF on the sign of uvel (:71-137) evaluated and selected, then the zero-velocity IF (:65-67);
    the divisor of :141 has magnitude >= 1.d-20. `ss11 ** n` (n = 2..5) are integer powers (lax.integer_pow;
    libgcc's __powidf2 for n >= 3, x*x for n = 2).
    """
    sNx, OLx = cfg.sNx, cfg.OLx
    recip_dxF, recip_deepFacC = grid.recip_dxF, grid.recip_deepFacC

    ix = loop_i(1-OLx+4, sNx+OLx-3)                                     # :63

#     ==================== integrate PQM profile over upwind cell IX-1   (:71-102, uvel > 0)
    if calc_CFL:                                                        # :74-80
        uCFL_p = (uvel[ix, iy] * delT
                  * recip_dxF[ix-1, iy]
                  * recip_deepFacC[kk])
    else:
        uCFL_p = uvel[ix, iy]

    ss11 = +1. - 2. * uCFL_p                                            # :82-83
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

    intF_p = (ivec[1] * fhat[1][ix-1, iy]                               # :96-100
              + ivec[2] * fhat[2][ix-1, iy]
              + ivec[3] * fhat[3][ix-1, iy]
              + ivec[4] * fhat[4][ix-1, iy]
              + ivec[5] * fhat[5][ix-1, iy])

#     ==================== integrate PQM profile over upwind cell IX+0   (:104-135, uvel < 0)
    if calc_CFL:                                                        # :107-113
        uCFL_m = (uvel[ix, iy] * delT
                  * recip_dxF[ix-0, iy]
                  * recip_deepFacC[kk])
    else:
        uCFL_m = uvel[ix, iy]

    ss11 = -1. - 2. * uCFL_m                                            # :115-116
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

    intF_m = (ivec[1] * fhat[1][ix-0, iy]                               # :129-133
              + ivec[2] * fhat[2][ix-0, iy]
              + ivec[3] * fhat[3][ix-0, iy]
              + ivec[4] * fhat[4][ix-0, iy]
              + ivec[5] * fhat[5][ix-0, iy])

    pos = uvel[ix, iy] > 0.                                             # :71
    uCFL = jnp.where(pos, uCFL_p, uCFL_m)
    intF = jnp.where(pos, intF_p, intF_m)

#-    to avoid potential underflow:
    intF = 0.5 * intF / jnp.copysign(MAX(jnp.abs(uCFL), 1.e-20, p="b"), uCFL)   # :141

#     ==================== calc. flux = upwind tracer * face-transport
    flux = flux.at[ix, iy].set(jnp.where(uvel[ix, iy] == 0.,            # :65-67, :144
                                         0., + ufac[ix, iy] * intF))
    return flux
