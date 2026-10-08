"""pkg/generic_advdiff/gad_ppm_flx_x.F: PPM fluxes on grid-cell edges in X (GAD_PPM_FLX_X).

Row routine (see gad_osc_hat_x.py): the row arrays carry the row index j and `iy` is the caller's row loop.
"""

import jax.numpy as jnp

from mitjax.farray import loop_i
from mitjax.ops.fortran_minmax import MAX


def gad_ppm_flx_x(kk, iy, calc_CFL, delT, uvel, ufac, fhat, flux, *, cfg, grid):
    """GAD_PPM_FLX_X(bi,bj,kk,iy, calc_CFL, delT, uvel, ufac, fhat, flux, myThid)
    @63cdc0b pkg/generic_advdiff/gad_ppm_flx_x.F:3-141

    C     | PPM_FLX_X: evaluate PPM flux on grid-cell edges.               |
    C       calc_CFL  :: TRUE to calc. CFL from vel.
    C       delT      :: time-step.
    C       uvel      :: vel.-comp in x-direction.
    C       ufac      :: vel.-flux in x-direction.
    C       fhat      :: row of poly. coeff.
    C       flux      :: adv.-flux in x-direction.

    Returns flux. `calc_CFL` is static; delT is a traced float. The point loop (:63-136) runs on all its points at
    once (each point reads only inputs); both branches of the IF on the sign of uvel (:71-125) are evaluated and the
    one the Fortran takes is selected, then the zero-velocity IF (:65-67). The division of :129 is finite on every
    lane (its divisor has magnitude >= 1.d-20). `ss22 ** n` with ss22 = +-1 is exact; `ss11 ** n` is an integer
    power (lax.integer_pow, libgcc's __powidf2 for n = 3; n = 2 is x*x in both). SIGN(a,b) with a >= 0 is
    copysign(a,b).
    """
    sNx, OLx = cfg.sNx, cfg.OLx
    recip_dxF, recip_deepFacC = grid.recip_dxF, grid.recip_deepFacC

    ix = loop_i(1-OLx+3, sNx+OLx-2)                                     # :63

#     ==================== integrate PPM profile over upwind cell IX-1   (:71-96, uvel > 0)
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
    ivec[1] = ss22 - ss11                                               # :86-90
    ivec[2] = (ss22 ** 2
               - ss11 ** 2)*(1. / 2.)
    ivec[3] = (ss22 ** 3
               - ss11 ** 3)*(1. / 3.)

    intF_p = (ivec[1] * fhat[1][ix-1, iy]                               # :92-94
              + ivec[2] * fhat[2][ix-1, iy]
              + ivec[3] * fhat[3][ix-1, iy])

#     ==================== integrate PPM profile over upwind cell IX+0   (:98-123, uvel < 0)
    if calc_CFL:                                                        # :101-107
        uCFL_m = (uvel[ix, iy] * delT
                  * recip_dxF[ix-0, iy]
                  * recip_deepFacC[kk])
    else:
        uCFL_m = uvel[ix, iy]

    ss11 = -1. - 2. * uCFL_m                                            # :109-110
    ss22 = -1.

#     ==================== integrate profile over region swept by face
    ivec = {}
    ivec[1] = ss22 - ss11                                               # :113-117
    ivec[2] = (ss22 ** 2
               - ss11 ** 2)*(1. / 2.)
    ivec[3] = (ss22 ** 3
               - ss11 ** 3)*(1. / 3.)

    intF_m = (ivec[1] * fhat[1][ix-0, iy]                               # :119-121
              + ivec[2] * fhat[2][ix-0, iy]
              + ivec[3] * fhat[3][ix-0, iy])

    pos = uvel[ix, iy] > 0.                                             # :71
    uCFL = jnp.where(pos, uCFL_p, uCFL_m)
    intF = jnp.where(pos, intF_p, intF_m)

#-    to avoid potential underflow:
    intF = 0.5 * intF / jnp.copysign(MAX(jnp.abs(uCFL), 1.e-20, p="b"), uCFL)   # :129

#     ==================== calc. flux = upwind tracer * face-transport
    flux = flux.at[ix, iy].set(jnp.where(uvel[ix, iy] == 0.,            # :65-67, :132
                                         0., + ufac[ix, iy] * intF))
    return flux
