"""pkg/generic_advdiff/gad_pqm_flx_r.F: PQM fluxes on grid-cell edges in R (GAD_PQM_FLX_R).

Column routine (see gad_osc_hat_r.py): the column arrays carry the column indices and `ix, iy` are the caller's
column loops.
"""

import jax.numpy as jnp

from mitjax.farray import loops_kji
from mitjax.ops.fortran_minmax import MAX


def gad_pqm_flx_r(ix, iy, delT, wvel, wfac, fhat, flux, *, cfg, grid):
    """GAD_PQM_FLX_R(bi,bj,ix,iy, delT, wvel, wfac, fhat, flux, myThid)
    @63cdc0b pkg/generic_advdiff/gad_pqm_flx_r.F:3-144

    C     | PQM_FLX_R: evaluate PQM flux on grid-cell edges.               |
    C       delT      :: time-step.
    C       wvel      :: vel.-comp in r-direction.
    C       wfac      :: vel.-flux in r-direction.
    C       fhat      :: col. of poly. coeff.
    C       flux      :: adv.-flux in r-direction.

    Returns flux. As GAD_PPM_FLX_R (gad_ppm_flx_r.py), with five coefficients; note that this routine tests the
    sign of wfac (:66, :72), not of wvel as GAD_PPM_FLX_R does.
    """
    Nr = cfg.Nr
    recip_drF = grid.recip_drF

    ir, iy, ix = loops_kji((+2, Nr), (iy.first, iy.last), (ix.first, ix.last))   # :64

#     ==================== integrate PQM profile over upwind cell IR-1   (:72-98, wfac < 0)
    wCFL_m = (wvel[ix, iy, ir]                                          # :75-76
              * delT[ir-1]*recip_drF[ir-1])

    ss11 = +1. + 2. * wCFL_m                                            # :78-79
    ss22 = +1.

#     ==================== integrate profile over region swept by face
    ivec = {}
    ivec[1] = ss22 - ss11                                               # :82-90
    ivec[2] = (ss22 ** 2
               - ss11 ** 2)*(1. / 2.)
    ivec[3] = (ss22 ** 3
               - ss11 ** 3)*(1. / 3.)
    ivec[4] = (ss22 ** 4
               - ss11 ** 4)*(1. / 4.)
    ivec[5] = (ss22 ** 5
               - ss11 ** 5)*(1. / 5.)

    intF_m = (ivec[1] * fhat[1][ix, iy, ir-1]                           # :92-96
              + ivec[2] * fhat[2][ix, iy, ir-1]
              + ivec[3] * fhat[3][ix, iy, ir-1]
              + ivec[4] * fhat[4][ix, iy, ir-1]
              + ivec[5] * fhat[5][ix, iy, ir-1])

#     ==================== integrate PQM profile over upwind cell IR+0   (:100-126, wfac > 0)
    wCFL_p = (wvel[ix, iy, ir]                                          # :103-104
              * delT[ir-0]*recip_drF[ir-0])

    ss11 = -1. + 2. * wCFL_p                                            # :106-107
    ss22 = -1.

#     ==================== integrate profile over region swept by face
    ivec = {}
    ivec[1] = ss22 - ss11                                               # :110-118
    ivec[2] = (ss22 ** 2
               - ss11 ** 2)*(1. / 2.)
    ivec[3] = (ss22 ** 3
               - ss11 ** 3)*(1. / 3.)
    ivec[4] = (ss22 ** 4
               - ss11 ** 4)*(1. / 4.)
    ivec[5] = (ss22 ** 5
               - ss11 ** 5)*(1. / 5.)

    intF_p = (ivec[1] * fhat[1][ix, iy, ir-0]                           # :120-124
              + ivec[2] * fhat[2][ix, iy, ir-0]
              + ivec[3] * fhat[3][ix, iy, ir-0]
              + ivec[4] * fhat[4][ix, iy, ir-0]
              + ivec[5] * fhat[5][ix, iy, ir-0])

    neg = wfac[ix, iy, ir] < 0.                                         # :72
    wCFL = jnp.where(neg, wCFL_m, wCFL_p)
    intF = jnp.where(neg, intF_m, intF_p)

#-    to avoid potential underflow:
    intF = -0.5 * intF / jnp.copysign(MAX(jnp.abs(wCFL), 1.e-20, p="b"), wCFL)   # :132

#     ==================== calc. flux = upwind tracer * face-transport
    flux = flux.at[ix, iy, ir].set(jnp.where(wfac[ix, iy, ir] == 0.,    # :66-68, :135
                                             0., + wfac[ix, iy, ir] * intF))
    return flux
