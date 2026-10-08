"""GAD_U3C4_IMPL_R: pentadiagonal matrix of implicit vertical advection, 3rd-order upwind / DST3 / 4th-order centered
(@63cdc0b pkg/generic_advdiff/gad_u3c4_impl_r.F)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.pkg.generic_advdiff.gad_h import ENUM_CENTERED_4TH, ENUM_DST3, oneSixth


def gad_u3c4_impl_r(k, iMin, iMax, jMin, jMax, advectionScheme, deltaTarg, rTrans, recip_hFac,
                    a5d, b5d, c5d, d5d, e5d, *, cfg, grid, params):
    """GAD_U3C4_IMPL_R( bi,bj,k, iMin,iMax,jMin,jMax, advectionScheme, deltaTarg, rTrans, recip_hFac,
                        a5d, b5d, c5d, d5d, e5d, myThid )
    @63cdc0b pkg/generic_advdiff/gad_u3c4_impl_r.F:6-201

    C     Compute matrix element to solve vertical advection implicitly
    C      using 3rd order upwind advection scheme,
    C         or 3rd order Direct Space and Time advection scheme,
    C         or 4th order Centered advection scheme.
    C     Method:
    C      contribution of vertical transport at interface k is added
    C      to matrix lines k and k-1
    C     k            :: vertical level
    C     iMin,iMax    :: computation domain
    C     jMin,jMax    :: computation domain
    C  advectionScheme :: advection scheme to use
    C     deltaTarg    :: time step
    C     rTrans       :: vertical volume transport
    C     recip_hFac   :: inverse of cell open-depth factor
    C     a5d          :: 2nd  lower diag of pentadiagonal matrix
    C     b5d          :: 1rst lower diag of pentadiagonal matrix
    C     c5d          :: main diag       of pentadiagonal matrix
    C     d5d          :: 1rst upper diag of pentadiagonal matrix
    C     e5d          :: 2nd  upper diag of pentadiagonal matrix

    Returns (a5d, b5d, c5d, d5d, e5d) (all are updated: :154-193). `k`, `iMin..jMax` and `advectionScheme` are
    static Python ints (GAD_IMPLICIT_R passes iMin = 1, iMax = sNx, jMin = 1, jMax = sNy: gad_implicit_r.F:78-79).
    `deltaTarg` is declared (Nr), `recip_hFac` and the matrices (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr). GRID.h: recip_rA,
    maskC, recip_drC, recip_drF, recip_deepFac2C, recip_deepFac2F, rkSign; PARAMS.h: recip_rhoFacC, recip_rhoFacF
    (`params`). The point loop body (:132-193) runs on the whole (i,j) range at once (each point reads inputs and
    writes its own matrix entries; levels k and k-1 are distinct).
    Ported: advectionScheme = ENUM_UPWIND_3RD (advect_xz/input.nlfs: saltAdvScheme=3, the only M1 caller), i.e. the
    ELSE branch :148-151; GO lane (global_ocean.cs32x15/input_ad, tempAdvScheme = saltAdvScheme = 30): the ENUM_DST3
    branch :139-147 (GRID.h recip_drC, recip_deepFac2F; PARAMS.h recip_rhoFacF). Not ported (raise): the
    ENUM_CENTERED_4TH branch :135-138 (when flagC4), and the `(ALLOW_AUTODIFF && TARGET_NEC_SX)`
    variant :96-130 (no M1 build defines TARGET_NEC_SX). `0.5 _d 0`, `1. _d 0`, `0. _d 0`, `2. _d 0` are doubles;
    oneSixth is GAD.h:97.
    """
    if cfg.ALLOW_AUTODIFF and cfg.TARGET_NEC_SX:
        raise NotImplementedError("GAD_U3C4_IMPL_R: the (ALLOW_AUTODIFF && TARGET_NEC_SX) variant is not ported")
    Nr = cfg.Nr
    recip_rA, maskC, recip_drF = grid.recip_rA, grid.maskC, grid.recip_drF
    recip_deepFac2C, rkSign = grid.recip_deepFac2C, grid.rkSign
    recip_rhoFacC = params.recip_rhoFacC

#--   process interior interface only:
    if k > 1 and k <= Nr:                                           # :83

        km2 = max(1, k-2)                                           # :85-92; MINMAX-INT: integer (no tie or NaN case)
        kp1 = min(Nr, k+1)  # MINMAX-INT: integer (no tie or NaN case)
        maskP1 = 1.
        maskM2 = 1.
        if k <= 2:
            maskM2 = 0.
        if k >= Nr:
            maskP1 = 0.
        flagC4 = (advectionScheme == ENUM_CENTERED_4TH
                  and k > 2 and k < Nr)

#--    Add centered, upwind and high-order contributions
        deltaTcfl = deltaTarg[k]                                    # :95  (read only by the DST3 branch)
        j = loop_j(jMin, jMax)                                      # :124-195
        i = loop_i(iMin, iMax)
        rCenter = 0.5*rTrans[i, j]*recip_rA[i, j]*rkSign           # :132-134
        mskM = maskC[i, j, km2]*maskM2
        mskP = maskC[i, j, kp1]*maskP1
        if flagC4:                                                  # :135-138
            raise NotImplementedError("GAD_U3C4_IMPL_R: the ENUM_CENTERED_4TH branch (flagC4) is not ported")
        elif advectionScheme == ENUM_DST3:                          # :139-147 (GO lane: cs32x15/input_ad)
            wCFL = (deltaTcfl*jnp.abs(rTrans[i, j])                 # :140-142
                    * recip_rA[i, j]*grid.recip_drC[k]
                    * grid.recip_deepFac2F[k]*params.recip_rhoFacF[k])
            rHigh = (1. - wCFL*wCFL)*oneSixth                       # :143  1. _d 0
            rUpwind = (2.*rHigh)*jnp.abs(rCenter)                   # :145  2. _d 0
            rC4km = rHigh*(rCenter+jnp.abs(rCenter))*mskM           # :146
            rC4kp = rHigh*(rCenter-jnp.abs(rCenter))*mskP           # :147
        else:                                                       # :148-151
            rUpwind = 2.*oneSixth*jnp.abs(rCenter)
            rC4km = oneSixth*(rCenter+jnp.abs(rCenter))*mskM
            rC4kp = oneSixth*(rCenter-jnp.abs(rCenter))*mskP

        a5d = a5d.at[i, j, k].set(a5d[i, j, k]                      # :154-158
                         + rC4km
                          *deltaTarg[k]
                          *recip_hFac[i, j, k]*recip_drF[k]
                          *recip_deepFac2C[k]*recip_rhoFacC[k])
        b5d = b5d.at[i, j, k].set(b5d[i, j, k]                      # :159-163
                         - ((rCenter+rUpwind) + rC4km)
                          *deltaTarg[k]
                          *recip_hFac[i, j, k]*recip_drF[k]
                          *recip_deepFac2C[k]*recip_rhoFacC[k])
        c5d = c5d.at[i, j, k].set(c5d[i, j, k]                      # :164-168
                         - ((rCenter-rUpwind) + rC4kp)
                          *deltaTarg[k]
                          *recip_hFac[i, j, k]*recip_drF[k]
                          *recip_deepFac2C[k]*recip_rhoFacC[k])
        d5d = d5d.at[i, j, k].set(d5d[i, j, k]                      # :169-173
                         + rC4kp
                          *deltaTarg[k]
                          *recip_hFac[i, j, k]*recip_drF[k]
                          *recip_deepFac2C[k]*recip_rhoFacC[k])
        b5d = b5d.at[i, j, k-1].set(b5d[i, j, k-1]                  # :174-178
                         - rC4km
                          *deltaTarg[k-1]
                          *recip_hFac[i, j, k-1]*recip_drF[k-1]
                          *recip_deepFac2C[k-1]*recip_rhoFacC[k-1])
        c5d = c5d.at[i, j, k-1].set(c5d[i, j, k-1]                  # :179-183
                         + ((rCenter+rUpwind) + rC4km)
                          *deltaTarg[k-1]
                          *recip_hFac[i, j, k-1]*recip_drF[k-1]
                          *recip_deepFac2C[k-1]*recip_rhoFacC[k-1])
        d5d = d5d.at[i, j, k-1].set(d5d[i, j, k-1]                  # :184-188
                         + ((rCenter-rUpwind) + rC4kp)
                          *deltaTarg[k-1]
                          *recip_hFac[i, j, k-1]*recip_drF[k-1]
                          *recip_deepFac2C[k-1]*recip_rhoFacC[k-1])
        e5d = e5d.at[i, j, k-1].set(e5d[i, j, k-1]                  # :189-193
                         - rC4kp
                          *deltaTarg[k-1]
                          *recip_hFac[i, j, k-1]*recip_drF[k-1]
                          *recip_deepFac2C[k-1]*recip_rhoFacC[k-1])

#--   process interior interface only: end
    return a5d, b5d, c5d, d5d, e5d
