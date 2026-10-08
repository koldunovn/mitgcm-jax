"""GAD_DST2U1_IMPL_R: pkg/generic_advdiff/gad_dst2u1_impl_r.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.pkg.generic_advdiff.gad_h import ENUM_DST2


def gad_dst2u1_impl_r(k, iMin, iMax, jMin, jMax, advectionScheme, deltaTarg, rTrans, recip_hFac, a3d, b3d, c3d, *,
                      cfg, grid, params):
    """GAD_DST2U1_IMPL_R( bi,bj,k, iMin,iMax,jMin,jMax, advectionScheme, deltaTarg, rTrans, recip_hFac,
                          a3d, b3d, c3d, myThid )
    @63cdc0b pkg/generic_advdiff/gad_dst2u1_impl_r.F:6-109

    C     Compute matrix element to solve vertical advection implicitly
    C     using DST 2nd.Order (=Lax-Wendroff) or 1rst Order Upwind scheme.
    C     Method:
    C      contribution of vertical transport at interface k is added
    C      to matrix lines k and k-1.
    C     k            :: vertical level
    C  advectionScheme :: advection scheme to use: either 2nd Order DST or 1rst Order Upwind
    C     iMin,iMax    :: computation domain
    C     jMin,jMax    :: computation domain
    C     deltaTarg    :: time step
    C     rTrans       :: vertical volume transport
    C     recip_hFac   :: inverse of cell open-depth factor
    C     a3d          :: lower diagonal of the tridiagonal matrix
    C     b3d          :: main  diagonal of the tridiagonal matrix
    C     c3d          :: upper diagonal of the tridiagonal matrix

    Returns (a3d, b3d, c3d) (updated at levels k and k-1, :86-101). `k`, `iMin..jMax` and `advectionScheme` are static
    Python ints (GAD_IMPLICIT_R passes 1, sNx, 1, sNy and its b5d, c5d, d5d as a3d, b3d, c3d: gad_implicit_r.F
    :207-211). `deltaTarg` is declared (Nr); `rTrans` (1-OLx:sNx+OLx,1-OLy:sNy+OLy); `recip_hFac` and the matrices
    (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr). `cfg`: the GAD kernels' static configuration (gad_calc_rhs.GadKernelCfg: Nr).
    GRID.h: recip_rA, recip_drF, recip_deepFac2C, rkSign; PARAMS.h: recip_rhoFacC (`params`). The point loop
    (:77-103) runs on the whole iMin..iMax x jMin..jMax range at once (each point writes its own matrix entries;
    levels k and k-1 are distinct). The commented-out w_CFL lines (:79-81, :85) are not ported. `0. _d 0`,
    `1. _d 0`, `0.5 _d 0` are doubles; ABS is exact."""
    Nr = cfg.Nr
    g = grid
    recip_rhoFacC = params.recip_rhoFacC
    rLimit = 0.                                                                 # :69  0. _d 0
    if advectionScheme == ENUM_DST2:
        rLimit = 1.                                                             # :70  1. _d 0
    if k > 1 and k <= Nr:                                                       # :73
        # :76 deltaTcfl = deltaTarg(k): read only by the commented-out w_CFL lines (:79-81, :85)
        j = loop_j(jMin, jMax)                                                  # :77-78
        i = loop_i(iMin, iMax)                                                  # :78
        rCenter = 0.5*rTrans[i, j]*g.recip_rA[i, j]*g.rkSign                    # :82  0.5 _d 0
        rUpwind = (jnp.abs(rCenter)                                             # :83-84
                   * (1. - rLimit))
        a3d = a3d.at[i, j, k].set(a3d[i, j, k]                                  # :86-89
                                  - (rCenter+rUpwind)*deltaTarg[k]
                                  * recip_hFac[i, j, k]*g.recip_drF[k]
                                  * g.recip_deepFac2C[k]*recip_rhoFacC[k])
        b3d = b3d.at[i, j, k].set(b3d[i, j, k]                                  # :90-93
                                  - (rCenter-rUpwind)*deltaTarg[k]
                                  * recip_hFac[i, j, k]*g.recip_drF[k]
                                  * g.recip_deepFac2C[k]*recip_rhoFacC[k])
        b3d = b3d.at[i, j, k-1].set(b3d[i, j, k-1]                              # :94-97
                                    + (rCenter+rUpwind)*deltaTarg[k-1]
                                    * recip_hFac[i, j, k-1]*g.recip_drF[k-1]
                                    * g.recip_deepFac2C[k-1]*recip_rhoFacC[k-1])
        c3d = c3d.at[i, j, k-1].set(c3d[i, j, k-1]                              # :98-101
                                    + (rCenter-rUpwind)*deltaTarg[k-1]
                                    * recip_hFac[i, j, k-1]*g.recip_drF[k-1]
                                    * g.recip_deepFac2C[k-1]*recip_rhoFacC[k-1])
    return a3d, b3d, c3d
