"""GAD_FLUXLIMIT_IMPL_R: tridiagonal matrix of implicit vertical advection with a flux limiter
(@63cdc0b pkg/generic_advdiff/gad_fluxlimit_impl_r.F)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.ops.fortran_minmax import MAX
from mitjax.ops.safe import safe_div
from mitjax.pkg.generic_advdiff.gad_flux_limiter_h import Limiter

# eesupp/inc/EEPARAMS.h:72   PARAMETER ( zeroRL = 0.0 _d 0 , oneRL  = 1.0 _d 0 )
zeroRL = 0.0
oneRL = 1.0
# gad_fluxlimit_impl_r.F:69-70   _RL CrMax; PARAMETER( CrMax = 1.D+6 )
CrMax = 1.0e+6


def gad_fluxlimit_impl_r(k, iMin, iMax, jMin, jMax, deltaTarg, rTrans, recip_hFac, tFld, a3d, b3d, c3d, *, cfg,
                         grid, params):
    """GAD_FLUXLIMIT_IMPL_R( bi,bj,k, iMin,iMax,jMin,jMax, deltaTarg, rTrans, recip_hFac, tFld, a3d, b3d, c3d,
                             myThid )
    @63cdc0b pkg/generic_advdiff/gad_fluxlimit_impl_r.F:6-139

    C     Compute matrix element to solve vertical advection implicitly
    C      using flux--limiter advection scheme.
    C     Method:
    C      contribution of vertical transport at interface k is added
    C      to matrix lines k and k-1.
    C     k            :: vertical level
    C     iMin, iMax   :: computation domain
    C     jMin, jMax   :: computation domain
    C     deltaTarg    :: time step
    C     rTrans       :: vertical volume transport
    C     recip_hFac   :: inverse of cell open-depth factor
    C     tFld         :: tracer field
    C     a3d          :: lower diagonal of the tridiagonal matrix
    C     b3d          :: main  diagonal of the tridiagonal matrix
    C     c3d          :: upper diagonal of the tridiagonal matrix

    Returns (a3d, b3d, c3d) (all are updated: :116-131). `k` and `iMin..jMax` are static Python ints (GAD_IMPLICIT_R
    passes 1, sNx, 1, sNy: gad_implicit_r.F:78-79; it passes its b5d, c5d, d5d as a3d, b3d, c3d: :230-232).
    `deltaTarg` is declared (Nr); `recip_hFac`, `tFld` and the matrices (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr). GRID.h:
    recip_rA, maskC, recip_drC, recip_drF, recip_deepFac2C, recip_deepFac2F, rkSign; PARAMS.h: recip_rhoFacC,
    recip_rhoFacF (`params`). Limiter(Cr) is GAD_FLUX_LIMITER.h (gad_flux_limiter_h.py). The local upwindFac(i,j)
    (:66) is written (:105-107) before it is read (:115) on the same iMin..iMax x jMin..jMax points. Both point loops
    run on the whole (i,j) range at once (each point reads inputs or its own upwindFac and writes its own matrix
    entries; levels k and k-1 are distinct). The IFs (:93-97, :98-102) are `where`s with the division guarded before
    the operation (`safe_div`) [L-AD-1]; SIGN is `copysign` (gfortran's -fsign-zero default); MAX is
    mitjax.ops.fortran_minmax.MAX (gfortran's value on a tie); MAX and the limiter are differentiated as written
    [L-AD-7]. `1. _d 0`, `-1. _d 0`, `0.5 _d 0` are doubles.
    """
    Nr = cfg.Nr
    recip_rA, maskC, recip_drC, recip_drF = grid.recip_rA, grid.maskC, grid.recip_drC, grid.recip_drF
    recip_deepFac2C, recip_deepFac2F, rkSign = grid.recip_deepFac2C, grid.recip_deepFac2F, grid.rkSign
    recip_rhoFacC, recip_rhoFacF = params.recip_rhoFacC, params.recip_rhoFacF

    km2 = max(1, k-2)                                               # :76-78; MINMAX-INT: integer (no tie or NaN case)
    km1 = max(1, k-1)  # MINMAX-INT: integer (no tie or NaN case)
    kp1 = min(Nr, k+1)  # MINMAX-INT: integer (no tie or NaN case)

#--   process interior interface only:
    if k > 1 and k <= Nr:                                           # :81

#--   Compute the upwind fraction:
        deltaTcfl = deltaTarg[k]                                    # :84
        upwindFac = rTrans.local("upwindFac")                       # :66  _RL upwindFac(1-OLx:sNx+OLx,1-OLy:sNy+OLy)
        j = loop_j(jMin, jMax)                                      # :85-109
        i = loop_i(iMin, iMax)
        w_CFL = (deltaTcfl*rTrans[i, j]*recip_rA[i, j]*recip_drC[k]
                           *recip_deepFac2F[k]*recip_rhoFacF[k])
        Rjp = (tFld[i, j, kp1]-tFld[i, j, k])*maskC[i, j, kp1]
        Rj = (tFld[i, j, k]-tFld[i, j, km1])
        Rjm = (tFld[i, j, km1]-tFld[i, j, km2])*maskC[i, j, km2]

        Cr = jnp.where(rTrans[i, j] < zeroRL, Rjm, Rjp)             # :93-97
        big = jnp.abs(Rj)*CrMax <= jnp.abs(Cr)                      # :98-102
        Cr = jnp.where(big, jnp.copysign(CrMax, Cr)*jnp.copysign(oneRL, Rj),
                       safe_div(Cr, Rj, ~big))

#        calculate upwind fraction using Limiter Function:
        upwindFac = upwindFac.at[i, j].set(1.                       # :105-107
                         - Limiter(Cr, p=("b", "a", "b", "a"))*(1. + jnp.abs(w_CFL)))   # :105-106
        upwindFac = upwindFac.at[i, j].set(MAX(-1., upwindFac[i, j], p="b"))  # :107

#--    Add centered & upwind contributions
        rCenter = 0.5*rTrans[i, j]*recip_rA[i, j]*rkSign           # :112-133
        rUpwind = jnp.abs(rCenter)*upwindFac[i, j]
        a3d = a3d.at[i, j, k].set(a3d[i, j, k]
                         - (rCenter+rUpwind)*deltaTarg[k]
                          *recip_hFac[i, j, k]*recip_drF[k]
                          *recip_deepFac2C[k]*recip_rhoFacC[k])
        b3d = b3d.at[i, j, k].set(b3d[i, j, k]
                         - (rCenter-rUpwind)*deltaTarg[k]
                          *recip_hFac[i, j, k]*recip_drF[k]
                          *recip_deepFac2C[k]*recip_rhoFacC[k])
        b3d = b3d.at[i, j, k-1].set(b3d[i, j, k-1]
                         + (rCenter+rUpwind)*deltaTarg[k-1]
                          *recip_hFac[i, j, k-1]*recip_drF[k-1]
                          *recip_deepFac2C[k-1]*recip_rhoFacC[k-1])
        c3d = c3d.at[i, j, k-1].set(c3d[i, j, k-1]
                         + (rCenter-rUpwind)*deltaTarg[k-1]
                          *recip_hFac[i, j, k-1]*recip_drF[k-1]
                          *recip_deepFac2C[k-1]*recip_rhoFacC[k-1])

#--   process interior interface only: end
    return a3d, b3d, c3d
