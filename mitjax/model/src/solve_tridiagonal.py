"""SOLVE_TRIDIAGONAL: model/src/solve_tridiagonal.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j, loops_kji
from mitjax.ops.safe import safe_div
from mitjax.ops.scan_k import level, scan_k, set_levels


def solve_tridiagonal(iMin, iMax, jMin, jMax, a3d, b3d, c3d, y3d, errCode, *, cfg):
    """SOLVE_TRIDIAGONAL( iMin,iMax, jMin,jMax, a3d, b3d, c3d, y3d, errCode, bi, bj, myThid )
    @63cdc0b model/src/solve_tridiagonal.F:7-302

    C     *==========================================================*
    C     | S/R SOLVE_TRIDIAGONAL
    C     | o Solve a tri-diagonal system A*X=Y (dimension Nr)
    C     *==========================================================*
    C     | o Used to solve implicitly vertical advection & diffusion
    C     *==========================================================*
    C     a3d     :: matrix lower diagnonal
    C     b3d     :: matrix main  diagnonal
    C     c3d     :: matrix upper diagnonal
    C     y3d     :: Y vector (R.H.S.);
    C     -- OUTPUT: --
    C     y3d     :: X = solution of A*X=Y
    C     errCode :: > 0 if singular matrix

    a3d, b3d, c3d, y3d: FArrays (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr). Returns (y3d, errCode) with errCode an int32
    array [tile] (one Fortran call per tile). Ported: the default branch (neither SOLVE_DIAGONAL_LOWMEMORY nor
    SOLVE_DIAGONAL_KINNER, :150, :226-295), which runs over the whole tile (iMin..jMax are not used there). The forward
    sweep (DO k=1,Nr) and the backward sweep (DO k=Nr,1,-1) are recursions in k: scan_k in the Fortran order. Where
    the Fortran tests a pivot (`.NE. 0. _d 0`), the division is guarded on that test (finite derivatives).
    """
    if cfg.cpp.SOLVE_DIAGONAL_LOWMEMORY:
        raise NotImplementedError("SOLVE_TRIDIAGONAL: SOLVE_DIAGONAL_LOWMEMORY (:87-146) is not ported")
    if cfg.cpp.SOLVE_DIAGONAL_KINNER:
        raise NotImplementedError("SOLVE_TRIDIAGONAL: SOLVE_DIAGONAL_KINNER (:152-222) is not ported")
    sz = cfg.size
    Nr = sz.Nr
    errCode = jnp.zeros(y3d.data.shape[0], jnp.int32)                   # :150  errCode = 0

    # :227-235  init. + copy to temp. array
    k, j, i = loops_kji((1, Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
    c3d_prime = c3d.local("c3d_prime").at[i, j, k].set(0.0)
    y3d_prime = y3d.local("y3d_prime").at[i, j, k].set(0.0)
    y3d_m1 = y3d.local("y3d_m1").at[i, j, k].set(y3d[i, j, k])

    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    # :241-254  forward sweep, k = 1
    nz = b3d[i, j, 1] != 0.0                                            # IF ( b3d(i,j,1).NE.0. _d 0 ) THEN
    recVar = safe_div(1.0, b3d[i, j, 1], nz)                            # recVar = 1. _d 0 / b3d(i,j,1)
    c3d_prime = c3d_prime.at[i, j, 1].set(jnp.where(nz, c3d[i, j, 1]*recVar, 0.0))
    y3d_prime = y3d_prime.at[i, j, 1].set(jnp.where(nz, y3d_m1[i, j, 1]*recVar, 0.0))
    errCode = jnp.where(jnp.any(~nz, axis=(1, 2)), 1, errCode)

    # :255-275  forward sweep, k = 2..Nr
    def forward(carry, x):
        cpm1, ypm1 = carry
        tmpVar = x["b"][i, j] - x["a"][i, j]*cpm1[i, j]                 # :258
        nzk = tmpVar != 0.0                                             # :259
        rv = safe_div(1.0, tmpVar, nzk)                                 # :260
        cpk = x["cp"].at[i, j].set(jnp.where(nzk, x["c"][i, j]*rv, 0.0))         # :261 / :266
        ypk = x["yp"].at[i, j].set(jnp.where(nzk, (x["y"][i, j]                  # :262-264 / :267
                                                   - x["a"][i, j]*ypm1[i, j]
                                                   )*rv, 0.0))
        return (cpk, ypk), {"cp": cpk, "yp": ypk, "err": jnp.any(~nzk, axis=(1, 2))}

    _, fw = scan_k(forward, (level(c3d_prime, 1), level(y3d_prime, 1)), range(2, Nr+1),
                   lambda k: {"a": level(a3d, k), "b": level(b3d, k), "c": level(c3d, k), "y": level(y3d_m1, k),
                              "cp": level(c3d_prime, k), "yp": level(y3d_prime, k)})
    c3d_prime = set_levels(c3d_prime, fw["cp"], range(2, Nr+1))
    y3d_prime = set_levels(y3d_prime, fw["yp"], range(2, Nr+1))
    errCode = jnp.where(jnp.any(fw["err"], axis=0), 1, errCode)         # :268  errCode = 1

    # :279-295  backward sweep
    y3d = y3d.at[i, j, Nr].set(y3d_prime[i, j, Nr])                     # :280-285  k = Nr

    def backward(ykp1, x):
        yk = x["y"].at[i, j].set(x["yp"][i, j]                          # :289-290
                                 - x["cp"][i, j]*ykp1[i, j])
        return yk, yk

    _, bw = scan_k(backward, level(y3d, Nr), range(Nr-1, 0, -1),
                   lambda k: {"y": level(y3d, k), "yp": level(y3d_prime, k), "cp": level(c3d_prime, k)})
    y3d = set_levels(y3d, bw, range(1, Nr))
    return y3d, errCode
