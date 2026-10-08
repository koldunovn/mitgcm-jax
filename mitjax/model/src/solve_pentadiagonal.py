"""SOLVE_PENTADIAGONAL: model/src/solve_pentadiagonal.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j, loops_kji
from mitjax.ops.safe import safe_div
from mitjax.ops.scan_k import level, scan_k, set_levels


def solve_pentadiagonal(iMin, iMax, jMin, jMax, a5d, b5d, c5d, d5d, e5d, y5d, errCode, *, cfg):
    """SOLVE_PENTADIAGONAL( iMin,iMax, jMin,jMax, a5d, b5d, c5d, d5d, e5d, y5d, errCode, bi, bj, myThid )
    @63cdc0b model/src/solve_pentadiagonal.F:10-432

    C     *==========================================================*
    C     | S/R SOLVE_PENTADIAGONAL
    C     | o Solve a penta-diagonal system A*X=Y (dimension Nr)
    C     *==========================================================*
    C     | o Used to solve implicitly vertical advection & diffusion
    C     a5d     :: 2nd  lower diagonal of the pentadiagonal matrix
    C     b5d     :: 1rst lower diagonal of the pentadiagonal matrix
    C     c5d     :: main diagonal       of the pentadiagonal matrix
    C     d5d     :: 1rst upper diagonal of the pentadiagonal matrix
    C     e5d     :: 2nd  upper diagonal of the pentadiagonal matrix
    C     y5d     :: Y vector (R.H.S.); -- OUTPUT: -- X = solution of A*X=Y
    C     errCode :: > 0 if singular matrix

    a5d..e5d, y5d: FArrays (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr). Returns (y5d, errCode): errCode an int32 array [tile]
    (one Fortran call per tile): 0 (:208, which overwrites the caller's -1) and 1 on a tile with a zero pivot. Ported: the
    default branch (neither SOLVE_DIAGONAL_LOWMEMORY nor SOLVE_DIAGONAL_KINNER, :316-424), which runs over the whole
    tile (iMin..jMax are not used there); requires INCLUDE_IMPLVERTADV_CODE or no ALLOW_AUTODIFF (:70). The forward
    sweep DO k=1,Nr reads levels k-1 and k-2, the backward sweep DO k=Nr,1,-1 levels k+1 and k+2: recursions in k,
    scan_k in the Fortran order with the first one (two) levels as single statements. Where the Fortran tests the
    pivot (`tmpVar.NE.0. _d 0`, :381), the division is guarded on that test (finite derivatives)."""
    if cfg.cpp.ALLOW_AUTODIFF and not cfg.cpp.INCLUDE_IMPLVERTADV_CODE:
        return y5d, errCode                                                     # :70  the routine is empty
    if cfg.cpp.SOLVE_DIAGONAL_LOWMEMORY:
        raise NotImplementedError("SOLVE_PENTADIAGONAL: SOLVE_DIAGONAL_LOWMEMORY (:96-204) is not ported")
    if cfg.cpp.SOLVE_DIAGONAL_KINNER:
        raise NotImplementedError("SOLVE_PENTADIAGONAL: SOLVE_DIAGONAL_KINNER (:210-315) is not ported")
    sz = cfg.size
    Nr = sz.Nr
    if Nr < 3:
        raise NotImplementedError("SOLVE_PENTADIAGONAL: Nr < 3 is not ported (the k >= 3 sweep is assumed)")
    errCode = jnp.zeros(y5d.data.shape[0], jnp.int32)                         # :208  errCode = 0

    # :321-332  Init. + copy to temp. array
    k, j, i = loops_kji((1, Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
    c5d_prime = c5d.local("c5d_prime").at[i, j, k].set(0.)
    d5d_prime = d5d.local("d5d_prime").at[i, j, k].set(0.)
    e5d_prime = e5d.local("e5d_prime").at[i, j, k].set(0.)
    y5d_prime = y5d.local("y5d_prime").at[i, j, k].set(0.)
    y5d_m1 = y5d.local("y5d_m1").at[i, j, k].set(y5d[i, j, k])

    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)

    def normalise(cp, dp, ep, yp):
        """:377-392 normalization of one level: (dp, ep, yp, zero-pivot flag per tile)."""
        tmpVar = cp[i, j]                                                       # :380
        nz = tmpVar != 0.                                                       # :381
        recVar = safe_div(1., tmpVar, nz)                                       # :382  1. _d 0 / tmpVar
        dp = dp.at[i, j].set(jnp.where(nz, dp[i, j]*recVar, 0.))                # :383 / :387  0. _d 0
        ep = ep.at[i, j].set(jnp.where(nz, ep[i, j]*recVar, 0.))                # :384 / :388
        yp = yp.at[i, j].set(jnp.where(nz, yp[i, j]*recVar, 0.))                # :385 / :389
        return dp, ep, yp, jnp.any(~nz, axis=(1, 2))                            # :390  errCode = 1

    # :341-350  k = 1: just copy terms
    cp1 = level(c5d_prime, 1).at[i, j].set(c5d[i, j, 1])
    dp1 = level(d5d_prime, 1).at[i, j].set(d5d[i, j, 1])
    ep1 = level(e5d_prime, 1).at[i, j].set(e5d[i, j, 1])
    yp1 = level(y5d_prime, 1).at[i, j].set(y5d_m1[i, j, 1])
    dp1, ep1, yp1, err1 = normalise(cp1, dp1, ep1, yp1)
    # :351-361  k = 2: subtract one term
    tmpVar = b5d[i, j, 2]                                                       # :355
    cp2 = level(c5d_prime, 2).at[i, j].set(c5d[i, j, 2] - tmpVar*dp1[i, j])    # :356
    dp2 = level(d5d_prime, 2).at[i, j].set(d5d[i, j, 2] - tmpVar*ep1[i, j])    # :357
    ep2 = level(e5d_prime, 2).at[i, j].set(e5d[i, j, 2])                       # :358
    yp2 = level(y5d_prime, 2).at[i, j].set(y5d_m1[i, j, 2] - tmpVar*yp1[i, j])   # :359
    dp2, ep2, yp2, err2 = normalise(cp2, dp2, ep2, yp2)
    errCode = jnp.where(err1 | err2, 1, errCode)

    # :362-374  k >= 3: subtract two terms (carry: d', e', y' of levels k-1 and k-2)
    def forward(carry, x):
        (dm1, em1, ym1), (dm2, em2, ym2) = carry
        tmpVar = x["b"][i, j] - x["a"][i, j]*dm2[i, j]                          # :366
        cp = x["cp"].at[i, j].set(x["c"][i, j] - tmpVar*dm1[i, j]               # :367-368
                                  - x["a"][i, j]*em2[i, j])
        dp = x["dp"].at[i, j].set(x["d"][i, j] - tmpVar*em1[i, j])             # :369
        ep = x["ep"].at[i, j].set(x["e"][i, j])                                 # :370
        yp = x["yp"].at[i, j].set(x["y"][i, j] - tmpVar*ym1[i, j]               # :371-372
                                  - x["a"][i, j]*ym2[i, j])
        dp, ep, yp, err = normalise(cp, dp, ep, yp)
        return ((dp, ep, yp), (dm1, em1, ym1)), {"cp": cp, "dp": dp, "ep": ep, "yp": yp, "err": err}

    _, fw = scan_k(forward, ((dp2, ep2, yp2), (dp1, ep1, yp1)), range(3, Nr+1),
                   lambda k: {"a": level(a5d, k), "b": level(b5d, k), "c": level(c5d, k), "d": level(d5d, k),
                              "e": level(e5d, k), "y": level(y5d_m1, k), "cp": level(c5d_prime, k),
                              "dp": level(d5d_prime, k), "ep": level(e5d_prime, k), "yp": level(y5d_prime, k)})
    errCode = jnp.where(jnp.any(fw["err"], axis=0), 1, errCode)
    # levels 1, 2 (single statements above) and 3..Nr (the scan) written back
    c5d_prime = set_levels(set_levels(c5d_prime, jnp.stack([cp1.data, cp2.data]), range(1, 3)), fw["cp"],
                           range(3, Nr+1))
    d5d_prime = set_levels(set_levels(d5d_prime, jnp.stack([dp1.data, dp2.data]), range(1, 3)), fw["dp"],
                           range(3, Nr+1))
    e5d_prime = set_levels(set_levels(e5d_prime, jnp.stack([ep1.data, ep2.data]), range(1, 3)), fw["ep"],
                           range(3, Nr+1))
    y5d_prime = set_levels(set_levels(y5d_prime, jnp.stack([yp1.data, yp2.data]), range(1, 3)), fw["yp"],
                           range(3, Nr+1))

    # :398-423  Backward sweep (starting from bottom)
    y5d = y5d.at[i, j, Nr].set(y5d_prime[i, j, Nr])                             # :400-405  k = Nr
    y5d = y5d.at[i, j, Nr-1].set(y5d_prime[i, j, Nr-1]                          # :406-412  k = Nr-1
                                 - y5d[i, j, Nr]*d5d_prime[i, j, Nr-1])

    def backward(carry, x):
        yp1_, yp2_ = carry                                                      # y5d(k+1), y5d(k+2)
        yk = x["y"].at[i, j].set(x["yp"][i, j]                                  # :413-420
                                 - yp1_[i, j]*x["dp"][i, j]
                                 - yp2_[i, j]*x["ep"][i, j])
        return (yk, yp1_), yk

    if Nr > 2:                                                                  # Nr >= 3 (raised above)
        _, bw = scan_k(backward, (level(y5d, Nr-1), level(y5d, Nr)), range(Nr-2, 0, -1),
                       lambda k: {"y": level(y5d, k), "yp": level(y5d_prime, k), "dp": level(d5d_prime, k),
                                  "ep": level(e5d_prime, k)})
        y5d = set_levels(y5d, bw, range(1, Nr-1))
    return y5d, errCode
