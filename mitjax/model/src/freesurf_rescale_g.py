"""FREESURF_RESCALE_G: model/src/freesurf_rescale_g.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j


def freesurf_rescale_g(k, gTracer, *, cfg, params, state, grid=None):
    """FREESURF_RESCALE_G( bi, bj, k, gTracer, myThid )   @63cdc0b model/src/freesurf_rescale_g.F:6-87

    C     | SUBROUTINE FREESURF_RESCALE_G
    C     | o Re-scale Gchange to be consistent with the new rStarFac / hFac (Non-Linear Free-Surface)

    Returns gTracer (the caller's (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr) array) with level k (a Python int) rescaled.
    Under NONLIN_FRSURF with nonlinFreeSurf > 0 and select_rStar > 0: gTracer(i,j,k) / rStarExpC(i,j) on every point
    (:53-61; rStarExpC of the State: CALC_R_STAR guards and counts its zero denominator, calc_r_star.F:306-311, so it
    is never 0). GOADK lane (global_ocean.90x40x15/code_ad, select_rStar = 0): the hFac_surfC arm (:73-82), at
    k = kSurfC gTracer*_hFacC/hFac_surfC (`grid`: GRID.h kSurfC, hFacC as the step sees them; `state`: hFac_surfC),
    the division guarded where it is not taken. Raise: the sigma arm (:62-72)."""
    if not cfg.cpp.NONLIN_FRSURF:
        return gTracer
    sz = cfg.size
    if params.nonlinFreeSurf > 0:                                               # :51
        if params.select_rStar > 0:                                             # :52
            if cfg.cpp.DISABLE_RSTAR_CODE:
                return gTracer
            j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)                                 # :54
            i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)                                 # :55
            gTracer = gTracer.at[i, j, k].set(gTracer[i, j, k]                  # :56-57
                                              / state.rStarExpC[i, j])
        elif params.selectSigmaCoord_ne_0:                                      # :62-72
            raise NotImplementedError("FREESURF_RESCALE_G: the hybrid sigma arm (:62-72) is not ported")
        else:                                                                   # :73-82 (GOADK lane)
            if grid is None:
                raise ValueError("FREESURF_RESCALE_G: the hFac_surfC arm needs `grid` (kSurfC)")
            j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)                                 # :74
            i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)                                 # :75
            at = grid.kSurfC[i, j] == k                                         # :76  IF (k.EQ.kSurfC(i,j,bi,bj))
            den = jnp.where(at, state.hFac_surfC[i, j], 1.)    # guard: divided only where k = kSurfC (wet surface)
            gTracer = gTracer.at[i, j, k].set(jnp.where(                        # :77-78
                at, gTracer[i, j, k]*grid.hFacC[i, j, k]/den, gTracer[i, j, k]))
    return gTracer
