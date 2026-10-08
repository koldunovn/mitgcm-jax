"""COST_TEST   @63cdc0b verification/global_ocean.cs32x15/code_ad/cost_test.F:1-87 (the experiment's own copy, which
the build compiles in place of pkg/cost/cost_test.F; GO lane, M2 Task 25 adjoint)."""

import jax.numpy as jnp

from mitjax.pkg.cost.cost_h import chain_sum, require_code_dir

CODE_DIR = "global_ocean.cs32x15/code_ad"


def cost_test(cost, *, cfg, theta, thetaLev, maskC):
    """COST_TEST( myThid )

    Branch of the build (COST_OPTIONS.h: ALLOW_COST_TEST and ALLOW_COST_TSQUARED defined): for every tile, DO j =
    1,sNy; DO i = 1,sNx; DO k = 1,Nr: IF ( maskC(i,j,k,bi,bj).NE.0. ) objf_test(bi,bj) = objf_test(bi,bj) +
    (theta - thetaLev)**2 (:72-81; `**2` is the product x*x, gfortran's integer power 2), one chain per tile in that
    point order, starting from COST_INIT_VARIA's objf_test (0. _d 0, cost_init_varia.F:43). thetaLev: READ_FLD_XYZ_RL(
    hydrogThetaFile, ' ', thetaLev, 0, myThid) (:51, the caller reads it: Model.cost_fixed["thetaLev"], interior
    only). The iLocOut/jLocOut/kLocOut set-up (:38-46) and ig/jg (:60, :63) are read only by the #ifndef
    ALLOW_COST_TSQUARED branch (:65-71, not compiled). A masked point adds nothing (the Fortran skips it: +0.0
    would change -0.0 only, and objf_test starts at +0.0, so the chain value is the same)."""
    require_code_dir(cfg, CODE_DIR)
    if not (cfg.cpp.flag("ALLOW_COST_TEST", "COST_OPTIONS.h") and cfg.cpp.flag("ALLOW_COST_TSQUARED",
                                                                               "COST_OPTIONS.h")):
        raise NotImplementedError("COST_TEST: only the ALLOW_COST_TSQUARED branch of this build is ported")
    sz = cfg.size
    j0, j1, i0, i1 = sz.OLy, sz.OLy + sz.sNy, sz.OLx, sz.OLx + sz.sNx
    th = theta.data[:, :, j0:j1, i0:i1]                         # [tile, k, j, i]
    tl = thetaLev.data[:, :, j0:j1, i0:i1]
    wet = maskC.data[:, :, j0:j1, i0:i1] != 0.
    d = th - tl
    term = jnp.where(wet, d*d, 0.)                              # (theta-thetaLev)**2 on the wet points
    # chain order j, i, k (:57-81): [tile, j, i, k], one chain per tile (cost_h.chain_sum: s = 0; s = s + ...).
    # objf_test enters at COST_INIT_VARIA's 0. _d 0: 0 + chain = the Fortran chain exactly (a sum of squares >= +0)
    terms = jnp.transpose(term, (0, 2, 3, 1))
    cost.objf_test = cost.objf_test + chain_sum(terms)
    return cost
