"""CG2D_NSA: model/src/cg2d_nsa.F @63cdc0b (the literal forward solver of the not-self-adjoint CG2D), and its
derivative wiring through the implicit rule of mitjax/ad/cg2d_rule.py (the same rule and operator as CG2D's,
mitjax/model/src/cg2d.py: CG2D_NSA solves the same system).

    cg2d_nsa(...)        the literal CG2D_NSA: `DO it2d=1, numIters` with the residual test inside the loop (a
                         `lax.while_loop`; the iterations are never differentiated);
    cg2d_nsa_solve(...)  what SOLVE_FOR_PRESSURE calls with useNSACGSolver (solve_for_pressure.F:297-305): the same
                         outputs, bit for bit on every lane; derivatives from the implicit rule.
"""

from dataclasses import dataclass
from functools import partial

import jax.numpy as jnp
from jax import lax

from mitjax.ad import cg2d_rule
from mitjax.eesupp.global_sum import tile_sum_fortran
from mitjax.farray import FArray, loop_i, loop_j
from mitjax.model.src.cg2d import EXCH_XY_RL, _cg2dh_of, _derivative_solve, _xy, cg2d_operator
from mitjax.ops.fortran_minmax import MAX_CHAIN
from mitjax.ops.safe import safe_div, safe_sqrt


def cg2d_nsa(cg2d_b, cg2d_x, numIters, nIterMin, *, cfg, cg2dh, params, ex):
    """CG2D_NSA( cg2d_b, cg2d_x, firstResidual, minResidualSq, lastResidual, numIters, nIterMin, myThid )
    @63cdc0b model/src/cg2d_nsa.F:15-398

    C     | SUBROUTINE CG2D_NSA
    C     | o Two-dimensional grid problem conjugate-gradient inverter
    C     |   (with preconditioner).
    C     | o This version is used only in the case when the matrix
    C     |   operator is Not "Self-Adjoint" (NSA). Any remaining residuals
    C     |   will be immediately reported to the National Security Agency
    C     | Con. grad is an iterative procedure for solving Ax = b.
    C     | It requires the A be symmetric.
    C     | This implementation assumes A is a five-diagonal matrix
    C     | of the form that arises in the discrete representation of
    C     | the del^2 operator in a two-dimensional space.
    C     cg2d_b    :: The source term or "right hand side" (Output: normalised RHS)
    C     cg2d_x    :: The solution (Input: first guess)
    C     firstResidual :: the initial residual before any iterations
    C     minResidualSq :: the lowest residual reached (squared)
    C     lastResidual  :: the actual residual reached
    C     numIters  :: Inp: the maximum number of iterations allowed
    C                  Out: the actual number of iterations used
    C     nIterMin  :: Inp: decide to store (if >=0) or not (if <0) lowest res. sol.
    C                  Out: iteration number corresponding to lowest residual

    cg2d_b, cg2d_x: FArrays (1-OLx:sNx+OLx,1-OLy:sNy+OLy) (:59-60); numIters, nIterMin: static Python ints
    (SOLVE_FOR_PRESSURE passes cg2dMaxIters and cg2dUseMinResSol-1, solve_for_pressure.F:277-278); `cg2dh`: CG2D.h
    (aW2d, aS2d, aC2d, pW, pS, pC, cg2dNorm, cg2dTolerance_sq, cg2dNormaliseRHS); `params`: PARAMS.h values read
    here, cg2dMinItersNSA (static int; tested against the iteration counter), debugLevel and printResidualFreq
    (static); `ex`: the exchanger.

    Returns (cg2d_b, cg2d_x, firstResidual, minResidualSq, lastResidual, numIters, nIterMin, printed): the Fortran
    outputs in argument order (numIters as an int32 scalar: the last iteration that ran, :354, :394; minResidualSq =
    -1 and nIterMin as given: CG2D_NSA never sets them, :135), then `printed` = {"sumRHS", "rhsMax"}, the values of
    the line ' cg2d_nsa: Sum(rhs),rhsMax = ' (:226-232).

    The iteration `DO it2d=1, numIters` with `IF ( it2d .LE. cg2dMinItersNSA .OR. err_sq .GE. cg2dTolerance_sq )`
    around its body (:239-371) is a `lax.while_loop` with that condition: once the test fails at an it2d past
    cg2dMinItersNSA, err_sq no longer changes and every later iteration is empty, so stopping there is the Fortran's
    result (actualIts keeps the last executed it2d). The work arrays have their declarations: cg2d_z, cg2d_q
    (1:sNx,1:sNy), cg2d_r and cg2d_s (1-OLx:sNx+OLx,1-OLy:sNy+OLy), zeroed before the first residual (:183-194);
    exchanges are EXCH_XY_RL (:173, :215, :304, :367). Tile partial sums are `DO j; DO i` chains from 0, tiles added
    in tile order (GLOBAL_SUM_TILE_RL). The halo points of cg2d_x hold the exchanged first guess (:173) and are never
    updated, as in the Fortran. Divisions guarded where the Fortran tests the divisor (:288, :332) and SQRT where it
    tests err_sq (:219-223, :389-393): finite values and derivatives on the untaken lanes.

    Ported for tutorial_tracer_adjsens/code_ad (ALLOW_AUTODIFF_TAMC defined: the numItersMax and cg2dNormaliseRHS
    checks :117-132 run, the normalisation :152-170 and :373-386 is compiled out). Raise: ALLOW_CG2D_NSA undefined
    (an empty routine, :68), a build without ALLOW_AUTODIFF_TAMC (the cg2dNormaliseRHS normalisation),
    printResidualFreq >= 1 with debugLevel >= debLevZero (the per-iteration print :357-364); the Fortran STOPs of
    :118-131 raise. nIterMin is not read (any value). numItersMax: the experiment's tamc.h PARAMETER (`params`). rhsMax
    (:140-150) only feeds the printed line under ALLOW_AUTODIFF_TAMC.
    """
    from mitjax.model.src.cg2d_h import debLevZero
    sz = cfg.size
    sNx, sNy, OLx, OLy = sz.sNx, sz.sNy, sz.OLx, sz.OLy
    if not cfg.cpp.ALLOW_CG2D_NSA:                                              # :68
        raise NotImplementedError("CG2D_NSA: compiled without ALLOW_CG2D_NSA (an empty routine)")
    if not cfg.cpp.ALLOW_AUTODIFF_TAMC:                                         # :152-170, :373-386
        raise NotImplementedError("CG2D_NSA: a build without ALLOW_AUTODIFF_TAMC (cg2dNormaliseRHS normalisation) "
                                  "is not ported")
    if params.debugLevel >= debLevZero and params.printResidualFreq >= 1:       # :225-232, :357-364
        raise NotImplementedError("CG2D_NSA: printResidualFreq >= 1 (per-iteration residual print) is not ported")
    if numIters > params.numItersMax:                                           # :118-123
        raise ValueError(f"CG2D_NSA: numIters > numItersMax ={params.numItersMax:10d}\n"
                         "ABNORMAL END: S/R CG2D_NSA")
    if cg2dh.cg2dNormaliseRHS:                                                  # :124-131
        raise ValueError("CG2D_NSA: cg2dNormaliseRHS is disabled\nset cg2dTargetResWunit (instead of "
                         "cg2dTargetResidual)\nABNORMAL END: S/R CG2D_NSA")
    aW2d, aS2d, aC2d = cg2dh.aW2d, cg2dh.aS2d, cg2dh.aC2d
    pW, pS, pC = cg2dh.pW, cg2dh.pS, cg2dh.pC
    cg2dNorm, cg2dTolerance_sq = cg2dh.cg2dNorm, cg2dh.cg2dTolerance_sq
    cg2dMinItersNSA = int(params.cg2dMinItersNSA)
    dt = cg2d_x.dtype
    nt = cg2d_x.ntiles

    # :109-114  local work arrays (NaN until the Fortran writes them), typed per tile (`ex.vary`)
    cg2d_z = FArray(ex.vary(jnp.full((nt, sNy, sNx), jnp.nan, dt)), "cg2d_z", i=(1, sNx), j=(1, sNy))
    cg2d_q = FArray(ex.vary(jnp.full((nt, sNy, sNx), jnp.nan, dt)), "cg2d_q", i=(1, sNx), j=(1, sNy))
    full = dict(i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy))
    cg2d_r = FArray(ex.vary(jnp.full((nt, sNy+2*OLy, sNx+2*OLx), jnp.nan, dt)), "cg2d_r", **full)
    cg2d_s = FArray(ex.vary(jnp.full((nt, sNy+2*OLy, sNx+2*OLx), jnp.nan, dt)), "cg2d_s", **full)

    # :134-137  Initialise auxiliary constant, some output variable and inverter
    minResidualSq = jnp.asarray(-1.0, dt)
    recip_eta_qrNM1 = jnp.asarray(1.0, dt)                                      # :137 (eta_qrNM1 = 1, :136, unused)

    # :139-150  Normalise RHS
    j = loop_j(1, sNy)
    i = loop_i(1, sNx)
    cg2d_b = cg2d_b.at[i, j].set(cg2d_b[i, j]*cg2dNorm)                         # :145
    # rhsMax = 0. _d 0; DO bj; DO bi; DO j; DO i: rhsMax = MAX(ABS(cg2d_b(i,j,bi,bj)),rhsMax) (:140-150), one chain
    # over every tile in tile order, the new value winning ties and NaN (the tutorial_tracer_adjsens/code_ad table)
    rhsMax = MAX_CHAIN(0.0, jnp.abs(cg2d_b[i, j]), p="a", acc="b", ex=ex)      # :146

    # :172-173  Update overlaps
    cg2d_x = EXCH_XY_RL(cg2d_x, ex=ex)

    # :175-212  Initial residual calculation
    cg2d_q = cg2d_q.at[i, j].set(0.)                                            # :185
    cg2d_z = cg2d_z.at[i, j].set(0.)                                            # :186
    jf = loop_j(1-OLy, sNy+OLy)                                                 # :189
    i_f = loop_i(1-OLx, sNx+OLx)                                                # :190
    cg2d_r = cg2d_r.at[i_f, jf].set(0.)                                         # :191
    cg2d_s = cg2d_s.at[i_f, jf].set(0.)                                         # :192
    cg2d_r = cg2d_r.at[i, j].set(cg2d_b[i, j] -
                                 (aW2d[i, j]*cg2d_x[i-1, j]
                                  + aW2d[i+1, j]*cg2d_x[i+1, j]
                                  + aS2d[i, j]*cg2d_x[i, j-1]
                                  + aS2d[i, j+1]*cg2d_x[i, j+1]
                                  + aC2d[i, j]*cg2d_x[i, j]
                                  ))                                            # :199-205
    errTile = tile_sum_fortran(cg2d_r[i, j]*cg2d_r[i, j])                       # :195, :206-207
    sumRHStile = tile_sum_fortran(cg2d_b[i, j])                                 # :196, :208
    cg2d_r = EXCH_XY_RL(cg2d_r, ex=ex)                                          # :215
    err_sq = ex.global_sum_tile(errTile)                                        # :216
    sumRHS = ex.global_sum_tile(sumRHStile)                                     # :217
    actualIts = jnp.asarray(0, jnp.int32)                                       # :218
    nz = err_sq != 0.                                                           # :219-223
    firstResidual = jnp.where(nz, safe_sqrt(err_sq, nz), 0.)

    # :239-371  DO it2d=1, numIters ; IF ( it2d .LE. cg2dMinItersNSA .OR. err_sq .GE. cg2dTolerance_sq )
    def cond(c):
        it2d, err_sq = c[0], c[1]
        return (it2d <= numIters) & ((it2d <= cg2dMinItersNSA) | (err_sq >= cg2dTolerance_sq))

    def body(c):
        it2d, err_sq, cg2d_x, cg2d_r, cg2d_s, cg2d_z, cg2d_q, recip_eta_qrNM1, actualIts = c
        # :253-275  Solve preconditioning equation and update conjugate direction vector "s"
        cg2d_z = cg2d_z.at[i, j].set(pC[i, j]*cg2d_r[i, j]
                                     + pW[i, j]*cg2d_r[i-1, j]
                                     + pW[i+1, j]*cg2d_r[i+1, j]
                                     + pS[i, j]*cg2d_r[i, j-1]
                                     + pS[i, j+1]*cg2d_r[i, j+1])               # :264-269
        eta_qrNtile = tile_sum_fortran(cg2d_z[i, j]*cg2d_r[i, j])               # :261, :270-271
        eta_qrN = ex.global_sum_tile(eta_qrNtile)                               # :277
        cgBeta = eta_qrN*recip_eta_qrNM1                                        # :283
        nzq = eta_qrN != 0.                                                     # :287-288
        recip_eta_qrNM1 = jnp.where(nzq, safe_div(1., eta_qrN, nzq), 0.)
        cg2d_s = cg2d_s.at[i, j].set(cg2d_z[i, j]
                                     + cgBeta*cg2d_s[i, j])                     # :294-295
        cg2d_s = EXCH_XY_RL(cg2d_s, ex=ex)                                      # :304
        # :306-332  Evaluate laplace operator on conjugate gradient vector  q = A.s
        cg2d_q = cg2d_q.at[i, j].set(aW2d[i, j]*cg2d_s[i-1, j]
                                     + aW2d[i+1, j]*cg2d_s[i+1, j]
                                     + aS2d[i, j]*cg2d_s[i, j-1]
                                     + aS2d[i, j+1]*cg2d_s[i, j+1]
                                     + aC2d[i, j]*cg2d_s[i, j])                 # :318-323
        alphaTile = tile_sum_fortran(cg2d_s[i, j]*cg2d_q[i, j])                 # :315, :324-325
        alphaSum = ex.global_sum_tile(alphaTile)                                # :330
        nza = alphaSum != 0.                                                    # :331-332
        alpha = jnp.where(nza, safe_div(eta_qrN, alphaSum, nza), 0.)
        # :334-354  Update simultaneously solution and residual vectors (and Iter number)
        cg2d_x = cg2d_x.at[i, j].set(cg2d_x[i, j]+alpha*cg2d_s[i, j])           # :346
        cg2d_r = cg2d_r.at[i, j].set(cg2d_r[i, j]-alpha*cg2d_q[i, j])           # :347
        errTile = tile_sum_fortran(cg2d_r[i, j]*cg2d_r[i, j])                   # :343, :348-349
        actualIts = it2d                                                        # :354
        err_sq = ex.global_sum_tile(errTile)                                    # :356
        cg2d_r = EXCH_XY_RL(cg2d_r, ex=ex)                                      # :367
        return (it2d + 1, err_sq, cg2d_x, cg2d_r, cg2d_s, cg2d_z, cg2d_q, recip_eta_qrNM1, actualIts)

    carry = (jnp.asarray(1, jnp.int32), err_sq, cg2d_x, cg2d_r, cg2d_s, cg2d_z, cg2d_q, recip_eta_qrNM1, actualIts)
    _, err_sq, cg2d_x, cg2d_r, cg2d_s, cg2d_z, cg2d_q, recip_eta_qrNM1, actualIts = lax.while_loop(cond, body, carry)

    # :388-394  Return parameters to caller
    nz = err_sq != 0.
    lastResidual = jnp.where(nz, safe_sqrt(err_sq, nz), 0.)
    numIters_out = actualIts
    nIterMin_out = jnp.asarray(nIterMin, jnp.int32)
    printed = {"sumRHS": sumRHS, "rhsMax": rhsMax}
    return cg2d_b, cg2d_x, firstResidual, minResidualSq, lastResidual, numIters_out, nIterMin_out, printed


# ---------------------------------------------------------------------------------------------------------------------
# derivative wiring: CG2D_NSA through the implicit rule (mitjax/ad/cg2d_rule.py), as cg2d.cg2d_solve

@dataclass(frozen=True)
class _SolveStaticNSA:
    """Static (hashable) configuration of the forward solve inside the rule."""
    cfg: object
    numIters: int
    nIterMin: int
    printResidualFreq: int
    debugLevel: int
    cg2dMinItersNSA: int
    numItersMax: int


def _forward_solve_nsa(static, A, b, x_first, op_consts, solve_consts):
    """cg2d_rule's forward_solve: the literal cg2d_nsa(); x_full = its cg2d_x on every lane (halos included)."""
    L = op_consts["ex"].layout
    cg2dh = _cg2dh_of(A, op_consts, solve_consts).replace(cg2dNormaliseRHS=False)
    out = cg2d_nsa(_xy(b, "cg2d_b", L), _xy(x_first, "cg2d_x", L), static.numIters, static.nIterMin, cfg=static.cfg,
                   cg2dh=cg2dh, params=static, ex=op_consts["ex"])
    cg2d_b, cg2d_x, firstResidual, minResidualSq, lastResidual, numIters, nIterMin, printed = out
    aux = {"cg2d_b": cg2d_b.data, "firstResidual": firstResidual, "minResidualSq": minResidualSq,
           "lastResidual": lastResidual, "numIters": numIters, "nIterMin": nIterMin, "sumRHS": printed["sumRHS"],
           "rhsMax": printed["rhsMax"]}
    return cg2d_x.data, aux


def cg2d_nsa_solve(cg2d_b, cg2d_x, numIters, nIterMin, *, cfg, cg2dh, params, ex):
    """CG2D_NSA as SOLVE_FOR_PRESSURE calls it with useNSACGSolver (solve_for_pressure.F:297-305), differentiable:
    the arguments and outputs of `cg2d_nsa` (same order), every output bit for bit cg2d_nsa()'s. Derivative: the
    implicit rule of CG2D (mitjax/model/src/cg2d.py cg2d_solve: same operator M = A/cg2dNorm, since CG2D_NSA scales
    cg2d_b by cg2dNorm and solves with the same normalised aW2d, aS2d, aC2d; same PCG derivative solve), with
    respect to cg2d_b and the operator, none with respect to the first guess; the tangent of cg2d_x is 0 on halo
    lanes, exact because SOLVE_FOR_PRESSURE's _EXCH_XY_RL( cg2d_x ) (solve_for_pressure.F:315) overwrites every halo.
    TAF differentiates CG2D_NSA's iterations instead (that is why the routine exists); the JAX derivative is the
    exact derivative of the converged solution (docs/PORTING_RULES.md §3)."""
    if cg2dh.cg2dNormaliseRHS:                                                  # cg2d_nsa.F:124-131
        raise ValueError("CG2D_NSA: cg2dNormaliseRHS is disabled\nset cg2dTargetResWunit (instead of "
                         "cg2dTargetResidual)\nABNORMAL END: S/R CG2D_NSA")
    L = ex.layout
    static = _SolveStaticNSA(cfg=cfg, numIters=int(numIters), nIterMin=int(nIterMin),
                             printResidualFreq=int(params.printResidualFreq), debugLevel=int(params.debugLevel),
                             cg2dMinItersNSA=int(params.cg2dMinItersNSA), numItersMax=int(params.numItersMax))
    interior = ex.tile_mask(cg2d_rule.interior_mask(L.sNx, L.sNy, L.OLx, L.OLy))
    A = {"aW2d": cg2dh.aW2d.data, "aS2d": cg2dh.aS2d.data, "aC2d": cg2dh.aC2d.data}
    op_consts = dict(ex.vary({"cg2dNorm": cg2dh.cg2dNorm, "pW": cg2dh.pW.data, "pS": cg2dh.pS.data,
                              "pC": cg2dh.pC.data}), ex=ex)
    solve_consts = {"cg2dTolerance_sq": cg2dh.cg2dTolerance_sq}
    x, aux = cg2d_rule.cg2d_implicit(partial(_forward_solve_nsa, static), cg2d_operator, _derivative_solve, A,
                                     cg2d_b.data, cg2d_x.data, op_consts, solve_consts, interior)
    printed = {"sumRHS": aux["sumRHS"], "rhsMax": aux["rhsMax"]}
    return (FArray(aux["cg2d_b"], cg2d_b.name, _dims=cg2d_b.dims), FArray(x, cg2d_x.name, _dims=cg2d_x.dims),
            aux["firstResidual"], aux["minResidualSq"], aux["lastResidual"], aux["numIters"], aux["nIterMin"],
            printed)


def cg2d_nsa_sum_rhs_message(sumRHS, rhsMax):
    """cg2d_nsa.F:229-230: WRITE(standardMessageUnit,'(A,1P2E22.14)') ' cg2d_nsa: Sum(rhs),rhsMax = ', sumRHS, rhsMax
    (a plain WRITE, debugLevel >= debLevZero)."""
    from mitjax.model.src.cg2d import fortran_1pe
    return " cg2d_nsa: Sum(rhs),rhsMax = " + fortran_1pe(float(sumRHS), 22, 14) + fortran_1pe(float(rhsMax), 22, 14)
