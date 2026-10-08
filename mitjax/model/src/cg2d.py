"""CG2D: model/src/cg2d.F @63cdc0b (the literal forward solver), its derivative wiring through the implicit rule
(mitjax/ad/cg2d_rule.py), and the STDOUT lines of the solver (cg2d.F and SOLVE_FOR_PRESSURE's monitor lines).

    cg2d(...)        the literal CG2D: the Fortran iteration with its residual stopping test (a `lax.while_loop`; the
                     iterations are never differentiated);
    cg2d_solve(...)  what SOLVE_FOR_PRESSURE calls (solve_for_pressure.F:308-312): the same outputs as cg2d(), bit for
                     bit on every lane (the rule's forward value is cg2d()'s array, halos included; Nikolay
                     2026-10-01), derivatives from the implicit rule (tangent and adjoint by a tight PCG of the same
                     operator, cg2d_rule.make_pcg_solve).
"""

import math
from dataclasses import dataclass
from functools import partial

import jax.numpy as jnp
from jax import lax

from mitjax.ad import cg2d_rule, modes
from mitjax.eesupp.global_sum import tile_sum_fortran
from mitjax.eesupp.global_sum_singlecpu import global_sum_singlecpu_rl
from mitjax.farray import FArray, loop_i, loop_j
from mitjax.model.src.cg2d_h import CG2DH, debLevE, debLevZero
from mitjax.ops.fortran_minmax import MAX_CHAIN
from mitjax.ops.safe import safe_div


# ---------------------------------------------------------------------------------------------------------------------
# exchanges called by CG2D (eesupp/src/exch_xy_rx.template, exch_s3d_rx.template) on FArrays

def EXCH_XY_RL(phi, *, ex):
    """EXCH_XY_RL( phi, myThid ) on a (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy) array (mitjax/eesupp map "XY")."""
    return FArray(ex.EXCH_XY_RL(phi.data), phi.name, tiled=phi.tiled, _dims=phi.dims)


def EXCH_S3D_RL(phi, myNz, *, ex):
    """EXCH_S3D_RL( phi, myNz, myThid ) on a cg2d work array declared (0:sNx+1,0:sNy+1,nSx,nSy) (cg2d.F:91-92):
    the array is placed inside a zero (1-OLx:sNx+OLx,1-OLy:sNy+OLy) array, exchanged with the probed map "S3D"
    (which writes the width-1 ring and reads interior points only, mitjax/eesupp/exchange.py), and read back. myNz=1
    (cg2d.F calls it with 1)."""
    if myNz != 1:
        raise NotImplementedError("EXCH_S3D_RL with myNz /= 1 is not ported")
    L = ex.layout
    (_, ilo, ihi), (_, jlo, jhi) = phi.dims
    if (ilo, ihi, jlo, jhi) != (0, L.sNx + 1, 0, L.sNy + 1):
        raise ValueError(f"EXCH_S3D_RL: {phi.decl()} is not declared (0:sNx+1,0:sNy+1)")
    big = jnp.zeros((phi.data.shape[0], L.ny, L.nx), phi.data.dtype)
    sl = (slice(None), slice(L.OLy - 1, L.OLy + L.sNy + 1), slice(L.OLx - 1, L.OLx + L.sNx + 1))
    big = ex.EXCH_S3D_RL(big.at[sl].set(phi.data))
    return FArray(big[sl], phi.name, tiled=phi.tiled, _dims=phi.dims)


# ---------------------------------------------------------------------------------------------------------------------
# CG2D

def cg2d(cg2d_b, cg2d_x, numIters, nIterMin, *, cfg, cg2dh, params, ex):
    """CG2D( cg2d_b, cg2d_x, firstResidual, minResidualSq, lastResidual, numIters, nIterMin, myThid )
    @63cdc0b model/src/cg2d.F:13-422

    C     | SUBROUTINE CG2D
    C     | o Two-dimensional grid problem conjugate-gradient inverter
    C     |   (with preconditioner).
    C     | Con. grad is an iterative procedure for solving Ax = b.
    C     | It requires the A be symmetric.
    C     | This implementation assumes A is a five-diagonal matrix
    C     | of the form that arises in the discrete representation of
    C     | the del^2 operator in a two-dimensional space.
    C     cg2d_b    :: The source term or "right hand side" (output: normalised RHS)
    C     cg2d_x    :: The solution (input: first guess)
    C     firstResidual :: the initial residual before any iterations
    C     minResidualSq :: the lowest residual reached (squared)
    C     lastResidual  :: the actual residual reached
    C     numIters  :: Inp: the maximum number of iterations allowed
    C                  Out: the actual number of iterations used
    C     nIterMin  :: Inp: decide to store (if >=0) or not (if <0) lowest res. sol.
    C                  Out: iteration number corresponding to lowest residual

    cg2d_b, cg2d_x: FArrays (1-OLx:sNx+OLx,1-OLy:sNy+OLy) (cg2d.F:51-52); numIters, nIterMin: static Python ints
    (SOLVE_FOR_PRESSURE passes cg2dMaxIters and cg2dUseMinResSol-1, solve_for_pressure.F:277-278); `cg2dh`: CG2D.h
    (aW2d, aS2d, aC2d, pW, pS, pC, cg2dNorm, cg2dTolerance_sq, cg2dNormaliseRHS); `params`: reads printResidualFreq
    and debugLevel (static); `ex`: the exchanger (exchanges, GLOBAL_SUM_TILE_RL, _GLOBAL_MAX_RL).

    Returns (cg2d_b, cg2d_x, firstResidual, minResidualSq, lastResidual, numIters, nIterMin, printed): the Fortran
    outputs in argument order (numIters, nIterMin as int32 scalars), then `printed` = {"sumRHS", "rhsMax"}, the values
    of the line ' cg2d: Sum(rhs),rhsMax = ' that cg2d.F:196-202 writes on every call (debugLevel >= debLevZero; the
    caller prints it with `cg2d_sum_rhs_message`: no host output inside traced model code).

    The iteration (:207-355, `DO 10 it2d=1, numIters` with `GOTO 11` on the residual test) is a `lax.while_loop`
    whose condition is the Fortran's: continue while it2d <= numIters and the last err_sq was not below
    cg2dTolerance_sq (the test before the loop, :204, and at the end of every iteration, :337). The work arrays have
    their Fortran declarations: cg2d_q (1:sNx,1:sNy), cg2d_r and cg2d_s (0:sNx+1,0:sNy+1), zeroed incl. their ring
    before the first residual (:142-147, the master change). Tile partial sums are `DO j; DO i` chains from 0
    (eesupp/global_sum.tile_sum_fortran) and the tiles are added by GLOBAL_SUM_TILE_RL in tile order
    (`ex.global_sum_tile`). Halo points of cg2d_x hold the exchanged, normalised first guess (:120-136) and are never
    updated (the iteration and the un-normalisation run over the interior only), as in the Fortran.

    Ported for the M1 variants (the four that call the solver: barotropic gyre, baroclinic gyre,
    global_ocean.90x40x15, tutorial_global_oce_optim/input_ad; all with cg2dUseMinResSol = 0, printResidualFreq = -1,
    debugLevel 1 or 2): nIterMin < 0 and printResidual = .FALSE.. Raise: nIterMin >= 0 (min-residual solution,
    :148-155, :190-193, :338-351, :358-369; selected by cg2dUseMinResSol = 1, which no solver-calling M1 run uses),
    printResidualFreq >= 1 (the per-iteration print :329-336 and the debug block :392-418, which also exchanges
    cg2d_x). Lane M4ADLAB (lab_sea/code_ad, CPP_EEOPTIONS.h:129): CG2D_SINGLECPU_SUM, the sums of r*r, q*r, s*q
    (:170-175 / :182-186, :226-242, :282-296, :312-328) by GLOBAL_SUM_SINGLECPU_RL of the products (localBuf,
    eesupp/global_sum_singlecpu.py) instead of tile partials; Sum(rhs) keeps GLOBAL_SUM_TILE_RL (:176, :187). GO lane: cg2dNormaliseRHS = .FALSE. (global_ocean.cs32x15, the
    W-unit target residual) skips the normalisation :117-133 and the un-normalisation :371-384.
    TARGET_NEC_SX lines are compiler directives only.
    """
    sz = cfg.size
    sNx, sNy = sz.sNx, sz.sNy
    if nIterMin >= 0:
        raise NotImplementedError("CG2D: nIterMin >= 0 (cg2dUseMinResSol = 1, min-residual solution, cg2d.F:148-155, "
                                  "190-193, 338-351, 358-369) is not ported")
    if params.debugLevel >= debLevZero and params.printResidualFreq >= 1:
        raise NotImplementedError("CG2D: printResidualFreq >= 1 (per-iteration residual print cg2d.F:329-336, debug "
                                  "block :392-418) is not ported")
    singlecpu = bool(cfg.cpp.CG2D_SINGLECPU_SUM)

    def gsum(prod):
        """The global sum of a [T, sNy, sNx] product: localBuf + GLOBAL_SUM_SINGLECPU_RL with CG2D_SINGLECPU_SUM, else
        the tile partials + GLOBAL_SUM_TILE_RL."""
        if singlecpu:
            return global_sum_singlecpu_rl(prod, cfg=cfg, ex=ex)
        return ex.global_sum_tile(tile_sum_fortran(prod))
    aW2d, aS2d, aC2d = cg2dh.aW2d, cg2dh.aS2d, cg2dh.aC2d
    pW, pS, pC = cg2dh.pW, cg2dh.pS, cg2dh.pC
    cg2dNorm, cg2dTolerance_sq = cg2dh.cg2dNorm, cg2dh.cg2dTolerance_sq
    dt = cg2d_x.dtype

    # :89-92  local work arrays (NaN until the Fortran writes them); typed per tile (`ex.vary`: varying over the
    # tile axis under shard_map, the identity on one device), as the loop carry they become
    nt = cg2d_x.ntiles
    cg2d_q = FArray(ex.vary(jnp.full((nt, sNy, sNx), jnp.nan, dt)), "cg2d_q", i=(1, sNx), j=(1, sNy))
    cg2d_r = FArray(ex.vary(jnp.full((nt, sNy + 2, sNx + 2), jnp.nan, dt)), "cg2d_r", i=(0, sNx + 1),
                    j=(0, sNy + 1))
    cg2d_s = FArray(ex.vary(jnp.full((nt, sNy + 2, sNx + 2), jnp.nan, dt)), "cg2d_s", i=(0, sNx + 1),
                    j=(0, sNy + 1))

    # :100-102  Initialise auxiliary constant, some output variable and inverter
    minResidualSq = jnp.asarray(-1.0, dt)
    eta_qrNM1 = jnp.asarray(1.0, dt)

    # :104-115  Normalise RHS
    j = loop_j(1, sNy)
    i = loop_i(1, sNx)
    cg2d_b = cg2d_b.at[i, j].set(cg2d_b[i, j]*cg2dNorm)
    # rhsMax = 0. _d 0; DO bj; DO bi; DO j; DO i: rhsMax = MAX(ABS(cg2d_b),rhsMax) (:105-115), the new value winning
    # NaN (the oracle table), one chain over every tile in tile order; _GLOBAL_MAX_RL (:119, inside cg2dNormaliseRHS)
    # is the identity for the oracle's single process (global_max.F)
    rhsMax = MAX_CHAIN(0.0, jnp.abs(cg2d_b[i, j]), p="a", acc="b", ex=ex)      # :111
    # :117-133  IF (cg2dNormaliseRHS) (static: CG2D.h; GO lane: the .FALSE. case skips :118-132 and :372-383)
    if cg2dh.cg2dNormaliseRHS:
        # :120-121  rhsNorm = 1. _d 0; IF ( rhsMax .NE. 0. ) rhsNorm = 1. _d 0 / rhsMax
        rhsNorm = jnp.where(rhsMax != 0., safe_div(1.0, rhsMax, rhsMax != 0.), 1.0)
        cg2d_b = cg2d_b.at[i, j].set(cg2d_b[i, j]*rhsNorm)                      # :126
        cg2d_x = cg2d_x.at[i, j].set(cg2d_x[i, j]*rhsNorm)                      # :127

    # :135-136  Update overlaps
    cg2d_x = EXCH_XY_RL(cg2d_x, ex=ex)

    # :138-180  Initial residual calculation
    j1 = loop_j(0, sNy + 1)                                                     # :142-147
    i1 = loop_i(0, sNx + 1)
    cg2d_r = cg2d_r.at[i1, j1].set(0.0)
    cg2d_s = cg2d_s.at[i1, j1].set(0.0)
    cg2d_r = cg2d_r.at[i, j].set(cg2d_b[i, j] -
                                 (aW2d[i, j]*cg2d_x[i-1, j]
                                  + aW2d[i+1, j]*cg2d_x[i+1, j]
                                  + aS2d[i, j]*cg2d_x[i, j-1]
                                  + aS2d[i, j+1]*cg2d_x[i, j+1]
                                  + aC2d[i, j]*cg2d_x[i, j]
                                  ))                                            # :163-169
    rr = cg2d_r[i, j]*cg2d_r[i, j]                                             # :157, :170-175
    sumRHStile = tile_sum_fortran(cg2d_b[i, j])                                 # :156, :176
    cg2d_r = EXCH_S3D_RL(cg2d_r, 1, ex=ex)                                      # :181
    err_sq = gsum(rr)                                                           # :182-186
    sumRHS = ex.global_sum_tile(sumRHStile)                                     # :187
    actualIts = jnp.asarray(0, jnp.int32)                                       # :188
    firstResidual = jnp.sqrt(err_sq)                                            # :189
    # :195-202  printResidual = .FALSE. (printResidualFreq < 1, checked above); the ' cg2d: Sum(rhs),rhsMax = ' line
    # is returned in `printed`

    # :204  IF ( err_sq .LT. cg2dTolerance_sq ) GOTO 11
    done0 = err_sq < cg2dTolerance_sq

    # :206-356  >>> BEGIN SOLVER <<<  DO 10 it2d=1, numIters
    def cond(c):
        it2d, done = c[0], c[1]
        return jnp.logical_not(done) & (it2d <= numIters)

    def body(c):
        it2d, done, cg2d_x, cg2d_r, cg2d_s, cg2d_q, eta_qrNM1, err_sq, actualIts = c
        # :209-238  Solve preconditioning equation and update conjugate direction vector "s"
        cg2d_q = cg2d_q.at[i, j].set(pC[i, j]*cg2d_r[i, j]
                                     + pW[i, j]*cg2d_r[i-1, j]
                                     + pW[i+1, j]*cg2d_r[i+1, j]
                                     + pS[i, j]*cg2d_r[i, j-1]
                                     + pS[i, j+1]*cg2d_r[i, j+1])               # :219-224
        eta_qrN = gsum(cg2d_q[i, j]*cg2d_r[i, j])                               # :213, :226-242
        cgBeta = eta_qrN/eta_qrNM1                                              # :245
        eta_qrNM1 = eta_qrN                                                     # :250
        cg2d_s = cg2d_s.at[i, j].set(cg2d_q[i, j]
                                     + cgBeta*cg2d_s[i, j])                     # :256-257
        # :263-264  Do exchanges that require messages i.e. between processes.
        cg2d_s = EXCH_S3D_RL(cg2d_s, 1, ex=ex)
        # :266-296  Evaluate laplace operator on conjugate gradient vector  q = A.s
        cg2d_q = cg2d_q.at[i, j].set(aW2d[i, j]*cg2d_s[i-1, j]
                                     + aW2d[i+1, j]*cg2d_s[i+1, j]
                                     + aS2d[i, j]*cg2d_s[i, j-1]
                                     + aS2d[i, j+1]*cg2d_s[i, j+1]
                                     + aC2d[i, j]*cg2d_s[i, j])                 # :276-281
        alpha = gsum(cg2d_s[i, j]*cg2d_q[i, j])                                 # :270, :282-296
        alpha = eta_qrN/alpha                                                   # :301
        # :303-321  Update simultaneously solution and residual vectors (and Iter number)
        cg2d_x = cg2d_x.at[i, j].set(cg2d_x[i, j]+alpha*cg2d_s[i, j])           # :310
        cg2d_r = cg2d_r.at[i, j].set(cg2d_r[i, j]-alpha*cg2d_q[i, j])           # :311
        rr = cg2d_r[i, j]*cg2d_r[i, j]                                         # :307, :312-318
        actualIts = it2d                                                        # :322
        err_sq = gsum(rr)                                                       # :324-328
        # :329-336  printResidual = .FALSE.
        # :337  IF ( err_sq .LT. cg2dTolerance_sq ) GOTO 11
        done = err_sq < cg2dTolerance_sq
        # :338-351  IF ( err_sq .LT. minResidualSq ): with nIterMin < 0, minResidualSq stays -1 (:101, :190-193
        # not taken), so the test is never true and nothing is stored
        # :353  CALL EXCH_S3D_RL( cg2d_r, 1, myThid ), skipped by the GOTO 11 of :337
        cg2d_r_ex = EXCH_S3D_RL(cg2d_r, 1, ex=ex)
        cg2d_r = FArray(jnp.where(done, cg2d_r.data, cg2d_r_ex.data), cg2d_r.name, _dims=cg2d_r.dims)
        return (it2d + 1, done, cg2d_x, cg2d_r, cg2d_s, cg2d_q, eta_qrNM1, err_sq, actualIts)

    carry = (jnp.asarray(1, jnp.int32), done0, cg2d_x, cg2d_r, cg2d_s, cg2d_q, eta_qrNM1, err_sq, actualIts)
    _, _, cg2d_x, cg2d_r, cg2d_s, cg2d_q, eta_qrNM1, err_sq, actualIts = lax.while_loop(cond, body, carry)
    # :356  11 CONTINUE

    # :358-369  nIterMin < 0: not taken
    # :371-384  IF (cg2dNormaliseRHS): Un-normalise the answer
    if cg2dh.cg2dNormaliseRHS:
        cg2d_x = cg2d_x.at[i, j].set(cg2d_x[i, j]/rhsNorm)                      # :378

    # :386-388  Return parameters to caller
    lastResidual = jnp.sqrt(err_sq)
    numIters_out = actualIts
    nIterMin_out = jnp.asarray(nIterMin, jnp.int32)
    # :392  debugLevel >= debLevE .AND. printResidualFreq == 1: excluded above
    printed = {"sumRHS": sumRHS, "rhsMax": rhsMax}
    return cg2d_b, cg2d_x, firstResidual, minResidualSq, lastResidual, numIters_out, nIterMin_out, printed


# ---------------------------------------------------------------------------------------------------------------------
# derivative wiring: CG2D through the implicit rule (mitjax/ad/cg2d_rule.py)

@dataclass(frozen=True)
class _SolveStatic:
    """Static (hashable) configuration of the forward solve inside the rule."""
    cfg: object
    numIters: int
    nIterMin: int
    printResidualFreq: int
    debugLevel: int
    cg2dNormaliseRHS: bool = True       # GO lane: CG2D.h's static switch (cg2d.F:117, :371)


def _xy(data, name, L):
    return FArray(data, name, i=(1 - L.OLx, L.sNx + L.OLx), j=(1 - L.OLy, L.sNy + L.OLy))


def _cg2dh_of(A, op_consts, solve_consts, cg2dNormaliseRHS=True):
    L = op_consts["ex"].layout
    return CG2DH(aW2d=_xy(A["aW2d"], "aW2d", L), aS2d=_xy(A["aS2d"], "aS2d", L), aC2d=_xy(A["aC2d"], "aC2d", L),
                 pW=_xy(op_consts["pW"], "pW", L), pS=_xy(op_consts["pS"], "pS", L), pC=_xy(op_consts["pC"], "pC", L),
                 cg2dNorm=op_consts["cg2dNorm"], cg2dTolerance_sq=solve_consts["cg2dTolerance_sq"],
                 cg2dNormaliseRHS=cg2dNormaliseRHS)


def _forward_solve(static, A, b, x_first, op_consts, solve_consts):
    """cg2d_rule's forward_solve: the literal cg2d(); x_full = its cg2d_x on every lane (halos included)."""
    L = op_consts["ex"].layout
    out = cg2d(_xy(b, "cg2d_b", L), _xy(x_first, "cg2d_x", L), static.numIters, static.nIterMin, cfg=static.cfg,
               cg2dh=_cg2dh_of(A, op_consts, solve_consts, static.cg2dNormaliseRHS), params=static,
               ex=op_consts["ex"])
    cg2d_b, cg2d_x, firstResidual, minResidualSq, lastResidual, numIters, nIterMin, printed = out
    aux = {"cg2d_b": cg2d_b.data, "firstResidual": firstResidual, "minResidualSq": minResidualSq,
           "lastResidual": lastResidual, "numIters": numIters, "nIterMin": nIterMin, "sumRHS": printed["sumRHS"],
           "rhsMax": printed["rhsMax"]}
    return cg2d_x.data, aux


def cg2d_operator(A, op_consts, v):
    """cg2d_rule's operator M v = A_norm v / cg2dNorm (A_norm: the normalised aW2d, aS2d, aC2d of INI_CG2D /
    UPDATE_CG2D), so that M x = cg2d_b is the system CG2D solves after scaling cg2d_b by cg2dNorm (cg2d.F:110).
    Reads the interior of v only: its halos are filled by EXCH_XY_RL (which writes every halo point from interior
    sources on every M1 layout: cg2d_rule.exchange_reads_interior_only). The stencil is cg2d.F:163-169's sum in its
    order. 0 on halo lanes."""
    ex = op_consts["ex"]
    L = ex.layout
    sNx, sNy = L.sNx, L.sNy
    aW2d, aS2d, aC2d = _xy(A["aW2d"], "aW2d", L), _xy(A["aS2d"], "aS2d", L), _xy(A["aC2d"], "aC2d", L)
    x = EXCH_XY_RL(_xy(v, "v", L), ex=ex)
    j = loop_j(1, sNy)
    i = loop_i(1, sNx)
    out = _xy(jnp.zeros_like(v), "Mv", L)
    out = out.at[i, j].set((aW2d[i, j]*x[i-1, j]
                            + aW2d[i+1, j]*x[i+1, j]
                            + aS2d[i, j]*x[i, j-1]
                            + aS2d[i, j+1]*x[i, j+1]
                            + aC2d[i, j]*x[i, j])
                           / op_consts["cg2dNorm"])
    return out.data


def cg2d_preconditioner(A, op_consts, r):
    """The derivative solve's preconditioner: CG2D's (pC, pW, pS) stencil (cg2d.F:219-224) times cg2dNorm (M is the
    normalised operator over cg2dNorm), on the interior, 0 on halo lanes; r's halos filled by EXCH_XY_RL. Symmetric
    (pW(i+1,j) couples i and i+1 both ways); only its effect on the PCG's convergence matters."""
    ex = op_consts["ex"]
    L = ex.layout
    sNx, sNy = L.sNx, L.sNy
    pW, pS, pC = _xy(op_consts["pW"], "pW", L), _xy(op_consts["pS"], "pS", L), _xy(op_consts["pC"], "pC", L)
    x = EXCH_XY_RL(_xy(r, "r", L), ex=ex)
    j = loop_j(1, sNy)
    i = loop_i(1, sNx)
    out = _xy(jnp.zeros_like(r), "Pr", L)
    out = out.at[i, j].set((pC[i, j]*x[i, j]
                            + pW[i, j]*x[i-1, j]
                            + pW[i+1, j]*x[i+1, j]
                            + pS[i, j]*x[i, j-1]
                            + pS[i, j+1]*x[i, j+1])
                           * op_consts["cg2dNorm"])
    return out.data


# the derivative solve: preconditioned CG of the same operator from 0 to 1e-13 (cg2d_rule.make_pcg_solve; inner
# products are the Fortran-order global sums through op_consts["ex"])
DERIVATIVE_TOL = 1e-13
DERIVATIVE_MAX_ITERS = 4000
_derivative_solve = cg2d_rule.make_pcg_solve(cg2d_operator, precond=cg2d_preconditioner, tol=DERIVATIVE_TOL,
                                             max_iters=DERIVATIVE_MAX_ITERS)


def cg2d_solve(cg2d_b, cg2d_x, numIters, nIterMin, *, cfg, cg2dh, params, ex):
    """CG2D as SOLVE_FOR_PRESSURE calls it (solve_for_pressure.F:308-312), differentiable: the arguments and
    outputs of `cg2d` (same order), every output bit for bit cg2d()'s (the rule's forward value is the forward
    solve's array on every lane, halos included; gate test_cg2d.py). Derivative: the implicit rule
    (mitjax/ad/cg2d_rule.cg2d_implicit) with respect to cg2d_b and, when mitjax/ad/modes.cg2d_operator_adjoint says so
    (the run's cg2dFullAdjoint as TAF's CG2D_MAD, or params.mjx_cg2d_derivative = "exact"), the operator (aW2d, aS2d,
    aC2d, cg2dNorm); none with respect to the first guess cg2d_x or cg2dTolerance_sq; the tangent of cg2d_x is the implicit derivative on
    the interior and 0 on halo lanes, exact because SOLVE_FOR_PRESSURE's _EXCH_XY_RL( cg2d_x ) (solve_for_pressure.F
    :315) overwrites every halo from interior sources (cg2d_rule docstring). The returned cg2d_b (CG2D's normalised
    RHS) and the diagnostics carry zero tangents: no M1 caller reads cg2d_b after CG2D.

    `ex` is an Exchanger or (inside shard_map) a ShardedExchanger: the float setup constants are made varying with
    `ex.vary` and padding tiles get `interior` False (`ex.tile_mask`), as the rule requires.

    The solver selection of SOLVE_FOR_PRESSURE (solve_for_pressure.F:280-321) is checked here: only the default
    branch (CALL CG2D, :306-313) is ported; DISCONNECTED_TILES (CG2D_EX0, :280-286), useSRCGSolver (CG2D_SR,
    :288-296) and ALLOW_CG2D_NSA with useNSACGSolver (CG2D_NSA, :297-305) raise; cg2dFullAdjoint's CG2D_STORE
    (:316-321) changes no forward value (GOADK lane)."""
    if cfg.cpp.DISCONNECTED_TILES:
        # vermix lane (M3 Task 30): :280-286 CG2D_EX0, the disconnected-tile solver (forward only: its derivative
        # wiring is pending; reverse mode through its while_loop raises)
        from mitjax.model.src.cg2d_ex0 import cg2d_ex0
        return cg2d_ex0(cg2d_b, cg2d_x, numIters, nIterMin, cfg=cfg, cg2dh=cg2dh, params=params, ex=ex)
    if params.useSRCGSolver:
        raise NotImplementedError("SOLVE_FOR_PRESSURE: useSRCGSolver (CG2D_SR) is not ported")
    if cfg.cpp.ALLOW_CG2D_NSA and params.useNSACGSolver:
        raise NotImplementedError("SOLVE_FOR_PRESSURE: useNSACGSolver (CG2D_NSA) is not ported")
    # GOADK lane (global_ocean.90x40x15/input_ad.bottomdrag): with cg2dFullAdjoint (ALLOW_AUTODIFF), :318-319
    # CALL CG2D_STORE( cg2d_x, .TRUE. ) copies cg2d_x into TAF's tape (pkg/autodiff/cg2d_mad.F:249-...): no forward
    # value changes; its STDOUT lines (debugLevel >= debLevZero) are host output. The backward is the implicit rule of
    # mitjax/ad/cg2d_rule.py with the operator active or passive as TAF's hand-written CG2D_MAD decides it from
    # cg2dFullAdjoint (mitjax/ad/modes.py; plan "Decisions 2026-10-02" item 3 revised; backward-only).
    L = ex.layout
    static = _SolveStatic(cfg=cfg, numIters=int(numIters), nIterMin=int(nIterMin),
                          printResidualFreq=int(params.printResidualFreq), debugLevel=int(params.debugLevel),
                          cg2dNormaliseRHS=bool(cg2dh.cg2dNormaliseRHS))
    interior = ex.tile_mask(cg2d_rule.interior_mask(L.sNx, L.sNy, L.OLx, L.OLy))
    A = {"aW2d": cg2dh.aW2d.data, "aS2d": cg2dh.aS2d.data, "aC2d": cg2dh.aC2d.data}
    op_consts = dict(ex.vary({"cg2dNorm": cg2dh.cg2dNorm, "pW": cg2dh.pW.data, "pS": cg2dh.pS.data,
                              "pC": cg2dh.pC.data}), ex=ex)
    solve_consts = {"cg2dTolerance_sq": cg2dh.cg2dTolerance_sq}
    x, aux = cg2d_rule.cg2d_implicit(partial(_forward_solve, static), cg2d_operator, _derivative_solve, A,
                                     cg2d_b.data, cg2d_x.data, op_consts, solve_consts, interior,
                                     operator_active=modes.cg2d_operator_adjoint(cfg, params))
    printed = {"sumRHS": aux["sumRHS"], "rhsMax": aux["rhsMax"]}
    return (FArray(aux["cg2d_b"], cg2d_b.name, _dims=cg2d_b.dims), FArray(x, cg2d_x.name, _dims=cg2d_x.dims),
            aux["firstResidual"], aux["minResidualSq"], aux["lastResidual"], aux["numIters"], aux["nIterMin"],
            printed)


# ---------------------------------------------------------------------------------------------------------------------
# STDOUT lines (host side: formatted from the returned values)

def fortran_1pe(x, w, d):
    """Fortran `1PEw.d` (gfortran): one digit before the point, d after, exponent `E+xx` (`+xxx` without the E for
    |exponent| >= 100), right-justified in w characters; correctly rounded (as Python's formatting)."""
    if math.isnan(x):
        return "NaN".rjust(w)
    s = f"{x:.{d}E}"
    mant, _, exp = s.partition("E")
    e = int(exp)
    es = f"E{'+' if e >= 0 else '-'}{abs(e):02d}" if abs(e) < 100 else f"{'+' if e >= 0 else '-'}{abs(e):03d}"
    out = mant + es
    return out.rjust(w) if len(out) <= w else "*" * w


def cg2d_sum_rhs_message(sumRHS, rhsMax):
    """cg2d.F:199-200: WRITE(standardMessageUnit,'(A,1P2E22.14)') ' cg2d: Sum(rhs),rhsMax = ', sumRHS,rhsMax
    (a plain WRITE: no PRINT_MESSAGE prefix)."""
    return " cg2d: Sum(rhs),rhsMax = " + fortran_1pe(float(sumRHS), 22, 14) + fortran_1pe(float(rhsMax), 22, 14)


def solve_for_pressure_cg2d_messages(firstResidual, minResidualSq, lastResidual, numIters, nIterMin):
    """The lines SOLVE_FOR_PRESSURE prints after CG2D at monitorFreq when debugLevel >= debLevA
    (solve_for_pressure.F:331-350; testreport's `PS` check-list variable is cg2d_init_res), without the PRINT_MESSAGE
    prefix: '(A20,1PE23.14)' 'cg2d_init_res =', '(A27,2I8)' 'cg2d_iters(min,last) =', 'cg2d_min_res  =' (only if
    minResidualSq >= 0, printed as its SQRT), 'cg2d_last_res ='. For the core lane's solve_for_pressure.py."""
    lines = ["cg2d_init_res =".rjust(20) + fortran_1pe(float(firstResidual), 23, 14),
             "cg2d_iters(min,last) =".rjust(27) + f"{int(nIterMin):8d}{int(numIters):8d}"]
    if float(minResidualSq) >= 0.:
        lines.append("cg2d_min_res  =".rjust(20) + fortran_1pe(math.sqrt(float(minResidualSq)), 23, 14))
    lines.append("cg2d_last_res =".rjust(20) + fortran_1pe(float(lastResidual), 23, 14))
    return lines
