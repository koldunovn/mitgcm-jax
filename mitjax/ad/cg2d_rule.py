"""Implicit derivative of the two-dimensional pressure solve (CG2D): the rule the cg2d forward kernel calls (Task 7c).

The forward solve is the literal CG2D of `model/src/cg2d.F` (@63cdc0b), written in M1 Task 12 in the style Task 8
chooses; this module never differentiates its iterations ([L-AD-8]). It takes the forward solve as a function and gives
the converged solution x of  operator(A, op_consts, x) = b  the derivative of the converged system (implicit function
theorem):

    dx = M^-1 (db - d_A operator(A, op_consts, x) . dA - d_c operator(A, op_consts, x) . dc),   M v = operator(A, c, v)

a `jax.custom_jvp` whose tangent is a `lax.custom_linear_solve` of M with a TIGHT solve from a zero first guess, used
for the tangent and (M is symmetric) for the transpose alike, so the tangent-linear model and the adjoint are exact
transposes of each other ([L-AD-9], [L-CONF-8]: custom_linear_solve's own JVP re-runs the loose, warm-started primal
solve on the tangent, which is not linear in the tangent; the ECCO port measured a dot test of 1.9e-9 with it and
1.3e-11 with this rule; `test_cg2d_rule.py` measures both on a synthetic operator).

Arrays are full tile arrays `[tile, j, i]` with halos; `interior` (bool, `[j, i]` or `[tile, j, i]`) marks the Fortran
points i = 1..sNx, j = 1..sNy, where cg2d.F computes the solution (every update loops `DO j=1,sNy; DO i=1,sNx`:
the initial residual cg2d.F:161-180, the iteration from cg2d.F:207).

Forward value (Nikolay 2026-10-01: faithful to the model): the rule's output x is the forward solve's array on EVERY
lane, halos included, bit for bit (`_forward_value`; gate `test_cg2d_rule.py::test_forward_equals_literal_and_dense`
with a planted halo-zeroed forward value as its negative control). For the literal CG2D the halo lanes hold stale
values: the exchanged, normalised first guess (cg2d.F:120-136: `cg2d_x*rhsNorm` on the interior, then
`EXCH_XY_RL( cg2d_x )`), never updated by the iteration (interior only) and not un-normalised (cg2d.F:376-378,
interior only); only a debug run (`debugLevel.GE.debLevE .AND. printResidualFreq.EQ.1`, cg2d.F:392-393) exchanges
them inside CG2D.

Derivative: the unknown of the implicit derivative is the interior, and the tangent is 0 on halo lanes (so the
transpose drops halo cotangents). This is exact for the Fortran's computation, although the stale halo values do
depend on x_first, b and cg2dNorm (through rhsNorm, cg2d.F:119-121): the only reader of CG2D's cg2d_x is
SOLVE_FOR_PRESSURE's `_EXCH_XY_RL( cg2d_x, myThid )` right after the call (solve_for_pressure.F:308-315 @63cdc0b;
CG2D_STORE at :316-321 comes after it), and that exchange overwrites every halo point from an interior source and
keeps every interior point: measured by the oracle's probe for all six M1 layouts (eesupp/exch_maps.py `XY` maps;
`exchange_reads_interior_only` checks it per layout, test_exchange.py). Hence d(EXCH_XY_RL o CG2D) = E dx_interior,
and the halo tangent of CG2D never reaches anything. A layout whose XY exchange leaves a halo point unwritten or reads
a halo source would break this argument: check it with `exchange_reads_interior_only` before using the rule there.

Halo lanes are zeroed only inside the derivative solves, where it is the correct linear algebra: their right-hand side,
their matvec and their solution ([L-AD-10]). A derivative solve that leaves values on halo lanes (a literal CG2D ends
with exchanged halos; fesom_jax's sharded PCG left partial sums there) makes `custom_linear_solve(symmetric=True)`
transpose the wrong operator: the halo cotangents are dropped instead of being added onto their owners. fesom_jax:
owner cotangents next to partition edges 48-63 % wrong while parameter gradients passed; `test_cg2d_rule.py` shows the
dot test failing without the zeroing.

Sharded (eesupp/sharded_exchange.py, inside `shard_map(check_vma=True)`): the operator's exchanges and the derivative
solve's inner products come from the ShardedExchanger carried in `op_consts["ex"]` (`exchanger_vdot`); the caller makes
`interior` and the invariant leaves of `op_consts` (cg2dNorm) varying with `ex.vary` before the call ([L-PAR-10]: JAX
0.10.1 re-traces `custom_linear_solve` at lowering and fails on an invariant -> varying pvary inside it), and padding
tiles get `interior` all False (`ex.tile_mask`): their solution and cotangent are 0 ([L-PAR-6]).

Contract of the functions (all pure functions of their arguments: a custom_jvp cannot close over traced values
[L-AD-18]; model data such as the exchange maps travel in `op_consts`, never in a closure):
    forward_solve(A, b, x_first, op_consts, solve_consts) -> (x_full, aux)
        the literal forward solve (CG2D): x_full its output array on every lane (the rule's forward value); aux: its
        diagnostics (firstResidual, numIters, ...), returned with zero tangent;
    operator(A, op_consts, v) -> M v
        reads only the interior of v (fills v's halo lanes itself, as cg2d.F does with EXCH_XY_RL/EXCH_S3D_RL before
        each stencil) and returns a field that is 0 on halo lanes; bilinear in (A, v) for CG2D. For CG2D:
        operator = (aW2d x(i-1) + aW2d(i+1) x(i+1) + aS2d x(j-1) + aS2d(j+1) x(j+1) + aC2d x) / cg2dNorm with the
        normalised coefficients of ini_cg2d.F:124-127 / update_cg2d.F:116-124 (aW2d = cg2dNorm * physical), so that
        M x = cg2d_b is the system CG2D solves after scaling cg2d_b by cg2dNorm (cg2d.F:110);
    derivative_solve(A, op_consts, solve_consts, r) -> y,  M y = r to a tight tolerance from y = 0
        never differentiated (custom_linear_solve treats it as a black box); default `make_pcg_solve(operator)`
        (inner products `vdot(op_consts, a, b)`: `tile_vdot` on one device without an exchanger, `exchanger_vdot` =
        the Fortran-order global sum through `op_consts["ex"]`, the same value for every P).
Setup constants are traced arguments, never closed over or recomputed ([E§11], [L-XLA-4]: a constant cg2dNorm let XLA
regroup `(cg2d_b*cg2dNorm)*rhsNorm`, a 1e-9 drift): `cg2dNorm` (ini_cg2d.F:146) in `op_consts`, `cg2dTolerance_sq`
(master: a CG2D.h common variable set once at ini_cg2d.F:163 as cg2dTolerance*cg2dTolerance) in `solve_consts`, read
by the forward solve only.

Derivatives: with respect to b and every floating leaf of A and op_consts (cg2dNorm included); none with respect to
x_first (the converged solution does not depend on the first guess) or solve_consts. TAF's cg2d (pkg/autodiff/
cg2d.flow:4-9, ACTIVE = 1,2: cg2d_b and cg2d_x only) treats the operator as passive; with `cg2dFullAdjoint=.FALSE.`
its adjoint is again a CG2D solve of the same symmetric operator (cg2d_mad.F), as here, but to the forward tolerance
(here: `make_pcg_solve`'s 1e-13). `cg2d_implicit(..., operator_active=False)` is that passive operator: the derivative
with respect to b only (a backward-only switch; CG2D's caller takes it from the run's cg2dFullAdjoint through
mitjax/ad/modes.py, plan "Decisions 2026-10-02" item 3 revised); where no control reaches the operator (M1, R5: linear
free surface) both settings give the same derivative.

The solution carries the remat name "cg2d_x" (`SAVE_NAME`; drivers/checkpoint.py keeps it in a rematerialised
reverse pass instead of re-running the literal iteration).
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.ad_checkpoint import checkpoint_name

__all__ = ["SAVE_NAME", "interior_mask", "cg2d_implicit", "make_pcg_solve", "tile_vdot", "exchanger_vdot",
           "default_vdot", "exchange_reads_interior_only"]

SAVE_NAME = "cg2d_x"


def interior_mask(sNx, sNy, OLx, OLy):
    """[sNy+2 OLy, sNx+2 OLx] bool, True on the Fortran interior i = 1..sNx, j = 1..sNy (array index i-1+OLx)."""
    m = np.zeros((sNy + 2 * OLy, sNx + 2 * OLx), bool)
    m[OLy:OLy + sNy, OLx:OLx + sNx] = True
    return m


def _mask_halo(interior, v):
    """v on the interior, 0 on halo lanes (a select: no arithmetic on interior values)."""
    return jnp.where(interior, v, jnp.zeros((), v.dtype))


# The two halo zeroings of the derivative solve, kept as separate named steps ([L-AD-10]); test_cg2d_rule.py replaces
# `_mask_solution` by the identity to show that the dot test fails without it.
def _mask_rhs(interior, r):
    return _mask_halo(interior, r)


def _mask_solution(interior, y):
    return _mask_halo(interior, y)


def _zero_tangent(x):
    if jnp.issubdtype(jnp.result_type(x), jnp.floating):
        return jnp.zeros_like(x)
    return np.zeros(np.shape(x), jax.dtypes.float0)


def _forward_value(interior, x_full):
    """The rule's forward output: the forward solve's array itself, every lane (halos included) bit for bit. A named
    step so that test_cg2d_rule.py can plant a halo-zeroed forward value and show the gate failing."""
    return x_full


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _implicit(fns, A, b, x_first, op_consts, solve_consts, interior):
    forward_solve = fns[0]
    x_full, aux = forward_solve(A, b, x_first, op_consts, solve_consts)
    return _forward_value(interior, x_full), aux


@_implicit.defjvp
def _implicit_jvp(fns, primals, tangents):
    _, operator, derivative_solve, operator_active = fns
    A, b, x_first, op_consts, solve_consts, interior = primals
    dA, db, _, dc, _, _ = tangents
    x, aux = _implicit(fns, *primals)
    if operator_active:
        # the unknown is the interior of the solution (the operator reads only the interior of its argument)
        x_int = _mask_halo(interior, x)
        # differentiate operator(A, c, x) = b at the converged x: M dx + d_(A,c) operator . (dA, dc) = db
        _, dMx = jax.jvp(lambda A_, c_: operator(A_, c_, x_int), (A, op_consts), (dA, dc))
        rhs = _mask_rhs(interior, db - dMx)
    else:
        # the operator passive (ad/modes.py; TAF's CG2D_MAD with cg2dFullAdjoint = .FALSE., cg2d_mad.F:220-236
        # aW2d_ad = aS2d_ad = aC2d_ad = pW_ad = pS_ad = pC_ad = 0): M dx = db, dA and dc are not read
        rhs = _mask_rhs(interior, db)

    def matvec(v):
        return _mask_halo(interior, operator(A, op_consts, _mask_halo(interior, v)))

    def solve(_, r):
        return _mask_solution(interior, derivative_solve(A, op_consts, solve_consts, _mask_rhs(interior, r)))

    dx = lax.custom_linear_solve(matvec, rhs, solve, symmetric=True)
    return (x, aux), (dx, jax.tree.map(_zero_tangent, aux))


def cg2d_implicit(forward_solve, operator, derivative_solve, A, b, x_first, op_consts, solve_consts, interior, *,
                  operator_active=True):
    """Solve operator(A, op_consts, x) = b with the literal forward solve; derivatives by the implicit rule.

    forward_solve, operator, derivative_solve: functions as in the module docstring (static: module-level functions or
    `functools.partial` of hashable objects; never closures over arrays). A, op_consts, solve_consts: pytrees of
    arrays (traced jit arguments; an exchanger travels in op_consts). x_first: the first guess (no derivative).
    interior: bool mask of the Fortran interior (`interior_mask`), `[j, i]` or `[tile, j, i]` (sharded: `ex.tile_mask`,
    False on padding tiles).

    Returns (x, aux): x = the forward solve's array on every lane, halos included, bit for bit (differentiable; its
    tangent is the implicit derivative on the interior and 0 on halo lanes, exact because the caller's EXCH_XY_RL
    overwrites the halos, see the module docstring; remat name "cg2d_x"); aux = the forward solve's diagnostics (zero
    tangent).

    operator_active (static bool, a backward-only switch: mitjax/ad/modes.py): True = the derivative with respect to b
    and to the operator (A, op_consts); False = with respect to b only, the operator passive (TAF's CG2D_MAD with
    cg2dFullAdjoint = .FALSE.). The forward value and the forward program are the same for both.
    """
    x, aux = _implicit((forward_solve, operator, derivative_solve, bool(operator_active)), A, b, x_first, op_consts,
                       solve_consts, jnp.asarray(interior))
    return checkpoint_name(x, SAVE_NAME), aux


def exchange_reads_interior_only(maps, name="XY"):
    """True when the exchange `name` of exch_maps.ExchangeMaps `maps` writes every halo point, takes every source from
    an interior point and keeps every interior point: then the exchange's output does not depend on the halo lanes of
    its input, the condition under which the rule's zero halo tangent is exact (module docstring)."""
    L = maps.layout
    interior = np.broadcast_to(L.interior(), L.shape2d).reshape(-1)
    ok = True
    for key in (name,) if name in maps.maps else (name + "_u", name + "_v"):
        src, comp, _ = maps.maps[key]
        written = comp > 0
        ok &= bool(np.all(written == ~interior)) and bool(np.all(interior[src[written]]))
    return ok


# ---------------------------------------------------------------------------------------------------------------------
# default tight derivative solve


def _varying_axes(a):
    return getattr(getattr(jax.typeof(a), "mat", None), "varying", frozenset())


def tile_vdot(a, b):
    """sum(a * b) of arrays holding every tile: per-tile partial sums (XLA's order), then the tiles added in fixed
    order 0, 1, 2, ... starting from 0 (global_sum_tile.F). Deterministic for a given tiling; single device only: an
    input varying over a shard_map axis (a per-device block) is an error, since this sum would miss the other
    devices' tiles (use `exchanger_vdot`)."""
    if _varying_axes(a) or _varying_axes(b):
        raise ValueError("tile_vdot inside shard_map sums one device's tiles only: pass vdot=exchanger_vdot with the "
                         "ShardedExchanger in op_consts['ex']")
    part = jnp.sum(a * b, axis=tuple(range(1, a.ndim)))
    s = jnp.zeros((), part.dtype)
    for t in range(part.shape[0]):
        s = s + part[t]
    return s


def _pcg(matvec, precond, vdot, b, tol, max_iters):
    """Preconditioned CG for M y = b from y = 0; stops when sqrt(r.r) <= tol * sqrt(b.b) or after max_iters.
    Returns (y, iterations, final relative residual)."""
    bb = vdot(b, b)
    tol_sq = (tol * tol) * bb
    y0 = jnp.zeros_like(b)
    z0 = precond(b)
    rz0 = vdot(b, z0)

    def cond(st):
        it, _, r, _, _ = st
        return (it < max_iters) & (vdot(r, r) > tol_sq)

    def body(st):
        it, y, r, p, rz = st
        q = matvec(p)
        alpha = rz / vdot(p, q)
        y = y + alpha * p
        r = r - alpha * q
        z = precond(r)
        rz_new = vdot(r, z)
        p = z + (rz_new / rz) * p
        return it + 1, y, r, p, rz_new

    it, y, r, _, _ = lax.while_loop(cond, body, (jnp.zeros((), jnp.int32), y0, b, z0, rz0))
    rel = jnp.sqrt(vdot(r, r) / jnp.where(bb > 0, bb, 1.0))
    return y, it, rel


def default_vdot(op_consts, a, b):
    """The derivative solve's default inner product: `exchanger_vdot` when op_consts carries an exchanger ("ex"),
    else `tile_vdot` (single device; it refuses per-device blocks under shard_map)."""
    if isinstance(op_consts, dict) and "ex" in op_consts:
        return exchanger_vdot(op_consts, a, b)
    return tile_vdot(a, b)


def exchanger_vdot(op_consts, a, b):
    """sum(a * b) over the Fortran interior of every tile: the product, then the per-tile `DO j; DO i` partials and
    GLOBAL_SUM_TILE_RL through the exchanger `op_consts["ex"]` (eesupp/global_sum.py): on one device and inside
    shard_map at any P the same additions in the same order, so the derivative solve is bitwise P-independent."""
    ex = op_consts["ex"]
    L = ex.layout
    prod = a * b
    return ex.global_sum_rl(prod[..., L.OLy:L.OLy + L.sNy, L.OLx:L.OLx + L.sNx])


def _pcg_solve(operator, precond, vdot, tol, max_iters, A, op_consts, solve_consts, r):
    def matvec(v):
        return operator(A, op_consts, v)
    pre = (lambda v: v) if precond is None else (lambda v: precond(A, op_consts, v))
    y, _, _ = _pcg(matvec, pre, partial(vdot, op_consts), r, tol, max_iters)
    return y


def make_pcg_solve(operator, precond=None, vdot=default_vdot, tol=1e-13, max_iters=2000):
    """A derivative_solve: preconditioned CG of `operator` from zero to the relative residual `tol` (default 1e-13, as
    the ECCO port's transpose solve; TAF's adjoint solve uses the forward tolerance) with at most `max_iters`
    iterations. precond(A, op_consts, r) -> z (e.g. the UPDATE_CG2D preconditioner pW, pS, pC carried in op_consts),
    None for none; vdot(op_consts, a, b): the global inner product (default `default_vdot`: the Fortran-order global
    sum through op_consts["ex"] when there is one, required under shard_map; else `tile_vdot` on one device). The PCG
    reads r only on the interior: `cg2d_implicit` zeroes r's and y's halo lanes."""
    return partial(_pcg_solve, operator, precond, vdot, float(tol), int(max_iters))
