"""`scan_k`: a Fortran `DO k` loop whose iterations depend on each other (a recursion in k), for physics code.

The allow-listed way to write a k recursion in `mitjax/model/` and `mitjax/pkg/` (docs/KERNEL_GUIDE.md §4; raw
`lax.scan` is banned there). Typical uses: the forward elimination and back substitution of a tridiagonal solve
(SOLVE_TRIDIAGONAL, IMPLDIFF), integrations from the surface or from the bottom.

What a Fortran reader writes, and what it means:

    C     DO k=2,Nr                                    (bet(k) needs bet(k-1))
    C      gam(i,j,k) = c(i,j,k-1)*bet(i,j,k-1)
    C      bet(i,j,k) = 1. _d 0 / ( b(i,j,k) - a(i,j,k)*gam(i,j,k) )
    C     ENDDO

    def level_k(k):                                  # what iteration k reads, at Fortran level k (a Python int)
        return {"c_km1": level(c, k-1), "a": level(a, k), "b": level(b, k), "gam": level(gam, k)}

    def iteration(bet_km1, x):                       # ONE pass of the loop body; bet_km1 = bet(:,:,k-1)
        gam_k = x["gam"].at[i, j].set(x["c_km1"][i, j]*bet_km1[i, j])
        bet_k = bet_km1.at[i, j].set(1.0/(x["b"][i, j] - x["a"][i, j]*gam_k[i, j]))
        return bet_k, {"gam": gam_k, "bet": bet_k}   # (what iteration k+1 reads, what level k keeps)

    bet_last, out = scan_k(iteration, level(bet, 1), range(2, Nr+1), level_k)
    gam = set_levels(gam, out["gam"], range(2, Nr+1))
    bet = set_levels(bet, out["bet"], range(2, Nr+1))

* `ks` is a Python `range` written as the Fortran DO statement: `DO k=2,Nr` is `range(2, Nr+1)`, `DO k=Nr-1,1,-1` is
  `range(Nr-1, 0, -1)`. Iterations run in exactly that order, one after the other, as in the Fortran.
* `level_k(k)` is called at trace time once for every k of `ks` (k a Python int): it returns the level-k inputs of
  iteration k (any pytree of arrays: `level(A, k)` slices of FArrays, scalars such as `recip_drF[k]`). Reading
  level k-1 or k+1 is written there, with its Fortran index.
* `iteration(carry, x)` is the loop body: `carry` holds what iteration k reads from iteration k-1 (k+1 for a downward
  loop), `x` the level-k inputs; it returns the new carry and the level-k results. It runs inside `lax.scan`, so it is
  traced once and must not branch on k (k-dependent values come in through `x`).
* The result `out` has the level-k results of every iteration, each leaf stacked along a level axis in INCREASING k
  (for a downward loop too); `set_levels(A, out_leaf, ks)` writes them into an FArray at levels `ks`.

Why a scan and not a Python loop: the body is traced and compiled once instead of Nr times (compile time and the
size of the program grow with Nr), while the values are bitwise those of the Python loop in the Fortran order
(mitjax/tests/test_scan_k.py). Reverse mode differentiates through it as through the unrolled loop.

This module may use JAX transforms (`mitjax/ops/` is outside the physics directories).
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from mitjax.farray import FArray, KIdx

__all__ = ["scan_k", "level", "set_levels"]


def level(A, k):
    """The level-k slice A(:,:,k) of an FArray declared with (i, j, k) as an FArray declared (i, j) with the same
    i, j bounds and tiling; k is a Fortran index (a Python int, or a traced level index KIdx of a level scan whose
    whole range lies) inside A's declared k bounds."""
    dims = {d[0]: d for d in A.dims}
    if "k" not in dims or "i" not in dims or "j" not in dims:
        raise ValueError(f"level: {A.decl()} is not an (i, j, k) array")
    _, klo, khi = dims["k"]
    if isinstance(k, KIdx):
        if not (klo <= k.lo and k.hi <= khi):
            raise IndexError(f"level: {A.name}(:,:,{k}) is outside the declared k = {klo}:{khi} of {A.decl()}")
        kk = k.value - klo
    elif not (isinstance(k, int) and klo <= k <= khi):
        raise IndexError(f"level: {A.name}(:,:,{k}) is outside the declared k = {klo}:{khi} of {A.decl()}")
    else:
        kk = k - klo
    data = A.data[:, kk] if A.tiled else A.data[kk]
    _, ilo, ihi = dims["i"]
    _, jlo, jhi = dims["j"]
    return FArray(data, A.name, i=(ilo, ihi), j=(jlo, jhi), tiled=A.tiled)     # same name at every k: one pytree


def set_levels(A, stacked, ks):
    """A with A(:,:,k) = the level-k slice of `stacked` for every k in `ks` (the `out` leaf of scan_k: an FArray
    declared (i, j) whose storage has a leading level axis in increasing k, or a plain array [level, tile, j, i]);
    every other level keeps A's values."""
    dims = {d[0]: d for d in A.dims}
    _, klo, khi = dims["k"]
    kk = sorted(ks)
    if not kk or kk[0] < klo or kk[-1] > khi:
        raise IndexError(f"set_levels: levels {kk[:1]}..{kk[-1:]} outside the declared k = {klo}:{khi} of "
                         f"{A.decl()}")
    data = stacked.data if isinstance(stacked, FArray) else stacked
    if data.shape[0] != len(kk):
        raise ValueError(f"set_levels: {data.shape[0]} stacked levels for {len(kk)} levels of {A.name}")
    pos = np.asarray(kk) - klo                         # storage positions of the Fortran levels
    if A.tiled:
        new = A.data.at[:, pos].set(jnp.moveaxis(data, 0, 1))
    else:
        new = A.data.at[pos].set(data)
    return FArray(new, A.name, tiled=A.tiled, _dims=A.dims)


def scan_k(iteration, carry, ks, level_k):
    """DO k = ks (Fortran order): carry, out_k = iteration(carry, level_k(k)). Returns (carry, out) with `out` the
    pytree of level-k results stacked along a leading level axis in increasing k (see the module docstring)."""
    if not isinstance(ks, range):
        raise TypeError("scan_k: ks must be a range written as the Fortran DO statement, e.g. range(2, Nr+1)")
    if len(ks) == 0:
        raise ValueError("scan_k: empty DO loop")
    xs = [level_k(k) for k in ks]
    stacked = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *xs)
    carry, out = lax.scan(iteration, carry, stacked)
    if ks.step < 0:                                    # stacked in loop order; return in increasing k
        out = jax.tree_util.tree_map(lambda a: a[::-1], out)
    return carry, out


def _carry_names(c_in, c_out):
    """`c_out` with the FArray names of `c_in` (a name is pytree aux data, and a routine may return its dummy
    argument under its own name: a lax.scan carry must keep its type); dims and tiling are left to the scan's check."""
    def isf(x):
        return isinstance(x, FArray)
    li = jax.tree_util.tree_leaves(c_in, is_leaf=isf)
    lo_, to = jax.tree_util.tree_flatten(c_out, is_leaf=isf)
    if len(li) != len(lo_):
        return c_out
    new = [FArray(b.data, a.name, tiled=b.tiled, _dims=b.dims) if isf(a) and isf(b) and a.name != b.name else b
           for a, b in zip(li, lo_)]
    return jax.tree_util.tree_unflatten(to, new)


def scan_levels(body, carry, lo, hi, period=2, down=False, with_out=False, peel=(0, 0)):
    """`DO k = lo, hi` (or `DO k = hi, lo, -1` with down=True) of a per-level CALLER -- DYNAMICS' level loop calling
    CALC_PHI_HYD(k), MOM_FLUXFORM(k), TIMESTEP(k), TEMP_INTEGRATE's calling GAD_CALC_RHS(k), ... -- as a lax.scan
    whose body calls the unchanged per-level routines with k a traced level index `KIdx` (mitjax/farray.py):
    `carry = body(k, carry)` for every k in the Fortran order (KERNEL_GUIDE §4, decided 2026-10-02).

    The body is traced for `period` consecutive levels per scan iteration (k of known parity, so the two-slot
    rotations kUp = 1+MOD(k+1,2) stay static choices): the per-level code is compiled `period` times instead of
    hi-lo+1 times. Levels left over after whole periods run with static ints. The caller runs the levels whose
    static branches differ (typically k = 1 and k = Nr) itself; a KIdx raises on any test its range lo..hi does
    not decide, so a missing peel cannot go unnoticed. Values are bitwise those of the unrolled loop
    (mitjax/tests/test_scan_k.py); reverse mode may sum a value's cotangent contributions in another order.

    with_out=True: `body` returns (carry, y) with y a pytree of level-k values (e.g. the dump-gate probes); then
    returns (carry, [(k, y_k) for every k in loop order]) with y_k taken out of the scan's stacked outputs.

    peel=(a, b): the levels lo .. lo+a-1 and hi-b+1 .. hi run with static ints (the levels whose branches a KIdx
    cannot decide), in the Fortran order around the scan of lo+a .. hi-b; all levels static when that is empty."""
    from mitjax.farray import KIdx
    if peel != (0, 0):
        mlo, mhi = lo + peel[0], hi - peel[1]
        if mlo > mhi:
            ks = range(lo, hi + 1) if not down else range(hi, lo - 1, -1)
            before, after = list(ks), []
        else:
            before = list(range(lo, mlo)) if not down else list(range(hi, mhi, -1))
            after = list(range(mhi + 1, hi + 1)) if not down else list(range(mlo - 1, lo - 1, -1))
        outs = []
        for k in before:
            r = body(k, carry)
            carry = r[0] if with_out else r
            outs += [(k, r[1])] if with_out else []
        if mlo <= mhi:
            r = scan_levels(body, carry, mlo, mhi, period, down, with_out)
            carry = r[0] if with_out else r
            outs += r[1] if with_out else []
        for k in after:
            r = body(k, carry)
            carry = r[0] if with_out else r
            outs += [(k, r[1])] if with_out else []
        return (carry, outs) if with_out else carry
    n = (hi - lo + 1) // period
    outs = []

    def call(k, c):
        r = body(k, c)
        return r if with_out else (r, None)

    if not down:
        first, sign = lo, 1
    else:
        first, sign = hi, -1
    if n >= 1:
        par0 = first % 2

        def it(c, p):
            k0 = first + sign * period * p
            ys = []
            for q in range(period):
                kq_first = first + sign * q
                kq_last = first + sign * (q + period * (n - 1))
                kk = KIdx(k0 + sign * q, min(kq_first, kq_last), max(kq_first, kq_last), (par0 + q) % 2)
                c_prev = c
                c, y = call(kk, c)
                c = _carry_names(c_prev, c)
                ys.append(y)
            return c, tuple(ys)
        carry, ystack = lax.scan(it, carry, jnp.arange(n, dtype=jnp.int32))
        if with_out:
            for p_ in range(n):
                for q in range(period):
                    outs.append((first + sign * (period * p_ + q), jax.tree_util.tree_map(lambda a: a[p_], ystack[q])))
    rest = range(first + sign * period * n, (hi + 1) if not down else (lo - 1), sign)
    for k in rest:
        carry, y = call(k, carry)
        if with_out:
            outs.append((k, y))
    return (carry, outs) if with_out else carry
