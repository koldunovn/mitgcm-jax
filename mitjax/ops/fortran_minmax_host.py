"""Host-side (numpy float64) Fortran MAX/MIN with gfortran's value at each call site, and the running-extremum chains
of the monitor (lane MON's mon_intrinsics.py, moved here by lane MINMAX, 2026-10-01). Not a Fortran file.

`MAX(a, b, p=)` / `MIN(a, b, p=)` take `p` as `mitjax.ops.fortran_minmax.MAX` does: "a" = the first argument wins a
tie (+0 vs -0) and a NaN, "b" = the second; the per-site value comes from $MJX_REFERENCE/minmax_sites
(tools/fortran_minmax_sites.py; the rule: docs/KERNEL_GUIDE.md §3, mitjax/ops/fortran_minmax.py).

The monitor statistics carry running extrema across every point of every tile, e.g. `theMin = MIN(theMin,tmpVal)`
in `DO bj; DO bi; DO k; DO j; DO i` order. `min_chain` / `max_chain` return exactly such a chain's result,
`m = init; DO (C order of values) IF (mask) m = OP(m, x)`, with the site's `p` ("a" = the running value m wins ties
and NaN, "b" = the new value x wins), computed without a Python loop. Host side: the monitor is evaluated on concrete
arrays outside the differentiated program (`monitor.monitor`).
"""

import numpy as np

LETTERS = ("a", "b")


def _check(p):
    if p not in LETTERS:
        raise ValueError(f"p must be 'a' or 'b', got {p!r}")


def MAX(a, b, *, p):
    """Fortran MAX(a, b) of two REAL*8 scalars, gfortran's value at the site: p="a": (b > a) ? b : a; "b": (a > b) ? a : b."""
    _check(p)
    a, b = np.float64(a), np.float64(b)
    if p == "a":
        return b if b > a else a
    return a if a > b else b


def MIN(a, b, *, p):
    """Fortran MIN(a, b) of two REAL*8 scalars, gfortran's value at the site: p="a": (b < a) ? b : a; "b": (a < b) ? a : b."""
    _check(p)
    a, b = np.float64(a), np.float64(b)
    if p == "a":
        return b if b < a else a
    return a if a < b else b


def _flat(values, mask=None):
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if mask is None:
        return v
    m = np.broadcast_to(np.asarray(mask, dtype=bool), np.shape(values)).reshape(-1)
    return v[m]


def _chain(init, values, mask, p, better):
    """m = init; DO ... IF (mask) m = OP(m, x) where p="a": m wins ties and NaN, p="b": x wins; `better(u, v)` is the
    strict comparison of the operation (u < v for MIN, u > v for MAX)."""
    _check(p)
    xs = _flat(values, mask)
    m = np.float64(init)
    if p == "a":
        # m = (x better than m) ? x : m.  A NaN x never enters; a NaN m stays (comparisons with NaN are false).
        if np.isnan(m):
            return m
        xs = xs[~np.isnan(xs)]
        if xs.size == 0:
            return m
        best = xs[0]
        for v in (xs.min(), xs.max()):
            if better(v, best):
                best = v
        if not better(best, m):
            return m
        return xs[np.argmax(xs == best)]           # first occurrence of the extremum (+0 == -0)
    # p="b": m = (m better than x) ? m : x.  A NaN on either side gives x, so the chain restarts after the last NaN.
    seq = np.concatenate([[m], xs])
    nan = np.isnan(seq)
    if nan.any():
        last = int(np.nonzero(nan)[0][-1])
        if last == seq.size - 1:
            return seq[-1]
        seq = seq[last + 1:]
    best = seq[0]
    for v in (seq.min(), seq.max()):
        if better(v, best):
            best = v
    return seq[np.nonzero(seq == best)[0][-1]]     # last occurrence of the extremum


def min_chain(init, values, mask=None, *, p):
    """`m = init; DO (C order of values) IF (mask) m = MIN(m, x)` with the site's `p` (module docstring).
    values: [tile, k, j, i] flattens C-wise in the Fortran loop order bj, bi, k, j, i."""
    return _chain(init, values, mask, p, lambda u, v: u < v)


def max_chain(init, values, mask=None, *, p):
    """`m = init; DO ... IF (mask) m = MAX(m, x)` with the site's `p`."""
    return _chain(init, values, mask, p, lambda u, v: u > v)


def first_masked(values, mask):
    """(value, found) of the first point in loop order where mask holds: `IF (mask .AND. noPnts) THEN theMin=tmpVal;
    noPnts=.FALSE.` (mon_calc_stats_rl.F:70-74)."""
    xs = _flat(values, mask)
    if xs.size == 0:
        return np.float64(0.0), False
    return xs[0], True
