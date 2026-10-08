"""Fortran MAX and MIN of REAL values with the value gfortran gives at each call site (lane MINMAX, 2026-10-01).

Every call states which argument wins a tie (+0 vs -0) and a NaN at that Fortran site:

    MAX(a, b, p="a")        # :NN   the first argument wins ties and NaN: (b > a) ? b : a
    MAX(a, b, p="b")        # :NN   the second argument wins:              (a > b) ? a : b
    MAX(a1, a2, a3, p="ab") # :NN   n arguments: the left fold M = a1; M = MAX(M, a_k), one letter per step,
                            #       "a" = the running value M wins, "b" = the new argument a_k wins
(MIN likewise with <). `p` comes from the oracle's own compilation of the site, the table
$MJX_REFERENCE/minmax_sites/<build>.json written by tools/fortran_minmax_sites.py, and the `# :NN` comment cites the
Fortran line; mitjax/tests/test_minmax_sites.py checks both for every call in mitjax/model and mitjax/pkg.

Why per site (docs/KERNEL_GUIDE.md §3): gfortran 11.2.0 at -O0 on x86-64 lowers MAX(a1,...,an) to
`M = a1; M = MAX_EXPR <a_k, M>` and each MAX_EXPR to one maxsd/minsd, which returns its source operand on a tie or a
NaN (Intel SDM). Which argument ends up as the source depends on the operand canonicalisation of the GIMPLE
statement (constants and loads of memory-resident variables go second) and on the RTL expander's and register
allocator's choices for a commutative operation, so it differs between call sites: e.g. MAX(0.D0,-0.D0) gives -0 in
GAD_DST3FL_ADV_X, and GAD_FLUX_LIMITER.h's MIN(2.D0,Cr) returns 2 for a NaN Cr inside the statement function but NaN
when the same expression is written inline (mitjax/tests/fortran_minmax/sfprobe.F). jnp.maximum/jnp.minimum
propagate NaN and return the first argument on a tie, which is gfortran's value at no site in general.

Derivative: JAX's own derivative of max/min (`jnp.maximum`/`jnp.minimum`: the selected argument's tangent, half of
each at an exact tie), the JAX-optimal default (Nikolay 2026-10-01, [L-CONF-12]); TAF follows the Fortran branch at
a tie (a later TAF switch). Only the forward value differs from jnp.maximum/minimum, and only on ties and NaN.

Allowed in physics code (`mitjax/model`, `mitjax/pkg`); `jax.custom_jvp` lives here, as in `mitjax/ops/safe.py`.
The host-side (numpy) counterparts and the running-extremum chains of the monitor are in
`mitjax/ops/fortran_minmax_host.py`.
"""

import os
import warnings

import jax
import jax.numpy as jnp

__all__ = ["MAX", "MIN", "MAX_CHAIN", "LETTERS", "build_winner", "MinMaxDefaultWarning"]

LETTERS = ("a", "b")


def _step(op, winner):
    """One gfortran MAX_EXPR/MIN_EXPR step `M = OP(M, x)` where `winner` ("a" = M, "b" = x) wins ties and NaN."""
    better = (lambda u, v: u > v) if op == "MAX" else (lambda u, v: u < v)
    ref = jnp.maximum if op == "MAX" else jnp.minimum

    @jax.custom_jvp
    def f(m, x):
        if winner == "a":
            return jnp.where(better(x, m), x, m)       # (x better than M) ? x : M
        return jnp.where(better(m, x), m, x)           # (M better than x) ? M : x

    @f.defjvp
    def _jvp(primals, tangents):
        return f(*primals), jax.jvp(ref, primals, tangents)[1]

    return f


_STEPS = {(op, w): _step(op, w) for op in ("MAX", "MIN") for w in LETTERS}


def _fold(op, args, p):
    if len(args) < 2:
        raise TypeError(f"{op} needs at least two arguments")
    if not isinstance(p, str) or len(p) != len(args) - 1 or any(c not in LETTERS for c in p):
        raise ValueError(f"{op} of {len(args)} arguments: p must be {len(args) - 1} letter(s) of 'a'/'b', got {p!r}")
    m = args[0]
    for w, x in zip(p, args[1:]):
        m = _STEPS[(op, w)](m, x)
    return m


def MAX(*args, p):
    """Fortran MAX(a1, a2, ...) of REAL values with gfortran's value at the site (module docstring for `p`)."""
    return _fold("MAX", args, p)


def MIN(*args, p):
    """Fortran MIN(a1, a2, ...) of REAL values with gfortran's value at the site (module docstring for `p`)."""
    return _fold("MIN", args, p)


def MAX_CHAIN(init, values, *, p, acc, ex=None):
    """A running Fortran MAX over the tiles and a DO nest, `m = init; DO bj; DO bi; DO ...; m = MAX(...)`, with
    gfortran's value at the site, vectorised.

    values: [T, ...] the operands the DO nest visits on each tile held, flattened C-wise in the loop order (outer to
    inner loop, then the statements of one iteration); `ex.all_tiles` (eesupp: every real tile in tile order, on every
    device and every P) joins the tiles into one chain in tile order, the chain of the oracle's single process (its
    _GLOBAL_MAX is then the identity, global_max.F); ex=None: all tiles are in `values`. `acc` is the Fortran argument
    that holds the running value ("a" or "b"), `p` the site's winner of ties and NaN (the table, as for MAX).
    - The new value wins (p != acc), e.g. `rhsMax = MAX(ABS(x), rhsMax)`: a NaN x makes m NaN and the next x replaces
      it, so the result is the maximum of the elements after the last NaN in loop order (init included when there is
      no NaN), or that NaN when it is the last element.
    - The running value wins (p == acc): a NaN x never enters; a NaN init stays.
    Ties: the values must hold no -0 (ABS results; init >= +0), so an equal value is the same number and only NaN
    can make the order matter. Derivative: JAX's own of the max/where expression (the selected elements' tangents)."""
    if p not in LETTERS or acc not in LETTERS:
        raise ValueError(f"MAX_CHAIN: p and acc must be 'a' or 'b', got {p!r}, {acc!r}")
    x = jnp.asarray(values)
    x = x.reshape(x.shape[0], -1)
    n = x.shape[1]
    isn = jnp.isnan(x)
    ninf = jnp.asarray(-jnp.inf, x.dtype)
    gather = (lambda a: a) if ex is None else ex.all_tiles
    if p == acc:                                        # the running value wins: NaN values never enter
        m = jnp.max(gather(jnp.max(jnp.where(isn, ninf, x), axis=1)))
        return jnp.where(jnp.isnan(init), init, jnp.maximum(init, m))
    pos = jnp.arange(n)
    last = jnp.max(jnp.where(isn, pos, -1), axis=1)                          # [T] last NaN, -1: none
    after = jnp.max(jnp.where(pos[None, :] > last[:, None], x, ninf), axis=1)
    lastval = jnp.take_along_axis(x, jnp.clip(last, 0, n - 1)[:, None], axis=1)[:, 0]
    v = jnp.where(last == n - 1, lastval, after)          # the tile's chain value from a NaN start, or its maximum
    summary = gather(jnp.stack([(last >= 0).astype(x.dtype), v], axis=1))     # [nTiles, 2] in tile order
    has, v = summary[:, 0] > 0, summary[:, 1]
    idx = jnp.arange(v.shape[0])
    s = jnp.max(jnp.where(has, idx, -1))                                      # the last tile with a NaN
    base = jnp.where(s >= 0, v[jnp.clip(s, 0, v.shape[0] - 1)], init)
    tail = idx > s
    tail_max = jnp.max(jnp.where(tail, v, ninf))
    return jnp.where(jnp.any(tail), jnp.where(jnp.isnan(base), tail_max, jnp.maximum(base, tail_max)), base)


class MinMaxDefaultWarning(UserWarning):
    """A build-dependent MAX/MIN site (plan decision 15) took its documented default winner (plan decision 7)."""


_WARNED = set()          # (build identity, site) warned about in this process: one warning per build and site, so a
                         # new build in the same process warns again (lane APIF); _WARNED.clear() forgets them all

STRICT_ENV = "MJX_MINMAX_STRICT"


def strict():
    """MJX_MINMAX_STRICT = 1: a known build without a table entry is refused (our tests: conftest.py sets it);
    unset or 0: it takes the documented default with one warning, as any other build (plan decision 7, revised
    2026-10-07). Any other value is an error."""
    v = os.environ.get(STRICT_ENV, "").strip()
    if v not in ("", "0", "1"):
        raise ValueError(f"{STRICT_ENV}={v!r}: 1 (refuse a known build without a table entry), 0 or unset")
    return v == "1"


def build_winner(cfg, table, site, defaults=None):
    """Plan decision 15 (build-dependent MAX/MIN winners): the winner `p` at `site` of the build `cfg` describes.

    Where gfortran compiles the same MAX/MIN statement to different winners in different builds (test_minmax_sites.py
    KNOWN_DIFFER), the kernel module holds a static table `MINMAX_BUILD = {"<pkg/.../file.F>:<line>": {build: p}}`
    naming the oracle builds that execute the statement and their winner from $MJX_REFERENCE/minmax_sites/<build>.json
    (the audit checks the table, the call's literal `p` and its `# MINMAX-BUILD: <build>` marker against that json),
    and `MINMAX_DEFAULT = {site: p}`, the documented default winner (`defaults`).
    The build of `cfg` is found by its identity, a hash of its preprocessed options and SIZE.h
    (mitjax/config/build_identity.py, docs plan 20261006 Task 4), not by its name: a renamed or moved copy of a
    verification experiment is the same build, an edited code directory is another one.
    - A verification build the oracle compiled (build_identity.KNOWN, label `<experiment>-<code dir>-<commit[:7]>`)
      with a table entry takes its measured winner.
    - A known build without an entry (the oracle never measured the statement in it, e.g. a namelist option of the
      user's run switches the statement on) is refused with MJX_MINMAX_STRICT=1 (strict(); our tests) and otherwise
      treated as any other build (plan decision 7, revised by Nikolay 2026-10-07: "default with warning for users,
      strict in our tests").
    - Any other build (plan decision 7) takes the documented default `defaults[site]` -- the winner of most of the
      measured builds that compile the site (KNOWN_DIFFER; the audit test_minmax_sites.py checks it) -- with one
      MinMaxDefaultWarning per build, site and process: bitwise agreement with that build's own gfortran is then not
      guaranteed at this statement. Without a default it raises NotImplementedError."""
    from mitjax.config import build_identity
    build = build_identity.label(cfg)
    if build is not None:
        hits = {b: p for b, p in table.get(site, {}).items() if b.rsplit("-", 1)[0] == build}
        if len(hits) == 1:
            return next(iter(hits.values()))
        if len(hits) > 1 or strict():
            raise NotImplementedError(f"{site}: the MAX/MIN winner of build {build} is build-dependent and not in the "
                                      f"kernel's table {table.get(site)} (plan decision 15; {STRICT_ENV}=1)")
    if defaults is None or site not in defaults:
        raise NotImplementedError(f"{site}: build-dependent MAX/MIN winner and no documented default "
                                  "(plan decisions 7, 15)")
    p = defaults[site]
    key = (build_identity.identity(cfg), site)
    if key not in _WARNED:
        _WARNED.add(key)
        if build is None:
            which = (f"this build ({cfg.experiment}/{cfg.code_dir}, identity {build_identity.identity(cfg)[:12]}) is "
                     "not a build the Fortran oracle measured")
        else:
            which = (f"this build ({build}, {cfg.experiment}/{cfg.code_dir}) is a verification build the Fortran "
                     "oracle compiled, but its winner at this statement was never measured")
        warnings.warn(f"{site}: the MAX/MIN winner is build-dependent (plan decision 15) and {which}: the documented "
                      f"default p={p!r} is used (plan decision 7); bitwise agreement with a gfortran build of it is "
                      "not guaranteed at this statement", MinMaxDefaultWarning, stacklevel=2)
    return p
