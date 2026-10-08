"""scan_k (mitjax/ops/scan_k.py), the allow-listed k recursion of the physics code (KERNEL_GUIDE §4):
bitwise equal to a Python loop in the Fortran order (upward and downward loops), its gradients equal to the
unrolled loop's and finite, and a planted reversed iteration order is caught by the same comparison."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax.farray import FArray, loop_i, loop_j
from mitjax.ops.scan_k import level, scan_k, set_levels

NT, OL, SN, NR = 2, 2, 5, 9
B = {"i": (1 - OL, SN + OL), "j": (1 - OL, SN + OL)}


def _fields(seed=7):
    rng = np.random.default_rng(seed)
    shp = (NT, NR, SN + 2 * OL, SN + 2 * OL)
    a = -rng.uniform(0.0, 1.0, shp)
    c = -rng.uniform(0.0, 1.0, shp)
    b = 1.0 - a - c + rng.uniform(0.0, 0.1, shp)
    y = 15.0 + 5.0 * rng.standard_normal(shp)
    return [jnp.asarray(x) for x in (a, b, c, y)]


def _fa(x, name):
    return FArray(x, name, k=(1, NR), **B)


def thomas_scan(a, b, c, y, order=None):
    """Forward elimination (DO k=2,Nr) and back substitution (DO k=Nr-1,1,-1) of SOLVE_TRIDIAGONAL's
    non-KINNER branch (solve_tridiagonal.F:239-295) written with scan_k; `order` plants a wrong loop order."""
    a, b, c, y = _fa(a, "a3d"), _fa(b, "b3d"), _fa(c, "c3d"), _fa(y, "y3d")
    j, i = loop_j(*B["j"]), loop_i(*B["i"])
    cp, yp = c.local("c3d_prime"), y.local("y3d_prime")
    recVar = 1.0 / level(b, 1)[i, j]
    cp1 = level(cp, 1).at[i, j].set(level(c, 1)[i, j] * recVar)
    yp1 = level(yp, 1).at[i, j].set(level(y, 1)[i, j] * recVar)
    cp = set_levels(cp, jax.tree_util.tree_map(lambda t: t[None], cp1), range(1, 2))
    yp = set_levels(yp, jax.tree_util.tree_map(lambda t: t[None], yp1), range(1, 2))

    def forward(carry, x):
        cpm1, ypm1 = carry
        rv = 1.0 / (x["b"][i, j] - x["a"][i, j] * cpm1[i, j])
        cpk = cpm1.at[i, j].set(x["c"][i, j] * rv)
        ypk = ypm1.at[i, j].set((x["y"][i, j] - x["a"][i, j] * ypm1[i, j]) * rv)
        return (cpk, ypk), {"cp": cpk, "yp": ypk}

    ks = order or range(2, NR + 1)
    _, out = scan_k(forward, (cp1, yp1), ks,
                    lambda k: {"a": level(a, k), "b": level(b, k), "c": level(c, k), "y": level(y, k)})
    cp = set_levels(cp, out["cp"], range(2, NR + 1))
    yp = set_levels(yp, out["yp"], range(2, NR + 1))

    def backward(ykp1, x):
        yk = ykp1.at[i, j].set(x["yp"][i, j] - x["cp"][i, j] * ykp1[i, j])
        return yk, yk

    yN = level(yp, NR)
    _, ys = scan_k(backward, yN, range(NR - 1, 0, -1), lambda k: {"yp": level(yp, k), "cp": level(cp, k)})
    yo = set_levels(y, ys, range(1, NR))
    yo = set_levels(yo, jax.tree_util.tree_map(lambda t: t[None], yN), range(NR, NR + 1))
    return yo.data


def thomas_loop(a, b, c, y, reverse_forward=False):
    """The same algorithm as plain Python loops over k on [tile, k, j, i] arrays, in the Fortran order."""
    cp = [None] * (NR + 1)
    yp = [None] * (NR + 1)
    rv = 1.0 / b[:, 0]
    cp[1], yp[1] = c[:, 0] * rv, y[:, 0] * rv
    ks = range(2, NR + 1)
    if reverse_forward:
        ks = range(NR, 1, -1)
        cp[NR], yp[NR] = cp[1], yp[1]           # a wrong order still has to start somewhere
    for k in ks:
        km = k - 1 if not reverse_forward else min(k + 1, NR)
        rv = 1.0 / (b[:, k - 1] - a[:, k - 1] * cp[km])
        cp[k] = c[:, k - 1] * rv
        yp[k] = (y[:, k - 1] - a[:, k - 1] * yp[km]) * rv
    out = [None] * (NR + 1)
    out[NR] = yp[NR]
    for k in range(NR - 1, 0, -1):
        out[k] = yp[k] - cp[k] * out[k + 1]
    return jnp.stack(out[1:], axis=1)


def test_scan_k_bitwise_vs_python_loop():
    a, b, c, y = _fields()
    got = np.asarray(jax.jit(thomas_scan)(a, b, c, y))
    want = np.asarray(jax.jit(thomas_loop)(a, b, c, y))
    assert np.all(np.isfinite(got)) and np.all(np.isfinite(want))
    assert np.array_equal(got.view(np.int64), want.view(np.int64)), \
        f"{np.count_nonzero(got != want)} of {got.size} points differ"


def test_scan_k_planted_reversed_order_fails():
    """Negative control: the forward sweep run DO k=Nr,2,-1 (wrong order) differs from the Fortran order."""
    a, b, c, y = _fields()
    want = np.asarray(jax.jit(thomas_loop)(a, b, c, y))
    bad = np.asarray(jax.jit(lambda *v: thomas_loop(*v, reverse_forward=True))(a, b, c, y))
    n_bad = np.count_nonzero(bad != want)
    print(f"reversed forward sweep: {n_bad} of {want.size} points differ")
    assert n_bad > 0.9 * want.size
    # and the scan with a planted wrong loop order is caught by the same bitwise comparison
    got_bad = np.asarray(jax.jit(lambda *v: thomas_scan(*v, order=range(NR, 1, -1)))(a, b, c, y))
    assert np.count_nonzero(got_bad != want) > 0


def test_scan_k_downward_loop_returns_increasing_k():
    x = jnp.arange(NT * NR * (SN + 2 * OL) ** 2, dtype=jnp.float64).reshape(NT, NR, SN + 2 * OL, SN + 2 * OL)
    A = _fa(x, "A")
    order = []

    def it(carry, xk):
        return carry + 1.0, xk[loop_i(*B["i"]), loop_j(*B["j"])] * 0.0 + carry

    def lev(k):
        order.append(k)
        return level(A, k)

    _, out = scan_k(it, jnp.float64(0.0), range(NR, 0, -1), lev)
    assert order == list(range(NR, 0, -1))
    # iteration n (0-based) wrote the value n at level NR - n; out is in increasing k
    assert np.array_equal(np.asarray(out[:, 0, 0, 0]), np.arange(NR - 1, -1, -1, dtype=float))


def test_scan_k_gradients_equal_unrolled_and_finite():
    a, b, c, y = _fields()
    w = jnp.asarray(np.random.default_rng(3).standard_normal(a.shape))

    def cost(f):
        return lambda a, b, c, y: jnp.sum(w * f(a, b, c, y))

    g_scan = jax.jit(jax.grad(cost(thomas_scan), argnums=(0, 1, 2, 3)))(a, b, c, y)
    g_loop = jax.jit(jax.grad(cost(thomas_loop), argnums=(0, 1, 2, 3)))(a, b, c, y)
    for gs, gl, name in zip(g_scan, g_loop, "abcy"):
        gs, gl = np.asarray(gs), np.asarray(gl)
        assert np.all(np.isfinite(gs)), name
        np.testing.assert_allclose(gs, gl, rtol=1e-13, atol=1e-13 * np.max(np.abs(gl)), err_msg=name)


def test_scan_k_rejects_non_range():
    with pytest.raises(TypeError):
        scan_k(lambda c, x: (c, x), 0.0, [1, 2, 3], lambda k: jnp.float64(k))


def _levels_body(k, c):
    """A per-level caller body of the converted kind: branches on k = NR and k = 1, a two-slot parity choice
    (1+MOD(NR-k,2)), level k-1 read, level k written, and the written array returned under another name."""
    acc, out = c
    kk = k - 1
    v = out.data[:, getattr(kk, "value", kk)]
    f = 2.0 if k == NR else (3.0 if k == 1 else 1.0)
    slot = 1 + (NR - k) % 2
    acc = acc * 0.5 + v * f + slot
    km1 = k - 2 if k >= 2 else kk
    w = out.data[:, getattr(km1, "value", km1)]
    return acc + 0.25 * w, FArray(out.data.at[:, getattr(kk, "value", kk)].set(acc), "renamed", k=(1, NR), **B)


@pytest.mark.parametrize("down", [False, True])
@pytest.mark.parametrize("nr_lo", [1, 3, NR - 1])
def test_scan_levels_peel_bitwise_vs_python_loop(down, nr_lo):
    """scan_levels(..., peel=(1, 1)) == the Python loop in the Fortran order, bitwise (any range, incl. an empty
    scanned middle); without the peel the KIdx raises at the undecided k = 1 / k = NR branch."""
    from mitjax.ops.scan_k import scan_levels
    x = jnp.asarray(np.random.default_rng(5).standard_normal((NT, NR, SN + 2 * OL, SN + 2 * OL)))
    c0 = (x[:, 0] * 0.0, _fa(x, "out"))

    def loop(c):
        for k in (range(NR, nr_lo - 1, -1) if down else range(nr_lo, NR + 1)):
            c = _levels_body(k, c)
        return c

    got = jax.jit(lambda c: scan_levels(_levels_body, c, nr_lo, NR, down=down, peel=(1, 1)))(c0)
    want = jax.jit(loop)(c0)
    for g, w in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want)):
        assert np.array_equal(np.asarray(g).view(np.int64), np.asarray(w).view(np.int64))
    if nr_lo == 1:                     # k = 1 (upward) / k = NR (downward) inside the scanned range
        with pytest.raises(TypeError):
            jax.jit(lambda c: scan_levels(_levels_body, c, nr_lo, NR, down=down))(c0)


def test_kidx_rsub_parity():
    from mitjax.farray import KIdx
    k = KIdx(jnp.int32(4), 2, 8, 0)
    m = 9 - k
    assert (m.lo, m.hi, m.parity) == (1, 7, 1) and 1 + m % 2 == 2
