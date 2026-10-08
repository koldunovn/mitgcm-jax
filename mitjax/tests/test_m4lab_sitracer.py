"""lab_sea/input (M4 step 5, lane M4LAB session 4): derivatives of the new SITRACER kernels (L3), which have no
forward-only solver on their path: SEAICE_TRACER_PHYS (the thermodynamic dilution / melt and the 'age' clock) and
SEAICE_ADVDIFF's tracer part (OS7MP advection of SItracer*mate, the scaling back with ALLOW_SITRACER_ADVCAP's cap and
the negative-value removal). The forward is gated end to end in test_m4lab_run.py (the 9-step run's %MON
seaice_sitracer lines and the siTrac records of pickup_seaice.0000000010).

Per kernel, at a smooth sea-ice state on every lane (m4lab_gate.smooth_sitracer_sf: every IF of the kernels decided
by margins far larger than the FD steps): the reverse-mode gradient of a weighted sum of the outputs is finite on every
lane (interior, halo) and nonzero; the tangent (jvp) and adjoint (vjp) agree in a dot test with random directions on
every lane (bar 1e-12 relative); a central FD along a random direction agrees with the tangent (bar 1e-7 relative,
best of h = 1e-1 .. 1e-5 times a direction of 1 % of the inputs). Negative control of the FD gate: SEAICE_TRACER_PHYS
with the derivative of its guarded divisions (growFact, expandFact) scaled by 1 + 1e-4, values unchanged (a
jax.custom_jvp wrapper of safe_div planted by monkeypatch): the best FD error rises above 1e-7."""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from mitjax.tests import m4lab_gate as L


@pytest.fixture(scope="module")
def md():
    m, _ = L.stub_model()
    return m, L.smooth_sitracer_sf(m)


def _weights(ys, seed):
    rng = np.random.default_rng(seed)
    return tuple(jnp.asarray(rng.standard_normal(np.shape(y))) for y in ys)


def _cost(f, sf, w):
    def J(x):
        return sum(jnp.sum(wi*yi) for wi, yi in zip(w, f(x, sf)))
    return J


def _check(m, sf, kernel, seed=11):
    f, xin = L.sitracer_kernel_fn(m, kernel)
    x0 = tuple(sf[n].data for n in xin)
    y0 = jax.jit(f)(x0, sf)
    w = _weights(y0, seed)
    J = jax.jit(_cost(f, sf, w))
    g = jax.jit(jax.grad(J))(x0)
    rng = np.random.default_rng(seed + 1)
    dx = tuple(jnp.asarray(rng.standard_normal(np.shape(x)))*jnp.abs(x)*1e-2 + 1e-6 for x in x0)
    dy = _weights(y0, seed + 2)
    _, tan = jax.jvp(lambda x: f(x, sf), (x0,), (dx,))
    _, vjp = jax.vjp(lambda x: f(x, sf), x0)
    adj = vjp(dy)[0]
    lhs = sum(float(jnp.sum(a*b)) for a, b in zip(tan, dy))
    rhs = sum(float(jnp.sum(a*b)) for a, b in zip(dx, adj))
    t = sum(float(jnp.sum(a*b)) for a, b in zip(g, dx))
    fd = []
    for h in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        xp = tuple(x + h*d for x, d in zip(x0, dx))
        xm = tuple(x - h*d for x, d in zip(x0, dx))
        fd.append(abs((float(J(xp)) - float(J(xm)))/(2*h) - t)/abs(t))
    return g, (lhs, rhs), t, fd, J, x0, dx


@pytest.mark.parametrize("kernel", ["tracer_phys", "advdiff"])
def test_sitracer_kernel_gradient(md, kernel):
    """Finite and nonzero gradient on every lane, dot test < 1e-12, FD < 1e-7 (measured: dev job of session 4)."""
    m, sf = md
    g, (lhs, rhs), t, fd, *_ = _check(m, sf, kernel)
    print(kernel, "dot", lhs, rhs, abs(lhs - rhs)/abs(lhs), "tangent", t, "FD rel. errors", fd)
    assert all(bool(jnp.all(jnp.isfinite(gi))) for gi in g)
    assert int(jnp.count_nonzero(g[0])) > 100, int(jnp.count_nonzero(g[0]))
    assert abs(lhs - rhs) <= 1e-12*abs(lhs), (lhs, rhs)
    assert min(fd) < 1e-7, fd


def _planted_safe_div(fac):
    """safe_div with its derivative scaled by `fac` and its value unchanged."""
    from mitjax.ops.safe import safe_div

    @jax.custom_jvp
    def sd(a, b, mask):
        return safe_div(a, b, mask)

    @sd.defjvp
    def _(primals, tangents):
        a, b, mask = primals
        da, db, _ = tangents
        y, dy = jax.jvp(lambda a_, b_: safe_div(a_, b_, mask), (a, b), (da, db))
        return y, dy*fac
    return sd


def test_sitracer_fd_negative_control(md, monkeypatch):
    """The FD gate bites: SEAICE_TRACER_PHYS with the derivative of growFact / expandFact (safe_div) scaled by
    1 + 1e-4 (values unchanged): the best FD error along the random direction is above 1e-7."""
    import mitjax.pkg.seaice.seaice_tracer_phys as TP
    m, sf = md
    monkeypatch.setattr(TP, "safe_div", _planted_safe_div(1. + 1e-4))
    _, _, t, fd, *_ = _check(m, sf, "tracer_phys")
    print("planted safe_div derivative x (1 + 1e-4): tangent", t, "FD rel. errors", fd)
    assert min(fd) > 1e-7, fd
