"""R1 tutorial_barotropic_gyre gradients through the scan driver and the checkpoint drivers (plan Task 12):

* full-run gradient finiteness: dJ/d(every leaf of the initial carry: all State fields, FFIELDS.h, phi0surf) through
  the 10-step run as `integrate` (mitjax/drivers/checkpoint.py, per-step remat) under grad.value_and_grad, finite on
  every point (halos, land); J = a weighted sum of etaN and vVel at the end; the "sqrt" schedule gives the same J
  and the same gradient (bitwise);
* FD h-sweep of J w.r.t. one initial etaN point and one initial uVel point (interior, wet) against the AD directional
  derivative, in a copy of the experiment with cg2dTargetResidual = 1e-13 (the forward solver's noise floor: the
  experiment's 1e-7 lets the FD wander with the iteration count; a change to the experiment, not a tolerance);
* TL vs adjoint dot test of the 10-step map (uVel0, etaN0) -> (etaN, uVel, vVel) with tile-edge cotangents.
"""

import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from mitjax.tests.test_r1_driver import model  # noqa: E402

_M = {}


def _model(tag, overrides=None):
    if tag not in _M:
        _M[tag] = model(tag, overrides)
    return _M[tag]


def ck_step(m):
    """integrate's step(model, st, x): one FORWARD_STEP with the counters in the carry, x = iloop."""
    def step(arrays, st, iloop):
        carry, t, it = st
        carry, t, it, _ = m.step(arrays, carry, iloop, t, it)
        return (carry, t, it)
    return step


def init_fn(theta, st):
    """theta = (State, ff, phi0surf) of the initial carry."""
    carry, t, it = st
    return ((theta[0], theta[1], theta[2], carry[3]), t, it)


def weights(shape):
    return jnp.cos(jnp.arange(int(np.prod(shape))).reshape(shape) * 0.37)


def final_cost(arrays, st):
    state = st[0][0]
    w = weights(state.etaN.data.shape)
    return jnp.sum(w * state.etaN.data) + jnp.sum(w[:, None] * state.vVel.data)


def setup(m, n=10):
    from mitjax.drivers.checkpoint import stack_steps
    carry = m.initial_carry()
    t, it = m.start_counters()
    st0 = (carry, t, it)
    xs = jnp.arange(1, n + 1, dtype=jnp.int32)
    theta = (carry[0], carry[1], carry[2])
    return st0, xs, theta


def test_full_gradient_finite_and_schedules():
    from mitjax.drivers.grad import value_and_grad
    m = _model("grad")
    st0, xs, theta = setup(m)
    J, g = value_and_grad(ck_step(m), theta, m.arrays, st0, xs, final_cost=final_cost, init_fn=init_fn,
                          schedule="step")
    bad = [jax.tree_util.keystr(p) for p, x in jax.tree_util.tree_flatten_with_path(g)[0]
           if jnp.issubdtype(x.dtype, jnp.floating) and not bool(jnp.all(jnp.isfinite(x)))]
    assert np.isfinite(float(J)) and not bad, bad
    gs = jax.tree_util.tree_leaves(g)
    assert any(float(jnp.max(jnp.abs(x))) > 0 for x in gs)
    J2, g2 = value_and_grad(ck_step(m), theta, m.arrays, st0, xs, final_cost=final_cost, init_fn=init_fn,
                            schedule="sqrt")
    assert float(J2) == float(J)
    for a, b in zip(gs, jax.tree_util.tree_leaves(g2)):
        assert np.array_equal(np.asarray(a), np.asarray(b))


TIGHT = {("data", "PARM02", "cg2dTargetResidual"): 1.e-13}


def _J_of(m, which, idx):
    """J as a function of a scalar perturbation of initial `which` at storage index idx, and its AD derivative."""
    from mitjax.drivers.checkpoint import integrate
    st0, xs, theta = setup(m)
    step = ck_step(m)

    def J(h):
        st = theta[0]
        f = getattr(st, which)
        st = st.replace(**{which: type(f)(f.data.at[idx].add(h), f.name, _dims=f.dims)})
        s_n, _ = integrate(step, m.arrays, init_fn((st, theta[1], theta[2]), st0), xs, schedule="step")
        return final_cost(m.arrays, s_n)
    Jj = jax.jit(J)
    ad = float(jax.jit(jax.grad(J))(jnp.float64(0.)))
    return Jj, ad


@pytest.mark.parametrize("which,idx,hs", [
    ("etaN", (0, 30, 20), (1e-1, 1e-2, 1e-3, 1e-4)),
    ("uVel", (0, 0, 30, 20), (1e-2, 1e-3, 1e-4, 1e-5)),
])
def test_fd_sweep(which, idx, hs):
    from mitjax.drivers.grad import fd_sweep
    m = _model("tight", TIGHT)
    Jj, ad = _J_of(m, which, idx)
    rows = fd_sweep(lambda x: Jj(x), jnp.float64(0.), jnp.float64(1.), hs, ad)
    for r in rows:
        print(f"{which} h={r.h:.0e} fd={r.fd:.15e} ad={r.ad:.15e} rel={r.rel_err:.2e} noise/h={r.noise:.1e}")
    assert ad != 0.0
    assert min(r.rel_err for r in rows) <= 1e-7, rows


def test_dot_test_tl_vs_adjoint():
    from mitjax.drivers.checkpoint import integrate
    from mitjax.drivers.grad import dot_test
    m = _model("grad")
    st0, xs, theta = setup(m)
    step = ck_step(m)
    st = theta[0]

    def f(x):
        u0, e0 = x
        s = st.replace(uVel=type(st.uVel)(u0, "uVel", _dims=st.uVel.dims),
                       etaN=type(st.etaN)(e0, "etaN", _dims=st.etaN.dims))
        s_n, _ = integrate(step, m.arrays, init_fn((s, theta[1], theta[2]), st0), xs, schedule="none")
        sn = s_n[0][0]
        return (sn.etaN.data, sn.uVel.data, sn.vVel.data)
    x = (st.uVel.data, st.etaN.data)
    k = jax.random.split(jax.random.PRNGKey(7), 5)
    v = (jax.random.normal(k[0], x[0].shape), jax.random.normal(k[1], x[1].shape))
    y = f(x)
    w = tuple(jax.random.normal(kk, a.shape) for kk, a in zip(k[2:], y))
    rows = dot_test(f, x, v, w, amps=(1.0, 1e-6))
    for amp, lhs, rhs, rel in rows:
        print(f"dot test amp={amp:g} <Jv,w>={lhs:.15e} <v,J^T w>={rhs:.15e} rel={rel:.2e}")
    assert all(rel <= 1e-12 for _, _, _, rel in rows), rows
