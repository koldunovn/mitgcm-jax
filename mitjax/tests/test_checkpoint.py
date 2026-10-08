"""mitjax/drivers/{checkpoint,grad}.py (plan Task 7c) on a toy step (seconds; smoke).

TEST FIXTURE (not model code): a toy "FORWARD_STEP" with a State {a, b, gNm1, it}: a nonlinear update read from a
per-step input and a traced model parameter, an Adams-Bashforth-like history `gNm1` carried in the State, an integer
iteration counter, and a model pytree holding a float parameter, a 1-D weight field and an integer index map.

Gates: scan == Python loop of the jitted step, bitwise, for every schedule; J and gradients (initial State and model
parameter) bitwise equal between schedules none / step / sqrt (several segment counts) and the nested chunked driver
(host chunks -> sqrt segments -> steps); the per-step cotangent statistics hook reports the statistics of dJ/dState at
every step boundary (== the chunked driver's per-boundary State cotangents) and leaves the gradients bitwise unchanged;
the model is a jit argument (no captured constants), and a planted closure is detected; FD and dot test on the toy.
"""

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.drivers import checkpoint as ck
from mitjax.drivers import grad as gr

N = 12
NX = 7


def _step(model, st, x):
    """Toy step: AB2-like extrapolation of a tendency G(a, b) with the history gNm1 (as adams_bashforth2.F carries
    gNm1), a nonlinear b update, an integer gather through the model's index map."""
    a, b, gNm1 = st["a"], st["b"], st["gNm1"]
    G = jnp.sin(a) * model["k"] + x["f"] * b[model["idx"]] * model["w"]
    ab = (1.5 + model["eps"]) * G - (0.5 + model["eps"]) * gNm1
    return {"a": a + 0.1 * ab, "b": b + 0.05 * a * a - 0.01 * b, "gNm1": G, "it": st["it"] + 1}


def _setup():
    model = {"k": jnp.asarray(0.9), "eps": jnp.asarray(0.01), "w": jnp.linspace(0.5, 1.5, NX),
             "idx": jnp.asarray(np.roll(np.arange(NX), 1))}
    st0 = {"a": jnp.linspace(0.0, 1.0, NX), "b": jnp.ones(NX), "gNm1": jnp.zeros(NX), "it": jnp.asarray(1)}
    xs = {"f": np.linspace(0.1, 0.5, N)}
    return model, st0, xs


def _final_cost(m, s):
    return jnp.sum(s["a"] ** 2) + jnp.sum(s["gNm1"])


def _cost(m, s, x):
    return jnp.sum(s["b"]) * 0.01


def _params_fn(theta, model):
    return dict(model, k=theta["k"])


def _init_fn(theta, st):
    return gr.replace_fields({k: theta[k] for k in ("a", "b")}, st)


def _stats(model, ct):
    """Per-boundary statistics of the State cotangent (a stand-in for MON_WRITESTATS_RL of adTheta etc.)."""
    return {"a_max": jnp.max(ct["a"]), "a_min": jnp.min(ct["a"]), "b_rms": jnp.sqrt(jnp.mean(ct["b"] ** 2))}


def _theta(st0, model):
    return {"a": st0["a"], "b": st0["b"], "k": model["k"]}


def _bits(tree):
    return jax.tree.map(lambda x: np.asarray(x).view(np.uint64) if np.asarray(x).dtype == np.float64
                        else np.asarray(x), tree)


def _equal(t1, t2):
    return all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(_bits(t1)), jax.tree.leaves(_bits(t2))))


def test_scan_equals_loop():
    """integrate (schedules none, step, sqrt with 1..12 segments) == the Python loop of the jitted step, bitwise."""
    model, st0, xs = _setup()
    s_loop = ck.run_loop(jax.jit(_step), model, st0, [jax.tree.map(lambda a: a[i], xs) for i in range(N)])
    runs = [("none", None), ("step", None)] + [("sqrt", S) for S in (None, 1, 2, 3, 4, 5, 12)]
    for sch, S in runs:
        s, acc = jax.jit(lambda m, s, x: ck.integrate(_step, m, s, x, schedule=sch, segments=S))(model, st0, xs)
        assert int(s["it"]) == 1 + N
        assert _equal(s, s_loop), (sch, S)
    assert ck.sqrt_segments(24) == 4 and ck.sqrt_segments(12) == 3 and ck.sqrt_segments(1) == 1


def test_all_schedules_same_gradient():
    """value_and_grad with schedules none / step / sqrt (segments 2, 3, 4, 5) and chunked_value_and_grad (chunk
    lengths 1, 3, 4, 12; boundary strides 1-3; in-chunk schedules step and sqrt = the nested schedule): J and the
    gradient w.r.t. the initial State (a, b) bitwise; the gradient w.r.t. the model parameter k within 4 ulp.

    Why not bitwise for k (measured 2026-10-01: 1.1e-16 relative, 1 ulp, for sqrt segments 2 and 4 and chunks of 3;
    bitwise for the others): a parameter read by every step gets the sum over steps of its per-step cotangents, and
    each scan transpose sums its own steps from zero, so segments and chunks group that sum differently. The State
    cotangent is passed from step to step unchanged in every schedule, hence bitwise."""
    model, st0, xs = _setup()
    th = _theta(st0, model)
    kw = dict(final_cost=_final_cost, cost=_cost, init_fn=_init_fn, params_fn=_params_fn)
    J0, g0 = gr.value_and_grad(_step, th, model, st0, xs, schedule="none", **kw)
    assert all(np.all(np.isfinite(np.asarray(v))) and np.any(np.asarray(v) != 0) for v in jax.tree.leaves(g0))
    runs = {"step": gr.value_and_grad(_step, th, model, st0, xs, schedule="step", **kw)}
    for S in (2, 3, 4, 5):
        runs[f"sqrt{S}"] = gr.value_and_grad(_step, th, model, st0, xs, schedule="sqrt", segments=S, **kw)
    for K in (1, 3, 4, 12):
        xs_fn, nch = gr.chunks_of(xs, K)
        for stride in (1, 2, 3):
            for sch in ("step", "sqrt"):
                r = gr.chunked_value_and_grad(_step, th, model, st0, n_chunks=nch, chunk_steps=K, xs_fn=xs_fn,
                                              schedule=sch, boundary_stride=stride, **kw)
                assert len(r.trace) == nch + 1 and all(np.isfinite(r.trace))
                runs[f"chunked K={K} stride={stride} {sch}"] = (r.loss, r.grad)
    bad = [name for name, (J, g) in runs.items()
           if not (float(J) == float(J0) and _equal({k: g[k] for k in ("a", "b")}, {k: g0[k] for k in ("a", "b")})
                   and abs(float(g["k"]) - float(g0["k"])) <= 4 * np.spacing(abs(float(g0["k"]))))]
    assert not bad, bad
    # the grouping effect is real (not a test artefact): at least one nested schedule differs from "none" in k
    assert any(float(g["k"]) != float(g0["k"]) for _, g in runs.values())


def test_cotangent_statistics_hook():
    """The stats hook gives the statistics of dJ/dState at every boundary 0..N (N+1 rows): row 0 == the gradient
    w.r.t. the initial State, row N == the seed (final cost), and every row == the chunked driver's 1-step boundary
    cotangents (to 1e-14 relative: different programs). The gradient with the hook is bitwise the gradient without."""
    model, st0, xs = _setup()
    th = _theta(st0, model)
    kw = dict(final_cost=_final_cost, cost=_cost, init_fn=_init_fn, params_fn=_params_fn)
    ref_stats = None
    for sch, S in (("step", None), ("none", None), ("sqrt", 3), ("sqrt", 5)):
        J0, g0 = gr.value_and_grad(_step, th, model, st0, xs, schedule=sch, segments=S, **kw)
        J, g, stats = gr.value_and_grad(_step, th, model, st0, xs, schedule=sch, segments=S, stats_fn=_stats, **kw)
        assert float(J) == float(J0) and _equal(g, g0), sch      # same schedule with and without the hook: bitwise
        assert all(np.asarray(v).shape == (N + 1,) for v in stats.values())
        if ref_stats is None:
            ref_stats = stats
        else:
            assert all(np.allclose(stats[k], ref_stats[k], rtol=1e-14, atol=0) for k in stats), (sch, S)
    stats = ref_stats
    _, g0 = gr.value_and_grad(_step, th, model, st0, xs, schedule="step", **kw)
    # row 0: the cotangent of the initial State a is dJ/da0 (init_fn replaces a)
    assert float(stats["a_max"][0]) == float(np.max(g0["a"])) and float(stats["a_min"][0]) == float(np.min(g0["a"]))
    # row N: the seed dJ/da_N = 2 a_N from the final cost
    s_n, _ = jax.jit(lambda m, s, x: ck.integrate(_step, m, s, x))(model, st0, xs)
    assert np.isclose(float(stats["a_max"][N]), float(np.max(2 * s_n["a"])), rtol=1e-15)
    # every row vs the chunked driver's boundary cotangents (seed first)
    cots = []
    xs_fn, nch = gr.chunks_of(xs, 1)
    gr.chunked_value_and_grad(_step, th, model, st0, n_chunks=nch, chunk_steps=1, xs_fn=xs_fn,
                              on_cot=cots.append, **kw)
    ref = [_stats(model, c) for c in reversed(cots)]
    for k in stats:
        np.testing.assert_allclose(np.asarray(stats[k]), [float(r[k]) for r in ref], rtol=1e-14, atol=0)


def test_model_is_a_jit_argument():
    """The drivers lower with no captured constants (model, inputs and controls are arguments). Negative control: the
    same objective closing over the model captures its arrays, and captured_constants reports them."""
    model = {"k": jnp.asarray(0.9), "eps": jnp.asarray(0.01), "w": jnp.asarray(np.linspace(0.5, 1.5, 4096)),
             "idx": jnp.asarray(np.roll(np.arange(4096), 1))}
    st0 = {"a": jnp.linspace(0.0, 1.0, 4096), "b": jnp.ones(4096), "gNm1": jnp.zeros(4096), "it": jnp.asarray(1)}
    xs = {"f": np.linspace(0.1, 0.5, N)}
    th = _theta(st0, model)
    J = gr._objective(_step, schedule="step", segments=None, cost=_cost, final_cost=_final_cost, init_fn=_init_fn,
                      params_fn=_params_fn)
    vg = jax.jit(jax.value_and_grad(J))
    assert gr.captured_constants(vg, th, model, st0, xs) == 0
    closure = jax.jit(jax.value_and_grad(lambda th, st0, xs: J(th, model, st0, xs)))
    captured = gr.captured_constants(closure, th, st0, xs)
    nbytes = model["w"].nbytes + model["idx"].nbytes
    print(f"captured by the planted closure: {captured} B (model w + idx: {nbytes} B)")
    assert captured >= nbytes, (captured, nbytes)
    # same values either way
    v1, v2 = vg(th, model, st0, xs), closure(th, st0, xs)
    assert np.isclose(float(v1[0]), float(v2[0]), rtol=1e-14)


def test_fd_and_dot_test_on_toy():
    """The toy gradient w.r.t. k agrees with central FD (plateau <= 1e-8); the dot test utility gives <= 1e-13 on the
    State map; a wrong transpose (planted custom_vjp with a factor 2) is caught by it."""
    model, st0, xs = _setup()
    th = _theta(st0, model)
    kw = dict(final_cost=_final_cost, cost=_cost, init_fn=_init_fn, params_fn=_params_fn)
    _, g = gr.value_and_grad(_step, th, model, st0, xs, **kw)
    d = jax.tree.map(jnp.zeros_like, th)
    d["k"] = jnp.asarray(1.0)
    rows = gr.fd_sweep(lambda t: gr.objective(_step, t, model, st0, xs, **kw), th, d, (1e-3, 1e-4, 1e-5, 1e-6),
                       float(g["k"]))
    errs = np.array([r.rel_err for r in rows])
    assert np.all(np.isfinite(errs)) and np.min(errs) <= 1e-8, rows    # np: Python's min() skips a NaN

    def f(ab):
        s, _ = ck.integrate(_step, model, dict(st0, a=ab["a"], b=ab["b"]), xs)
        return {"a": s["a"], "b": s["b"]}
    rng = np.random.default_rng(0)
    x = {"a": st0["a"], "b": st0["b"]}
    v = jax.tree.map(lambda z: jnp.asarray(rng.standard_normal(z.shape)), x)
    w = jax.tree.map(lambda z: jnp.asarray(rng.standard_normal(z.shape)), x)
    for amp, lhs, rhs, rel in gr.dot_test(f, x, v, w, amps=(1.0, 1e-6)):
        assert rel <= 1e-13, (amp, rel)

    # negative control of the utility: a linear map whose VJP applies A instead of A^T (custom_linear_solve with a
    # wrong transpose_solve) fails the dot test; with the true transpose it passes
    A = jnp.asarray(rng.standard_normal((4, 4)))
    z0, vz, wz = jnp.ones(4), jnp.arange(4.0), jnp.arange(4.0) - 2.0

    def op(tr):
        return lambda z: jax.lax.custom_linear_solve(lambda y: y, z, lambda _, b: A @ b, lambda _, b: tr @ b)
    assert gr.dot_test(op(A.T), z0, vz, wz)[2] < 1e-14
    assert gr.dot_test(op(A), z0, vz, wz)[2] > 1e-3

    # amplification(): a dead reverse path (a zero in the cotangent-norm trace) or a NaN is reported and fails the bars
    assert gr.amplification([1.0, 1.001, 1.002, 1.003], 1).passes
    for trace in ([1.0, 0.0, 0.0, 0.0], [1.0, 1.001, float("nan"), 1.003]):
        amp = gr.amplification(trace, 1)
        assert not amp.passes and amp.bad_rates > 0, (trace, amp)
