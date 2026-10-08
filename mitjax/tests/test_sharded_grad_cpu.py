"""Sharded gradients on fake CPU devices (plan Task 17, tier 1x): R2 tutorial_baroclinic_gyre, J = weighted sum of
theta and etaN after 10 steps (mitjax/tests/shardgrad_gate.py), dJ/d(initial State, FFIELDS.h, phi0surf) through
`integrate` with the sqrt schedule: mitjax/drivers/sharded_grad.py (jax.vjp of grad._objective inside
jit(shard_map(check_vma=True)), ShardedExchanger) at P = 1, 4, 3 vs the single-device driver grad.value_and_grad.

[L-PAR-11] says fake CPU devices gave NaN or deadlocked for the ECCO gradient driver; here they were measured first
(dev jobs 27832216, 27832341-3): no deadlock, no NaN, and the numbers asserted below
(this file passed in dev jobs 27832902, 27832930 and 27833582). The real-GPU gate is test_sharded_grad_gpu.py (tier 2).
Cost: each program is traced (~70 s) and compiled (~350 s) once; a warm gradient takes ~1 s. Five whole-window
programs and the three chunk programs: ~45 min.

* The sharded driver on a 1-device mesh == the single-device driver: J and every gradient point bitwise (L-PAR-1).
* P=4 (one tile per device, every tile edge a partition edge) and P=3 (Tloc = 2, two padding tiles = replicas of tile
  1): J bitwise; every gradient point finite incl. the padding tiles, where it is exactly 0; the gradient == P=1 to
  rounding: max over leaves of max|g_P - g_1| / max|g_1| <= TOL (measured 2.55e-15 at P=4 and P=3, 1.3e5 points
  differ; the sharded exchanges' transposes sum a point's copies in another order than the single-device scatter-add,
  test_sharded_exchange.test_sharded_transpose_and_padding; the P=1 mesh, same code without remote rounds, is
  bitwise); the final State of the program's own forward pass bitwise the P=1 mesh's.
* Tile-edge cotangents across devices: a cost on tile 1 only (device 0 at P=4) has a gradient on the other devices'
  tiles only through the ppermute transposes: nonzero next to the tile edges, == the P=1 mesh within TOL.
* The sharded chunked driver (host-parked boundaries, chunk VJPs) == the whole-window sharded program, bitwise.
* Linearity guard of the reverse program ([L-AD-32]): seed 0 -> exactly 0 everywhere; seed 2 -> 2 x seed 1 bitwise.
* Negative controls, measured to bite: (a) every ppermute round transposed in the FORWARD direction (custom_vjp; the
  forward is bitwise unchanged) -> the gradient comparison fails (measured: per-leaf relative 1.1e10 over 10 steps,
  689 over 3 steps; 2.6e5 points); (b) the exchanger's invariant -> varying cast removed (`vary` = identity) ->
  tracing fails in the cg2d solve's while_loop (TypeError: carry "manual axis types do not match", [L-PAR-10]; 4.9 s).
"""

import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

import jax  # noqa: E402

# P vs P=1 gradient, per-leaf relative (module docstring: measured 2.55e-15; the wrong-transpose control gives 689)
TOL = 1e-14

_C = {}


@pytest.fixture(autouse=True)
def _release_executables():
    """Free the previous test's compiled programs before compiling the next (values are kept as numpy in _C). Each
    XLA:CPU kernel holds ~3 memory mappings and a process may hold vm.max_map_count = 65530; the five whole-window
    programs of this file kept alive together exceeded it: LLVM "Unable to allocate section memory" -> abort (tier1x
    27833481, dev 27834318/27834790/27836249). Changes no value."""
    import gc
    jax.clear_caches()
    gc.collect()
    yield


def _G():
    from mitjax.tests import shardgrad_gate as G
    return G


def _su():
    if "su" not in _C:
        _C["su"] = _G().Setup(_G().model())
    return _C["su"]


def _p1():
    if "p1" not in _C:
        J, g = _su().p1_value_and_grad()
        _C["p1"] = (float(J), jax.device_get(g))
    return _C["p1"]


@pytest.mark.parametrize("nproc", [1, 4, 3])
def test_sharded_gradient_equals_p1(nproc):
    G = _G()
    J1, g1 = _p1()
    assert not G.nonfinite(g1) and any(np.any(x != 0) for _, x in G.leaves(g1))
    sd = G.Sharded(_su(), nproc)
    J, g, st = jax.device_get(sd.value_and_grad())
    assert float(J) == J1, (float(J), J1)
    assert not G.nonfinite(g), G.nonfinite(g)                    # padded layout: padding tiles included
    assert not G.padding_nonzero(sd.sh, g), G.padding_nonzero(sd.sh, g)
    glob, per, npts = G.summary(G.compare(g1, sd.sh.unpad_tree(g)))
    print(f"P={nproc} vs P=1 gradient: global rel {glob:.3e}, per-leaf rel {per:.3e}, {npts} points differ")
    st = sd.sh.unpad_tree(st)
    if nproc == 1:
        assert npts == 0, (glob, per, npts)                      # the same code without remote rounds: bitwise
        _C["st1"] = st
    else:
        assert per <= TOL, (glob, per, npts)
        if "st1" in _C:
            d = {k: v for k, v in G.compare(_C["st1"], st).items() if v[2]}
            assert not d, d                                      # the forward pass: bitwise
    _C[f"sd{nproc}"] = sd
    if nproc == 4:
        _C["g4"], _C["st4"] = g, st


def test_tile_edge_cotangents_cross_device():
    """J0 = the cost on tile 1 (storage 0) only. At P=4 tile 1 is device 0's, so dJ0/dtheta0 on tiles 2-4 (devices
    1-3) arrives only through the ppermute transposes: nonzero at wet points of the first interior column of tile 2
    (east of tile 1) and the first interior row of tile 3 (north), levels 1 and 5, and == the P=1 mesh within TOL
    (the same compiled programs as above: the weights are an argument). Measured: 1.65e-15 per leaf; max |gradient|
    on tiles 2-4 0.14, 0.10, 0.0066."""
    G = _G()
    su = _su()
    w0 = G.weights(su.m, tiles=(0,))
    out = {}
    for P in (1, 4):
        sd = _C.get(f"sd{P}") or G.Sharded(su, P)
        J, g, _ = jax.device_get(sd.value_and_grad(w=w0))
        out[P] = (float(J), sd.sh.unpad_tree(g))
    assert out[4][0] == out[1][0]
    sz = su.m.cfg.size
    OLx, OLy = sz.OLx, sz.OLy
    mask = np.asarray(su.m.grid.maskC.data)
    g4, g1 = np.asarray(out[4][1][0].theta.data), np.asarray(out[1][1][0].theta.data)
    pts = [(1, 0, OLy + 15, OLx), (1, 4, OLy + 5, OLx), (2, 0, OLy, OLx + 15), (2, 4, OLy, OLx + 5)]
    for p in pts:
        assert mask[p] == 1.0 and g4[p] != 0.0, (p, g4[p])
    glob, per, npts = G.summary(G.compare(out[1][1], out[4][1]))
    print(f"tile-1 cost: P=4 vs P=1 mesh: global rel {glob:.3e}, per-leaf rel {per:.3e}, {npts} points differ; "
          f"gradient on tiles 2-4: {[float(np.max(np.abs(g4[t]))) for t in (1, 2, 3)]}")
    assert per <= TOL, (glob, per, npts)


def test_sharded_chunked_driver_equals_whole_window_p4():
    """sharded_grad.sharded_chunked_value_and_grad (2 chunks x 5 steps, per-step checkpoint inside a chunk, chunk
    boundaries parked on the host in the padded layout and placed again) == the whole-window sharded program (sqrt
    schedule) at P=4: J and every gradient point bitwise (as grad.chunked_value_and_grad == value_and_grad at P=1,
    test_checkpoint.py); the per-chunk cotangent-norm trace has 3 finite, nonzero entries."""
    from mitjax.drivers import sharded_grad as SG
    from mitjax.drivers.checkpoint import take_steps
    G = _G()
    su = _su()
    sd = _C.get("sd4") or G.Sharded(su, 4)
    if "g4" in _C:
        J4, g4 = _p1()[0], _C["g4"]
    else:
        J4, g4, _ = jax.device_get(sd.value_and_grad())
        J4 = float(J4)
    xs = np.asarray(su.xs)
    res = SG.sharded_chunked_value_and_grad(sd.sh, su.step, sd.theta, sd.model, sd.st0, n_chunks=2, chunk_steps=5,
                                            xs_fn=lambda c: take_steps(xs, 5 * c, 5 * (c + 1)),
                                            final_cost=su.final_cost, init_fn=G.init_fn, schedule="step")
    assert res.loss == J4, (res.loss, J4)
    d = {k: v for k, v in G.compare(g4, jax.device_get(res.grad)).items() if v[2]}
    assert not d, d
    assert len(res.trace) == 3 and all(np.isfinite(t) and t > 0 for t in res.trace), res.trace


def test_linearity_guard_p4():
    G = _G()
    sd = _C.get("sd4") or G.Sharded(_su(), 4)
    g1 = _C["g4"] if "g4" in _C else jax.device_get(sd.value_and_grad()[1])
    _, gz, _ = jax.device_get(sd.value_and_grad(0.0))
    assert sum(int(np.count_nonzero(x)) for _, x in G.leaves(gz)) == 0
    assert not G.nonfinite(gz)
    _, g2, _ = jax.device_get(sd.value_and_grad(2.0))
    d = {k: v for k, v in G.compare(jax.tree.map(lambda a: 2.0 * np.asarray(a), g1), g2).items() if v[2]}
    assert not d, d


def test_negative_control_wrong_transpose_bites():
    G = _G()
    J1, g1 = _p1()
    sd = G.Sharded(_su(), 4, ex_fn=lambda ex: G.as_exchanger(G.WrongTransposeExchanger, ex))
    J, g, st = jax.device_get(sd.value_and_grad())
    assert float(J) == J1                                         # the forward is untouched
    if "st4" in _C:
        d = {k: v for k, v in G.compare(_C["st4"], sd.sh.unpad_tree(st)).items() if v[2]}
        assert not d, d
    glob, per, npts = G.summary(G.compare(g1, sd.sh.unpad_tree(g)))
    print(f"wrong ppermute transpose: global rel {glob:.3e}, per-leaf rel {per:.3e}, {npts} points differ")
    assert per > TOL and npts > 0, (glob, per, npts)              # the gate above fails


def test_negative_control_missing_vary_fails():
    G = _G()
    sd = G.Sharded(_su(), 4, ex_fn=lambda ex: G.as_exchanger(G.NoVaryExchanger, ex))
    with pytest.raises(TypeError, match="manual axis types"):
        sd.vg.lower(sd.theta, sd.model, sd.st0, _su().xs, G.SG.seed(1.0))
