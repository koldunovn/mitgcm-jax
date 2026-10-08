"""R5 optim adjoint on the sharded model, fake CPU devices (plan Task 17, tier 1x; helpers r5_shardgrad.py): the
gradient of COST_FINAL's fc w.r.t. the xx_qnet record through 10 steps (adjoint_run.sharded_problem, per-step
checkpoint) by drivers/sharded_grad.py (jax.vjp of grad._objective inside jit(shard_map(check_vma=True))) on a
TileSharding of optim's 2x2 tiles, against the single-device driver (the first TAF match's program).

* P=1 mesh (the sharded code without remote rounds) == single device: J and every gradient point bitwise.
* P=4 (one tile per device; every tile edge a partition edge): J bitwise (COST_FINAL gathers its inputs to every
  tile in tile order, ex.all_tiles, and sums as one process does); the gradient == single device within TOL (the
  exchange transposes sum a point's copies in another order than the single-device scatter-add, as SHARDGRAD
  measured for R2: 2.55e-15); every point finite incl. the padded layout; linearity guard: seed 0 -> exactly 0,
  seed 2 == 2 x seed 1 bitwise.
* P=2 (tiles 1-2 | 3-4) and P=3 (tiles 1-2 | 3-4 | two padding tiles: the same device edges, the same gradient bit
  for bit) do not meet TOL (1.32e-14; lane FLOOR job 27952965: P=2 / 3 1.319e-14, P=4 7.11e-15) and are judged by
  plan decision 18:
  max|g_P - g_1| <= C_FLOOR x FLOOR, FLOOR the window's 1-ulp floor at P = 1 (the largest relative gradient change
  of 1-ulp relative +-1 injections, fixed seed, into the adjoint state at each step boundary; measured by
  scripts/floor_optim_perturb.py perturb, whose uninstrumented base equals g_1 bit for bit). Negative control: a
  planted difference of 1.5 C_FLOOR x FLOOR at one point of the real g_P fails the same criterion.
Each program compiles once (~10 min on a CPU node); results are kept as numpy and the jit caches cleared between
programs (vm.max_map_count, PORTING_LESSONS).
"""

import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

TOL = 1e-14
C_FLOOR = 10                # plan decision 18: the bar as a multiple of the window's P = 1 1-ulp floor
FLOOR = 2.006550987402107e-14   # the window's 1-ulp floor at P = 1, at boundary K = 8 (lane FLOOR job 27952964)
FLOOR_P = {2, 3}            # the P judged by the floor (decision 18); every other P by TOL, which it meets
_C = {}


def _meets(g, g1, nproc):
    """(ok, measure, bar): R5's max|g_P - g_1| / max|g_1| against TOL, or against C_FLOOR x FLOOR for FLOOR_P."""
    from mitjax.tests import r5_shardgrad as RS
    bar = C_FLOOR * FLOOR if nproc in FLOOR_P else TOL
    r = RS.rel(g, g1)
    return r <= bar, r, bar


@pytest.fixture(autouse=True)
def _release():
    from mitjax.tests import r5_shardgrad as RS
    RS.release()
    yield


def _single():
    from mitjax.tests import r5_shardgrad as RS
    if "su" not in _C:
        _C["su"] = RS.Setup("t1x-shard")
    if "p1" not in _C:
        _C["p1"] = _C["su"].p1()
        RS.release()
    return _C["su"], _C["p1"]


@pytest.mark.parametrize("nproc", [1, 2, 3, 4])
def test_sharded_optim_gradient(nproc):
    from mitjax.tests import r5_gate as r5
    from mitjax.tests import r5_shardgrad as RS
    su, (J1, g1) = _single()
    assert all(abs(g1[p] - v) / abs(v) < 1e-14 for p, v in zip(r5.GRDCHK_POINTS, r5.TAF_ADJOINT_GRADIENT))
    sd = RS.Sharded(su, nproc)
    J, gp = sd.value_and_grad()
    g = sd.unpad(gp)
    assert np.all(np.isfinite(gp))
    assert J == J1, (J, J1)
    if nproc == 1:
        assert RS.bits_differ(g, g1) == 0
        return
    assert np.all(gp[sd.sh.layout.nTiles:] == 0.0)
    ok, r, bar = _meets(g, g1, nproc)
    print(f"P={nproc}: max|g_P - g_1| / max|g_1| = {r:.3e}, bar {bar:.3e}, ratio to the floor {r / FLOOR:.3f}")
    assert ok, (r, bar)
    if nproc in FLOOR_P:
        # negative control: a planted difference of 1.5 C_FLOOR FLOOR max|g_1| at the largest |g_1| fails the bar
        gb = g.copy()
        i = np.unravel_index(np.argmax(np.abs(g1)), g1.shape)
        gb[i] += 1.5 * C_FLOOR * FLOOR * np.max(np.abs(g1))
        okb, rb, _ = _meets(gb, g1, nproc)
        print(f"P={nproc} control: planted {rb:.3e} > bar {bar:.3e}")
        assert not okb, (rb, bar)
    J0, g0 = sd.value_and_grad(0.0)
    J2, g2 = sd.value_and_grad(2.0)
    assert np.all(g0 == 0.0)
    assert RS.bits_differ(g2, 2.0 * gp) == 0
