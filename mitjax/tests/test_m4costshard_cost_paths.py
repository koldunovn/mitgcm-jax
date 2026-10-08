"""The cost path on the traced model and on a TileSharding's blocks (lane M4COSTSHARD session 2, tier 1x).

1. Model.cost_final reads its grid and tables from the `arrays` argument (drivers/model.py; the adjoint drivers pass
   the traced model through adjoint_run.final_cost_gathered): for every cost build the drivers run -- the ice / ecco
   / ctrl arm (1D_ocean_ice_column, lab_sea: COST_AVERAGESFIELDS' ecco_tab, COST_GENCOST_ALL's R_low,
   CTRL_COST_DRIVER's maskC), COST_ATLANTIC_HEAT (global_ocean.90x40x15), COST_TEST (global_ocean.cs32x15),
   COST_TRACER (tutorial_tracer_adjsens) and the COST_WEIGHTS arm (tutorial_global_oce_optim) -- COST_FINAL on the
   initial cost state with `arrays=` the set-up's Arrays while the set-up's own grid and Arrays are poisoned (None)
   gives the default call's cost.h state, bit for bit. Negative controls (measured, job 27909680: 4 / 4 plants bite):
   each site reverted to the set-up closure (ATLANTIC_HEAT's maskC, CTRL_COST_DRIVER's maskC, R_low, ecco_tab) fails
   its case.
2. SEAICE_COST_TEST (pkg/seaice/seaice_cost_test.py) with the exchanger: only 1D_ocean_ice_column (one tile)
   compiles ALLOW_COST_ICE, so the multi-tile case is planted: random AREA / HEFF / rA on the tiles of
   tutorial_baroclinic_gyre (4 tiles, P = 3: Tloc = 2, two padding tiles) and global_ocean.90x40x15 (36 tiles, P = 4),
   a random objf_ice [nTiles], cost_ice_flag 1 and 2, the clock active and inactive. objf_ice under
   jit(shard_map(check_vma=True)) equals the single-device Exchanger's bit for bit, which equals ex=None (one code
   path) and a float64 numpy replay of the Fortran chain (:92-95 / :105-108, objf_ice(bi,bj) + tempVar*rA*fld point
   by point; the gate XLA flags have no FMA). Negative controls (each measured to bite): no gather (the pre-fix
   routine under shard_map: a shape error); every device starting from objf_ice of tiles 1..Tloc (no tile_index);
   the cost_tracer form objf_ice + all_tiles(chain from 0) (another rounding).
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax import paths
from mitjax.eesupp.exch_maps import load_maps
from mitjax.eesupp.exchange import Exchanger
from mitjax.eesupp.shard import TileSharding
from mitjax.farray import FArray, loop_i, loop_j
from mitjax.ops.fortran_minmax import MIN
from mitjax.pkg.seaice import seaice_cost_test as SCT


def _bits(a, b):
    a, b = np.ascontiguousarray(np.asarray(a)), np.ascontiguousarray(np.asarray(b))
    assert a.shape == b.shape and a.dtype == b.dtype, (a.shape, b.shape, a.dtype, b.dtype)
    if a.dtype.kind == "f":
        a, b = a.view(f"i{a.itemsize}"), b.view(f"i{b.itemsize}")
    return int(np.count_nonzero(a != b))


# ---------------------------------------------------------------------------------------------------------------------
# 1. Model.cost_final's grid and tables from `arrays`

COST_BUILDS = [("1D_ocean_ice_column", "input_ad"), ("lab_sea", "input_ad"), ("global_ocean.90x40x15", "input_ad"),
               ("global_ocean.cs32x15", "input_ad"), ("tutorial_tracer_adjsens", "input_ad"),
               ("tutorial_global_oce_optim", "input_ad")]


def _leaves(tree):
    return [np.asarray(x) for x in jax.tree.leaves(tree)]


def cost_final_mismatches(m):
    """(bits differing between COST_FINAL with the set-up's grid and with `arrays=` while the set-up is poisoned,
    the number of compared leaves)."""
    c0 = m.initial_carry()
    pk = c0[4]

    def run(**kw):
        cost, _ = m.cost_final(jax.tree.map(lambda x: x, pk["cost"]), pk.get("genarr"), state=c0[0], out={},
                               ecco=pk.get("ecco"), **kw)
        return _leaves(cost)
    ref = run()
    A, grid, arrays = m.arrays, m.grid, m.arrays
    m.grid, m.arrays = None, None                   # poisoned: any read of the set-up's grid / Arrays raises
    try:
        new = run(arrays=A)
    finally:
        m.grid, m.arrays = grid, arrays
    assert len(ref) == len(new)
    return sum(_bits(a, b) for a, b in zip(ref, new)), len(ref)


@pytest.mark.parametrize("exp,inp", COST_BUILDS, ids=[e for e, _ in COST_BUILDS])
def test_cost_final_reads_arrays(exp, inp, tmp_path):
    from mitjax.drivers.model import Model
    from mitjax.drivers.run import load_experiment, make_rundir
    exp_dir = paths.UPSTREAM / "verification" / exp
    m = Model(load_experiment(exp_dir, inp), make_rundir(exp_dir, inp, tmp_path / "run"))
    assert m.cfg.cpp.ALLOW_COST
    n, nleaves = cost_final_mismatches(m)
    assert n == 0 and nleaves > 0, (n, nleaves)


# ---------------------------------------------------------------------------------------------------------------------
# 2. SEAICE_COST_TEST on a planted multi-tile case

class _Cpp:
    def flag(self, name, *_):
        return name == "ALLOW_COST_ICE"


CLOCK = dict(endTime=jnp.float64(36000.0), startTime=jnp.float64(0.0), lastinterval=jnp.float64(14400.0),
             deltaTClock=jnp.float64(3600.0))
LAYOUTS = [("tutorial_baroclinic_gyre", 3), ("global_ocean.90x40x15", 4)]


def planted(L, seed=0):
    """Random AREA, HEFF, rA FArrays (halos included) on the layout's tiles and a random objf_ice [nTiles]."""
    rng = np.random.default_rng(seed)
    shape = (L.nTiles, L.sNy + 2 * L.OLy, L.sNx + 2 * L.OLx)
    dims = dict(i=(1 - L.OLx, L.sNx + L.OLx), j=(1 - L.OLy, L.sNy + L.OLy))
    f = {n: FArray(jnp.asarray(rng.uniform(lo, hi, shape)), n, **dims)
         for n, lo, hi in (("AREA", 0.0, 1.0), ("HEFF", 0.0, 3.0), ("rA", 1e9, 4e10))}
    return f, jnp.asarray(rng.uniform(0.0, 1e12, L.nTiles))


def _cfg(L):
    return SimpleNamespace(cpp=_Cpp(), size=SimpleNamespace(sNx=L.sNx, sNy=L.sNy))


def numpy_replay(L, f, objf, flag, myTime):
    """float64 replay of seaice_cost_test.F:79-82, :92-95 / :105-108 (tile by tile, DO j, DO i)."""
    c = {k: float(v) for k, v in CLOCK.items()}
    o = np.array(objf, dtype=np.float64)
    if not myTime > c["endTime"] - c["lastinterval"]:
        return o
    tempVar = 1.0 / ((1.0 + min(c["endTime"] - c["startTime"], c["lastinterval"])) / c["deltaTClock"])
    fld = np.asarray((f["HEFF"] if flag == 1 else f["AREA"]).data)
    rA = np.asarray(f["rA"].data)
    for t in range(L.nTiles):
        s = o[t]
        for j in range(L.sNy):
            for i in range(L.sNx):
                s = s + tempVar * rA[t, L.OLy + j, L.OLx + i] * fld[t, L.OLy + j, L.OLx + i]
        o[t] = s
    return o


def _call(fn, L, ex, f, objf, flag, myTime):
    cost = SimpleNamespace(objf_ice=objf)
    return fn(cost, myTime, 0, cfg=_cfg(L), sp=SimpleNamespace(cost_ice_flag=flag), AREA=f["AREA"], HEFF=f["HEFF"],
              rA=f["rA"], ex=ex, **CLOCK).objf_ice


def sharded(fn, maps, P, f, objf, flag, myTime):
    L = maps.layout
    sh = TileSharding(maps, P)
    spec = jax.tree.map(lambda x: sh.TILES, f, is_leaf=lambda x: isinstance(x, FArray))

    def body(ex, fl, o, t):
        return _call(fn, L, ex, fl, o, flag, t)
    prog = sh.shard_map(body, in_specs=(sh.TILES, spec, sh.REP, sh.REP), out_specs=sh.REP)
    return np.asarray(prog(sh.ex, sh.put_tree(f), objf, jnp.float64(myTime))), sh


# planted errors (negative controls): the routine's code with one change each
def _chain(cost, L, f, flag, o):
    tempVar = 1.0 / ((1.0 + MIN(CLOCK["endTime"] - CLOCK["startTime"], CLOCK["lastinterval"], p="b"))
                     / CLOCK["deltaTClock"])
    fld = f["HEFF"] if flag == 1 else f["AREA"]
    for j in range(1, L.sNy + 1):
        for i in range(1, L.sNx + 1):
            jj, ii = loop_j(j, j), loop_i(i, i)
            o = o + tempVar * f["rA"][ii, jj][:, 0, 0] * fld[ii, jj][:, 0, 0]
    return o


def _planted_no_tile_index(cost, myTime, myIter, *, cfg, sp, AREA, HEFF, rA, ex=None, **clk):
    L = SimpleNamespace(sNx=cfg.size.sNx, sNy=cfg.size.sNy)
    T = AREA.data.shape[0]
    o = _chain(cost, L, dict(AREA=AREA, HEFF=HEFF, rA=rA), int(sp.cost_ice_flag), cost.objf_ice[:T])
    o = ex.all_tiles(o)
    cost.objf_ice = jnp.where(myTime > (clk["endTime"] - clk["lastinterval"]), o, cost.objf_ice)
    return cost


def _planted_tracer_form(cost, myTime, myIter, *, cfg, sp, AREA, HEFF, rA, ex=None, **clk):
    L = SimpleNamespace(sNx=cfg.size.sNx, sNy=cfg.size.sNy)
    loc = _chain(cost, L, dict(AREA=AREA, HEFF=HEFF, rA=rA), int(sp.cost_ice_flag),
                 jnp.zeros((AREA.data.shape[0],), jnp.float64))
    o = cost.objf_ice + ex.all_tiles(loc)
    cost.objf_ice = jnp.where(myTime > (clk["endTime"] - clk["lastinterval"]), o, cost.objf_ice)
    return cost


def _pre_fix(cost, myTime, myIter, *, ex=None, **kw):
    return SCT.seaice_cost_test(cost, myTime, myIter, ex=None, **kw)


@pytest.mark.parametrize("exp,P", LAYOUTS, ids=[e for e, _ in LAYOUTS])
def test_seaice_cost_test_sharded_equals_P1(exp, P):
    maps = load_maps(exp)
    L = maps.layout
    assert len(jax.devices()) >= P and L.nTiles > 1
    f, objf = planted(L)
    for flag in (1, 2):
        for myTime in (3600.0, 25200.0):           # inactive (<= endTime - lastinterval = 21600), active
            ref = numpy_replay(L, f, objf, flag, myTime)
            one = np.asarray(_call(SCT.seaice_cost_test, L, Exchanger(maps), f, objf, flag, jnp.float64(myTime)))
            none = np.asarray(_call(SCT.seaice_cost_test, L, None, f, objf, flag, jnp.float64(myTime)))
            got, sh = sharded(SCT.seaice_cost_test, maps, P, f, objf, flag, myTime)
            assert _bits(one, ref) == 0 and _bits(none, one) == 0, (flag, myTime)
            assert _bits(got, one) == 0, (flag, myTime, np.max(np.abs(got - one)))
            assert (myTime > 21600.0) == bool(np.any(one != np.asarray(objf)))
    if exp == "tutorial_baroclinic_gyre":
        assert sh.blocks.Tpad > L.nTiles              # the padded layout is exercised


@pytest.mark.parametrize("exp,P", LAYOUTS, ids=[e for e, _ in LAYOUTS])
def test_seaice_cost_test_negative_controls(exp, P):
    maps = load_maps(exp)
    L = maps.layout
    f, objf = planted(L)
    one = np.asarray(_call(SCT.seaice_cost_test, L, Exchanger(maps), f, objf, 2, jnp.float64(25200.0)))
    with pytest.raises(Exception, match="(?i)incompatible shapes|broadcast"):
        sharded(_pre_fix, maps, P, f, objf, 2, 25200.0)
    for fn in (_planted_no_tile_index, _planted_tracer_form):
        got, _ = sharded(fn, maps, P, f, objf, 2, 25200.0)
        assert _bits(got, one) > 0, fn.__name__
