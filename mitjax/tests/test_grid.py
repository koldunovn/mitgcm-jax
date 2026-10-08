"""Grid gates (plan Task 10, tier1x): every GRID.h field the oracle dumps, bitwise on all points, for every M1 variant,
and the negative controls measured to bite in each.

The grid of a variant is built as INITIALISE_FIXED builds it (mitjax/tests/grid_gate.py `build_grid`: INI_GRID,
SET_GRID_FACTORS, INI_DEPTHS with the bathymetry read through mitjax/io/mds.py, INI_MASKS_ETC with the probed exchanges
of mitjax/eesupp, INI_CORI), eagerly on the CPU backend under the gate XLA flags (conftest.py), and compared with the
registered dumps-on oracle run (G00_geometry; S00_begin hFac arrays) by element equality on every point of every tile,
halos and land included, with both sides required finite everywhere. Dumped fields that are not GRID.h fields set by
these routines are named exemptions (grid_gate.NOT_TASK10, and the hFac arrays UPDATE_R_STAR rescales under r*).

Negative controls (measured, 2026-10-01, all nine variants): delR(Nr) moved by one ulp makes 4 (baroclinic gyre) to
15136 (barotropic gyre) points of 4-16 fields differ; skipping every exchange makes 3264 (barotropic gyre) to 416024
(global_ocean.90x40x15) points of 16-22 fields differ. Each control asserts that the gate fails on its planted error.

The P=4 == P=1 gate of the grid is pending (the grid build is eager; sharding it needs the sharded driver).
"""

import dataclasses

import numpy as np
import pytest

from mitjax.tests import grid_gate as gg

IDS = [f"{e}/{i}" for e, i in gg.VARIANTS]


@pytest.mark.parametrize("exp,inp", gg.VARIANTS, ids=IDS)
def test_grid_bitwise(exp, inp):
    grid = gg.build_grid(exp, inp)
    result, exempt = gg.compare(grid, exp, inp)
    bad = gg.failures(result)
    assert not bad, (f"{exp}/{inp}: fields differing from the oracle "
                     f"(n, ndiff, nonfinite ours, nonfinite oracle, ndiff bits): {bad}")
    npts = sum(v[0] for v in result.values())
    assert len(result) >= 62 and npts > 0, f"{exp}/{inp}: compared only {len(result)} fields / {npts} points"
    for name, reason in exempt.items():
        assert reason, f"{name}: exemption without a reason"
    # every GRID.h field the routines set and the oracle dumps is gated: the horizontal metrics, masks, hFac factors
    for name in ("xC", "yG", "dxC", "dyU", "rAz", "recip_rAs", "maskC", "maskW", "maskS", "maskInW", "R_low",
                 "Ro_surf", "rLowW", "rSurfS", "recip_Rcol", "drF", "rF", "fCori", "tanPhiAtV"):
        assert name in result, f"{exp}/{inp}: {name} not gated"
    assert any(k.startswith("hFacC") for k in result), f"{exp}/{inp}: hFacC not gated"


@pytest.mark.parametrize("exp,inp", gg.VARIANTS, ids=IDS)
def test_negative_control_delR_one_ulp(exp, inp):
    """Planted error: delR(Nr) moved by one ulp. Bites in every variant (at least drF(Nr), drC(Nr+1) and their
    reciprocals differ)."""
    params = gg.perturbed_delR(gg.grid_params(exp, inp))
    bad = gg.failures(gg.compare(gg.build_grid(exp, inp, params=params), exp, inp)[0])
    assert {"drF", "recip_drF"} <= set(bad), f"{exp}/{inp}: the delR control did not bite as expected: {sorted(bad)}"
    assert sum(v[1] for v in bad.values()) >= 4


@pytest.mark.parametrize("exp,inp", gg.VARIANTS, ids=IDS)
def test_negative_control_no_exchange(exp, inp):
    """Planted error: every exchange returns its input (halos of R_low, Ro_surf, hFacW/S, rSurfW/S, rLowW/S never
    filled). Bites in every variant: R_low and the hFac-derived masks differ at halo points."""
    grid = gg.build_grid(exp, inp, ex=gg.NoExchange())
    bad = gg.failures(gg.compare(grid, exp, inp)[0])
    assert {"R_low", "maskC", "maskW", "maskS"} <= set(bad), f"{exp}/{inp}: no-exchange control: {sorted(bad)}"
    # the differing points include halo points (the gate covers halos)
    ds, it, _ = gg.oracle(exp, inp)
    ref = ds.field(it, "G00_geometry", "R_low")[:, 0]
    ours = np.asarray(grid.R_low.data)
    sz = gg.experiment(exp, inp).cfg.size
    halo = np.ones(ref.shape[1:], bool)
    halo[sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] = False
    assert np.count_nonzero((ours != ref) & halo[None]) > 0


def test_unported_options_raise():
    """Options no M1 variant executes raise instead of running untested code (PORTING_RULES 1)."""
    from mitjax.model.src.ini_grid import ini_grid
    from mitjax.model.src.ini_masks_etc import ini_masks_etc
    exp, inp = "tutorial_baroclinic_gyre", "input"
    cfg = gg.experiment(exp, inp).cfg
    p = gg.grid_params(exp, inp)
    for change, where in ((dict(cosPower=1.0), "cosPower"), (dict(rotateGrid=True), "rotateGrid"),
                          (dict(selectSigmaCoord=1), "selectSigmaCoord"), (dict(delXFile="dx.bin"), "delXFile")):
        with pytest.raises(NotImplementedError):
            ini_grid(cfg=cfg, params=dataclasses.replace(p, **change))
    grid = gg.build_grid(exp, inp)
    with pytest.raises(NotImplementedError):
        ini_masks_etc(grid, cfg=cfg, params=dataclasses.replace(p, useMin4hFacEdges=True), ex=gg.exchanger(exp))


def test_grid_pytree_and_immutability():
    import jax

    from mitjax.model.grid import Grid
    exp, inp = "advect_xy", "input"
    grid = gg.build_grid(exp, inp)
    leaves, tree = jax.tree_util.tree_flatten(grid)
    back = jax.tree_util.tree_unflatten(tree, leaves)
    assert back.names() == grid.names() and all(np.array_equal(np.asarray(a), np.asarray(b))
                                                for a, b in zip(leaves, jax.tree_util.tree_leaves(back)))
    with pytest.raises(AttributeError):
        grid.hFacC = None
    with pytest.raises(KeyError):
        grid.replace(notAGridField=1)
    with pytest.raises(AttributeError, match="has not been set"):
        Grid().hFacC


@pytest.mark.parametrize("exp,inp", gg.VARIANTS, ids=IDS)
def test_grid_extra_fields_bitwise(exp, inp):
    """kSurfC/W/S, kLowC, topoZ, cosFac*/sqCosFac*, deepFac*, hybrid sigma, rkSign, gravitySign (ported in Task 10,
    dumped since the jaxdump extension at b73c05c, run job 27828737, all nine variants INVISIBLE): bitwise on every
    point (measured 2026-10-01: 27 fields, 3246-83194 points per variant, no difference)."""
    r = gg.compare_extra(gg.build_grid(exp, inp), exp, inp)
    bad = {k: v for k, v in r.items() if v[0] == "shape" or any(v[1:])}
    assert not bad and len(r) == len(gg.EXTRA_G) + len(gg.EXTRA_V), bad


@pytest.mark.parametrize("exp,inp", [("tutorial_barotropic_gyre", "input"), ("advect_xz", "input")],
                         ids=["tutorial_barotropic_gyre/input", "advect_xz/input"])
def test_grid_extra_fields_negative_control(exp, inp):
    """No exchange: kSurfC/W/S and kLowC differ at halo points (measured: 180-324 points per field)."""
    r = gg.compare_extra(gg.build_grid(exp, inp, ex=gg.NoExchange()), exp, inp)
    assert all(r[n][1] > 100 for n in ("kSurfC", "kSurfW", "kSurfS", "kLowC")), r
