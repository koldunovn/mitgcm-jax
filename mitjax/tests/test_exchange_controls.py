"""mitjax/eesupp/{exchange,exch_maps}.py (plan Task 7b), tier 1x: the adjoint identity, several levels, unprobed
options, and the negative controls of the probe gate (test_exchange.py), each measured to bite on the real maps."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax.eesupp import exch_maps as EM
from mitjax.eesupp import exchange as X
from mitjax.eesupp.exchange import Exchanger
from mitjax.tests.exch_gate import ROUTINES, bits, probe_dumps, probe_mismatches

LAYOUTS = tuple(EM.M1_PROBES)


def _transpose_np(maps, name, y):
    """Explicit transpose of a scalar exchange from its map (numpy scatter-add), independent of JAX's AD."""
    src, comp, sign = maps.maps[name]
    x = np.zeros_like(y)
    keep = comp == 0
    np.add.at(x, (Ellipsis, np.nonzero(keep)[0]), y[..., keep])
    sel = comp == 1
    np.add.at(x, (Ellipsis, src[sel]), y[..., sel] * sign[sel])
    return (x,)


def _transpose_vec_np(maps, name, yu, yv):
    """The same for a vector exchange (u and v outputs, sources in either input)."""
    xu, xv = np.zeros_like(yu), np.zeros_like(yv)
    for key, y, own_target in ((name + "_u", yu, xu), (name + "_v", yv, xv)):
        src, comp, sign = maps.maps[key]
        keep = comp == 0
        np.add.at(own_target, (Ellipsis, np.nonzero(keep)[0]), y[..., keep])
        for c, target in ((1, xu), (2, xv)):
            sel = comp == c
            np.add.at(target, (Ellipsis, src[sel]), y[..., sel] * sign[sel])
    return xu, xv


@pytest.mark.parametrize("exp", LAYOUTS)
def test_adjoint_identity(exp):
    """<E x, y> == <x, E^T y> with E^T from jax.vjp on random [tile, k=3, j, i] data (all lanes), and E^T == an explicit
    scatter-add transpose built from the map (<= 1e-15 of the field's max: the scatter-adds sum a few terms in another
    order), for every scalar and vector exchange."""
    maps = EM.load_maps(exp)
    ex = Exchanger(maps)
    L = maps.layout
    rng = np.random.default_rng(11)
    shp = (L.nTiles, 3, L.ny, L.nx)
    for name in list(EM.SCALAR) + list(EM.VECTOR):
        vec = name in EM.VECTOR
        xs = tuple(rng.standard_normal(shp) for _ in range(1 + vec))
        ys = tuple(rng.standard_normal(shp) for _ in range(1 + vec))
        f = (lambda u, v, n=name: ex.vector(n, u, v)) if vec else (lambda u, n=name: (ex.scalar(n, u),))
        out, pull = jax.vjp(f, *xs)
        xt = pull(tuple(jnp.asarray(y) for y in ys))
        lhs = sum(float(np.vdot(o, y)) for o, y in zip(out, ys))
        rhs = sum(float(np.vdot(x, t)) for x, t in zip(xs, xt))
        assert abs(lhs - rhs) <= 1e-13 * abs(lhs), (exp, name, lhs, rhs)
        flat = lambda a: np.moveaxis(np.asarray(a), 0, -3).reshape(3, -1)   # noqa: E731
        if vec:
            ref = _transpose_vec_np(maps, name, flat(ys[0]), flat(ys[1]))
        else:
            ref = _transpose_np(maps, name, flat(ys[0]))
        for t, r in zip(xt, ref):     # scatter-adds of a few terms in another order: ulps of the field's max
            err = np.max(np.abs(flat(t) - r)) / np.max(np.abs(r))
            assert err <= 1e-15, (exp, name, err)


def test_several_levels_and_unprobed_options():
    """EXCH_3D_RL on 3 levels: level k is the probe output plus k*D (D = 10*cbase keeps the codes exact; no sign flips
    in M1), bitwise; an exchange option the probe did not measure is an error."""
    exp = "tutorial_baroclinic_gyre"
    maps = EM.load_maps(exp)
    ds, it = probe_dumps(exp)
    ex = Exchanger(maps)
    cbase = ds.scalar(it, EM.STAGE, "xCbase")
    pu, _ = EM.probe_inputs(maps.layout, cbase)
    D = 10.0 * cbase
    a = np.stack([pu + k * D for k in range(3)], axis=1)
    got = np.asarray(jax.jit(lambda e, a: e.EXCH_3D_RL(a))(ex, a))
    want = ds.field(it, EM.STAGE, "x3D")[:, 0]
    for k in range(3):
        assert np.array_equal(bits(got[:, k]), bits(want + k * D)), k
    for call in (lambda: ex.EXCH_UV_DGRID_3D_RL(a, a, False), lambda: ex.EXCH_UV_3D_RL(a, a, False),
                 lambda: ex.EXCH_SM_3D_RL(a, False)):
        with pytest.raises(NotImplementedError):
            call()


def _with_map(maps, key, src=None, comp=None, sign=None):
    s, c, g = (x.copy() for x in maps.maps[key])
    for arr, new in ((s, src), (c, comp), (g, sign)):
        if new is not None:
            arr[new[0]] = new[1]
    return dataclasses.replace(maps, maps=dict(maps.maps, **{key: (s, c, g)}))


def test_negative_controls_bite():
    """Planted errors, each on the real maps of one layout, each failing the probe gate where it must:
    one sign flipped in a vector map (EXCH_UV_XY_RL u output, one written halo point; also flips that point's +0
    probe to -0); one wrong source; -0 turned into +0 by an `+ 0.0` in the exchange formula. The map builder refuses a
    probe whose interior changed; map files are never overwritten; a wrong sha256 is refused."""
    exp = "tutorial_baroclinic_gyre"
    maps = EM.load_maps(exp)
    ds, it = probe_dumps(exp)
    assert sum(probe_mismatches(Exchanger(maps), ds, it).values()) == 0
    p = int(np.nonzero(maps.maps["UVs_u"][1] > 0)[0][0])          # first written halo point of the u output

    flipped = probe_mismatches(Exchanger(_with_map(maps, "UVs_u", sign=(p, -1))), ds, it)
    bad = {k: v for k, v in flipped.items() if v}
    print(f"sign flipped at one point: {bad}")
    assert bad == {"xUVs_u": 1, "zpUVs_u": 1, "zmUVs_u": 1}, bad

    wrong_src = probe_mismatches(Exchanger(_with_map(maps, "XY", src=(p, maps.maps["XY"][0][p] + 1))), ds, it)
    print(f"one wrong source: { {k: v for k, v in wrong_src.items() if v} }")
    assert wrong_src["xT"] == 1

    orig = X.apply_map
    try:
        # (XLA:CPU folds a constant `+ 0.0` away under jit, measured 2026-10-01, so the planted error is a select)
        X.apply_map = lambda m, own, fa, fb: (lambda o: jnp.where(o == 0.0, jnp.zeros_like(o), o))(
            orig(m, own, fa, fb))
        jax.clear_caches()
        lost = probe_mismatches(Exchanger(maps), ds, it)
    finally:
        X.apply_map = orig
        jax.clear_caches()
    lost_bad = {k: v for k, v in lost.items() if v}
    print(f"-0 lost: {lost_bad}")
    assert lost_bad and all(k.startswith("zm") for k in lost_bad), lost_bad

    L = maps.layout
    cbase = ds.scalar(it, EM.STAGE, "xCbase")
    pu, _ = EM.probe_inputs(L, cbase)
    tampered = ds.field(it, EM.STAGE, "xT")[:, 0].copy()
    tampered[0, L.OLy, L.OLx] = pu[0, L.OLy, L.OLx + 1]                # interior point copied from its neighbour
    s, c, g = EM.decode_field(tampered, cbase, L)
    with pytest.raises(ValueError, match="interior point"):
        EM.finish_map("XY", s, c, g, L)

    path = EM.map_path(exp, L)
    with pytest.raises(FileExistsError):
        maps.save(path)
    with pytest.raises(ValueError, match="sha256"):
        EM.ExchangeMaps.load(path, "0" * 64)


def test_routine_table_complete():
    """Every probed exchange output is exercised by the gate (exch_gate.ROUTINES), and every map has a routine."""
    fields = {f for flds, _ in ROUTINES for f in flds}
    assert fields == set(EM.SCALAR.values()) | {f for pair in EM.VECTOR.values() for f in pair}
