"""SBO gates (M1 sub-lane FORCING, tier1x): SBO_CALC on the oracle's dumped state of global_ocean.90x40x15 == the
SBO.h scalars dumped at S19_sbo_calc (bitwise, every dumped scalar) for the initial call and the end of every dumped
step; SBO_OUTPUT's %SBO records == the oracle STDOUT records character for character; negative controls (sum order,
tile order, rotation) measured to bite; finite gradients and FD.

Inputs of SBO_CALC (do_the_model_io.F:181, after the step): uVel, vVel, etaN at S16_blocking_exchanges; rhoInSitu at
S04_oceanic_phys (written by FIND_RHO in DO_OCEANIC_PHYS only); hFacC as the step left it (the last r* stage that
dumps it, S12_calc_rstar); sIceLoad at P01. The initial call (THE_MAIN_LOOP before the first step: occurrence 0 of
S19 at nIter0) reads the S00_begin state of the first step.
"""

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.eesupp import global_sum as gs
from mitjax.model.state import State
from mitjax.pkg.sbo import sbo_calc as sc
from mitjax.pkg.sbo.sbo_calc import SboCommon, sbo_calc
from mitjax.pkg.sbo.sbo_output import new_stdout, sbo_output
from mitjax.pkg.sbo.sbo_readparms import sbo_readparms
from mitjax.tests import forcing_gate as fg

EXP = ("global_ocean.90x40x15", "input")


def _inputs(c, it, initial):
    """(state, grid, ff) of the SBO_CALC call at the end of step `it` (or the initial call)."""
    if initial:
        st = c.state_from(it, "S00_begin", ("etaN", "uVel", "vVel", "rhoInSitu"))
        hs, fs = "S00_begin", "S00_begin"
    else:
        d = c.state_from(it, "S16_blocking_exchanges", ("etaN", "uVel", "vVel"))
        r = c.state_from(it, "S04_oceanic_phys", ("rhoInSitu",))
        st = State({"etaN": d.etaN, "uVel": d.uVel, "vVel": d.vVel, "rhoInSitu": r.rhoInSitu})
        hs = "S12_calc_rstar" if c.has(it, "S12_calc_rstar", "hFacC") else fg.hfac_stage(c, it, "S16_blocking_exchanges")
        fs = "P01_external_forcing_surf"
    g = c.grid.replace(hFacC=c.fa(it, hs, "hFacC", c.grid.hFacC))
    ff = c.initial_ffields()
    ff = ff.replace(sIceLoad=c.fa(it, fs, "sIceLoad", ff.sIceLoad))
    return st, g, ff


def _calls(c):
    """(iteration key, occurrence, initial?, myIter of the call, myTime of the call) of every dumped SBO_CALC call."""
    out = []
    for it in c.iterations():
        n = c.ds.n_occ((it, "S19_sbo_calc", "mass"))
        if it == c.tp.nIter0 and n == 2:
            out.append((it, 0, True, it, c.my_time(it)))
        out.append((it, n - 1, False, it + 1, c.my_time(it + 1)))
    return out


def _compare_scalars(c, it, occ, sbo):
    res = {}
    for n in SboCommon.names():
        if (it, "S19_sbo_calc", n) in c.ds.index:
            ref = c.ds.scalar(it, "S19_sbo_calc", n, occ=occ)
            ours = float(np.asarray(getattr(sbo, n)))
            res[n] = (ours, ref, bool(np.float64(ours).view(np.int64) == np.float64(ref).view(np.int64)))
    return res


def test_sbo_calc_bitwise_and_stdout():
    """Every dumped SBO.h scalar bitwise at every dumped call (initial + 3 steps), and the four %SBO records of
    SBO_OUTPUT equal the oracle STDOUT's records of the same calls, character for character."""
    c = fg.ctx(*EXP)
    sp = sbo_readparms(c.e, c.fp)
    want = [ln.rstrip("\n") for ln in open(c.rundir / "output.txt") if "%SBO" in ln]
    got_all = []
    calls = _calls(c)
    assert len(calls) == 4
    for it, occ, initial, myIter, myTime in calls:
        st, g, ff = _inputs(c, it, initial)
        sbo = sbo_calc(myTime, myIter, cfg=c.cfg, grid=g, fp=c.fp, state=st, ff=ff)
        res = _compare_scalars(c, it, occ, sbo)
        assert len(res) >= 19, sorted(res)
        bad = {k: v for k, v in res.items() if not v[2]}
        assert not bad, (it, occ, bad)
        out = new_stdout()
        sbo_output(myTime, myIter, sbo, sp=sp, nIter0=c.tp.nIter0, deltaTClock=c.tp.deltaTClock, stdout=out)
        got_all += out.units[6]
    assert len(got_all) == 16 and got_all == want[:16], (got_all, want[:16])


def test_negative_control_sum_order():
    """Planted errors: (a) the tile partial sums of SBO_CALC added with jnp.sum (tree order) instead of the Fortran
    chain; (b) the tiles of GLOBAL_SUM_TILE_RL added in reverse order. Each bites: at least one dumped scalar
    differs (measured)."""
    c = fg.ctx(*EXP)
    it, occ, initial, myIter, myTime = _calls(c)[1]
    st, g, ff = _inputs(c, it, initial)
    orig_chain, orig_gsum = sc._chain, sc._gsum

    def tree_chain(terms):
        t = jnp.asarray(terms)
        return jnp.sum(t.reshape(t.shape[0], -1), axis=1)

    def reversed_gsum(part, ex):
        return gs.global_sum_tile(jnp.asarray(part)[::-1])
    counts = {}
    for name, (attr, fn) in (("tree_chain", ("_chain", tree_chain)), ("reversed_tiles", ("_gsum", reversed_gsum))):
        setattr(sc, attr, fn)
        try:
            sbo = sbo_calc(myTime, myIter, cfg=c.cfg, grid=g, fp=c.fp, state=st, ff=ff)
        finally:
            sc._chain, sc._gsum = orig_chain, orig_gsum
        res = _compare_scalars(c, it, occ, sbo)
        counts[name] = sum(1 for v in res.values() if not v[2])
    assert all(n > 0 for n in counts.values()), counts


def test_negative_control_rotation():
    """Planted error: UE/VN taken without the C-to-A averaging (uVel, vVel themselves). Bites: the current OAM
    xoamc and zoamc differ."""
    c = fg.ctx(*EXP)
    it, occ, initial, myIter, myTime = _calls(c)[1]
    st, g, ff = _inputs(c, it, initial)
    orig = sc.rotate_uv2en_rl
    sc.rotate_uv2en_rl = lambda u, v, ue, vn, *a, **k: (u, v, u, v)
    try:
        sbo = sbo_calc(myTime, myIter, cfg=c.cfg, grid=g, fp=c.fp, state=st, ff=ff)
    finally:
        sc.rotate_uv2en_rl = orig
    res = _compare_scalars(c, it, occ, sbo)
    assert not res["zoamc"][2] and not res["xoamc"][2], res


def test_sbo_gradient_finite_and_fd():
    """d zoamc / d uVel and d mass / d etaN through SBO_CALC: finite on every lane (halos, land); central FD at a
    wet interior point (both are linear in these inputs: FD is exact up to rounding, 1e-6 relative, with steps above
    the forward-noise floor)."""
    c = fg.ctx(*EXP)
    it, occ, initial, myIter, myTime = _calls(c)[1]
    st0, g, ff = _inputs(c, it, initial)
    sz = c.sz

    def f(u, eta, name):
        st = st0.replace(uVel=type(st0.uVel)(u, "uVel", tiled=True, _dims=st0.uVel.dims),
                         etaN=type(st0.etaN)(eta, "etaN", tiled=True, _dims=st0.etaN.dims))
        return getattr(sbo_calc(myTime, myIter, cfg=c.cfg, grid=g, fp=c.fp, state=st, ff=ff), name)
    u, eta = st0.uVel.data, st0.etaN.data
    gu = jax.grad(lambda u_: f(u_, eta, "zoamc"))(u)
    ge = jax.grad(lambda e_: f(u, e_, "mass"))(eta)
    assert np.isfinite(np.asarray(gu)).all() and np.isfinite(np.asarray(ge)).all()
    t, j, i = 5, sz.OLy + 5, sz.OLx + 5
    assert float(g.maskC.data[t, 0, j, i]) == 1.0 and float(g.maskW.data[t, 0, j, i]) == 1.0
    h = 1e-3
    du = jnp.zeros_like(u).at[t, 0, j, i].set(h)
    fd = (float(f(u + du, eta, "zoamc")) - float(f(u - du, eta, "zoamc"))) / (2 * h)
    ad = float(gu[t, 0, j, i])
    assert ad != 0.0 and abs(fd - ad) <= 1e-6 * abs(ad), (fd, ad)
    # mass ~ 1.4e21: an h of 1e-3 m moves it by ~2e11, below the forward-noise floor (~3e5 per rounding, 1e-6
    # relative, measured); mass is linear in etaN, so a 1 m step is exact up to rounding (1e-9 relative)
    h = 1.0
    de = jnp.zeros_like(eta).at[t, j, i].set(h)
    fd = (float(f(u, eta + de, "mass")) - float(f(u, eta - de, "mass"))) / (2 * h)
    ad = float(ge[t, j, i])
    assert ad != 0.0 and abs(fd - ad) <= 1e-6 * abs(ad), (fd, ad)
