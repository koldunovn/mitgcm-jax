"""Forcing gates (M1 sub-lane FORCING, tier1x): INI_FORCING, LOAD_FIELDS_DRIVER / EXTERNAL_FIELDS_LOAD (periodic
forcing read and time interpolation), FREEZE_SURFACE, EXTERNAL_FORCING_SURF + FORCING_SURF_RELAX against the oracle's
own dumps of every M1 variant, bitwise on every point incl. halos (mitjax/tests/forcing_gate.py); the forcing record
and weight sequence against the oracle STDOUT and over a full cycle of steps against the exact time arithmetic;
negative controls measured to bite; finite gradients on all lanes and FD at smooth points.
"""

import dataclasses
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax.model.src import external_fields_load as efl
from mitjax.model.src.external_fields_load import external_fields_load, get_periodic_interval
from mitjax.model.src.external_forcing_surf import external_forcing_surf
from mitjax.model.src.freeze_surface import freeze_surface
from mitjax.model.src.load_fields_driver import load_fields_driver
from mitjax.model import state as stmod
from mitjax.tests import forcing_gate as fg
from mitjax.tests import grid_gate as gg

IDS = [f"{e}/{i}" for e, i in gg.VARIANTS]
PERIODIC = [("global_ocean.90x40x15", "input"), ("tutorial_global_oce_optim", "input_ad")]
FREEZE = PERIODIC                     # allowFreezing=.TRUE. (coverage: freeze_surface runs in these two)
# P01 inputs not dumped: PmEpR (SURFACE.h) at the first step of global_ocean.90x40x15 (written by the
# initialisation's INTEGR_CONTINUITY from hDivFlow, integr_continuity.F:140-162; read by external_forcing_surf.F:271-285
# with nonlinFreeSurf > 0 and useRealFreshWaterFlux). At later steps PmEpR = -EmPmR of the previous step's
# EXTERNAL_FORCING_SURF (integr_continuity.F:171-178, every point), which P01 of that step dumps.
PMEPR_MISSING = {("global_ocean.90x40x15", "input")}


def _load_chain(c, steps=None, patch=None, driver=True):
    """FFields after LOAD_FIELDS_DRIVER at every dumped step, starting from INI_FFIELDS + INI_FORCING; at each step
    the dumped fields of group f at S00_begin are the input (what the previous step left in FFIELDS.h), the time
    records taux0/1, ... and loadedRec (not dumped) are carried by the chain; theta from S00_begin. `driver=False`
    calls EXTERNAL_FIELDS_LOAD directly (optim: LOAD_FIELDS_DRIVER raises on useCTRL)."""
    ff = c.initial_ffields()
    out = {}
    for it in (steps or c.iterations()):
        ff = fg.ffields_with(c, ff, it, "S00_begin", fg.FORCING_F + fg.SURF_F[:4])
        st = c.state_from(it, "S00_begin", ("theta",))
        if driver:
            ff = load_fields_driver(c.my_time(it), it, ff, cfg=c.cfg, fp=c.fp, state=st, rw=c.rw, ex=c.ex)
        else:
            ff = external_fields_load(c.my_time(it), it, ff, cfg=c.cfg, fp=c.fp, rw=c.rw, ex=c.ex)
        out[it] = ff
    return out


def _uses_ctrl(c):
    return bool(c.cfg.cpp.ALLOW_CTRL)


# --------------------------------------------------------------------------------------------- INI_FORCING
@pytest.mark.parametrize("exp,inp", gg.VARIANTS, ids=IDS)
def test_ini_forcing_bitwise(exp, inp):
    """INI_FFIELDS + INI_FORCING == group f at S00_begin of the first dumped step (nothing writes these fields
    between INITIALISE_VARIA and the S00 anchor), every point."""
    c = fg.ctx(exp, inp)
    ff = c.initial_ffields()
    it = c.iterations()[0]
    res = {n: fg.compare(getattr(ff, n), c.ref(it, "S00_begin", n))
           for n in fg.FORCING_F + fg.SURF_F[:4] if n in ff and c.has(it, "S00_begin", n)}
    assert len(res) >= 10, res
    assert not fg.bad(res), res


# --------------------------------------------------------------------------------------------- LOAD_FIELDS
@pytest.mark.parametrize("exp,inp", gg.VARIANTS, ids=IDS)
def test_load_fields_bitwise(exp, inp):
    """LOAD_FIELDS_DRIVER (EXTERNAL_FIELDS_LOAD) == group f at S02_load_fields at every dumped step, every point; the
    chain carries taux0/1 etc. and loadedRec from step to step as the Fortran does."""
    c = fg.ctx(exp, inp)
    chain = _load_chain(c, driver=not _uses_ctrl(c))
    for it, ff in chain.items():
        res = {n: fg.compare(getattr(ff, n), c.ref(it, "S02_load_fields", n))
               for n in fg.FORCING_F + fg.SURF_F[:4] if n in ff and c.has(it, "S02_load_fields", n)}
        assert len(res) >= 10, res
        assert not fg.bad(res), (it, fg.bad(res))
    if c.fp.periodicExternalForcing:
        it = c.iterations()[-1]
        assert np.abs(np.asarray(chain[it].fu.data)).max() > 0 and chain[it].loadedRec > 0


def test_load_fields_driver_ctrl_raises():
    """useCTRL (optim): CTRL_MAP_GENTIM2D belongs to pkg/ctrl (another lane): LOAD_FIELDS_DRIVER refuses to run
    without it instead of skipping the call."""
    c = fg.ctx(*PERIODIC[1])
    with pytest.raises(NotImplementedError, match="CTRL_MAP_GENTIM2D"):
        _load_chain(c, steps=c.iterations()[:1], driver=True)


@pytest.mark.parametrize("exp,inp", PERIODIC, ids=[f"{e}/{i}" for e, i in PERIODIC])
def test_record_sequence_stdout(exp, inp):
    """The EXTERNAL_FIELDS_LOAD records (record indices, loadedRec, weights printed F14.10; 'Reading new data') of
    every step of the run equal the oracle STDOUT lines, character for character, in order."""
    c = fg.ctx(exp, inp)
    from mitjax.model.src.ini_ffields import ini_ffields
    want = [ln.rstrip("\n") for ln in open(c.rundir / "output.txt") if ln.startswith(" EXTERNAL_FIELDS_LOAD")]
    ff = ini_ffields(cfg=c.cfg)
    got = []
    for n in range(c.tp.nTimeSteps):
        it = c.tp.nIter0 + n
        ff = external_fields_load(c.my_time(it), it, ff, cfg=c.cfg, fp=c.fp, rw=c.rw, ex=c.ex, stdout=got)
    assert want and got == want


def _exact_interval(cycle, period, dt, t):
    """GET_PERIODIC_INTERVAL in exact rational arithmetic (record mid-points at (n-1/2)*period, periodic): the record
    pair around t, the weight of the second record, the record of t - dt."""
    C, P, T, D = Fraction(cycle), Fraction(period), Fraction(t), Fraction(dt)
    nb = int(C / P)
    loc = (T - P / 2) % C
    r1 = 1 + int(loc // P)
    r2 = 1 + r1 % nb
    w2 = (loc - P * (r1 - 1)) / P
    r0 = 1 + int(((T - P / 2 - D) % C) // P)
    return r0, r1, r2, w2


@pytest.mark.parametrize("exp,inp", PERIODIC, ids=[f"{e}/{i}" for e, i in PERIODIC])
def test_record_sequence_full_cycle(exp, inp):
    """Pure index test over 1.5 forcing cycles of steps from nIter0: the record indices of get_periodic_interval
    equal the exact rational ones at every step; the weights equal the exactly rounded rational weight (every
    operand of :118-128 is an integer number of seconds below 2**53, so fmod and the subtraction are exact and the
    one division rounds once); the steps where EXTERNAL_FIELDS_LOAD reads new records are exactly the steps whose
    time passes a record mid-point (plain test :100, intime1 != loadedRec) and, for the AUTODIFF test (:92), those
    plus nIter0."""
    c = fg.ctx(exp, inp)
    fp, tp = c.fp, c.tp
    nsteps = int(1.5 * fp.externForcingCycle / tp.deltaTClock)
    loaded, reads_plain, reads_ad, prev = 0, [], [], None
    for n in range(nsteps):
        it = tp.nIter0 + n
        t = c.my_time(it)
        r0, r1, r2, w1, w2 = get_periodic_interval(fp.externForcingCycle, fp.externForcingPeriod, tp.deltaTClock, t)
        e0, e1, e2, ew2 = _exact_interval(fp.externForcingCycle, fp.externForcingPeriod, tp.deltaTClock, t)
        assert (r0, r1, r2) == (e0, e1, e2), (it, (r0, r1, r2), (e0, e1, e2))
        assert float(w2) == float(ew2) and float(w1) == 1.0 - float(ew2), (it, w2, ew2)
        assert 0.0 <= float(w2) < 1.0
        if r2 != loaded:
            reads_plain.append(it)
            loaded = r2
        if r1 != r0 or it == tp.nIter0:
            reads_ad.append(it)
        # the record pair changes exactly when the time passes a mid-point (n-1/2)*period
        if prev is not None:
            crossed = (Fraction(t) - Fraction(fp.externForcingPeriod) / 2) // Fraction(fp.externForcingPeriod) != \
                (Fraction(prev) - Fraction(fp.externForcingPeriod) / 2) // Fraction(fp.externForcingPeriod)
            assert crossed == (r1 != r0), it
        prev = t
    per = int(fp.externForcingPeriod / tp.deltaTClock)
    assert reads_plain[0] == tp.nIter0 and len(reads_plain) == 1 + (nsteps - 1 + per // 2) // per or \
        len(reads_plain) >= nsteps // per, reads_plain
    assert reads_plain == reads_ad, (reads_plain, reads_ad)


# --------------------------------------------------------------------------------------------- FREEZE + SURF
def _p01_inputs(c, it, prev_p01_ff=None):
    """(ff, state, phi0surf, ok_fields) given the oracle's inputs of DO_OCEANIC_PHYS at step it."""
    ff = _load_chain(c, steps=[i for i in c.iterations() if i <= it], driver=not _uses_ctrl(c))[it]
    ff = fg.ffields_with(c, ff, it, fg.forcing_input_stage(c, it), fg.FORCING_F)
    names = ("theta", "salt", "PmEpR")
    st = c.state_from(it, "S00_begin", ("theta", "salt"))
    pmepr = stmod.declare("PmEpR", c.sz)
    if prev_p01_ff is not None:                       # integr_continuity.F:173-178 of the previous step
        pmepr = type(pmepr)(-prev_p01_ff, "PmEpR", tiled=pmepr.tiled, _dims=pmepr.dims)
    else:                                             # not dumped (PMEPR_MISSING); read only by the ungated field
        pmepr = type(pmepr)(jnp.zeros(pmepr.data.shape), "PmEpR", tiled=pmepr.tiled, _dims=pmepr.dims)
    st = stmod.State({"theta": st.theta, "salt": st.salt, "PmEpR": pmepr})
    g = c.grid
    hs = fg.hfac_stage(c, it, "P01_external_forcing_surf")
    if hs != "S00_begin" or c.has(it, "S00_begin", "hFacC"):
        g = g.replace(hFacC=c.fa(it, hs, "hFacC", g.hFacC))
    phi0 = c.fa(it, "S00_begin", "phi0surf", stmod.declare("etaN", c.sz))
    return ff, st, g, phi0, names


def _run_p01(c, it, prev=None):
    ff, st, g, phi0, _ = _p01_inputs(c, it, prev)
    sz = c.sz
    if c.fp.allowFreezing:
        st, ff = freeze_surface(c.my_time(it), it, ff, cfg=c.cfg, grid=g, fp=c.fp, state=st)
    ff, st, phi0 = external_forcing_surf(1-sz.OLx, sz.sNx+sz.OLx, 1-sz.OLy, sz.sNy+sz.OLy, c.my_time(it), it, ff,
                                         cfg=c.cfg, grid=g, fp=c.fp, state=st, phi0surf=phi0)
    return ff, st, phi0


@pytest.mark.parametrize("exp,inp", gg.VARIANTS, ids=IDS)
def test_external_forcing_surf_bitwise(exp, inp):
    """FREEZE_SURFACE (allowFreezing) + EXTERNAL_FORCING_SURF (+ FORCING_SURF_RELAX) on the dumped inputs ==
    group f at P01_external_forcing_surf (surfaceForcingU/V/T/S, EmPmR, phi0surf) at every dumped step, every point.
    PmEpR: see PMEPR_MISSING (global_ocean first step: surfaceForcingS not gated, reported)."""
    c = fg.ctx(exp, inp)
    prev = None
    gated = 0
    for it in c.iterations():
        ff, st, phi0 = _run_p01(c, it, prev)
        names = list(fg.SURF_F)
        if (exp, inp) in PMEPR_MISSING and prev is None:
            names.remove("surfaceForcingS")
        res = {n: fg.compare(getattr(ff, n), c.ref(it, "P01_external_forcing_surf", n)) for n in names}
        if c.cfg.cpp.ATMOSPHERIC_LOADING:
            res["phi0surf"] = fg.compare(phi0, c.ref(it, "P01_external_forcing_surf", "phi0surf"))
        assert not fg.bad(res), (it, fg.bad(res))
        gated += len(res)
        if (exp, inp) in PMEPR_MISSING:
            prev = jnp.asarray(np.asarray(c.ref(it, "P01_external_forcing_surf", "EmPmR"))[:, 0])
    assert gated >= 3 * 5


@pytest.mark.parametrize("exp,inp", FREEZE, ids=[f"{e}/{i}" for e, i in FREEZE])
def test_freeze_surface_bitwise(exp, inp):
    """FREEZE_SURFACE on theta of S00_begin == theta at S04_oceanic_phys (all levels, every point) at every dumped
    step; the branch fires (cold points exist) in the dumped steps."""
    c = fg.ctx(exp, inp)
    fired = 0
    for it in c.iterations():
        st = c.state_from(it, "S00_begin", ("theta",))
        g = c.grid
        hs = fg.hfac_stage(c, it, "P01_external_forcing_surf")
        g = g.replace(hFacC=c.fa(it, hs, "hFacC", g.hFacC))
        ff = fg.ffields_with(c, c.initial_ffields(), it, "S00_begin", ())
        st2, ff2 = freeze_surface(c.my_time(it), it, ff, cfg=c.cfg, grid=g, fp=c.fp, state=st)
        res = fg.compare(st2.theta, c.ref(it, "S04_oceanic_phys", "theta"))
        assert not any(res[1:]), (it, res)
        fired += int(np.count_nonzero(np.asarray(ff2.adjustColdSST_diag.data) != 0))
    assert fired > 0, "FREEZE_SURFACE never changed theta in the dumped steps: the gate does not exercise :55-59"


# --------------------------------------------------------------------------------------------- negative controls
def test_negative_control_record_off_by_one():
    """Planted error: every record index one month later (intime0+1, intime1+1). Bites: fu/fv/Qnet/EmPmR at S02 of
    global_ocean differ at (nearly) every wet point."""
    c = fg.ctx(*PERIODIC[0])
    orig = efl.get_periodic_interval

    def shifted(*a):
        r0, r1, r2, w1, w2 = orig(*a)
        n = int(round(c.fp.externForcingCycle / c.fp.externForcingPeriod))
        return r0 % n + 1, r1 % n + 1, r2 % n + 1, w1, w2
    efl.get_periodic_interval = shifted
    try:
        ff = _load_chain(c, steps=c.iterations()[:1])[c.iterations()[0]]
    finally:
        efl.get_periodic_interval = orig
    it = c.iterations()[0]
    res = {n: fg.compare(getattr(ff, n), c.ref(it, "S02_load_fields", n)) for n in ("fu", "fv", "Qnet", "EmPmR")}
    assert all(v[1] > 1000 for v in res.values()), res


def test_negative_control_weights_swapped():
    """Planted error: bWght and aWght swapped. Bites at step nIter0+1 (weights 0.4667/0.5333; at nIter0 both are
    0.5 and a swap is invisible: measured, the control uses the second step)."""
    c = fg.ctx(*PERIODIC[0])
    orig = efl.get_periodic_interval

    def swapped(*a):
        r0, r1, r2, w1, w2 = orig(*a)
        return r0, r1, r2, w2, w1
    efl.get_periodic_interval = swapped
    try:
        chain = _load_chain(c, steps=c.iterations()[:2])
    finally:
        efl.get_periodic_interval = orig
    it0, it1 = c.iterations()[:2]
    res0 = {n: fg.compare(getattr(chain[it0], n), c.ref(it0, "S02_load_fields", n)) for n in ("fu", "Qnet")}
    res1 = {n: fg.compare(getattr(chain[it1], n), c.ref(it1, "S02_load_fields", n)) for n in ("fu", "Qnet")}
    assert not fg.bad(res0), res0
    assert all(v[1] > 1000 for v in res1.values()), res1


def test_negative_control_relax_sign():
    """Planted error: the restoring term of FORCING_SURF_RELAX with the opposite sign (theta - SST -> SST - theta,
    via negated lambda). Bites: surfaceForcingT at P01 of global_ocean differs at the relaxed points."""
    c = fg.ctx(*PERIODIC[0])
    it = c.iterations()[1]
    ff, st, g, phi0, _ = _p01_inputs(c, it, jnp.asarray(np.asarray(
        c.ref(c.iterations()[0], "P01_external_forcing_surf", "EmPmR"))[:, 0]))
    ff = ff.replace(lambdaThetaClimRelax=type(ff.lambdaThetaClimRelax)(
        -ff.lambdaThetaClimRelax.data, "lambdaThetaClimRelax", tiled=True, _dims=ff.lambdaThetaClimRelax.dims))
    sz = c.sz
    st, ff = freeze_surface(c.my_time(it), it, ff, cfg=c.cfg, grid=g, fp=c.fp, state=st)
    ff, st, phi0 = external_forcing_surf(1-sz.OLx, sz.sNx+sz.OLx, 1-sz.OLy, sz.sNy+sz.OLy, c.my_time(it), it, ff,
                                         cfg=c.cfg, grid=g, fp=c.fp, state=st, phi0surf=phi0)
    res = fg.compare(ff.surfaceForcingT, c.ref(it, "P01_external_forcing_surf", "surfaceForcingT"))
    assert res[1] > 1000, res


# --------------------------------------------------------------------------------------------- gradients
def _surf_cost_fn(c, it):
    ff0, st0, g, phi0, _ = _p01_inputs(c, it, jnp.asarray(np.asarray(
        c.ref(c.iterations()[0], "P01_external_forcing_surf", "EmPmR"))[:, 0]))
    sz = c.sz
    w = jnp.asarray(np.random.default_rng(1).standard_normal(ff0.surfaceForcingT.data.shape))

    def cost(qnet, theta):
        ff = ff0.replace(Qnet=type(ff0.Qnet)(qnet, "Qnet", tiled=True, _dims=ff0.Qnet.dims))
        st = st0.replace(theta=type(st0.theta)(theta, "theta", tiled=True, _dims=st0.theta.dims))
        st, ff = freeze_surface(c.my_time(it), it, ff, cfg=c.cfg, grid=g, fp=c.fp, state=st)
        ff, st, _ = external_forcing_surf(1-sz.OLx, sz.sNx+sz.OLx, 1-sz.OLy, sz.sNy+sz.OLy, c.my_time(it), it, ff,
                                          cfg=c.cfg, grid=g, fp=c.fp, state=st, phi0surf=phi0)
        return jnp.sum(w * (ff.surfaceForcingT.data + ff.surfaceForcingS.data))
    return cost, ff0.Qnet.data, st0.theta.data


def test_gradient_finite_and_fd():
    """d cost / d (Qnet, theta) through FREEZE_SURFACE + EXTERNAL_FORCING_SURF + FORCING_SURF_RELAX (global_ocean,
    second dumped step): finite on every lane (halos, land); central FD at a smooth wet interior point (theta above
    freezing) matches to 1e-7 relative (the cost is linear in Qnet and piecewise linear in theta: FD is exact up to
    rounding)."""
    c = fg.ctx(*PERIODIC[0])
    it = c.iterations()[1]
    cost, q, th = _surf_cost_fn(c, it)
    gq, gt = jax.grad(cost, argnums=(0, 1))(q, th)
    assert np.isfinite(np.asarray(gq)).all() and np.isfinite(np.asarray(gt)).all()
    sz = c.sz
    idx = (5, 0, sz.OLy + 5, sz.OLx + 5)
    assert float(c.grid.maskC.data[idx]) == 1.0 and float(th[idx]) > -1.0
    for x, gx, which in ((q, gq, 0), (th, gt, 1)):
        h = 1e-3 * max(1.0, abs(float(x[idx[:1] + idx[2:]] if x.ndim == 3 else x[idx])))
        e = jnp.zeros_like(x).at[idx[:1] + idx[2:] if x.ndim == 3 else idx].set(h)
        args_p = (q + e, th) if which == 0 else (q, th + e)
        args_m = (q - e, th) if which == 0 else (q, th - e)
        fd = (float(cost(*args_p)) - float(cost(*args_m))) / (2 * h)
        ad = float(gx[idx[:1] + idx[2:]] if x.ndim == 3 else gx[idx])
        assert ad != 0.0 and abs(fd - ad) <= 1e-7 * abs(ad), (which, fd, ad)
