"""r* / nonlinear free surface (M1 lane RSTAR, plan Task 15a): CALC_R_STAR, UPDATE_R_STAR, RESET_NLFS_VARS,
UPDATE_SURF_DR against the oracle's substep dumps (mitjax/tests/rstar_gate.py: inputs teacher-forced from the dumps,
outputs compared on every point of every tile, halos included, by element equality, bit pattern and finiteness).

Variants: global_ocean.90x40x15/input (select_rStar = 2, nonlinFreeSurf = 4, doResetHFactors; steps 36000-36002),
advect_xz/input.nlfs (the same r* settings, staggerTimeStep, momStepping = .FALSE.; steps 0, 1, 2, 9), and for
UPDATE_SURF_DR the linear free surface of the NONLIN_FRSURF build, advect_xz/input and input.pqm.
Each negative control is a planted error measured to bite on the same data. tier1x (about a minute on a compute node).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax.farray import FArray
from mitjax.model.src.calc_r_star import calc_r_star_host
from mitjax.tests import rstar_gate as R

GO = ("global_ocean.90x40x15", "input")
NLFS = ("advect_xz", "input.nlfs")


def _ids(v):
    return f"{v[0]}/{v[1]}"


def _assert_bitwise(res, what):
    bad = R.failures(res)
    assert not bad, f"{what}: {bad}"
    assert all(v[0] > 0 for v in res.values()), f"{what}: nothing compared {res}"


def _steps(v, stage):
    _, its, _ = R.oracle(*v)
    return [it for it in its if stage in R.stages(*v, it)]


# ---------------------------------------------------------------------------------------------------------------
# CALC_R_STAR (forward_step.F:949)

@pytest.mark.parametrize("v", R.RSTAR_VARIANTS, ids=_ids)
def test_calc_r_star_bitwise(v):
    """S07 rStarFac + S11 etaH -> S12 (rStarFacC/W/S, rStarDhCDt) and G00 R of the next step (rStarFacNm1*,
    rStarExp*, rStarDh*Dt; pStarFacK untouched), every dumped step; counters all zero, no STOP and no warning on the
    host; no zero denominator at :306-311 on any lane of the inputs (the guard is inert on the gated data)."""
    steps = _steps(v, "S12_calc_rstar")
    assert len(steps) == (3 if v == GO else 4)
    n_next = 0
    for it in steps:
        res, counters, zero_den = R.gate_calc_r_star(*v, it)
        _assert_bitwise(res, f"{_ids(v)} it={it}")
        n_next += any(k.startswith("G00+1/") for k in res)
        assert zero_den == 0
        for k in ("icntc1", "icntw", "icnts", "icntc2"):
            assert np.asarray(counters[k]).dtype == np.int32 and not np.asarray(counters[k]).any(), k
        lines, nw = calc_r_star_host(counters, it + 1, 0, cfg=R.experiment(*v).cfg)
        assert lines == [] and nw == 0
    assert n_next == 2


# ---------------------------------------------------------------------------------------------------------------
# UPDATE_R_STAR (.TRUE. at forward_step.F:839, .FALSE. at :475), RESET_NLFS_VARS (:467)

@pytest.mark.parametrize("v", R.RSTAR_VARIANTS, ids=_ids)
def test_update_r_star_latest_bitwise(v):
    """UPDATE_R_STAR(.TRUE.): hFacC/W/S, recip_hFacC vs S07 and recip_hFacW/S vs G00 of the next step."""
    for it in _steps(v, "S07_update_rstar_T"):
        res = R.gate_update_r_star(*v, it, True)
        _assert_bitwise(res, f"{_ids(v)} it={it}")


def test_update_r_star_nm1_and_reset_global_ocean_bitwise():
    """doResetHFactors (global_ocean): RESET_NLFS_VARS (pStarFacK vs G00 R of the next step) and
    UPDATE_R_STAR(.FALSE.) (hFacC/W/S, recip_hFacC vs S01; recip_hFacW/S vs G00 of the step, which the Fortran
    computed at the previous :839 from the same rStarFac: the inputs are measured equal here)."""
    steps = _steps(GO, "S01_update_rstar_F")
    assert len(steps) == 3
    for it in steps:
        _assert_bitwise(R.gate_update_r_star(*GO, it, False), f"S01 it={it}")
        if it + 1 in steps:
            _assert_bitwise(R.gate_reset_nlfs_vars(*GO, it), f"reset it={it}")
            for c in ("C", "W", "S"):            # rStarFacNm1 of step it+1 == rStarFac used at :839 of step it
                a = R.field(*GO, it + 1, "G00_geometry", f"rStarFacNm1{c}")
                b = R.field(*GO, it, "S07_update_rstar_T", f"rStarFac{c}")
                assert np.array_equal(a.view(np.int64), b.view(np.int64))


@pytest.mark.parametrize("v", R.SURFDR_VARIANTS, ids=_ids)
def test_update_surf_dr_linear_fs_bitwise(v):
    """UPDATE_SURF_DR (forward_step.F:852, nonlinFreeSurf = 0): hFacC = h0FacC, recip_hFacC vs S00 of the next
    step, hFacW/S untouched."""
    _, its, _ = R.oracle(*v)
    n = 0
    for it in its:
        if it + 1 in its:
            _assert_bitwise(R.gate_update_surf_dr(*v, it), f"{_ids(v)} it={it}")
            n += 1
    assert n == 2


# ---------------------------------------------------------------------------------------------------------------
# INITIALISE_VARIA's sequence (initialise_varia.F:297-346) and the operator UPDATE_CG2D builds from our hFac

@pytest.mark.parametrize("v", R.RSTAR_VARIANTS, ids=_ids)
def test_init_sequence_bitwise(v):
    """INI_NLFS_VARS (core lane) -> CALC_R_STAR -> UPDATE_R_STAR(.TRUE.) -> CALC_R_STAR == S00 (r group), G00 R and
    G00 recip_hFacW/S of the first step: 20 fields, every point."""
    res, cs, eta_ok, _ = R.init_chain(*v)
    assert eta_ok[0] > 0 and not any(eta_ok[1:]), eta_ok
    _assert_bitwise(res, _ids(v))
    assert len(res) == 20
    for c in cs:
        assert not any(np.asarray(c[k]).any() for k in ("icntc1", "icntw", "icnts", "icntc2"))


def _cg2d_on(hW, hS, cg2dh, myIter):
    from mitjax.model.src.cg2d_h import ini_parms_cg2d
    from mitjax.model.src.update_cg2d import update_cg2d
    from mitjax.tests import cg2d_gate as G
    e = R.experiment(*GO)
    p = ini_parms_cg2d(e)
    g, s = G.teacher_geometry(*GO)
    g.hFacW, g.hFacS = hW, hS
    f = jax.jit(lambda c, g, s, p, x: update_cg2d(c, None, myIter, cfg=e.cfg, grid=g, surface=s, params=p, ex=x))
    return f(cg2dh, g, s, p, R.exchanger(GO[0]))


def _cg2d_chain(hfac_of_step):
    """INI_CG2D, UPDATE_CG2D at nIter0 on our init-sequence hFac, then UPDATE_CG2D of every dumped step on
    hfac_of_step(it); yields (it, CG2D.h)."""
    from mitjax.tests import cg2d_gate as G
    c, prm = G.run_ini_cg2d(*GO)
    _, _, _, g0 = R.init_chain(*GO)
    c = _cg2d_on(g0.hFacW, g0.hFacS, c, prm.nIter0)
    _, its, _ = R.oracle(*GO)
    for it in its:
        hW, hS = hfac_of_step(it)
        c = _cg2d_on(hW, hS, c, it + 1)
        yield it, c


def _our_hfac(it):
    sz = R.experiment(*GO).cfg.size
    g = R.run_update_r_star(*GO, True, R.teacher_grid(*GO, it, "S01_update_rstar_F"),
                            R.teacher_state(*GO, it, "S00_begin"))
    return g.hFacW, g.hFacS


def _compare_s08(c, it):
    return {n: R.compare(getattr(c, n), R.field(*GO, it, "S08_update_cg2d", n))
            for n in ("aW2d", "aS2d", "aC2d", "pW", "pS", "pC")}


def test_update_cg2d_on_our_hfac_global_ocean_bitwise():
    """UPDATE_CG2D (forward_step.F:869, CG2D lane) on the hFacW/S of our UPDATE_R_STAR(.TRUE.) == S08 at every
    dumped step: the operator the solver gets is built from the hFac the r* routines leave."""
    n = 0
    for it, c in _cg2d_chain(_our_hfac):
        _assert_bitwise(_compare_s08(c, it), f"S08 it={it}")
        n += 1
    assert n == 3


# ---------------------------------------------------------------------------------------------------------------
# P = 4 == P = 1 (36 tiles of the exch2 layout on 4 fake CPU devices)

def test_rstar_P4_equals_P1_global_ocean():
    """CALC_R_STAR + UPDATE_R_STAR(.TRUE.) inside jit(shard_map(check_vma=True)) on 4 devices == P = 1 bit for bit
    (every output field, every point) and the counters gathered in tile order on every device."""
    for it in (36000, 36002):
        (s4, c4, g4), (s1, c1, g1) = R.run_sharded(*GO, it)
        for n in R.CALC_OUT_G00 + R.R_FAC:
            a, b = np.asarray(getattr(s4, n).data), np.asarray(getattr(s1, n).data)
            assert np.array_equal(a.view(np.int64), b.view(np.int64)), n
        for n in R.HFAC + ("recip_hFacW", "recip_hFacS"):
            a, b = np.asarray(getattr(g4, n).data), np.asarray(getattr(g1, n).data)
            assert np.array_equal(a.view(np.int64), b.view(np.int64)), n
        for k in c4:
            assert c4[k].shape == (36,) and np.array_equal(c4[k], np.asarray(c1[k])), k


# ---------------------------------------------------------------------------------------------------------------
# gradients

def test_rstar_gradient_finite_fd_dot():
    """d J / d etaH of CALC_R_STAR -> UPDATE_R_STAR(.TRUE.) (J weights every output on every lane,
    rstar_gate.objective):
    finite on every lane (halos, land) and non-zero on every wet interior point; FD (central, h = 1e-2 m) at six
    random wet points within 1e-7 relative (measured, global_ocean step 36000: max 1.7e-8 at h = 1e-2, 8.0e-8 at
    1e-3, 9.5e-7 at 1e-4: round-off of J ~ 141 grows as h shrinks, J is nearly linear in etaH); tangent (jvp) vs
    adjoint (vjp) dot test to 1e-13 (measured 2.6e-15)."""
    eta, grid, state = R.objective_inputs(*GO, 36000)
    w = R.weights(*GO, grid, state)
    J = R.objective(*GO, grid, state, w)
    x0 = eta.data
    g = np.asarray(jax.jit(jax.grad(J))(x0))
    assert np.all(np.isfinite(g))
    Jj = jax.jit(J)
    kS = np.asarray(grid.kSurfC.data)
    sz = R.experiment(*GO).cfg.size
    inner = (slice(None), slice(sz.OLy, sz.OLy+sz.sNy), slice(sz.OLx, sz.OLx+sz.sNx))
    assert np.all(g[inner][kS[inner] <= sz.Nr] != 0.)
    rng = np.random.default_rng(1)
    wet = np.argwhere(kS[:, sz.OLy:sz.OLy+sz.sNy, sz.OLx:sz.OLx+sz.sNx] <= sz.Nr)
    worst = 0.
    for t, jj, ii in wet[rng.choice(len(wet), 6, replace=False)]:
        idx = (t, jj + sz.OLy, ii + sz.OLx)
        h = 1e-2
        xp = x0.at[idx].add(h)
        xm = x0.at[idx].add(-h)
        fd = (float(Jj(xp)) - float(Jj(xm)))/(2*h)
        worst = max(worst, abs(fd - g[idx])/max(abs(g[idx]), 1e-300))
    assert worst < 1e-7, worst
    v = jnp.asarray(rng.standard_normal(x0.shape))
    tl = float(jax.jit(lambda x, v: jax.jvp(J, (x,), (v,))[1])(x0, v))
    ad = float(np.sum(g*np.asarray(v)))
    assert abs(tl - ad) <= 1e-13*max(abs(tl), abs(ad)), (tl, ad)


# ---------------------------------------------------------------------------------------------------------------
# negative controls (planted errors, each measured to bite on the data above)

class _NoExchange:
    """The exchanger with every halo exchange the identity (planted error); reductions unchanged."""

    def __init__(self, ex):
        self.ex = ex

    def EXCH_XY_RL(self, phi):
        return phi

    def EXCH_UV_XY_RL(self, u, v, withSigns):
        return u, v

    def all_tiles(self, per_tile):
        return self.ex.all_tiles(per_tile)


def test_negative_controls_bite():
    it = 36001
    # 1. CALC_R_STAR without its exchanges (:259-260): halo points of rStarFac*, rStarDh*, rStarExp*
    eta = R._xy(R.field(*GO, it, "S11_integr_continuity", "etaH"), "etaH", R.experiment(*GO).cfg.size)
    from mitjax.model.src.calc_r_star import calc_r_star
    e = R.experiment(*GO)
    out, _ = calc_r_star(eta, None, None, cfg=e.cfg, grid=R.teacher_grid(*GO, it, "S07_update_rstar_T"),
                         params=R.params(*GO), state=R.teacher_state(*GO, it, "S07_update_rstar_T"),
                         ex=_NoExchange(R.exchanger(GO[0])))
    bite = sum(R.compare(getattr(out, n), R.field(*GO, it, "S12_calc_rstar", n))[1] for n in R.R_FAC)
    assert bite > 100, bite
    # 2. etaH of the wrong stage (S06_dynamics: before SOLVE_FOR_PRESSURE / INTEGR_CONTINUITY)
    eta6 = R._xy(R.field(*GO, it, "S06_dynamics", "etaH"), "etaH", e.cfg.size)
    out, _ = R.run_calc_r_star(*GO, eta6, R.teacher_grid(*GO, it, "S07_update_rstar_T"),
                               R.teacher_state(*GO, it, "S07_update_rstar_T"))
    bite = sum(R.compare(getattr(out, n), R.field(*GO, it, "S12_calc_rstar", n))[1] for n in R.R_FAC)
    assert bite > 1000, bite
    # 3. UPDATE_R_STAR(.TRUE.) on the wrong time level (rStarFacNm1 instead of rStarFac)
    st = R.teacher_state(*GO, it, "S00_begin")
    st = st.replace(rStarFacC=st.rStarFacNm1C, rStarFacW=st.rStarFacNm1W, rStarFacS=st.rStarFacNm1S)
    g = R.run_update_r_star(*GO, True, R.teacher_grid(*GO, it, "S01_update_rstar_F"), st)
    bite = sum(R.compare(getattr(g, n), R.field(*GO, it, "S07_update_rstar_T", n))[1] for n in R.HFAC)
    assert bite > 1000, bite
    # 4. UPDATE_CG2D on the hFac of the wrong stage (S01: UPDATE_R_STAR(.FALSE.)'s, one step behind)
    def wrong(i):
        sz = e.cfg.size
        return (R._xyz(R.field(*GO, i, "S01_update_rstar_F", "hFacW"), "hFacW", sz),
                R._xyz(R.field(*GO, i, "S01_update_rstar_F", "hFacS"), "hFacS", sz))
    bites = [sum(v[1] for v in _compare_s08(c, i).values()) for i, c in _cg2d_chain(wrong)]
    assert all(b > 100 for b in bites), bites
    # 5. UPDATE_SURF_DR reading the wrong thickness field (h0FacW at C points): advect_xz/input
    v = ("advect_xz", "input")
    grid = R.teacher_grid(*v, 0, "S00_begin")
    out = R.run_update_surf_dr(*v, grid.replace(h0FacC=FArray(grid.h0FacW.data, "h0FacC", _dims=grid.h0FacC.dims)))
    bite = sum(R.compare(getattr(out, n), R.field(*v, 1, "S00_begin", n))[1] for n in ("hFacC", "recip_hFacC"))
    assert bite > 100, bite


def test_unported_options_raise():
    """Options no M1 variant uses raise at trace time."""
    from mitjax.model.src.calc_r_star import calc_r_star
    from mitjax.model.src.reset_nlfs_vars import reset_nlfs_vars
    from mitjax.model.src.update_surf_dr import update_surf_dr
    e = R.experiment(*GO)
    st = R.teacher_state(*GO, 36000, "S00_begin")
    air = R.params(*GO).replace(static=dict(fluidIsAir=True))
    with pytest.raises(NotImplementedError, match="fluidIsAir"):
        calc_r_star(None, None, None, cfg=e.cfg, grid=None, params=air, state=st, ex=None)
    with pytest.raises(NotImplementedError, match="fluidIsAir"):
        reset_nlfs_vars(None, None, cfg=e.cfg, params=air, state=st)
    # UPDATE_SURF_DR's nonlinFreeSurf > 0 arms are ported (lane B session 9, gated by adjustment.cs input.nlfs and the
    # GOADK input_ad family); without the State they refuse instead of reading nothing
    with pytest.raises(ValueError, match="needs the State"):
        update_surf_dr(True, None, None, cfg=e.cfg, grid=None, params=R.params(*GO))
    with pytest.raises(RuntimeError, match="ABNORMAL END: S/R CALC_R_STAR"):
        c = {k: np.zeros(36, np.int32) for k in ("icntc1", "icntw", "icnts", "icntc2")}
        c["maxhFacC"] = np.zeros(36)
        c["icntw"][5] = 3
        calc_r_star_host(c, 7, 0, cfg=e.cfg)


# ---------------------------------------------------------------------------------------------------------------
# counters (the dumped runs never trip them: all zero above) on a synthetic etaH

class _ReversedTiles:
    """The exchanger with the per-tile gather in reversed tile order (planted error)."""

    def __init__(self, ex):
        self.ex = ex

    def EXCH_XY_RL(self, phi):
        return self.ex.EXCH_XY_RL(phi)

    def EXCH_UV_XY_RL(self, u, v, withSigns):
        return self.ex.EXCH_UV_XY_RL(u, v, withSigns)

    def all_tiles(self, per_tile):
        return self.ex.all_tiles(per_tile)[::-1]


def _synthetic_counts(ex=None):
    """etaH of step 36000 raised by 1.5 column depths at the first two wet interior points of tile A and the last of
    tile B (rStarFacC > hFacSup = 2 there), lowered by 0.9 column depths at the first wet interior point of tile C
    (rStarFacC < hFacInf = 0.2), A < B < C the first ocean tiles from tile 4, 20, 30 on; returns (counters, expected
    per-tile counts from the formula :103-105 in numpy on the same inputs, (A, B, C) 0-based)."""
    sz = R.experiment(*GO).cfg.size
    it = 36000
    grid = R.teacher_grid(*GO, it, "S07_update_rstar_T")
    state = R.teacher_state(*GO, it, "S07_update_rstar_T")
    eta = np.array(R.field(*GO, it, "S11_integr_continuity", "etaH")[:, 0])
    H = np.asarray(grid.Ro_surf.data) - np.asarray(grid.R_low.data)
    kS = np.asarray(grid.kSurfC.data)
    def wet(t):
        return [(jj, ii) for jj in range(sz.OLy, sz.OLy+sz.sNy) for ii in range(sz.OLx, sz.OLx+sz.sNx)
                if kS[t, jj, ii] <= sz.Nr]
    A, B, C = (next(t for t in range(t0, 36) if len(wet(t)) >= 2) for t0 in (3, 19, 29))
    pts = [(A, wet(A)[0], 1.5), (A, wet(A)[1], 1.5), (B, wet(B)[-1], 1.5), (C, wet(C)[0], -0.9)]
    for t, (jj, ii), f in pts:
        eta[t, jj, ii] += f*H[t, jj, ii]
    out, c = R.run_calc_r_star(*GO, R._xy(eta[:, None], "etaH", sz), grid, state, ex=ex)
    sl = (slice(None), slice(sz.OLy, sz.OLy+sz.sNy+1), slice(sz.OLx, sz.OLx+sz.sNx+1))     # j, i = 1..sN+1
    rC = np.where(kS[sl] <= sz.Nr, (eta[sl] + np.asarray(grid.Ro_surf.data)[sl] - np.asarray(grid.R_low.data)[sl])
                  * np.asarray(grid.recip_Rcol.data)[sl], 1.)
    p = R.params(*GO)
    want = dict(icntc1=(rC < float(p.hFacInf)).sum(axis=(1, 2)), icntc2=(rC > float(p.hFacSup)).sum(axis=(1, 2)),
                maxhFacC=np.where(rC > float(p.hFacSup), rC, 0.).max(axis=(1, 2)))
    return jax.tree.map(np.asarray, c), want, (A, B, C)


def test_counters_synthetic_and_host_stop():
    """Per-tile counters in tile order (numpy recount of the formula on the same inputs: a consistency check, not a
    Fortran oracle; the Fortran oracle of the counters is a pending replay), the host's
    warnings / STOP with the tile's bi,bj; planted error: the gather in reversed tile order bites."""
    c, want, (A, B, C) = _synthetic_counts()
    nSx = R.experiment(*GO).cfg.size.nSx
    bij = {t: f"{t % nSx + 1:4d}{t // nSx + 1:4d}" for t in (A, B, C)}
    for k in ("icntc1", "icntc2"):
        assert np.array_equal(c[k], want[k]), (k, c[k], want[k])
    assert np.array_equal(c["maxhFacC"], want["maxhFacC"])
    assert c["icntc2"][A] == 2 and c["icntc2"][B] >= 1 and c["icntc1"][C] >= 1 and A < B < C
    cfg = R.experiment(*GO).cfg
    with pytest.raises(RuntimeError) as err:
        calc_r_star_host(c, 36001, 0, cfg=cfg)
    msg = str(err.value)
    assert f"WARNING: r*FacC < hFacInf at{int(c['icntc1'][C]):8d} pts : bi,bj,Thid,Iter={bij[C]}   1     36001" \
        in msg, msg
    assert msg.index(f"{bij[A]}   1") < msg.index(f"{bij[B]}   1") < msg.index(f"{bij[C]}   1")
    no_stop = dict(c, icntc1=np.zeros_like(c["icntc1"]), icntw=np.zeros_like(c["icntw"]),
                   icnts=np.zeros_like(c["icnts"]))
    lines, nw = calc_r_star_host(no_stop, 36001, 0, cfg=cfg)
    assert nw == 2 and len(lines) == 4, lines
    assert lines[0] == f"WARNING: r*FacC > hFacSup at       2 pts : bi,bj,Thid,Iter={bij[A]}   1     36001"
    assert lines[2].endswith(f"{bij[B]}   1     36001")
    m4, m20 = float(c["maxhFacC"][A]), float(c["maxhFacC"][B])
    assert lines[1] == f"WARNING: max(hFacC) is {R_e14(m4)}"
    assert lines[3] == f"WARNING: max(hFacC) is {R_e14(max(m4, m20))}"           # the running MAX over tiles
    cr, _, _ = _synthetic_counts(ex=_ReversedTiles(R.exchanger(GO[0])))
    assert not np.array_equal(cr["icntc2"], want["icntc2"])                      # the planted order bites


def R_e14(x):
    from mitjax.model.src.calc_r_star import _e14_6
    return _e14_6(x)


# ---------------------------------------------------------------------------------------------------------------
# the RSTAR replay harness: synthetic inputs on the real global_ocean grid (reference/replay_rstar)

def _replay_run():
    runs = R.replay_runs()
    assert len(runs) == 1 and runs[0][:2] == GO, runs
    return runs[0]


def test_replay_harness_bitwise_and_stderr():
    """The harness's CALC_R_STAR (A: area weighted; B: the simple average, vectorInvariantMomentum with
    selectKEscheme = 1), UPDATE_R_STAR(.TRUE.) and (.FALSE.) (recip_hFacW/S included), RESET_NLFS_VARS: every output
    bitwise on every point; columns pushed above hFacSup in several tiles (also at the halo column/row sNx+1, sNy+1
    the counters read before the exchange): calc_r_star_host's lines == the Fortran's lines on STDERR.0000, text for
    text (counts, bi,bj, iteration, the running max(hFacC) in E14.6, numbWrite carried across the two calls)."""
    res, ours, ref, (cA, cB) = R.replay_gate(*_replay_run())
    _assert_bitwise(res, "replay")
    assert len(res) == 13 + 6 + 6 + 13 + 1
    assert np.count_nonzero(np.asarray(cA["icntc2"])) >= 5 and np.count_nonzero(np.asarray(cB["icntc2"])) >= 5
    assert len(ref) >= 20 and ours == ref, (ours[:6], ref[:6])


def test_replay_negative_controls_bite():
    """Planted: the counters gathered in reversed tile order (the STDERR lines differ); call B with the area
    weighting (rStarFacW/S of B differ)."""
    exp, inp, rundir = _replay_run()
    _, ours, ref, _ = R.replay_gate(exp, inp, rundir, ex=_ReversedTiles(R.exchanger(exp)))
    assert ours != ref
    res, _, _, _ = R.replay_gate(exp, inp, rundir, planted_area_weight=True)
    bite = res["B_rStarFacW"][1] + res["B_rStarFacS"][1]
    assert bite > 1000, bite
