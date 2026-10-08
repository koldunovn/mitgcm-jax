"""Gates of pkg/ctrl, pkg/cost and pkg/grdchk on tutorial_global_oce_optim/input_ad (lane ctrl, plan Task 16 prep).

Oracle: mitjax/tests/ctrl_cost_gate.py (runs ZERO = testreport's variant, XX = planted nonzero xx_qnet, FD = the
grdchk run). Bitwise = equal bit patterns at every point of every tile, halos included. All tier1x.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax.tests import ctrl_cost_gate as G

KINDS = ("zero", "xx")


def _sz():
    return G.experiment().cfg.size


# ------------------------------------------------------------------------------------------------- controls

@pytest.mark.parametrize("kind", KINDS)
def test_ctrl_effective_record_bitwise(kind):
    """CTRL_MAP_INI_GENTIM2D: our effective record == the oracle's xx_qnet.effective.0000000000 file (interior)."""
    sz = _sz()
    eff, _, _ = G.ctrl_init(kind)
    oeff = G.read_tiled_xy(G.run_top(kind), "xx_qnet.effective.0000000000", sz)
    inner = (slice(None), slice(sz.OLy, sz.OLy + sz.sNy), slice(sz.OLx, sz.OLx + sz.sNx))
    assert G.bits_equal(np.asarray(eff[1][0].data)[inner], oeff[inner])
    if kind == "xx":
        assert np.count_nonzero(oeff[inner]) == 2315          # every wet surface point carries the planted control


@pytest.mark.parametrize("kind", KINDS)
def test_ctrl_map_forcing_bitwise_all_steps(kind):
    """S02 forcing + control -> our CTRL_MAP_GENTIM2D + CTRL_MAP_FORCING == the dumped S03 forcing, every point incl.
    halos, all 10 steps; with the planted control the step changes Qnet at 2957 points (so the gate sees it)."""
    eff, _, genarr = G.ctrl_init(kind)
    for it in range(10):
        out, genarr = G.ctrl_step(kind, it, eff, genarr)
        for n in G.FORCING:
            assert G.n_bits_differ(out[n].data, G.field2(kind, it, "S03_ctrl_map_forcing", n)) == 0, (it, n)
    changed = G.n_bits_differ(G.field2(kind, 9, "S02_load_fields", "Qnet"), G.field2(kind, 9, "S03_ctrl_map_forcing",
                                                                                    "Qnet"))
    assert changed == (2957 if kind == "xx" else 0)


def test_ctrl_negative_controls_bite():
    """Planted errors measured to bite: (a) the zero control for the planted run (2957 points of Qnet), (b) one
    control value changed by 1e-6 W/m^2 at the first grdchk point (measured: that point of Qnet and its periodic
    halo copy, 2 points; a 1-ulp change of the control does NOT bite here: it is absorbed by the rounding of
    Qnet + xx, |Qnet| ~ 2e2 vs |xx| <= 50), (c) the scalar exchanges skipped (the halo points that differ)."""
    sz = _sz()
    # (a)
    eff, _, genarr = G.ctrl_init("xx", xx_in=G.control_input("zero")[0])
    out, _ = G.ctrl_step("xx", 0, eff, genarr)
    assert G.n_bits_differ(out["Qnet"].data, G.field2("xx", 0, "S03_ctrl_map_forcing", "Qnet")) == 2957
    # (b)
    xx, _ = G.control_input("xx")
    t, j, i = 0, 2 - 1 + sz.OLy, 43 - 1 + sz.OLx
    bumped = xx[1][0].data.at[t, j, i].add(1e-6)
    eff, _, genarr = G.ctrl_init("xx", xx_in={1: [G.fa2(bumped, "xx", sz)]})
    out, _ = G.ctrl_step("xx", 0, eff, genarr)
    assert G.n_bits_differ(out["Qnet"].data, G.field2("xx", 0, "S03_ctrl_map_forcing", "Qnet")) == 2
    # (c)
    from mitjax.pkg.ctrl import ctrl_map_forcing as cmf
    eff, _, genarr = G.ctrl_init("xx")

    class NoXY:
        def __init__(self, ex):
            self.ex = ex

        def EXCH_XY_RL(self, a):
            return a

        def EXCH_UV_XY_RL(self, u, v, s):
            return self.ex.EXCH_UV_XY_RL(u, v, s)
    real = G.exchanger
    try:
        G.exchanger = lambda: NoXY(real())
        out, _ = G.ctrl_step("xx", 0, eff, genarr)
    finally:
        G.exchanger = real
    del cmf
    n = G.n_bits_differ(out["Qnet"].data, G.field2("xx", 0, "S03_ctrl_map_forcing", "Qnet"))
    assert n > 0, n


# ------------------------------------------------------------------------------------------------- cost

@pytest.mark.parametrize("kind", KINDS)
def test_cost_bitwise_per_step_and_final(kind):
    """fc, glofc after COST_TILE at every step (S18) and the per-tile objf_temp_tut, objf_hflux_tut, tile_fc and fc
    after COST_FINAL (E01) bitwise; STDOUT lines of COST_FINAL identical to the oracle's."""
    eff, _, genarr = G.ctrl_init(kind)
    _, genarr = G.ctrl_step(kind, 0, eff, genarr)

    def check(it, cost):
        for n in ("fc", "glofc"):
            assert G.bits_equal(np.asarray(getattr(cost, n)), G.dumps(kind).scalar(it, "S18_cost_tile", n)), (it, n)
    cost, loc, early = G.cost_run(kind, genarr["xx_gentim2d"], per_step=check)
    for n in ("objf_temp_tut", "objf_hflux_tut", "tile_fc"):
        assert G.n_bits_differ(np.asarray(getattr(cost, n)), G.dumped_tiles(kind, 9, "E01_cost_final", n)) == 0, n
    assert G.bits_equal(np.asarray(cost.fc), G.dumps(kind).scalar(9, "E01_cost_final", "fc"))
    from mitjax.pkg.cost.cost_final import cost_final_lines
    ours = cost_final_lines(cost, loc, cfg=G.experiment().cfg, params=G.cost_params(), early_fc=early)
    lines = [ln.text for ln in G.printed(kind)[0]]
    start = next(k for k, t in enumerate(lines) if "early fc =" in t)
    theirs = [lines[start]] + lines[start + 1:start + 9] + [lines[start + 9]] + \
        [next(t for t in lines[start:] if "global fc =" in t)]
    strip = lambda t: t.split(") ", 1)[1] if t.startswith("(PID.TID") else t     # noqa: E731
    assert [strip(t) for t in theirs] == ours
    if kind == "zero":
        assert ours[-1] == " global fc =   6.20023228182337E+00"


def test_cost_negative_control_bites():
    """A planted error in one cost weight (wti(1) of Err_levitus_15layer.bin times (1 + 2**-52)) changes fc."""
    eff, _, genarr = G.ctrl_init("zero")
    _, genarr = G.ctrl_step("zero", 0, eff, genarr)
    tw = G.cost_inputs("zero")[0].copy()
    tw[0] = np.nextafter(tw[0], np.inf)
    cost, _, _ = G.cost_run("zero", genarr["xx_gentim2d"], tmpwti=tw)
    assert not G.bits_equal(np.asarray(cost.fc), G.dumps("zero").scalar(9, "E01_cost_final", "fc"))
    assert G.n_bits_differ(np.asarray(cost.objf_temp_tut), G.dumped_tiles("zero", 9, "E01_cost_final",
                                                                         "objf_temp_tut")) == 4


def test_cost_gradient_wrt_control_finite_and_fd():
    """d fc / d xx through CTRL_MAP_INI_GENTIM2D -> CTRL_MAP_GENTIM2D -> COST_FINAL on the dumped state (no time
    stepping: the hflux term is the only one that sees the control): finite at every lane incl. halos, land and
    padding-free tiles; equal to a central FD at the first grdchk point and to the closed form
    2*mult_hflux_tut*tmpC*whfluxm*xx."""
    sz = _sz()
    kind = "xx"
    xx0 = G.control_input(kind)[0][1][0].data
    from mitjax.pkg.cost.cost_final import cost_final
    from mitjax.pkg.cost.cost_init_varia import cost_init_varia
    from mitjax.pkg.cost.cost_weights import cost_weights
    cfg, g = G.experiment().cfg, G.grid(kind)
    tw, errh, thetalev = G.cost_inputs(kind)
    whfluxm, wtheta = cost_weights(cfg=cfg, tmpwti=tw, Err_hflux=errh, ex=G.exchanger(), xy=lambda d, n: G.fa2(d, n, sz))

    def fc_of(xx):
        eff, _, genarr = G.ctrl_init(kind, xx_in={1: [G.fa2(xx, "xx", sz)]})
        _, genarr = G.ctrl_step(kind, 0, eff, genarr)
        cost = cost_init_varia(cfg=cfg, ntiles=4)
        cost.cMeanTheta = G.fa3(G.dumps(kind).field(0, "S17_monitor", "theta"), "cMeanTheta", sz, sz.Nr)
        cost, _ = cost_final(cost, cfg=cfg, params=G.cost_params(), maskC=g["maskC"], wtheta=wtheta,
                             thetalev=thetalev, whfluxm=whfluxm, xx_gentim2d=genarr["xx_gentim2d"])
        return cost.fc

    grad = np.asarray(jax.grad(fc_of)(xx0))
    assert np.all(np.isfinite(grad))
    t, j, i = 0, 2 - 1 + sz.OLy, 43 - 1 + sz.OLx
    best = np.inf
    for h in (1e-1, 1e-2, 1e-3):
        fp = float(fc_of(xx0.at[t, j, i].add(h)))
        fm = float(fc_of(xx0.at[t, j, i].add(-h)))
        best = min(best, abs((fp - fm) / (2 * h) - grad[t, j, i]) / abs(grad[t, j, i]))
    assert best < 1e-7, best
    tmpC = 1.0 / 2315.0
    closed = 2 * 2.0 * tmpC * float(whfluxm.data[t, j, i]) * float(xx0[t, j, i])
    assert abs(grad[t, j, i] - closed) <= 1e-12 * abs(closed)
    # land and halo lanes: the control there never reaches the cost
    land = np.asarray(g["maskC"].data[:, 0]) == 0
    assert np.all(grad[land] == 0.0)


# ------------------------------------------------------------------------------------------------- grdchk

def _grdchk_kw():
    from mitjax.pkg.grdchk import grdchk as gc
    sz = _sz()
    mC = G.dumps("zero").field(0, "G00_geometry", "maskC")
    nw = gc.ctrl_init_wet_nwetctile(mC, sz)
    gm = gc.grdchk_get_mask(nw, ncvargrd="c", ncvarnrmax=1, ncvarxmax=sz.sNx, ncvarymax=sz.sNy, ncvarrecs=1)
    return nw, gm, dict(gm=gm, maskC=mC, sz=sz, iLocTile=1, jLocTile=1, ncvargrd="c", ncvarrecs=1, ncvarnrmax=1,
                        ncvarxmax=sz.sNx, ncvarymax=sz.sNy)


def test_grdchk_positions_match_oracle():
    """The 3 points of data.grdchk (nbeg=1, nstep=1, nend=3) -> (i, j, k, bi, bj, iobc, rec) as the oracle prints
    them ("grdchk pos:" lines of the FD run), the packed component index icomp = 1, 2, 3, ncvarcomp = 2315 (the
    oracle's "ph-test icomp, ncvarcomp" line) and the wet counts of CTRL_SUMMARY; the storage index of each point."""
    from mitjax.io import stdout as so
    from mitjax.pkg.grdchk import grdchk as gc
    sz = _sz()
    nw, gm, kw = _grdchk_kw()
    assert gm.ncvarcomp == 2315
    lines = G.printed("fd")[0]
    assert any(ln.text.split() == ["ph-test", "icomp,", "ncvarcomp,", "ichknum", "1", "2315", "1"] for ln in lines)
    summ = [ln.text for ln in lines if "bi,bj,#(c/s/w):" in ln.text]
    counts = []
    for t in summ:
        f = t.split()
        counts.append(int(f[f.index("bi,bj,#(c/s/w):") + 3]))
    assert counts == [int(x) for x in nw.sum(axis=1)]
    pts = gc.grdchk_points(nbeg=1, nstep=1, nend=3, **kw)
    oracle = so.grdchk(lines).points
    ours = [(r.itilepos, r.jtilepos, r.layer, r.itile, r.jtile, r.obcspos, r.icvrec) for _, r in pts]
    assert ours == [(p.i, p.j, p.k, p.bi, p.bj, p.iobc, p.rec) for p in oracle]
    assert [ic for ic, _ in pts] == [1, 2, 3]
    assert [gc.storage_index(r, sz) for _, r in pts] == [(0, 3, 44), (0, 3, 45), (0, 3, 46)]
    assert all(r.ierr == 0 for _, r in pts)


def test_grdchk_negative_control_bites():
    """A planted off-by-one (counting from nbeg = 2, or a wet mask shifted by one point) gives other positions."""
    from mitjax.pkg.grdchk import grdchk as gc
    nw, gm, kw = _grdchk_kw()
    good = [(r.itilepos, r.jtilepos) for _, r in gc.grdchk_points(nbeg=1, nstep=1, nend=3, **kw)]
    off = [(r.itilepos, r.jtilepos) for _, r in gc.grdchk_points(nbeg=2, nstep=1, nend=4, **kw)]
    assert off != good
    kw2 = dict(kw, maskC=np.roll(kw["maskC"], 1, axis=-1))
    shifted = [(r.itilepos, r.jtilepos) for _, r in gc.grdchk_points(nbeg=1, nstep=1, nend=3, **kw2)]
    assert shifted != good


def test_grdchk_fd_formula_reproduces_oracle_fd():
    """gfd from the oracle's printed perturbed costs and grdchk_eps = 0.1 with grdchk_epsfac = 2 (central) equals
    the oracle's printed `ADM finite-diff_grad` within the rounding of the printout: the costs are printed with 15
    significant digits (1PE22.14), so each carries up to 0.5e-14 relative error, which the difference propagates as
    (|fp| + |fm|)*0.5e-14/(2*eps); the printed gfd adds its own 0.5e-14 relative."""
    from mitjax.io import stdout as so
    from mitjax.pkg.grdchk import grdchk as gc
    eps = G.experiment().params["data.grdchk:grdchk_nml:grdchk_eps"]
    for p in so.grdchk(G.printed("fd")[0]).points:
        gfd = gc.fd_gradient(p.fcpertplus, p.fcpertminus, float(eps), gc.grdchk_epsfac(True))
        bound = (abs(p.fcpertplus) + abs(p.fcpertminus)) * 0.5e-14 / (2 * float(eps)) \
            + 0.5e-14 * abs(p.adm["finite-diff_grad"])
        assert abs(gfd - p.adm["finite-diff_grad"]) <= bound, (gfd, p.adm["finite-diff_grad"], bound)
