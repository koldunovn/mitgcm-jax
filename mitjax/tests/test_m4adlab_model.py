"""lab_sea/input_ad (M4 step 7 part 2, lane M4ADLAB session 3): model/src and eesupp arms of the code_ad build, gated
bitwise against lane A's dumps-on run job27855987-jdon (code_ad forward), iterations 0-2, teacher-forced.

1. CG2D with CG2D_SINGLECPU_SUM (verification/lab_sea/code_ad/CPP_EEOPTIONS.h:129): the residual and the CG scalars
   eta_qrN, alpha, err_sq as GLOBAL_SUM_SINGLECPU_RL of the products over the global array in global order
   (model/src/cg2d.F:170-186, :226-242, :282-296, :312-328; eesupp/src/global_sum_singlecpu.F:142-146), replayed from
   C01_cg2d_inputs: C02's cg2d_x on every point and numIters, nIterMin, firstResidual, minResidualSq, lastResidual
   bitwise. Control: CG2D_SINGLECPU_SUM planted off (tile partials + GLOBAL_SUM_TILE_RL) changes the solution.
2. pkg/down_slope (G1): DWNSLP_INIT_FIXED's site tables and Gamma, as the oracle's down_slope.0000.log prints them
   (all 214 lines, Gamma in 1PE14.6) and its STDOUT `DWNSLP_INIT: DWNSLP_NbSite=` lines. Control: DWNSLP_drFlow
   planted 29. (Gamma lines differ).
3. The whole FORWARD_STEP of input_ad (session 4: the real Model, no stub left; sessions 3: ecco / ctrl / cost planted
   off),
   steps 0-2 chained FREE from the Model's own cold-start carry, every stage the probes hold compared on every point
   (53 stages per iteration: EXF, SEAICE_MODEL, KPP, GMRedi, DOWN_SLOPE's CALC_RHO / CALC_FLOW / APPLY, dynamics with
   CG2D_SINGLECPU_SUM, the multi-dimensional DST3 advection with the vertical GAD_DST3_ADV_R, staggerTimeStep, the
   monitor's state). Not teacher-forced: the CD scheme's carried velocities (useCDscheme) are not dumped at S00_begin,
   so a teacher carry at iterations >= 1 is not the oracle's state (measured: dev job 27894429 differs from D01 on,
   the free chain 27894754 is bitwise). Controls at iteration 0: DWNSLP_APPLY planted to the identity (T11_temp_gT
   and T21_salt_gS differ), DWNSLP_CALC_RHO planted to the plain level k (rhoInSitu below the bottom differs).
   Session 4: S03_ctrl_map_forcing and S18_cost_tile are compared too (ctrl / cost on). Blind spot: C01/C02 (cg2d:
   item 1).
"""

import dataclasses

import pytest

from mitjax.tests import cg2d_gate as C

EXP = ("lab_sea", "input_ad")

pytestmark = pytest.mark.filterwarnings("ignore")


def _bad(res):
    return {k: v for k, v in res.items() if v[1] or v[2] or v[3]}


def test_cg2d_singlecpu_sum():
    from mitjax.model.src import cg2d as cg2d_mod
    from mitjax.tests.m4col_gate import CppPlant
    for it in (0, 1, 2):
        out = C.replay_cg2d(*EXP, it)
        res = C.compare_solution(*EXP, it, out)
        assert set(res) == {"cg2d_x", *C.SCALARS} and _bad(res) == {}, (it, _bad(res))
    cfg = C.experiment(*EXP).cfg
    planted = dataclasses.replace(cfg, cpp=CppPlant(cfg.cpp, off=("CG2D_SINGLECPU_SUM",)))

    def tile_sums(*a, cfg, **kw):
        return cg2d_mod.cg2d(*a, cfg=planted, **kw)
    bad = {it: _bad(C.compare_solution(*EXP, it, C.replay_cg2d(*EXP, it, cg2d_fn=tile_sums))) for it in (0, 1)}
    assert all("cg2d_x" in b for b in bad.values()), bad


def test_down_slope_init_log_and_stdout():
    from mitjax import paths
    from mitjax.pkg.down_slope.dwnslp_init_fixed import dwnslp_init_fixed
    from mitjax.pkg.down_slope.dwnslp_readparms import dwnslp_readparms
    from mitjax.tests import m4adlab_gate as A
    m, _ = A.stub_model()
    rd = paths.REFERENCE_RUNS / A.EXP[0] / A.EXP[1] / A.JDON / "rundir"
    want = (rd / "down_slope.0000.log").read_text().splitlines()
    assert len(want) == 214 and m.dwnslp_log == want
    so = [ln.split(") ", 1)[1] for ln in (rd / "output.txt").read_text().splitlines()
          if "DWNSLP_INIT: DWNSLP_NbSite=" in ln]
    assert len(so) == 4 and m.dwnslp_stdout == so, (m.dwnslp_stdout, so)
    dp = dwnslp_readparms(m.exp, tempStepping=True, saltStepping=True, debugLevel=int(m.params.debugLevel))
    assert (dp.DWNSLP_slope, dp.DWNSLP_rec_mu, dp.DWNSLP_drFlow) == (5.e-3, 1.e+4, 30.)
    planted = dwnslp_init_fixed(dataclasses.replace(dp, DWNSLP_drFlow=29.), cfg=m.cfg, grid=m.grid,
                                usingPCoords=False)["log"]
    assert sum(a != b for a, b in zip(planted, want)) > 0


def test_free_steps_bitwise(monkeypatch):
    import jax
    from mitjax.tests import m4adlab_gate as A
    m, ds = A.stub_model()
    res = A.free_steps(m, ds, its=(0, 1, 2))
    for it, (stages, bad) in res.items():
        assert len(stages) >= 55 and {"P02_rho_sigma_ivdc", "P05_gmredi_tensor", "S09_solve_for_pressure",
                                      "T11_temp_gT", "T21_salt_gS", "S17_monitor", "I04_growth",
                                      "S03_ctrl_map_forcing", "S18_cost_tile"} <= set(stages), it
        assert bad == {}, (it, bad)
    jax.clear_caches()
    import mitjax.pkg.down_slope.dwnslp_apply as DA
    monkeypatch.setattr(DA, "dwnslp_apply", lambda tracer, g, *a, **k: g)
    bad = A.free_steps(m, ds, its=(0,))[0][1]
    assert ("T11_temp_gT", "gT_loc") in bad and ("T21_salt_gS", "gS_loc") in bad, bad
    monkeypatch.undo()
    jax.clear_caches()
    import mitjax.pkg.down_slope.dwnslp_calc_rho as DR
    from mitjax.model.src.find_rho import find_rho_2d
    from mitjax.model.src.do_oceanic_phys import _level
    sz = m.cfg.size

    def plain(tFld, sFld, rhoLoc, k, *, cfg, grid, params, eos, state):
        return find_rho_2d(1-sz.OLx, sz.sNx+sz.OLx, 1-sz.OLy, sz.sNy+sz.OLy, k, _level(tFld, k), _level(sFld, k),
                           rhoLoc, k, cfg=cfg, grid=grid, params=params, eos=eos, state=state)
    monkeypatch.setattr(DR, "dwnslp_calc_rho", plain)
    bad = A.free_steps(m, ds, its=(0,))[0][1]
    assert ("P02_rho_sigma_ivdc", "rhoInSitu") in bad, bad
