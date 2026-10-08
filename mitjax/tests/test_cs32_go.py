"""global_ocean.cs32x15/input (plan Task 25, M2), GO lane session 8: the step of the run driver's Model on the cube
(nIter0 = 72000; r*, staggerTimeStep, GM_AdvForm with the 'gkw91' taper, cg2dTargetResWunit, and ALLOW_NONHYDROSTATIC /
ALLOW_ADDFLUID / SHORTWAVE_HEATING / ALLOW_BALANCE_FLUXES / ALLOW_BALANCE_RELAX / ALLOW_SEAICE compiled with their
switches off) vs lane A's dumps of steps 72000-72002, every point of every tile, bit patterns (helpers:
mitjax/tests/cs32_gate.py).

* the front of step 72000 up to S04_oceanic_phys: S00, S02, P01 EXTERNAL_FORCING_SURF (the balance and relaxation
  switches off, FORCING_SURF_RELAX's r* rescaling), P02, P03, P05 GMREDI_CALC_TENSOR (GMREDI_SLOPE_PSI 'gkw91'), P06
  GMREDI_DO_EXCH, S04; control: FORCING_SURF_RELAX without its staggerTimeStep r* rescaling -> P01 differs;
* CG2D without RHS normalisation (cg2dNormaliseRHS = .FALSE.) on the dumped C01 inputs of the 3 iterations, literal and
  through the implicit rule: C02's cg2d_x and scalars bit for bit; control: with the normalisation -> differs;
* SOLVE_FOR_PRESSURE (CALC_DIV_GHAT with NH / ADDFLUID compiled and off) on a teacher State: S09 etaN bitwise;
* steps 72000-72002 from the initial carry, no teacher forcing: every dumped stage bitwise (30 stages incl. the
  per-level D00a / D00c, the staggered THERMODYNAMICS T01-T23 with the penetrating short-wave); the whole run and
  P=6 == P=1: test_cs32_go_run.py.
Costs: about 4 min on a CPU compute node.
"""

import dataclasses

import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

from mitjax.tests import cs32_gate as S  # noqa: E402


@pytest.fixture(scope="module")
def r():
    return S.cs32_run()


def test_front_to_s04_bitwise(r, monkeypatch):
    out, ncmp, _ = S.run_compare(r, 1, until="S04_oceanic_phys")
    print("cs32x15 front stages compared:", dict(sorted(ncmp.items())))
    need = {"S00_begin", "S02_load_fields", "P01_external_forcing_surf", "P02_rho_sigma_ivdc", "P03_mxlayer",
            "P05_gmredi_tensor", "P06_gmredi_exch", "S04_oceanic_phys"}
    assert {st for (_, st) in ncmp} >= need
    assert {k: v for k, v in out.items() if v} == {}
    import mitjax.model.src.external_forcing_surf as EFS
    orig = EFS.forcing_surf_relax
    monkeypatch.setattr(EFS, "forcing_surf_relax", lambda *a, fp, **k: orig(
        *a, fp=dataclasses.replace(fp, staggerTimeStep=False), **k))
    out, _, _ = S.run_compare(r, 1, until="P01_external_forcing_surf")
    assert {"surfaceForcingT", "surfaceForcingS"} & set(out[(r.its[0], "P01_external_forcing_surf")])


def test_cg2d_without_rhs_normalisation(r, monkeypatch):
    assert r.m.cg2dh.cg2dNormaliseRHS is False
    for it in r.its:
        for solve in ("literal", "rule"):
            res = S.replay_cg2d(r, it, solve)
            assert all(v[1] == 0 and v[2] == 0 for v in res.values()), (it, solve, res)
    m = r.m
    monkeypatch.setattr(m, "cg2dh", m.cg2dh.replace(cg2dNormaliseRHS=True))
    res = S.replay_cg2d(r, r.its[0], "literal")
    assert res["cg2d_x"][1] > 0, res


def test_solve_for_pressure_teacher(r):
    for it in r.its:
        res, diag = S.replay_solve_for_pressure(r, it)
        assert res["etaN"][0] > 0 and not any(res["etaN"][1:]), (it, res)


def test_steps_bitwise(r):
    """Steps 72000-72002 from the Model's initial carry with no teacher forcing (DYNAMICS with MOM_VECINV on the
    cube: lane B's wiring): every dumped stage bitwise, D00a_phi_hyd / D00c_mom_vecinv per level included."""
    out, ncmp, _ = S.run_compare(r, 3)
    print("cs32x15 stages compared:", len(ncmp), "fields:", sum(ncmp.values()))
    for it in r.its:
        got = {st for (i, st) in ncmp if i == it}
        assert {"S04_oceanic_phys", "D00c_mom_vecinv", "S06_dynamics", "S09_solve_for_pressure", "S12_calc_rstar",
                "S14_thermodynamics_stagger", "T13_temp_impl", "T23_salt_impl", "S16_blocking_exchanges"} <= got, (
            it, sorted(got))
    assert len(ncmp) >= 90 and {k: v for k, v in out.items() if v} == {}
