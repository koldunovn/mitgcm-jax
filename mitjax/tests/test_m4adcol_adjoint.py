"""Lane M4ADCOL (M4 step 7, session 3): the adjoint of 1D_ocean_ice_column/input_ad vs TAF, through the driver Model
(useECCO as data.pkg sets it) and drivers/adjoint_run.GenarrAdjoint on the grdchk control xx_theta (genarr3d iarr 1,
weight ones_64b.bin): fc(xx) = COST_FINAL(THE_MAIN_LOOP(CTRL_MAP_INI_GENARR(xx))), jax.value_and_grad through the
per-step-checkpointed scan. TAF's adjoint mode equals the forward for this run (data.autodiff sets only dumpAdByRec;
pkg/autodiff/autodiff_readparms.adjoint_mode_check, test_m4adcol_setup.py), and the sea-ice build has no dynamics
(no LSR solve to differentiate).

Gated (testreport's criterion, MATCH_CRIT = 10 digits, verification/testreport:1154): `ADM adjoint_gradient` (admGrd)
at the four grdchk points (xx_theta (1,1,k), k = 1..4) >= 10 digits vs TAF's results/output_adm.txt (measured 12-14,
dev job 27880879), computed and as printed by adm_lines; `ADM ref_cost_function` in every printed digit; the gradient
finite everywhere and zero off the wet interior; a repeat bitwise. Negative control: the control weight planted to
1 + 1e-7 (CTRL_MAP_INI_GENARR divides by its square root) fails the 10-digit gate at every point.
Costs: about 10 min on a CPU node (the gradient program compiles in ~7 min).
"""

import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

from mitjax.tests import adcol_gate as A  # noqa: E402


def _digits(a, b):
    """testreport's digit count of two numbers (-log10 of the relative difference)."""
    if a == b:
        return 16
    return int(np.floor(-np.log10(abs(a - b) / max(abs(a), abs(b)))))


@pytest.fixture(scope="module")
def adj():
    from mitjax import paths as P
    from mitjax.drivers.adjoint_run import GenarrAdjoint
    from mitjax.drivers.model import Model
    from mitjax.drivers.run import load_experiment, make_rundir
    from mitjax.io import stdout as so
    from mitjax.tests import advect_gate as ag
    exp_dir = P.UPSTREAM / "verification" / A.EXP[0]
    e = load_experiment(exp_dir, A.EXP[1])
    m = Model(e, make_rundir(exp_dir, A.EXP[1], ag.out_dir("adcol-adjoint")))
    a = GenarrAdjoint(m, key=(3, 1))
    sz = m.cfg.size
    pts = [(0, k - 1, sz.OLy, sz.OLx) for k in (1, 2, 3, 4)]          # data.grdchk: (1,1,k), k = 1..4, tile 1
    taf = so.grdchk(so.read_stdout(P.UPSTREAM / "verification" / A.EXP[0] / "results" / "output_adm.txt"))
    fc, g = a.value_and_grad()
    fc2, g2 = a.value_and_grad()
    return dict(m=m, a=a, pts=pts, taf=taf, fc=float(fc), g=np.asarray(g), fc2=float(fc2), g2=np.asarray(g2))


def test_admgrd_vs_taf(adj):
    from mitjax.drivers.adjoint_run import adm_lines
    m, g, pts, taf = adj["m"], adj["g"], adj["pts"], adj["taf"]
    assert [(p.i, p.j, p.k, p.bi, p.bj) for p in taf.points] == [(1, 1, k, 1, 1) for k in (1, 2, 3, 4)]
    want = [p.adm["adjoint_gradient"] for p in taf.points]
    got = [float(g[p]) for p in pts]
    print(got, want, [_digits(x, y) for x, y in zip(got, want)])
    assert all(_digits(x, y) >= 10 for x, y in zip(got, want)), (got, want)
    lines = []
    for p in pts:
        lines += adm_lines(adj["fc"], g[p], 0.0)
    printed = [float(ln.split("=")[1]) for ln in lines if "adjoint_gradient" in ln]
    assert all(_digits(x, y) >= 10 for x, y in zip(printed, want)), (printed, want)
    ref = [ln for ln in lines if "ref_cost_function" in ln]
    assert all(f"{float(r.split('=')[1]):.14E}" == f"{taf.fcref:.14E}" for r in ref), (ref, taf.fcref)
    assert np.all(np.isfinite(g))
    sz = m.cfg.size
    interior = np.zeros(g.shape, bool)
    interior[..., sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] = True
    wet = interior & (np.asarray(m.grid.maskC.data) != 0)
    assert np.count_nonzero(g[~wet]) == 0 and np.count_nonzero(g[wet]) > 0


def test_repeat_bitwise(adj):
    assert adj["fc2"] == adj["fc"] and np.array_equal(adj["g2"].view(np.uint64), adj["g"].view(np.uint64))


def test_negative_control_weight(adj):
    """The genarr3d weight planted to 1 + 1e-7 (a traced input of the differentiated function: no recompile)."""
    import jax
    a, pts, taf = adj["a"], adj["pts"], adj["taf"]
    w = jax.tree_util.tree_map(lambda x: x * (1.0 + 1e-7), a.model.pkc["genarr_w"])
    fcb, gb = a.value_and_grad(model=a.model.replace(pkc={**a.model.pkc, "genarr_w": w}))
    assert float(fcb) == adj["fc"]                       # the first guess is zero: fc does not see the weight
    gb = np.asarray(gb)
    d = [_digits(float(gb[p]), t.adm["adjoint_gradient"]) for p, t in zip(pts, taf.points)]
    print(d)
    assert max(d) < 10, d
