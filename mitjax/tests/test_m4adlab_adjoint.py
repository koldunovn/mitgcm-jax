"""Lane M4ADLAB (M4 step 7 part 2, session 5): the adjoint of lab_sea/input_ad vs TAF, through the driver Model
(every package as data.pkg sets it: sea ice with the LSR dynamics, EXF, KPP, GMRedi, down_slope, pkg/ecco, pkg/ctrl)
and drivers/adjoint_run.Adjoint on the grdchk control: xx_atemp (gentim2d iarr 1), record 1 of its two, weight
ones_64b.bin: fc(xx) = COST_FINAL(THE_MAIN_LOOP(CTRL_MAP_INI_GENTIM2D(xx))), jax.value_and_grad through the
per-step-checkpointed scan. The run's own settings: TAF's adjoint mode equals the forward (data.autodiff is empty;
test_m4adlab_targets.py), the LSR derivative is the transpose of the executed sweeps (SEAICE_LSR_ADJOINT_ITER, plan
decision 14, mitjax/ad/lsr_sweeps.py), CG2D's derivative CG2D_MAD's (operator passive: no NONLIN_FRSURF;
mitjax/ad/modes.py).

Gated:
- The exact derivative of the forward: admGrd at the five grdchk points (tile (1,1), (i, 8), i = 6..10) equals our
  own central finite difference of fc at eps = 1e-2 (the jitted forward of the same driver; its FD at the grdchk's
  eps 1e-3 prints the FD oracle's -- TAF's -- fc+ / fc- / finite-diff_grad in every digit, also gated here) within
  FD_TOL = 1e-5 relative (measured 1.1e-7 .. 7.9e-7, dev job 27898125; the FD noise at eps 1e-2 is
  ~5e-7, fc printed to 15 digits). Negative controls: the gentim2d weight planted to 1 + 1e-4 (CTRL_MAP_INI_GENTIM2D
  divides by its square root: 5e-5 relative) fails it at every point, and so do TAF's own adjoint gradients
  (1e-4 .. 9e-4 relative from FD: the reason for the next item).
- The upstream's other derivatives of the same run at testreport's MATCH_CRIT (10 digits, verification/
  testreport:1154): admGrd vs TAF's tangent-linear model (results/output_tlm.txt.gz, `TLM  tangent-lin_grad`,
  checkpoint69p, code_ad + input_ad: measured 12 digits at all five points) and vs the Tapenade adjoint
  (results/output_tap_adj.txt, `ADM  adjoint_gradient`, code_tap + input_tap = input_ad's data with the Tapenade
  build: measured 12-13 digits). Negative control: the weight planted to 1 + 1e-7 fails both at every point.
- TAF's adjoint (results/output_adm.txt, TAF_ADM_GRAD of test_m4adlab_targets.py) is NOT matched: 3-4 digits
  (relative 9.7e-5, 1.1e-4, 4.9e-4, 8.8e-4, 7.6e-4). TAF's adjoint differs by the same amounts from TAF's own
  tangent-linear model, from the Tapenade adjoint and from the FD of the forward (its own finite-diff_grad column,
  = ours in every digit): the gap is in TAF's adjoint of this run, not in our forward or derivative
  (docs/ISSUES_UPSTREAM.md; MITgcm#1041). The measured digits are gated; the 10-digit match is a strict
  xfail.
- `ADM ref_cost_function` in every printed digit; the gradient finite on every lane and zero off the interior; a
  repeat bitwise.
Costs: about 30 min on a CPU node (the gradient program compiles in ~18 min, the forward in ~3 min; the controls
reuse both).
"""

import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

EXP = ("lab_sea", "input_ad")
POINTS_I = (6, 7, 8, 9, 10)


def _digits(a, b):
    """testreport's digit count of two numbers (-log10 of the relative difference)."""
    if a == b:
        return 16
    return int(np.floor(-np.log10(abs(a - b) / max(abs(a), abs(b)))))


@pytest.fixture(scope="module")
def adj():
    from mitjax import paths as P
    from mitjax.drivers.adjoint_run import Adjoint
    from mitjax.drivers.model import Model
    from mitjax.drivers.run import load_experiment, make_rundir
    from mitjax.io import stdout as so
    from mitjax.tests import advect_gate as ag
    exp_dir = P.UPSTREAM / "verification" / EXP[0]
    e = load_experiment(exp_dir, EXP[1])
    m = Model(e, make_rundir(exp_dir, EXP[1], ag.out_dir("adlab-adjoint")))
    a = Adjoint(m, iarr=1, rec=1)
    sz = m.cfg.size
    pts = [(0, sz.OLy + 8 - 1, sz.OLx + i - 1) for i in POINTS_I]       # data.grdchk: tile (1,1), (i, 8), rec 1
    taf = so.grdchk(so.read_stdout(exp_dir / "results" / "output_adm.txt"))
    fc, g = a.value_and_grad()
    fc2, g2 = a.value_and_grad()
    return dict(m=m, a=a, pts=pts, taf=taf, fc=float(fc), g=np.asarray(g), fc2=float(fc2), g2=np.asarray(g2))


def _upstream_adjoints():
    """{"tlm": TAF's TLM tangent-lin_grad, "tap": the Tapenade ADM adjoint_gradient} at the five points, in order."""
    import gzip
    import re
    from mitjax import paths as P
    res = P.UPSTREAM / "verification" / EXP[0] / "results"
    tlm = gzip.open(res / "output_tlm.txt.gz", "rt", errors="replace").read()
    tap = (res / "output_tap_adj.txt").read_text(errors="replace")
    out = {"tlm": [float(x) for x in re.findall(r"TLM\s+tangent-lin_grad\s+=\s+(\S+)", tlm)],
           "tap": [float(x) for x in re.findall(r"ADM\s+adjoint_gradient\s+=\s+(\S+)", tap)]}
    for text in (tlm, tap):          # the same five grdchk points as output_adm.txt
        assert re.findall(r"grdchk pos: i,j,k=\s+(\d+)\s+(\d+)\s+(\d+)", text) == \
            [(str(i), "8", "1") for i in POINTS_I]
    return out


def test_admgrd_vs_taf_tlm_and_tapenade(adj):
    g, pts = adj["g"], adj["pts"]
    up = _upstream_adjoints()
    got = [float(g[p]) for p in pts]
    for k in ("tlm", "tap"):
        dig = [_digits(x, y) for x, y in zip(got, up[k])]
        print(k, up[k], dig)
        assert len(up[k]) == 5 and min(dig) >= 10, (k, got, up[k], dig)


def test_negative_control_weight_10_digits(adj):
    """The gentim2d weight planted to 1 + 1e-7 moves the gradient by 5e-8: the 10-digit gates fail at every point."""
    import jax
    a, pts = adj["a"], adj["pts"]
    w = jax.tree_util.tree_map(lambda x: x * (1.0 + 1e-7), a.model.pkc["ctrl_weight"])
    fcb, gb = a.value_and_grad(model=a.model.replace(pkc={**a.model.pkc, "ctrl_weight": w}))
    assert float(fcb) == adj["fc"]
    gb = np.asarray(gb)
    up = _upstream_adjoints()
    for k in ("tlm", "tap"):
        dig = [_digits(float(gb[p]), y) for p, y in zip(pts, up[k])]
        print(k, dig)
        assert max(dig) < 10, (k, dig)


FD_TOL = 1e-5
FD_EPS = 1e-2
MEASURED_TAF_DIGITS = (4, 3, 3, 3, 3)          # dev job 27897240 (relative 9.7e-5 .. 8.8e-4)


def _fd(a, pts, eps, model=None):
    return [g for (_, _, _, g) in a.grdchk_fd(pts, eps=eps, model=model)]


def test_admgrd_equals_fd_of_forward(adj):
    a, g, pts, taf = adj["a"], adj["g"], adj["pts"], adj["taf"]
    # the forward of this driver at the grdchk's own perturbations is the FD oracle's (= TAF's) in every digit
    fd3 = a.grdchk_fd(pts)
    assert [(f"{fp:.14E}", f"{fm:.14E}", f"{gf:.14E}") for (_, fp, fm, gf) in fd3] == \
        [(f"{p.fcpertplus:.14E}", f"{p.fcpertminus:.14E}", f"{p.adm['finite-diff_grad']:.14E}") for p in taf.points]
    fd = _fd(a, pts, FD_EPS)
    got = [float(g[p]) for p in pts]
    rel = [abs(x - y) / abs(y) for x, y in zip(got, fd)]
    print("rel(admGrd, FD eps 1e-2):", rel)
    assert max(rel) < FD_TOL, rel
    # control 1: TAF's adjoint gradients are not the derivative of this forward at FD_TOL
    taf_rel = [abs(t.adm["adjoint_gradient"] - y) / abs(y) for t, y in zip(taf.points, fd)]
    print("rel(TAF admGrd, FD eps 1e-2):", taf_rel)
    assert min(taf_rel) > FD_TOL, taf_rel


def test_negative_control_weight_fd(adj):
    """The gentim2d weight planted to 1 + 1e-4 (model.pkc["ctrl_weight"], a traced input: no recompile) moves the
    gradient by 5e-5 relative: the FD gate (with the unplanted forward's FD) fails at every point."""
    import jax
    a, pts = adj["a"], adj["pts"]
    w = jax.tree_util.tree_map(lambda x: x * (1.0 + 1e-4), a.model.pkc["ctrl_weight"])
    fcb, gb = a.value_and_grad(model=a.model.replace(pkc={**a.model.pkc, "ctrl_weight": w}))
    assert float(fcb) == adj["fc"]                       # the first guess is zero: fc does not see the weight
    gb = np.asarray(gb)
    fd = _fd(a, pts, FD_EPS)
    rel = [abs(float(gb[p]) - y) / abs(y) for p, y in zip(pts, fd)]
    print(rel)
    assert min(rel) > FD_TOL, rel


def test_admgrd_vs_taf_measured_digits(adj):
    """The measured state of the TAF comparison (a change either way is news): 3-4 digits at the five points."""
    g, pts, taf = adj["g"], adj["pts"], adj["taf"]
    dig = tuple(_digits(float(g[p]), t.adm["adjoint_gradient"]) for p, t in zip(pts, taf.points))
    print(dig)
    assert dig == MEASURED_TAF_DIGITS, dig


@pytest.mark.xfail(strict=True, reason="admGrd vs TAF's adjoint: 3-4 digits, not MATCH_CRIT's 10. TAF's adjoint "
                                       "differs by 1e-4 .. 9e-4 from TAF's own TLM, the Tapenade adjoint and the FD "
                                       "of the forward, which ours matches (docs/ISSUES_UPSTREAM.md)")
def test_admgrd_vs_taf(adj):
    from mitjax.drivers.adjoint_run import adm_lines
    from mitjax.tests.test_m4adlab_targets import TAF_ADM_GRAD
    g, pts, taf = adj["g"], adj["pts"], adj["taf"]
    assert [(p.i, p.j, p.k, p.bi, p.bj) for p in taf.points] == [(i, 8, 1, 1, 1) for i in POINTS_I]
    want = [p.adm["adjoint_gradient"] for p in taf.points]
    assert [f"{w:.14E}" for w in want] == list(TAF_ADM_GRAD)
    got = [float(g[p]) for p in pts]
    print(got, want, [_digits(x, y) for x, y in zip(got, want)])
    assert all(_digits(x, y) >= 10 for x, y in zip(got, want)), (got, want)
    lines = []
    for p in pts:
        lines += adm_lines(adj["fc"], g[p], 0.0)
    printed = [float(ln.split("=")[1]) for ln in lines if "adjoint_gradient" in ln]
    assert all(_digits(x, y) >= 10 for x, y in zip(printed, want)), (printed, want)


def test_ref_cost_finite_interior(adj):
    m, g, taf = adj["m"], adj["g"], adj["taf"]
    assert f"{adj['fc']:.14E}" == f"{taf.fcref:.14E}", (adj["fc"], taf.fcref)
    assert np.all(np.isfinite(g))
    sz = m.cfg.size
    interior = np.zeros(g.shape, bool)
    interior[..., sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] = True
    assert np.count_nonzero(g[~interior]) == 0 and np.count_nonzero(g[interior]) > 0


def test_repeat_bitwise(adj):
    assert adj["fc2"] == adj["fc"] and np.array_equal(adj["g2"].view(np.uint64), adj["g"].view(np.uint64))
