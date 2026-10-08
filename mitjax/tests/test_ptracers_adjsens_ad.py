"""tutorial_tracer_adjsens/input_ad: the adjoint (PTRACERS lane, plan Task 26), through the
driver Model and mitjax/drivers/adjoint_run.GenarrAdjoint. The control xx_ptr1 (genarr3d) enters the initial State
inside the differentiated function (adjoint_run.genarr_apply: CTRL_MAP_INI_GENARR, then CONVECTIVE_ADJUSTMENT_INI); fc is
COST_FINAL's mult_tracer*objf_tracer accumulated by COST_TILE's COST_TRACER over the 4 steps.

Digit target, stated before measuring: admGrd >= 12 digits vs TAF. The control is the initial passive tracer, so
dfc/dxx is the transpose of a map linear in the tracer at a fixed flow: the dynamics adjoint (CG2D_NSA's TAF adjoint vs
our implicit cg2d rule) does not enter it. Measured: every printed digit (1PE22.14) at all 5 points.

Gated:
1. `ADM ref_cost_function` (admCst) and `ADM adjoint_gradient` (admGrd) at the 5 grdchk points vs
   results/output_adm.txt, every printed digit; the gradient finite everywhere, zero off the wet interior.
2. Our GRDCHK finite differences (+-grdchk_eps = 1e-4 at the points: fcpertplus, fcpertminus, finite-diff_grad) vs
   lane A's FD oracle runs (job27832347-fdzero) and the results' ADM lines, every printed digit.
3. Trust protocol: the FD h-sweep (the best h within 1e-6 relative of the adjoint), a repeat of the gradient
   (bitwise); the tangent-linear (jvp) vs adjoint dot test is in test_ptracers_adjsens_admon.py.
4. The tangent-linear model at the 5 points (jvp on unit vectors) vs results/output_tlm*.txt.gz (`TLM
   tangent-lin_grad`), every printed digit.
5. Negative control: mult_tracer times 1 + 1e-7 changes the printed fc and admGrd misses the gate.

The `%MON ad_*` blocks (ADMONITOR's ad_dynstat / ad_forcing and ADPTRACERS_MONITOR) are gated in
test_ptracers_adjsens_admon.py. Tier 1x; the variant compiles a gradient program (~16 min on a CPU node), the
forward (~4 min) and the tangent; jax caches are cleared between programs (PORTING_LESSONS vm.max_map_count).
"""

import functools
import gc
import gzip

import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

# input_ad.som81: the forward is gated (test_ptracers_adjsens.py::test_som81_forward); its gradient program did not
# finish within a 30-min CPU dev job (27840319, 27840429): not gated here until it is measured
VARIANTS = ("input_ad",)
POINTS_IJK = [(i, 5, 4) for i in (15, 16, 17, 18, 19)]      # data.grdchk iGloPos.., nend = 4 (5 points), tile (1,1)


def _suffix(inp):
    return "" if inp == "input_ad" else "." + inp.split(".", 1)[1]


def results_lines(inp, kind="adm"):
    from mitjax import paths
    d = paths.UPSTREAM / "verification/tutorial_tracer_adjsens/results"
    if kind == "adm":
        text = (d / f"output_adm{_suffix(inp)}.txt").read_text(errors="replace")
    else:
        text = gzip.open(d / f"output_tlm{_suffix(inp)}.txt.gz", "rt", errors="replace").read()
    return [ln.rstrip() for ln in text.splitlines()]


def oracle_fd_lines(inp):
    """Lane A's FD oracle run (the forward of the TAF build at +-grdchk_eps, adjoint gradient zeroed)."""
    from mitjax import paths
    d = paths.REFERENCE_RUNS / "tutorial_tracer_adjsens" / inp / "job27832347-fdzero" / "rundir"
    return [ln.rstrip() for ln in (d / "output.txt").read_text(errors="replace").splitlines()]


def digits(a, b):
    """testreport's digit count (tools/testreport_jax: -log10 of the relative difference)."""
    if a == b:
        return 16
    return int(np.floor(-np.log10(abs(a - b) / max(abs(a), abs(b)))))


def points(m):
    sz = m.cfg.size
    return [(0, k - 1, sz.OLy + j - 1, sz.OLx + i - 1) for (i, j, k) in POINTS_IJK]


@functools.lru_cache(maxsize=None)
def adjoint(inp):
    from mitjax.drivers.adjoint_run import GenarrAdjoint
    from mitjax.tests.test_ptracers_latlon import model
    return GenarrAdjoint(model(("tutorial_tracer_adjsens", inp), "adjsens-ad"))


@functools.lru_cache(maxsize=None)
def gradient(inp):
    a = adjoint(inp)
    fc, g = a.value_and_grad()
    return float(fc), np.asarray(g)


def _clear():
    import jax
    jax.clear_caches()
    gc.collect()


@pytest.mark.parametrize("inp", VARIANTS)
def test_admgrd_admcst_vs_taf(inp):
    from mitjax.drivers.adjoint_run import adm_lines
    a = adjoint(inp)
    st = a.initial_state()[0][0]                     # init(zeros) is the Model's initial State, bitwise
    for n in ("theta", "salt", "pTracer_01"):
        x, y = np.asarray(getattr(st, n).data), np.asarray(getattr(a.m.state0, n).data)
        assert np.array_equal(x.view(np.uint64), y.view(np.uint64)), n
    fc, g = gradient(inp)
    assert np.all(np.isfinite(g))
    sz = a.m.cfg.size
    wet = np.zeros(g.shape, bool)
    wet[:, :, sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] = True
    wet &= np.asarray(a.m.grid.maskC.data) != 0
    assert np.all(g[~wet] == 0.0)
    ref = results_lines(inp)
    want = [ln for ln in ref if " ADM  ref_cost_function" in ln or " ADM  adjoint_gradient" in ln]
    got = []
    for p in points(a.m):
        got += [ln for ln in adm_lines(fc, g[p], 0.0) if "finite-diff" not in ln]
    assert got == want, [(x, y) for x, y in zip(got, want) if x != y]


@pytest.mark.parametrize("inp", VARIANTS[:1])
def test_gradient_repeat_bitwise(inp):
    """A second evaluation of the gradient program gives the same fc and gradient, bit for bit."""
    fc, g = gradient(inp)
    fc2, g2 = adjoint(inp).value_and_grad()
    assert float(fc2) == fc and np.array_equal(np.asarray(g2).view(np.uint64), g.view(np.uint64))


@pytest.mark.parametrize("inp", VARIANTS[:1])
def test_negative_control_cost_weight(inp):
    """mult_tracer times 1 + 1e-7 (a planted cost weight): the printed fc and admGrd differ from TAF's."""
    import jax.numpy as jnp
    a = adjoint(inp)
    cf = dict(a.model.pkc["cost_fixed"])
    cf["params"] = dict(cf["params"], mult_tracer=cf["params"]["mult_tracer"]*(1.0 + 1e-7))
    model = a.model.replace(pkc={**a.model.pkc, "cost_fixed": cf})
    fcb, gb = a.value_and_grad(model=model)
    gb = np.asarray(gb)
    ref = [float(ln.split("=")[1]) for ln in results_lines(inp) if " ADM  adjoint_gradient" in ln]
    reffc = float([ln for ln in results_lines(inp) if " ADM  ref_cost_function" in ln][0].split("=")[1])
    assert f"{float(fcb):.14E}" != f"{reffc:.14E}"
    assert min(digits(float(gb[p]), t) for p, t in zip(points(a.m), ref)) < 12
    del jnp


@pytest.mark.parametrize("inp", VARIANTS)
def test_grdchk_fd_vs_oracle(inp):
    """fcpertplus / fcpertminus and finite-diff_grad at the 5 points == lane A's FD oracle (every printed digit),
    and finite-diff_grad == the results' (the TAF build's FD)."""
    from mitjax.drivers.adjoint_run import adm_lines
    _clear()
    a = adjoint(inp)
    fd = a.grdchk_fd(points(a.m))
    orc = oracle_fd_lines(inp)
    plus = [ln for ln in orc if "grdchk perturb(+)fc" in ln]
    minus = [ln for ln in orc if "grdchk perturb(-)fc" in ln]
    gfd = [ln for ln in orc if " ADM  finite-diff_grad" in ln]
    assert len(plus) == len(minus) == len(gfd) == 5
    for k, (p, fp, fm, g) in enumerate(fd):
        assert plus[k].endswith(f"{fp:.14E}") and minus[k].endswith(f"{fm:.14E}"), (p, fp, fm, plus[k], minus[k])
        assert adm_lines(0.0, 0.0, g)[2] == gfd[k], (adm_lines(0.0, 0.0, g)[2], gfd[k])
    want = [ln for ln in results_lines(inp) if " ADM  finite-diff_grad" in ln]
    assert [adm_lines(0.0, 0.0, g)[2] for (_, _, _, g) in fd] == want


@pytest.mark.parametrize("inp", VARIANTS[:1])
def test_fd_h_sweep(inp):
    """FD h-sweep at the first point (central differences of our forward): the best h agrees with the adjoint
    gradient (TAF's printed admGrd, which ours equals to every digit: test_admgrd_admcst_vs_taf) within 1e-6 relative,
    and h -> 0 loses digits (the forward-noise floor)."""
    _clear()
    a = adjoint(inp)
    p = points(a.m)[0]
    ref = float([ln for ln in results_lines(inp) if " ADM  adjoint_gradient" in ln][0].split("=")[1])
    errs = {}
    for h in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        _, fp, fm, gfd = a.grdchk_fd([p], eps=h)[0]
        errs[h] = abs(gfd - ref) / abs(ref)
    assert min(errs.values()) < 1e-6, errs
    assert errs[1e-6] > min(errs.values()), errs


@pytest.mark.parametrize("inp", VARIANTS[:1])
def test_tlm_vs_taf(inp):
    _clear()
    _tlm(inp)


def _tlm(inp):
    """The tangent-linear model (jax.jvp) on the unit vector of each check point == `TLM tangent-lin_grad` of
    results/output_tlm.txt.gz (every printed digit), and `TLM ref_cost_function`."""
    from mitjax.io.fortran_format import fortran_write
    a = adjoint(inp)
    want = [ln for ln in results_lines(inp, "tlm") if " TLM  " in ln and "finite-diff" not in ln]
    got = []
    for p in points(a.m):
        e = np.zeros(np.shape(a.theta0))
        e[p] = 1.0
        fc, tl = a.jvp(e)
        for lit, val in ((" TLM  ref_cost_function      =", fc), (" TLM  tangent-lin_grad       =", tl)):
            got.append(("(PID.TID 0000.0001) " + fortran_write("(A30,1PE22.14)", lit, float(val))).rstrip())
    assert got == want, [(x, y) for x, y in zip(got, want) if x != y]
