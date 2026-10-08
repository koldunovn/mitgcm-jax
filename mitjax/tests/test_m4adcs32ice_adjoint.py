"""global_ocean.cs32x15/input_ad.seaice: the adjoint vs TAF (M4 step 7, last item; plan decision 17 revised; lane
M4ADCS32ICE session 2). fc(xx) = COST_FINAL(THE_MAIN_LOOP(CTRL_MAP_INI_GENARR(xx))) for the grdchk control xx_theta
(drivers/adjoint_run.GenarrAdjoint(key=)); the 4 grdchk points i = 1..4, j = 1, k = 1, tile (1,1); the CG2D derivative
follows the run's cg2dFullAdjoint = .FALSE. (operator passive, TAF's CG2D_MAD and TAF's TLM).

Two settings of the backward pass (the forward program is the same function in both; fc bit for bit):
* exact mode -- no backward-only switch, the LSR derivative A1 by the explicit option sp.mjx_lsr_derivative =
  "sweeps" (mitjax/ad/modes.lsr_derivative; this build has no SEAICE_LSR_ADJOINT_ITER): the derivative of our
  forward, which is what TAF's TLM computes (TAF's tangent never sets inAdMode). Gates: admGrd vs TAF's TLM
  (results/output_tlm.seaice.txt.gz, `TLM tangent-lin_grad`) >= 10 digits (testreport's 10); our central FD at eps
  1e-3 (grdchk_eps x 0.1) within FD_TOL; the dot test; finite on every lane, zero off the interior; repeat bitwise;
  control: the planted cost weight (data.cost mult_test x (1 + 1e-7)) misses the TLM gate.
* run mode -- the run's data.autodiff (useApproxAdvectionInAdMode = .TRUE.: scheme 33 -> 30 in the reverse sweep,
  mitjax/ad/approx_advection.py via drivers/ad_switches.with_run_switches) with A1: admGrd vs TAF's ADM
  (results/output_adm.seaice.txt). Effect test: fc identical, gradient differs from the exact mode's by more than the
  10-digit gate at every point; its negative control (exact vs exact) shows no difference.
Measured (lane M4ADCS32ICE session 2): exact vs TAF's TLM 11, 11, 12, 11 digits (job 27906279); run
vs TAF's ADM 8, 7, 7, 7 (pinned in RUN_TAF_ADM_DIGITS, 10 digits a strict xfail).
About 50-60 min on a CPU node: two gradient programs (~15-20 min compile each), the tangent, the forward (FD).
"""

import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

from mitjax.tests import m4adcs32ice_gate as A  # noqa: E402

V = "seaice"
GATE_DIGITS = 10                        # testreport's digits (plan decision 17)
FD_EPS = 1e-3                           # grdchk_eps x 0.1 (FD h-sweep: truncation ~1e-8 there, measured)
FD_TOL = 1e-7                           # |admGrd - FD(1e-3)| / |FD| (measured 1.1e-8 .. 4.0e-8, job 27906279)
DOT_TOL = 1e-12                         # jvp vs <grad, v> (measured 1.1e-14, job 27906279)
RUN_TAF_ADM_DIGITS = (8, 7, 7, 7)       # run mode vs TAF's ADM, measured (rel 8.1e-9, 1.5e-8, 2.5e-8, 3.3e-8; dev job
                                        # 27906489 on c46bcc2): TAF's sea-ice adjoint of the last step (LSR without
                                        # SEAICE_LSR_ADJOINT_ITER) -- handoff session 2


def _digits(a, b):
    if a == b:
        return 16
    return int(np.floor(-np.log10(abs(a - b) / max(abs(a), abs(b)))))


def _release():
    import gc

    import jax
    jax.clear_caches()
    gc.collect()


def _taf():
    import gzip
    import re

    from mitjax import paths
    from mitjax.io import stdout as so
    res = paths.UPSTREAM / "verification" / "global_ocean.cs32x15" / "results"
    adm = so.grdchk(so.read_stdout(res / "output_adm.seaice.txt"))
    tlm = gzip.open(res / "output_tlm.seaice.txt.gz", "rt", errors="replace").read()
    return dict(adm=[p.adm["adjoint_gradient"] for p in adm.points],
                tlm=[float(x) for x in re.findall(r"TLM\s+tangent-lin_grad\s+=\s+(\S+)", tlm)])


@pytest.fixture(scope="module")
def adj():
    import jax

    from mitjax.ad.modes import lsr_derivative, with_lsr_derivative
    from mitjax.drivers.ad_switches import with_run_switches
    from mitjax.drivers.adjoint_run import GenarrAdjoint
    from mitjax.io import stdout as so
    from mitjax.pkg.ctrl.ctrl_readparms import ctrl_readparms_genarr
    m = A.run(V).m
    key = next((3, g.iarr) for g in ctrl_readparms_genarr(m.exp.run, 3, m.cfg) if g.file.strip() == "xx_theta")
    o = so.grdchk(so.read_stdout(A.fd_top(V) / "rundir" / "output.txt"))
    sz = m.cfg.size
    P = [(p.bi - 1 + (p.bj - 1) * sz.nSx, p.k - 1, p.j - 1 + sz.OLy, p.i - 1 + sz.OLx) for p in o.points]
    a = GenarrAdjoint(m, key=key)
    exact = a.model.replace(pkc={**a.model.pkc, "sp": with_lsr_derivative(a.model.pkc["sp"], "sweeps")})
    run = with_run_switches(m, exact)
    out = dict(m=m, a=a, P=P, taf=_taf(), lsr_default=lsr_derivative(m.cfg, a.model.pkc["sp"]),
               lsr_exact=lsr_derivative(m.cfg, exact.pkc["sp"]))
    try:
        fc, g = a.value_and_grad(model=exact)
        out.update(fc=float(fc), g=np.asarray(g))
        fc2, g2 = a.value_and_grad(model=exact)
        out.update(fc2=float(fc2), g2=np.asarray(g2))
        cf = exact.pkc["cost_fixed"]
        prm = {**cf["params"], "mult_test": cf["params"]["mult_test"] * (1.0 + 1e-7)}
        fcb, gb = a.value_and_grad(model=exact.replace(pkc={**exact.pkc, "cost_fixed": {**cf, "params": prm}}))
        out.update(fcb=float(fcb), gb=np.asarray(gb))
    finally:
        _release()
    try:
        fcr, gr = a.value_and_grad(model=run)
        out.update(fcr=float(fcr), gr=np.asarray(gr))
    finally:
        _release()
    try:
        rng = np.random.default_rng(20261005)
        v = np.zeros(out["g"].shape)
        sl = (Ellipsis, slice(sz.OLy, sz.OLy + sz.sNy), slice(sz.OLx, sz.OLx + sz.sNx))
        v[sl] = rng.standard_normal(v[sl].shape)
        v *= np.asarray(m.grid.maskC.data)
        _, jv = a.jvp(jax.numpy.asarray(v), model=exact)
        out.update(dot=(float(jv), float(np.sum(out["g"] * v))))
    finally:
        _release()
    try:
        out["fc_default"] = float(a.cost())
        out["fd"] = a.grdchk_fd(P, eps=FD_EPS, model=exact)
    finally:
        _release()
    yield out
    _release()


def test_lsr_option(adj):
    """The build has no SEAICE_LSR_ADJOINT_ITER: by default the LSR is forward only; the explicit option gives A1."""
    assert adj["lsr_default"] == "forward_only" and adj["lsr_exact"] == "sweeps"


def test_forward_identical_every_setting(adj):
    """fc of the default program (while_loop LSR), the exact mode (A1 scan) and the run mode (switch on) bit for bit,
    = TAF's ref_cost_function in every printed digit."""
    fc = adj["fc"]
    assert adj["fc_default"] == fc == adj["fc2"] == adj["fcr"]
    from mitjax.io import stdout as so
    from mitjax import paths
    res = paths.UPSTREAM / "verification" / "global_ocean.cs32x15" / "results"
    assert f"{fc:.14E}" == f"{so.grdchk(so.read_stdout(res / 'output_adm.seaice.txt')).fcref:.14E}"


def test_exact_mode_matches_taf_tlm(adj):
    """Exact mode admGrd vs TAF's TLM >= GATE_DIGITS at the 4 points."""
    got = [float(adj["g"][p]) for p in adj["P"]]
    d = [_digits(x, y) for x, y in zip(got, adj["taf"]["tlm"])]
    print("exact admGrd", [f"{x:.14E}" for x in got], "vs TAF TLM digits", d)
    assert min(d) >= GATE_DIGITS, (got, adj["taf"]["tlm"])


def test_negative_control_cost_weight(adj):
    """Planted cost weight x (1 + 1e-7): fc changes and the TLM gate fails at every point."""
    assert adj["fcb"] != adj["fc"]
    assert max(_digits(float(adj["gb"][p]), y) for p, y in zip(adj["P"], adj["taf"]["tlm"])) < GATE_DIGITS


def test_exact_mode_vs_fd(adj):
    """Our central FD of the jitted forward at eps 1e-3 within FD_TOL of the exact admGrd; control: TAF's ADM (the
    run mode's approximate derivative) misses it."""
    rel = [abs(float(adj["g"][p]) - fd) / abs(fd) for (p, _, _, fd) in adj["fd"]]
    print("exact vs FD(1e-3) rel", rel)
    assert max(rel) <= FD_TOL
    rel_taf = [abs(y - fd) / abs(fd) for y, (_, _, _, fd) in zip(adj["taf"]["adm"], adj["fd"])]
    assert min(rel_taf) > FD_TOL, rel_taf


def test_dot_test(adj):
    jv, gv = adj["dot"]
    rel = abs(jv - gv) / max(abs(jv), abs(gv))
    print("dot test", jv, gv, rel)
    assert rel <= DOT_TOL


def test_finite_interior_repeat(adj):
    """Both gradients finite on every lane, zero off the interior; the exact gradient repeats bit for bit."""
    m = adj["m"]
    sz = m.cfg.size
    for g in (adj["g"], adj["gr"]):
        interior = np.zeros(g.shape, bool)
        interior[..., sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] = True
        assert np.all(np.isfinite(g))
        assert np.count_nonzero(g[~interior]) == 0
    assert np.array_equal(adj["g"].view(np.uint64), adj["g2"].view(np.uint64))


def _effect(g1, g2, P):
    return [_digits(float(g1[p]), float(g2[p])) for p in P]


def test_approx_advection_switch_effect(adj):
    """The backward-only switch has an effect at every grdchk point (fc identical: test_forward_identical_*);
    negative control: the same comparison of the exact gradient with its repeat shows none."""
    d = _effect(adj["gr"], adj["g"], adj["P"])
    print("run vs exact digits", d)
    assert max(d) < GATE_DIGITS
    assert min(_effect(adj["g2"], adj["g"], adj["P"])) >= GATE_DIGITS       # control: no effect -> would fail above


def test_run_mode_vs_taf_adm(adj):
    """Run mode admGrd vs TAF's ADM: the measured digits (RUN_TAF_ADM_DIGITS) at every point. Localised (adjoint
    monitor, handoff session 2): in the reverse of the last step TAF's ocean adjoints equal ours (adtheta every
    statistic, adsalt >= 9 digits) while its sea-ice adjoints do not (aduice ~10x ours, adarea, adhsnow): TAF's LSR
    adjoint with the overwritten tape (seaice_lsr.F:805-807), which no backward-only switch reproduces."""
    got = [float(adj["gr"][p]) for p in adj["P"]]
    d = [_digits(x, y) for x, y in zip(got, adj["taf"]["adm"])]
    print("run admGrd", [f"{x:.14E}" for x in got], "vs TAF ADM digits", d)
    assert RUN_TAF_ADM_DIGITS is not None and d == list(RUN_TAF_ADM_DIGITS), d


@pytest.mark.xfail(strict=True, reason="TAF's adjoint of input_ad.seaice includes its LSR adjoint without "
                   "SEAICE_LSR_ADJOINT_ITER (wrong by construction, seaice_lsr.F:805-807; MITgcm#1041): 7-8 digits")
def test_run_mode_taf_adm_ten_digits(adj):
    got = [float(adj["gr"][p]) for p in adj["P"]]
    assert min(_digits(x, y) for x, y in zip(got, adj["taf"]["adm"])) >= GATE_DIGITS
