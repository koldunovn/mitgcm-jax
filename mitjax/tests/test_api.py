"""The thin API (mitjax/api.py, docs plan 20261006 Task 6) against the gated drivers and TAF's verification outputs.

* run: tutorial_barotropic_gyre/input through `exp.run` gives drivers/run.run's output.txt and pickups byte for byte;
  `compare(run, exp.results())` passes testreport (>= 10 digits on cg2d_init_res); negative control: the same output
  with every cg2d_init_res value perturbed by 1e-6 (relative) FAILs.
* gradient + grdchk: 1D_ocean_ice_column/input_ad (control xx_theta from data.grdchk) against the gate numbers of
  test_m4adcol_adjoint.py / test_m4adcol_fd.py: admGrd >= 10 digits vs TAF's results/output_adm.txt at the 4 grdchk
  points (gradient and grdchk), ref_cost_function and the perturbed costs fc+ / fc- in every printed digit (TAF's
  lines; equal to lane A's FD oracle), compare(chk, exp.results()) passes testreport's adm check (admGrd decides);
  the adxx_theta file holds our gradient's interior (MDS layout) and its .meta is the MDS writer's text of lane A's
  oracle run (where it exists); negative control: the grdchk output with every adjoint_gradient value perturbed by
  1e-7 (relative) FAILs compare.
* mode: global_ocean.cs32x15/input_ad (the cheapest live case where "run" and "exact" differ: no data.autodiff
  switch is set, but cg2dFullAdjoint = .FALSE. with nonlinFreeSurf = 4 keeps the CG2D operator passive in TAF's
  adjoint, mitjax/ad/modes.py; input_ad.seaice_dynmix needs the LSR and 5 sea-ice steps): mode="run" >= 10 digits vs
  TAF at the 4 grdchk points; mode="exact" has the same fc bit for bit and a gradient that misses TAF's 10 digits
  (measured 6-7 by test_cs32_ad_adjoint.py) -- the negative control of mode="run": the wrong switch set fails the gate.
* sharded: tutorial_global_oce_optim/input_ad (4 tiles of 45x20, gentim2d xx_qnet) gradient at devices=2 vs
  devices=1: fc bit for bit, gradient finite (max relative difference printed).
* errors: an unknown variant, a variant without data.ctrl, an unported option -- each names the file or routine.
* lane APIF: cg2d_derivative="run" with mode="exact" = mode="run" bit for bit on cs32x15/input_ad (one more gradient
  program, ~9 min); gradient() vs grdchk() adjoint gradients bit for bit with jax.clear_caches() between programs.
* CLI: `python -m mitjax compare` on the API run's output (exit 0) and on the perturbed one (exit 1); the lazy
  `import mitjax` (no driver imported).
Costs (CPU node): run ~3 min, 1D column ~15 min, cs32x15 ~35 min (two gradient programs), optim ~15 min.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

from mitjax import paths  # noqa: E402

ADMGRD_DIGITS = 10                       # testreport's MATCH_CRIT (verification/testreport:1154), the existing gates'


def _digits(a, b):
    if a == b:
        return 16
    return int(np.floor(-np.log10(abs(a - b) / max(abs(a), abs(b)))))


def _out(tag):
    from mitjax.tests import advect_gate as ag
    return ag.out_dir(f"api-{tag}")


def _exp(name):
    return paths.UPSTREAM / "verification" / name


def _taf(name):
    from mitjax.io import stdout as so
    return so.grdchk(so.read_stdout(_exp(name) / "results" / "output_adm.txt"))


def _release():
    import gc

    import jax
    jax.clear_caches()
    gc.collect()


def _perturb(src, dst, needle, factor):
    """A copy of output file `src` with the value after '=' of every line containing `needle` times `factor`."""
    out = []
    for ln in Path(src).read_text().splitlines():
        if needle in ln and "=" in ln:
            head, val = ln.rsplit("=", 1)
            ln = f"{head}= {float(val.replace('D', 'E')) * factor:.14E}"
        out.append(ln)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text("\n".join(out) + "\n")
    return Path(dst)


# ---------------------------------------------------------------------------------------------------------- run

@pytest.fixture(scope="module")
def gyre():
    import mitjax
    from mitjax.drivers.run import run as driver_run
    exp = mitjax.load(_exp("tutorial_barotropic_gyre"), variant="input")
    ref = driver_run(_exp("tutorial_barotropic_gyre"), "input", _out("gyre-driver"))
    r = exp.run(out=_out("gyre-api"))
    _release()
    return dict(exp=exp, driver_output=Path(ref), run=r)


def test_run_bitwise_vs_driver(gyre):
    r, ref = gyre["run"], gyre["driver_output"]
    assert r.output.read_bytes() == ref.read_bytes()
    pk = sorted(p.name for p in ref.parent.iterdir() if p.name.startswith("pickup"))
    assert pk and sorted(p.name for p in r.rundir.iterdir() if p.name.startswith("pickup")) == pk
    for n in pk:
        assert (r.rundir / n).read_bytes() == (ref.parent / n).read_bytes(), n
    assert {"etaN", "uVel", "vVel", "theta"} <= set(r.fields), sorted(r.fields)
    assert all(np.all(np.isfinite(v)) for v in r.fields.values())
    assert r.interior("etaN").shape[-2:] == (r.size.sNy, r.size.sNx)


def test_compare_run_passes_and_perturbed_fails(gyre):
    import mitjax
    res = gyre["exp"].results()
    assert res.output == _exp("tutorial_barotropic_gyre") / "results" / "output.txt" and res.output_adm is None
    rep = mitjax.compare(gyre["run"], res)
    print(rep.summary)
    assert rep.verdict == "pass", rep.summary
    bad = _perturb(gyre["run"].output, _out("gyre-perturbed") / "output.txt", "cg2d_init_res", 1.0 + 1e-6)
    rep_bad = mitjax.compare(bad, res, kind="fwd")
    print(rep_bad.summary)
    assert rep_bad.verdict == "FAIL", rep_bad.summary


def test_cli_compare_exit_codes(gyre):
    env = dict(os.environ, PYTHONPATH=str(paths.REPO))
    base = [sys.executable, "-m", "mitjax", "compare"]
    ok = subprocess.run(base + [str(gyre["run"].output), str(_exp("tutorial_barotropic_gyre")), "--variant",
                                "input"], env=env, capture_output=True, text=True)
    assert ok.returncode == 0, (ok.stdout, ok.stderr[-2000:])
    bad = _perturb(gyre["run"].output, _out("gyre-perturbed-cli") / "output.txt", "cg2d_init_res", 1.0 + 1e-6)
    no = subprocess.run(base + [str(bad), str(_exp("tutorial_barotropic_gyre")), "--variant", "input"], env=env,
                        capture_output=True, text=True)
    assert no.returncode == 1, (no.stdout, no.stderr[-2000:])


def test_lazy_import_and_cli_help():
    env = dict(os.environ, PYTHONPATH=str(paths.REPO))
    code = ("import sys, mitjax; assert 'mitjax.api' not in sys.modules and 'mitjax.drivers' not in sys.modules; "
            "from mitjax import paths; assert 'mitjax.api' not in sys.modules; f = mitjax.load; "
            "assert 'mitjax.api' in sys.modules and f.__module__ == 'mitjax.api'")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    for cmd in ("run", "gradient", "grdchk", "compare"):
        h = subprocess.run([sys.executable, "-m", "mitjax", cmd, "--help"], env=env, capture_output=True, text=True)
        assert h.returncode == 0 and "usage" in h.stdout, (cmd, h.stderr[-1000:])


# ------------------------------------------------------------------------------------------- gradient + grdchk

@pytest.fixture(scope="module")
def col():
    import mitjax
    exp = mitjax.load(_exp("1D_ocean_ice_column"), variant="input_ad")
    g = exp.gradient(out=_out("col-gradient"))
    _release()
    chk = exp.grdchk(out=_out("col-grdchk"))
    _release()
    return dict(exp=exp, g=g, chk=chk, taf=_taf("1D_ocean_ice_column"))


def test_gradient_vs_taf(col):
    g, taf = col["g"], col["taf"]
    assert g.control == "xx_theta" and g.mode == "run" and list(g.adxx) == [1]
    assert f"{g.fc:.14E}" == f"{taf.fcref:.14E}", (g.fc, taf.fcref)
    sz = col["exp"].config.cfg.size
    pts = [(p.bi - 1 + (p.bj - 1) * sz.nSx, p.k - 1, p.j - 1 + sz.OLy, p.i - 1 + sz.OLx) for p in taf.points]
    a = g.adxx[1]
    d = [_digits(float(a[p]), t.adm["adjoint_gradient"]) for p, t in zip(pts, taf.points)]
    print("gradient admGrd digits vs TAF:", d)
    assert len(pts) == 4 and min(d) >= ADMGRD_DIGITS, d
    assert np.all(np.isfinite(a))


def test_adxx_file(col):
    """adxx_theta.0000000000.data holds our gradient's interior (big-endian float64, MDS layout), and its .meta is
    the text lane A's oracle run of the same experiment has in adxx_theta.0000000000.meta (the same MDS writer,
    MDS_WRITE_FIELD; skipped without the oracle). The DATA of that run are not a reference: job27856073-fdzero is a
    forward gfortran build (its adxx_theta is zeros by construction), and no TAF adxx file exists in the reference
    data -- TAF's gradient is known only through results/output_adm.txt, checked at the grdchk points by
    test_gradient_vs_taf / test_grdchk_vs_taf (lane API session 2)."""
    g = col["g"]
    sz = col["exp"].config.cfg.size
    data = g.rundir / "adxx_theta.0000000000.data"
    assert data in g.files and (g.rundir / "adxx_theta.0000000000.meta") in g.files
    ours = np.fromfile(data, dtype=">f8")
    want = g.adxx[1][0, :, sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx]
    assert np.array_equal(ours, want.ravel()), (ours[:4], want.ravel()[:4])
    orc = paths.REFERENCE_RUNS / "1D_ocean_ice_column" / "input_ad" / "job27856073-fdzero" / "rundir"
    if not (orc / "adxx_theta.0000000000.meta").exists():
        pytest.skip("lane A's oracle run is not available (no .meta to compare)")
    meta_ours = (g.rundir / "adxx_theta.0000000000.meta").read_text()
    meta_orc = (orc / "adxx_theta.0000000000.meta").read_text()
    print(meta_ours)
    assert meta_ours == meta_orc, (meta_ours, meta_orc)


def test_grdchk_vs_taf(col):
    from mitjax.io import stdout as so
    chk, taf = col["chk"], col["taf"]
    ours = so.grdchk(so.read_stdout(chk.output))
    assert [(p.i, p.j, p.k, p.bi, p.bj, p.rec) for p in ours.points] == \
        [(p.i, p.j, p.k, p.bi, p.bj, p.rec) for p in taf.points]
    assert f"{ours.fcref:.14E}" == f"{taf.fcref:.14E}"
    for o, t in zip(ours.points, taf.points):
        print(o.k, o.adm, t.adm, o.fcpertplus, t.fcpertplus, o.fcpertminus, t.fcpertminus)
        assert f"{o.adm['ref_cost_function']:.14E}" == f"{t.adm['ref_cost_function']:.14E}"
        assert _digits(o.adm["adjoint_gradient"], t.adm["adjoint_gradient"]) >= ADMGRD_DIGITS
        assert f"{o.fcpertplus:.14E}" == f"{t.fcpertplus:.14E}", (o.k, o.fcpertplus, t.fcpertplus)
        assert f"{o.fcpertminus:.14E}" == f"{t.fcpertminus:.14E}", (o.k, o.fcpertminus, t.fcpertminus)
    # the GRDCHK_PRINT block's (p) / (c) rows: TAF's text
    want = [ln.split(") ", 1)[1] for ln in _exp("1D_ocean_ice_column").joinpath("results", "output_adm.txt")
            .read_text().splitlines() if "grdchk output (p)" in ln or "grdchk output (c)" in ln
            or "grdchk output h." in ln or " EPS =" in ln]
    got = [ln.split(") ", 1)[1] for ln in chk.lines if "grdchk output (p)" in ln or "grdchk output (c)" in ln
           or "grdchk output h." in ln or " EPS =" in ln]
    assert got == want, (got, want)


def test_compare_grdchk_passes_and_perturbed_fails(col):
    import mitjax
    res = col["exp"].results()
    rep = mitjax.compare(col["chk"], res)
    print(rep.summary, [(v.name, v.digits) for v in rep.run.variables])
    assert rep.verdict == "pass", rep.summary
    bad = _perturb(col["chk"].output, _out("col-grdchk-perturbed") / "output.txt", "ADM  adjoint_gradient",
                   1.0 + 1e-7)
    rep_bad = mitjax.compare(bad, res, kind="adm")
    print(rep_bad.summary)
    assert rep_bad.verdict == "FAIL", rep_bad.summary


# ------------------------------------------------------------------------------------------------------- mode

@pytest.fixture(scope="module")
def cs32():
    import mitjax
    from mitjax.drivers.grdchk import grdchk_positions, grdchk_settings, storage_point
    exp = mitjax.load(_exp("global_ocean.cs32x15"), variant="input_ad")
    out = {}
    for mode in ("run", "exact"):
        out[mode] = exp.gradient(out=_out(f"cs32-{mode}"), mode=mode)
        _release()
    # lane APIF (contract 9): mode "exact" with the run's CG2D derivative (TAF's passive operator)
    out["exact_cg2d_run"] = exp.gradient(out=_out("cs32-exact-cg2drun"), mode="exact", cg2d_derivative="run")
    _release()
    m = exp.model(_out("cs32-points"))
    s = grdchk_settings(m.exp)
    from mitjax.drivers.grdchk import control_by_name
    ctl = control_by_name(m, s["grdchkvarname"])
    pts = grdchk_positions(m, ctl, s)
    out["pts"] = [storage_point(ctl, r, m.cfg.size) for _, r in pts]
    out["pos"] = [(r.itilepos, r.jtilepos, r.layer, r.itile, r.jtile) for _, r in pts]
    return out


def test_mode_run_matches_taf_exact_differs(cs32):
    taf = _taf("global_ocean.cs32x15")
    assert cs32["pos"] == [(p.i, p.j, p.k, p.bi, p.bj) for p in taf.points]
    want = [p.adm["adjoint_gradient"] for p in taf.points]
    run, exact = cs32["run"], cs32["exact"]
    d_run = [_digits(float(run.adxx[1][p]), w) for p, w in zip(cs32["pts"], want)]
    d_exact = [_digits(float(exact.adxx[1][p]), w) for p, w in zip(cs32["pts"], want)]
    print("cs32x15 admGrd digits vs TAF: run", d_run, "exact", d_exact)
    assert run.fc == exact.fc                                         # backward-only: the forward is the same
    assert min(d_run) >= ADMGRD_DIGITS, d_run
    assert max(d_exact) < ADMGRD_DIGITS, d_exact                     # the wrong switch set misses the gate


def test_cg2d_derivative_override(cs32):
    """Lane APIF (contract 9): gradient(mode="exact", cg2d_derivative="run") is the program of mode="run" on this run
    (input_ad sets no data.autodiff switch: the modes differ only in the CG2D derivative), fc and every gradient value
    bit for bit. Control (the argument ignored): that program is mode="exact"'s, which the test above measures to
    miss TAF's digits and to differ from mode="run" (asserted here too)."""
    run, exact, over = cs32["run"], cs32["exact"], cs32["exact_cg2d_run"]
    assert over.fc == run.fc and over.adxx[1].tobytes() == run.adxx[1].tobytes()
    assert exact.adxx[1].tobytes() != run.adxx[1].tobytes()           # the planted arm (= mode "exact") differs


def test_clear_caches_values_unchanged(col):
    """Lane APIF (contract 8): gradient() and grdchk() now call jax.clear_caches() between programs; the gradient of
    gradient() and grdchk()'s adjoint gradient at the check points (separate programs, caches cleared in between)
    are bit for bit the same, and so are their reference costs."""
    g, chk = col["g"], col["chk"]
    sz = col["exp"].config.cfg.size
    for c in chk.checks:
        p = (c["bi"] - 1 + (c["bj"] - 1) * sz.nSx, c["k"] - 1, c["j"] - 1 + sz.OLy, c["i"] - 1 + sz.OLx)
        assert float(g.adxx[1][p]) == c["adjoint_gradient"], (p, float(g.adxx[1][p]), c["adjoint_gradient"])
        assert c["fcref"] == g.fc


# ---------------------------------------------------------------------------------------------------- sharded

def test_sharded_gradient_fc_bitwise():
    import jax

    import mitjax
    assert len(jax.devices()) >= 2, jax.devices()
    exp = mitjax.load(_exp("tutorial_global_oce_optim"), variant="input_ad")
    g1 = exp.gradient(out=_out("optim-p1"))
    _release()
    g2 = exp.gradient(out=_out("optim-p2"), devices=2)
    _release()
    assert g1.control == g2.control == "xx_qnet" and list(g1.adxx) == list(g2.adxx)
    a, b = g1.adxx[1], g2.adxx[1]
    rel = float(np.max(np.abs(a - b)) / np.max(np.abs(a)))
    print("optim fc", g1.fc, g2.fc, "max rel gradient difference P=2 vs P=1:", rel)
    assert g2.fc == g1.fc
    assert np.all(np.isfinite(b)) and a.shape == b.shape


# ----------------------------------------------------------------------------------------------------- errors

def test_error_unknown_variant():
    import mitjax
    with pytest.raises(FileNotFoundError, match="no input dir .*input.nope"):
        mitjax.load(_exp("tutorial_barotropic_gyre"), variant="input.nope")


def test_error_no_data_ctrl():
    import mitjax
    exp = mitjax.load(_exp("tutorial_barotropic_gyre"), variant="input")
    with pytest.raises(FileNotFoundError, match="data.ctrl"):
        exp.gradient(out=_out("gyre-nogradient"))


def test_error_unported_option():
    import mitjax
    exp = mitjax.load(_exp("vermix"), variant="input.gglLC")
    with pytest.raises(NotImplementedError, match="GGL90_ADD_STOKESDRIFT"):
        exp.run(out=_out("vermix-gglLC"))
    _release()
