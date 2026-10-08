"""global_ocean.cs32x15 input_ad.seaice / input_ad.seaice_dynmix: the whole forward run and GRDCHK's perturbed runs
(M4 step 7, last item; lane M4ADCS32ICE). The code_ad build's forward through the driver (helpers:
mitjax/tests/m4adcs32ice_gate.py).

* the whole run (2 / 5 steps) through the run driver (make_rundir + drivers.run.forward, as `python -m mitjax run`):
  every %MON record and MONITOR banner of the oracle's reference forward sweep and COST_FINAL's lines (early fc, the
  12 objf_test lines, local fc, global fc) identical to the oracle STDOUT (lane A's plain run job27855988-plain);
* GRDCHK (grdchk_main.F:359-432; xx_theta, genarr3d, grdchk_eps 1e-2, the 4 points nbeg 1, nstep 1, nend 4 at the
  positions the FD oracle prints): our fc at the zero control and at the 8 perturbations, and the finite
  differences, equal to lane A's FD oracle run job27856074-fdzero in every printed digit (1PE22.14); negative
  control: a planted cost weight (data.cost mult_test x (1 + 1e-7), a jit argument) misses every digit gate.
Costs: about 10-15 min per variant on a CPU compute node (the run driver's compile, the cost program, 9 runs).
"""

import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

from mitjax.tests import m4adcs32ice_gate as A  # noqa: E402


@pytest.fixture(autouse=True)
def _drop_compiled():
    import gc

    import jax
    yield
    jax.clear_caches()
    gc.collect()


@pytest.mark.parametrize("v", ["seaice", "dynmix"])
def test_whole_run_monitor_and_cost_lines(v):
    from mitjax.tests import cube_run_gate as C
    m, res, o = C.whole_run(*A.EXPS[v], tag=f"m4adcs32ice-whole-{v}")
    raw = o.raw
    k0 = next(n for n, x in enumerate(raw) if "Begin MONITOR dynamic field statistics" in x) - 1
    k1 = next(n for n, x in enumerate(raw) if "Start of S/R COST_FINAL" in x)
    sel = lambda x: "%MON " in x or "MONITOR " in x and "statistics" in x       # noqa: E731
    ours, theirs = [x for x in res.records if sel(x)], [x for x in raw[k0:k1] if sel(x)]
    print(v, len(ours), "monitor lines vs", len(theirs))
    assert len(ours) == len(theirs) > 0
    bad = [(a, b) for a, b in zip(ours, theirs) if a != b]
    assert bad == [], bad[:6]
    k2 = next(n for n, x in enumerate(raw) if " global fc =" in x)
    ca = [x.rstrip() for x in res.records if "fc =" in x or "objf_" in x]
    cb = [x.rstrip() for x in raw[k1:k2 + 1] if "fc =" in x or "objf_" in x]
    assert len(ca) == 15 and ca == cb, (ca, cb)


@pytest.mark.parametrize("v", ["seaice", "dynmix"])
def test_grdchk_perturbed_costs_every_digit(v):
    from mitjax.drivers.adjoint_run import GenarrAdjoint, adm_lines
    from mitjax.io import stdout as so
    from mitjax.pkg.ctrl.ctrl_readparms import ctrl_readparms_genarr
    r = A.run(v)
    m = r.m
    key = next((3, g.iarr) for g in ctrl_readparms_genarr(m.exp.run, 3, m.cfg) if g.file.strip() == "xx_theta")
    o = so.grdchk(so.read_stdout(A.fd_top(v) / "rundir" / "output.txt"))
    sz = m.cfg.size
    pts = [(p.bi - 1 + (p.bj - 1) * sz.nSx, p.k - 1, p.j - 1 + sz.OLy, p.i - 1 + sz.OLx) for p in o.points]
    assert len(pts) == 4
    a = GenarrAdjoint(m, key=key)
    fc = float(a.cost())
    fd = a.grdchk_fd(pts)
    print(v, "fc", f"{fc:.14E}", [(f"{fp:.14E}", f"{fm:.14E}", f"{g:.14E}") for _, fp, fm, g in fd])
    ours = []
    for (p, fp, fm, gfd), op in zip(fd, o.points):
        assert f"{fp:.14E}" == f"{op.fcpertplus:.14E}" and f"{fm:.14E}" == f"{op.fcpertminus:.14E}", (p, fp, fm)
        assert f"{fc:.14E}" == f"{op.adm['ref_cost_function']:.14E}", fc
        ours += adm_lines(fc, 0.0, gfd)
    keep = ("ref_cost_function", "finite-diff_grad")
    raw = (A.fd_top(v) / "rundir" / "output.txt").read_text(errors="replace").splitlines()
    theirs = [ln.rstrip() for ln in raw if " ADM  " in ln and any(k in ln for k in keep)]
    assert [ln for ln in ours if any(k in ln for k in keep)] == theirs, (ours, theirs)
    # negative control: a planted cost weight misses the digit gates (fc and one perturbed cost)
    cf = a.model.pkc["cost_fixed"]
    prm = {**cf["params"], "mult_test": cf["params"]["mult_test"] * (1.0 + 1e-7)}
    planted = a.model.replace(pkc={**a.model.pkc, "cost_fixed": {**cf, "params": prm}})
    fcb = float(a.cost(model=planted))
    fpb = float(a.cost(a.theta0.at[pts[0]].add(1e-2), model=planted))
    assert f"{fcb:.14E}" != f"{o.points[0].adm['ref_cost_function']:.14E}"
    assert f"{fpb:.14E}" != f"{o.points[0].fcpertplus:.14E}"
    assert np.isfinite(fcb)
