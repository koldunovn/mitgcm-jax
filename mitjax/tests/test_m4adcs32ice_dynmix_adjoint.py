"""global_ocean.cs32x15/input_ad.seaice_dynmix: the adjoint vs TAF (M4 step 7, last item; plan decision 17 revised;
lane M4ADCS32ICE session 2). The run sets SEAICEuseFREEDRIFTswitchInAd = .TRUE. (data.autodiff): TAF's reverse sweep
differentiates the free drift in place of the LSR (seaice_dynsolver.F:303-346 in AD mode; autodiff_inadmode_set_ad.F
:69-72), its TLM the LSR. The port follows it with the backward-only switch of mitjax/ad/freedrift_switch.py
(drivers/ad_switches.with_run_switches).

* run mode (the switch on): admGrd vs TAF's ADM (results/output_adm.seaice_dynmix.txt) >= 10 digits at the 4 grdchk
  points (xx_theta, i = 1..4, j = 1, k = 1, tile (1,1)); control: the planted cost weight misses it.
* exact mode (no switch, the LSR derivative A1 by sp.mjx_lsr_derivative = "sweeps"): vs TAF's TLM
  (output_tlm.seaice_dynmix.txt.gz) -- measured digits pinned (EXACT_TAF_TLM_DIGITS).
* fc of the default, exact and run programs bit for bit.
* effect test of the switch on a cost that sees the ice dynamics (the grdchk points barely do), on the live fixture:
  SEAICE_DYNSOLVER of iteration 36000 (inputs from lane A's dumps job27855988-jdon, m4off_gate.inputs_at), J =
  <w, (uIce, vIce)> after the solver: its outputs bit for bit with the switch on and off; the gradient with respect to
  HEFF and the ocean uVel differs by O(1) between the switch and A1; with the switch it equals the gradient of
  <clip'(uIce_LSR) w, (uice_fd, vice_fd)> (SEAICE_FREEDRIFT's outputs: the AD-mode block's uIce = uIce_fd, then
  SEAICE_clipVelocities, seaice_dynsolver.F:388-410, whose derivative mask TAF takes from the uice / vice it stores
  before the clip, :390-392, i.e. the forward's LSR velocity) to rounding; negative controls: the same "differs"
  comparison of A1 with itself fails, and the reference without the clip mask misses (session 2's test did that:
  0.30, job 27907059).
About 45-55 min on a CPU node (two gradient programs, the kernel programs).
"""

import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

from mitjax.tests import m4adcs32ice_gate as A  # noqa: E402

V = "dynmix"
GATE_DIGITS = 10
EXACT_TAF_TLM_DIGITS = (11, 10, 10, 10)  # measured: dev job 27906492 (rel 8.8e-12, 2.7e-11, 2.9e-11, 4.0e-11)
KERNEL_TOL = 1e-12                      # switch-on gradient vs the free-drift gradient, relative to max |g|


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
    adm = so.grdchk(so.read_stdout(res / "output_adm.seaice_dynmix.txt"))
    tlm = gzip.open(res / "output_tlm.seaice_dynmix.txt.gz", "rt", errors="replace").read()
    return dict(fcref=adm.fcref, adm=[p.adm["adjoint_gradient"] for p in adm.points],
                tlm=[float(x) for x in re.findall(r"TLM\s+tangent-lin_grad\s+=\s+(\S+)", tlm)])


@pytest.fixture(scope="module")
def adj():
    from mitjax.ad.modes import with_lsr_derivative
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
    out = dict(m=m, a=a, P=P, taf=_taf(), run_sp=run.pkc["sp"], exact_sp=exact.pkc["sp"])
    try:
        fcr, gr = a.value_and_grad(model=run)
        out.update(fcr=float(fcr), gr=np.asarray(gr))
        cf = run.pkc["cost_fixed"]
        prm = {**cf["params"], "mult_test": cf["params"]["mult_test"] * (1.0 + 1e-7)}
        fcb, gb = a.value_and_grad(model=run.replace(pkc={**run.pkc, "cost_fixed": {**cf, "params": prm}}))
        out.update(fcb=float(fcb), gb=np.asarray(gb))
    finally:
        _release()
    try:
        fce, ge = a.value_and_grad(model=exact)
        out.update(fce=float(fce), ge=np.asarray(ge))
    finally:
        _release()
    try:
        out["fc_default"] = float(a.cost())
    finally:
        _release()
    yield out
    _release()


def test_forward_identical_every_setting(adj):
    assert adj["fc_default"] == adj["fce"] == adj["fcr"]
    assert f"{adj['fcr']:.14E}" == f"{adj['taf']['fcref']:.14E}"


def test_run_mode_matches_taf_adm(adj):
    got = [float(adj["gr"][p]) for p in adj["P"]]
    d = [_digits(x, y) for x, y in zip(got, adj["taf"]["adm"])]
    print("run admGrd", [f"{x:.14E}" for x in got], "vs TAF ADM digits", d)
    assert min(d) >= GATE_DIGITS, (got, adj["taf"]["adm"])


def test_negative_control_cost_weight(adj):
    assert adj["fcb"] != adj["fcr"]
    assert max(_digits(float(adj["gb"][p]), y) for p, y in zip(adj["P"], adj["taf"]["adm"])) < GATE_DIGITS


def test_exact_mode_vs_taf_tlm(adj):
    got = [float(adj["ge"][p]) for p in adj["P"]]
    d = [_digits(x, y) for x, y in zip(got, adj["taf"]["tlm"])]
    print("exact admGrd", [f"{x:.14E}" for x in got], "vs TAF TLM digits", d,
          "run vs exact digits", [_digits(float(adj["gr"][p]), float(adj["ge"][p])) for p in adj["P"]])
    assert EXACT_TAF_TLM_DIGITS is not None and d == list(EXACT_TAF_TLM_DIGITS), d


def test_finite_and_interior(adj):
    sz = adj["m"].cfg.size
    for g in (adj["gr"], adj["ge"]):
        interior = np.zeros(g.shape, bool)
        interior[..., sz.OLy:sz.OLy + sz.sNy, sz.OLx:sz.OLx + sz.sNx] = True
        assert np.all(np.isfinite(g)) and np.count_nonzero(g[~interior]) == 0


@pytest.fixture(scope="module")
def kern():
    """SEAICE_DYNSOLVER of iteration 36000 from the oracle's inputs, J = <w, (uIce, vIce)> after it, its gradient with
    respect to (HEFF, ocean uVel) with the switch (run) and with A1 (exact), and the gradient of <m w, (uice_fd,
    vice_fd)> (the free drift, Y03) in the exact program, with m = 1 (no clip) and m = clip'(uIce_LSR) (the derivative
    of seaice_dynsolver.F:401-404, MAX(MIN(u, 0.40), -0.40), at the pre-clip LSR velocity Y06_lsr)."""
    import jax
    import jax.numpy as jnp

    from mitjax.ad.modes import with_lsr_derivative
    from mitjax.drivers.ad_switches import with_run_switches
    from mitjax.tests import m4adlab_gate as L
    from mitjax.tests import m4off_gate as O
    r = A.run(V)
    m = r.m
    exact_sp = with_lsr_derivative(m.arrays.pkc["sp"], "sweeps")
    run_sp = with_run_switches(m, m.arrays.replace(pkc={**m.arrays.pkc, "sp": exact_sp})).pkc["sp"]
    it = r.its[0]
    sf, ff, exf, st = O.inputs_at(m, r.ds, it, "I00_seaice_begin")
    op = m.arrays.pkc["op"]
    fn = L.dynsolver_fn(m, cfg=m.cfg)
    t, i = L.clock(m, it)[:2]
    rng = np.random.default_rng(36000)
    sz = m.cfg.size
    w = np.zeros(sf["UICE"].data.shape)
    sl = (Ellipsis, slice(sz.OLy, sz.OLy + sz.sNy), slice(sz.OLx, sz.OLx + sz.sNx))
    w[sl] = rng.standard_normal(w[sl].shape)
    wu, wv = jnp.asarray(w), jnp.asarray(np.roll(w, 1, axis=-1))

    def J(heff, uvel, sp, stage, u, v):
        rw = lambda a, d: type(a)(d, a.name, tiled=a.tiled, _dims=a.dims)   # noqa: E731  (traced: no numpy)
        sf2 = {**sf, "HEFF": rw(sf["HEFF"], heff)}
        st2 = st.replace(uVel=rw(st.uVel, uvel))
        out, _ = fn(sp, op, sf2, ff, exf, st2, t, i)
        return jnp.sum(wu * out[stage][u].data) + jnp.sum(wv * out[stage][v].data), out

    x = (sf["HEFF"].data, st.uVel.data)
    res = {}
    for name, sp in (("run", run_sp), ("exact", exact_sp)):
        (val, out), g = jax.value_and_grad(lambda h, u: J(h, u, sp, "I01_dynsolver", "UICE", "VICE"),
                                           argnums=(0, 1), has_aux=True)(*x)
        res[name] = dict(val=float(val), uIce=np.asarray(out["I01_dynsolver"]["UICE"].data),
                         g=[np.asarray(y) for y in g])
        _release()
    from mitjax.ops.fortran_minmax import MAX, MIN
    out0, _ = fn(exact_sp, op, sf, ff, exf, st, t, i)
    clip = lambda a: MAX(MIN(a, 0.40, p="b"), -0.40, p="a")   # noqa: E731  (clip_velocities, :401-404)
    pre = [out0["Y06_lsr"][n].data for n in ("UICE", "VICE")]
    post = [out0["I01_dynsolver"][n].data for n in ("UICE", "VICE")]
    res["clip_is_last"] = all(np.array_equal(np.asarray(clip(a)), np.asarray(b)) for a, b in zip(pre, post))
    mask = [jax.jvp(clip, (a,), (jnp.ones_like(a),))[1] for a in pre]
    res["n_clipped_under_w"] = int(np.sum((np.asarray(mask[0]) != 1.0) & (np.asarray(wu) != 0))
                                   + np.sum((np.asarray(mask[1]) != 1.0) & (np.asarray(wv) != 0)))
    _release()
    for name, (a, b) in (("fd", (wu, wv)), ("fd_clip", (wu * mask[0], wv * mask[1]))):
        def Jfd(h, u, a=a, b=b):
            sf2 = {**sf, "HEFF": type(sf["HEFF"])(h, sf["HEFF"].name, tiled=sf["HEFF"].tiled, _dims=sf["HEFF"].dims)}
            st2 = st.replace(uVel=type(st.uVel)(u, st.uVel.name, tiled=st.uVel.tiled, _dims=st.uVel.dims))
            o, _ = fn(exact_sp, op, sf2, ff, exf, st2, t, i)
            return jnp.sum(a * o["Y03_freedrift"]["uice_fd"].data) + jnp.sum(b * o["Y03_freedrift"]["vice_fd"].data)
        res[name] = dict(g=[np.asarray(y) for y in jax.grad(Jfd, argnums=(0, 1))(*x)])
        _release()
    return res


def test_kernel_switch_forward_identical(kern):
    assert kern["run"]["val"] == kern["exact"]["val"]
    assert np.array_equal(kern["run"]["uIce"].view(np.uint64), kern["exact"]["uIce"].view(np.uint64))


def _maxrel(g1, g2):
    return max(float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(b))), 1e-300) for a, b in zip(g1, g2))


def test_kernel_switch_effect(kern):
    """The switch changes the ice-dynamics gradient by O(1) (relative to max |g|); negative control: A1 vs A1 shows no
    change (the assertion would fail)."""
    d = _maxrel(kern["run"]["g"], kern["exact"]["g"])
    print("kernel: switch vs A1 max rel", d)
    assert d > 1e-2
    assert _maxrel(kern["exact"]["g"], kern["exact"]["g"]) == 0.0


def test_kernel_switch_is_free_drift(kern):
    """With the switch the gradient of <w, (uIce, vIce)> is that of <clip'(uIce_LSR) w, (uice_fd, vice_fd)>
    (seaice_dynsolver.F:312-313 in AD mode, then the clip :401-404 with TAF's mask from the stored pre-clip velocity
    :390-392), to rounding; A1's is not. Negative control: without the clip mask the reference misses (the clip binds
    under w: SEAICE_clipVelocities = .TRUE. in data.seaice)."""
    assert kern["clip_is_last"]                     # I01's UICE / VICE = the clip of Y06_lsr's: nothing in between
    assert kern["n_clipped_under_w"] > 0
    d = _maxrel(kern["run"]["g"], kern["fd_clip"]["g"])
    print("kernel: clipped points under w", kern["n_clipped_under_w"], "switch vs clip-masked free drift max rel", d,
          "vs unmasked", _maxrel(kern["run"]["g"], kern["fd"]["g"]),
          "A1 vs clip-masked free drift", _maxrel(kern["exact"]["g"], kern["fd_clip"]["g"]))
    assert d <= KERNEL_TOL
    assert _maxrel(kern["run"]["g"], kern["fd"]["g"]) > 1e-2
    assert _maxrel(kern["exact"]["g"], kern["fd_clip"]["g"]) > 1e-2
