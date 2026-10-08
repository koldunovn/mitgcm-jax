"""global_ocean.cs32x15/input_ad.seaice: the reverse pass of EXF_WIND + EXF_BULKFORMULAE's solve4Stress = .FALSE. arm
is finite where the wind stress is zero (lane M4ADCS32ICE session 2; plan rule "masked/halo/padding lanes compute
finite values", mitjax/ops/safe.py).

EXF_BULKFORMULAE without ALLOW_BULK_LARGEYEAGER04 (code_ad) takes ustar = SQRT(wStress/atmrho), tau =
SQRT(wStress*atmrho) (exf_bulkformulae.F:331, :333) inside `IF ( atemp .NE. 0. )` (:283). The port computes the arm on
every lane and selects it with `where(atemp /= 0)`; unguarded, the discarded roots on an atemp = 0 lane with wStress = 0
have the derivative 1/(2 SQRT(0)) = inf and the zero cotangent of the discarded branch gives 0*inf = NaN. The guard is
EXF_WIND's own AD guard of the same root (`IF ( wStress .LE. 0. ) ustar = 0.`, exf_wind.F:194-199): mask wStress > 0,
fill +0. = SQRT(+0.); EXF_WIND's wStress = SQRT(usSq) is guarded by its `IF ( usSq .NE. 0. )` (:170-178).

The live fixture (lane A's dumps job27855988-jdon, iteration 36000) has wStress > 0 on every interior lane (min
1.3e-3) and atemp = 0 on 1724 of the 6144 interior lanes, so the hazard is latent here; it is planted: the C-grid stress
around atemp = 0 lanes set to 0 (ustress at i, i+1 and vstress at j, j+1: usSq = 0, :158-165). The cost is the sum of
EXF_BULKFORMULAE's hs, hl, evap over the interior.
* forward: the kernels on the fixture give the oracle's X03 wStress and X04 hs, hl, evap bit for bit (interior), the
  guard changes no value (guarded vs unguarded bit for bit), and the planted zeros change hs, hl, evap only where
  they change a neighbour's wStress (the planted cells sit where atemp = 0);
* the gradient with respect to the stress fields (the path of the xx_fu / xx_fv controls, through EXF_WIND) and with
  respect to EXF_FIELDS.h's wStress (EXF_BULKFORMULAE alone) is finite on every lane;
* negative controls: EXF_BULKFORMULAE's roots unguarded -> NaN in d/d(wStress) (measured: the stress path stays
  finite, EXF_WIND's own `where` discards the NaN cotangent at usSq = 0 -- so the hazard was latent in this build);
  EXF_WIND's root unguarded -> NaN in d/d(ustress, vstress).
Not covered, by construction of the Fortran: wStress = 0 on an atemp /= 0 lane is singular in the Fortran forward
itself (huol = ... /(ustar*ustar), :373-375), so no guard can make its derivative finite without changing a forward
value (a candidate in docs/ISSUES_UPSTREAM.md).
About 1-2 min on a CPU node (Model set-up from the oracle run directory, two small jitted kernels per case).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mitjax.xla_flags import set_gate_xla_flags

set_gate_xla_flags()

from mitjax.tests import m4adcs32ice_gate as A  # noqa: E402

IT = 36000
NPLANT = 8                      # planted zero-stress cells (atemp = 0 interior lanes, spread over tiles)


@pytest.fixture(scope="module")
def fx():
    from mitjax.farray import FArray
    from mitjax.pkg.exf.exf_getforcing import exf_tsf
    from mitjax.tests import m4cs32ice_gate as G
    from mitjax.tests import m4off_gate as O
    r = A.run("seaice")
    m = r.m
    # EXF_FIELDS.h and FFIELDS.h (gcmSST) as the oracle has them before X03 (m4cs32ice_gate.before); the State is
    # the Model's initial one (read here only for useRelativeWind / sstExtrapol, both off in this run)
    f = dict(m.pk0["exf"])
    for n, v in f.items():
        if hasattr(v, "data") and v.data.dtype != jnp.int32:
            a = G.before(r.ds, IT, "X03_exf_wind", n)
            if a is not None:
                f[n] = O.like(v, a)
    # gcmSST = theta(k = ks) (load_fields_driver.F:178, at the start of the step): S00_begin's theta, level 1
    th = np.asarray(r.ds.field(IT, "S00_begin", "theta"))
    ff = m.ff0.replace(gcmSST=O.like(m.ff0.gcmSST, th[:, 0]))
    st = m.state0
    assert not m.arrays.pkc["exfp"].useRelativeWind
    exfp = m.arrays.pkc["exfp"]
    Tsf = exf_tsf(f, cfg=m.cfg, exf=exfp, grid=m.grid, params=m.params, state=st, ff=ff)
    sz = m.cfg.size
    o = sz.OLx
    at = np.asarray(f["atemp"].data)
    off = np.argwhere(at[:, o:o + sz.sNy, o:o + sz.sNx] == 0)
    sel = off[np.linspace(0, len(off) - 1, NPLANT).astype(int)]
    us, vs = np.asarray(f["ustress"].data).copy(), np.asarray(f["vstress"].data).copy()
    for t, jj, ii in sel:
        j, i = jj + o, ii + o
        us[t, j, i] = us[t, j, i + 1] = 0.
        vs[t, j, i] = vs[t, j + 1, i] = 0.
    planted = (jnp.asarray(us), jnp.asarray(vs))
    clean = (f["ustress"].data, f["vstress"].data)

    def J(uv, wind=None, bulk=None):
        import mitjax.pkg.exf.exf_bulkformulae as EB
        from mitjax.pkg.exf.exf_wind import exf_wind
        g = dict(f)
        g["ustress"] = FArray(uv[0], f["ustress"].name, tiled=f["ustress"].tiled, _dims=f["ustress"].dims)
        g["vstress"] = FArray(uv[1], f["vstress"].name, tiled=f["vstress"].tiled, _dims=f["vstress"].dims)
        # the run's gentim2d controls (xx_qnet, xx_empmr, xx_fu, xx_fv) have no xx_wspeed file: EXF_WIND adds none
        g = exf_wind(jnp.float64(0.), IT, g, cfg=m.cfg, exf=exfp, params=m.params, state=st,
                     ctrl=dict(xx_gentim2d={}, files={}))
        g = EB.exf_bulkformulae(Tsf, jnp.float64(0.), IT, g, cfg=m.cfg, exf=exfp, params=m.params, state=st)
        sl = (slice(None), slice(o, o + sz.sNy), slice(o, o + sz.sNx))
        return (jnp.sum(g["hs"].data[sl]) + jnp.sum(g["hl"].data[sl]) + jnp.sum(g["evap"].data[sl])), g

    def Jw(wst, g0):
        """EXF_BULKFORMULAE alone, as a function of EXF_FIELDS.h's wStress (g0: EXF_WIND's outputs)."""
        import mitjax.pkg.exf.exf_bulkformulae as EB
        g = dict(g0)
        g["wStress"] = FArray(wst, g0["wStress"].name, tiled=g0["wStress"].tiled, _dims=g0["wStress"].dims)
        g = EB.exf_bulkformulae(Tsf, jnp.float64(0.), IT, g, cfg=m.cfg, exf=exfp, params=m.params, state=st)
        sl = (slice(None), slice(o, o + sz.sNy), slice(o, o + sz.sNx))
        return jnp.sum(g["hs"].data[sl]) + jnp.sum(g["hl"].data[sl]) + jnp.sum(g["evap"].data[sl])

    yield dict(r=r, m=m, J=J, Jw=Jw, planted=planted, clean=clean, sel=sel, o=o, sz=sz)
    jax.clear_caches()


def _grad(fx, uv):
    return jax.jit(jax.grad(lambda x: fx["J"](x)[0]))(uv)


def test_fixture_has_latent_hazard(fx):
    """The planted cells are atemp = 0 interior lanes where the clean wStress is > 0 (the hazard is latent)."""
    r = fx["r"]
    o = fx["o"]
    ws = r.ds.field(IT, "X03_exf_wind", "wStress")[..., o:-o, o:-o]
    assert np.all(ws > 0) and len(fx["sel"]) == NPLANT


def _fwd(fx, uv):
    return jax.jit(lambda x: fx["J"](x))(uv)[1]


def _bits(a):
    return np.asarray(a.data).view(np.uint64)


def _grad_w(fx):
    """d(sum hs + hl + evap)/d(wStress) of EXF_BULKFORMULAE alone at EXF_WIND's planted outputs."""
    g0 = _fwd(fx, fx["planted"])
    return np.asarray(jax.jit(jax.grad(lambda w: fx["Jw"](w, g0)))(g0["wStress"].data))


def test_forward_bitwise_guard_and_planted(fx, monkeypatch):
    """The fixture is the oracle's step (X03 wStress, X04 hs, hl, evap bit for bit, interior); the guard changes no
    value (guarded vs unguarded kernels bit for bit, clean and planted inputs); with the planted zeros hs, hl, evap
    change only where wStress changes (the planted faces feed the neighbours' usSq), and wStress is +0. at the
    planted cells (atemp = 0 lanes)."""
    import mitjax.pkg.exf.exf_bulkformulae as EB
    o = fx["o"]
    g, gp = _fwd(fx, fx["clean"]), _fwd(fx, fx["planted"])
    monkeypatch.setattr(EB, "safe_sqrt", lambda x, mask, fill=0.0: jnp.sqrt(x))
    jax.clear_caches()
    gu, gpu = _fwd(fx, fx["clean"]), _fwd(fx, fx["planted"])
    for n in ("hs", "hl", "evap", "wStress"):
        assert np.array_equal(_bits(g[n]), _bits(gu[n])) and np.array_equal(_bits(gp[n]), _bits(gpu[n])), n
    same = _bits(gp["wStress"]) == _bits(g["wStress"])        # the planted faces also feed the neighbours' usSq
    for n in ("hs", "hl", "evap"):
        assert np.array_equal(_bits(gp[n])[same], _bits(g[n])[same]), n
    for st_, n in (("X03_exf_wind", "wStress"), ("X04_exf_bulkformulae", "hs"), ("X04_exf_bulkformulae", "hl"),
                   ("X04_exf_bulkformulae", "evap")):            # the fixture is the oracle's step (interior)
        want = np.asarray(fx["r"].ds.field(IT, st_, n))[..., o:-o, o:-o].reshape(-1)
        got = np.asarray(g[n].data)[..., o:-o, o:-o].reshape(-1)
        assert np.array_equal(want.view(np.uint64), got.view(np.uint64)), (st_, n)
    ws = np.asarray(gp["wStress"].data)
    for t, jj, ii in fx["sel"]:
        v = ws[t, jj + o, ii + o]
        assert v == 0.0 and not np.signbit(v)


def test_gradient_finite_with_zero_stress(fx):
    """With the planted zero-stress cells: d(sum hs + hl + evap)/d(ustress, vstress) through EXF_WIND +
    EXF_BULKFORMULAE and d/d(wStress) of EXF_BULKFORMULAE alone are finite on every lane, and nonzero somewhere."""
    gu, gv = _grad(fx, fx["planted"])
    gw = _grad_w(fx)
    for x in (gu, gv, gw):
        x = np.asarray(x)
        assert np.all(np.isfinite(x)), np.argwhere(~np.isfinite(x))[:5]
    assert np.count_nonzero(np.asarray(gu)) > 0 and np.count_nonzero(gw) > 0


def test_control_bulkformulae_unguarded(fx, monkeypatch):
    """Negative control: EXF_BULKFORMULAE's two roots unguarded (jnp.sqrt, as before session 2) -> NaN in
    d/d(wStress) at the planted cells. (Measured and printed: d/d(ustress, vstress) stays finite then, because
    EXF_WIND's own guard -- a `where` on usSq /= 0 -- discards the NaN cotangent of wStress at those cells.)"""
    import mitjax.pkg.exf.exf_bulkformulae as EB
    monkeypatch.setattr(EB, "safe_sqrt", lambda x, mask, fill=0.0: jnp.sqrt(x))
    jax.clear_caches()
    gw = _grad_w(fx)
    gu, gv = _grad(fx, fx["planted"])
    bad = int(np.count_nonzero(~np.isfinite(gw)))
    print("unguarded EXF_BULKFORMULAE: non-finite d/d(wStress) lanes", bad, "; d/d(ustress, vstress):",
          int(np.count_nonzero(~np.isfinite(np.asarray(gu)))) + int(np.count_nonzero(~np.isfinite(np.asarray(gv)))))
    assert bad > 0
def test_control_wind_unguarded(fx, monkeypatch):
    """Negative control: EXF_WIND's wStress = SQRT(usSq) unguarded -> NaN in the gradient at the planted cells
    (EXF_WIND's guard is load-bearing on this path too)."""
    import mitjax.pkg.exf.exf_wind as EW
    monkeypatch.setattr(EW, "safe_sqrt", lambda x, mask, fill=0.0: jnp.sqrt(x))
    jax.clear_caches()
    gu, gv = _grad(fx, fx["planted"])
    bad = int(np.count_nonzero(~np.isfinite(np.asarray(gu)))) + int(np.count_nonzero(~np.isfinite(np.asarray(gv))))
    print("unguarded EXF_WIND: non-finite gradient lanes", bad)
    assert bad > 0
