"""GAD_IMPLICIT_R's implicit vertical advection (gad_implicit_r.F:162-259; M1 lane GO, session 3) with
GAD_DST2U1_IMPL_R (ENUM_DST2, global_ocean.90x40x15/input_ad: tempVertAdvScheme = saltVertAdvScheme = 20,
temp/saltImplVertAdv) and the tri-diagonal solve, teacher-forced from the input_ad oracle dumps (kind jdon, steps
0-2): T12/T22 (gT_loc, gS_loc after TIMESTEP_TRACER), T13/T23 kappaRk, T01 wFld (THERMODYNAMICS' residual
transport, GM bolus included) -> GAD_IMPLICIT_R -> T13/T23 bitwise on every point.

recip_hFacNew is built here by the formula of the arm input_ad runs (thermodynamics.F:233-243: nonlinFreeSurf = 2,
select_rStar = 0: 1/hFac_surfC at kSurfC, recip_hFacC elsewhere; hFac_surfC as CALC_SURF_DR leaves it at the end of
the step = G00 of the next step); THERMODYNAMICS' own arm is the M2 lane's. The penta-diagonal path
(SOLVE_PENTADIAGONAL with GAD_U3C4_IMPL_R) and the flux-limiter path (GAD_FLUXLIMIT_IMPL_R) are gated in-model by
advect_xz/input.nlfs (test_r3_nlfs.py). The exchange maps of input_ad's exch1 layout (4 tiles of 45x20) are decoded
in memory from the run's own exchange probe (go_gate.setup)."""

import jax

from mitjax.farray import loops_kji
from mitjax.tests import go_gate as G

EXP = ("global_ocean.90x40x15", "input_ad")


def _run(s, it, scale_w=1.):
    from mitjax.pkg.generic_advdiff.gad_h import GAD_SALINITY, GAD_TEMPERATURE
    from mitjax.pkg.generic_advdiff.gad_implicit_r import gad_implicit_r
    cfg = s.cfg
    sz = cfg.size
    g = s.grid
    nxt = G.dumped(s, it + 1, "G00_geometry")
    st = G.teacher_state(s, it, [("S00_begin", None)])
    hs = G._farr(nxt["hFac_surfC"], st.etaN.local("hFac_surfC"))
    k, j, i = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
    rh = st.theta.local("recip_hFacNew").at[i, j, k].set(0.)
    import jax.numpy as jnp
    kk = jnp.arange(1, sz.Nr+1)[None, :, None, None]
    ks = g.kSurfC.data[:, None]
    hsd = hs.data[:, None]
    at_s = kk == ks
    rh = rh.__class__(jnp.where(at_s, 1./jnp.where(at_s, hsd, 1.), g.recip_hFacC.data), rh.name, tiled=True,
                      _dims=rh.dims)                                   # thermodynamics.F:236-240
    t, i_ = G.counters(s, it)
    out = {}
    w = G._farr(G.dumped(s, it, "T01_residual_flow")["wFld"], st.wVel.local("wFld"))
    w = w.__class__(w.data*scale_w, w.name, tiled=True, _dims=w.dims)
    for tid, fld, pre, post, name, impl, scheme in (
            (GAD_TEMPERATURE, "theta", "T12_temp_step", "T13_temp_impl", "gT_loc", s.params.tempImplVertAdv,
             s.params.tempVertAdvScheme),
            (GAD_SALINITY, "salt", "T22_salt_step", "T23_salt_impl", "gS_loc", s.params.saltImplVertAdv,
             s.params.saltVertAdvScheme)):
        d12, d13 = G.dumped(s, it, pre), G.dumped(s, it, post)
        gl = G._farr(d12[name], st.theta.local("g_loc"))
        kap = G._farr(d13["kappaRk"], st.theta.local("kappaRk"))
        f = jax.jit(lambda kap, rh, w, tr, gl, p, t, i_: gad_implicit_r(
            impl, scheme, tid, p.dTtracerLev, kap, rh, w, tr, gl, t, i_, cfg=cfg, grid=g, params=p))
        res = f(kap, rh, w, getattr(st, fld), gl, s.params, t, i_)
        out[post] = G.compare(res.data, d13[name])
    return out


def test_dst2_implicit_vertical_advection_bitwise():
    """input_ad steps 0-2: T13 and T23 bitwise (implicit DST2 vertical advection + implicit diffusion); negative
    control: wFld scaled by (1 + 2^-40) bites."""
    s = G.setup(*EXP, kind="jdon")
    assert s.params.tempImplVertAdv and s.params.tempVertAdvScheme == 20
    for it in s.its[:-1]:
        r = _run(s, it)
        assert all(not any(v[1:]) for v in r.values()), (it, r)
    r = _run(s, s.its[0], scale_w=1. + 2.**-40)
    assert all(v[3] > 100 for v in r.values()), r
