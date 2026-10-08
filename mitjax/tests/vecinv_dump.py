"""In-dump gate of MOM_VECINV (VECINV lane, coordinator 2026-10-01): the tendency stage `D00c_mom_vecinv` (gU, gV,
guDissip, gvDissip of level k right after MOM_VECINV, dynamics.F:527) of lane A's dumps-on runs (`jdon`,
reference/reference_runs.py) recomputed by mitjax/pkg/mom_vecinv from the dumped model state and geometry.

Inputs, each from the oracle's own dump of the same iteration (nothing synthetic):
  * geometry: stage G00_geometry (GRID.h, incl. recip_hFacW/S, deep/rho factors, cosFac) and S00_begin
    (hFacC/W/S, recip_hFacC: under NONLIN_FRSURF the current ones);
  * state: uVel, vVel, wVel at S00_begin (DYNAMICS reads them unchanged: nothing between S00_begin and
    dynamics.F:527 writes them);
  * DYNAMICS' own set-up (dynamics.F:296-319): gU = gV = 0 on every level before the k loop only #ifdef ALLOW_AUTODIFF
    (:296-305; without it the points MOM_VECINV does not write keep the gU/gV of S00_begin: measured on cs32x15, where
    taking 0 there left 40951 halo points of gU wrong from the second step on), fVerU/fVerV(kUp, kDown) = 0 always
    (:306-309), so fVerUkm of level 1 is 0 and of level k the fVerUkp of level k-1 (points outside iMin..iMax stay 0);
    kappaRU/RV from CALC_VISCOSITY (mitjax/model/src/calc_viscosity.py) with viscArNr(k) = viscAr (ini_parms.F:545,
    the run's data); MOM_VISC.h from MOM_INIT_FIXED (mitjax/pkg/mom_common/mom_init_fixed.py) on the dumped grid;
    recip_rhoFacC = 1/rhoFacC (set_ref_state.F:379; 1 at :75 when rhoFacC is not set: the same value 1 here);
  * PARAMS.h: the run's own files through ini_parms_dyn with its VECINV arm (vecinv_replay.run_params); where
    ini_parms_dyn is not ported yet (the cube runs) the harness's record of the run's PARAMS.h with the VECINV arm over
    it; the W2 tile view from lane B's W2 set-up.
"""

import sys
from pathlib import Path

import numpy as np

import jax.numpy as jnp

from mitjax.farray import FArray
from mitjax.tests import vecinv_replay as vr

REPO = Path(__file__).resolve().parents[2]

STAGE = "D00c_mom_vecinv"
GEOM3 = ("maskC", "maskW", "maskS", "recip_hFacW", "recip_hFacS")
GEOM3_S00 = ("hFacC", "hFacW", "hFacS", "recip_hFacC")
GEOM3_NLFS = ("h0FacW", "h0FacS")
GEOM2 = ("rA", "rAw", "rAs", "rAz", "recip_rA", "recip_rAw", "recip_rAs", "recip_rAz", "dxC", "dyC", "dxG", "dyG",
         "dxV", "dyU", "recip_dxC", "recip_dyC", "recip_dxG", "recip_dyG", "recip_dxV", "recip_dyU", "recip_dxF",
         "recip_dyF", "fCoriG", "fCoriCos", "angleCosC", "angleSinC")
GEOMY = ("cosFacU", "cosFacV", "sqCosFacU", "sqCosFacV")
GEOMR = ("drF", "recip_drF", "recip_deepFacC", "recip_deepFac2C", "deepFacC", "deepFac2C", "rhoFacC")
GEOMRP1 = ("recip_drC", "deepFac2F", "rhoFacF", "rVel2wUnit")


def _rr():
    if str(REPO / "reference") not in sys.path:
        sys.path.insert(0, str(REPO / "reference"))
    import reference_runs as rr
    return rr


def dumpset(exp, inp):
    from mitjax.io.dump import DumpSet
    rr = _rr()
    return DumpSet(rr.run_top(rr.find_run(exp, inp, "jdon")) / "dumps")


def _vert(ds, it, name):
    """A kind-V record (vertical profile, every point the same per level) as its 1-D values."""
    f = ds.field(it, "G00_geometry", name)
    v = f[0, :, 0, 0]
    assert np.all(f == v[None, :, None, None]), name
    return v


def grid_params_visc(R, ds, it):
    """(Grid, Params, visc) of iteration `it` from the dumps (see the module docstring)."""
    from mitjax.model.grid import Grid
    from mitjax.pkg.mom_common.mom_init_fixed import mom_init_fixed
    Nr = R.Nr
    b2 = vr._b(R)
    b3 = dict(b2, k=(1, Nr))
    f = {}
    for n in GEOM3:
        f[n] = FArray(jnp.asarray(ds.field(it, "G00_geometry", n)), n, **b3)
    for n in GEOM3_S00:
        f[n] = FArray(jnp.asarray(ds.field(it, "S00_begin", n)), n, **b3)
    if R.cfg.cpp.flag("NONLIN_FRSURF", "MOM_COMMON_OPTIONS.h"):
        for n in GEOM3_NLFS:
            f[n] = FArray(jnp.asarray(ds.field(it, "G00_geometry", n)), n, **b3)
    for n in GEOM2:
        f[n] = FArray(jnp.asarray(ds.field(it, "G00_geometry", n)[:, 0]), n, **b2)
    for n in GEOMY:
        a = ds.field(it, "G00_geometry", n)[:, 0]                   # (j) profile per tile, constant along i
        assert np.all(a == a[:, :, :1]), n
        f[n] = FArray(jnp.asarray(a[:, :, 0]), n, j=b2["j"])
    one = {n: _vert(ds, it, n) for n in GEOMR + GEOMRP1}
    for n in GEOMR:
        if n != "rhoFacC":
            f[n] = FArray(jnp.asarray(one[n]), n, k=(1, Nr), tiled=False)
    for n in ("recip_drC", "deepFac2F"):
        f[n] = FArray(jnp.asarray(one[n]), n, k=(1, Nr+1), tiled=False)
    f["rkSign"] = jnp.float64(_vert(ds, it, "rkSign")[0])
    f["gravitySign"] = jnp.float64(_vert(ds, it, "gravitySign")[0])
    grid = Grid(f)
    params, source = vr.run_params(R)
    if source != "ini_parms_dyn":                                   # the harness record holds overridden factors
        params = params.replace(traced={
            "recip_rhoFacC": FArray(jnp.asarray(1. / one["rhoFacC"]), "recip_rhoFacC", k=(1, Nr), tiled=False),
            "rhoFacF": FArray(jnp.asarray(one["rhoFacF"]), "rhoFacF", k=(1, Nr+1), tiled=False),
            "rVel2wUnit": FArray(jnp.asarray(one["rVel2wUnit"]), "rVel2wUnit", k=(1, Nr+1), tiled=False)})
    visc = vr.CommonBlock(**{n: v for n, v in mom_init_fixed(cfg=R.cfg, grid=grid, params=params).items()
                             if n in ("L2_D", "L2_Z", "L3_D", "L3_Z", "L4rdt_D", "L4rdt_Z", "deepFacAdv")})
    return grid, params, visc


def viscArNr(R):
    """viscArNr(k) = viscAr (ini_parms.F:545; the run's data PARM01 viscAr, else set_defaults.F's 0): used only where
    run_params falls back to the harness record (ini_parms_dyn's own viscArNr otherwise)."""
    v = R.exp.params.values.get("data:parm01:viscar", np.float64(0.))
    return FArray(jnp.full((R.Nr,), np.float64(v)), "viscArNr", k=(1, R.Nr), tiled=False)


def state_inputs(R, ds, it):
    return {n: jnp.asarray(ds.field(it, "S00_begin", n)) for n in ("uVel", "vVel", "wVel", "gU", "gV")}


def dynamics_fn(R, mods, w2):
    """f(fields, grid, params, visc) -> {name: [tile, k, j, i]} of gU, gV, guDissip, gvDissip after MOM_VECINV at every
    level, with DYNAMICS' own set-up (module docstring)."""
    from mitjax.model.src.calc_viscosity import calc_viscosity
    from mitjax.model.state import State
    sNx, sNy, OLx, OLy, Nr = (R.size[k] for k in ("sNx", "sNy", "OLx", "OLy", "Nr"))
    b2 = dict(i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy))
    b3 = dict(b2, k=(1, Nr))
    bk1 = dict(b2, k=(1, Nr+1))
    iMin, iMax, jMin, jMax = 0, sNx+1, 0, sNy+1                      # dynamics.F:191-192

    def f(fields, grid, params, visc):
        T = fields["uVel"].shape[0]
        zero2 = FArray(jnp.zeros((T, sNy+2*OLy, sNx+2*OLx)), "zero", **b2)
        if R.cfg.cpp.flag("ALLOW_AUTODIFF"):                       # dynamics.F:296-305
            gU0 = gV0 = FArray(jnp.zeros((T, Nr, sNy+2*OLy, sNx+2*OLx)), "gU", **b3)
        else:
            gU0, gV0 = FArray(fields["gU"], "gU", **b3), FArray(fields["gV"], "gV", **b3)
        kap = FArray(jnp.full((T, Nr+1, sNy+2*OLy, sNx+2*OLx), jnp.nan), "kappaRU", **bk1)
        p = params
        if "viscArNr" not in params.traced_items():                # harness-record fallback (cube runs)
            p = params.replace(traced={"viscArNr": viscArNr(R)}, static={"interViscAr_pCell": False,
                                                                         "pCellMix_select": 0})
        kappaRU, kappaRV = calc_viscosity(iMin, iMax, jMin, jMax, kap, kap, cfg=R.cfg, params=p)
        state = State({"uVel": FArray(fields["uVel"], "uVel", **b3), "vVel": FArray(fields["vVel"], "vVel", **b3),
                       "wVel": FArray(fields["wVel"], "wVel", **b3), "gU": gU0, "gV": gV0})
        fUm, fVm = zero2, zero2                                     # fVerU/V(kUp) = 0 (dynamics.F:306-309)
        out = {n: [] for n in ("gU", "gV", "guDissip", "gvDissip")}
        for k in range(1, Nr+1):
            fUp, fVp, guD, gvD, state = mods["mom_vecinv"].mom_vecinv(
                k, iMin, iMax, jMin, jMax, kappaRU, kappaRV, fUm, fVm, zero2, zero2, zero2, zero2, 0., 0,
                cfg=R.cfg, grid=grid, params=p, state=state, visc=visc, w2=w2, ctrlf=vr.ctrlf_for(R.cfg, zero2))
            fUm, fVm = fUp, fVp
            out["gU"].append(state.gU.data[:, k-1])
            out["gV"].append(state.gV.data[:, k-1])
            out["guDissip"].append(guD.data)
            out["gvDissip"].append(gvD.data)
        return {n: jnp.stack(a, axis=1) for n, a in out.items()}

    return f


def reference(R, ds, it):
    """{name: [tile, k, j, i]} of the D00c records of iteration `it`."""
    return {n: np.concatenate([ds.field(it, STAGE, f"{n}_k{k:03d}") for k in range(1, R.Nr+1)], axis=1)
            for n in ("gU", "gV", "guDissip", "gvDissip")}
