"""pkg/mom_common/mom_init_fixed.F: fixed MOM_VISC.h fields, the viscosity length scales (MOM_INIT_FIXED)."""

import jax.numpy as jnp

from mitjax.farray import FArray, loop_i, loop_j
from mitjax.ops.safe import safe_div, safe_pow

_OPT = "MOM_COMMON_OPTIONS.h"       # mom_init_fixed.F:1


def mom_init_fixed(*, cfg, grid, params):
    """MOM_INIT_FIXED(myThid)   @63cdc0b pkg/mom_common/mom_init_fixed.F:8-235

    C     Initialize fixed quantities
    C     for momentum (common to fluxform & vecinv) packages

    Returns {name: value} of the MOM_VISC.h fields it sets: deepFacAdv(Nr) (/MOM_GRID_COPY/), L2_D, L2_Z, L3_D, L3_Z,
    L4rdt_D, L4rdt_Z (/MOM_VISC_LENGTH/), and viscAh_W, viscA4_W (/MOM_VISC_NH/, (i,j,k), #ifdef ALLOW_NONHYDROSTATIC).
    An initialisation routine, run once on the host (eagerly, as the grid builders). useAreaViscLength (:99, :123)
    and useNHMTerms (:50) are static; deltaTMom.NE.0 (:43) chooses between two values (a `where`, the division
    guarded). `L2**1.5` is a REAL exponent (`1.5` REAL*4, exact): pow, guarded at 0 (safe_pow; value pow(0, 1.5) = 0
    = the fill); `**2` integer_pow. `2. _d 0`, `0.03125 _d 0`, `1. _d 0` are double.

    Not ported, raise: MOM_USE_OLD_DEEP_VERT_ADV (:49-55), ALLOW_3D_VISCAH/VISCA4 (:61-80, :157-176), ALLOW_SMAG_3D
    (:142-151), ALLOW_BOTTOMDRAG_ROUGHNESS (:178-226) (all #undef in the M2 builds) and MOM_DIAGNOSTICS_INIT with
    useDiagnostics (:228-232; pkg/diagnostics is not ported). twoThird (:41) is read only under ALLOW_SMAG_3D.
    """
    for name in ("MOM_USE_OLD_DEEP_VERT_ADV", "ALLOW_3D_VISCAH", "ALLOW_3D_VISCA4", "ALLOW_SMAG_3D",
                 "ALLOW_BOTTOMDRAG_ROUGHNESS"):
        if cfg.cpp.flag(name, _OPT):
            raise NotImplementedError(f"MOM_INIT_FIXED: {name} is not ported")
    sNx, sNy, OLx, OLy, Nr = cfg.size.sNx, cfg.size.sNy, cfg.size.OLx, cfg.size.OLy, cfg.size.Nr
    rA, rAz = grid.rA, grid.rAz
    recip_dxF, recip_dyF, recip_dxV, recip_dyU = grid.recip_dxF, grid.recip_dyF, grid.recip_dxV, grid.recip_dyU
    deepFacC = grid.deepFacC
    p = params
    out = {}

    recip_dt = safe_div(1., p.deltaTMom, p.deltaTMom != 0., fill=1.)   # :42-43

    deepFacAdv = FArray(jnp.full((Nr,), jnp.nan), "deepFacAdv", k=(1, Nr), tiled=False)
    for k in range(1, Nr+1):                                        # :46-48
        deepFacAdv = deepFacAdv.at[k].set(1.)
    if p.useNHMTerms:                                               # :50-54
        for k in range(1, Nr+1):
            deepFacAdv = deepFacAdv.at[k].set(deepFacC[k])
    out["deepFacAdv"] = deepFacAdv

    j = loop_j(1-OLy, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx)
    if cfg.cpp.flag("ALLOW_NONHYDROSTATIC", _OPT):                 # :81-91
        T = rA.data.shape[0]
        shp = (T, Nr, sNy+2*OLy, sNx+2*OLx)
        b3 = dict(i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy), k=(1, Nr))
        viscAh_W = FArray(jnp.full(shp, jnp.nan), "viscAh_W", **b3)
        viscA4_W = FArray(jnp.full(shp, jnp.nan), "viscA4_W", **b3)
        for k in range(1, Nr+1):
            viscAh_W = viscAh_W.at[i, j, k].set(p.viscAhW)
            viscA4_W = viscA4_W.at[i, j, k].set(p.viscA4W)
        out["viscAh_W"], out["viscA4_W"] = viscAh_W, viscA4_W

#--   Length scales at divergence (tracer) points (:93-116)
    L2_D = rA.local("L2_D").at[i, j].set(rA[i, j])                  # :94-98
    if not p.useAreaViscLength:                                     # :99-109
        a, b = recip_dxF[i, j], recip_dyF[i, j]
        c = (a != 0.) | (b != 0.)
        L2_D = L2_D.at[i, j].set(jnp.where(c, safe_div(2., (a**2+b**2), c), L2_D[i, j]))
    L3_D = rA.local("L3_D").at[i, j].set(safe_pow(L2_D[i, j], 1.5, L2_D[i, j] > 0.))   # :110-116
    L4rdt_D = rA.local("L4rdt_D").at[i, j].set(0.03125*recip_dt
                                               *L2_D[i, j]**2)

#--   Length scales at vorticity points (:118-140)
    L2_Z = rA.local("L2_Z").at[i, j].set(rAz[i, j])                 # :118-122
    if not p.useAreaViscLength:                                     # :123-133
        a, b = recip_dxV[i, j], recip_dyU[i, j]
        c = (a != 0.) | (b != 0.)
        L2_Z = L2_Z.at[i, j].set(jnp.where(c, safe_div(2., (a**2+b**2), c), L2_Z[i, j]))
    L3_Z = rA.local("L3_Z").at[i, j].set(safe_pow(L2_Z[i, j], 1.5, L2_Z[i, j] > 0.))   # :134-140
    L4rdt_Z = rA.local("L4rdt_Z").at[i, j].set(0.03125*recip_dt
                                               *L2_Z[i, j]**2)
    out.update(L2_D=L2_D, L3_D=L3_D, L4rdt_D=L4rdt_D, L2_Z=L2_Z, L3_Z=L3_Z, L4rdt_Z=L4rdt_Z)

    if cfg.cpp.flag("ALLOW_DIAGNOSTICS", _OPT) and p.useDiagnostics:      # :228-232
        raise NotImplementedError("MOM_INIT_FIXED: MOM_DIAGNOSTICS_INIT (pkg/diagnostics is not ported)")
    return out
