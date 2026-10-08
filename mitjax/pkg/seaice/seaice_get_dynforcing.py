"""SEAICE_GET_DYNFORCING: pkg/seaice/seaice_get_dynforcing.F @63cdc0b (lane M4OFF, the C-grid build)."""

import jax.numpy as jnp

from mitjax.farray import FArray, loop_i, loop_j
from mitjax.model.grid import deg2rad
from mitjax.ops.safe import safe_sqrt
from mitjax.pkg.seaice.seaice_params_h import ZERO


def seaice_get_dynforcing(uIce, vIce, AREA, SIMaskU, SIMaskV, TAUX, TAUY, myTime, myIter, *, cfg, sp, exfp, exf,
                          grid, op, ff=None):
    """SEAICE_GET_DYNFORCING( uIce, vIce, AREA, SIMaskU, SIMaskV, taux, tauy, myTime, myIter, myThid )
    @63cdc0b pkg/seaice/seaice_get_dynforcing.F:9-281

    C     | o compute surface stress from atmopheric forcing fields

    `exf` EXF_FIELDS.h (uwind, vwind), `exfp` ExfParams (useAtmWind, useRelativeWind), `grid` (fCori, yC), `ff`
    FFIELDS.h (fu, fv: read by the `ELSE` arm only). Returns (TAUX, TAUY): the ice surface stress at U and V points;
    fu/fv are not written in this build.
    Ported: ALLOW_EXF with SEAICE_EXTERNAL_FLUXES (:110-143, the open-ocean stress, not compiled) and useEXF with
    useAtmWind (:145-206), useRelativeWind = .FALSE. (:157-167 raises); the `ELSE` arm :207-237 without
    HACK_FOR_GMAO_CPL (lane M4CS32ICE: useAtmWind = .FALSE., the stress from fu/fv scaled by SEAICE_drag(_south) /
    OCEAN_drag with the yC < 0 choice at the C point, every point :221-234; HACK_FOR_GMAO_CPL raises when compiled).
    usingPCoords (ks, read by nothing else here) raises. Diagnostics (:244-279,
    useDiagnostics) are output only. The (i,j) loops are independent per point: vectorised; the pointwise IFs are
    wheres with both arms finite (the SQRT guarded on its own arm). SIGN(a,b) is gfortran's copysign; SIN, COS of the
    turning angle are XLA's (as OSTRES)."""
    if not cfg.cpp.flag("ALLOW_EXF") or not cfg.cpp.flag("SEAICE_EXTERNAL_FLUXES", "SEAICE_OPTIONS.h"):
        raise NotImplementedError("SEAICE_GET_DYNFORCING: only ALLOW_EXF with SEAICE_EXTERNAL_FLUXES is ported")
    if op.usingPCoords:                                                        # :98
        raise NotImplementedError("SEAICE_GET_DYNFORCING: usingPCoords is not ported")
    if not (cfg.use_flag("useEXF") and exfp.useAtmWind):                       # :145-149
        return _stress_from_fu_fv(SIMaskU, SIMaskV, TAUX, TAUY, cfg=cfg, sp=sp, grid=grid, ff=ff)   # :207-237
    if exfp.useRelativeWind:                                                   # :157-167
        raise NotImplementedError("SEAICE_GET_DYNFORCING: useRelativeWind (:157-167) is not ported")
    del uIce, vIce, AREA
    sz = cfg.size
    OLx, OLy, sNx, sNy = sz.OLx, sz.OLy, sz.sNx, sz.sNy
    SINWIN = jnp.sin(sp.SEAICE_airTurnAngle*deg2rad)                           # :100
    COSWIN = jnp.cos(sp.SEAICE_airTurnAngle*deg2rad)                           # :101
    j = loop_j(1-OLy, sNy+OLy)                                                 # :151-156
    i = loop_i(1-OLx, sNx+OLx)
    uTmp = exf["uwind"][i, j]                                                  # :153
    vTmp = exf["vwind"][i, j]                                                  # :154
    AAA = uTmp**2+vTmp**2                                                      # :171
    AAA = jnp.where(AAA <= sp.SEAICE_EPS_SQ,                                   # :172-176
                    sp.SEAICE_EPS,
                    safe_sqrt(AAA, AAA > sp.SEAICE_EPS_SQ))
    CDAIR = jnp.where(grid.yC[i, j] < ZERO,                                    # :177-181
                      sp.SEAICE_rhoAir*sp.SEAICE_drag_south*AAA,
                      sp.SEAICE_rhoAir*sp.SEAICE_drag*AAA)
    dims = dict(i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy))
    uTmp = FArray(uTmp, "uTmp", **dims)
    vTmp = FArray(vTmp, "vTmp", **dims)
    CDAIR = FArray(CDAIR, "CDAIR", **dims)
    fCori = grid.fCori
    j = loop_j(1-OLy+1, sNy+OLy)                                               # :184-205
    i = loop_i(1-OLx+1, sNx+OLx)
    TAUX = TAUX.at[i, j].set(0.5*(                                             # :187-194
        CDAIR[i, j]*(
            COSWIN*uTmp[i, j]
            - jnp.copysign(SINWIN, fCori[i, j])*vTmp[i, j])
        + CDAIR[i-1, j]*(
            COSWIN*uTmp[i-1, j]
            - jnp.copysign(SINWIN, fCori[i-1, j])*vTmp[i-1, j])
    )*SIMaskU[i, j])
    TAUY = TAUY.at[i, j].set(0.5*(                                             # :196-203
        CDAIR[i, j]*(
            jnp.copysign(SINWIN, fCori[i, j])*uTmp[i, j]
            + COSWIN*vTmp[i, j])
        + CDAIR[i, j-1]*(
            jnp.copysign(SINWIN, fCori[i, j-1])*uTmp[i, j-1]
            + COSWIN*vTmp[i, j-1])
    )*SIMaskV[i, j])
    return TAUX, TAUY


def _stress_from_fu_fv(SIMaskU, SIMaskV, TAUX, TAUY, *, cfg, sp, grid, ff):
    """seaice_get_dynforcing.F:207-237 (the `ELSE` arm of `useEXF .AND. useAtmWind`, lane M4CS32ICE): the wind stress
    is available on U and V points (fu, fv of FFIELDS.h, from EXF_MAPFIELDS); CDAIR = SEAICE_drag(_south)/OCEAN_drag
    by the sign of yC at the C point (i,j), taux = CDAIR*fu*SIMaskU, tauy = CDAIR*fv*SIMaskV on every point. The
    pointwise IF is a where (both arms finite: OCEAN_drag is a nonzero parameter)."""
    if cfg.cpp.flag("HACK_FOR_GMAO_CPL"):                                      # :213-219
        raise NotImplementedError("SEAICE_GET_DYNFORCING: HACK_FOR_GMAO_CPL (:213-219) is not ported")
    sz = cfg.size
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)                                        # :221-222
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    CDAIR = jnp.where(grid.yC[i, j] < ZERO,                                    # :224-228
                      sp.SEAICE_drag_south/sp.OCEAN_drag,
                      sp.SEAICE_drag/sp.OCEAN_drag)
    TAUX = TAUX.at[i, j].set(CDAIR*ff.fu[i, j]                                 # :229-230
                             * SIMaskU[i, j])
    TAUY = TAUY.at[i, j].set(CDAIR*ff.fv[i, j]                                 # :231-232
                             * SIMaskV[i, j])
    return TAUX, TAUY
