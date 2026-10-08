"""OSTRES: pkg/seaice/ostres.F @63cdc0b (B-grid ocean surface stress)."""

import jax.numpy as jnp

from mitjax.eesupp.exch_rs import EXCH_UV_XY_RS, _rewrap
from mitjax.farray import loop_i, loop_j
from mitjax.model.grid import deg2rad
from mitjax.pkg.seaice.seaice_params_h import ONE, QUART


def ostres(COR_ICE, sf, ff, *, cfg, sp, ex):
    """ostres( COR_ICE, myThid )   @63cdc0b pkg/seaice/ostres.F:4-98

    C     | o Calculate ocean surface stress

    `sf` SEAICE.h (WINDX, WINDY, DWATN, UICE, VICE, GWATX, GWATY, AREA), `ff` FFields (fu, fv). Returns (sf, ff):
    WINDX/WINDY after their exchange (:43), fu/fv blended under the ice (:65-92) and exchanged (:93).
    Ported: SEAICE_BGRID_DYNAMICS with SEAICE_BICE_STRESS and SEAICE_EXTERNAL_FLUXES (the stress from the wind,
    :45-62, is not compiled); other builds raise. The (I,J) loop is independent per point: vectorised. SIGN(a,b) is
    gfortran's copysign; SIN, COS are XLA's (bitwise glibc, mitjax/ops/libm.py)."""
    for o, want in (("SEAICE_BGRID_DYNAMICS", True), ("SEAICE_BICE_STRESS", True), ("SEAICE_EXTERNAL_FLUXES", True)):
        if cfg.cpp.flag(o, "SEAICE_OPTIONS.h") != want:
            raise NotImplementedError(f"OSTRES: the build without {o} is not ported")
    sz = cfg.size
    sf = dict(sf)
    SINWIN = jnp.sin(sp.SEAICE_airTurnAngle*deg2rad)                           # :37
    COSWIN = jnp.cos(sp.SEAICE_airTurnAngle*deg2rad)                           # :38
    del SINWIN, COSWIN                                                         # (read only in :45-62, not compiled)
    SINWAT = jnp.sin(sp.SEAICE_waterTurnAngle*deg2rad)                         # :39
    COSWAT = jnp.cos(sp.SEAICE_waterTurnAngle*deg2rad)                         # :40
    u, v = ex.EXCH_UV_XY_RL(sf["WINDX"].data, sf["WINDY"].data, True)          # :43
    sf["WINDX"], sf["WINDY"] = _rewrap(u, sf["WINDX"]), _rewrap(v, sf["WINDY"])
    j = loop_j(1, sz.sNy)                                                      # :65-92
    i = loop_i(1, sz.sNx)
    DWATN, UICE, VICE, GWATX, GWATY, AREA = (sf[n] for n in ("DWATN", "UICE", "VICE", "GWATX", "GWATY", "AREA"))
    fuIce = QUART*(DWATN[i, j]+DWATN[i, j+1])*(                                # :69-76
        COSWAT
        * (UICE[i, j]-GWATX[i, j]
           + UICE[i, j+1]-GWATX[i, j+1])
        - jnp.copysign(SINWAT, COR_ICE[i, j])
        * (VICE[i, j]-GWATY[i, j]
           + VICE[i, j+1]-GWATY[i, j+1]))
    fvIce = QUART*(DWATN[i, j]+DWATN[i+1, j])*(                                # :77-84
        jnp.copysign(SINWAT, COR_ICE[i, j])
        * (UICE[i, j]-GWATX[i, j]
           + UICE[i+1, j]-GWATX[i+1, j])
        + COSWAT
        * (VICE[i, j]-GWATY[i, j]
           + VICE[i+1, j]-GWATY[i+1, j]))
    fu = ff.fu.at[i, j].set((ONE-AREA[i, j])*ff.fu[i, j]                      # :85-86
                            + AREA[i, j]*fuIce)
    fv = ff.fv.at[i, j].set((ONE-AREA[i, j])*ff.fv[i, j]                      # :87-88
                            + AREA[i, j]*fvIce)
    fu, fv = EXCH_UV_XY_RS(fu, fv, True, ex=ex)                                # :93
    return sf, ff.replace(fu=fu, fv=fv)
