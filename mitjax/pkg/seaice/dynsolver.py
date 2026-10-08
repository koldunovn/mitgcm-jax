"""DYNSOLVER: pkg/seaice/dynsolver.F @63cdc0b (B-grid sea-ice dynamics driver)."""

import jax.numpy as jnp
import numpy as np

from mitjax.farray import FArray, loop_i, loop_j
from mitjax.model.grid import deg2rad
from mitjax.ops.libm import glibc_exp
from mitjax.ops.safe import safe_sqrt
from mitjax.pkg.seaice.ostres import ostres
from mitjax.pkg.seaice.seaice_params_h import HALF, ONE, QUART


def kgeo_level(sf):
    """The level KGEO holds on every point (KGEO is set once by SEAICE_INIT_FIXED, a host integer field). A KGEO that
    varies in space raises (the per-point level gather of dynsolver.F:81-84 is not ported)."""
    k = np.unique(np.asarray(sf["KGEO"].data))
    if k.size != 1:
        raise NotImplementedError("DYNSOLVER: a spatially varying KGEO (no SEAICE_BICE_STRESS) is not ported")
    return int(k[0])


def dynsolver(myTime, myIter, sf, ff, exf, *, cfg, sp, op, grid, state, ex, kgeo):
    """DYNSOLVER( myTime, myIter, myThid )   @63cdc0b pkg/seaice/dynsolver.F:10-350

    C     | o Ice dynamics using LSR solver
    C     |   Zhang and Hibler,   JGR, 102, 8691-8702, 1997

    `sf` SEAICE.h / SEAICE_GRID.h (dict), `ff` FFields (pLoad, sIceLoad, fu, fv), `exf` EXF_FIELDS.h (uwind, vwind),
    `state` (uVel, vVel, etaN), `kgeo` the level of KGEO (kgeo_level, static). Returns (sf, ff).
    Ported: SEAICE_BGRID_DYNAMICS with EXPLICIT_SSH_SLOPE and ATMOSPHERIC_LOADING (this build), SEAICEuseDYNAMICS =
    .FALSE.: the forcing (:66-235: GWATX/Y, AMASS and COR_ICE, the wind drag DAIRN and WINDX/Y, FORCEX/Y with the
    surface tilt, FORCEX0/Y0, PRESS0, SEAICE_zMax/zMin) and OSTRES (:301). SEAICEuseDYNAMICS (:237-290, LSR) raises;
    so do ALLOW_AUTODIFF, SEAICE_ALLOW_CLIPVELS, ALLOW_OBCS when compiled. The point loops are independent:
    vectorised. `IF ( AAA .LE. SEAICE_EPS_SQ )` (:129) is a pointwise where; the SQRT of the discarded arm is guarded
    (safe_sqrt on AAA > SEAICE_EPS_SQ) so that its derivative at AAA = 0 is finite. `IF ( YC .LT. ZERO )` (:143) is a
    where on the grid. SIGN(a,b) is gfortran's copysign; SIN, COS are XLA's (bitwise glibc); EXP is glibc_exp. The
    REAL*4 literal `0.` of COR_ICE (:102) is exact. ECCEN, ECM2 (:68-69) are set and not read in this path."""
    for o, want in (("SEAICE_BGRID_DYNAMICS", True), ("EXPLICIT_SSH_SLOPE", True), ("SEAICE_ALLOW_CLIPVELS", False)):
        if cfg.cpp.flag(o, "SEAICE_OPTIONS.h") != want:
            raise NotImplementedError(f"DYNSOLVER: {o} {'undefined' if want else 'defined'} is not ported")
    if not cfg.cpp.flag("ATMOSPHERIC_LOADING"):
        raise NotImplementedError("DYNSOLVER: ATMOSPHERIC_LOADING undefined (:173-191) is not ported")
    if cfg.cpp.flag("ALLOW_AUTODIFF"):
        raise NotImplementedError("DYNSOLVER: ALLOW_AUTODIFF is not ported")
    if sp.SEAICEuseDYNAMICS:
        raise NotImplementedError("DYNSOLVER: SEAICEuseDYNAMICS (LSR, :237-290) is not ported")
    sz = cfg.size
    OLx, OLy, sNx, sNy = sz.OLx, sz.OLy, sz.sNx, sz.sNy
    sf = dict(sf)
    RHOICE = sp.SEAICE_rhoIce                                                  # :66
    RHOAIR = sp.SEAICE_rhoAir                                                  # :67
    PSTAR = sp.SEAICE_strength                                                 # :70
    SINWIN = jnp.sin(sp.SEAICE_airTurnAngle*deg2rad)                           # :73
    COSWIN = jnp.cos(sp.SEAICE_airTurnAngle*deg2rad)                           # :74
    # :77-95 geostrophic currents at the KGEO level
    j = loop_j(0, sNy+1)
    i = loop_i(0, sNx+1)
    uVel, vVel = state.uVel, state.vVel
    sf["GWATX"] = sf["GWATX"].at[i, j].set(HALF*(uVel[i, j, kgeo]             # :81-82
                                                 + uVel[i, j-1, kgeo]))
    sf["GWATY"] = sf["GWATY"].at[i, j].set(HALF*(vVel[i, j, kgeo]             # :83-84
                                                 + vVel[i-1, j, kgeo]))
    # :98-114 mass and Coriolis term
    HEFF = sf["HEFF"]
    COR_ICE = FArray(jnp.zeros_like(HEFF.data), "COR_ICE", tiled=HEFF.tiled, _dims=HEFF.dims)   # :63, :100-104
    j = loop_j(1, sNy)
    i = loop_i(1, sNx)
    sf["AMASS"] = sf["AMASS"].at[i, j].set(RHOICE*QUART*(                     # :107-109
        HEFF[i, j] + HEFF[i-1, j]
        + HEFF[i, j-1] + HEFF[i-1, j-1]))
    COR_ICE = COR_ICE.at[i, j].set(sf["AMASS"][i, j] * grid.fCoriG[i, j])     # :110
    # :120-161 wind stress
    UWIND, VWIND = exf["uwind"], exf["vwind"]
    U1 = QUART*(UWIND[i-1, j-1]+UWIND[i-1, j]                                  # :124-125
                + UWIND[i, j-1]+UWIND[i, j])
    V1 = QUART*(VWIND[i-1, j-1]+VWIND[i-1, j]                                  # :126-127
                + VWIND[i, j-1]+VWIND[i, j])
    AAA = U1**2+V1**2                                                          # :128
    AAA = jnp.where(AAA <= sp.SEAICE_EPS_SQ,                                   # :129-133
                    sp.SEAICE_EPS,
                    safe_sqrt(AAA, AAA > sp.SEAICE_EPS_SQ))
    sgn = jnp.copysign(SINWIN, grid.fCori[i, j])
    DAIRN = RHOAIR*sp.OCEAN_drag*(2.70+0.142*AAA+0.0764*AAA*AAA)              # :135-136
    sf["WINDX"] = sf["WINDX"].at[i, j].set(DAIRN*(COSWIN*U1-sgn*V1))           # :137-138
    sf["WINDY"] = sf["WINDY"].at[i, j].set(DAIRN*(sgn*U1+COSWIN*V1))           # :139-140
    AREA = sf["AREA"]
    DAIRN = jnp.where(grid.yC[i, j] < 0.0,                                     # :143-153 (YC = GRID.h yC)
                      RHOAIR*(sp.SEAICE_drag_south*AAA*AREA[i, j]
                              + sp.OCEAN_drag*(2.70+0.142*AAA
                                               + 0.0764*AAA*AAA)*(ONE-AREA[i, j])),
                      RHOAIR*(sp.SEAICE_drag*AAA*AREA[i, j]
                              + sp.OCEAN_drag*(2.70+0.142*AAA
                                               + 0.0764*AAA*AAA)*(ONE-AREA[i, j])))
    sf["DAIRN"] = sf["DAIRN"].at[i, j].set(DAIRN)
    sf["FORCEX"] = sf["FORCEX"].at[i, j].set(DAIRN*(COSWIN*U1-sgn*V1))         # :154-155
    sf["FORCEY"] = sf["FORCEY"].at[i, j].set(DAIRN*(sgn*U1+COSWIN*V1))         # :156-157
    # :163-235 surface tilt (EXPLICIT_SSH_SLOPE) and ice strength
    j = loop_j(1-OLy, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx)
    phiSurf = grid.Bo_surf[i, j]*state.etaN[i, j]                              # :170
    if op.useRealFreshWaterFlux:                                               # :175-191
        phiSurf = phiSurf + (ff.pLoad[i, j]                                    # :178-181
                             + ff.sIceLoad[i, j]*op.gravity*op.sIceLoadFac
                             )*op.recip_rhoConst
    else:
        phiSurf = phiSurf + ff.pLoad[i, j]*op.recip_rhoConst                   # :187-188
    phiSurf = FArray(phiSurf, "phiSurf", i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy))
    j = loop_j(1-OLy+1, sNy+OLy)                                               # :193-210
    i = loop_i(1-OLx+1, sNx+OLx)
    AMASS, SIMaskU, SIMaskV = sf["AMASS"], sf["SIMaskU"], sf["SIMaskV"]
    sf["FORCEX"] = sf["FORCEX"].at[i, j].set(                                  # :196-200
        sf["FORCEX"][i, j]
        - AMASS[i, j]
        * ((phiSurf[i, j]-phiSurf[i-1, j])*SIMaskU[i, j]
           + (phiSurf[i, j-1]-phiSurf[i-1, j-1])*SIMaskV[i, j-1]
           )*HALF*grid.recip_dxV[i, j])
    sf["FORCEY"] = sf["FORCEY"].at[i, j].set(                                  # :201-205
        sf["FORCEY"][i, j]
        - AMASS[i, j]
        * ((phiSurf[i, j]-phiSurf[i, j-1])*SIMaskV[i, j]
           + (phiSurf[i-1, j]-phiSurf[i-1, j-1])*SIMaskV[i-1, j]
           )*HALF*grid.recip_dyU[i, j])
    sf["FORCEX0"] = sf["FORCEX0"].at[i, j].set(sf["FORCEX"][i, j])            # :207
    sf["FORCEY0"] = sf["FORCEY0"].at[i, j].set(sf["FORCEY"][i, j])            # :208
    j = loop_j(1-OLy, sNy+OLy)                                                 # :212-233
    i = loop_i(1-OLx, sNx+OLx)
    PRESS0 = (PSTAR*sf["HEFF"][i, j]                                           # :225-226
              * glibc_exp(-sp.SEAICE_cStar*(ONE-sf["AREA"][i, j])))
    sf["SEAICE_zMax"] = sf["SEAICE_zMax"].at[i, j].set(sp.SEAICE_zetaMaxFac*PRESS0)   # :228
    sf["SEAICE_zMin"] = sf["SEAICE_zMin"].at[i, j].set(sp.SEAICE_zetaMin*jnp.ones_like(PRESS0))   # :230
    sf["PRESS0"] = sf["PRESS0"].at[i, j].set(PRESS0*sf["HEFFM"][i, j])        # :231
    sf, ff = ostres(COR_ICE, sf, ff, cfg=cfg, sp=sp, ex=ex)                    # :301
    return sf, ff
