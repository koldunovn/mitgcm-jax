"""SEAICE_OCEAN_STRESS: pkg/seaice/seaice_ocean_stress.F @63cdc0b (lane M4OFF, the C-grid build)."""

import jax.numpy as jnp

from mitjax.eesupp.exch_rs import EXCH_UV_XY_RS
from mitjax.farray import loop_i, loop_j
from mitjax.model.grid import deg2rad
from mitjax.pkg.seaice.seaice_params_h import HALF, ONE


def seaice_ocean_stress(windTauX, windTauY, myTime, myIter, sf, ff, *, cfg, sp, op, grid, state, ex):
    """SEAICE_OCEAN_STRESS( windTauX, windTauY, myTime, myIter, myThid )
    @63cdc0b pkg/seaice/seaice_ocean_stress.F:6-145

    C     | o Calculate ocean surface stress

    `sf` SEAICE.h (DWATN, uIce, vIce, AREA), `ff` FFields (fu, fv), `state` (uVel, vVel), `grid` (fCori). Returns ff
    with fu/fv blended under the ice (:101-140) and exchanged (:142). windTauX/Y are read only by the
    useHB87StressCoupling arm (:64-98), which raises. The (I,J) loop is independent per point: vectorised. SIGN(a,b) is
    gfortran's copysign; SIN, COS of the turning angle are XLA's (as OSTRES)."""
    del windTauX, windTauY
    if sp.useHB87stressCoupling:                                               # :64-98
        raise NotImplementedError("SEAICE_OCEAN_STRESS: useHB87StressCoupling (:64-98) is not ported")
    sz = cfg.size
    if op.usingPCoords:                                                        # :55-59
        kSrf = sz.Nr
    else:
        kSrf = 1
    SINWAT = jnp.sin(sp.SEAICE_waterTurnAngle*deg2rad)                         # :61 (_RS SINWAT)
    COSWAT = jnp.cos(sp.SEAICE_waterTurnAngle*deg2rad)                         # :62
    DWATN, uIce, vIce, AREA = sf["DWATN"], sf["UICE"], sf["VICE"], sf["AREA"]
    uVel, vVel, fCori = state.uVel, state.vVel, grid.fCori
    j = loop_j(1, sz.sNy)                                                      # :107-138
    i = loop_i(1, sz.sNx)
    fuIceLoc = (HALF*(DWATN[i, j]+DWATN[i-1, j])                               # :109-119
                * COSWAT
                * (uIce[i, j]-uVel[i, j, kSrf])
                - jnp.copysign(SINWAT, fCori[i, j]) * 0.5
                * (DWATN[i, j]
                   * 0.5*(vIce[i, j]-vVel[i, j, kSrf]
                          + vIce[i, j+1]-vVel[i, j+1, kSrf])
                   + DWATN[i-1, j]
                   * 0.5*(vIce[i-1, j]-vVel[i-1, j, kSrf]
                          + vIce[i-1, j+1]-vVel[i-1, j+1, kSrf])))
    fvIceLoc = (HALF*(DWATN[i, j]+DWATN[i, j-1])                               # :120-130
                * COSWAT
                * (vIce[i, j]-vVel[i, j, kSrf])
                + jnp.copysign(SINWAT, fCori[i, j]) * 0.5
                * (DWATN[i, j]
                   * 0.5*(uIce[i, j]-uVel[i, j, kSrf]
                          + uIce[i+1, j]-uVel[i+1, j, kSrf])
                   + DWATN[i, j-1]
                   * 0.5*(uIce[i, j-1]-uVel[i, j-1, kSrf]
                          + uIce[i+1, j-1]-uVel[i+1, j-1, kSrf])))
    areaW = 0.5 * (AREA[i, j] + AREA[i-1, j]) * sp.SEAICEstressFactor         # :131-132
    areaS = 0.5 * (AREA[i, j] + AREA[i, j-1]) * sp.SEAICEstressFactor         # :133-134
    fu = ff.fu.at[i, j].set((ONE-areaW)*ff.fu[i, j]+areaW*fuIceLoc)           # :135
    fv = ff.fv.at[i, j].set((ONE-areaS)*ff.fv[i, j]+areaS*fvIceLoc)           # :136
    fu, fv = EXCH_UV_XY_RS(fu, fv, True, ex=ex)                                # :142
    return ff.replace(fu=fu, fv=fv)
