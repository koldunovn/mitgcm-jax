"""FREEZE_SURFACE: model/src/freeze_surface.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j

Tfreezing_value = -1.9      # freeze_surface.F:48  Tfreezing = -1.9 _d 0 (double literal)


def freeze_surface(myTime, myIter, ff, *, cfg, grid, fp, state):
    """FREEZE_SURFACE( myTime, myIter, myThid )   @63cdc0b model/src/freeze_surface.F:7-69

    C     | S/R FREEZE_SURFACE
    C     | o Check water temperature and limit range of temperature
    C     | appropriately.

    `state` (theta of level k written), `ff` (adjustColdSST_diag written), `grid` (drF, hFacC), `fp.dTtracerLev`.
    Returns (state, ff). The pointwise IF (:55-62) is a select; both branches are finite on every point (dTtracerLev
    > 0, set by INI_PARMS), so no guard is needed."""
    sz = cfg.size
    if fp.usingPCoords:                                                         # :42-46
        k = sz.Nr
    else:
        k = 1
    Tfreezing = Tfreezing_value                                                 # :48
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    th = state.theta[i, j, k]
    cold = th < Tfreezing                                                       # :55
    adj = (Tfreezing - th) * grid.drF[k] * grid.hFacC[i, j, k] / float(fp.dTtracerLev[k - 1])   # :56-58
    adjustColdSST_diag = ff.adjustColdSST_diag.at[i, j].set(jnp.where(cold, adj, 0.))            # :56-58, :61
    theta = state.theta.at[i, j, k].set(jnp.where(cold, Tfreezing, th))                          # :59
    return state.replace(theta=theta), ff.replace(adjustColdSST_diag=adjustColdSST_diag)
