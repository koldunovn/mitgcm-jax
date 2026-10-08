"""CONVECTIVE_WEIGHTS: model/src/convective_weights.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.ops.safe import safe_div


def convective_weights(k, rhoKm1, rhoK, weightA, weightB, convectCount, *, cfg, grid):
    """CONVECTIVE_WEIGHTS( bi, bj, k, rhoKm1, rhoK, weightA, weightB, convectCount, myThid )
    @63cdc0b model/src/convective_weights.F:6-80

    C Calculates the weights used to represent convective mixing
    C between two layers.
    C Mixing is represented by:
    C                       T(k-1) = T(k-1) + A * ( T(k) - T(k-1) )
    C                       T(k)   = T(k)   + B * ( T(k-1) - T(k) )
    C In the stable case, A = B = 0
    C In the unstable case, A and B are non-zero and are chosen so as to
    C conserve total volume of T.
    C     rhoKm1  :: rho in level k-1
    C     rhoK    :: rho in level  k
    C     weightA :: weight for tracer @ level k-1
    C     weightB :: weight for tracer @ level  k
    C     convectCount :: counter to diagnose where convection occurs

    k: Python int (2..Nr). rhoKm1, rhoK, weightA, weightB: (1-OLx:sNx+OLx,1-OLy:sNy+OLy); convectCount: the same with
    k=(1,Nr). Returns (weightA, weightB, convectCount). Compiled under INCLUDE_CONVECT_CALL or INCLUDE_CONVECT_INI_CALL
    (:46); the macro `_hFacC` is hFacC (as in every ported routine, e.g. forcing_surf_relax.py). Every point reads only its own inputs: one vectorised statement
    per assignment. The IF (:57-73) is a `where` with both branches finite (d2/dS, d1/dS guarded where the THEN
    branch is not taken: dS = 0 on land). `0.` and `1.` are exact REAL*4 literals."""
    sz = cfg.size
    g = grid
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)                                         # :55
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)                                         # :56
    unstable = ((g.hFacC[i, j, k-1]*g.hFacC[i, j, k] > 0.)                      # :57-59
                & ((rhoK[i, j]-rhoKm1[i, j])*g.rkSign*g.gravitySign < 0.))
    d1 = g.hFacC[i, j, k-1]*g.drF[k-1]                                          # :62
    d2 = g.hFacC[i, j, k]*g.drF[k]                                              # :63
    dS = d1+d2                                                                  # :64
    weightA = weightA.at[i, j].set(jnp.where(unstable, safe_div(d2, dS, unstable), 0.))     # :65 / :70
    weightB = weightB.at[i, j].set(jnp.where(unstable, safe_div(d1, dS, unstable), 0.))     # :66 / :71
    convectCount = convectCount.at[i, j, k].set(jnp.where(unstable, 1., 0.))                # :67 / :72
    return weightA, weightB, convectCount
