"""SEAICE_OCEANDRAG_COEFFS: pkg/seaice/seaice_oceandrag_coeffs.F @63cdc0b (lane M4OFF session 3, the C-grid build)."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.ops.safe import safe_sqrt
from mitjax.pkg.seaice.seaice_params_h import ZERO


def seaice_oceandrag_coeffs(uIceLoc, vIceLoc, HEFFMLoc, CwatC, iStep, myTime, myIter, *, cfg, sp, op, grid, state):
    """SEAICE_OCEANDRAG_COEFFS( uIceLoc, vIceLoc, HEFFMLoc, CwatC, iStep, myTime, myIter, myThid )
    @63cdc0b pkg/seaice/seaice_oceandrag_coeffs.F:9-116

    C     | o Compute the drag coefficients for ice-ocean drag,
    C     |   so that we can use the same code for different solvers

    Returns CwatC (the SEAICE.h field DWATN at the call sites): the points 1-OLx..sNx+OLx-1, 1-OLy..sNy+OLy-1
    written (:72-73), the last row and column as given. Ported: the C-grid build without ALLOW_OBCS (the
    #else of OBCS_UVICE_OLD, :83-92). The (i,j) loop is independent per point: vectorised; the IF of :105-106 is a
    `where` and the SQRT is evaluated only on the lanes where the Fortran evaluates it (safe_sqrt on the IF's
    condition, the guard before the operation; lane M4ADLAB session 2: the former `where(cond, sqrt(tempVar), c)`
    had the same values but a backward 0*inf = NaN at tempVar = 0, i.e. on land and wherever ice and ocean move
    alike: measured, dev job 27893261)."""
    del iStep, myTime, myIter
    if cfg.cpp.flag("ALLOW_OBCS"):
        raise NotImplementedError("SEAICE_OCEANDRAG_COEFFS: ALLOW_OBCS is not ported")
    sz = cfg.size
    if op.usingPCoords:                                                        # :63-67
        kSrf = sz.Nr
    else:
        kSrf = 1
    tempMin = sp.SEAICEdWatMin*sp.SEAICEdWatMin                                # :68
    uVel, vVel = state.uVel, state.vVel
    maskInW, maskInS, YC = grid.maskInW, grid.maskInS, grid.yC
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy-1)                                      # :72
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx-1)                                      # :73
    tempVar = 0.25*(                                                           # :84-92
        ((uIceLoc[i, j]-uVel[i, j, kSrf])
         * maskInW[i, j]
         + (uIceLoc[i+1, j]-uVel[i+1, j, kSrf])
         * maskInW[i+1, j])**2
        + ((vIceLoc[i, j]-vVel[i, j, kSrf])
           * maskInS[i, j]
           + (vIceLoc[i, j+1]-vVel[i, j+1, kSrf])
           * maskInS[i, j+1])**2)
    dragCoeff = jnp.where(YC[i, j] < ZERO,                                     # :94-98
                          sp.SEAICE_waterDrag_south*op.rhoConst,
                          sp.SEAICE_waterDrag*op.rhoConst)
    c = sp.SEAICEdWatMin*jnp.ones_like(tempVar)                                # :99
    cond = dragCoeff*dragCoeff * tempVar > tempMin
    c = jnp.where(cond,                                                        # :105-106
                  dragCoeff*safe_sqrt(tempVar, cond), c)
    CwatC = CwatC.at[i, j].set(c * HEFFMLoc[i, j])                             # :107
    return CwatC
