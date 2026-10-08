"""CTRL_BOUND_2D   @63cdc0b pkg/ctrl/ctrl_bound.F:74-125"""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j, loops_kji


def ctrl_bound_2d(fieldCur, mask2D, boundsVec, *, sz):
    """CTRL_BOUND_2D( fieldCur, mask2D, boundsVec, myThid )

    C     | o in forward mode: impose bounds on ctrl vector values
    C     | o in adjoint mode: do nothing ... or emulate local minimum

    boundsVec: the 5 static bounds (xx_gentim2d_bounds(1:5,iarr), host floats: a namelist value deciding a branch
    of set-up code). Returns fieldCur."""
    if boundsVec[0] < boundsVec[3]:                                    # :101 boundsVec(1).LT.boundsVec(4)
        j = loop_j(1 - sz.OLy, sz.sNy + sz.OLy)                        # :106
        i = loop_i(1 - sz.OLx, sz.sNx + sz.OLx)                        # :107
        wet = mask2D[i, j] != 0.0                                      # :108
        f = fieldCur[i, j]
        f = jnp.where(wet & (f > boundsVec[3]), boundsVec[3], f)       # :109-111
        f = jnp.where(wet & (f < boundsVec[0]), boundsVec[0], f)       # :112-114
        fieldCur = fieldCur.at[i, j].set(f)
    return fieldCur


def ctrl_bound_3d(fieldCur, mask3D, boundsVec, *, sz):
    """CTRL_BOUND_3D( fieldCur, mask3D, boundsVec, myThid )   @63cdc0b pkg/ctrl/ctrl_bound.F:14-68 (GOADK lane)

    C     | o in forward mode: impose bounds on ctrl vector values
    C     | o in adjoint mode: do nothing ... or emulate local minimum

    boundsVec: the 5 static bounds (xx_genarr3d_bounds(1:5,iarr), host floats: a namelist value deciding a branch of
    set-up code). The clipping loop runs on the interior 1..sNx, 1..sNy, 1..Nr (:43-60; each point reads only
    itself). Returns fieldCur."""
    if boundsVec[0] < boundsVec[3]:                                    # :41 boundsVec(1).LT.boundsVec(4)
        k, j, i = loops_kji((1, sz.Nr), (1, sz.sNy), (1, sz.sNx))      # :46-48
        wet = mask3D[i, j, k] != 0.0                                   # :49
        f = fieldCur[i, j, k]
        f = jnp.where(wet & (f > boundsVec[3]), boundsVec[3], f)       # :50-52
        f = jnp.where(wet & (f < boundsVec[0]), boundsVec[0], f)       # :53-55
        fieldCur = fieldCur.at[i, j, k].set(f)
    return fieldCur
