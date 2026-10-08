"""GGL90_ADD_STOKESDRIFT: pkg/ggl90/ggl90_add_stokesdrift.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.model.grid import PI
from mitjax.ops.libm import glibc_exp
from mitjax.ops.safe import safe_sqrt


def ggl90_add_stokesdrift(uRes, vRes, uFld, vFld, k, *, cfg, grid, ggl, state):
    """GGL90_ADD_STOKESDRIFT( uRes, vRes, uFld, vFld, k, bi, bj, myThid )
    @63cdc0b pkg/ggl90/ggl90_add_stokesdrift.F:7-79

    C  Add Stokes-drift contribution to Eulerien velocity to get residual flow
    C   uRes, vRes   :: residual flow with Stokes-drift added
    C   uFld, vFld   :: Eulerien horizontal velocity, 2 compon.
    C   k            :: current vertical level

    uRes, vRes, uFld, vFld: FArrays (1-OLx:sNx+OLx,1-OLy:sNy+OLy); k a Python int; `state` FFIELDS.h
    (surfaceForcingU/V). Returns (uRes, vRes), unchanged without ALLOW_GGL90_LANGMUIR or useLANGMUIR (:291, :301).
    EXP is glibc_exp; SQRT(ABS(surfaceForcing)) is guarded on |surfaceForcing| > 0 (value Fortran's: SQRT(+0) = +0,
    SIGN keeps the sign of the forcing)."""
    if not (cfg.cpp.flag("ALLOW_GGL90_LANGMUIR", "GGL90_OPTIONS.h") and ggl.useLANGMUIR):
        return uRes, vRes
    sz = cfg.size
    recip_Lasq = 1.0/ggl.LC_num                                        # :305  1. _d 0 / LC_num
    recip_Lasq = recip_Lasq * recip_Lasq                                # :306
    depthFac = recip_Lasq*glibc_exp(4.0*PI/ggl.LC_lambda*grid.rC[k])   # :307  EXP( 4. _d 0 *PI/LC_lambda*rC(k) )
    j, i = loop_j(1-sz.OLy, sz.sNy+sz.OLy), loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    sfU, sfV = state.surfaceForcingU[i, j], state.surfaceForcingV[i, j]
    uStar = jnp.copysign(safe_sqrt(jnp.abs(sfU), jnp.abs(sfU) > 0.0), sfU)     # :310-311
    stokesU = uStar * depthFac                                          # :312
    vStar = jnp.copysign(safe_sqrt(jnp.abs(sfV), jnp.abs(sfV) > 0.0), sfV)     # :313-314
    stokesV = vStar * depthFac                                          # :315
    uRes = uRes.at[i, j].set(uFld[i, j] + stokesU*grid.maskW[i, j, k])  # :322
    vRes = vRes.at[i, j].set(vFld[i, j] + stokesV*grid.maskS[i, j, k])  # :323
    return uRes, vRes
