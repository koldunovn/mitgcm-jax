"""CALC_IVDC: model/src/calc_ivdc.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j


def calc_ivdc(iMin, iMax, jMin, jMax, K, sigmaR, myTime, myIter, *, grid, state):
    """CALC_IVDC( bi, bj, iMin, iMax, jMin, jMax, K, sigmaR, myTime, myIter, myThid )
    @63cdc0b model/src/calc_ivdc.F:2-56

    C     *==========================================================*
    C     | SUBROUTINE CALC_IVDC
    C     | o Calculates Implicit Vertical Diffusivity for Convection
    C     \\==========================================================*

    sigmaR: FArray (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr); K a Python int. Returns IVDConvCount (DYNVARS.h, `state`) with
    level K set to 1 where the column is statically unstable and 0 elsewhere on iMin:iMax, jMin:jMax (:44-53); every
    other point keeps its value. The literal `0.` (:47) is REAL*4 zero (exact).
    """
    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    # :47  IF ( -sigmaR(i,j,k)*gravitySign.GT.0. ) THEN 1. _d 0 ELSE 0. _d 0
    return state.IVDConvCount.at[i, j, K].set(
        jnp.where(-sigmaR[i, j, K]*grid.gravitySign > 0.0, 1.0, 0.0))
