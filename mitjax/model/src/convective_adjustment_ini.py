"""CONVECTIVE_ADJUSTMENT_INI: model/src/convective_adjustment_ini.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import FArray
from mitjax.model.src.convective_adjustment import _level
from mitjax.model.src.convective_weights import convective_weights
from mitjax.model.src.convectively_mixtracer import convectively_mixtracer
from mitjax.model.src.find_rho import find_rho_2d
from mitjax.pkg.ptracers.ptracers_convect import ptracers_convect


def convective_adjustment_ini(myTime, myIter, *, cfg, grid, params, eos, state, ptr=None, ptf=None):
    """CONVECTIVE_ADJUSTMENT_INI( bi, bj, myTime, myIter, myThid )
    @63cdc0b model/src/convective_adjustment_ini.F:10-186

    C     | SUBROUTINE CONVECTIVE_ADJUSTMENT_INI
    C     | o Driver for vertical mixing or similar parameterization
    C     | Same prognostic code logic as S/R CONVECTIVE_ADJUSTMENT,
    C     | but different time history behavior in forward-reverse
    C     | adjoint operation.

    Returns (state, ptf). Compiled under INCLUDE_CONVECT_INI_CALL (:52; raise otherwise). The statements of
    CONVECTIVE_ADJUSTMENT without its DIFFERENT_MULTIPLE test (commented out here, :76-77, :181) and without the
    diagnostics; the level loop (:119-178) is the same recursion, a level scan (mitjax/model/src/
    convective_adjustment.py docstring). myTime, myIter are not read."""
    if not cfg.cpp.INCLUDE_CONVECT_INI_CALL:                                    # :52
        raise NotImplementedError("CONVECTIVE_ADJUSTMENT_INI: compiled without INCLUDE_CONVECT_INI_CALL")
    sz = cfg.size
    Nr = sz.Nr
    iMin = 1-sz.OLx                                                             # :80-83
    iMax = sz.sNx+sz.OLx
    jMin = 1-sz.OLy
    jMax = sz.sNy+sz.OLy
    # :104-116  IF ( rkSign*gravitySign .GT. 0. ) "<=> usingZCoords" (:105): decided on the host from usingZCoords
    if not params.usingZCoords:
        raise NotImplementedError("CONVECTIVE_ADJUSTMENT_INI: the pressure-coordinate branch (:110-116) is not "
                                  "ported")
    kTop, kBottom, kDir, deltaK = 2, Nr, 1, -1                                  # :106-109
    theta, salt = state.theta, state.salt
    lev = _level(theta, 1)
    rhoKm1 = lev.local("rhoKm1")
    rhoK = lev.local("rhoK")
    weightA = lev.local("weightA")
    weightB = lev.local("weightB")
    ConvectCount = FArray(jnp.zeros_like(theta.data), "ConvectCount", tiled=theta.tiled, _dims=theta.dims)  # :92-98
    usePTRACERS = cfg.cpp.ALLOW_PTRACERS and params.usePTRACERS                 # :169-175
    # :119  DO k=kTop,kBottom,kDir as a level scan (KERNEL_GUIDE §4; no level branches)
    def level_k(k, c):
        rhoKm1, rhoK, weightA, weightB, ConvectCount, theta, salt, ptf = c
        rhoKm1 = find_rho_2d(iMin, iMax, jMin, jMax, k+deltaK,                  # :132-137
                             _level(theta, k-1), _level(salt, k-1), rhoKm1, k-1,
                             cfg=cfg, grid=grid, params=params, eos=eos, state=state)
        rhoK = find_rho_2d(iMin, iMax, jMin, jMax, k+deltaK,                    # :140-145
                           _level(theta, k), _level(salt, k), rhoK, k,
                           cfg=cfg, grid=grid, params=params, eos=eos, state=state)
        weightA, weightB, ConvectCount = convective_weights(                    # :152-155
            k, rhoKm1, rhoK, weightA, weightB, ConvectCount, cfg=cfg, grid=grid)
        theta = convectively_mixtracer(k, weightA, weightB, theta, cfg=cfg)    # :158-161
        salt = convectively_mixtracer(k, weightA, weightB, salt, cfg=cfg)      # :164-167
        if usePTRACERS:                                                         # :171-175
            ptf = ptracers_convect(k, weightA, weightB, cfg=cfg, ptr=ptr, ptf=ptf)
        return rhoKm1, rhoK, weightA, weightB, ConvectCount, theta, salt, ptf

    from mitjax.ops.scan_k import scan_levels
    c = scan_levels(level_k, (rhoKm1, rhoK, weightA, weightB, ConvectCount, theta, salt, ptf), kTop, kBottom)
    rhoKm1, rhoK, weightA, weightB, ConvectCount, theta, salt, ptf = c
    return state.replace(theta=theta, salt=salt), ptf
