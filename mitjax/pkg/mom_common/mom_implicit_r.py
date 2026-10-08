"""pkg/mom_common/mom_u_implicit_r.F (MOM_U_IMPLICIT_R) and mom_v_implicit_r.F (MOM_V_IMPLICIT_R): implicit vertical
viscosity of the momentum tendencies (M3 lane MLAdjust: implicitViscosity)."""

import jax.numpy as jnp

from mitjax.farray import loops_kji
from mitjax.model.grid import oneRS
from mitjax.model.src.solve_tridiagonal import solve_tridiagonal

_OPT = "MOM_COMMON_OPTIONS.h"


def _implicit_r(comp, kappaR, myTime, myIter, *, cfg, grid, params, state):
    """The shared statements of MOM_U_IMPLICIT_R / MOM_V_IMPLICIT_R (the two files differ only in U/W vs V/S names
    and the PARAMETER ranges :50-51): returns the new gU (comp "U") or gV ("V") of `state`."""
    sz = cfg.size
    Nr = sz.Nr
    p, g = params, grid
    if p.selectImplicitDrag >= 1:                                               # :151-207
        raise NotImplementedError(f"MOM_{comp}_IMPLICIT_R: selectImplicitDrag >= 1 (implicit bottom drag) is not "
                                  "ported")
    if p.momImplVertAdv and Nr > 1:                                             # :210-283
        raise NotImplementedError(f"MOM_{comp}_IMPLICIT_R: momImplVertAdv is not ported")
    if cfg.cpp.flag("ALLOW_SOLVE4_PS_AND_DRAG") and p.selectImplicitDrag == 2:   # :322-347
        raise NotImplementedError(f"MOM_{comp}_IMPLICIT_R: ALLOW_SOLVE4_PS_AND_DRAG is not ported")
    if cfg.cpp.flag("ALLOW_DIAGNOSTICS", _OPT) and p.useDiagnostics:            # :88-98, :350-397
        raise NotImplementedError(f"MOM_{comp}_IMPLICIT_R: the diagnostics blocks are not ported")
    if comp == "U":
        iMin, iMax, jMin, jMax = 1, sz.sNx+1, 1, sz.sNy                          # :50-51
        mask, recip_hFac, gF = g.maskW, g.recip_hFacW, state.gU
    else:
        iMin, iMax, jMin, jMax = 1, sz.sNx, 1, sz.sNy+1                          # :50-51
        mask, recip_hFac, gF = g.maskS, g.recip_hFacS, state.gV
    # :78-87  diagonalNumber = 1; b5d = 0. _d 0, c5d = 1. _d 0, d5d = 0. _d 0 on every point
    k, j, i = loops_kji((1, Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
    b5d = gF.local("b5d").at[i, j, k].set(0.)
    c5d = gF.local("c5d").at[i, j, k].set(1.)
    d5d = gF.local("d5d").at[i, j, k].set(0.)
    diagonalNumber = 1
    if p.implicitViscosity and Nr > 1:                                          # :109
        diagonalNumber = 3                                                      # :112
        k, j, i = loops_kji((2, Nr), (jMin, jMax), (iMin, iMax))                # :114-125  1rst lower diagonal
        wet = mask[i, j, k-1] == oneRS
        b5d = b5d.at[i, j, k].set(jnp.where(
            wet, -p.deltaTMom
            * recip_hFac[i, j, k]*g.recip_drF[k]
            * g.recip_deepFac2C[k]*p.recip_rhoFacC[k]
            * kappaR[i, j, k]*g.recip_drC[k]
            * g.deepFac2F[k]*p.rhoFacF[k], b5d[i, j, k]))
        k, j, i = loops_kji((1, Nr-1), (jMin, jMax), (iMin, iMax))              # :127-138  1rst upper diagonal
        wet = mask[i, j, k+1] == oneRS
        d5d = d5d.at[i, j, k].set(jnp.where(
            wet, -p.deltaTMom
            * recip_hFac[i, j, k]*g.recip_drF[k]
            * g.recip_deepFac2C[k]*p.recip_rhoFacC[k]
            * kappaR[i, j, k+1]*g.recip_drC[k+1]
            * g.deepFac2F[k+1]*p.rhoFacF[k+1], d5d[i, j, k]))
        k, j, i = loops_kji((1, Nr), (jMin, jMax), (iMin, iMax))                # :140-146  Main diagonal
        c5d = c5d.at[i, j, k].set(1. - (b5d[i, j, k] + d5d[i, j, k]))
    if diagonalNumber == 1:                                                     # :285-306
        raise NotImplementedError(f"MOM_{comp}_IMPLICIT_R: the 1-diagonal solve (:285-306) is not ported")
    errCode = -1                                                                # :309
    gF, errCode = solve_tridiagonal(iMin, iMax, jMin, jMax, b5d, c5d, d5d, gF, errCode, cfg=cfg)   # :310-314
    # :315-317 IF (errCode.GE.1) STOP: not raised inside a traced program (the solve guards its zero pivots)
    return gF


def mom_u_implicit_r(kappaRU, myTime, myIter, *, cfg, grid, params, state):
    """MOM_U_IMPLICIT_R( kappaRU, bi, bj, myTime, myIter, myThid )   @63cdc0b pkg/mom_common/mom_u_implicit_r.F:8-400

    C     | o Solve implicitly vertical advection & diffusion of momentum, zonal component

    Returns the State with gU solved (DYNVARS.h gU is updated in place in the Fortran). Ported: implicitViscosity
    with Nr > 1 (the tri-diagonal matrix :109-149; the masks tested against oneRS; _recip_hFacW = recip_hFacW, no
    HFACC_MACROS override; products left to right, the unary minus on the whole product, -(a*b) = (-a)*b bit for
    bit) and its SOLVE_TRIDIAGONAL (:307-317) on iMin..iMax = 1..sNx+1, jMin..jMax = 1..sNy (:50-51). The k loops
    are independent across k (each level of b5d/d5d reads only inputs; c5d its own level): k-vectorised nests.
    Raise: selectImplicitDrag >= 1 (:151-207), momImplVertAdv (:210-283), the 1-diagonal solve, ALLOW_SOLVE4_PS_AND_DRAG
    with selectImplicitDrag = 2, the diagnostics. `1. _d 0`, `0. _d 0` exact."""
    return state.replace(gU=_implicit_r("U", kappaRU, myTime, myIter, cfg=cfg, grid=grid, params=params,
                                        state=state))


def mom_v_implicit_r(kappaRV, myTime, myIter, *, cfg, grid, params, state):
    """MOM_V_IMPLICIT_R( kappaRV, bi, bj, myTime, myIter, myThid )   @63cdc0b pkg/mom_common/mom_v_implicit_r.F:8-400

    C     | o Solve implicitly vertical advection & diffusion of momentum, meridional component

    As mom_u_implicit_r with maskS, recip_hFacS, gV, kappaRV and iMin..iMax = 1..sNx, jMin..jMax = 1..sNy+1."""
    return state.replace(gV=_implicit_r("V", kappaRV, myTime, myIter, cfg=cfg, grid=grid, params=params,
                                        state=state))
