"""UPDATE_R_STAR: model/src/update_r_star.F @63cdc0b."""

from mitjax.farray import loops_kji
from mitjax.ops.safe import safe_div

import jax.numpy as jnp


def update_r_star(useLatest, myTime, myIter, *, cfg, grid, state):
    """UPDATE_R_STAR( useLatest, myTime, myIter, myThid )   @63cdc0b model/src/update_r_star.F:6-134

    C     | SUBROUTINE UPDATE_R_STAR
    C     | o Update the thickness fractions (hFacC,W,S)
    C     |   according to the surface r-position = Non-Linear FrSurf
    C     useLatest :: if true use rStarFacC, else use rStarFacNm1C
    C     myTime    :: Current time in simulation
    C     myIter    :: Current iteration number in simulation

    Returns the Grid with the GRID.h arrays the routine writes: hFacC, hFacW, hFacS (every point) and recip_hFacC,
    recip_hFacW, recip_hFacS (where maskC/W/S .NE. 0; every other point keeps its prior value, :71-76, :106-111).
    Reads h0FacC/W/S, maskC/W/S and the prior recip_hFac* from `grid`; rStarFacC/W/S (useLatest = .TRUE.,
    :48-80) or rStarFacNm1C/W/S (useLatest = .FALSE., :82-115) from `state`. `useLatest` is a static Python bool (a
    LOGICAL argument that selects the branch).

    Ported: the `#else` arms of USE_MASK_AND_NO_IF (:70-77, :105-112), which every M1 build compiles; the macro
    _hFacC is hFacC (GRID.h, no HFACC_MACROS override in M1). Raise: USE_MASK_AND_NO_IF (:63-69, :98-104),
    DISABLE_RSTAR_CODE (the routine is then empty, but no caller reaches it: FORWARD_STEP and INITIALISE_VARIA call it
    only without DISABLE_RSTAR_CODE). The exchanges after the loop (:125-128) are commented out in the Fortran.

    Vectorisation: the DO k / DO j / DO i iterations (:50-80, :84-115) are independent (each point reads only
    inputs of the routine and the hFac it has just written at the same point): one k-vectorised nest. The IF on
    the mask is a `where` whose division is guarded by the same mask (safe_div; hFac = h0Fac*rStarFac > 0 there)."""
    if cfg.cpp.USE_MASK_AND_NO_IF:
        raise NotImplementedError("UPDATE_R_STAR: USE_MASK_AND_NO_IF (update_r_star.F:63-69, :98-104) is not ported")
    if cfg.cpp.DISABLE_RSTAR_CODE:
        raise NotImplementedError("UPDATE_R_STAR: DISABLE_RSTAR_CODE (empty routine) is not ported")
    sz = cfg.size
    g = grid
    k, j, i = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
    if useLatest:
        fC, fW, fS = state.rStarFacC, state.rStarFacW, state.rStarFacS                     # :55-60
    else:
        fC, fW, fS = state.rStarFacNm1C, state.rStarFacNm1W, state.rStarFacNm1S            # :90-95
    hFacC = g.hFacC.at[i, j, k].set(g.h0FacC[i, j, k]
                                    * fC[i, j])                                             # :55-56 / :90-91
    hFacW = g.hFacW.at[i, j, k].set(g.h0FacW[i, j, k]
                                    * fW[i, j])                                             # :57-58 / :92-93
    hFacS = g.hFacS.at[i, j, k].set(g.h0FacS[i, j, k]
                                    * fS[i, j])                                             # :59-60 / :94-95
    # :71-76 (useLatest): `1. _d 0 / _hFacC`; :106-111: `oneRS / _hFacC` (oneRS = 1.0 _d 0, EEPARAMS.h:69): the same
    # double 1.0; masks compared with `0.` (REAL*4 zero) and with zeroRS = 0.0 _d 0 (EEPARAMS.h:69): the same zero.
    recip_hFacC = _recip(g.recip_hFacC, g.maskC, hFacC, i, j, k)                            # :71-72 / :106-107
    recip_hFacW = _recip(g.recip_hFacW, g.maskW, hFacW, i, j, k)                            # :73-74 / :108-109
    recip_hFacS = _recip(g.recip_hFacS, g.maskS, hFacS, i, j, k)                            # :75-76 / :110-111
    return g.replace(hFacC=hFacC, hFacW=hFacW, hFacS=hFacS,
                     recip_hFacC=recip_hFacC, recip_hFacW=recip_hFacW, recip_hFacS=recip_hFacS)


def _recip(recip, mask, hFac, i, j, k):
    """IF (mask(i,j,k).NE.0.) recip(i,j,k) = 1. _d 0 / hFac(i,j,k)  (else the prior value)."""
    wet = mask[i, j, k] != 0.
    return recip.at[i, j, k].set(jnp.where(wet, safe_div(1., hFac[i, j, k], wet), recip[i, j, k]))
