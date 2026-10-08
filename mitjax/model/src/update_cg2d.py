"""UPDATE_CG2D: model/src/update_cg2d.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.eesupp.exch_rs import EXCH_XY_RS
from mitjax.ops.safe import safe_div


def update_cg2d(cg2dh, myTime, myIter, *, cfg, grid, surface, params, ex):
    """UPDATE_CG2D( myTime, myIter, myThid )   @63cdc0b model/src/update_cg2d.F:7-195

    C     | SUBROUTINE UPDATE_CG2D
    C     | o Update 2d conjugate gradient solver operators
    C     |   account for Free-Surf effect on total column thickness
    C     | This routine is based on INI_CG2D, and simplified. It is
    C     | used when the non-linear free surface mode is activated
    C     | or when bottom depth is part of the control vector.

    Called by FORWARD_STEP (forward_step.F:866-871: momStepping and nonlinFreeSurf > 2) and INITIALISE_VARIA
    (initialise_varia.F:325-327: nonlinFreeSurf > 2) under NONLIN_FRSURF; the caller decides. Reads CG2D.h
    (`cg2dh`: cg2dNorm, and pW, pS, pC, aC2d whose points this routine does not write), (GRID.h) dyG, dxG, drF, the
    current hFacW, hFacS, recip_dxC, recip_dyC, rA from `grid`, recip_Bo from `surface`, the Cg2dParams `params`.
    `myIter`: a Python int, or anything when cg2dPreCondFreq is 0 or 1 (then the decision :57-62 does not depend on
    it). Returns the updated CG2DH (aW2d, aS2d, aC2d and, when the preconditioner is updated, pW, pS, pC).

    Vectorisation: the k loop :98-112 is a sum over k: a Python loop over k in the Fortran order, each level
    vectorised over i, j; every other loop computes each point from inputs only. Pointwise IFs (:168-186) are
    `where`s with the division guarded before it.

    Ported for the M1 variants (global_ocean.90x40x15: nonlinFreeSurf = 4): the default operator (:98-112) and the
    shallow-atmosphere main diagonal (:144-153). Raise: ALLOW_SOLVE4_PS_AND_DRAG with selectImplicitDrag = 2
    (:78-97), ALLOW_OBCS (:120-127), deepAtmosphere (:131-142). ALLOW_AUTODIFF zeroes aC2d incl. halos (:73-75) and is
    ported.
    """
    sz = cfg.size
    sNx, sNy, OLx, OLy, Nr = sz.sNx, sz.sNy, sz.OLx, sz.OLy, sz.Nr
    if cfg.cpp.ALLOW_SOLVE4_PS_AND_DRAG and params.selectImplicitDrag == 2:
        raise NotImplementedError("UPDATE_CG2D: selectImplicitDrag = 2 (update_cg2d.F:78-97) is not ported")
    if cfg.cpp.ALLOW_OBCS:
        raise NotImplementedError("UPDATE_CG2D: ALLOW_OBCS (maskInC factors, update_cg2d.F:120-127) is not ported")
    if params.deepAtmosphere:
        raise NotImplementedError("UPDATE_CG2D: deepAtmosphere (update_cg2d.F:131-142) is not ported")
    dyG, dxG, drF, hFacW, hFacS = grid.dyG, grid.dxG, grid.drF, grid.hFacW, grid.hFacS
    recip_dxC, recip_dyC, rA = grid.recip_dxC, grid.recip_dyC, grid.rA
    recip_Bo = surface.recip_Bo
    aW2d, aS2d, aC2d, pW, pS, pC = cg2dh.aW2d, cg2dh.aS2d, cg2dh.aC2d, cg2dh.pW, cg2dh.pS, cg2dh.pC
    cg2dNorm = cg2dh.cg2dNorm
    implicSurfPress, implicDiv2DFlow = params.implicSurfPress, params.implicDiv2DFlow
    freeSurfFac, deltaTMom, deltaTFreeSurf = params.freeSurfFac, params.deltaTMom, params.deltaTFreeSurf
    cg2dpcOffDFac = params.cg2dpcOffDFac

    # :56-62  Decide when to update cg2d Preconditioner
    if params.cg2dPreCondFreq == 0:
        updatePreCond = False
    elif params.cg2dPreCondFreq == 1:
        updatePreCond = True                    # MOD(myIter,1) .EQ. 0 for every myIter
    else:
        if not isinstance(myIter, int):
            raise NotImplementedError("UPDATE_CG2D: cg2dPreCondFreq > 1 with a traced myIter is not ported")
        updatePreCond = (myIter == params.nIter0)
        if myIter % params.cg2dPreCondFreq == 0:
            updatePreCond = True

    # :64-77  Initialise laplace operator
    jf = loop_j(1 - OLy, sNy + OLy)
    i_f = loop_i(1 - OLx, sNx + OLx)
    aW2d = aW2d.at[i_f, jf].set(0.0)
    aS2d = aS2d.at[i_f, jf].set(0.0)
    if cfg.cpp.ALLOW_AUTODIFF:
        aC2d = aC2d.at[i_f, jf].set(0.0)                                        # :73-75
    # :98-112
    j = loop_j(1, sNy + 1)
    i = loop_i(1, sNx + 1)
    for k in range(1, Nr + 1):
        faceArea = dyG[i, j]*drF[k] \
            * hFacW[i, j, k]                                                    # :102-103
        aW2d = aW2d.at[i, j].set(aW2d[i, j]
                                 + faceArea*recip_dxC[i, j])                    # :104-105
        faceArea = dxG[i, j]*drF[k] \
            * hFacS[i, j, k]                                                    # :106-107
        aS2d = aS2d.at[i, j].set(aS2d[i, j]
                                 + faceArea*recip_dyC[i, j])                    # :108-109
    # :116-129
    aW2d = aW2d.at[i, j].set(aW2d[i, j]*cg2dNorm
                             * implicSurfPress*implicDiv2DFlow)                 # :118-119
    aS2d = aS2d.at[i, j].set(aS2d[i, j]*cg2dNorm
                             * implicSurfPress*implicDiv2DFlow)                 # :123-124
    # :130-154  compute matrix main diagonal (deepAtmosphere = .FALSE.: :144-153)
    j1 = loop_j(1, sNy)
    i1 = loop_i(1, sNx)
    aC2d = aC2d.at[i1, j1].set(-(
        aW2d[i1, j1] + aW2d[i1+1, j1]
        + aS2d[i1, j1] + aS2d[i1, j1+1]
        + freeSurfFac*cg2dNorm*recip_Bo[i1, j1]
        * rA[i1, j1]/deltaTMom/deltaTFreeSurf
    ))                                                                          # :146-151

    if updatePreCond:
        # :160-161  Update overlap regions
        aC2d = EXCH_XY_RS(aC2d, ex=ex)
        # :163-190  Initialise preconditioner
        aC = aC2d[i, j]
        pC = pC.at[i, j].set(jnp.where(aC == 0., 1.0,
                                       safe_div(1.0, aC, aC != 0.)))            # :168-172
        pW_tmp = aC2d[i, j]+aC2d[i-1, j]                                        # :173
        pW = pW.at[i, j].set(jnp.where(pW_tmp == 0., 0.0,                       # :174-179
                                       -safe_div(aW2d[i, j], (cg2dpcOffDFac*pW_tmp)**2, pW_tmp != 0.)))
        pS_tmp = aC2d[i, j]+aC2d[i, j-1]                                        # :180
        pS = pS.at[i, j].set(jnp.where(pS_tmp == 0., 0.0,                       # :181-186
                                       -safe_div(aS2d[i, j], (cg2dpcOffDFac*pS_tmp)**2, pS_tmp != 0.)))

    return cg2dh.replace(aW2d=aW2d, aS2d=aS2d, aC2d=aC2d, pW=pW, pS=pS, pC=pC)
