"""CD_CODE_SCHEME: pkg/cd_code/cd_code_scheme.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j


def cd_code_scheme(k, dPhiHydX, dPhiHydY, guFld, gvFld, guCor, gvCor, myTime, myIter, *, cfg, grid, params,
                   state):
    """CD_CODE_SCHEME( bi,bj,k, dPhiHydX,dPhiHydY, guFld,gvFld, guCor,gvCor, myTime, myIter, myThid )
    @63cdc0b pkg/cd_code/cd_code_scheme.F:7-252

    C !DESCRIPTION:
    C The C-D scheme. The less said the better :-)
    C  k                    :: vertical level
    C     dPhiHydX,Y        :: Gradient (X & Y dir.) of Hydrostatic Potential
    C  guFld,gvFld          :: Acceleration (U & V compon.) from the C grid
    C  guCor,gvCor          :: Coriolis terms (2 compon.) computed on C grid
    C  myTime               :: current time
    C  myIter               :: current time-step number

    Returns (state, guCor, gvCor): the State with CD_CODE_VARS.h uVelD, vVelD, uNM1, vNM1 written at level k (a
    Python int). Reads (DYNVARS.h) etaN, uVel, vVel, (CD_CODE_VARS.h) etaNm1, uVelD, vVelD, uNM1, vNM1 from `state`;
    (GRID.h, SURFACE.h) Bo_surf, recip_dxC, recip_dyC, maskW, maskS, fCori from `grid`; epsAB_CD, rCD, deltaTMom,
    cfFacMom, pfFacMom (traced) and staggerTimeStep (static) from `params`. guCor, gvCor: the caller's arrays,
    written on jMin..jMax x iMin..iMax only (the dummy initialisation :88-94 is commented out in the Fortran).

    Ported (CD_CODE_OPTIONS.h: CD_CODE_NO_AB_CORIOLIS undefined, the `#else` arms :108-109, :133, :190): every loop
    in its statement order; points are independent inside each nest. `IF (myIter.EQ.0)` (:66-72) on the traced
    counter is a `where` of two finite values (ab05 = -0. _d 0 at myIter = 0, kept as the signed zero). Raise:
    ALLOW_OBCS (maskInC factors :122-124, :178-180), CD_CODE_NO_AB_CORIOLIS."""
    if cfg.cpp.ALLOW_OBCS:
        raise NotImplementedError("CD_CODE_SCHEME: ALLOW_OBCS (maskInC factors) is not ported")
    if cfg.cpp.flag("CD_CODE_NO_AB_CORIOLIS", "CD_CODE_OPTIONS.h"):
        raise NotImplementedError("CD_CODE_SCHEME: CD_CODE_NO_AB_CORIOLIS is not ported")
    sz = cfg.size
    OLx, OLy, sNx, sNy = sz.OLx, sz.OLy, sz.sNx, sz.sNy
    g = grid
    etaN, etaNm1, uVel, vVel = state.etaN, state.etaNm1, state.uVel, state.vVel
    uVelD, vVelD, uNM1, vNM1 = state.uVelD, state.vVelD, state.uNM1, state.vNM1
    iMin, iMax = 1-OLx+1, sNx+OLx-1                                             # :59-60  PARAMETER
    jMin, jMax = 1-OLy+1, sNy+OLy-1                                             # :61

    # :66-72  Adams-Bashforth weighting factors
    it0 = myIter == 0
    ab15 = jnp.where(it0, 1., 1.5 + params.epsAB_CD)                            # :67 1. _d 0 / :70 1.5 _d 0 + epsAB_CD
    ab05 = jnp.where(it0, -0., -0.5 - params.epsAB_CD)                          # :68 -0. _d 0 / :71 -0.5 _d 0 - epsAB_CD

    # :74-81  stagger time stepping: grad Phi_Hyp is not in gU,gV and needs to be added
    if params.staggerTimeStep:
        phxFac = params.pfFacMom                                                # :76
        phyFac = params.pfFacMom                                                # :77
    else:
        phxFac = 0.                                                             # :79  0. (REAL*4, exact)
        phyFac = 0.                                                             # :80

    pF = dPhiHydX.local("pF")
    vF = dPhiHydX.local("vF")
    aF = dPhiHydX.local("aF")
    # :97-111  Pressure extrapolated forward in time
    j = loop_j(1-OLy, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx)
    pF = pF.at[i, j].set(g.Bo_surf[i, j]*(ab15*etaN[i, j] + ab05*etaNm1[i, j]))   # :108-109

    # :113-127  grady(p) + gV
    j = loop_j(1-OLy+1, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx)
    aF = aF.at[i, j].set((gvFld[i, j]                                            # :117-121
                          - (g.recip_dyC[i, j]*(pF[i, j]-pF[i, j-1])
                             + phyFac*dPhiHydY[i, j])
                          )*g.maskS[i, j, k])
    # :128-137  Average to Vd point and add coriolis
    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    vF = vF.at[i, j].set((aF[i, j]+aF[i-1, j+1]                                  # :130-133
                          + (aF[i-1, j]+aF[i, j+1]))*0.25
                         * g.maskW[i, j, k]
                         - (g.fCori[i, j]
                            + g.fCori[i-1, j])*0.5
                         * (ab15*uVel[i, j, k] + ab05*uNM1[i, j, k]))
    # :139-143  Step forward Vd
    vVelD = vVelD.at[i, j, k].set(vVelD[i, j, k] + params.deltaTMom*vF[i, j])    # :141
    # :144-159  Relax D grid V to C grid V
    vVelD = vVelD.at[i, j, k].set((params.rCD*vVelD[i, j, k]                     # :147-157
                                   + (1. - params.rCD)
                                   * (ab15*((vVel[i, j, k]+vVel[i-1, j+1, k])
                                            + (vVel[i-1, j, k]+vVel[i, j+1, k])
                                            )*0.25
                                      + ab05*((vNM1[i, j, k]+vNM1[i-1, j+1, k])
                                              + (vNM1[i-1, j, k]+vNM1[i, j+1, k])
                                              )*0.25
                                      ))*g.maskW[i, j, k])
    # :160-167  Calculate coriolis force on U
    guCor = guCor.at[i, j].set((g.fCori[i, j]                                    # :163-166
                                + g.fCori[i-1, j])*0.5
                               * vVelD[i, j, k]*params.cfFacMom)

    # :169-183  gradx(p)+gU
    j = loop_j(1-OLy, sNy+OLy)
    i = loop_i(1-OLx+1, sNx+OLx)
    aF = aF.at[i, j].set((guFld[i, j]                                            # :173-177
                          - (g.recip_dxC[i, j]*(pF[i, j]-pF[i-1, j])
                             + phxFac*dPhiHydX[i, j])
                          )*g.maskW[i, j, k])
    # :184-193  Average to Ud point and add coriolis
    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    vF = vF.at[i, j].set((aF[i, j]+aF[i+1, j-1]                                  # :186-190
                          + (aF[i+1, j]+aF[i, j-1]))*0.25
                         * g.maskS[i, j, k]
                         + (g.fCori[i, j]
                            + g.fCori[i, j-1])*0.5
                         * (ab15*vVel[i, j, k] + ab05*vNM1[i, j, k]))
    # :203-208  Step forward Ud
    uVelD = uVelD.at[i, j, k].set(uVelD[i, j, k] + params.deltaTMom*vF[i, j])    # :206
    # :209-223  Relax D grid U to C grid U
    uVelD = uVelD.at[i, j, k].set((params.rCD*uVelD[i, j, k]                     # :212-222
                                   + (1. - params.rCD)
                                   * (ab15*((uVel[i, j, k]+uVel[i+1, j-1, k])
                                            + (uVel[i, j-1, k]+uVel[i+1, j, k])
                                            )*0.25
                                      + ab05*((uNM1[i, j, k]+uNM1[i+1, j-1, k])
                                              + (uNM1[i, j-1, k]+uNM1[i+1, j, k])
                                              )*0.25
                                      ))*g.maskS[i, j, k])
    # :224-232  Calculate coriolis force on V
    gvCor = gvCor.at[i, j].set(-((g.fCori[i, j]                                  # :227-230 (Fortran unary minus
                                  + g.fCori[i, j-1])*0.5                         # applies to the whole product)
                                 * uVelD[i, j, k]*params.cfFacMom))

    # :234-240  Save "previous time level" variables
    j = loop_j(1-OLy, sNy+OLy)
    i = loop_i(1-OLx, sNx+OLx)
    uNM1 = uNM1.at[i, j, k].set(uVel[i, j, k])                                   # :237
    vNM1 = vNM1.at[i, j, k].set(vVel[i, j, k])                                   # :238
    return state.replace(uVelD=uVelD, vVelD=vVelD, uNM1=uNM1, vNM1=vNM1), guCor, gvCor
