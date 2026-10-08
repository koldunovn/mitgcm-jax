"""TIMESTEP: model/src/timestep.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j
from mitjax.model.src.adams_bashforth2 import adams_bashforth2
from mitjax.model.src.apply_forcing import apply_forcing_u, apply_forcing_v


def timestep(iMin, iMax, jMin, jMax, k, dPhiHydX, dPhiHydY, phiSurfX, phiSurfY, guDissip, gvDissip, myTime, myIter,
             *, cfg, grid, params, state, ff, fp):
    """TIMESTEP( bi, bj, iMin, iMax, jMin, jMax, k, dPhiHydX,dPhiHydY, phiSurfX, phiSurfY, guDissip, gvDissip,
    myTime, myIter, myThid )   @63cdc0b model/src/timestep.F:10-429

    C     | S/R TIMESTEP
    C     | o Step model fields forward in time
    C     dPhiHydX,Y :: Gradient (X & Y directions) of Hydrostatic Potential
    C     phiSurfX,Y :: Gradient (X & Y directions) of Surface Potential
    C     guDissip   :: dissipation tendency (all explicit terms), u component
    C     gvDissip   :: dissipation tendency (all explicit terms), v component

    Writes gU, gV (DYNVARS.h: the new velocity before the pressure correction) and guNm1, gvNm1 (AB2 history);
    returns the State. `ff`: FFIELDS.h, `fp`: the forcing parameters (FORCING lane: APPLY_FORCING_U/V). Not compiled in M1 builds and not ported:
    ALLOW_ADAMSBASHFORTH_3 (:150-171, :186-196 raise), ALLOW_CD_CODE (:227-320, global_ocean / optim: raises), the
    non-hydrostatic / implicit-int-grav-wave and debug blocks. GO lane (plan Task 15a): ALLOW_CD_CODE with
    useCDscheme (:97-100, :228-269: CD_CODE_SCHEME writes uVelD, vVelD, uNM1, vNM1 into the State; the `#else` arms
    of CD_CODE_NO_AB_MOMENTUM) and the NONLIN_FRSURF r* rescaling (:272-284: gUtmp/rStarExpW, gVtmp/rStarExpS; the
    divisions are as written: rStarExpW/S are never 0 on wet or dry points, calc_r_star.F:306-311 guards their
    zero denominator and counts it); the sigma arm (:285-300) raises. Lane B (Task 25, input.nlfs): the hFac_surf arm
    (:301-317) at the surface level kSurfW/S, the division guarded by its own condition (mitjax/ops/safe.py). The REAL test `implicSurfPress.NE.oneRL` (:328) is a
    `where` of two finite values (implicSurfPress is traced). Points are independent."""
    ab3 = bool(cfg.cpp.ALLOW_ADAMSBASHFORTH_3)     # GO lane: ADAMS_BASHFORTH3 of gU, gV with guNm, gvNm (:162-172)
    if cfg.cpp.ALLOW_CD_CODE and cfg.cpp.flag("CD_CODE_NO_AB_MOMENTUM", "CD_CODE_OPTIONS.h"):
        raise NotImplementedError("TIMESTEP: CD_CODE_NO_AB_MOMENTUM (:148-158, :195-205, :237-260) is not ported")
    sz = cfg.size
    Nr = sz.Nr
    gU, gV = state.gU, state.gV
    # AB histories: DYNVARS.h guNm, gvNm (pairs, ALLOW_ADAMSBASHFORTH_3) or guNm1, gvNm1
    guNm1, gvNm1 = (state.guNm, state.gvNm) if ab3 else (state.guNm1, state.gvNm1)
    uVel, vVel = state.uVel, state.vVel
    maskW, maskS = grid.maskW, grid.maskS
    psFac = (params.pfFacMom*(1. - params.implicSurfPress)                      # :80-81  1. _d 0
             * grid.recip_deepFacC[k]*params.recip_rhoFacC[k])
    phFac = params.pfFacMom                                                     # :84
    guExt = dPhiHydX.local("guExt")
    gvExt = dPhiHydX.local("gvExt")
    gUtmp = dPhiHydX.local("gUtmp")
    gVtmp = dPhiHydX.local("gVtmp")
    gu_AB = dPhiHydX.local("gu_AB")
    gv_AB = dPhiHydX.local("gv_AB")
    gUdPx = dPhiHydX.local("gUdPx")
    gVdPy = dPhiHydX.local("gVdPy")
    j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)                                         # :89
    i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)                                         # :90
    guExt = guExt.at[i, j].set(0.)                                              # :91  0. _d 0
    gvExt = gvExt.at[i, j].set(0.)                                              # :92
    gUtmp = gUtmp.at[i, j].set(0.)                                              # :93
    gVtmp = gVtmp.at[i, j].set(0.)                                              # :94
    gUdPx = gUdPx.at[i, j].set(0.)                                              # :95
    gVdPy = gVdPy.at[i, j].set(0.)                                              # :96
    if cfg.cpp.ALLOW_CD_CODE:                                                   # :97-100 (GO lane)
        guCor = dPhiHydX.local("guCor").at[i, j].set(0.)                        # :98  0. _d 0
        gvCor = dPhiHydX.local("gvCor").at[i, j].set(0.)                        # :99

    if params.momForcing:                                                       # :104-114
        old = dict(gU=gU, gV=gV) if cfg.cpp.USE_OLD_EXTERNAL_FORCING else {}    # lane B: the State's gU, gV
        guExt = apply_forcing_u(guExt, iMin, iMax, jMin, jMax, k, myTime, myIter, cfg=cfg, grid=grid,
                                fp=fp, ff=ff, params=params, **({"gU": gU} if old else {}))
        gvExt = apply_forcing_v(gvExt, iMin, iMax, jMin, jMax, k, myTime, myIter, cfg=cfg, grid=grid,
                                fp=fp, ff=ff, params=params, **({"gV": gV} if old else {}))

    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    if not params.staggerTimeStep and not params.implicitIntGravWave:          # :116-126
        gU = gU.at[i, j, k].set(gU[i, j, k] - phFac*dPhiHydX[i, j])             # :120
        gV = gV.at[i, j, k].set(gV[i, j, k] - phFac*dPhiHydY[i, j])             # :121

    if params.momViscosity and params.momDissip_In_AB:                         # :129-136
        gU = gU.at[i, j, k].set(gU[i, j, k] + guDissip[i, j])                   # :132
        gV = gV.at[i, j, k].set(gV[i, j, k] + gvDissip[i, j])                   # :133

    if params.momForcing and params.momForcingOutAB != 1:                      # :139-146
        gU = gU.at[i, j, k].set(gU[i, j, k] + guExt[i, j])                      # :142
        gV = gV.at[i, j, k].set(gV[i, j, k] + gvExt[i, j])                      # :143

    if ab3:                                                                     # :162-172 (GO lane)
        from mitjax.model.src.adams_bashforth3 import adams_bashforth3
        gU, guNm1, gu_AB = adams_bashforth3(k, Nr, gU, guNm1, gu_AB, params.mom_StartAB, myIter,
                                            cfg=cfg, params=params)
        gV, gvNm1, gv_AB = adams_bashforth3(k, Nr, gV, gvNm1, gv_AB, params.mom_StartAB, myIter,
                                            cfg=cfg, params=params)
    else:
        gU, guNm1, gu_AB = adams_bashforth2(k, Nr, gU, guNm1, gu_AB, params.mom_StartAB, myIter,   # :174-179
                                            cfg=cfg, params=params)
        gV, gvNm1, gv_AB = adams_bashforth2(k, Nr, gV, gvNm1, gv_AB, params.mom_StartAB, myIter,   # :180-185
                                            cfg=cfg, params=params)
    hist = dict(guNm=guNm1, gvNm=gvNm1) if ab3 else dict(guNm1=guNm1, gvNm1=gvNm1)

    gUtmp = gUtmp.at[i, j].set(gU[i, j, k])                                     # :200
    gVtmp = gVtmp.at[i, j].set(gV[i, j, k])                                     # :201

    if params.momForcing and params.momForcingOutAB == 1:                      # :209-216
        gUtmp = gUtmp.at[i, j].set(gUtmp[i, j] + guExt[i, j])                   # :212
        gVtmp = gVtmp.at[i, j].set(gVtmp[i, j] + gvExt[i, j])                   # :213

    if params.momViscosity and not params.momDissip_In_AB:                     # :219-226
        gUtmp = gUtmp.at[i, j].set(gUtmp[i, j] + guDissip[i, j])                # :222
        gVtmp = gVtmp.at[i, j].set(gVtmp[i, j] + gvDissip[i, j])                # :223

    if cfg.cpp.ALLOW_CD_CODE and params.useCDscheme:                           # :228-269 (GO lane)
        # :232-235 step forward the D-grid velocity with gUtmp, gVtmp = gU,V +dissip +forcing; Coriolis on the C grid
        state = state.replace(gU=gU, gV=gV, **hist)
        from mitjax.pkg.cd_code.cd_code_scheme import cd_code_scheme
        state, guCor, gvCor = cd_code_scheme(k, dPhiHydX, dPhiHydY, gUtmp, gVtmp, guCor, gvCor, myTime, myIter,
                                             cfg=cfg, grid=grid, params=params, state=state)
        gUtmp = gUtmp.at[i, j].set(gUtmp[i, j] + guCor[i, j])                   # :264  (#else of NO_AB_MOMENTUM)
        gVtmp = gVtmp.at[i, j].set(gVtmp[i, j] + gvCor[i, j])                   # :265

    if cfg.cpp.NONLIN_FRSURF and not params.vectorInvariantMomentum and params.nonlinFreeSurf > 1:   # :272-319
        if params.select_rStar > 0:                                             # :275
            if not cfg.cpp.DISABLE_RSTAR_CODE:
                gUtmp = gUtmp.at[i, j].set(gUtmp[i, j]/state.rStarExpW[i, j])   # :280  (GO lane)
                gVtmp = gVtmp.at[i, j].set(gVtmp[i, j]/state.rStarExpS[i, j])   # :281
        elif params.selectSigmaCoord_ne_0:                                      # :285-300
            raise NotImplementedError("TIMESTEP: the hybrid sigma rescaling (:285-300) is not ported")
        else:                                                                   # :301-317 (lane B, Task 25)
            from mitjax.ops.safe import safe_div
            atW = grid.kSurfW[i, j] == k                                        # :305
            atS = grid.kSurfS[i, j] == k                                        # :309
            gUtmp = gUtmp.at[i, j].set(jnp.where(                               # :306-307 _hFacW: hFacW
                atW, safe_div(gUtmp[i, j]*grid.hFacW[i, j, k], state.hFac_surfW[i, j], atW), gUtmp[i, j]))
            gVtmp = gVtmp.at[i, j].set(jnp.where(                               # :310-311
                atS, safe_div(gVtmp[i, j]*grid.hFacS[i, j, k], state.hFac_surfS[i, j], atS), gVtmp[i, j]))

    if params.staggerTimeStep or params.implicitIntGravWave:                   # :321-327
        gUdPx = gUdPx.at[i, j].set(-phFac*dPhiHydX[i, j] - psFac*phiSurfX[i, j])   # :324
        gVdPy = gVdPy.at[i, j].set(-phFac*dPhiHydY[i, j] - psFac*phiSurfY[i, j])   # :325
    else:                                                                       # :328-334 ELSEIF on a REAL
        ns = params.implicSurfPress != 1.                                       # oneRL = 1.0 _d 0
        gUdPx = gUdPx.at[i, j].set(jnp.where(ns, -psFac*phiSurfX[i, j], gUdPx[i, j]))   # :331
        gVdPy = gVdPy.at[i, j].set(jnp.where(ns, -psFac*phiSurfY[i, j], gVdPy[i, j]))   # :332

    gU = gU.at[i, j, k].set(uVel[i, j, k]                                       # :375-377
                            + params.deltaTMom*(gUtmp[i, j] + gUdPx[i, j])
                            * maskW[i, j, k])
    gV = gV.at[i, j, k].set(vVel[i, j, k]                                       # :384-386
                            + params.deltaTMom*(gVtmp[i, j] + gVdPy[i, j])
                            * maskS[i, j, k])
    return state.replace(gU=gU, gV=gV, **hist)
