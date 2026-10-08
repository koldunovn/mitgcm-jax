"""CALC_PHI_HYD: model/src/calc_phi_hyd.F @63cdc0b (the OCEANIC and ATMOSPHERIC branches)."""

import jax.numpy as jnp

from mitjax.farray import FArray, loop_i, loop_j
from mitjax.model.src.calc_grad_phi_hyd import calc_grad_phi_hyd
from mitjax.model.src.diags_phi_hyd import diags_phi_hyd
from mitjax.model.src.diags_phi_rlow import diags_phi_rlow
from mitjax.ops.fortran_minmax import MAX, MIN


def calc_phi_hyd(iMin, iMax, jMin, jMax, k, phiHydF, phiHydC, dPhiHydX, dPhiHydY, myTime, myIter, *, cfg, grid,
                 params, state, phi0surf, eos=None):
    """CALC_PHI_HYD( bi, bj, iMin, iMax, jMin, jMax, k, phiHydF, phiHydC, dPhiHydX, dPhiHydY, myTime, myIter,
    myThid )   @63cdc0b model/src/calc_phi_hyd.F:10-676

    C     | SUBROUTINE CALC_PHI_HYD
    C     | o Integrate the hydrostatic relation to find the Hydros. Potential (ocean: Pressure/rho ;
    C     |   atmos = geopotential)
    C     phiHydF  :: hydrostatic potential anomaly at middle between 2 centers k & k+1 (interface k+1)
    C     phiHydC  :: hydrostatic potential anomaly at cell center
    C     dPhiHydX,Y :: gradient (X & Y dir.) of hydrostatic potential anom.

    Returns (state, phiHydF, phiHydC, dPhiHydX, dPhiHydY): the State carries totPhiHyd and phiHydLow (DIAGS_PHI_HYD,
    DIAGS_PHI_RLOW). Ported: buoyancyRelation = 'OCEANIC' (:130-291) with alphaRho = rhoInSitu (:151-155), both
    integr_GeoPot arms and both uniformFreeSurfLev arms; then CALC_GRAD_PHI_HYD (:621-630), DIAGS_PHI_RLOW, DIAGS_PHI_HYD.
    GO lane: quasiHydrostatic (MOM_QUASIHYDROSTATIC, :179-186) and the NONLIN_FRSURF terms with surfPhiFac = 0.
    (select_rStar /= 0: ddRloc + surfPhiFac*etaH at :229, :275). Lane B (Task 25): 'ATMOSPHERIC' (:417-602) with
    alphaRho = rhoInSitu (:440-446) and the integr_GeoPot = 0, 1, 2/3 arms; the Exner factors
    (x/atm_Po)**atm_kappa are glibc's pow evaluated on the host (Params rF_Po_kappa, rC_Po_kappa: ini_parms._atm_traced),
    the arithmetic around them is the Fortran's, in its order. Raise: 'OCEANICP' (:294-415),
    useFVgradPhi (selectSigmaCoord), ALLOW_SHELFICE. GOADK lane: addSurfPhiAnom (select_rStar = 0 with nonlinFreeSurf
    >= 4: surfPhiFac = 1., :96-100, and the surface phiHydF of :191-200 with uniformFreeSurfLev at k = 1).
    The arm `implicitIntGravWave .OR. myIter.LT.0` (:138: FIND_RHO_2D) is static here: implicitIntGravWave raises and
    myIter >= nIter0 >= 0 in every forward run (the_main_loop counters), so nIter0 < 0 raises. Pointwise IFs on
    kSurfC are `where`s; MAX/MIN with mitjax.ops.fortran_minmax."""
    if params.buoyancyRelation not in ("OCEANIC", "ATMOSPHERIC"):                # :130, :417 (ATMOSPHERIC: lane B)
        raise NotImplementedError("CALC_PHI_HYD: only buoyancyRelation='OCEANIC' / 'ATMOSPHERIC' is ported")
    if cfg.cpp.ALLOW_SHELFICE:
        raise NotImplementedError("CALC_PHI_HYD: ALLOW_SHELFICE terms are not ported")
    # vermix lane (M3 Task 30): INI_PRESSURE calls with myIter = -1 (a static Python int): the FIND_RHO_2D arm of
    # :138-149 (`eos` required); in the time loop myIter >= nIter0 >= 0 (traced) and the rhoInSitu arm (:151-155) runs
    rho_from_eos = isinstance(myIter, int) and myIter < 0
    if params.implicitIntGravWave or (params.nIter0 < 0 and not rho_from_eos):
        raise NotImplementedError("CALC_PHI_HYD: alphaRho from FIND_RHO_2D (:138-149) in the time loop is not ported")
    if rho_from_eos and eos is None:
        raise ValueError("CALC_PHI_HYD: myIter < 0 (FIND_RHO_2D, :144-149) needs `eos`")
    addSurfPhiAnom = params.select_rStar == 0 and params.nonlinFreeSurf >= 4    # :96
    surfPhiFac = 0.                                                             # :99  0. (REAL*4, exact)
    if addSurfPhiAnom:                                                          # :100 (GOADK lane: bottomdrag NLFS=4)
        surfPhiFac = 1.                                                         #  1. (REAL*4, exact)
    sz = cfg.size
    Nr = sz.Nr
    g = grid
    gravity, recip_rhoConst = params.gravity, params.recip_rhoConst
    gravFacC, gravFacF = params.gravFacC, params.gravFacF
    halfRL = 0.5                                                                # EEPARAMS.h:73
    if k == 1:                                                                  # :121-127
        jA = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
        iA = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
        phiHydF = phiHydF.at[iA, jA].set(0.)
    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    if rho_from_eos and params.buoyancyRelation == "OCEANIC":                   # :138-149 (vermix lane)
        from mitjax.model.src.find_rho import find_rho_2d
        (_, ilo, ihi), (_, jlo, jhi), (_, klo, _) = state.theta.dims
        lev = lambda A: FArray(A.data[:, k - klo], A.name, i=(ilo, ihi), j=(jlo, jhi))   # noqa: E731 A(.,.,k)
        alphaRho = find_rho_2d(iMin, iMax, jMin, jMax, k, lev(state.theta), lev(state.salt),
                               phiHydF.local("alphaRho"), k, cfg=cfg, grid=grid, params=params, eos=eos, state=state)
    else:
        alphaRho = phiHydF.local("alphaRho").at[i, j].set(state.rhoInSitu[i, j, k])  # :151-155 (ATMOSPHERIC: :440-446)
    if cfg.cpp.ALLOW_MOM_COMMON and params.quasiHydrostatic:                     # :179-186 (GO lane)
        from mitjax.pkg.mom_common.mom_quasihydrostatic import mom_quasihydrostatic
        alphaRho = mom_quasihydrostatic(k, state.uVel, state.vVel, alphaRho, myTime, myIter,
                                        cfg=cfg, grid=grid, params=params)       # :182-184
    if cfg.cpp.NONLIN_FRSURF and addSurfPhiAnom and params.uniformFreeSurfLev and k == 1:   # :191-200 (GOADK)
        phiHydF = phiHydF.at[i, j].set(surfPhiFac*state.etaH[i, j]
                                       * gravity*alphaRho[i, j]*recip_rhoConst)
    atSurf = g.kSurfC[i, j] == k
    if params.buoyancyRelation == "ATMOSPHERIC":                                # :417-602 (lane B, Task 25)
        if params.selectSigmaCoord_ne_0:                                        # :463-490 useFVgradPhi
            raise NotImplementedError("CALC_PHI_HYD: the ATMOSPHERIC useFVgradPhi arm (:463-490) is not ported")
        phiHydC, phiHydF = _atmospheric(k, i, j, phiHydF, phiHydC, alphaRho, atSurf, surfPhiFac, cfg=cfg, grid=g,
                                        params=params, state=state)
    elif params.integr_GeoPot == 1:                                             # :205-243
        if params.uniformFreeSurfLev:
            phiHydC = phiHydC.at[i, j].set(phiHydF[i, j]                        # :215-217
                                           + halfRL*g.drF[k]*gravFacC[k]*gravity
                                           * alphaRho[i, j]*recip_rhoConst)
            phiHydF = phiHydF.at[i, j].set(phiHydF[i, j]                        # :218-220
                                           + g.drF[k]*gravFacC[k]*gravity
                                           * alphaRho[i, j]*recip_rhoConst)
        else:
            ddRloc = g.Ro_surf[i, j]-g.rC[k]                                    # :227
            if cfg.cpp.NONLIN_FRSURF:
                ddRloc = ddRloc + surfPhiFac*state.etaH[i, j]                   # :229 (GO lane)
            phiHydC = phiHydC.at[i, j].set(jnp.where(
                atSurf,
                ddRloc*gravFacC[k]*gravity*alphaRho[i, j]*recip_rhoConst,     # :231-232
                phiHydF[i, j] + halfRL*g.drF[k]*gravFacC[k]*gravity           # :234-236
                * alphaRho[i, j]*recip_rhoConst))
            phiHydF = phiHydF.at[i, j].set(phiHydC[i, j]                        # :238-240
                                           + halfRL*g.drF[k]*gravFacC[k]*gravity
                                           * alphaRho[i, j]*recip_rhoConst)
    else:                                                                       # :245-291
        dRlocM = halfRL*g.drC[k]*gravFacF[k]                                    # :251
        if k == 1:
            dRlocM = (g.rF[k]-g.rC[k])*gravFacF[k]                              # :252
        if k == Nr:
            dRlocP = (g.rC[k]-g.rF[k+1])*gravFacF[k+1]                          # :254
        else:
            dRlocP = halfRL*g.drC[k+1]*gravFacF[k+1]                            # :256
        if params.uniformFreeSurfLev:                                           # :258-266
            phiHydC = phiHydC.at[i, j].set(phiHydF[i, j]                        # :261-262
                                           + dRlocM*gravity*alphaRho[i, j]*recip_rhoConst)
            phiHydF = phiHydF.at[i, j].set(phiHydC[i, j]                        # :263-264
                                           + dRlocP*gravity*alphaRho[i, j]*recip_rhoConst)
        else:                                                                   # :267-288
            rec_dRm = 1.0/(g.rF[k]-g.rC[k])                                     # :268  oneRL/
            rec_dRp = 1.0/(g.rC[k]-g.rF[k+1])                                   # :269
            ddRloc = g.Ro_surf[i, j]-g.rC[k]                                    # :273
            if cfg.cpp.NONLIN_FRSURF:
                ddRloc = ddRloc + surfPhiFac*state.etaH[i, j]                   # :275 (GO lane)
            phiHydC = phiHydC.at[i, j].set(jnp.where(
                atSurf,
                (MAX(0., ddRloc, p="b")*rec_dRm*dRlocM                          # :277-279
                 + MIN(0., ddRloc, p="b")*rec_dRp*dRlocP
                 )*gravity*alphaRho[i, j]*recip_rhoConst,
                phiHydF[i, j] + dRlocM*gravity*alphaRho[i, j]*recip_rhoConst))    # :281-282
            phiHydF = phiHydF.at[i, j].set(phiHydC[i, j]                        # :284-285
                                           + dRlocP*gravity*alphaRho[i, j]*recip_rhoConst)
    if params.selectSigmaCoord_ne_0:                                            # :97 useFVgradPhi, :633-654
        raise NotImplementedError("CALC_PHI_HYD: useFVgradPhi (CALC_GRAD_PHI_FV) is not ported")
    if params.momPressureForcing:                                               # :624-630
        dPhiHydX, dPhiHydY = calc_grad_phi_hyd(k, iMin, iMax, jMin, jMax, phiHydC, alphaRho, dPhiHydX, dPhiHydY,
                                               myTime, myIter, cfg=cfg, grid=grid, params=params, phi0surf=phi0surf,
                                               state=state)
    state = diags_phi_rlow(k, iMin, iMax, jMin, jMax, phiHydF, phiHydC, alphaRho, myTime, myIter,   # :660-665
                           cfg=cfg, grid=grid, params=params, state=state, phi0surf=phi0surf)
    state = diags_phi_hyd(k, iMin, iMax, jMin, jMax, phiHydC, myTime, myIter,                       # :668-671
                          cfg=cfg, grid=grid, params=params, state=state, phi0surf=phi0surf)
    return state, phiHydF, phiHydC, dPhiHydX, dPhiHydY


def _atmospheric(k, i, j, phiHydF, phiHydC, alphaRho, atSurf, surfPhiFac, *, cfg, grid, params, state):
    """CALC_PHI_HYD's 'ATMOSPHERIC' integration of d Phi / d pi (calc_phi_hyd.F:492-606; lane B, Task 25) on the
    point set (i, j) of level k. pKF(k) = (rF(k)/atm_Po)**atm_kappa, pKC(k) = (rC(k)/atm_Po)**atm_kappa: glibc's pow
    on the host (Params, ini_parms._atm_traced). Pointwise IFs on kSurfC are `where`s; MAX/MIN(zeroRL, ...) with
    mitjax.ops.fortran_minmax. Returns (phiHydC, phiHydF)."""
    Nr = cfg.size.Nr
    g = grid
    halfRL = 0.5                                                                # EEPARAMS.h:73
    atm_Cp = params.atm_Cp
    pKF, pKC = params.rF_Po_kappa, params.rC_Po_kappa
    if params.integr_GeoPot == 0:                                               # :492-524
        if k == 1:                                                              # :503-508
            ddPIm = atm_Cp*(pKF[k] - pKC[k])                                    # :504-505
        else:
            ddPIm = atm_Cp*(pKC[k-1] - pKC[k])*halfRL                           # :507-508
        if k == Nr:                                                             # :510-515
            ddPIp = atm_Cp*(pKC[k] - pKF[k+1])                                  # :511-512
        else:
            ddPIp = atm_Cp*(pKC[k] - pKC[k+1])*halfRL                           # :514-515
        phiHydC = phiHydC.at[i, j].set(phiHydF[i, j] + ddPIm*alphaRho[i, j])    # :520
        phiHydF = phiHydF.at[i, j].set(phiHydC[i, j] + ddPIp*alphaRho[i, j])    # :521
    elif params.integr_GeoPot == 1:                                             # :527-557
        ddPIm = atm_Cp*(pKF[k] - pKC[k])                                        # :537-538
        ddPIp = atm_Cp*(pKC[k] - pKF[k+1])                                      # :539-540
        ddRloc = g.Ro_surf[i, j]-g.rC[k]                                        # :544
        if cfg.cpp.NONLIN_FRSURF:
            ddRloc = ddRloc + surfPhiFac*state.etaH[i, j]                       # :546
        phiHydC = phiHydC.at[i, j].set(jnp.where(
            atSurf,
            ddRloc*g.recip_drF[k]*2.0*ddPIm*alphaRho[i, j],                    # :548-549  2. _d 0
            phiHydF[i, j] + ddPIm*alphaRho[i, j]))                              # :551
        phiHydF = phiHydF.at[i, j].set(phiHydC[i, j] + ddPIp*alphaRho[i, j])    # :553
    elif params.integr_GeoPot in (2, 3):                                        # :559-600
        if k == 1:                                                              # :569-574
            ddPIm = atm_Cp*(pKF[k] - pKC[k])                                    # :570-571
        else:
            ddPIm = atm_Cp*(pKC[k-1] - pKC[k])*halfRL                           # :573-574
        if k == Nr:                                                             # :576-581
            ddPIp = atm_Cp*(pKC[k] - pKF[k+1])                                  # :577-578
        else:
            ddPIp = atm_Cp*(pKC[k] - pKC[k+1])*halfRL                           # :580-581
        rec_dRm = 1.0/(g.rF[k]-g.rC[k])                                         # :583  oneRL/
        rec_dRp = 1.0/(g.rC[k]-g.rF[k+1])                                       # :584
        ddRloc = g.Ro_surf[i, j]-g.rC[k]                                        # :588
        if cfg.cpp.NONLIN_FRSURF:
            ddRloc = ddRloc + surfPhiFac*state.etaH[i, j]                       # :590
        phiHydC = phiHydC.at[i, j].set(jnp.where(
            atSurf,
            (MAX(0., ddRloc, p="b")*rec_dRm*ddPIm                               # :592-594
             + MIN(0., ddRloc, p="b")*rec_dRp*ddPIp
             )*alphaRho[i, j],
            phiHydF[i, j] + ddPIm*alphaRho[i, j]))                              # :596
        phiHydF = phiHydF.at[i, j].set(phiHydC[i, j] + ddPIp*alphaRho[i, j])    # :598
    else:
        raise ValueError("CALC_PHI_HYD: Bad integr_GeoPot option !")           # :605 STOP
    return phiHydC, phiHydF
