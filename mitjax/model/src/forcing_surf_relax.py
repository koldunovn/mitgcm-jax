"""FORCING_SURF_RELAX: model/src/forcing_surf_relax.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import loop_i, loop_j


def forcing_surf_relax(iMin, iMax, jMin, jMax, myTime, myIter, ff, *, cfg, grid, fp, state, sp=None, area=None):
    """FORCING_SURF_RELAX( iMin, iMax, jMin, jMax, myTime, myIter, myThid )   @63cdc0b
    model/src/forcing_surf_relax.F:7-258

    C     | SUBROUTINE FORCING_SURF_RELAX
    C     | o Calculate relaxation surface forcing terms
    C     |   for temperature and salinity

    `ff` FFIELDS.h (surfaceForcingT/S written, lambda*ClimRelax, SST, SSS read), `state` (theta, salt), `grid`
    (drF, hFacC as the step sees it). Returns the new FFields.

    Ported: the restoring terms :93-106 (GO lane: ALLOW_SEAICE compiled with useSEAICE off, as in
    global_ocean.cs32x15, it is the plain branch; lane M4OFF: useSEAICE with SEAICErestoreUnderIce .TRUE. (`sp`,
    SEAICE_PARAMS.h), as in offline_exf_seaice/input.thermo, also takes it; lane M4LAB: the no-restore-under-ice arm
    :74-91, useSEAICE .AND. .NOT.SEAICErestoreUnderIce, with `area` = SEAICE.h AREA, as in lab_sea/input). GOADK lane (global_ocean.90x40x15/code_ad): the NONLIN_FRSURF staggerTimeStep
    rescaling :112-157 with select_rStar = 0 and selectSigmaCoord = 0 (the surface point ks = kSurfC: forcing times
    recip_hFacC(ks)*hFac_surfC, :144-155; with DISABLE_RSTAR_CODE / DISABLE_SIGMA_CODE their arms are empty); PTRACERS
    lane (tutorial_tracer_adjsens/input_ad.som81; also global_ocean.cs32x15, GO lane): the r* rescaling :117-125
    (forcing times rStarExpC); the sigma rescaling raises. GO lane: ALLOW_BALANCE_RELAX compiled with its two
    switches off (global_ocean.cs32x15). Raises: balanceThetaClimRelax / balanceSaltClimRelax (:167-238).
    DIAGNOSTICS_SCALE_FILL :241-255 is output only (not ported)."""
    sz = cfg.size
    useSEAICE = any(k.lower() == "useseaice" for k, _ in cfg.use) and cfg.use_flag("useSEAICE")
    if cfg.cpp.ALLOW_SEAICE and useSEAICE:                                      # :73-74 (GO lane: the switch)
        if sp is None:
            raise ValueError("FORCING_SURF_RELAX: useSEAICE needs SEAICE_PARAMS.h `sp` (SEAICErestoreUnderIce)")
        noRestoreUnderIce = not sp.SEAICErestoreUnderIce                       # :75
        if noRestoreUnderIce and area is None:
            raise ValueError("FORCING_SURF_RELAX: .NOT.SEAICErestoreUnderIce needs SEAICE.h AREA (`area`)")
    else:
        noRestoreUnderIce = False
    if fp.usingPCoords:                                                         # :63-67
        ks = sz.Nr
    else:
        ks = 1
    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    if noRestoreUnderIce:                                                       # :75-90 (lane M4LAB)
        # :80-83  surfaceForcingT = -lambdaThetaClimRelax*(1.-AREA)*(theta(ks)-SST)*drF(ks)*_hFacC(ks)
        sfT = ff.surfaceForcingT.at[i, j].set(
            -ff.lambdaThetaClimRelax[i, j]*(1.-area[i, j])
            * (state.theta[i, j, ks] - ff.SST[i, j])
            * grid.drF[ks] * grid.hFacC[i, j, ks])
        # :85-88  surfaceForcingS = -lambdaSaltClimRelax*(1.-AREA)*(salt(ks)-SSS)*drF(ks)*_hFacC(ks)
        sfS = ff.surfaceForcingS.at[i, j].set(
            -ff.lambdaSaltClimRelax[i, j]*(1.-area[i, j])
            * (state.salt[i, j, ks] - ff.SSS[i, j])
            * grid.drF[ks] * grid.hFacC[i, j, ks])
    else:
        # :96-99  surfaceForcingT = -lambdaThetaClimRelax*(theta(ks)-SST)*drF(ks)*_hFacC(ks)
        sfT = ff.surfaceForcingT.at[i, j].set(
            -ff.lambdaThetaClimRelax[i, j]
            * (state.theta[i, j, ks] - ff.SST[i, j])
            * grid.drF[ks] * grid.hFacC[i, j, ks])
        # :101-104  surfaceForcingS = -lambdaSaltClimRelax*(salt(ks)-SSS)*drF(ks)*_hFacC(ks)
        sfS = ff.surfaceForcingS.at[i, j].set(
            -ff.lambdaSaltClimRelax[i, j]
            * (state.salt[i, j, ks] - ff.SSS[i, j])
            * grid.drF[ks] * grid.hFacC[i, j, ks])
    if cfg.cpp.NONLIN_FRSURF and fp.staggerTimeStep and fp.nonlinFreeSurf > 0:  # :112-157 (GOADK lane)
        if fp.select_rStar > 0:                                                 # :116-126
            if not cfg.cpp.DISABLE_RSTAR_CODE:                                  # :117-125 (PTRACERS lane: som81)
                sfT = sfT.at[i, j].set(sfT[i, j]
                                       * state.rStarExpC[i, j])                 # :120-121
                sfS = sfS.at[i, j].set(sfS[i, j]
                                       * state.rStarExpC[i, j])                 # :122-123
        elif fp.selectSigmaCoord != 0:                                          # :127-143
            if not cfg.cpp.DISABLE_SIGMA_CODE:
                raise NotImplementedError("FORCING_SURF_RELAX: the sigma rescaling (:128-142) is not ported")
        else:                                                                   # :144-155
            at = grid.kSurfC[i, j] == ks                                        # IF (ks.EQ.kSurfC(i,j,bi,bj))
            sfT = sfT.at[i, j].set(jnp.where(at, sfT[i, j]
                                             * grid.recip_hFacC[i, j, ks]*state.hFac_surfC[i, j], sfT[i, j]))
            sfS = sfS.at[i, j].set(jnp.where(at, sfS[i, j]
                                             * grid.recip_hFacC[i, j, ks]*state.hFac_surfC[i, j], sfS[i, j]))
    if cfg.cpp.ALLOW_BALANCE_RELAX and (fp.balanceThetaClimRelax or fp.balanceSaltClimRelax):   # :167, :203
        raise NotImplementedError("FORCING_SURF_RELAX: balanceThetaClimRelax / balanceSaltClimRelax (ALLOW_BALANCE_RELAX "
                                  ":165-239) is not ported")
    return ff.replace(surfaceForcingT=sfT, surfaceForcingS=sfS)
