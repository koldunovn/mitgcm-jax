"""EXTERNAL_FORCING_SURF: model/src/external_forcing_surf.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j
from mitjax.model.grid import UNSET_RL
from mitjax.model.src.forcing_surf_relax import forcing_surf_relax


def _use(cfg, name):
    return cfg.use_flag(name) if any(k.lower() == name.lower() for k, _ in cfg.use) else False


def external_forcing_surf(iMin, iMax, jMin, jMax, myTime, myIter, ff, *, cfg, grid, fp, state, phi0surf=None,
                          ptr=None, sp=None, area=None):
    """EXTERNAL_FORCING_SURF( iMin, iMax, jMin, jMax, myTime, myIter, myThid )   @63cdc0b
    model/src/external_forcing_surf.F:14-407

    C     | SUBROUTINE EXTERNAL_FORCING_SURF
    C     | o Determines forcing terms based on external fields
    C     |   relaxation terms etc.

    `ff` FFIELDS.h (surfaceForcingU/V/T/S written, EmPmR masked), `state` (theta, salt read; PmEpR written with
    staggerTimeStep, read with nonlinFreeSurf + useRealFreshWaterFlux), `phi0surf` (SURFACE.h /SURF_FIXED/, written
    under ATMOSPHERIC_LOADING; not carried by State). Returns (ff, state, phi0surf).

    Ported: :139-168 (surfaceForcingT/S = 0, EmPmR*maskInC with useRealFreshWaterFlux, PmEpR = -EmPmR with
    staggerTimeStep), FORCING_SURF_RELAX :173-177, the surface fluxes :218-237, the fresh-water branches :261-351,
    ATMOSPHERIC_LOADING :355-392. Note: :261-391 sit inside the bi,bj loop of :205-404, as in the Fortran.
    PTRACERS lane: usePTRACERS -> PTRACERS_FORCING_SURF (:179-198; `ptr`: PTRACERS_PARAMS.h, the State's pTracer
    and surfaceForcingPTr fields). PTRACERS lane: SHORTWAVE_HEATING (:69-71, :80-83, :228-230: Qnet - Qsw*tmpFac).
    GO lane: ALLOW_BALANCE_FLUXES :90-115 with its switches off (selectBalanceEmPmR = 0, balanceQnet = .FALSE.:
    nothing to do). Raises: REMOVE_MEAN_RS of :98-114 when a balance switch is on, useSALT_PLUME :239-253 and useSHELFICE :394-400 (no M1 variant switches these on).
    Lane M4OFF: `sp` SEAICE_PARAMS.h (useSEAICE), read by FORCING_SURF_RELAX (SEAICErestoreUnderIce); lane M4LAB:
    `area` SEAICE.h AREA for its no-restore-under-ice arm."""
    sz = cfg.size
    if cfg.cpp.ALLOW_BALANCE_FLUXES:                                            # :90-115 (GO lane)
        noSeaice = (not _use(cfg, "useSEAICE")) or _use(cfg, "useThSIce")       # .NOT.useSeaice .OR. useThSIce
        if (fp.selectBalanceEmPmR >= 1 and noSeaice) or (fp.balanceQnet and noSeaice):   # :98-99, :110
            raise NotImplementedError("EXTERNAL_FORCING_SURF: REMOVE_MEAN_RS (selectBalanceEmPmR >= 1 or "
                                      "balanceQnet) is not ported")
    if fp.usingPCoords:                                                         # :74-78
        ks = sz.Nr
    else:
        ks = 1
    recip_Cp = 1.0 / fp.HeatCapacity_Cp                                         # :79  1. _d 0 / HeatCapacity_Cp
    mass2rUnit = fp.mass2rUnit
    if cfg.cpp.SHORTWAVE_HEATING:                                               # :80-83 (PTRACERS lane)
        tmpFac = 1.0                                                            # oneRS
        if fp.selectPenetratingSW <= 0:
            tmpFac = 0.0                                                        # zeroRS

    # :139-168
    jh = loop_j(1-sz.OLy, sz.sNy+sz.OLy)
    ih = loop_i(1-sz.OLx, sz.sNx+sz.OLx)
    sfT = ff.surfaceForcingT.at[ih, jh].set(0.)                                 # :144
    sfS = ff.surfaceForcingS.at[ih, jh].set(0.)                                 # :145
    EmPmR = ff.EmPmR
    if fp.useRealFreshWaterFlux:                                                # :149-156
        EmPmR = EmPmR.at[ih, jh].set(EmPmR[ih, jh]*grid.maskInC[ih, jh])       # :153
    PmEpR = state.PmEpR
    if fp.staggerTimeStep:                                                      # :160-166
        PmEpR = PmEpR.at[ih, jh].set(-EmPmR[ih, jh])                            # :163
    ff = ff.replace(surfaceForcingT=sfT, surfaceForcingS=sfS, EmPmR=EmPmR)
    state = state.replace(PmEpR=PmEpR)

    if fp.doThetaClimRelax or fp.doSaltClimRelax:                               # :173-177
        ff = forcing_surf_relax(iMin, iMax, jMin, jMax, myTime, myIter, ff, cfg=cfg, grid=grid, fp=fp, state=state,
                                sp=sp, area=area)

    if cfg.cpp.ALLOW_PTRACERS and _use(cfg, "usePTRACERS"):                     # :179-198 (PTRACERS lane)
        # PTRACERS_FORCING_SURF( surfaceForcingS, bi, bj, iMin, iMax, jMin, jMax, ... ) (:189-193), the build's own
        # version (an experiment's code directory may replace it); `ptr`: PTRACERS_PARAMS.h (the caller's params)
        from mitjax.pkg.ptracers.ptracers_fields_h import ptf_of_state, state_with_ptf
        from mitjax.pkg.ptracers.ptracers_forcing_surf import routine_of_build
        ptf = routine_of_build(cfg)(ff.surfaceForcingS, iMin, iMax, jMin, jMax, myTime, myIter, cfg=cfg, grid=grid,
                                    params=fp, ptr=ptr, ptf=ptf_of_state(state, ptr), ff=ff)
        state = state_with_ptf(state, ptf)

    # :205-404 (bi,bj loop)
    j = loop_j(jMin, jMax)
    i = loop_i(iMin, iMax)
    sfU = ff.surfaceForcingU.at[i, j].set(ff.fu[i, j]*mass2rUnit)              # :222
    sfV = ff.surfaceForcingV.at[i, j].set(ff.fv[i, j]*mass2rUnit)              # :224
    if cfg.cpp.SHORTWAVE_HEATING:                                               # :226-231 with :228-230
        sfT = ff.surfaceForcingT.at[i, j].set(
            ff.surfaceForcingT[i, j] - (ff.Qnet[i, j]
                                        - ff.Qsw[i, j]*tmpFac)*recip_Cp*mass2rUnit)
    else:
        sfT = ff.surfaceForcingT.at[i, j].set(                                  # :226-231
            ff.surfaceForcingT[i, j] - (ff.Qnet[i, j])*recip_Cp*mass2rUnit)
    sfS = ff.surfaceForcingS.at[i, j].set(                                      # :233-234
        ff.surfaceForcingS[i, j] - ff.saltFlux[i, j]*mass2rUnit)

    if cfg.cpp.ALLOW_SALT_PLUME and _use(cfg, "useSALT_PLUME"):                 # :239-253
        raise NotImplementedError("EXTERNAL_FORCING_SURF: SALT_PLUME_FORCING_SURF is not ported")

    theta_ks = state.theta[i, j, ks]
    salt_ks = state.salt[i, j, ks]
    if (fp.nonlinFreeSurf > 0 or fp.usingPCoords) and fp.useRealFreshWaterFlux:   # :261-262
        if fp.temp_EvPrRn != UNSET_RL:                                          # :268-277
            sfT = sfT.at[i, j].set(sfT[i, j] + state.PmEpR[i, j]*(fp.temp_EvPrRn - theta_ks)*mass2rUnit)
        if fp.salt_EvPrRn != UNSET_RL:                                          # :279-288
            sfS = sfS.at[i, j].set(sfS[i, j] + state.PmEpR[i, j]*(fp.salt_EvPrRn - salt_ks)*mass2rUnit)
    else:
        if fp.convertFW2Salt == -1.:                                            # :296  (-1. REAL*4, exact)
            if fp.temp_EvPrRn != UNSET_RL:                                      # :299-309
                sfT = sfT.at[i, j].set(sfT[i, j] + ff.EmPmR[i, j]*(theta_ks - fp.temp_EvPrRn)*mass2rUnit)
            if fp.salt_EvPrRn != UNSET_RL:                                      # :310-320
                sfS = sfS.at[i, j].set(sfS[i, j] + ff.EmPmR[i, j]*(salt_ks - fp.salt_EvPrRn)*mass2rUnit)
        else:
            if fp.temp_EvPrRn != UNSET_RL:                                      # :325-335
                sfT = sfT.at[i, j].set(sfT[i, j] + ff.EmPmR[i, j]*(float(fp.tRef[ks - 1]) - fp.temp_EvPrRn)
                                       * mass2rUnit)
            if fp.salt_EvPrRn != UNSET_RL:                                      # :336-346
                sfS = sfS.at[i, j].set(sfS[i, j] + ff.EmPmR[i, j]*(fp.convertFW2Salt - fp.salt_EvPrRn)
                                       * mass2rUnit)
    ff = ff.replace(surfaceForcingU=sfU, surfaceForcingV=sfV, surfaceForcingT=sfT, surfaceForcingS=sfS)

    if cfg.cpp.ATMOSPHERIC_LOADING and fp.usingZCoords:                         # :355-392
        if phi0surf is None:
            raise ValueError("EXTERNAL_FORCING_SURF: ATMOSPHERIC_LOADING writes phi0surf; pass it")
        if fp.useRealFreshWaterFlux:                                            # :367-374
            phi0surf = phi0surf.at[i, j].set(
                (ff.pLoad[i, j] + ff.sIceLoad[i, j]*fp.gravity*fp.sIceLoadFac)*fp.recip_rhoConst)
        else:                                                                   # :375-381
            phi0surf = phi0surf.at[i, j].set(ff.pLoad[i, j]*fp.recip_rhoConst)

    if cfg.cpp.ALLOW_SHELFICE and _use(cfg, "useSHELFICE"):                     # :394-400
        raise NotImplementedError("EXTERNAL_FORCING_SURF: SHELFICE_FORCING_SURF is not ported")
    return ff, state, phi0surf
