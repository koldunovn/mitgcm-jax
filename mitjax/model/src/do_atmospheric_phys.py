"""DO_ATMOSPHERIC_PHYS: model/src/do_atmospheric_phys.F @63cdc0b."""


def do_atmospheric_phys(myTime, myIter, *, cfg, params, state, grid=None):
    """DO_ATMOSPHERIC_PHYS( myTime, myIter, myThid )   @63cdc0b model/src/do_atmospheric_phys.F:10-156

    C     | SUBROUTINE DO_ATMOSPHERIC_PHYS
    C     | o Controlling routine for atmospheric physics and parameterization

    Returns the State. Everything it does sits under `IF ( fluidIsAir )` (:69-115) or in atmospheric packages
    (AIM, FIZHI, ...); an ocean (fluidIsAir = .FALSE.) passes through, except in an ALLOW_AUTODIFF build, whose
    `ELSE` arm (:101-113, R5 arm) sets rhoInSitu = 0. (REAL*4 literal, exact) on every point of every level.
    fluidIsAir (lane B, Task 25; `grid` for maskC): the virtual potential temperature anomaly in rhoInSitu
    (:76-100) on every point of every level, thetaRef = thetaConst under select_rStar >= 1 or selectSigmaCoord >= 1,
    else tRef(k). The atmospheric packages (useFIZHI, useAtm_Phys, useAIM: :117-149) raise."""
    if params.fluidIsAir:                                                       # :69-100 (lane B, Task 25)
        if any(dict(cfg.use).get(n, False) for n in ("useFIZHI", "useAtm_Phys", "useAIM")):
            raise NotImplementedError("DO_ATMOSPHERIC_PHYS: FIZHI / ATM_PHYS / AIM are not ported")
        from mitjax.farray import loop_i, loop_j
        sz = cfg.size
        j = loop_j(1-sz.OLy, sz.sNy+sz.OLy)                                     # :89
        i = loop_i(1-sz.OLx, sz.sNx+sz.OLx)                                     # :90
        rhoInSitu = state.rhoInSitu
        for k in range(1, sz.Nr+1):                                             # :81
            if params.select_rStar >= 1 or params.selectSigmaCoord >= 1:        # :82
                thetaRef = params.thetaConst                                    # :84
            else:
                thetaRef = params.tRef[k]                                       # :87
            rhoInSitu = rhoInSitu.at[i, j, k].set(                              # :91-94
                (state.theta[i, j, k]
                 * (state.salt[i, j, k]*params.atm_Rq + 1.)                     # oneRL
                 - thetaRef)*grid.maskC[i, j, k])
        return state.replace(rhoInSitu=rhoInSitu)
    if cfg.cpp.ALLOW_AUTODIFF:                                                  # :101-113 (R5 arm)
        from mitjax.farray import loops_kji
        sz = cfg.size
        k, j, i = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))
        state = state.replace(rhoInSitu=state.rhoInSitu.at[i, j, k].set(0.))   # :108
    return state
