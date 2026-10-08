"""pkg/mom_common/mom_quasihydrostatic.F: quasi-hydrostatic terms added to the buoyancy (MOM_QUASIHYDROSTATIC)."""

from mitjax.farray import loop_i, loop_j

_OPT = "MOM_COMMON_OPTIONS.h"       # mom_quasihydrostatic.F:1


def mom_quasihydrostatic(k, uFld, vFld, effectiveBuoy, myTime, myIter, *, cfg, grid, params):
    """MOM_QUASIHYDROSTATIC(bi, bj, k, uFld, vFld, effectiveBuoy, myTime, myIter, myThid)
    @63cdc0b pkg/mom_common/mom_quasihydrostatic.F:3-153

    C     | o SUBROUTINE MOM_QUASIHYDROSTATIC
    C     |   Add Quasi-Hydrostatic Terms to buoyancy
    C  k             :: vertical level
    C  uFld          :: zonal flow
    C  vFld          :: meridional flow
    C  myTime        :: current time in simulation
    C  myIter        :: current iteration number
    C  effectiveBuoy :: Density (z-coord) / specific volume (p-coord) anomaly

    Returns effectiveBuoy. `k` and the PARAMS.h selectors (select3dCoriScheme, useNHMTerms, usingZCoords,
    staggerTimeStep) are static; myTime, myIter are not read. uFld, vFld are the 3-D fields. Ported: the z-coordinate
    scaling (:70-74) and both terms (:99-123). Not ported (raise): the p-coordinate scalings (:78-90) and, under
    #ifdef ALLOW_QHYD_STAGGER_TS, staggerTimeStep = .TRUE. (:125-141: ADAMS_BASHFORTH2 of the QHydGwNm state, model
    core lane) -- global_ocean.90x40x15 runs that branch, so its gate waits for adams_bashforth2. The point loops
    run on the whole (i,j) range; `0. _d 0` is the double zero.
    """
    OLx, OLy, sNx, sNy = cfg.size.OLx, cfg.size.OLy, cfg.size.sNx, cfg.size.sNy
    iMin, iMax = 0, sNx+1                                           # :58  PARAMETER( iMin = 0 , iMax = sNx+1 )
    jMin, jMax = 0, sNy+1                                           # :59  PARAMETER( jMin = 0 , jMax = sNy+1 )
    fCoriCos, angleCosC, angleSinC = grid.fCoriCos, grid.angleCosC, grid.angleSinC
    gravitySign, recip_deepFacC = grid.gravitySign, grid.recip_deepFacC
    halfRL = 0.5                                                    # EEPARAMS.h:73

    if params.select3dCoriScheme >= 1 or params.useNHMTerms:        # :68

        if params.usingZCoords:                                     # :70-74
#--   Z-coordinate case: Input is density anomaly
            scalingFactor = (params.rhoConst*gravitySign
                             * params.recip_gravity*params.recip_gravFacC[k])
        elif params.fluidIsWater:                                   # :78-84
            raise NotImplementedError("MOM_QUASIHYDROSTATIC: P-coordinate, oceanic case is not ported")
        else:                                                       # :86-89
            raise NotImplementedError("MOM_QUASIHYDROSTATIC: P-coordinate, Ideal-Gas case is not ported")

        gWinBuoy = effectiveBuoy.local("gWinBuoy")                  # :61
        jA = loop_j(1-OLy, sNy+OLy)                                 # :93-97
        iA = loop_i(1-OLx, sNx+OLx)
        gWinBuoy = gWinBuoy.at[iA, jA].set(0.)

        j = loop_j(jMin, jMax)
        i = loop_i(iMin, iMax)
        if params.select3dCoriScheme >= 1:                          # :99-110
            gWinBuoy = gWinBuoy.at[i, j].set(fCoriCos[i, j]*(
                angleCosC[i, j]*halfRL
                * (uFld[i, j, k] + uFld[i+1, j, k])
                - angleSinC[i, j]*halfRL
                * (vFld[i, j, k] + vFld[i, j+1, k])
                ))

        if params.useNHMTerms:                                      # :112-123
            gWinBuoy = gWinBuoy.at[i, j].set(gWinBuoy[i, j]
                + ((uFld[i, j, k]*uFld[i, j, k]
                    + uFld[i+1, j, k]*uFld[i+1, j, k])
                   + (vFld[i, j, k]*vFld[i, j, k]
                      + vFld[i, j+1, k]*vFld[i, j+1, k])
                   )*halfRL*params.recip_rSphere*recip_deepFacC[k])

        if cfg.cpp.flag("ALLOW_QHYD_STAGGER_TS", _OPT) and params.staggerTimeStep:   # :125-141
            raise NotImplementedError("MOM_QUASIHYDROSTATIC: staggerTimeStep with ALLOW_QHYD_STAGGER_TS "
                                      "(ADAMS_BASHFORTH2 of QHydGwNm) is not ported yet")

        effectiveBuoy = effectiveBuoy.at[i, j].set(effectiveBuoy[i, j]   # :143-148
                                                   + scalingFactor*gWinBuoy[i, j])

    return effectiveBuoy
