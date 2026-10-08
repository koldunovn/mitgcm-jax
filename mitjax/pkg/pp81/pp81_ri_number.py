"""PP81_RI_NUMBER: pkg/pp81/pp81_ri_number.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j
from mitjax.model.src.find_rho import find_rho_2d
from mitjax.ops.fortran_minmax import MAX
from mitjax.ops.scan_k import level

# pp81_ri_number.F:53-54  PARAMETER( p5=0.5, p125=0.125 )  (REAL*4 literals, exact); used only under PP81_SMOOTH_RI
p5 = 0.5
p125 = 0.125
# pp81_ri_number.F:56-57  PARAMETER    (  epsilon = 1.D-10 )
epsilon = 1.0e-10


def pp81_ri_number(K, iMin, iMax, jMin, jMax, RiNumber, myTime, *, cfg, grid, params, eos, state):
    """PP81_RI_NUMBER( bi, bj, K, iMin, iMax, jMin, jMax, RiNumber, myTime, myThid )
    @63cdc0b pkg/pp81/pp81_ri_number.F:7-138

    C     | SUBROUTINE PP81_RI_NUMBER                                |
    C     | o Compute gradient Richardson number for Pacanowski and  |
    C     |   Philander (1981) mixing scheme                         |
    C     bi, bj - array indices on which to apply calculations
    C     iMin, iMax, jMin, jMax
    C            - array boundaries
    C     k      - depth level
    C     myTime - Current time in simulation
    C     RiNumber - (output) Richardson number

    K a Python int (2..Nr in PP81_CALC); RiNumber: FArray (1-OLx:sNx+OLx,1-OLy:sNy+OLy), returned with iMin:iMax,
    jMin:jMax written (:96-117), every other point keeping its value. `state`: DYNVARS.h (theta, salt, uVel, vVel; and
    totPhiHyd for the pressure of FIND_RHO_2D). rhoKm1, rhoK are locals (NaN outside iMin:iMax, jMin:jMax, never read
    there). Both densities use the pressure of level K (kRef = K, :83-92). PP81_SMOOTH_RI (:62, :119-133) is
    #undef'd in PP81_OPTIONS.h of every M3 build and raises. The (i, j) loop reads only inputs: vectorised.
    """
    if cfg.cpp.flag("PP81_SMOOTH_RI", "PP81_OPTIONS.h"):
        raise NotImplementedError("PP81_RI_NUMBER: PP81_SMOOTH_RI is not ported")
    Km1 = max(1, K-1)                                                   # :67  MINMAX-INT: integer level index
    rhoKm1 = RiNumber.local("rhoKm1")
    rhoK = RiNumber.local("rhoK")
    rhoKm1 = find_rho_2d(iMin, iMax, jMin, jMax, K,                     # :83-87
                         level(state.theta, Km1), level(state.salt, Km1), rhoKm1, Km1,
                         cfg=cfg, grid=grid, params=params, eos=eos, state=state)
    rhoK = find_rho_2d(iMin, iMax, jMin, jMax, K,                       # :88-92
                       level(state.theta, K), level(state.salt, K), rhoK, K,
                       cfg=cfg, grid=grid, params=params, eos=eos, state=state)
    uVel, vVel = state.uVel, state.vVel
    j = loop_j(jMin, jMax)                                              # :96
    i = loop_i(iMin, iMax)                                              # :97
    tempu = (.5*(uVel[i, j, Km1]+uVel[i+1, j, Km1]                      # :98-100
                 - (uVel[i, j, K]+uVel[i+1, j, K]))
             * grid.recip_drC[K])
    tempv = (.5*(vVel[i, j, Km1]+vVel[i, j+1, Km1]                      # :101-103
                 - (vVel[i, j, K]+vVel[i, j+1, K]))
             * grid.recip_drC[K])
    RiFlux = tempu*tempu+tempv*tempv                                    # :104
    buoyFreq = - (params.gravity*params.mass2rUnit                      # :109-110
                  * (rhoKm1[i, j] - rhoK[i, j])*grid.recip_drC[K])
    return RiNumber.at[i, j].set(buoyFreq/MAX(RiFlux, epsilon, p="b"))  # :115
