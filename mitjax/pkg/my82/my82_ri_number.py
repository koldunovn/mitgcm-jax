"""MY82_RI_NUMBER: pkg/my82/my82_ri_number.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j
from mitjax.model.src.find_rho import find_rho_2d
from mitjax.ops.fortran_minmax import MAX
from mitjax.ops.scan_k import level

# my82_ri_number.F:58-59  PARAMETER( p5=0.5D0, p125=0.125D0 )  (used only under MY82_SMOOTH_RI)
p5 = 0.5
p125 = 0.125
# my82_ri_number.F:61-62  PARAMETER    (  epsilon = 1.D-10 )
epsilon = 1.0e-10


def my82_ri_number(K, iMin, iMax, jMin, jMax, RiNumber, buoyFreq, vertShear, myTime, *, cfg, grid, params, eos,
                   state):
    """MY82_RI_NUMBER( bi, bj, K, iMin, iMax, jMin, jMax, RiNumber, buoyFreq, vertShear, myTime, myThid )
    @63cdc0b pkg/my82/my82_ri_number.F:7-142

    C     | SUBROUTINE MY82_RI_NUMBER                                |
    C     | o Compute gradient Richardson number for Mellor and      |
    C     |   Yamada (1981) turbulence model                         |
    C     RiNumber  - (output) Richardson number
    C     buoyFreq  - (output) (neg.) buoyancy frequency -N^2
    C     vertShear - (output) vertical shear of velocity

    K a Python int; RiNumber, buoyFreq, vertShear: FArrays (1-OLx:sNx+OLx,1-OLy:sNy+OLy), returned with iMin:iMax,
    jMin:jMax written (:99-121), other points keeping their values. Both densities at the pressure of level K
    (kRef = K, :86-95). MY82_SMOOTH_RI (#undef'd in every M3 build) raises. The (i, j) loop reads only inputs.
    """
    if cfg.cpp.flag("MY82_SMOOTH_RI", "MY82_OPTIONS.h"):
        raise NotImplementedError("MY82_RI_NUMBER: MY82_SMOOTH_RI is not ported")
    Km1 = max(1, K-1)                                                   # :70  MINMAX-INT: integer level index
    rhoKm1 = RiNumber.local("rhoKm1")
    rhoK = RiNumber.local("rhoK")
    rhoKm1 = find_rho_2d(iMin, iMax, jMin, jMax, K,                     # :86-90
                         level(state.theta, Km1), level(state.salt, Km1), rhoKm1, Km1,
                         cfg=cfg, grid=grid, params=params, eos=eos, state=state)
    rhoK = find_rho_2d(iMin, iMax, jMin, jMax, K,                       # :91-95
                       level(state.theta, K), level(state.salt, K), rhoK, K,
                       cfg=cfg, grid=grid, params=params, eos=eos, state=state)
    uVel, vVel = state.uVel, state.vVel
    j = loop_j(jMin, jMax)                                              # :99
    i = loop_i(iMin, iMax)                                              # :100
    tempu = (.5*(uVel[i, j, Km1]+uVel[i+1, j, Km1]                      # :101-103
                 - (uVel[i, j, K]+uVel[i+1, j, K]))
             * grid.recip_drC[K])
    tempv = (.5*(vVel[i, j, Km1]+vVel[i, j+1, Km1]                      # :104-106
                 - (vVel[i, j, K]+vVel[i, j+1, K]))
             * grid.recip_drC[K])
    vertShear = vertShear.at[i, j].set(tempu*tempu+tempv*tempv)         # :107
    buoyFreq = buoyFreq.at[i, j].set(params.gravity*params.mass2rUnit   # :112-113
                                     * (rhoKm1[i, j] - rhoK[i, j])*grid.recip_drC[K])
    RiNumber = RiNumber.at[i, j].set(-buoyFreq[i, j]/MAX(vertShear[i, j], epsilon, p="b"))   # :118
    return RiNumber, buoyFreq, vertShear
