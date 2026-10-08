"""OPPS_INTERFACE: pkg/opps/opps_interface.F @63cdc0b."""

import jax.numpy as jnp

from mitjax.farray import FArray
from mitjax.pkg.opps.opps_calc import opps_calc
from mitjax.pkg.opps.opps_h import NTIME_MAX


def opps_interface(iMin, iMax, jMin, jMax, myTime, myIter, *, cfg, grid, params, eos, state, op,
                   nTimeMax=NTIME_MAX):
    """OPPS_INTERFACE( bi, bj, iMin, iMax, jMin, jMax, myTime, myIter, myThid )
    @63cdc0b pkg/opps/opps_interface.F:6-213

    C     | SUBROUTINE OPPS_INTERFACE                                      |
    C     | o Driver for OPPS mixing scheme that can be called             |
    C     |   instead of convective_adjustment.                            |
    C     |   Reference: Paluszkiewicz+Romea, Dynamics of Atmospheres and  |
    C     |   Oceans (1997) 26, pp. 95-130                                 |
    C     | o Support for passive tracers by joint treatment of            |
    C     |   active (theta, salt) and passive tracers.
    C     iMin, iMax, jMin, jMax :: Loop range
    C     myTime :: Current time in simulation
    C     myIter :: Current iteration in simulation

    Returns (theta, salt, OPPSconvectCount, counters): DYNVARS.h theta and salt with every level of the columns
    (I, J) of iMin:iMax, jMin:jMax where kSurfC .LE. Nr replaced by OPPS_CALC's tracerLoc (:178-181), every other
    point unchanged; OPPSconvectCount is the local the Fortran passes to DIAGNOSTICS_FILL (:193-198, output only),
    zero except where a plume mixed (:101-107, :421 of opps_calc.F). The DO J / DO I loop over columns (:109-191) is
    vectorised (columns are independent; OPPS_CALC takes the column mask, see mitjax/pkg/opps/opps_calc.py).
    `counters` ({"ntimeOver": int32 [tile]}) go to the host check opps_calc.opps_calc_host after the step.
    nTimeMax: the static bound of OPPS_CALC's time integration (opps_h.NTIME_MAX).

    Ported: #else of ALLOW_PTRACERS (nTracer = nTracerInUse = 2, :69-70, :88-89); ALLOW_PTRACERS raises. Both
    useGCMwVel branches (:112-123). ALLOW_OPPS_DEBUG (:132-146, :153-177) only prints (no STDOUT in the port; the
    tMin..sMaxNew values are read by nothing else). ALLOW_OPPS_SNAPSHOT is #undef'd (raises).
    """
    sz = cfg.size
    Nr = sz.Nr
    if cfg.cpp.ALLOW_PTRACERS:
        raise NotImplementedError("OPPS_INTERFACE: ALLOW_PTRACERS (passive tracers in OPPS) is not ported")
    if cfg.cpp.flag("ALLOW_OPPS_SNAPSHOT", "OPPS_OPTIONS.h"):
        raise NotImplementedError("OPPS_INTERFACE: ALLOW_OPPS_SNAPSHOT is not ported")
    nTracer = 2                                                         # :69-70  PARAMETER( nTracer = 2 )
    nTracerInUse = 2                                                    # :88-89
    theta, salt = state.theta, state.salt
    lo_i, lo_j = 1-sz.OLx, 1-sz.OLy
    shp = theta.data.shape                                              # [tile, k, j, i]
    OPPSconvectCount = jnp.zeros(shp)                                   # :101-107
    jj = jnp.arange(shp[2])[:, None] + lo_j
    ii = jnp.arange(shp[3])[None, :] + lo_i
    inLoop = (jj >= jMin) & (jj <= jMax) & (ii >= iMin) & (ii <= iMax)  # DO J=jMin,jMax / DO I=iMin,iMax (:109-110)
    col = inLoop[None] & (grid.kSurfC.data <= Nr)                       # :111  IF ( kSurfC(I,J) .LE. Nr )
    if op.useGCMwVel:                                                   # :112-117
        wVelLoc = state.wVel.data
    else:                                                               # :118-123
        wVelLoc = jnp.broadcast_to(-op.VERTICAL_VELOCITY, shp)
    kMax = grid.kLowC.data                                              # :147
    tracerLoc, OPPSconvectCount, counters = opps_calc(                  # :148-152
        [theta.data, salt.data], OPPSconvectCount, wVelLoc, kMax, nTracer, nTracerInUse, col, myTime, myIter,
        cfg=cfg, grid=grid, params=params, eos=eos, state=state, op=op, nTimeMax=nTimeMax)
    c4 = col[:, None]
    theta = FArray(jnp.where(c4, tracerLoc[0], theta.data), theta.name, tiled=theta.tiled, _dims=theta.dims)  # :178-181
    salt = FArray(jnp.where(c4, tracerLoc[1], salt.data), salt.name, tiled=salt.tiled, _dims=salt.dims)
    return theta, salt, OPPSconvectCount, counters
