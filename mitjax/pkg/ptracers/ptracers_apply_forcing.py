"""PTRACERS_APPLY_FORCING: pkg/ptracers/ptracers_apply_forcing.F @63cdc0b."""

from mitjax.farray import loop_i, loop_j


def k_surface(routine, *, cfg, params):
    """kSurface (:57-65): 0 for an atmosphere, -1 for z coordinates with useShelfIce, Nr for p coordinates, else 1.
    Only the last is ported (the others raise)."""
    if params.fluidIsAir:                                                       # :57-58
        raise NotImplementedError(f"{routine}: fluidIsAir (kSurface = 0) is not ported")
    if params.usingZCoords and params.useShelfIce:                              # :59-60
        raise NotImplementedError(f"{routine}: useShelfIce (kSurface = -1, kSurfC test :90-100) is not ported")
    if params.usingPCoords:                                                     # :61-62
        raise NotImplementedError(f"{routine}: usingPCoords (kSurface = Nr) is not ported")
    return 1                                                                    # :63-64


def check_unported(routine, iTracer, *, cfg, params, ptr):
    """GCHEM_ADD_TENDENCY (:71-78), PTRACERS_linFSConserve (:102-112), RBCS_ADD_TENDENCY (:114-121): raise when
    active (no ported run uses them)."""
    if cfg.cpp.flag("ALLOW_GCHEM", "PTRACERS_OPTIONS.h") and cfg.use_flag("useGCHEM"):
        raise NotImplementedError(f"{routine}: GCHEM_ADD_TENDENCY (useGCHEM) is not ported")
    if ptr.PTRACERS_linFSConserve[iTracer-1]:
        raise NotImplementedError(f"{routine}: PTRACERS_linFSConserve (meanSurfCorPTr) is not ported")
    if cfg.cpp.flag("ALLOW_RBCS", "PTRACERS_OPTIONS.h") and cfg.use_flag("useRBCS"):
        raise NotImplementedError(f"{routine}: RBCS_ADD_TENDENCY (useRBCS) is not ported")


def ptracers_apply_forcing(gPtracer, surfForcPtr, iMin, iMax, jMin, jMax, k, iTracer, myTime, myIter, *, cfg, grid,
                           params, ptr):
    """PTRACERS_APPLY_FORCING( gPtracer, surfForcPtr, iMin,iMax,jMin,jMax, k, bi, bj, iTracer, myTime, myIter,
    myThid )   @63cdc0b pkg/ptracers/ptracers_apply_forcing.F:7-126

    C     Apply passive tracer forcing, i.e., sources and sinks of tracer,
    C      by adding forcing terms to the tendency array
    C  gPtracer             :: the tendency array
    C  surfForcPtr          :: surface forcing term
    C  iMin iMax jMin jMax  :: working range of tile for applying forcing
    C  k                    :: vertical level number
    C  iTracer              :: tracer number

    gPtracer, surfForcPtr: (1-OLx:sNx+OLx,1-OLy:sNy+OLy); k, iTracer Python ints. Returns gPtracer: at the surface
    level (k .EQ. kSurface, kSurface = 1 in the ported runs, :57-65) gPtracer + surfForcPtr*recip_drF(k)*recip_hFacC
    on 0:sNx+1, 0:sNy+1 (:80-89; the loop bounds of the code, not iMin..jMax, which are commented out); other levels
    unchanged. Raise: kSurface /= 1, GCHEM, linFSConserve, RBCS (`check_unported`)."""
    check_unported("PTRACERS_APPLY_FORCING", iTracer, cfg=cfg, params=params, ptr=ptr)
    kSurface = k_surface("PTRACERS_APPLY_FORCING", cfg=cfg, params=params)
    sz = cfg.size
    if k == kSurface:                                                           # :80
        j = loop_j(0, sz.sNy+1)                                                 # :83
        i = loop_i(0, sz.sNx+1)                                                 # :84
        gPtracer = gPtracer.at[i, j].set(gPtracer[i, j]                         # :85-87
                                         + surfForcPtr[i, j]
                                         * grid.recip_drF[k]*grid.recip_hFacC[i, j, k])
    return gPtracer
