"""TIMESTEP_TRACER: model/src/timestep_tracer.F @63cdc0b."""

from mitjax.farray import loops_kji


def timestep_tracer(deltaTLev, tracer, gTracer, myTime, myIter, *, cfg):
    """TIMESTEP_TRACER( bi, bj, deltaTLev, tracer, gTracer, myTime, myIter, myThid )
    @63cdc0b model/src/timestep_tracer.F:7-69

    C     | S/R TIMESTEP_TRACER
    C     | o Step tracer field forward in time
    C     deltaTLev :: time-step [s] (vertical dependent)
    C     tracer    :: tracer field at current time step
    C     gTracer   :: input: tracer tendency ; output: updated tracer

    Returns gTracer. tracer, gTracer: (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr); deltaTLev(Nr) not tiled. The points of the
    k/j/i nest (:51-60) are independent (each reads and writes its own point): a k-vectorised nest. Under ALLOW_OBCS
    (:56-63) the only code line, the maskInC factor (:62), is commented out: nothing to port."""
    sz = cfg.size
    k, j, i = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))     # :51-53
    gTracer = gTracer.at[i, j, k].set(tracer[i, j, k]                                          # :54-55
                                      + deltaTLev[k]*gTracer[i, j, k])
    return gTracer
