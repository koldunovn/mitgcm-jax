"""CYCLE_TRACER: model/src/cycle_tracer.F @63cdc0b."""

from mitjax.farray import loops_kji


def cycle_tracer(tracer, gTracer, myTime, myIter, *, cfg):
    """CYCLE_TRACER( bi, bj, tracer, gTracer, myTime, myIter, myThid )   @63cdc0b model/src/cycle_tracer.F:6-53

    C     | S/R CYCLE_TRACER
    C     | o Cycles the time-stepping arrays for a tracer field
    C     tracer  :: tracer field
    C     gTracer :: tracer tendency (input: updated tracer)

    Returns tracer (gTracer is not written). Both (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr); the copy (:44-50) is
    independent point by point: a k-vectorised nest."""
    sz = cfg.size
    k, j, i = loops_kji((1, sz.Nr), (1-sz.OLy, sz.sNy+sz.OLy), (1-sz.OLx, sz.sNx+sz.OLx))     # :44-46
    tracer = tracer.at[i, j, k].set(gTracer[i, j, k])                                          # :47
    return tracer
