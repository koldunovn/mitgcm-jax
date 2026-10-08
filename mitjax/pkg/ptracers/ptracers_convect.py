"""PTRACERS_CONVECT: pkg/ptracers/ptracers_convect.F @63cdc0b."""

from mitjax.model.src.convectively_mixtracer import convectively_mixtracer


def ptracers_convect(k, weightA, weightB, *, cfg, ptr, ptf):
    """PTRACERS_CONVECT( bi, bj, k, weightA, weightB, myThid )   @63cdc0b pkg/ptracers/ptracers_convect.F:10-75

    C     do passive tracers convection
    C  weightA :: weight for level K-1
    C  weightB :: weight for level K

    `ptr`: PTRACERS_PARAMS.h/PTRACERS_START.h; `ptf`: PTRACERS_FIELDS.h. Returns `ptf` with pTracer mixed across
    interface k for every tracer 1..PTRACERS_numInUse with PTRACERS_StepFwd (static). The CADJ STORE (:55-61) is a
    TAF directive (no forward effect)."""
    for iTracer in range(1, ptr.PTRACERS_numInUse+1):                           # :52
        if ptr.PTRACERS_StepFwd[iTracer-1]:                                     # :53
            ptf = ptf.set("pTracer", iTracer, convectively_mixtracer(           # :63-66
                k, weightA, weightB, ptf.pTracer[iTracer-1], cfg=cfg))
    return ptf
