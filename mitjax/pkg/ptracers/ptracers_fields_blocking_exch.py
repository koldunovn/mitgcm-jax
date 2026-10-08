"""PTRACERS_FIELDS_BLOCKING_EXCH: pkg/ptracers/ptracers_fields_blocking_exch.F @63cdc0b."""

from mitjax.eesupp.exch_rs import EXCH_3D_RL
from mitjax.pkg.generic_advdiff.gad_exch_som import gad_exch_som

_OPT = "PTRACERS_OPTIONS.h"


def ptracers_fields_blocking_exch(*, cfg, ptr, ptf, ex):
    """PTRACERS_FIELDS_BLOCKING_EXCH( myThid )   @63cdc0b pkg/ptracers/ptracers_fields_blocking_exch.F:8-63

    C     Exchange data to update overlaps for passive tracers

    Returns `ptf` with pTracer exchanged (EXCH_3D_RL, Nr levels, :47-48) for every tracer 1..PTRACERS_numInUse with
    PTRACERS_StepFwd, and its SOM moments (GAD_EXCH_SOM, :49-54) under PTRACERS_ALLOW_DYN_STATE with SOM advection."""
    Nr = cfg.size.Nr
    for iTracer in range(1, ptr.PTRACERS_numInUse+1):                           # :43
        if ptr.PTRACERS_StepFwd[iTracer-1]:                                     # :44
            ptf = ptf.set("pTracer", iTracer, EXCH_3D_RL(ptf.pTracer[iTracer-1], Nr, ex=ex))   # :47-48
            if cfg.cpp.flag("PTRACERS_ALLOW_DYN_STATE", _OPT) and ptr.PTRACERS_SOM_Advection[iTracer-1]:   # :49-54
                ptf = ptf.set("som", iTracer, gad_exch_som(ptf.som[iTracer-1], Nr, ex=ex))
    return ptf
