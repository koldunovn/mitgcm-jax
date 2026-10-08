"""PTRACERS_INIT_FIXED: pkg/ptracers/ptracers_init_fixed.F @63cdc0b."""

from mitjax.pkg.generic_advdiff.gad_h import (ENUM_CENTERED_2ND, ENUM_CENTERED_4TH, ENUM_SOM_LIMITER,
                                              ENUM_SOM_PRATHER, ENUM_UPWIND_3RD)

_OPT = "PTRACERS_OPTIONS.h"
# GAD_ADVSCHEME_GET (pkg/generic_advdiff/gad_advscheme.F:124-128) is >= 0 only for the schemes GAD_INIT_FIXED
# registers (pkg/generic_advdiff/gad_init_fixed.F:44-62, GAD_ADVSCHEME_SET of every ENUM_* of GAD.h); any other
# number gives minSize < 0 and PTRACERS_INIT_FIXED stops (:59-71)
_KNOWN_SCHEMES = (1, 2, 3, 4, 7, 20, 30, 33, 40, 41, 42, 50, 51, 52, 77, 80, 81)


def ptracers_init_fixed(ptr, *, cfg, multiDimAdvection):
    """PTRACERS_INIT_FIXED( myThid )   @63cdc0b pkg/ptracers/ptracers_init_fixed.F:8-169

    C     Initialize PTRACERS constant

    `ptr`: the PtracersParams of PTRACERS_READPARMS; `multiDimAdvection`: PARAMS.h. Returns `ptr` with the internal
    per-tracer flags of :47-104: PTRACERS_MultiDimAdv, PTRACERS_SOM_Advection, PTRACERS_AdamsBashGtr,
    PTRACERS_AdamsBash_Tr (static tuples; tracers above PTRACERS_numInUse keep the initial values of :47-52).

    Not kept: the overlap check sizes (tracMinSize, GAD_OlMinSize update :58-74, :109-123: they only feed the
    overlap check of GAD_CHECK and a message), useMultiDimAdvec (:82-83, PARAMS.h; read only by checks and by
    GAD_INIT_FIXED's message in the ported runs). PTRACERS_INIT_FIXED_DYNAMIC (:130-137, PTRACERS_ALLOW_DYN_STATE:
    allocates the SOM moments) is the zero initialisation in PTRACERS_INIT_VARIA. PTRACERS_MNC_INIT and
    PTRACERS_DIAGNOSTICS_INIT (:150-162) are output set-up (not ported). The scheme check of :59-71 raises."""
    num = ptr.PTRACERS_num
    multi = [bool(multiDimAdvection)] * num                                    # :48
    som = [False] * num                                                         # :49
    abg = [False] * num                                                         # :50
    abt = [False] * num                                                         # :51
    for n in range(ptr.PTRACERS_numInUse):                                      # :56  DO iTracer = 1, PTRACERS_numInUse
        sch = ptr.PTRACERS_advScheme[n]
        if sch != 0 and sch not in _KNOWN_SCHEMES:                              # :59-71
            raise ValueError(f"PTRACERS_INIT_FIXED: invalid Adv. Scheme number={sch:6d} for tracer #{n+1:6d}")
        if sch in (ENUM_CENTERED_2ND, ENUM_UPWIND_3RD, ENUM_CENTERED_4TH, 0):  # :76-81
            multi[n] = False
        abg[n] = sch in (ENUM_CENTERED_2ND, ENUM_UPWIND_3RD, ENUM_CENTERED_4TH)  # :84-87
        if not ptr.PTRACERS_doAB_onGpTr:                                        # :88-91
            abt[n] = abg[n]
            abg[n] = False
        som[n] = ENUM_SOM_PRATHER <= sch <= ENUM_SOM_LIMITER                   # :93-95
        if not cfg.cpp.flag("PTRACERS_ALLOW_DYN_STATE", _OPT) and som[n]:       # :96-107
            raise ValueError("PTRACERS_INIT_FIXED: trying to use 2nd.Order-Moment Advection without dynamical "
                             "internal state data structures compiled")
    return ptr.replace(static=dict(PTRACERS_MultiDimAdv=tuple(multi), PTRACERS_SOM_Advection=tuple(som),
                                   PTRACERS_AdamsBashGtr=tuple(abg), PTRACERS_AdamsBash_Tr=tuple(abt)))
