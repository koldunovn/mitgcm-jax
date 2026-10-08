"""RBCS_READPARMS: pkg/rbcs/rbcs_readparms.F @63cdc0b (M3 Task 30, tutorial_reentrant_channel)."""

import numpy as np

from mitjax.params_io import RunParams, fortran_default

_RP = "pkg/rbcs/rbcs_readparms.F"
_F = "data.rbcs"
_G = "RBCS_PARM01"

# RBCS_SIZE.h:13  PARAMETER( maskLEN = 3 )
maskLEN = 3

# (name, rbcs_readparms.F line of its default :101-125): the RBCS_PARM01 variables of RBCS_PARAMS.h
_DEFAULTS = (("useRBCuVel", 101), ("useRBCvVel", 102), ("useRBCtemp", 103), ("useRBCsalt", 104),
             ("tauRelaxU", 105), ("tauRelaxV", 106), ("tauRelaxT", 107), ("tauRelaxS", 108),
             ("relaxMaskUFile", 109), ("relaxMaskVFile", 110), ("relaxUFile", 114), ("relaxVFile", 115),
             ("relaxTFile", 116), ("relaxSFile", 117), ("rbcsIniter", 118), ("rbcsForcingPeriod", 119),
             ("rbcsForcingCycle", 120), ("rbcsForcingOffset", 121), ("rbcsVanishingTime", 122),
             ("rbcsSingleTimeFiles", 123), ("rbcsIter0", 125), ("useRBCptracers", 133))
# the relaxation time scales (_RL) are traced in Params; every other value selects a branch or a record at trace
# time (the forcing period / cycle / offset, the vanishing time, deltaTrbcs as host floats): static
REALS = ("tauRelaxU", "tauRelaxV", "tauRelaxT", "tauRelaxS")


def _blank(s):
    return str(s).strip() == ""


def rbcs_readparms(exp, *, deltaTClock, startTime):
    """RBCS_READPARMS( myThid )   @63cdc0b pkg/rbcs/rbcs_readparms.F:7-286

    C     | SUBROUTINE RBCS_READPARMS
    C     | o Routine to initialize RBCS variables and constants.

    `exp`: mitjax.config.params.Experiment; deltaTClock, startTime: PARAMS.h (INI_PARMS). Returns
    (static, traced): the RBCS_PARAMS.h values by Fortran name (tauRelaxU/V/T/S in `traced`, as float64; the others
    in `static`) plus relaxMaskTrFile (maskLEN entries, :161-170), or None when useRBCS is .FALSE.
    (:89-96). Defaults :101-125 from the cited lines (`fortran_default`); deltaTrbcs = deltaTClock (:124). The
    consistency checks :163-271 stop as the Fortran (RuntimeError naming ABNORMAL END) when errCount > 0.
    ALLOW_PTRACERS (RBCS_PARM02, :146, :248-268) is not ported: a build that compiles it raises."""
    cfg = exp.cfg
    if not cfg.use_flag("useRBCS"):                                # :89-96
        return None
    if cfg.cpp.ALLOW_PTRACERS:                                     # :126-132, :145-147, :247-268
        raise NotImplementedError("RBCS_READPARMS: ALLOW_PTRACERS (RBCS_PARM02, useRBCpTrNum) is not ported")
    rp = RunParams(exp.run)
    v = {}
    for name, line in _DEFAULTS:                                   # :101-125 defaults, :144 READ
        v[name] = rp.get(_F, _G, name, default=fortran_default(f"{_RP}:{line}", name, exp))
    v["deltaTrbcs"] = rp.get(_F, _G, "deltaTrbcs") if rp.has(_F, _G, "deltaTrbcs") else deltaTClock   # :124
    # :43-45  locSize = 4 (no ALLOW_PTRACERS); :111-113 relaxMaskFile(1:locSize) = ' '
    locSize = 4
    elems, _ = rp.get_array(_F, _G, "relaxMaskFile")
    relaxMaskFile = [" "] * locSize
    for idx, val in elems.items():
        i = idx[0] if isinstance(idx, tuple) else idx
        relaxMaskFile[int(i) - 1] = val
    relaxMaskTrFile = [" "] * maskLEN                              # :161-163
    errCount = 0                                                   # :164
    for irbc in range(1, locSize + 1):                             # :165-171
        if irbc <= maskLEN:
            relaxMaskTrFile[irbc - 1] = relaxMaskFile[irbc - 1]
        elif not _blank(relaxMaskFile[irbc - 1]):
            errCount += 1
    if cfg.cpp.DISABLE_RBCS_MOM and (v["useRBCuVel"] or v["useRBCvVel"]):   # :181-193
        errCount += 1
    if v["rbcsIniter"] != 0:                                       # :195-202
        errCount += 1
    elif (startTime < v["rbcsForcingOffset"] + 0.5 * v["rbcsForcingPeriod"]       # :203-221 (0.5 REAL*4: exact)
          and not v["rbcsSingleTimeFiles"]):
        if v["rbcsForcingCycle"] > 0.0:
            pass                                                   # :205-212 a warning only
        else:
            errCount += 1                                          # :213-220
    for flag, tau in (("useRBCuVel", "tauRelaxU"), ("useRBCvVel", "tauRelaxV"), ("useRBCtemp", "tauRelaxT"),
                      ("useRBCsalt", "tauRelaxS")):                # :222-245
        if v[flag] and v[tau] <= 0.0:
            errCount += 1
    if errCount > 0:                                               # :270-276
        raise RuntimeError(f"RBCS_READPARMS: detected {errCount:3d} fatal error(s); "
                           "ABNORMAL END: S/R RBCS_READPARMS")
    static = {k: (float(x) if isinstance(x, float) else x) for k, x in v.items() if k not in REALS}
    static["relaxMaskTrFile"] = tuple(relaxMaskTrFile)
    traced = {k: np.float64(v[k]) for k in REALS}
    return static, traced
