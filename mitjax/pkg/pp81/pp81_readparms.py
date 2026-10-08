"""PP81_READPARMS: pkg/pp81/pp81_readparms.F @63cdc0b."""

import math

import numpy as np

from mitjax.model.grid import UNSET_RL
from mitjax.params_io import RunParams, fortran_default
from mitjax.pkg.pp81.pp81_h import PP81

_RP = "pkg/pp81/pp81_readparms.F"
_F = "data.pp81"
_G = "PP81_PARM01"

# (name, pp81_readparms.F line of its default) for the namelist variables the ported code reads (:64-73)
_DEFAULTS = (("PPnRi", 64), ("PPviscMin", 65), ("PPdiffMin", 66), ("PPviscMax", 67), ("PPnu0", 68),
             ("PPalpha", 69))


def pp81_readparms(exp, *, params):
    """PP81_READPARMS( myThid )   @63cdc0b pkg/pp81/pp81_readparms.F:7-127

    C     Initialize PP81 parameters, read in data.pp81

    `exp`: mitjax.config.params.Experiment (the run's data.pp81 as the model reads it, and the build); `params`:
    PARAMS.h (`viscArNr`, read at :120). Returns a `PP81` (PP81.h) or None when usePP81 is .FALSE. (:48-57: the
    routine returns after a warning). A namelist variable the file does not set takes its default of :64-69, read from
    that line by `fortran_default`; RiLimit (:70 UNSET_RL) is not a namelist variable of PP81_PARM01 (:35-44), so it
    is always derived at :117-123. Host side, float64, once (no jit): the setup-time IFs on REAL values (:102-116) are
    Python tests; the `**` of :120 with a REAL exponent is the C library pow (glibc, as gfortran's).
    PPdumpFreq, PPMixingMaps, PPwriteState (:71-73) select output only and are not carried.
    """
    if not exp.cfg.use_flag("usePP81"):                                # :48-57
        return None
    rp = RunParams(exp.run)
    v = {name: rp.get(_F, _G, name, default=fortran_default(f"{_RP}:{line}", name, exp))
         for name, line in _DEFAULTS}                                   # :64-69 defaults, :84 READ
    PPnRi = int(v["PPnRi"])
    f = {k: np.float64(v[k]) for k in ("PPviscMin", "PPdiffMin", "PPviscMax", "PPnu0", "PPalpha")}
    if f["PPviscMax"] <= 0.0:                                           # :102-106
        raise RuntimeError("PPviscMax must be greater than zero\nABNORMAL END: S/R PP81_READPARMS")
    if f["PPalpha"] == 0.0:                                             # :107-111
        raise RuntimeError("PPalpha must not be zero\nABNORMAL END: S/R PP81_READPARMS")
    if PPnRi == 0:                                                      # :112-116
        raise RuntimeError("PPnRi must not be zero\nABNORMAL END: S/R PP81_READPARMS")
    RiLimit = np.float64(UNSET_RL)                                      # :70  RiLimit = UNSET_RL
    if RiLimit == UNSET_RL:                                             # :117-123
        RiLimit = np.float64(PPnRi)                                     # :118  RiLimit = PPnRi
        viscArNr1 = float(np.asarray(params.viscArNr.data)[0])          # viscArNr(1)
        RiLimit = np.float64((                                          # :119-122
            math.pow(float((np.float64(f["PPnu0"]) + np.float64(viscArNr1)) / f["PPviscMax"]),
                     float(np.float64(1.0) / RiLimit))
            - np.float64(1.0))
            / f["PPalpha"])
    return PP81({"PP81isOn": True, "PPnRi": PPnRi, **f, "RiLimit": RiLimit})
