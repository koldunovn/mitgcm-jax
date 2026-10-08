"""MY82_READPARMS: pkg/my82/my82_readparms.F @63cdc0b."""

import numpy as np

from mitjax.params_io import RunParams, fortran_default
from mitjax.pkg.my82.my82_h import MY82

_RP = "pkg/my82/my82_readparms.F"
_F = "data.my82"
_G = "MY_PARM01"

# (name, my82_readparms.F line of its default): namelist variables the ported code reads (:61-63)
_DEFAULTS = (("MYviscMax", 61), ("MYdiffMax", 62), ("MYhblScale", 63))


def my82_readparms(exp):
    """MY82_READPARMS( myThid )   @63cdc0b pkg/my82/my82_readparms.F:7-109

    C     Initialize MY82 parameters, read in data.pp

    `exp`: mitjax.config.params.Experiment. Returns a `MY82` (MY82.h) or None when useMY82 is .FALSE. (:45-54).
    MYviscMax, MYdiffMax, MYhblScale: the run's data.my82 or the default of :61-63 (`fortran_default`); RiMax is
    not in MY_PARM01 (:35-41): always its default `0.1950 _d 0` (:64). The checks :96-105 stop as the Fortran.
    MYdumpFreq, MYMixingMaps, MYwriteState (:65-67) select output only. Host side, float64, once.
    """
    if not exp.cfg.use_flag("useMY82"):                                # :45-54
        return None
    rp = RunParams(exp.run)
    f = {name: np.float64(rp.get(_F, _G, name, default=fortran_default(f"{_RP}:{line}", name, exp)))
         for name, line in _DEFAULTS}                                   # :61-63 defaults, :78 READ
    f["RiMax"] = np.float64(0.1950)                                     # :64  RiMax = 0.1950 _d 0
    if f["MYviscMax"] <= 0.0:                                           # :96-100
        raise RuntimeError("MYviscMax must be greater than zero\nABNORMAL END: S/R MY82_READPARMS")
    if f["MYdiffMax"] <= 0.0:                                           # :101-105
        raise RuntimeError("MYdiffMax must be greater than zero\nABNORMAL END: S/R MY82_READPARMS")
    return MY82({"MYisOn": True, **f})
