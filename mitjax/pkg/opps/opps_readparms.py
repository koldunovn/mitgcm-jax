"""OPPS_READPARMS: pkg/opps/opps_readparms.F @63cdc0b."""

import numpy as np

from mitjax.params_io import RunParams, fortran_default
from mitjax.pkg.opps.opps_h import OPPS

_RP = "pkg/opps/opps_readparms.F"
_F = "data.opps"
_G = "OPPS_PARM01"

# (name, opps_readparms.F line of its default, static type or None for REAL)
_DEFAULTS = (("MAX_ABE_ITERATIONS", 65, int), ("OPPSdebugLevel", 66, int), ("PlumeRadius", 67, None),
             ("STABILITY_THRESHOLD", 68, None), ("FRACTIONAL_AREA", 69, None), ("MAX_FRACTIONAL_AREA", 70, None),
             ("VERTICAL_VELOCITY", 71, None), ("ENTRAINMENT_RATE", 72, None), ("useGCMwVel", 73, bool))


def opps_readparms(exp):
    """OPPS_READPARMS( myThid )   @63cdc0b pkg/opps/opps_readparms.F:7-108

    C     Initialize OPPS parameters, read in data.opps

    `exp`: mitjax.config.params.Experiment. Returns an `OPPS` (OPPS.h) or None when useOPPS is .FALSE. (:49-58).
    Each namelist variable: the run's data.opps or its default of :65-73 (`fortran_default`); e2 = 2.*ENTRAINMENT_RATE
    (:104; `2.` is REAL*4, exact). Host side, float64, once.
    """
    if not exp.cfg.use_flag("useOPPS"):                                # :49-58
        return None
    rp = RunParams(exp.run)
    f = {}
    for name, line, typ in _DEFAULTS:                                   # :65-73 defaults, :86 READ
        v = rp.get(_F, _G, name, default=fortran_default(f"{_RP}:{line}", name, exp))
        f[name] = typ(v) if typ is not None else np.float64(v)
    f["e2"] = np.float64(2.0)*f["ENTRAINMENT_RATE"]                     # :104  e2 = 2.*ENTRAINMENT_RATE
    return OPPS({"OPPSisOn": True, **f})
