"""SEAICE_PARAMS.h and SEAICE_SIZE.h of pkg/seaice @63cdc0b: the sea-ice parameters as one pytree, its PARAMETERs.

`SeaiceParams.r` holds the REAL parameters (numpy float64 leaves: traced when a SeaiceParams is a jit argument, so XLA
never folds them, params_io.params_pytree), incl. the REAL array `SEAICE_PDF` (a float64 vector of nITD; lane M4LAB
session 3); `SeaiceParams.s` the LOGICAL, INTEGER and CHARACTER parameters (static). Both are read by Fortran name: `sp.SEAICE_rhoIce`, `sp.usePW79thermodynamics`. Values are set
by SEAICE_READPARMS (seaice_readparms.py) and SEAICE_INIT_FIXED (seaice_init_fixed.py: SEAICE_mcPheePiston if unset).

`OceanParams` carries the PARAMS.h values the sea-ice routines read (REAL ones traced, flags static), by Fortran name.

PARAMETERs (module constants, with their lines):
"""

import dataclasses

import numpy as np

from mitjax.params_io import params_pytree

ZERO = 0.0                 # SEAICE_PARAMS.h:654  PARAMETER ( ZERO = 0.0 _d 0, ONE = 1.0 _d 0, TWO = 2.0 _d 0 )
ONE = 1.0                  # SEAICE_PARAMS.h:654
TWO = 2.0                  # SEAICE_PARAMS.h:654
QUART = 0.25               # SEAICE_PARAMS.h:656  PARAMETER ( QUART = 0.25 _d 0, HALF = 0.5 _d 0 )
HALF = 0.5                 # SEAICE_PARAMS.h:656
siEps = 1.0e-5             # SEAICE_PARAMS.h:658  PARAMETER ( siEps = 1. _d -5 )
MCPHEE_TAPER_FAC = 12.5    # SEAICE_PARAMS.h:664  PARAMETER ( MCPHEE_TAPER_FAC = 12.5 _d 0 , STANTON_NUMBER =
STANTON_NUMBER = 0.0056    # SEAICE_PARAMS.h:665  &            0.0056 _d 0, USTAR_BASE = 0.0125 _d 0 )
USTAR_BASE = 0.0125        # SEAICE_PARAMS.h:665
nITD_ITD = 5               # SEAICE_SIZE.h:22  PARAMETER (nITD = 5)   (#ifdef SEAICE_ITD)
nITD_NOITD = 7             # SEAICE_SIZE.h:24  PARAMETER (nITD = 7)   (#else)
SItrMaxNum = 3             # SEAICE_SIZE.h:29  PARAMETER(SItrMaxNum = 3 )   (lane M4LAB session 4: ALLOW_SITRACER)
GAD_HEFF = 1               # SEAICE_PARAMS.h:670  PARAMETER ( GAD_HEFF  = 1,   (lane M4ADCS32ICE: tape keys and
GAD_AREA = 2               # SEAICE_PARAMS.h:671  &            GAD_AREA  = 2,   SEAICE_ADVECTION's maxpass STOP)
GAD_SNOW = 3               # SEAICE_PARAMS.h:672  &            GAD_SNOW  = 3,
GAD_SALT = 4               # SEAICE_PARAMS.h:673  &            GAD_SALT  = 4,
GAD_SITR = 7               # SEAICE_PARAMS.h:676  &            GAD_SITR  = 7)
# SEAICE_TRACER.h:35-37: the REAL arrays (SItrMaxNum) of the sea-ice tracers, traced like SEAICE_PDF (lane M4LAB
# session 4)
SITR_REAL_ARRAYS = ("SItrFromOcean0", "SItrFromOceanFrac", "SItrFromFlood0", "SItrFromFloodFrac", "SItrExpand0")


def nITD(cfg):
    """SEAICE_SIZE.h:20-25: nITD = 5 with SEAICE_ITD, else 7."""
    return nITD_ITD if cfg.cpp.flag("SEAICE_ITD", "SEAICE_OPTIONS.h") else nITD_NOITD


@params_pytree
@dataclasses.dataclass(frozen=True)
class SeaiceParams:
    """SEAICE_PARAMS.h: `r` {name: float64} REAL parameters (traced), `s` ((name, value), ...) the others (static)."""
    r: dict
    s: tuple

    def __getattr__(self, name):
        r = object.__getattribute__(self, "r")
        if name in r:
            return r[name]
        for k, v in object.__getattribute__(self, "s"):
            if k == name:
                return v
        raise AttributeError(f"SeaiceParams has no parameter {name!r}")

    def replace(self, **kw):
        """A copy with parameters replaced (REAL ones into `r`, the others into `s`)."""
        r = dict(self.r)
        s = dict(self.s)
        for k, v in kw.items():
            if k == "SEAICE_PDF" or k in SITR_REAL_ARRAYS:                     # the REAL arrays (traced)
                r[k] = np.asarray(v, np.float64)
            elif k in r or (k not in s and isinstance(v, (float, np.floating))):
                r[k] = np.float64(v)
            else:
                s[k] = v
        return SeaiceParams(r=r, s=tuple(sorted(s.items())))


@params_pytree
@dataclasses.dataclass(frozen=True)
class OceanParams:
    """The PARAMS.h values the ported sea-ice routines read: REAL ones traced, flags static.
    `temp_EvPrRn_set` is the host flag of `temp_EvPrRn .NE. UNSET_RL` (seaice_growth.F:2345-2357)."""
    celsius2K: float
    rhoConst: float
    recip_rhoConst: float
    rhoConstFresh: float
    HeatCapacity_Cp: float
    gravity: float
    recip_gravity: float
    sIceLoadFac: float
    temp_EvPrRn: float
    useRealFreshWaterFlux: bool = dataclasses.field(metadata=dict(static=True))
    nonlinFreeSurf: int = dataclasses.field(metadata=dict(static=True))
    temp_EvPrRn_set: bool = dataclasses.field(metadata=dict(static=True))
    usingPCoords: bool = dataclasses.field(metadata=dict(static=True))
    # lane M4OFF: EEPARAMS.h useCubedSphereExchange (SEAICE_ADVECTION's passes, static)
    useCubedSphereExchange: bool = dataclasses.field(default=False, metadata=dict(static=True))
    # lane M4OFF session 3 (SEAICE_LSR): PARAMS.h debugLevel (static), deltaTClock, GRID.h globalArea (traced)
    debugLevel: int = dataclasses.field(default=0, metadata=dict(static=True))
    deltaTClock: float = np.float64(0.0)
    globalArea: float = np.float64(0.0)
    # lane M4CS32ICE (SEAICE_GROWTH with ALLOW_BALANCE_FLUXES compiled, :2440-2475, :2572-2664): PARAMS.h
    # selectBalanceEmPmR and balanceQnet (static; the defaults are set_defaults.F's after INI_PARMS)
    selectBalanceEmPmR: int = dataclasses.field(default=0, metadata=dict(static=True))
    balanceQnet: bool = dataclasses.field(default=False, metadata=dict(static=True))
