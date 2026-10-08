"""EXF_PARAM.h and EXF_CONSTANTS.h of pkg/exf @63cdc0b: the EXF parameters as one pytree.

`ExfParams.r` holds the REAL parameters (numpy float64 leaves: traced when an ExfParams is a jit argument, so XLA
never folds them, params_io.params_pytree); `ExfParams.s` the LOGICAL, INTEGER and CHARACTER parameters (static).
Both are read by Fortran name: `exf.atmrho`, `exf.useAtmWind`, `exf.atempfile`. Values are set by EXF_READPARMS
(exf_readparms.py) and the start times by EXF_INIT_FIXED (exf_init_fixed.py).

EXF_CONSTANTS.h PARAMETERs (module constants, with their lines):
"""

import dataclasses

import numpy as np

from mitjax.params_io import params_pytree

exf_half = 0.5             # EXF_CONSTANTS.h:31  exf_half =  0.5 _d 0
exf_one = 1.0              # EXF_CONSTANTS.h:32  exf_one  =  1.0 _d 0
exf_two = 2.0              # EXF_CONSTANTS.h:33  exf_two  =  2.0 _d 0
stefanBoltzmann = 5.670e-8  # EXF_CONSTANTS.h:46  PARAMETER ( stefanBoltzmann = 5.670 _d -8 )
karman = 0.4               # EXF_CONSTANTS.h:47  PARAMETER ( karman = 0.4 _d 0 )
ustofu11 = 0.381800        # EXF_CONSTANTS.h:73  ustofu11    =         0.381800 _d 0
clindrag_1 = 0.000065      # EXF_CONSTANTS.h:75  clindrag_1  =         0.000065 _d 0
clindrag_2 = 0.000490      # EXF_CONSTANTS.h:76  clindrag_2  =         0.000490 _d 0
niter_bulk = 2             # EXF_CONSTANTS.h:87  PARAMETER ( niter_bulk = 2 )


@params_pytree
@dataclasses.dataclass(frozen=True)
class ExfParams:
    """EXF_PARAM.h: `r` {name: float64} REAL parameters (traced), `s` ((name, value), ...) the others (static)."""
    r: dict
    s: tuple

    def __getattr__(self, name):
        r = object.__getattribute__(self, "r")
        if name in r:
            return r[name]
        for k, v in object.__getattribute__(self, "s"):
            if k == name:
                return v
        raise AttributeError(f"ExfParams has no parameter {name!r}")

    def replace(self, **kw):
        """A copy with parameters replaced (REAL ones into `r`, the others into `s`)."""
        r = dict(self.r)
        s = dict(self.s)
        for k, v in kw.items():
            if k in r or (k not in s and isinstance(v, (float, np.floating))):
                r[k] = np.float64(v)
            else:
                s[k] = v
        return ExfParams(r=r, s=tuple(sorted(s.items())))
