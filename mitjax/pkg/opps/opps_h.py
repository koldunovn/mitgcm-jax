"""OPPS.h: the Ocean Penetrative Plume Scheme parameters by Fortran name (pkg/opps/OPPS.h @63cdc0b).

`OPPS` holds the common-block variables of OPPS.h that OPPS_READPARMS sets: the INTEGERs MAX_ABE_ITERATIONS and
OPPSdebugLevel and the LOGICALs OPPSisOn, useGCMwVel (static), the REAL parameters (numpy float64 scalars, jit
leaves). OPPS.h has no fields: the convection count is a local of OPPS_INTERFACE.
"""

import jax
import numpy as np

# The static bound of OPPS_CALC's time integration DO nn=1,ntime (opps_calc.F:369; ntime from :334-337 has no bound in
# the Fortran): twice the largest ntime measured over every oracle column (main-session decision 2026-10-02, lane
# COLMIX session 2). Measured (smallest bound without overflow, by bisection on the port, which is bitwise on all of
# them): 49 / 49 / 50 over the 48 x 25 replay columns of passes MDJWF / JMD95Z / JMD95Z+useGCMwVel
# (reference/replay_colmix, runs 27841070-27841072), <= 1 on the vermix model columns at the dumped iterations: 2 x 50.
# An overflow is counted and STOPs on the host (opps_calc.opps_calc_host); it is never silent.
NTIME_MAX = 100

# name -> OPPS.h line of the declaration
PARAMETERS = {"MAX_ABE_ITERATIONS": 44, "OPPSdebugLevel": 47, "PlumeRadius": 51, "STABILITY_THRESHOLD": 52,
              "FRACTIONAL_AREA": 53, "MAX_FRACTIONAL_AREA": 54, "VERTICAL_VELOCITY": 55, "ENTRAINMENT_RATE": 56,
              "e2": 57, "OPPSisOn": 77, "useGCMwVel": 77}


def _is_static(v):
    return isinstance(v, (bool, str, int)) and not isinstance(v, (np.floating, float))


@jax.tree_util.register_pytree_node_class
class OPPS:
    """The OPPS.h common blocks by Fortran name (`op.PlumeRadius`, `op.MAX_ABE_ITERATIONS`)."""

    __slots__ = ("_f",)

    def __init__(self, fields=None):
        object.__setattr__(self, "_f", dict(fields or {}))

    def __getattr__(self, name):
        f = object.__getattribute__(self, "_f")
        if name in f:
            return f[name]
        if name in PARAMETERS:
            raise AttributeError(f"{name} (OPPS.h) has not been set by any ported routine yet")
        raise AttributeError(name)

    def __setattr__(self, name, value):
        raise AttributeError("OPPS is immutable; use op.replace(...)")

    def replace(self, **fields):
        for k in fields:
            if k not in PARAMETERS:
                raise KeyError(f"{k} is not an OPPS.h variable")
        return OPPS({**self._f, **fields})

    def tree_flatten(self):
        keys = sorted(self._f)
        static = tuple((k, self._f[k]) for k in keys if _is_static(self._f[k]))
        dyn = tuple(k for k in keys if not _is_static(self._f[k]))
        return tuple(self._f[k] for k in dyn), (dyn, static)

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        dyn, static = aux
        return cls({**dict(zip(dyn, leaves)), **dict(static)})
