"""MY82.h: the Mellor and Yamada (1982) level-2.0 parameters and fields by Fortran name (pkg/my82/MY82.h @63cdc0b).

`MY82` holds the common-block variables of MY82.h: the parameters MY82_READPARMS and MY82_INIT_VARIA set (REAL
parameters numpy float64 scalars, jit leaves) and the fields MYhbl, MYviscAr, MYdiffKr (FArrays with their MY82.h
declarations). Immutable: `my.replace(MYviscAr=...)` returns a new one. MYdumpFreq, MYmixingMaps, MYwriteState
select output only and are not carried. The magic PARAMETERs of Mellor & Yamada (M. Satoh, p.315) are module
constants.
"""

import jax
import numpy as np

from mitjax.farray import FArray

# MY82.h:36-40  PARAMETER( A1 = 0.92D0 ) ... PARAMETER( C1 = 0.08D0 )
A1 = 0.92
A2 = 0.74
B1 = 16.6
B2 = 10.1
C1 = 0.08

# name -> MY82.h line of the declaration
DECLARATIONS = {"MYhbl": 54, "MYviscAr": 55, "MYdiffKr": 56}
PARAMETERS = {"alpha1": 42, "alpha2": 42, "beta1": 43, "beta2": 43, "beta3": 43, "beta4": 43, "RiMax": 44,
              "MYhblScale": 45, "MYviscMax": 46, "MYdiffMax": 46, "MYisOn": 59}


def declare(name, size, data):
    """A MY82.h field from a [tile, k, j, i] (MYviscAr, MYdiffKr) or [tile, j, i] (MYhbl) array."""
    ij = dict(i=(1-size.OLx, size.sNx+size.OLx), j=(1-size.OLy, size.sNy+size.OLy))
    if name == "MYhbl":
        return FArray(data, name, **ij)
    if name in DECLARATIONS:
        return FArray(data, name, **ij, k=(1, size.Nr))
    raise KeyError(f"{name} is not a MY82.h field")


def _is_static(v):
    return isinstance(v, (bool, str, int)) and not isinstance(v, (np.floating, float))


@jax.tree_util.register_pytree_node_class
class MY82:
    """The MY82.h common blocks by Fortran name (`my.RiMax`, `my.beta1`, `my.MYviscAr`)."""

    __slots__ = ("_f",)

    def __init__(self, fields=None):
        object.__setattr__(self, "_f", dict(fields or {}))

    def __getattr__(self, name):
        f = object.__getattribute__(self, "_f")
        if name in f:
            return f[name]
        if name in DECLARATIONS or name in PARAMETERS:
            raise AttributeError(f"{name} (MY82.h) has not been set by any ported routine yet")
        raise AttributeError(name)

    def __setattr__(self, name, value):
        raise AttributeError("MY82 is immutable; use my.replace(...)")

    def replace(self, **fields):
        for k in fields:
            if k not in DECLARATIONS and k not in PARAMETERS:
                raise KeyError(f"{k} is not a MY82.h variable")
        return MY82({**self._f, **fields})

    def tree_flatten(self):
        keys = sorted(self._f)
        static = tuple((k, self._f[k]) for k in keys if _is_static(self._f[k]))
        dyn = tuple(k for k in keys if not _is_static(self._f[k]))
        return tuple(self._f[k] for k in dyn), (dyn, static)

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        dyn, static = aux
        return cls({**dict(zip(dyn, leaves)), **dict(static)})
