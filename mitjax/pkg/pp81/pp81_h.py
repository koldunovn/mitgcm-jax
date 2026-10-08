"""PP81.h: the Pacanowski and Philander (1981) parameters and fields by Fortran name (pkg/pp81/PP81.h @63cdc0b).

`PP81` holds the common-block variables of PP81.h: the parameters PP81_READPARMS sets (PPnRi an INTEGER, static; the
REAL parameters numpy float64 scalars, jit leaves, so that XLA never folds them) and the fields PPviscAr, PPdiffKr
(FArrays with their PP81.h declarations, PP81_INIT_VARIA / PP81_CALC). Immutable: `pp.replace(PPviscAr=...)` returns
a new one, as a Fortran routine writes the common block. PPdumpFreq, PPmixingMaps, PPwriteState select output only
and are not carried.
"""

import jax
import numpy as np

from mitjax.farray import FArray

# name -> PP81.h line of the declaration
DECLARATIONS = {"PPviscAr": 49, "PPdiffKr": 50}
# PP81.h:38-47: INTEGER PPnRi (static); REAL parameters (leaves)
PARAMETERS = {"PPnRi": 38, "PPviscMin": 41, "PPdiffMin": 41, "PPviscMax": 41, "PPnu0": 42, "PPalpha": 42,
              "RiLimit": 42, "PP81isOn": 53}


def declare(name, size, data):
    """A PP81.h field (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy) from a [tile, k, j, i] array."""
    if name not in DECLARATIONS:
        raise KeyError(f"{name} is not a PP81.h field")
    return FArray(data, name, i=(1-size.OLx, size.sNx+size.OLx), j=(1-size.OLy, size.sNy+size.OLy),
                  k=(1, size.Nr))


def _is_static(v):
    return isinstance(v, (bool, str, int)) and not isinstance(v, (np.floating, float))


@jax.tree_util.register_pytree_node_class
class PP81:
    """The PP81.h common blocks by Fortran name (`pp.PPnRi`, `pp.RiLimit`, `pp.PPviscAr`)."""

    __slots__ = ("_f",)

    def __init__(self, fields=None):
        object.__setattr__(self, "_f", dict(fields or {}))

    def __getattr__(self, name):
        f = object.__getattribute__(self, "_f")
        if name in f:
            return f[name]
        if name in DECLARATIONS or name in PARAMETERS:
            raise AttributeError(f"{name} (PP81.h) has not been set by any ported routine yet")
        raise AttributeError(name)

    def __setattr__(self, name, value):
        raise AttributeError("PP81 is immutable; use pp.replace(...)")

    def replace(self, **fields):
        for k in fields:
            if k not in DECLARATIONS and k not in PARAMETERS:
                raise KeyError(f"{k} is not a PP81.h variable")
        return PP81({**self._f, **fields})

    def tree_flatten(self):
        keys = sorted(self._f)
        static = tuple((k, self._f[k]) for k in keys if _is_static(self._f[k]))
        dyn = tuple(k for k in keys if not _is_static(self._f[k]))
        return tuple(self._f[k] for k in dyn), (dyn, static)

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        dyn, static = aux
        return cls({**dict(zip(dyn, leaves)), **dict(static)})
