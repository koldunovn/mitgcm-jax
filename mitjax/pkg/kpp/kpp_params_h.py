"""KPP_PARAMS.h and KPP.h: the KPP parameters, fixed arrays and fields by Fortran name (pkg/kpp @63cdc0b).

`Kpp` holds the KPP_PARAMS.h common-block variables the ported routines read (KPP_READPARMS's parameters: LOGICAL and
INTEGER static, REAL float64 leaves, traced when a `Kpp` is a jit argument so that XLA never folds them [L-XLA-4];
KPP_INIT_FIXED's derived values Vtc, cg, deltaz, deltau and arrays zgrid, hwide, wmt, wst; KPP_INIT_VARIA's nzmax),
each array an `FArray` with its KPP_PARAMS.h declaration. Immutable: `kpp.replace(...)`.

The KPP.h fields (KPPviscAz, KPPdiffKzS, KPPdiffKzT, KPPghat, KPPhbl, KPPfrac) are a dict {name: FArray} as in
`kpp_calc_dummy.KPP_FIELDS` (the PTRACERS lane's convention, kept); `kpp_fields` declares them.

PARAMETERs of KPP_PARAMS.h:16-23 (`mdiff = 3`, `Nrm1 = Nr-1`, `Nrp1 = Nr+1`, `Nrp2 = Nr+2`,
`imt=(sNx+2*OLx)*(sNy+2*OLy)`) and :137-138 (`nni = 890, nnj = 480`) are module constants / functions of SIZE.h.
"""

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.farray import FArray
from mitjax.pkg.kpp.kpp_calc_dummy import KPP_FIELDS

MDIFF = 3                      # KPP_PARAMS.h:18  parameter( mdiff = 3    )
NNI, NNJ = 890, 480            # KPP_PARAMS.h:138 parameter (nni = 890, nnj = 480)


def imt(size):
    """KPP_PARAMS.h:23  parameter( imt=(sNx+2*OLx)*(sNy+2*OLy) )"""
    return (size.sNx + 2*size.OLx)*(size.sNy + 2*size.OLy)


# the KPP_PARAMS.h parameters (line of the declaration): REAL -> float64 leaves; LOGICAL / INTEGER -> static
PARAMETERS = {
    "kpp_freq": 44, "kpp_dumpFreq": 45, "KPPwriteState": 68, "KPP_ghatUseTotalDiffus": 68, "KPPuseDoubleDiff": 69,
    "LimitHblStable": 70, "KPPuseSWfrac3D": 71, "minKPPhbl": 77,
    "epsln": 92, "phepsi": 92, "epsilon": 92, "vonk": 92, "dB_dz": 92, "conc1": 93, "conam": 94, "concm": 94,
    "conc2": 94, "zetam": 94, "conas": 95, "concs": 95, "conc3": 95, "zetas": 95,
    "Ricr": 116, "cekman": 116, "cmonob": 116, "concv": 116, "Vtc": 116, "hbf": 117,
    "deltaz": 141, "deltau": 141, "zmin": 141, "zmax": 141, "umin": 141, "umax": 141,
    "num_v_smooth_Ri": 158, "Riinfty": 159, "BVSQcon": 159, "difm0": 160, "difs0": 160, "dift0": 160,
    "difmcon": 161, "difscon": 161, "diftcon": 161, "Rrho0": 177, "dsfmax": 177, "cstar": 190, "cg": 190,
    # derived on the host by KPP_READPARMS (static): kpp_freq .EQ. deltaTClock decides KPP_CALC's update test
    "kpp_freq_eq_deltaTClock": None,
}
ARRAYS = ("nzmax", "zgrid", "hwide", "wmt", "wst")


def _is_static(v):
    return isinstance(v, (bool, str, int)) and not isinstance(v, (np.floating, float))


@jax.tree_util.register_pytree_node_class
class Kpp:
    """The KPP_PARAMS.h common blocks by Fortran name: `kpp.Ricr`, `kpp.zgrid`, `kpp.wmt`, `kpp.nzmax`."""

    __slots__ = ("_f",)

    def __init__(self, fields=None):
        object.__setattr__(self, "_f", dict(fields or {}))

    def __getattr__(self, name):
        f = object.__getattribute__(self, "_f")
        if name in f:
            return f[name]
        if name in PARAMETERS or name in ARRAYS:
            raise AttributeError(f"{name} (KPP_PARAMS.h) has not been set by any ported routine yet")
        raise AttributeError(name)

    def __setattr__(self, name, value):
        raise AttributeError("Kpp is immutable; use kpp.replace(...)")

    def __contains__(self, name):
        return name in self._f

    def names(self):
        return sorted(self._f)

    def replace(self, **fields):
        for k in fields:
            if k not in PARAMETERS and k not in ARRAYS:
                raise KeyError(f"{k} is not a KPP_PARAMS.h variable carried by mitjax/pkg/kpp/kpp_params_h.py")
        return Kpp({**self._f, **fields})

    def tree_flatten(self):
        keys = sorted(self._f)
        static = tuple((k, self._f[k]) for k in keys if _is_static(self._f[k]))
        dyn = tuple(k for k in keys if not _is_static(self._f[k]))
        return tuple(self._f[k] for k in dyn), (dyn, static)

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        dyn, static = aux
        return cls({**dict(zip(dyn, leaves)), **dict(static)})

    def __repr__(self):
        return f"Kpp({', '.join(self.names())})"


def declare_xy(name, data, size):
    """An (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy) array from [tile, j, i] data."""
    return FArray(jnp.asarray(data), name, i=(1-size.OLx, size.sNx+size.OLx), j=(1-size.OLy, size.sNy+size.OLy))


def declare_xyz(name, data, size, nk=None):
    """An (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy) array (k=(1, nk), default Nr) from [tile, k, j, i] data."""
    return FArray(jnp.asarray(data), name, i=(1-size.OLx, size.sNx+size.OLx), j=(1-size.OLy, size.sNy+size.OLy),
                  k=(1, size.Nr if nk is None else nk))


def kpp_fields(size, values=None):
    """The KPP.h fields {name: FArray} (KPP.h:28-38: KPPviscAz, KPPdiffKzS, KPPdiffKzT, KPPghat (Nr levels), KPPhbl,
    KPPfrac), from `values` {name: [tile, (k), j, i] array}; a missing one is NaN (not yet written)."""
    T = size.nSx*size.nSy
    ny, nx = size.sNy + 2*size.OLy, size.sNx + 2*size.OLx
    values = values or {}
    out = {}
    for n in KPP_FIELDS:
        two_d = n in ("KPPhbl", "KPPfrac")
        shape = (T, ny, nx) if two_d else (T, size.Nr, ny, nx)
        a = jnp.asarray(values[n], jnp.float64) if n in values else jnp.full(shape, jnp.nan, jnp.float64)
        out[n] = declare_xy(n, a, size) if two_d else declare_xyz(n, a, size)
    return out
