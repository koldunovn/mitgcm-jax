"""GGL90.h: the GGL90 (and IDEMIX, Langmuir) parameters and fields by Fortran name (pkg/ggl90/GGL90.h @63cdc0b).

`Ggl90` holds the common-block variables of GGL90.h that the ported routines read or write: the parameters set by
GGL90_READPARMS (LOGICAL, INTEGER and CHARACTER values static; REAL values float64 leaves, traced when a `Ggl90` is a
jit argument, so that XLA never folds them [L-XLA-4]) and the fields of GGL90_INIT_VARIA / GGL90_CALC /
GGL90_IDEMIX, each an `FArray` with its GGL90.h declaration (`DECLARATIONS`, one row per declaration with its header
line). A REAL parameter that selects code (`GGL90diffTKEh .GT. 0`, `IDEMIX_tau_h .GT. 0`) is also kept as a host
value, read with `ggl.static_float(name)` (decided once on the host; a perturbation of such a parameter must not
cross the threshold). Immutable: `ggl.replace(GGL90TKE=...)` returns a new one, as a Fortran routine writes the
common block.

Not carried: mskCor (#ifdef ALLOW_GGL90_SMOOTH, GGL90.h:108-111; the option raises), GGL90dumpFreq, GGL90mixingMaps
(output only, GGL90_OUTPUT is not ported) and GGL90isOn (set nowhere in pkg/ggl90 at 63cdc0b).
"""

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.farray import FArray

# GGL90.h:66-69 (PARAMETERs, REAL*8 literals with a D exponent)
SQRTTWO = 1.41421356237310           # GGL90.h:67  PARAMETER ( SQRTTWO = 1.41421356237310D0 )
GGL90eps = 2.23e-16                  # GGL90.h:69  PARAMETER ( GGL90eps = 2.23D-16 )

# name -> (GGL90.h line, shape kind); kinds: xy (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy), xyz (... ,Nr,nSx,nSy)
DECLARATIONS = {
    "GGL90TKE": (92, "xyz"), "GGL90viscArU": (93, "xyz"), "GGL90viscArV": (94, "xyz"), "GGL90diffKr": (95, "xyz"),
    # #ifdef ALLOW_GGL90_IDEMIX (GGL90.h:113-168)
    "IDEMIX_E": (149, "xyz"), "IDEMIX_F_B": (150, "xy"), "IDEMIX_F_S": (151, "xy"),
}

# the parameters (GGL90.h line): REAL -> float64 leaves; LOGICAL / INTEGER / CHARACTER -> static
PARAMETERS = {
    "GGL90TKEFile": 71, "GGL90ck": 74, "GGL90ceps": 74, "GGL90alpha": 75, "GGL90m2": 75, "GGL90diffTKEh": 76,
    "GGL90mixingLengthMin": 77, "GGL90TKEmin": 78, "GGL90TKEsurfMin": 78, "GGL90TKEbottom": 78,
    "GGL90viscMax": 79, "GGL90diffMax": 79, "mxlMaxFlag": 81, "adMxlMaxFlag": 82,
    "GGL90writeState": 99, "GGL90_dirichlet": 100, "mxlSurfFlag": 100, "calcMeanVertShear": 100,
    "useIDEMIX": 101, "useLANGMUIR": 102,
    # #ifdef ALLOW_GGL90_IDEMIX
    "IDEMIX_tau_v": 135, "IDEMIX_tau_h": 135, "IDEMIX_gamma": 135, "IDEMIX_jstar": 135, "IDEMIX_mu0": 136,
    "IDEMIX_diff_min": 136, "IDEMIX_mixing_efficiency": 137, "IDEMIX_diff_max": 137, "IDEMIX_frac_F_b": 138,
    "IDEMIX_frac_F_s": 138, "IDEMIX_tidal_file": 161, "IDEMIX_wind_file": 161, "IDEMIX_include_GM": 165,
    "IDEMIX_include_GM_bottom": 165,
    # #ifdef ALLOW_GGL90_LANGMUIR
    "LC_Gamma": 174, "LC_num": 174, "LC_lambda": 174,
}


def _bounds(kind, size):
    sNx, sNy, OLx, OLy, Nr = size.sNx, size.sNy, size.OLx, size.OLy, size.Nr
    return {"xy": dict(i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy)),
            "xyz": dict(i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy), k=(1, Nr))}[kind]


def declare(name, size, fill=np.nan):
    """A GGL90.h array with its declared bounds, every point `fill` (NaN: not yet written by any routine)."""
    _, kind = DECLARATIONS[name]
    b = _bounds(kind, size)
    shape = tuple(hi - lo + 1 for ax in ("k", "j", "i") if ax in b for lo, hi in [b[ax]])
    if size.nPx != 1 or size.nPy != 1:
        raise NotImplementedError("ggl90: a multi-process tiling (nPx*nPy > 1) is not ported")
    shape = (size.nSx * size.nSy,) + shape
    return FArray(jnp.full(shape, fill, dtype=jnp.float64), name, **b)


def wrap(name, data, size):
    """A GGL90.h array from a plain [tile, (k), j, i] array, with its declaration."""
    _, kind = DECLARATIONS[name]
    return FArray(jnp.asarray(data, jnp.float64), name, **_bounds(kind, size))


def _is_static(v):
    return isinstance(v, (bool, str, int)) and not isinstance(v, (np.floating, float))


@jax.tree_util.register_pytree_node_class
class Ggl90:
    """The GGL90.h common blocks by Fortran name: `ggl.GGL90ck`, `ggl.mxlMaxFlag`, `ggl.GGL90TKE`.

    REAL parameters are numpy float64 scalars (jit leaves), arrays FArrays (leaves), logicals, strings and integers
    static (part of the tree structure, so a branch on them is a Python `if`); `static_float(name)` is the host value
    of a REAL that selects code."""

    __slots__ = ("_f", "_h")

    def __init__(self, fields=None, host=None):
        object.__setattr__(self, "_f", dict(fields or {}))
        object.__setattr__(self, "_h", tuple(sorted(dict(host or {}).items())))

    def __getattr__(self, name):
        f = object.__getattribute__(self, "_f")
        if name in f:
            return f[name]
        if name in DECLARATIONS or name in PARAMETERS:
            raise AttributeError(f"{name} (GGL90.h) has not been set by any ported routine yet")
        raise AttributeError(name)

    def __setattr__(self, name, value):
        raise AttributeError("Ggl90 is immutable; use ggl.replace(...)")

    def __contains__(self, name):
        return name in self._f

    def names(self):
        return sorted(self._f)

    def static_float(self, name):
        """The host value of a REAL GGL90.h parameter that selects code (set by GGL90_READPARMS)."""
        for k, v in self._h:
            if k == name:
                return v
        raise AttributeError(f"GGL90.h {name}: no host value")

    def replace(self, **fields):
        for k in fields:
            if k not in DECLARATIONS and k not in PARAMETERS:
                raise KeyError(f"{k} is not a GGL90.h variable carried by mitjax/pkg/ggl90/ggl90_h.py")
        return Ggl90({**self._f, **fields}, dict(self._h))

    def tree_flatten(self):
        keys = sorted(self._f)
        static = tuple((k, self._f[k]) for k in keys if _is_static(self._f[k]))
        dyn = tuple(k for k in keys if not _is_static(self._f[k]))
        return tuple(self._f[k] for k in dyn), (dyn, static, self._h)

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        dyn, static, host = aux
        return cls({**dict(zip(dyn, leaves)), **dict(static)}, dict(host))

    def __repr__(self):
        return f"Ggl90({', '.join(self.names())})"
