"""GMREDI.h: the GM/Redi parameters and fields by Fortran name (pkg/gmredi/GMREDI.h @63cdc0b).

`Gmredi` holds the common-block variables of GMREDI.h that the ported routines read or write: the parameters set by
GMREDI_READPARMS (logicals, strings and integers static; REAL parameters float64 leaves, traced when a `Gmredi` is a
jit argument, so that XLA never folds them [L-XLA-4]), the coefficient arrays of GMREDI_INIT_FIXED and the tensor
(and bolus stream-function) arrays of GMREDI_INIT_VARIA / GMREDI_CALC_TENSOR, each array an `FArray` with its
GMREDI.h declaration (`DECLARATIONS` below, one row per declaration with its header line). Immutable:
`gm.replace(Kwx=..., ...)` returns a new one, as a Fortran routine writes the common block.

Not carried (their options raise where the Fortran would read them): the Visbeck,
Bates and GEOM arrays (GMREDI.h under GM_VISBECK_VARIABLE_K, GM_BATES_K3D, GM_GEOM_VARIABLE_K; M3 lane MLAdjust
carries GM_LeithQG_K of ALLOW_GM_LEITH_QG), and the fm07 / sub-meso / BVP parameters (GM_facTrL2dz, subMeso_*, GM_BVP_*): no M1 experiment
selects them.
"""

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.farray import FArray

# GMREDI.h:15-18  _RL op5, op25; PARAMETER( op5 = 0.5 _d 0 ), PARAMETER( op25 = 0.25 _d 0 )
op5 = 0.5
op25 = 0.25

# name -> (GMREDI.h line, shape kind, type); kinds as mitjax/model/grid.py: xy, xyz, r (Nr, not tiled)
DECLARATIONS = {
    "GM_isoFac2d": (247, "xy", "RS"), "GM_bolFac2d": (248, "xy", "RS"),
    "GM_isoFac1d": (249, "r", "RS"), "GM_bolFac1d": (250, "r", "RS"),
    "Kwx": (272, "xyz", "RL"), "Kwy": (273, "xyz", "RL"), "Kwz": (274, "xyz", "RL"),
    "Kux": (280, "xyz", "RL"), "Kvy": (281, "xyz", "RL"),
    "Kuz": (288, "xyz", "RL"), "Kvz": (289, "xyz", "RL"),            # #ifdef GM_EXTRA_DIAGONAL (GMREDI.h:284-294)
    "GM_PsiX": (299, "xyz", "RL"), "GM_PsiY": (300, "xyz", "RL"),    # #ifdef GM_BOLUS_ADVEC (GMREDI.h:296-301)
    # GOADK lane (M2, global_ocean.90x40x15/code_ad): #ifdef GM_READ_K3D_REDI / GM_READ_K3D_GM (GMREDI.h:254-263)
    "GM_inpK3dRedi": (257, "xyz", "RL"), "GM_inpK3dGM": (262, "xyz", "RL"),
    # M3 lane MLAdjust (input.QGLthGM): #ifdef ALLOW_GM_LEITH_QG (GMREDI.h:357-362)
    "GM_LeithQG_K": (360, "xyz", "RL"),
}

# the parameters (GMREDI.h line): REAL -> float64 leaves; LOGICAL / CHARACTER -> static
PARAMETERS = {
    "GM_AdvForm": 42, "GM_AdvSeparate": 43, "GM_useBVP": 44, "GM_useSubMeso": 45, "GM_ExtraDiag": 46,
    "GM_InMomAsStress": 47, "GM_useBatesK3d": 50, "GM_useLeithQG": 57, "GM_useGEOM": 58,
    "GM_taper_scheme": 90, "GM_iso2dFile": 91, "GM_iso1dFile": 92, "GM_bol2dFile": 93, "GM_bol1dFile": 94,
    "GM_K3dRediFile": 95, "GM_K3dGMFile": 96,
    "GM_isopycK": 153, "GM_background_K": 154, "GM_maxSlope": 155, "GM_Kmin_horiz": 156, "GM_Small_Number": 157,
    "GM_slopeSqCutoff": 158, "GM_Scrit": 159, "GM_Sd": 159, "GM_isoFac_calcK": 160, "GM_Visbeck_alpha": 169,
    "GM_Visbeck_maxSlope": 173, "GM_rMaxSlope": 232, "GM_skewflx": 233,
}


def _bounds(kind, size):
    sNx, sNy, OLx, OLy, Nr = size.sNx, size.sNy, size.OLx, size.OLy, size.Nr
    return {"xy": dict(i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy)),
            "xyz": dict(i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy), k=(1, Nr)),
            "r": dict(k=(1, Nr), tiled=False)}[kind]


def declare(name, size, fill=np.nan):
    """A GMREDI.h array with its declared bounds, every point `fill` (NaN: not yet written by any routine)."""
    line, kind, _ = DECLARATIONS[name]
    b = _bounds(kind, size)
    tiled = b.pop("tiled", True)
    shape = tuple(hi - lo + 1 for ax in ("k", "j", "i") if ax in b for lo, hi in [b[ax]])
    if tiled:
        if size.nPx != 1 or size.nPy != 1:
            raise NotImplementedError("gmredi: a multi-process tiling (nPx*nPy > 1) is not ported")
        shape = (size.nSx * size.nSy,) + shape
    return FArray(jnp.full(shape, fill, dtype=jnp.float64), name, **b, tiled=tiled)


def wrap(name, data, size):
    """A GMREDI.h array from a plain [tile, (k), j, i] (or [k]) array, with its declaration."""
    line, kind, _ = DECLARATIONS[name]
    b = _bounds(kind, size)
    tiled = b.pop("tiled", True)
    return FArray(data, name, **b, tiled=tiled)


def _is_static(v):
    return isinstance(v, (bool, str, int)) and not isinstance(v, (np.floating, float))


@jax.tree_util.register_pytree_node_class
class Gmredi:
    """The GMREDI.h common blocks by Fortran name: `gm.GM_maxSlope`, `gm.GM_taper_scheme`, `gm.Kwx`.

    REAL parameters are numpy float64 scalars (jit leaves), arrays FArrays (leaves), logicals, strings and integers
    static (part of the tree structure, so a branch on them is a Python `if`)."""

    __slots__ = ("_f",)

    def __init__(self, fields=None):
        object.__setattr__(self, "_f", dict(fields or {}))

    def __getattr__(self, name):
        f = object.__getattribute__(self, "_f")
        if name in f:
            return f[name]
        if name in DECLARATIONS or name in PARAMETERS:
            raise AttributeError(f"{name} (GMREDI.h) has not been set by any ported routine yet")
        raise AttributeError(name)

    def __setattr__(self, name, value):
        raise AttributeError("Gmredi is immutable; use gm.replace(...)")

    def __contains__(self, name):
        return name in self._f

    def names(self):
        return sorted(self._f)

    def replace(self, **fields):
        for k in fields:
            if k not in DECLARATIONS and k not in PARAMETERS:
                raise KeyError(f"{k} is not a GMREDI.h variable carried by mitjax/pkg/gmredi/gmredi_h.py")
        return Gmredi({**self._f, **fields})

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
        return f"Gmredi({', '.join(self.names())})"
