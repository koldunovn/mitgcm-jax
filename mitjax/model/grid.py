"""GRID.h: the model grid as Fortran-index arrays (plan Task 10), and the PARAMS.h values the grid routines read.

`Grid` holds the common-block variables of `model/inc/GRID.h` @63cdc0b that the grid routines set (INI_GRID and its
callees, SET_GRID_FACTORS, INI_DEPTHS, INI_MASKS_ETC, ADD_WALLS2MASKS, INI_CORI), plus the few `SURFACE.h` arrays they
also write (topoZ, h0FacC/W/S), each an `FArray` with its Fortran name and declared bounds (`DECLARATIONS` below, one
row per declaration with its header line). Storage is `[tile, k, j, i]` (mitjax/farray.py); tiles in the order
bi + (bj-1)*nSx (one process; the W2 tile order under pkg/exch2 for the default single facet, mitjax/eesupp/tiles.py).
A Grid is a pytree (its arrays are the leaves; field names are static), immutable: a routine returns
`grid.replace(xC=..., ...)` with the fields it wrote, as a Fortran routine writes the common block.

Where the grid is built (plan Task 10, [F§1] "grid builders run in numpy on the host, result device_put"): the grid
routines run ONCE, eagerly (op by op, no jit) on the CPU backend, under the gate XLA flags, and are never part of a
compiled model program; their result is a pytree of device arrays that drivers pass to jit as an argument
([L-ARCH-5]). Eager jax rather than numpy because FArray (Fortran-index arrays, the decided physics style) is built
on jax arrays (`.at[...].set`); eager dispatch runs every jax.numpy operation as its own XLA computation, so no
expression is fused or re-associated and each operation is one IEEE operation, as in the gfortran -O0 oracle; sin,
cos and tan of XLA:CPU are bitwise glibc 2.28 (Task 7c, mitjax/ops/libm.py), division is correctly rounded at the
gate flags. Cost (measured on a login node, 2026-10-01): 1-11 s per variant for the whole INITIALISE_FIXED grid
sequence, mostly per-operation dispatch and first-call compilation of the eager operations (the i/j recursions of
INI_LOCAL_GRID and the k sums of INI_MASKS_ETC are unrolled in Python).

The PARAMS.h / SET_GRID.h values the grid routines read (`GridParams`, built by `ini_parms_grid`) live in
mitjax/model/src/ini_parms.py; the RS exchanges in mitjax/eesupp/exch_rs.py; READ_REC_XY_RS in mitjax/pkg/rw/read_rec.py.
"""

import jax
import jax.numpy as jnp
import numpy as np

from mitjax.farray import FArray

# eesupp/inc/EEPARAMS.h:80-85 @63cdc0b
UNSET_RL = 1.234567e5       # EEPARAMS.h:81  PARAMETER ( UNSET_RL = 1.234567D5 )
UNSET_RS = 1.234567e5       # EEPARAMS.h:83  PARAMETER ( UNSET_RS = 1.234567D5 )
PRECFLOAT32 = 32            # EEPARAMS.h:63  PARAMETER ( precFloat32 = 32 )
# eesupp/inc/EEPARAMS.h:68-73: zeroRS = 0.0 _d 0, oneRS = 1.0 _d 0, zeroRL = 0.0 _d 0, oneRL = 1.0 _d 0,
# halfRL = 0.5 _d 0 (exact in binary; written as literals in the kernels)
zeroRS, oneRS = 0.0, 1.0    # EEPARAMS.h:69
zeroRL, oneRL = 0.0, 1.0    # EEPARAMS.h:72
halfRL = 0.5                # EEPARAMS.h:73
# model/inc/PARAMS.h:15-18 @63cdc0b (PARAMETERs, folded by gfortran in REAL*8: 2*PI is exact, /360 one rounding)
PI = 3.14159265358979323844   # PARAMS.h:16  PARAMETER ( PI = 3.14159265358979323844D0 )
deg2rad = 2.0 * PI / 360.0    # PARAMS.h:18  PARAMETER ( deg2rad = 2.D0*PI/360.D0 )

# ---------------------------------------------------------------------------------------------------------------
# Declarations: name -> (header, line, shape kind, type). Shape kinds:
#   xy   (1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)        xyz  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,nSx,nSy)
#   y    (1-OLy:sNy+OLy,nSx,nSy)                      r    (Nr)        rp1  (Nr+1)        0  scalar
DECLARATIONS = {
    # GRID.h COMMON /GRID_RL/ (GRID.h:314-335)
    "cosFacU": ("GRID.h", 320, "y", "RL"), "cosFacV": ("GRID.h", 321, "y", "RL"),
    "sqCosFacU": ("GRID.h", 322, "y", "RL"), "sqCosFacV": ("GRID.h", 323, "y", "RL"),
    "deepFacC": ("GRID.h", 324, "r", "RL"), "deepFac2C": ("GRID.h", 325, "r", "RL"),
    "deepFacF": ("GRID.h", 326, "rp1", "RL"), "deepFac2F": ("GRID.h", 327, "rp1", "RL"),
    "recip_deepFacC": ("GRID.h", 328, "r", "RL"), "recip_deepFac2C": ("GRID.h", 329, "r", "RL"),
    "recip_deepFacF": ("GRID.h", 330, "rp1", "RL"), "recip_deepFac2F": ("GRID.h", 331, "rp1", "RL"),
    "gravitySign": ("GRID.h", 332, "0", "RL"), "rkSign": ("GRID.h", 333, "0", "RL"),
    # GRID.h COMMON /GRID_RS/ (GRID.h:416-493)
    "dxC": ("GRID.h", 432, "xy", "RS"), "dxF": ("GRID.h", 433, "xy", "RS"), "dxG": ("GRID.h", 434, "xy", "RS"),
    "dxV": ("GRID.h", 435, "xy", "RS"), "dyC": ("GRID.h", 436, "xy", "RS"), "dyF": ("GRID.h", 437, "xy", "RS"),
    "dyG": ("GRID.h", 438, "xy", "RS"), "dyU": ("GRID.h", 439, "xy", "RS"),
    "rLowW": ("GRID.h", 440, "xy", "RS"), "rLowS": ("GRID.h", 441, "xy", "RS"),
    "Ro_surf": ("GRID.h", 442, "xy", "RS"), "rSurfW": ("GRID.h", 443, "xy", "RS"),
    "rSurfS": ("GRID.h", 444, "xy", "RS"),
    "recip_dxC": ("GRID.h", 445, "xy", "RS"), "recip_dxF": ("GRID.h", 446, "xy", "RS"),
    "recip_dxG": ("GRID.h", 447, "xy", "RS"), "recip_dxV": ("GRID.h", 448, "xy", "RS"),
    "recip_dyC": ("GRID.h", 449, "xy", "RS"), "recip_dyF": ("GRID.h", 450, "xy", "RS"),
    "recip_dyG": ("GRID.h", 451, "xy", "RS"), "recip_dyU": ("GRID.h", 452, "xy", "RS"),
    "xC": ("GRID.h", 453, "xy", "RS"), "xG": ("GRID.h", 454, "xy", "RS"), "yC": ("GRID.h", 455, "xy", "RS"),
    "yG": ("GRID.h", 456, "xy", "RS"),
    "rA": ("GRID.h", 457, "xy", "RS"), "rAw": ("GRID.h", 458, "xy", "RS"), "rAs": ("GRID.h", 459, "xy", "RS"),
    "rAz": ("GRID.h", 460, "xy", "RS"),
    "recip_rA": ("GRID.h", 461, "xy", "RS"), "recip_rAw": ("GRID.h", 462, "xy", "RS"),
    "recip_rAs": ("GRID.h", 463, "xy", "RS"), "recip_rAz": ("GRID.h", 464, "xy", "RS"),
    "maskInC": ("GRID.h", 465, "xy", "RS"), "maskInW": ("GRID.h", 466, "xy", "RS"),
    "maskInS": ("GRID.h", 467, "xy", "RS"),
    "maskC": ("GRID.h", 468, "xyz", "RS"), "maskW": ("GRID.h", 469, "xyz", "RS"), "maskS": ("GRID.h", 470, "xyz", "RS"),
    "drC": ("GRID.h", 471, "rp1", "RS"), "drF": ("GRID.h", 472, "r", "RS"),
    "recip_drC": ("GRID.h", 473, "rp1", "RS"), "recip_drF": ("GRID.h", 474, "r", "RS"),
    "rC": ("GRID.h", 475, "r", "RS"), "rF": ("GRID.h", 476, "rp1", "RS"),
    "aHybSigmF": ("GRID.h", 477, "rp1", "RS"), "bHybSigmF": ("GRID.h", 478, "rp1", "RS"),
    "aHybSigmC": ("GRID.h", 479, "r", "RS"), "bHybSigmC": ("GRID.h", 480, "r", "RS"),
    "dAHybSigF": ("GRID.h", 481, "r", "RS"), "dBHybSigF": ("GRID.h", 482, "r", "RS"),
    "dBHybSigC": ("GRID.h", 483, "rp1", "RS"), "dAHybSigC": ("GRID.h", 484, "rp1", "RS"),
    "tanPhiAtU": ("GRID.h", 485, "xy", "RS"), "tanPhiAtV": ("GRID.h", 486, "xy", "RS"),
    "angleCosC": ("GRID.h", 487, "xy", "RS"), "angleSinC": ("GRID.h", 488, "xy", "RS"),
    "u2zonDir": ("GRID.h", 489, "xy", "RS"), "v2zonDir": ("GRID.h", 490, "xy", "RS"),
    "fCori": ("GRID.h", 491, "xy", "RS"), "fCoriG": ("GRID.h", 492, "xy", "RS"),
    "fCoriCos": ("GRID.h", 493, "xy", "RS"),
    # GRID.h COMMON /GRID_VAR_RS/ (GRID.h:500-511)
    "hFacC": ("GRID.h", 504, "xyz", "RS"), "hFacW": ("GRID.h", 505, "xyz", "RS"),
    "hFacS": ("GRID.h", 506, "xyz", "RS"),
    "recip_hFacC": ("GRID.h", 507, "xyz", "RS"), "recip_hFacW": ("GRID.h", 508, "xyz", "RS"),
    "recip_hFacS": ("GRID.h", 509, "xyz", "RS"),
    "R_low": ("GRID.h", 510, "xy", "RS"), "recip_Rcol": ("GRID.h", 511, "xy", "RS"),
    # GRID.h COMMON /GRID_I/ (GRID.h:528-534)
    "kSurfC": ("GRID.h", 531, "xy", "INTEGER"), "kSurfW": ("GRID.h", 532, "xy", "INTEGER"),
    "kSurfS": ("GRID.h", 533, "xy", "INTEGER"), "kLowC": ("GRID.h", 534, "xy", "INTEGER"),
    # SURFACE.h arrays the grid routines write
    "Bo_surf": ("SURFACE.h", 24, "xy", "RL"), "recip_Bo": ("SURFACE.h", 25, "xy", "RL"),   # INI_LINEAR_PHISURF
    "topoZ": ("SURFACE.h", 26, "xy", "RS"),                       # INI_DEPTHS
    "h0FacC": ("SURFACE.h", 107, "xyz", "RS"), "h0FacW": ("SURFACE.h", 108, "xyz", "RS"),
    "h0FacS": ("SURFACE.h", 109, "xyz", "RS"),                    # INI_MASKS_ETC (#ifdef NONLIN_FRSURF)
    # ADVECT lane (M2, advect_cs): the W2 tile view of pkg/exch2/W2_EXCH2_TOPOLOGY.h /W2_EXCH2_TOPO_I/ that the
    # cube passes read at myTile = W2_myTileList(bi,bj) (gad_advection.F:254-259, gad_som_advect.F:232-237): one
    # value per tile in tile storage order (kind "t": [tile, 1, 1]); set by the driver's Model on the cube
    "exch2_myFace": ("W2_EXCH2_TOPOLOGY.h", 70, "t", "INTEGER"),
    "exch2_isWedge": ("W2_EXCH2_TOPOLOGY.h", 83, "t", "INTEGER"),
    "exch2_isNedge": ("W2_EXCH2_TOPOLOGY.h", 84, "t", "INTEGER"),
    "exch2_isEedge": ("W2_EXCH2_TOPOLOGY.h", 85, "t", "INTEGER"),
    "exch2_isSedge": ("W2_EXCH2_TOPOLOGY.h", 86, "t", "INTEGER"),
}

W2_TILE_VIEW = ("exch2_myFace", "exch2_isWedge", "exch2_isNedge", "exch2_isEedge", "exch2_isSedge")


def w2_tile_view(grid, w2, size):
    """ADVECT lane: the grid with the W2 tile view (W2_TILE_VIEW) of the topology `w2` (pkg/exch2 W2Common):
    tile t of the storage is W2_myTileList(bi,bj) with bi fastest (as cs_corner_flags orders them)."""
    import numpy as np
    vals = {n: [] for n in W2_TILE_VIEW}
    for bj in range(1, size.nSy + 1):
        for bi in range(1, size.nSx + 1):
            t = w2.W2_myTileList[bi, bj]
            for n in W2_TILE_VIEW:
                vals[n].append(int(getattr(w2, n)[t]))
    return grid.replace(**{n: FArray(jnp.asarray(np.array(v, np.int32)).reshape(-1, 1, 1), n,
                                     **bounds("t", size)) for n, v in vals.items()})


def bounds(kind, size):
    """FArray keyword bounds of a declaration kind (see DECLARATIONS) for an experiment's SIZE.h."""
    sNx, sNy, OLx, OLy, Nr = size.sNx, size.sNy, size.OLx, size.OLy, size.Nr
    return {"xy": dict(i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy)),
            "xyz": dict(i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy), k=(1, Nr)),
            "xyzp1": dict(i=(1-OLx, sNx+OLx), j=(1-OLy, sNy+OLy), k=(1, Nr+1)),   # lane B: FFIELDS.h SWFrac3D
            "y": dict(j=(1-OLy, sNy+OLy)),
            "t": dict(i=(1, 1), j=(1, 1)),          # ADVECT lane: one value per tile (W2 tile view)
            "r": dict(k=(1, Nr), tiled=False),
            "rp1": dict(k=(1, Nr+1), tiled=False)}[kind]


def n_tiles(size):
    """nSx*nSy tiles of a single-process build (nPx = nPy = 1 in every M1 SIZE.h)."""
    if size.nPx != 1 or size.nPy != 1:
        raise NotImplementedError("grid: a multi-process tiling (nPx*nPy > 1) is not ported")
    return size.nSx * size.nSy


def declare(name, size, fill=np.nan):
    """A GRID.h (or SURFACE.h) array with its declared bounds, every point `fill`. The default NaN marks points no
    routine has written yet (a gate then shows them); routines write the initial values the Fortran writes."""
    hdr, line, kind, typ = DECLARATIONS[name]
    if kind == "0":
        raise ValueError(f"{name} ({hdr}:{line}) is a scalar")
    b = bounds(kind, size)
    tiled = b.pop("tiled", True)
    dims = []
    for axis in ("k", "j", "i"):
        if axis in b:
            lo, hi = b[axis]
            dims.append(hi - lo + 1)
    shape = ((n_tiles(size),) if tiled else ()) + tuple(dims)
    dtype = jnp.int32 if typ == "INTEGER" else jnp.float64
    if typ == "INTEGER" and isinstance(fill, float) and np.isnan(fill):
        fill = -999999                     # integer arrays: an impossible index marks unwritten points
    return FArray(jnp.full(shape, fill, dtype=dtype), name, **b, tiled=tiled)


def local(name, size, fill=np.nan, **b):
    """A routine-local array with explicit Fortran bounds (i=(lo,hi), j=..., k=...; tiled unless tiled=False)."""
    tiled = b.pop("tiled", True)
    dims = []
    for axis in ("k", "j", "i"):
        if axis in b:
            lo, hi = b[axis]
            dims.append(hi - lo + 1)
    shape = ((n_tiles(size),) if tiled else ()) + tuple(dims)
    return FArray(jnp.full(shape, fill, dtype=jnp.float64), name, **b, tiled=tiled)


@jax.tree_util.register_pytree_node_class
class Grid:
    """The GRID.h common blocks (and the SURFACE.h arrays the grid routines write) by Fortran name: `grid.hFacC`.

    Immutable; `grid.replace(name=value, ...)` returns a new Grid. Scalars (rkSign, gravitySign) are float64 jax
    scalars, arrays are FArrays."""

    __slots__ = ("_f",)

    def __init__(self, fields=None):
        object.__setattr__(self, "_f", dict(fields or {}))

    def __getattr__(self, name):
        f = object.__getattribute__(self, "_f")
        if name in f:
            return f[name]
        if name in DECLARATIONS:
            hdr, line, _, _ = DECLARATIONS[name]
            raise AttributeError(f"{name} ({hdr}:{line}) has not been set by any grid routine yet")
        raise AttributeError(name)

    def __setattr__(self, name, value):
        raise AttributeError("Grid is immutable; use grid.replace(...)")

    def __contains__(self, name):
        return name in self._f

    def names(self):
        return sorted(self._f)

    def replace(self, **fields):
        for k in fields:
            if k not in DECLARATIONS:
                raise KeyError(f"{k} is not a declared grid field (mitjax/model/grid.py DECLARATIONS)")
        return Grid({**self._f, **fields})

    def tree_flatten(self):
        keys = tuple(sorted(self._f))
        return tuple(self._f[k] for k in keys), keys

    @classmethod
    def tree_unflatten(cls, keys, leaves):
        return cls(dict(zip(keys, leaves)))

    def __repr__(self):
        return f"Grid({', '.join(self.names())})"
