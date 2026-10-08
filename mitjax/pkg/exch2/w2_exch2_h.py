"""W2_EXCH2_SIZE.h, W2_EXCH2_TOPOLOGY.h and W2_EXCH2_PARAMS.h (pkg/exch2 @63cdc0b): the W2 topology common blocks,
filled on the host by the W2 set-up (W2_EEBOOT, mitjax/pkg/exch2/w2_eeboot.py).

    w2 = W2Common.declare(cfg.size)       # the common blocks, every element as COMMON storage starts (0, 0., ' ')
    w2.exch2_tBasex[tN]                   # exch2_tBasex(tN), Fortran index
    w2.exch2_pij[k, n, tN] = 1            # exch2_pij(k,n,tN) = 1

Integer set-up code, run once before the model (never traced): the arrays are host-side Fortran-index arrays
(`FIntArray`, numpy storage, Fortran declaration bounds, bounds-checked reads and in-place writes as in a COMMON
block), the arithmetic is Python integer arithmetic with the Fortran semantics made explicit where they differ from
Python's (`idiv`: INTEGER division truncates toward zero; `imod`: MOD takes the sign of the dividend; `nint`: NINT of a
REAL*4 rounds half away from zero; REAL*4 products stay REAL*4, `r4`).

Initial values: the COMMON blocks are static storage that gfortran zero-initialises (no DATA statement initialises
them); W2_EEBOOT (w2_eeboot.F:44-74), W2_READPARMS, W2_E2SETUP and W2_MAP_PROCS then set what they set.
"""

from dataclasses import dataclass, fields

import numpy as np

# W2_EXCH2_TOPOLOGY.h:21-25  directions
W2_NORTH = 1                # PARAMETER ( W2_NORTH = 1 )
W2_SOUTH = 2                # PARAMETER ( W2_SOUTH = 2 )
W2_EAST = 3                 # PARAMETER ( W2_EAST  = 3 )
W2_WEST = 4                 # PARAMETER ( W2_WEST  = 4 )

# W2_EXCH2_SIZE.h:307-308 (the build uses pkg/exch2's own header: no experiment override, see sizes())
W2_maxNbFacets = 10         # PARAMETER ( W2_maxNbFacets = 10 )
W2_maxNeighbours = 8        # PARAMETER ( W2_maxNeighbours = 8 )


# ---------------------------------------------------------------------------------------------------------------
# Fortran integer and REAL*4 semantics

def idiv(a, b):
    """INTEGER a/b: the quotient truncated toward zero (F2008 7.1.5.2.1), not Python's floor."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def imod(a, b):
    """MOD(a, b) for INTEGERs: a - INT(a/b)*b, the sign of a."""
    return a - idiv(a, b) * b


def r4(x):
    """A REAL*4 value (a literal without a D exponent, or the result of a REAL*4 operation)."""
    return np.float32(x)


def nint(x):
    """NINT of a REAL: the nearest integer, halves away from zero."""
    x = float(x)
    n = int(abs(x) + 0.5)
    return n if x >= 0 else -n


def int_(x):
    """INT of a REAL: truncation toward zero."""
    return int(float(x))


# ---------------------------------------------------------------------------------------------------------------
# host-side Fortran-index arrays

class FIntArray:
    """A host-side array with its Fortran declaration (`FIntArray("exch2_pij", 4, 8, 72)` is
    `INTEGER exch2_pij(4,8,72)`; a bound is `hi` (lower bound 1) or `(lo, hi)`). `a[i, j]` reads a(i,j) and
    `a[i, j] = v` writes it, with the indices checked against the declaration. Storage: numpy, index order as
    declared."""

    def __init__(self, name, *bounds, dtype=np.int64, fill=0):
        self.name = name
        self.bounds = tuple((1, b) if isinstance(b, int) else tuple(b) for b in bounds)
        shape = tuple(hi - lo + 1 for lo, hi in self.bounds)
        self.data = np.full(shape, fill, dtype=dtype)

    def _sidx(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        if len(idx) != len(self.bounds):
            raise IndexError(f"{self.name}{idx}: {len(idx)} indices for {len(self.bounds)} dimensions")
        out = []
        for n, (x, (lo, hi)) in enumerate(zip(idx, self.bounds), start=1):
            if isinstance(x, bool) or not isinstance(x, (int, np.integer)):
                raise TypeError(f"{self.name}: index {n} must be an integer, got {x!r}")
            if not lo <= x <= hi:
                raise IndexError(f"{self.name}({', '.join(map(str, idx))}): index {n} = {x} is outside the declared "
                                 f"{lo}:{hi}")
            out.append(int(x) - lo)
        return tuple(out)

    def __getitem__(self, idx):
        v = self.data[self._sidx(idx)]
        if self.data.dtype.kind in "iu":
            return int(v)
        return v

    def __setitem__(self, idx, value):
        self.data[self._sidx(idx)] = value

    def values(self, lo=None, hi=None):
        """a(lo:hi) of a 1-D array as a tuple of Python values."""
        if len(self.bounds) != 1:
            raise ValueError(f"{self.name}: values() of a {len(self.bounds)}-D array")
        lo = self.bounds[0][0] if lo is None else lo
        hi = self.bounds[0][1] if hi is None else hi
        return tuple(self[i] for i in range(lo, hi + 1))


# ---------------------------------------------------------------------------------------------------------------
# the common blocks

def sizes(size):
    """W2_EXCH2_SIZE.h:309-314 from SIZE.h: W2_maxNbTiles = nSx*nSy*nPx*nPy * 2 (and the stack sizes)."""
    W2_maxNbTiles = size.nSx * size.nSy * size.nPx * size.nPy * 2        # W2_EXCH2_SIZE.h:309
    return {"W2_maxNbTiles": W2_maxNbTiles,
            "W2_ioBufferSize": W2_maxNbTiles * size.sNx * size.sNy,        # :310
            "W2_maxXStackNx": W2_maxNbTiles * size.sNx,                    # :311
            "W2_maxXStackNy": W2_maxNbTiles * size.sNy,                    # :312
            "W2_maxYStackNx": W2_maxNbTiles * size.sNx,                    # :313
            "W2_maxYStackNy": W2_maxNbTiles * size.sNy}                    # :314


@dataclass
class W2Common:
    """/W2_EXCH2_TOPO_I/, /W2_EXCH2_HALO_SPEC/, /W2_CUMSUM_TOPO_I/, /W2_MAP_TILE2PROC/, /W2_EXCH2_COMMFLAG/,
    /EXCH2_FILLVAL_R*/ (W2_EXCH2_TOPOLOGY.h:63-184) and /W2_EXCH2_PARM_I/, /W2_EXCH2_PARM_L/, /W2_EXCH2_BUILD_I/,
    /W2_EXCH2_PARM_R/ (W2_EXCH2_PARAMS.h:224-270). Scalars are Python values, arrays FIntArray. `size`: SIZE.h."""
    size: object
    W2_maxNbTiles: int
    # W2_EXCH2_TOPOLOGY.h:63-93 /W2_EXCH2_TOPO_I/
    exch2_global_Nx: int
    exch2_global_Ny: int
    exch2_xStack_Nx: int
    exch2_xStack_Ny: int
    exch2_yStack_Nx: int
    exch2_yStack_Ny: int
    exch2_nTiles: int
    exch2_myFace: FIntArray
    exch2_mydNx: FIntArray
    exch2_mydNy: FIntArray
    exch2_tNx: FIntArray
    exch2_tNy: FIntArray
    exch2_tBasex: FIntArray
    exch2_tBasey: FIntArray
    exch2_txGlobalo: FIntArray
    exch2_tyGlobalo: FIntArray
    exch2_txXStackLo: FIntArray
    exch2_tyXStackLo: FIntArray
    exch2_txYStackLo: FIntArray
    exch2_tyYStackLo: FIntArray
    exch2_isWedge: FIntArray
    exch2_isNedge: FIntArray
    exch2_isEedge: FIntArray
    exch2_isSedge: FIntArray
    exch2_nNeighbours: FIntArray
    exch2_neighbourId: FIntArray
    exch2_opposingSend: FIntArray
    exch2_neighbourDir: FIntArray
    exch2_pij: FIntArray
    exch2_oi: FIntArray
    exch2_oj: FIntArray
    # :118-124 /W2_EXCH2_HALO_SPEC/
    exch2_iLo: FIntArray
    exch2_iHi: FIntArray
    exch2_jLo: FIntArray
    exch2_jHi: FIntArray
    # :136-140 /W2_CUMSUM_TOPO_I/ (W2_CUMSUM_USE_MATRIX's /W2_CUMSUM_MATRIX/ :141-145 is not in this build)
    W2_tMC1: int
    W2_tMC2: int
    W2_cumSum_facet: FIntArray
    # :159-168 /W2_MAP_TILE2PROC/
    W2_tileProc: FIntArray
    W2_tileIndex: FIntArray
    W2_myTileList: FIntArray
    W2_procTileList: FIntArray
    # :171-172 /W2_EXCH2_COMMFLAG/
    W2_myCommFlag: FIntArray
    # :177-184 /EXCH2_FILLVAL_R*/
    e2FillValue_RL: float
    e2FillValue_RS: float
    e2FillValue_R4: np.float32
    e2FillValue_R8: float
    # W2_EXCH2_PARAMS.h:224-236 /W2_EXCH2_PARM_I/
    preDefTopol: int
    nFacets: int
    facet_dims: FIntArray
    nBlankTiles: int
    blankList: FIntArray
    W2_mapIO: int
    W2_oUnit: object
    W2_printMsg: int
    # :240-242 /W2_EXCH2_PARM_L/
    W2_useE2ioLayOut: bool
    # :255-261 /W2_EXCH2_BUILD_I/
    facet_owns: FIntArray
    facet_pij: FIntArray
    facet_oi: FIntArray
    facet_oj: FIntArray
    # :269-270 /W2_EXCH2_PARM_R/  Real*4 facet_link(4, W2_maxNbFacets)
    facet_link: FIntArray
    # W2_EXCH2_TOPOLOGY.h:141-145 /W2_CUMSUM_MATRIX/ W2_cumSum_tiles(2,W2_maxNbTiles,W2_maxNbTiles): only in a build
    # with W2_CUMSUM_USE_MATRIX (adjustment.cs-32x32x1's W2_OPTIONS.h); None otherwise
    W2_cumSum_tiles: object = None

    @classmethod
    def declare(cls, size, cumsum_matrix=False):
        """Every common-block variable as declared, holding what zero-initialised static storage holds
        (cumsum_matrix: the build defines W2_CUMSUM_USE_MATRIX, which adds /W2_CUMSUM_MATRIX/)."""
        mT = sizes(size)["W2_maxNbTiles"]
        mN, mF = W2_maxNeighbours, W2_maxNbFacets
        nPxy = size.nPx * size.nPy

        def per_tile(name):
            return FIntArray(name, mT)

        def per_nb(name):
            return FIntArray(name, mN, mT)

        return cls(
            size=size, W2_maxNbTiles=mT,
            exch2_global_Nx=0, exch2_global_Ny=0, exch2_xStack_Nx=0, exch2_xStack_Ny=0,
            exch2_yStack_Nx=0, exch2_yStack_Ny=0, exch2_nTiles=0,
            exch2_myFace=per_tile("exch2_myFace"), exch2_mydNx=per_tile("exch2_mydNx"),
            exch2_mydNy=per_tile("exch2_mydNy"), exch2_tNx=per_tile("exch2_tNx"), exch2_tNy=per_tile("exch2_tNy"),
            exch2_tBasex=per_tile("exch2_tBasex"), exch2_tBasey=per_tile("exch2_tBasey"),
            exch2_txGlobalo=per_tile("exch2_txGlobalo"), exch2_tyGlobalo=per_tile("exch2_tyGlobalo"),
            exch2_txXStackLo=per_tile("exch2_txXStackLo"), exch2_tyXStackLo=per_tile("exch2_tyXStackLo"),
            exch2_txYStackLo=per_tile("exch2_txYStackLo"), exch2_tyYStackLo=per_tile("exch2_tyYStackLo"),
            exch2_isWedge=per_tile("exch2_isWedge"), exch2_isNedge=per_tile("exch2_isNedge"),
            exch2_isEedge=per_tile("exch2_isEedge"), exch2_isSedge=per_tile("exch2_isSedge"),
            exch2_nNeighbours=per_tile("exch2_nNeighbours"),
            exch2_neighbourId=per_nb("exch2_neighbourId"), exch2_opposingSend=per_nb("exch2_opposingSend"),
            exch2_neighbourDir=per_nb("exch2_neighbourDir"), exch2_pij=FIntArray("exch2_pij", 4, mN, mT),
            exch2_oi=per_nb("exch2_oi"), exch2_oj=per_nb("exch2_oj"),
            exch2_iLo=per_nb("exch2_iLo"), exch2_iHi=per_nb("exch2_iHi"),
            exch2_jLo=per_nb("exch2_jLo"), exch2_jHi=per_nb("exch2_jHi"),
            W2_tMC1=0, W2_tMC2=0, W2_cumSum_facet=FIntArray("W2_cumSum_facet", 2, mF, mF),
            W2_tileProc=per_tile("W2_tileProc"), W2_tileIndex=per_tile("W2_tileIndex"),
            W2_myTileList=FIntArray("W2_myTileList", size.nSx, size.nSy),
            W2_procTileList=FIntArray("W2_procTileList", size.nSx, size.nSy, nPxy),
            W2_myCommFlag=FIntArray("W2_myCommFlag", mN, size.nSx, size.nSy, dtype="<U1", fill=" "),
            e2FillValue_RL=0.0, e2FillValue_RS=0.0, e2FillValue_R4=np.float32(0.0), e2FillValue_R8=0.0,
            preDefTopol=0, nFacets=0, facet_dims=FIntArray("facet_dims", 2 * mF), nBlankTiles=0,
            blankList=per_tile("blankList"), W2_mapIO=0, W2_oUnit=0, W2_printMsg=0, W2_useE2ioLayOut=False,
            facet_owns=FIntArray("facet_owns", 2, mF), facet_pij=FIntArray("facet_pij", 4, 4, mF),
            facet_oi=FIntArray("facet_oi", 4, mF), facet_oj=FIntArray("facet_oj", 4, mF),
            facet_link=FIntArray("facet_link", 4, mF, dtype=np.float32, fill=0.0),
            W2_cumSum_tiles=FIntArray("W2_cumSum_tiles", 2, mT, mT) if cumsum_matrix else None,
        )

    def field_names(self):
        return [f.name for f in fields(self) if f.name not in ("size",)]
