"""Gather tables of the exch2 exchanges on any W2 topology, cubed sphere included, replayed on the host from the
Fortran loops (pkg/exch2 @63cdc0b): EXCH2_RX1_CUBE (scalar passes) and the cube-specific statements of every
exchange routine, executed literally on symbolic arrays.

    @63cdc0b pkg/exch2/exch2_rx1_cube.template:84-262, exch2_put_rx1.template, exch2_get_rx1.template,
             exch2_3d_rx.template:49-120, exch2_s3d_rx.template:44-56, exch2_sm_3d_rx.template:58-181,
             exch2_z_3d_rx.template:56-289, exch2_uv_agrid_3d_rx.template:59-208,
             exch2_uv_bgrid_3d_rx.template:74-412, exch2_uv_3d_rx.template:61-229,
             exch2_uv_dgrid_3d_rx.template:58-198

Symbolic replay: every point of every array holds (comp, src, sign, neg), meaning "the value of input array `comp`
(1 = u-like / scalar, 2 = v-like) at flat point `src`, multiplied by `sign` (the product of the `negOne` factors it
went through) and negated `neg` times (mod 2)". A Fortran copy statement copies the tuple, `x*negOne` multiplies
sign, `-x` flips neg. EXCH2_RX1_CUBE is replayed as the Fortran runs it: every tile PUTs its neighbours' buffers
from the array as it is before the pass (exch2_put_rx1.template: source (isl, jsl) of target (itl, jtl) through the
sender's pij/oi/oj), then every tile GETs them in tile and neighbour order (exch2_get_rx1.template), over the ranges
of EXCH2_GET_SCAL_BOUNDS. The result of a routine made of copies, `*negOne` and `-x` is one gather per output
array, `ExchangeMaps`' format (src, comp, sign) plus `neg`; it is exact: copies compose, a product of factors +-1
is +-1, and on x86-64 a multiplication by -1 or +1 leaves a NaN's sign and payload alone while negation flips the
sign bit, so neither the order of the factors nor their grouping changes a bit of the result.

The C-grid vector exchanges (EXCH2_UV_3D_RX and EXCH2_UV_DGRID_3D_RX) are not copies: their two EXCH2_RX2_CUBE
passes compute sa1*A1 + sa2*A2 (mitjax/pkg/exch2/exch2_rx2_cube.py); the cube statements that follow them (the
corner fixes of exch2_uv_3d_rx.template:79-227, the D-grid sign changes of exch2_uv_dgrid_3d_rx.template:91-196) are
replayed here as one post-pass gather on the outputs of the passes.

W2_FILL_NULL_REGIONS and W2_USE_R1_ONLY must be undefined (pkg/exch2/W2_OPTIONS.h and every M2 experiment's);
`cube_exchange_programs` checks it. Single process (W2_myCommFlag 'P').
"""

import numpy as np

from mitjax.pkg.exch2.exch2_get_scal_bounds import exch2_get_scal_bounds
from mitjax.pkg.exch2.w2_exch2_h import imod


def _range(lo, hi, stride):
    """DO x = lo, hi, stride."""
    return range(lo, hi + stride, stride)


class Sym:
    """Symbolic arrays 1..ncomp over the flattened [tile, j, i] points of one level of `layout`."""

    def __init__(self, layout, ncomp=2):
        self.L = layout
        n = layout.npoints
        self.comp = {c: np.full(n, c, np.int64) for c in range(1, ncomp + 1)}
        self.src = {c: np.arange(n, dtype=np.int64) for c in range(1, ncomp + 1)}
        self.sign = {c: np.ones(n, np.int64) for c in range(1, ncomp + 1)}
        self.neg = {c: np.zeros(n, bool) for c in range(1, ncomp + 1)}

    def p(self, tile, i, j):
        L = self.L
        if not (1 - L.OLx <= i <= L.sNx + L.OLx and 1 - L.OLy <= j <= L.sNy + L.OLy):
            raise IndexError(f"({i},{j}) outside the array (tile {tile})")
        return ((tile - 1) * L.ny + (j - 1 + L.OLy)) * L.nx + (i - 1 + L.OLx)

    def get(self, c, tile, i, j):
        q = self.p(tile, i, j)
        return (self.comp[c][q], self.src[c][q], self.sign[c][q], self.neg[c][q])

    def put(self, c, tile, i, j, val, mul=1, negate=False):
        q = self.p(tile, i, j)
        a, s, g, n = val
        self.comp[c][q], self.src[c][q], self.sign[c][q], self.neg[c][q] = a, s, g * mul, bool(n) ^ negate

    def copy(self, c, tile, i, j, d, i2, j2, mul=1, negate=False):
        """array_c(i,j) = [-] array_d(i2,j2) [* negOne] on tile `tile`."""
        self.put(c, tile, i, j, self.get(d, tile, i2, j2), mul, negate)

    def snapshot(self, c):
        return (self.comp[c].copy(), self.src[c].copy(), self.sign[c].copy(), self.neg[c].copy())

    def table(self, c, own_comp):
        """(src int32, comp int8 (0 = keep), sign int8, neg bool) of array c (own_comp: its own input id)."""
        n = self.L.npoints
        keep = ((self.comp[c] == own_comp) & (self.src[c] == np.arange(n)) & (self.sign[c] == 1) & ~self.neg[c])
        return (self.src[c].astype(np.int32), np.where(keep, 0, self.comp[c]).astype(np.int8),
                np.where(keep, 1, self.sign[c]).astype(np.int8), np.where(keep, False, self.neg[c]))


def rx1_cube(sym, c, w2, updateCorners, myOLw=None, myOLe=None, myOLs=None, myOLn=None, exchWidthX=None):
    """EXCH2_RX1_CUBE( array_c, .FALSE., 'T ', myOLw, myOLe, myOLs, myOLn, 1, exchWidthX, exchWidthY, cornerMode )
    on the symbolic array c (exch2_rx1_cube.template:84-262; signOption is not read by the RX1 path)."""
    L, sz = sym.L, w2.size
    myOLw = L.OLx if myOLw is None else myOLw
    myOLe = L.OLx if myOLe is None else myOLe
    myOLs = L.OLy if myOLs is None else myOLs
    myOLn = L.OLy if myOLn is None else myOLn
    exchWidthX = L.OLx if exchWidthX is None else exchWidthX
    i1Lo, i1Hi, j1Lo, j1Hi = 1 - myOLw, sz.sNx + myOLe, 1 - myOLs, sz.sNy + myOLn      # :86-89
    old = sym.snapshot(c)
    bufs = {}
    # -- Post sends into buffer (:99-132; exch2_put_rx1.template)
    for bj in range(1, sz.nSy + 1):
        for bi in range(1, sz.nSx + 1):
            thisTile = w2.W2_myTileList[bi, bj]
            for N in range(1, w2.exch2_nNeighbours[thisTile] + 1):
                farTile = w2.exch2_neighbourId[N, thisTile]
                oN = w2.exch2_opposingSend[N, thisTile]
                tIlo, tIhi, tJlo, tJhi, tiStride, tjStride = exch2_get_scal_bounds(
                    "T ", exchWidthX, updateCorners, farTile, oN, w2=w2)
                tgT = w2.exch2_neighbourId[N, thisTile]                         # put_rx1: tgT
                itb, jtb = w2.exch2_tBasex[tgT], w2.exch2_tBasey[tgT]
                isb, jsb = w2.exch2_tBasex[thisTile], w2.exch2_tBasey[thisTile]
                pi = (w2.exch2_pij[1, N, thisTile], w2.exch2_pij[2, N, thisTile])
                pj = (w2.exch2_pij[3, N, thisTile], w2.exch2_pij[4, N, thisTile])
                oi, oj = w2.exch2_oi[N, thisTile], w2.exch2_oj[N, thisTile]
                buf = []
                for jtl in _range(tJlo, tJhi, tjStride):
                    for itl in _range(tIlo, tIhi, tiStride):
                        itc, jtc = itl + itb, jtl + jtb
                        isc = pi[0] * itc + pi[1] * jtc + oi
                        jsc = pj[0] * itc + pj[1] * jtc + oj
                        isl, jsl = isc - isb, jsc - jsb
                        if not (i1Lo <= isl <= i1Hi and j1Lo <= jsl <= j1Hi):
                            raise RuntimeError(f"EXCH2_PUT_RX1: source ({isl},{jsl}) out of bounds (tile {thisTile}, "
                                               f"neighbour {N})")
                        q = sym.p(thisTile, isl, jsl)
                        buf.append(tuple(a[q] for a in old))                     # e2Bufr1_RX(iLoc) = array(isl,jsl)
                bufs[(thisTile, N)] = buf
    # -- Extract from buffer (:227-260; exch2_get_rx1.template, commSetting 'P')
    for bj in range(1, sz.nSy + 1):
        for bi in range(1, sz.nSx + 1):
            thisTile = w2.W2_myTileList[bi, bj]
            for N in range(1, w2.exch2_nNeighbours[thisTile] + 1):
                tIlo, tIhi, tJlo, tJhi, tiStride, tjStride = exch2_get_scal_bounds(
                    "T ", exchWidthX, updateCorners, thisTile, N, w2=w2)
                if w2.W2_myCommFlag[N, bi, bj] != "P":
                    raise NotImplementedError("EXCH2_GET_RX1: commSetting other than 'P' is not ported")
                soT = w2.exch2_neighbourId[N, thisTile]
                oNb = w2.exch2_opposingSend[N, thisTile]
                buf = bufs[(soT, oNb)]
                k = 0
                for jtl in _range(tJlo, tJhi, tjStride):
                    for itl in _range(tIlo, tIhi, tiStride):
                        sym.put(c, thisTile, itl, jtl, buf[k])
                        k += 1
                if k != len(buf):
                    raise RuntimeError("EXCH2_GET_RX1: buffer length differs from the PUT's")


def _tiles(w2):
    sz = w2.size
    for bj in range(1, sz.nSy + 1):
        for bi in range(1, sz.nSx + 1):
            yield w2.W2_myTileList[bi, bj]


# ---------------------------------------------------------------------------------------------------------------------
# the routines (one level: myNz = 1, so the k loops run once)

def exch2_3d(sym, c, w2):
    """EXCH2_3D_RX (exch2_3d_rx.template:49-65; :67-118 only under W2_FILL_NULL_REGIONS)."""
    rx1_cube(sym, c, w2, False)                                               # :56-59 EXCH_IGNORE_CORNERS
    rx1_cube(sym, c, w2, True)                                                # :61-64 EXCH_UPDATE_CORNERS


def exch2_s3d(sym, c, w2):
    """EXCH2_S3D_RX on (0:sNx+1, 0:sNy+1) arrays held inside the full array (exch2_s3d_rx.template:44-54): width 1,
    corners ignored."""
    rx1_cube(sym, c, w2, False, 1, 1, 1, 1, 1)


def exch2_sm_3d(sym, c, w2, withSigns, useCubedSphereExchange):
    """EXCH2_SM_3D_RX (exch2_sm_3d_rx.template:58-179)."""
    sz = w2.size
    negOne = -1 if withSigns else 1                                           # :61-62
    exchWidthX, exchWidthY = sz.OLx, sz.OLy
    rx1_cube(sym, c, w2, False)                                               # :68-71
    rx1_cube(sym, c, w2, True)                                                # :72-75
    if useCubedSphereExchange and withSigns:                                  # :80
        for t in _tiles(w2):
            myFace = w2.exch2_myFace[t]
            if imod(myFace, 2) == 1:
                if w2.exch2_isNedge[t] == 1:
                    for j in range(1, exchWidthY + 1):
                        for i in range(1 - sz.OLx, sz.sNx + sz.OLx + 1):
                            sym.copy(c, t, i, sz.sNy + j, c, i, sz.sNy + j, mul=negOne)
                if w2.exch2_isWedge[t] == 1:
                    for j in range(1 - sz.OLy, sz.sNy + sz.OLy + 1):
                        for i in range(1, exchWidthX + 1):
                            sym.copy(c, t, 1 - i, j, c, 1 - i, j, mul=negOne)
            else:
                if w2.exch2_isEedge[t] == 1:
                    for j in range(1 - sz.OLy, sz.sNy + sz.OLy + 1):
                        for i in range(1, exchWidthX + 1):
                            sym.copy(c, t, sz.sNx + i, j, c, sz.sNx + i, j, mul=negOne)
                if w2.exch2_isSedge[t] == 1:
                    for j in range(1, exchWidthY + 1):
                        for i in range(1 - sz.OLx, sz.sNx + sz.OLx + 1):
                            sym.copy(c, t, i, 1 - j, c, i, 1 - j, mul=negOne)


def exch2_z_3d(sym, c, w2, useCubedSphereExchange):
    """EXCH2_Z_3D_RX (exch2_z_3d_rx.template:56-289)."""
    sz = w2.size
    sNx, sNy, OLx, OLy = sz.sNx, sz.sNy, sz.OLx, sz.OLy
    saved = {}
    if useCubedSphereExchange:                                                # :63-84
        for t in _tiles(w2):
            saved[t] = (sym.get(c, t, 1, sNy + 1), sym.get(c, t, sNx + 1, 1))   # phiNW, phiSE (:79-80)
    rx1_cube(sym, c, w2, False)                                               # :86-89
    rx1_cube(sym, c, w2, True)                                                # :90-93
    if not useCubedSphereExchange:                                            # :95
        return
    for t in _tiles(w2):
        phiNW, phiSE = saved[t]
        mFace = w2.exch2_myFace[t]
        E, W, N, S = (w2.exch2_isEedge[t] == 1, w2.exch2_isWedge[t] == 1, w2.exch2_isNedge[t] == 1,
                      w2.exch2_isSedge[t] == 1)
        if imod(mFace, 2) == 0:                                               # :103
            if E:                                                             # :106
                for j in range(sNy + OLy, 2 - OLy - 1, -1):                   # :108
                    for i in range(sNx + 1, sNx + OLx + 1):
                        sym.copy(c, t, i, j, c, i, j - 1)                     # :110
                if N:                                                         # :115
                    for j in range(sNy + 2, sNy + OLy + 1):
                        i = sNx - sNy + j                                     # :118
                        sym.copy(c, t, sNx + 1, j, c, i, sNy + 1)             # :119
            if S:                                                             # :132
                for j in range(1 - OLy, 0 + 1):
                    for i in range(sNx + OLx, 2 - OLx - 1, -1):               # :135
                        sym.copy(c, t, i, j, c, i - 1, j)                     # :136
                if E:                                                         # :141
                    sym.put(c, t, sNx + 1, 1, phiSE)                          # :143
                    for i in range(sNx + 2, sNx + OLx + 1):
                        j = sNx + 2 - i                                       # :145
                        sym.copy(c, t, i, 1, c, sNx + 1, j)                   # :146
                if W:                                                         # :158
                    for j in range(1 - OLy, 0 + 1):
                        sym.copy(c, t, 1, j, c, j, 1)                         # :161
            if W and N:                                                       # :172-173
                for i in range(2 - OLx, 0 + 1):
                    j = sNy + 2 - i                                           # :176
                    sym.copy(c, t, i, sNy + 1, c, 1, j)                       # :177
        else:
            if N:                                                             # :194
                for j in range(sNy + 1, sNy + OLy + 1):
                    for i in range(sNx + OLx, 2 - OLx - 1, -1):               # :197
                        sym.copy(c, t, i, j, c, i - 1, j)                     # :198
                if E:                                                         # :203
                    for i in range(sNx + 2, sNx + OLx + 1):
                        j = sNy - sNx + i                                     # :206
                        sym.copy(c, t, i, sNy + 1, c, sNx + 1, j)             # :207
            if W:                                                             # :220
                for j in range(sNy + OLy, 2 - OLy - 1, -1):                   # :222
                    for i in range(1 - OLx, 0 + 1):
                        sym.copy(c, t, i, j, c, i, j - 1)                     # :224
                if N:                                                         # :229
                    sym.put(c, t, 1, sNy + 1, phiNW)                          # :231
                    for j in range(sNy + 2, sNy + OLy + 1):
                        i = sNy + 2 - j                                       # :233
                        sym.copy(c, t, 1, j, c, i, sNy + 1)                   # :234
                if S:                                                         # :246
                    for i in range(1 - OLx, 0 + 1):
                        sym.copy(c, t, i, 1, c, 1, i)                         # :249
            if E and S:                                                       # :262-263
                for j in range(2 - OLy, 0 + 1):
                    i = sNx + 2 - j                                           # :266
                    sym.copy(c, t, sNx + 1, j, c, i, 1)                       # :267


def exch2_uv_agrid_3d(sym, w2, withSigns, useCubedSphereExchange, U=1, V=2):
    """EXCH2_UV_AGRID_3D_RX (exch2_uv_agrid_3d_rx.template:59-206)."""
    sz = w2.size
    sNx, sNy, OLx, OLy = sz.sNx, sz.sNy, sz.OLx, sz.OLy
    exchWidthX, exchWidthY = OLx, OLy
    negOne = -1 if withSigns else 1                                           # :68-69
    rx1_cube(sym, U, w2, False)                                               # :72-75
    rx1_cube(sym, U, w2, True)                                                # :76-79
    rx1_cube(sym, V, w2, False)                                               # :81-84
    rx1_cube(sym, V, w2, True)                                                # :85-88
    if not useCubedSphereExchange:                                            # :94
        return
    for t in _tiles(w2):
        myFace = w2.exch2_myFace[t]
        loc = {(cc, i, j): sym.get(cc, t, i, j) for cc in (U, V)              # uLoc, vLoc (:102-107)
               for j in range(1 - OLy, sNy + OLy + 1) for i in range(1 - OLx, sNx + OLx + 1)}
        if imod(myFace, 2) == 1:
            if w2.exch2_isNedge[t] == 1:
                for j in range(1, exchWidthY + 1):
                    for i in range(1 - OLx, sNx + OLx + 1):
                        sym.put(U, t, i, sNy + j, loc[(V, i, sNy + j)], mul=negOne)
                        sym.put(V, t, i, sNy + j, loc[(U, i, sNy + j)])
            if w2.exch2_isWedge[t] == 1:
                for j in range(1 - OLy, sNy + OLy + 1):
                    for i in range(1, exchWidthX + 1):
                        sym.put(U, t, 1 - i, j, loc[(V, 1 - i, j)])
                        sym.put(V, t, 1 - i, j, loc[(U, 1 - i, j)], mul=negOne)
        else:
            if w2.exch2_isEedge[t] == 1:
                for j in range(1 - OLy, sNy + OLy + 1):
                    for i in range(1, exchWidthX + 1):
                        sym.put(U, t, sNx + i, j, loc[(V, sNx + i, j)])
                        sym.put(V, t, sNx + i, j, loc[(U, sNx + i, j)], mul=negOne)
            if w2.exch2_isSedge[t] == 1:
                for j in range(1, exchWidthY + 1):
                    for i in range(1 - OLx, sNx + OLx + 1):
                        sym.put(U, t, i, 1 - j, loc[(V, i, 1 - j)], mul=negOne)
                        sym.put(V, t, i, 1 - j, loc[(U, i, 1 - j)])


def exch2_uv_bgrid_3d(sym, w2, withSigns, useCubedSphereExchange, U=1, V=2):
    """EXCH2_UV_BGRID_3D_RX (exch2_uv_bgrid_3d_rx.template:74-410; the W2_FILL_NULL_REGIONS part not compiled)."""
    sz = w2.size
    sNx, sNy, OLx, OLy = sz.sNx, sz.sNy, sz.OLx, sz.OLy
    exchWidthX, exchWidthY = OLx, OLy
    negOne = -1 if withSigns else 1                                           # :80-81
    saved = {}
    if useCubedSphereExchange:                                                # :83-110
        for t in _tiles(w2):
            saved[t] = {"uNW": sym.get(U, t, 1, sNy + 1), "vNW": sym.get(V, t, 1, sNy + 1),
                        "uSE": sym.get(U, t, sNx + 1, 1), "vSE": sym.get(V, t, sNx + 1, 1)}
    rx1_cube(sym, U, w2, False)                                               # :112-115
    rx1_cube(sym, U, w2, True)                                                # :116-119
    rx1_cube(sym, V, w2, False)                                               # :121-124
    rx1_cube(sym, V, w2, True)                                                # :125-128
    if not useCubedSphereExchange:                                            # :134
        return
    for t in _tiles(w2):
        myFace = w2.exch2_myFace[t]
        E, W, N, S = (w2.exch2_isEedge[t] == 1, w2.exch2_isWedge[t] == 1, w2.exch2_isNedge[t] == 1,
                      w2.exch2_isSedge[t] == 1)
        loc = {(cc, i, j): sym.get(cc, t, i, j) for cc in (U, V)              # uLoc, vLoc
               for j in range(1 - OLy, sNy + OLy + 1) for i in range(1 - OLx, sNx + OLx + 1)}
        odd = imod(myFace, 2) == 1
        if odd:
            if N:
                for j in range(1, exchWidthY + 1):
                    for i in range(1 - OLx, sNx + OLx - 1 + 1):
                        sym.put(U, t, i + 1, sNy + j, loc[(V, i, sNy + j)], mul=negOne)
                        sym.put(V, t, i + 1, sNy + j, loc[(U, i, sNy + j)])
            if W:
                for j in range(1 - OLy, sNy + OLy - 1 + 1):
                    for i in range(1, exchWidthX + 1):
                        sym.put(U, t, 1 - i, j + 1, loc[(V, 1 - i, j)])
                        sym.put(V, t, 1 - i, j + 1, loc[(U, 1 - i, j)], mul=negOne)
        else:
            if E:
                for j in range(1 - OLy, sNy + OLy - 1 + 1):
                    for i in range(1, exchWidthX + 1):
                        sym.put(U, t, sNx + i, j + 1, loc[(V, sNx + i, j)])
                        sym.put(V, t, sNx + i, j + 1, loc[(U, sNx + i, j)], mul=negOne)
            if S:
                for j in range(1, exchWidthY + 1):
                    for i in range(1 - OLx, sNx + OLx - 1 + 1):
                        sym.put(U, t, i + 1, 1 - j, loc[(V, i, 1 - j)], mul=negOne)
                        sym.put(V, t, i + 1, 1 - j, loc[(U, i, 1 - j)])
        # corners (after the k loop of the edges, per tile)
        if W and S:
            if odd:
                for i in range(1, OLx + 1):
                    sym.copy(V, t, 1 - i, 1, U, 1, 1 - i, mul=negOne)
                    sym.copy(U, t, 1 - i, 1, V, 1, 1 - i)
            else:
                for i in range(1, OLx + 1):
                    sym.copy(U, t, 1, 1 - i, V, 1 - i, 1, mul=negOne)
                    sym.copy(V, t, 1, 1 - i, U, 1 - i, 1)
        if E and S:
            if odd:
                for i in range(2, OLx + 1):
                    sym.copy(U, t, sNx + 1, 2 - i, V, sNx + i, 1)
                    sym.copy(V, t, sNx + 1, 2 - i, U, sNx + i, 1, mul=negOne)
            else:
                sym.put(U, t, sNx + 1, 1, saved[t]["uSE"])
                sym.put(V, t, sNx + 1, 1, saved[t]["vSE"])
                for i in range(2, OLx + 1):
                    sym.copy(U, t, sNx + i, 1, V, sNx + 1, 2 - i, mul=negOne)
                    sym.copy(V, t, sNx + i, 1, U, sNx + 1, 2 - i)
        if E and N:
            if odd:
                for i in range(2, OLx + 1):
                    sym.copy(U, t, sNx + i, sNy + 1, V, sNx + 1, sNy + i)
                    sym.copy(V, t, sNx + i, sNy + 1, U, sNx + 1, sNy + i, mul=negOne)
            else:
                for i in range(2, OLx + 1):
                    sym.copy(U, t, sNx + 1, sNy + i, V, sNx + i, sNy + 1, mul=negOne)
                    sym.copy(V, t, sNx + 1, sNy + i, U, sNx + i, sNy + 1)
        if W and N:
            if odd:
                sym.put(U, t, 1, sNy + 1, saved[t]["uNW"])
                sym.put(V, t, 1, sNy + 1, saved[t]["vNW"])
                for i in range(2, OLx + 1):
                    sym.copy(U, t, 1, sNy + i, V, 2 - i, sNy + 1)
                    sym.copy(V, t, 1, sNy + i, U, 2 - i, sNy + 1, mul=negOne)
            else:
                for i in range(2, OLx + 1):
                    sym.copy(U, t, 2 - i, sNy + 1, V, 1, sNy + i, mul=negOne)
                    sym.copy(V, t, 2 - i, sNy + 1, U, 1, sNy + i)


def exch2_uv_3d_corner_fix(sym, w2, withSigns, U=1, V=2):
    """The useCubedSphereExchange part of EXCH2_UV_3D_RX (exch2_uv_3d_rx.template:79-227; DO_CORNER_COPY_V2U is
    #undef'd at :1 of the template, W2_FILL_NULL_REGIONS undefined): applied to the outputs of its two
    EXCH2_RX2_CUBE passes. `-vPhi` is a negation (neg), not a multiplication."""
    sz = w2.size
    sNx, sNy, OLx, OLy = sz.sNx, sz.sNy, sz.OLx, sz.OLy
    neg = bool(withSigns)
    for t in _tiles(w2):
        E, W, N, S = (w2.exch2_isEedge[t] == 1, w2.exch2_isWedge[t] == 1, w2.exch2_isNedge[t] == 1,
                      w2.exch2_isSedge[t] == 1)
        if W and S:                                                           # :119-140
            if OLx >= 2 and OLy >= 2:
                sym.copy(U, t, 0, 0, V, 1, 0)                                 # uPhi(0,0)=vPhi(1,0)
                sym.copy(V, t, 0, 0, U, 0, 1)                                 # vPhi(0,0)=uPhi(0,1)
        if W and N:                                                           # :141-168
            if OLx >= 2 and OLy >= 2:
                sym.copy(U, t, 0, sNy + 1, V, 1, sNy + 2, negate=neg)         # uPhi(0,sNy+1)=[-]vPhi(1,sNy+2)
                sym.copy(V, t, 0, sNy + 2, U, 0, sNy, negate=neg)             # vPhi(0,sNy+2)=[-]uPhi(0,sNy)
        if E and S:                                                           # :169-196
            if OLx >= 2 and OLy >= 2:
                sym.copy(U, t, sNx + 2, 0, V, sNx, 0, negate=neg)             # uPhi(sNx+2,0)=[-]vPhi(sNx,0)
                sym.copy(V, t, sNx + 1, 0, U, sNx + 2, 1, negate=neg)         # vPhi(sNx+1,0)=[-]uPhi(sNx+2,1)
        if E and N:                                                           # :197-218
            if OLx >= 2 and OLy >= 2:
                sym.copy(U, t, sNx + 2, sNy + 1, V, sNx, sNy + 2)             # uPhi(sNx+2,sNy+1)=vPhi(sNx,sNy+2)
                sym.copy(V, t, sNx + 1, sNy + 2, U, sNx + 2, sNy)             # vPhi(sNx+1,sNy+2)=uPhi(sNx+2,sNy)


def exch2_uv_dgrid_signs(sym, w2, withSigns, uPhi=1, vPhi=2):
    """The useCubedSphereExchange part of EXCH2_UV_DGRID_3D_RX after its EXCH2_UV_3D_RX call
    (exch2_uv_dgrid_3d_rx.template:91-196): `x*negOne` on the facet-edge halos."""
    sz = w2.size
    sNx, sNy, OLx, OLy = sz.sNx, sz.sNy, sz.OLx, sz.OLy
    exchWidthX, exchWidthY = OLx, OLy
    negOne = -1 if withSigns else 1                                           # :65-66
    for t in _tiles(w2):
        myFace = w2.exch2_myFace[t]
        if imod(myFace, 2) == 1:
            if w2.exch2_isNedge[t] == 1:
                for j in range(1, exchWidthY + 1):
                    for i in range(1 - OLx, sNx + OLx + 1):
                        sym.copy(uPhi, t, i, sNy + j, uPhi, i, sNy + j, mul=negOne)
            if w2.exch2_isWedge[t] == 1:
                for j in range(1 - OLy, sNy + OLy + 1):
                    for i in range(1, exchWidthX + 1):
                        sym.copy(vPhi, t, 1 - i, j, vPhi, 1 - i, j, mul=negOne)
        else:
            if w2.exch2_isEedge[t] == 1:
                for j in range(1 - OLy, sNy + OLy + 1):
                    for i in range(1, exchWidthX + 1):
                        sym.copy(vPhi, t, sNx + i, j, vPhi, sNx + i, j, mul=negOne)
            if w2.exch2_isSedge[t] == 1:
                for j in range(1, exchWidthY + 1):
                    for i in range(1 - OLx, sNx + OLx + 1):
                        sym.copy(uPhi, t, i, 1 - j, uPhi, i, 1 - j, mul=negOne)


# ---------------------------------------------------------------------------------------------------------------------
# all exchange kinds of an experiment

# kind -> (Fortran call, withSigns) ; the kinds of mitjax/eesupp/exch_maps.py plus the withSigns=.FALSE. variants
CUBE_SCALAR_KINDS = {"XY": None, "3D": None, "Z": None, "S3D": None, "SMs": True, "SMn": False}
CUBE_COPY_VECTOR_KINDS = {"As": ("A", True), "An": ("A", False), "Bs": ("B", True), "Bn": ("B", False)}
CUBE_RX2_KINDS = {"UVs": ("C", True), "UVn": ("C", False), "UV3s": ("C", True), "UV3n": ("C", False),
                  "Ds": ("D", True), "Dn": ("D", False)}


def cube_exchange_programs(w2, layout, useCubedSphereExchange, cpp=None):
    """(maps, rx2) of every exchange kind on the W2 topology `w2`:
      maps: {key: (src int32, comp int8, sign int8)} for the copy kinds (scalar keys "XY", ...; vector keys
            "As_u", "As_v", ...), as exch_maps.ExchangeMaps.maps (no copy kind negates: checked);
      rx2:  {kind: (swap, passes, post)} for the C-grid kinds: the EXCH2_RX2_CUBE passes of exch2_rx2_cube.py and
            `post` = {1: (src, comp, sign, neg) of u, 2: ... of v}, the post-pass gather (comp 1/2 = the u/v outputs
            of the passes, 0 = keep)."""
    return (cube_copy_maps(w2, layout, useCubedSphereExchange, cpp),
            cube_rx2_programs(w2, layout, useCubedSphereExchange, cpp))


def _check_w2_options(cpp):
    if cpp is not None:
        for opt in ("W2_FILL_NULL_REGIONS", "W2_USE_R1_ONLY"):
            try:
                on = bool(getattr(cpp, opt))
            except KeyError:
                on = False
            if on:
                raise NotImplementedError(f"cube exchanges: {opt} is not ported")


def cube_copy_maps(w2, layout, useCubedSphereExchange, cpp=None):
    """The `maps` part of cube_exchange_programs: the copy kinds (CUBE_SCALAR_KINDS, CUBE_COPY_VECTOR_KINDS)."""
    _check_w2_options(cpp)
    L = layout
    maps = {}
    for kind, withSigns in CUBE_SCALAR_KINDS.items():
        s = Sym(L, 1)
        if kind in ("XY", "3D"):
            exch2_3d(s, 1, w2)
        elif kind == "Z":
            exch2_z_3d(s, 1, w2, useCubedSphereExchange)
        elif kind == "S3D":
            exch2_s3d(s, 1, w2)
        else:
            exch2_sm_3d(s, 1, w2, withSigns, useCubedSphereExchange)
        src, comp, sign, neg = s.table(1, 1)
        if neg.any():
            raise AssertionError(f"{kind}: a copy kind negates")
        maps[kind] = (src, comp, sign)
    for kind, (grid, withSigns) in CUBE_COPY_VECTOR_KINDS.items():
        s = Sym(L, 2)
        if grid == "A":
            exch2_uv_agrid_3d(s, w2, withSigns, useCubedSphereExchange)
        else:
            exch2_uv_bgrid_3d(s, w2, withSigns, useCubedSphereExchange)
        for c, suf in ((1, "_u"), (2, "_v")):
            src, comp, sign, neg = s.table(c, c)
            if neg.any():
                raise AssertionError(f"{kind}: a copy kind negates")
            maps[kind + suf] = (src, comp, sign)
    return maps


def cube_rx2_programs(w2, layout, useCubedSphereExchange, cpp=None):
    """The `rx2` part of cube_exchange_programs: the C-grid kinds (CUBE_RX2_KINDS)."""
    from mitjax.pkg.exch2.exch2_rx2_cube import exch2_rx2_cube_tables
    _check_w2_options(cpp)
    L = layout
    rx2 = {}
    for kind, (grid, withSigns) in CUBE_RX2_KINDS.items():
        if grid == "C":
            passes = [exch2_rx2_cube_tables(w2, L, withSigns, "Cg", False),   # exch2_uv_3d_rx.template:70-73
                      exch2_rx2_cube_tables(w2, L, withSigns, "Cg", True)]    # :74-77
            swap = False
            s = Sym(L, 2)
            if useCubedSphereExchange:
                exch2_uv_3d_corner_fix(s, w2, withSigns, U=1, V=2)
        else:
            # EXCH2_UV_DGRID_3D_RX: EXCH2_UV_3D_RX( vPhi, uPhi, .FALSE. ) (exch2_uv_dgrid_3d_rx.template:86-88)
            passes = [exch2_rx2_cube_tables(w2, L, False, "Cg", False),
                      exch2_rx2_cube_tables(w2, L, False, "Cg", True)]
            swap = True
            s = Sym(L, 2)
            if useCubedSphereExchange:
                exch2_uv_3d_corner_fix(s, w2, False, U=2, V=1)               # its corner fixes on (vPhi, uPhi)
                exch2_uv_dgrid_signs(s, w2, withSigns, uPhi=1, vPhi=2)       # :91-196
        post = {c: s.table(c, c) for c in (1, 2)}
        rx2[kind] = (swap, passes, post)
    return rx2
